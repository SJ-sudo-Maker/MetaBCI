# -*- coding: utf-8 -*-
"""
LWSleepNet: Lightweight Sleep Staging Network.

A lightweight deep learning model for single-channel EEG sleep staging,
featuring dual-branch depthwise separable convolutions for multi-resolution
feature extraction and patch-based multi-head self-attention for temporal
dependency modeling.

Reference
---------
Yang, C., Li, B., Li, Y., He, Y., & Zhang, Y. (2023). LWSleepNet: A
lightweight attention-based deep learning model for sleep staging with
singlechannel EEG. Digital Health, 9, 20552076231188206.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

from .base import SkorchNet, _glorot_weight_zero_bias


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv1d(nn.Module):
    """1D depthwise separable convolution: dw conv → pw conv.

    Splits a standard conv into a per-channel spatial filter followed by
    a 1×1 channel mixer, drastically reducing parameters.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding,
            groups=in_channels, bias=False,
        )
        self.pw = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class InvertedResidual1d(nn.Module):
    """MobileNetV2-style inverted residual bottleneck for 1D signals.

    Expands channels, applies depthwise conv, then projects back.
    Residual connection when input/output shapes match.
    """

    def __init__(self, in_channels, out_channels, kernel_size=9,
                 expand_ratio=2):
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels)

        self.conv = nn.Sequential(OrderedDict([
            ("expand", nn.Conv1d(in_channels, hidden, 1, bias=False)),
            ("bn1", nn.BatchNorm1d(hidden)),
            ("act1", nn.GELU()),
            ("dw", nn.Conv1d(hidden, hidden, kernel_size,
                             padding=kernel_size // 2,
                             groups=hidden, bias=False)),
            ("bn2", nn.BatchNorm1d(hidden)),
            ("act2", nn.GELU()),
            ("project", nn.Conv1d(hidden, out_channels, 1, bias=False)),
            ("bn3", nn.BatchNorm1d(out_channels)),
        ]))

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MultiResolutionBranch(nn.Module):
    """Dual-branch multi-resolution feature extractor.

    Small kernel (k=5) captures high-frequency components (beta/gamma,
    ~20 Hz+), large kernel (k=51, ~0.5 s receptive field) captures
    low-frequency components (delta/theta, ~2 Hz).
    """

    def __init__(self, in_channels=1, out_channels=32,
                 small_kernel=5, large_kernel=51):
        super().__init__()

        self.small_branch = nn.Sequential(OrderedDict([
            ("dsconv", DepthwiseSeparableConv1d(
                in_channels, out_channels, small_kernel,
                padding=small_kernel // 2)),
            ("bn", nn.BatchNorm1d(out_channels)),
            ("act", nn.GELU()),
        ]))

        self.large_branch = nn.Sequential(OrderedDict([
            ("dsconv", DepthwiseSeparableConv1d(
                in_channels, out_channels, large_kernel,
                padding=large_kernel // 2)),
            ("bn", nn.BatchNorm1d(out_channels)),
            ("act", nn.GELU()),
        ]))

    def forward(self, x):
        s = self.small_branch(x)
        l = self.large_branch(x)
        return torch.cat([s, l], dim=1)


class TemporalAttentionBlock(nn.Module):
    """Patch-based multi-head self-attention block for temporal modeling.

    Divides the feature sequence into fixed-length patches and applies
    multi-head self-attention within each patch. Depthwise separable
    convolutions before and after MHA project features into a higher-
    dimensional space and back.
    """

    def __init__(self, d_model=64, patch_size=25, n_heads=8,
                 expand_ratio=2):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        hidden = d_model * expand_ratio

        self.pre_conv = nn.Sequential(OrderedDict([
            ("dw", nn.Conv1d(d_model, d_model, 3, padding=1,
                             groups=d_model, bias=False)),
            ("pw_expand", nn.Conv1d(d_model, hidden, 1, bias=False)),
            ("act", nn.GELU()),
            ("pw_project", nn.Conv1d(hidden, d_model, 1, bias=False)),
        ]))

        self.mha = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True,
        )

        self.post_conv = nn.Sequential(OrderedDict([
            ("dw", nn.Conv1d(d_model, d_model, 3, padding=1,
                             groups=d_model, bias=False)),
            ("pw", nn.Conv1d(d_model, d_model, 1, bias=False)),
        ]))
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        P = self.patch_size
        N = T // P
        residual = x

        # Pre-MHA projection
        x = self.pre_conv(x)                       # (B, C, T)

        # Reshape into patches for per-patch self-attention
        x = x.view(B, C, N, P)                     # (B, C, N, P)
        x = x.permute(0, 2, 3, 1).contiguous()     # (B, N, P, C)
        x = x.view(B * N, P, C)                    # (B*N, P, C)

        x, _ = self.mha(x, x, x)                   # (B*N, P, C)

        # Reconstruct sequence
        x = x.view(B, N, P, C)                     # (B, N, P, C)
        x = x.permute(0, 3, 1, 2).contiguous()     # (B, C, N, P)
        x = x.view(B, C, T)                        # (B, C, T)

        # Post-MHA projection + residual
        x = self.post_conv(x)
        x = self.norm(x + residual)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@SkorchNet
class LWSleepNet(nn.Module):
    """LWSleepNet: lightweight single-channel EEG sleep staging model.

    Architecture
    ------------
    1. MultiResolutionBranch — dual-branch depthwise separable conv
       (k=5 for high-freq, k=51 for low-freq), outputs 64 channels.
    2. InvertedResidual1d — bottleneck for cross-channel recalibration.
    3. TemporalAttentionBlock ×3 — patch-based (P=25) MHA (8 heads)
       with pre/post depthwise separable conv projections.
    4. Global average pool → Dropout(0.5) → Linear(5).

    The design targets ~180K parameters for deployment on portable devices.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (1 for single-channel Fpz-Cz).
    n_samples : int
        Samples per epoch (3000 for 30 s @ 100 Hz).
    n_classes : int
        Number of sleep stages (5: W, N1, N2, N3, REM).

    Examples
    --------
    >>> # X: (n_epochs, n_channels, n_samples)
    >>> model = LWSleepNet(X.shape[1], X.shape[2], 5)
    >>> model.fit(X[train_idx], y[train_idx])
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super().__init__()

        self.mrfe = MultiResolutionBranch(
            in_channels=n_channels,
            out_channels=32,
            small_kernel=5,
            large_kernel=51,
        )

        self.bottleneck = InvertedResidual1d(
            in_channels=64, out_channels=64,
            kernel_size=9, expand_ratio=2,
        )

        self.ta_block1 = TemporalAttentionBlock(
            d_model=64, patch_size=25, n_heads=8,
        )
        self.ta_block2 = TemporalAttentionBlock(
            d_model=64, patch_size=25, n_heads=8,
        )
        self.ta_block3 = TemporalAttentionBlock(
            d_model=64, patch_size=25, n_heads=8,
        )

        self.head = nn.Sequential(OrderedDict([
            ("gap", nn.AdaptiveAvgPool1d(1)),
            ("flatten", nn.Flatten()),
            ("drop", nn.Dropout(0.5)),
            ("fc", nn.Linear(64, n_classes)),
        ]))

        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        _glorot_weight_zero_bias(self)

    def forward(self, X: Tensor, **kwargs) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        X : Tensor, shape (B, n_channels, n_samples)

        Returns
        -------
        Tensor, shape (B, n_classes)
        """
        x = self.mrfe(X)
        x = self.bottleneck(x)
        x = self.ta_block1(x)
        x = self.ta_block2(x)
        x = self.ta_block3(x)
        return self.head(x)

    def cal_backbone(self, X: Tensor, **kwargs) -> Tensor:
        """Return features before the classification head."""
        x = self.mrfe(X)
        x = self.bottleneck(x)
        x = self.ta_block1(x)
        x = self.ta_block2(x)
        x = self.ta_block3(x)
        return x
