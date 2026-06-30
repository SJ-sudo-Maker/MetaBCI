# -*- coding: utf-8 -*-
"""
ParaSleep: Lightweight Sleep Staging Network.

A lightweight deep learning model for single-channel EEG sleep staging,
featuring dual-branch depthwise separable convolutions for multi-resolution
feature extraction and patch-based multi-head self-attention for temporal
dependency modeling.

Reference
---------
Yang, C., Li, B., Li, Y., He, Y., & Zhang, Y. (2023). ParaSleep: A
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
class ParaSleep(nn.Module):
    """ParaSleep: lightweight single-channel EEG sleep staging model.

    Two architectures are available:

    **Base (use_temporal_attention=False, default):**
    1. MultiResolutionBranch — dual-branch depthwise separable conv
       (k=5 for high-freq, k=51 for low-freq), outputs 64 channels.
    2. InvertedResidual1d — bottleneck for cross-channel recalibration.
    3. TemporalAttentionBlock ×3 — patch-based (P=25) MHA (8 heads)
       with pre/post depthwise separable conv projections.
    4. Global average pool → Dropout(0.5) → Linear(n_classes).

    Context epochs are treated as CNN input channels (n_channels=context).

    **Temporal Attention (use_temporal_attention=True):**
    1. Per-epoch shared feature extractor:
       MultiResolutionBranch(1ch) → InvertedResidual1d →
       TemporalAttentionBlock → GAP → (64,)
    2. Learnable position encoding + 1-layer TransformerEncoder
       across context epochs.
    3. Center-epoch feature → Dropout(0.5) → Linear(n_classes).

    Context epochs are processed independently first, then related via
    self-attention, explicitly modeling sleep stage transitions.

    Parameters
    ----------
    n_channels : int
        Base mode: number of EEG channels (=context window size).
        TA mode: number of context epochs.
    n_samples : int
        Samples per epoch (3000 for 30 s @ 100 Hz).
    n_classes : int
        Number of sleep stages (5: W, N1, N2, N3, REM).
    use_temporal_attention : bool
        If True, use the TA architecture with per-epoch shared CNN +
        Transformer. Default False (base architecture).
    use_aux : bool
        If True, add 4-class and 3-class auxiliary classification heads.
    d_model : int
        Feature dimension (TA mode only). Default 64.
    nhead : int
        Number of attention heads (TA mode only). Default 4.
    dim_feedforward : int
        Transformer FFN hidden dim (TA mode only). Default 128.
    target_index : str
        Which epoch to classify in TA mode.
        "center" — middle epoch (C//2), for center context.
        "last"   — last epoch (C-1), for causal context.
        Default "center".
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 use_temporal_attention: bool = False,
                 use_aux: bool = False,
                 d_model: int = 64, nhead: int = 4,
                 dim_feedforward: int = 128,
                 target_index: str = "center"):
        super().__init__()
        self.use_temporal_attention = use_temporal_attention
        self.use_aux = use_aux
        self.n_epochs = n_channels  # same param, semantic depends on mode
        self.target_index = target_index

        if target_index not in ("center", "last"):
            raise ValueError(f"target_index must be 'center' or 'last', got {target_index!r}")

        if use_temporal_attention:
            self._init_ta_architecture(
                n_classes, d_model, nhead, dim_feedforward)
        else:
            self._init_base_architecture(n_channels, n_classes)

        if use_aux:
            in_dim = d_model if use_temporal_attention else 64
            self.head4 = nn.Linear(in_dim, 4)
            self.head3 = nn.Linear(in_dim, 3)

        self._reset_parameters()

    # ------------------------------------------------------------------
    # Architecture initializers
    # ------------------------------------------------------------------

    def _init_base_architecture(self, n_channels, n_classes):
        """Original ParaSleep: epochs as CNN channels."""
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
        self._feat_dim = 64

    def _init_ta_architecture(self, n_classes, d_model, nhead,
                              dim_feedforward):
        """TA architecture: per-epoch shared CNN + Transformer."""
        # Per-epoch feature extractor (shared weights)
        self.mrfe = MultiResolutionBranch(
            in_channels=1, out_channels=32,
            small_kernel=5, large_kernel=51,
        )
        self.bottleneck = InvertedResidual1d(
            in_channels=64, out_channels=64,
            kernel_size=9, expand_ratio=2,
        )
        self.intra_attn = TemporalAttentionBlock(
            d_model=64, patch_size=25, n_heads=8,
        )
        self.epoch_gap = nn.AdaptiveAvgPool1d(1)

        # Temporal Transformer
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_epochs, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1, activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1)

        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(d_model, n_classes)
        self._feat_dim = d_model

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reset_parameters(self):
        _glorot_weight_zero_bias(self)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: Tensor, **kwargs):
        """Forward pass.

        Parameters
        ----------
        X : Tensor
            Base mode: shape (B, n_channels, n_samples).
            TA mode:  shape (B, n_epochs, n_samples).

        Returns
        -------
        Tensor (B, n_classes), or tuple (out5, out4, out3) if use_aux=True
        """
        if self.use_temporal_attention:
            return self._forward_ta(X)
        return self._forward_base(X)

    def _forward_base(self, X: Tensor):
        x = self.mrfe(X)
        x = self.bottleneck(x)
        x = self.ta_block1(x)
        x = self.ta_block2(x)
        x = self.ta_block3(x)
        out5 = self.head(x)
        if self.use_aux:
            feat = self.head[:-1](x)
            out4 = self.head4(feat)
            out3 = self.head3(feat)
            return out5, out4, out3
        return out5

    def _forward_ta(self, X: Tensor):
        B, C, T = X.shape

        # Per-epoch feature extraction (shared weights)
        feats = []
        for i in range(C):
            xi = X[:, i:i+1, :]                     # (B, 1, T)
            f = self.mrfe(xi)                        # (B, 64, T')
            f = self.bottleneck(f)                   # (B, 64, T')
            f = self.intra_attn(f)                   # (B, 64, T')
            f = self.epoch_gap(f).squeeze(-1)        # (B, 64)
            feats.append(f)
        feats = torch.stack(feats, dim=1)            # (B, C, 64)

        # Temporal Transformer
        feats = feats + self.pos_embed[:, :C, :]     # (B, C, 64)
        feats = self.transformer(feats)              # (B, C, 64)

        # Target epoch classification
        if self.target_index == "last":
            idx = C - 1
        else:
            idx = C // 2
        x = feats[:, idx, :]                         # (B, 64)
        x = self.dropout(x)
        out5 = self.fc(x)

        if self.use_aux:
            out4 = self.head4(x)
            out3 = self.head3(x)
            return out5, out4, out3
        return out5

    def cal_backbone(self, X: Tensor, **kwargs) -> Tensor:
        """Return features before the classification head."""
        if self.use_temporal_attention:
            B, C, T = X.shape
            feats = []
            for i in range(C):
                xi = X[:, i:i+1, :]
                f = self.mrfe(xi)
                f = self.bottleneck(f)
                f = self.intra_attn(f)
                f = self.epoch_gap(f).squeeze(-1)
                feats.append(f)
            feats = torch.stack(feats, dim=1)
            feats = feats + self.pos_embed[:, :C, :]
            feats = self.transformer(feats)
            # Return target epoch features
            idx = C - 1 if self.target_index == "last" else C // 2
            return feats[:, idx, :]

        x = self.mrfe(X)
        x = self.bottleneck(x)
        x = self.ta_block1(x)
        x = self.ta_block2(x)
        x = self.ta_block3(x)
        return x
