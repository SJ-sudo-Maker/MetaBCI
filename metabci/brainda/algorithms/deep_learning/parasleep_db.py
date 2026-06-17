# -*- coding: utf-8 -*-
"""
ParaSleep-DB: Dual-Branch variant with gradient isolation + transitive training.

Reference: Zhang et al. (2023) "A Two-Branch Trade-Off Neural Network for
Balanced Scoring Sleep Stages on Multiple Cohorts." Frontiers in Neuroscience.

Key changes from ParaSleep:
  - Dual classification head (ELB + SLB)
  - SLB receives detached features (gradient isolation from backbone)
  - Transitive α schedule during training
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

from .parasleep import (
    MultiResolutionBranch, InvertedResidual1d, TemporalAttentionBlock,
)
from .base import SkorchNet, _glorot_weight_zero_bias


class DualBranchHead(nn.Module):
    """Two-branch classifier with gradient isolation on SLB.

    ELB (Epoch Learning Branch): standard cross-entropy, full gradient flow
        to backbone. Learns general sleep features.

    SLB (Sequence Learning Branch): operates on detached features, no
        gradient to backbone. Uses class-balanced loss to improve N1
        without corrupting backbone features.
    """

    def __init__(self, in_features=64, n_classes=5, hidden=128):
        super().__init__()

        # ELB: simple linear (keep backbone features clean)
        self.elb = nn.Linear(in_features, n_classes)

        # SLB: deeper head with Focal Loss focus (detached input)
        self.slb = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(in_features, hidden)),
            ("act", nn.GELU()),
            ("drop", nn.Dropout(0.3)),
            ("fc2", nn.Linear(hidden, n_classes)),
        ]))

    def forward(self, feat: Tensor, detach_slb: bool = True):
        """Forward pass.

        Parameters
        ----------
        feat : Tensor, shape (B, 64)
            Backbone features after GAP.
        detach_slb : bool
            If True, SLB branch sees detached features (gradient isolation).
            Set False during inference (no gradient concern).

        Returns
        -------
        elb_out : Tensor, shape (B, n_classes)
        slb_out : Tensor, shape (B, n_classes)
        """
        elb_out = self.elb(feat)

        if detach_slb:
            slb_in = feat.detach()
        else:
            slb_in = feat

        slb_out = self.slb(slb_in)
        return elb_out, slb_out


@SkorchNet
class ParaSleepDB(nn.Module):
    """ParaSleep with dual-branch classification head.

    Architecture
    ------------
    1. MultiResolutionBranch — dual-kernel depthwise separable conv
    2. InvertedResidual1d — bottleneck
    3. TemporalAttentionBlock ×3 — patch-based MHA
    4. Global Avg Pool → 64-d features
    5. DualBranchHead:
       - ELB: Linear(64, 5) — standard CE loss
       - SLB: Linear(64→128→5) — Focal Loss on detached features

    Parameters
    ----------
    n_channels : int
        Input channels (3 for 3-epoch context windows).
    n_samples : int
        Samples per epoch (3000 for 30s @ 100Hz).
    n_classes : int
        Number of sleep stages (5).
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super().__init__()

        self.mrfe = MultiResolutionBranch(
            in_channels=n_channels, out_channels=32,
            small_kernel=5, large_kernel=51,
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

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.head = DualBranchHead(in_features=64, n_classes=n_classes)

        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        _glorot_weight_zero_bias(self)

    def forward(self, X: Tensor, detach_slb: bool = True, **kwargs):
        """Forward pass.

        Parameters
        ----------
        X : Tensor, shape (B, n_channels, n_samples)
        detach_slb : bool
            If True, gradient isolation on SLB (training mode).
            Set False for inference.
        """
        x = self.mrfe(X)
        x = self.bottleneck(x)
        x = self.ta_block1(x)
        x = self.ta_block2(x)
        x = self.ta_block3(x)
        x = self.gap(x)
        feat = self.flatten(x)  # (B, 64)
        return self.head(feat, detach_slb=detach_slb)

    def cal_backbone(self, X: Tensor, **kwargs) -> Tensor:
        """Return backbone features before classification head."""
        x = self.mrfe(X)
        x = self.bottleneck(x)
        x = self.ta_block1(x)
        x = self.ta_block2(x)
        x = self.ta_block3(x)
        x = self.gap(x)
        return self.flatten(x)
