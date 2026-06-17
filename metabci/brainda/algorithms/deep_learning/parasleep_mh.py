# -*- coding: utf-8 -*-
"""
ParaSleep-MH: Multi-Head variant inspired by ZleepAnlystNet's separating training.

5 parallel classification heads, each with class-specific loss weights.
All heads backprop to the shared backbone — no gradient isolation.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

from .parasleep import (
    MultiResolutionBranch, InvertedResidual1d, TemporalAttentionBlock,
)
from .base import SkorchNet, _glorot_weight_zero_bias


class MultiHead(nn.Module):
    """5 heads, each focusing on one sleep stage via asymmetric loss weights."""

    def __init__(self, in_features=64, n_classes=5, hidden=32):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(in_features, hidden)),
                ("act", nn.GELU()),
                ("fc2", nn.Linear(hidden, n_classes)),
            ])) for _ in range(n_classes)
        ])

    def forward(self, feat: Tensor):
        """Returns list of (B,5) logits, one per head."""
        return [h(feat) for h in self.heads]


@SkorchNet
class ParaSleepMH(nn.Module):
    """ParaSleep with 5 class-specific classification heads.

    Parameters
    ----------
    n_channels : int (3 for 3-epoch context windows)
    n_samples : int (3000 for 30s @ 100Hz)
    n_classes : int (5)
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super().__init__()

        self.mrfe = MultiResolutionBranch(
            in_channels=n_channels, out_channels=32, small_kernel=5, large_kernel=51,
        )
        self.bottleneck = InvertedResidual1d(
            in_channels=64, out_channels=64, kernel_size=9, expand_ratio=2,
        )
        self.ta_block1 = TemporalAttentionBlock(d_model=64, patch_size=25, n_heads=8)
        self.ta_block2 = TemporalAttentionBlock(d_model=64, patch_size=25, n_heads=8)
        self.ta_block3 = TemporalAttentionBlock(d_model=64, patch_size=25, n_heads=8)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.head = MultiHead(in_features=64, n_classes=n_classes)

        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self):
        _glorot_weight_zero_bias(self)

    def forward(self, X: Tensor, **kwargs):
        x = self.mrfe(X)
        x = self.bottleneck(x)
        x = self.ta_block1(x)
        x = self.ta_block2(x)
        x = self.ta_block3(x)
        x = self.gap(x)
        feat = self.flatten(x)
        return self.head(feat)  # list of 5 tensors, each (B, 5)

    def cal_backbone(self, X: Tensor, **kwargs) -> Tensor:
        x = self.mrfe(X)
        x = self.bottleneck(x)
        x = self.ta_block1(x)
        x = self.ta_block2(x)
        x = self.ta_block3(x)
        x = self.gap(x)
        return self.flatten(x)
