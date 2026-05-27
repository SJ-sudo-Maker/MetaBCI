# -*- coding: utf-8 -*-
"""
Sleep Paradigm for passive BCI sleep staging.

Extends BaseParadigm to support continuous sleep EEG data without discrete
event markers. Works with SleepEDFDataset.
"""
from typing import Optional, List

import numpy as np

from .base import BaseParadigm
from ..datasets.base import BaseDataset


class SleepParadigm(BaseParadigm):
    """Passive monitoring paradigm for sleep staging.

    Parameters
    ----------
    channels : list, optional
        EEG channels to use. Default None (use all dataset channels).
    srate : float, optional
        Target sampling rate. Default None (use dataset srate).
    """

    def is_valid(self, dataset: BaseDataset) -> bool:
        """Check if the dataset is compatible with this paradigm."""
        return getattr(dataset, "paradigm", None) == "sleep"


def sleep_preprocess_hook(raw, caches):
    """Raw hook: apply sleep-specific bandpass filter (0.5-40 Hz)."""
    raw.filter(0.5, 40, picks="eeg", verbose=False)
    return raw, caches


def sleep_normalize_hook(X, y, meta, caches):
    """Data hook: Z-score normalize each epoch independently."""
    # X shape: (n_epochs, n_channels, n_samples)
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-8
    X = (X - mean) / std
    return X, y, meta, caches
