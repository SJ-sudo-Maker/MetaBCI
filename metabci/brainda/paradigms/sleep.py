# -*- coding: utf-8 -*-
"""
Sleep Paradigm for passive BCI sleep staging.

Extends BaseParadigm to support continuous sleep EEG data without discrete
event markers. Provides configurable context window building (center/causal)
and multi-level label mapping (5-class / 4-class / 3-class).

Works with SleepEDFDataset.

Examples
--------
>>> paradigm = SleepParadigm(
...     channels=["EEG Fpz-Cz"],
...     srate=100,
...     context=3,
...     context_mode="center",
...     label_mode="5class",
... )
>>> X, y, meta = paradigm.get_data(dataset, subjects=["4001"])
>>> X_windows, y_windows = paradigm.build_windows(X, y)
>>> y_4class = paradigm.map_labels(y, "4class")
"""
from typing import Optional, List, Tuple

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
    context : int
        Number of consecutive epochs in each context window (odd).
        Default 3.
    context_mode : str
        "center" — equal context before and after the target epoch.
        "causal" — only past epochs (suitable for online inference).
        Default "center".
    label_mode : str
        "5class" — W, N1, N2, N3, REM
        "4class" — W, N1+N2, N3, REM
        "3class" — W, NREM, REM
        Default "5class".
    """

    # Label remapping tables
    _MAP_4CLASS = np.array([0, 1, 1, 2, 3], dtype=np.int64)  # N1+N2 merged
    _MAP_3CLASS = np.array([0, 1, 1, 1, 2], dtype=np.int64)  # N1+N2+N3=NREM

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        srate: Optional[float] = None,
        context: int = 3,
        context_mode: str = "center",
        label_mode: str = "5class",
    ):
        super().__init__(
            channels=channels,
            srate=srate,
        )
        if context % 2 != 1:
            raise ValueError(f"context must be odd, got {context}")
        if context_mode not in ("center", "causal"):
            raise ValueError(f"context_mode must be 'center' or 'causal', got {context_mode!r}")
        if label_mode not in ("5class", "4class", "3class"):
            raise ValueError(f"label_mode must be '5class', '4class', or '3class', got {label_mode!r}")

        self.context = context
        self.context_mode = context_mode
        self.label_mode = label_mode

    def is_valid(self, dataset: BaseDataset) -> bool:
        """Check if the dataset is compatible with this paradigm."""
        return getattr(dataset, "paradigm", None) == "sleep"

    # ------------------------------------------------------------------
    # Chronological epoch extraction
    # ------------------------------------------------------------------

    def extract_epochs(self, raw, annot, sfreq: float, epoch_sec: int = 30):
        """Extract epochs in strict chronological order from annotations.

        Unlike BaseParadigm.get_data() which groups epochs by event class,
        this preserves the original sleep timeline (crucial for context windows).

        Parameters
        ----------
        raw : mne.io.Raw
            Pre-loaded continuous EEG (single channel).
        annot : mne.Annotations
            Sleep stage annotations from Hypnogram.
        sfreq : float
            Sampling frequency.
        epoch_sec : int
            Epoch duration in seconds.

        Returns
        -------
        X : ndarray (n_epochs, n_channels, n_samples)
        y : ndarray (n_epochs,)
        onsets : ndarray (n_epochs,)
        """
        data = raw.get_data()  # (n_channels, n_samples)
        epoch_samples = int(epoch_sec * sfreq)

        X_list, y_list, onset_list = [], [], []
        for a in annot:
            desc = str(a["description"]).strip()
            # Support both digit labels ("0"-"4") and text labels ("Sleep stage W", etc.)
            if desc.lstrip("-").isdigit():
                stage_id = int(desc)
            else:
                from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
                stage_id = SleepEDFDataset.STAGE_MAP.get(desc, -1)
            if stage_id not in (0, 1, 2, 3, 4):
                continue
            n_epochs = int(a["duration"] // epoch_sec)
            for i in range(n_epochs):
                onset_sec = a["onset"] + i * epoch_sec
                start_sample = int(onset_sec * sfreq)
                end_sample = start_sample + epoch_samples
                if end_sample > data.shape[1]:
                    break
                epoch_data = data[:, start_sample:end_sample]
                X_list.append(epoch_data)
                y_list.append(stage_id)
                onset_list.append(onset_sec)

        if len(X_list) == 0:
            raise RuntimeError("No valid epochs extracted")

        X = np.stack(X_list).astype(np.float32)  # (n, ch, samples)
        y = np.array(y_list, dtype=np.int64)
        onsets = np.array(onset_list)

        # Verify strict chronological order
        assert np.all(np.diff(onsets) > 0), \
            "Epochs are not in strict chronological order!"
        return X, y, onsets

    # ------------------------------------------------------------------
    # Context window construction
    # ------------------------------------------------------------------

    def build_windows(self, X: np.ndarray, y: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Build context windows from consecutive single-channel epochs.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_samples)
            Raw epoch data (single-channel assumed; first channel used).
        y : ndarray, shape (n_epochs,)
            Integer sleep stage labels.

        Returns
        -------
        Xw : ndarray, shape (n_windows, context, n_samples)
            Context-windowed epochs.
        yw : ndarray, shape (n_windows,)
            Labels for the target (center/last) epoch of each window.
        """
        n = X.shape[0]
        if n < self.context:
            raise ValueError(
                f"Need at least {self.context} epochs, got {n}"
            )

        Xw_list, yw_list = [], []

        if self.context_mode == "causal":
            # Left-only: [t-context+1, ..., t] → predict t
            for i in range(self.context - 1, n):
                Xw_list.append(X[i - self.context + 1 : i + 1, 0, :])
                yw_list.append(y[i])
        else:
            # Center: [t-half, ..., t+half] → predict t
            half = self.context // 2
            for i in range(half, n - half):
                Xw_list.append(X[i - half : i + half + 1, 0, :])
                yw_list.append(y[i])

        Xw = np.stack(Xw_list)  # (n_windows, context, n_samples)
        yw = np.array(yw_list, dtype=np.int64)

        # Auto-remap labels if not 5class
        if self.label_mode != "5class":
            yw = self.map_labels(yw, self.label_mode)

        return Xw, yw

    # ------------------------------------------------------------------
    # Label mapping
    # ------------------------------------------------------------------

    @classmethod
    def map_labels(cls, y: np.ndarray, target: str) -> np.ndarray:
        """Remap 5-class labels to 4-class or 3-class.

        Parameters
        ----------
        y : ndarray
            Integer labels in 5-class space (0=W, 1=N1, 2=N2, 3=N3, 4=REM).
        target : str
            "4class" — {W:0, N1+N2:1, N3:2, REM:3}
            "3class" — {W:0, NREM:1, REM:2}

        Returns
        -------
        ndarray of remapped labels.
        """
        if target == "4class":
            return cls._MAP_4CLASS[y]
        elif target == "3class":
            return cls._MAP_3CLASS[y]
        else:
            raise ValueError(f"Unknown target: {target!r}")

    @property
    def n_classes(self) -> int:
        """Number of classes for the configured label_mode."""
        return {"5class": 5, "4class": 4, "3class": 3}[self.label_mode]

    @property
    def class_names(self) -> List[str]:
        """Class names for the configured label_mode."""
        return {
            "5class": ["W", "N1", "N2", "N3", "REM"],
            "4class": ["W", "N1+N2", "N3", "REM"],
            "3class": ["W", "NREM", "REM"],
        }[self.label_mode]


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
