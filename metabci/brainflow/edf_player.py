# -*- coding: utf-8 -*-
"""
EDF file replay as simulated real-time data source.

Allows testing the SleepOnlineWorker pipeline without actual EEG hardware.
"""

from typing import Optional, List, Tuple, Dict
import threading
import time

import numpy as np
import mne

from .amplifiers import BaseAmplifier, Marker
from .workers import ProcessWorker


class EDFSleepPlayer(BaseAmplifier):
    """Replay a sleep EDF recording as if it were live streaming data.

    Loads a pre-recorded EDF file, extracts a single EEG channel, and
    streams it through the standard brainflow Marker → Worker pipeline
    at a controlled rate (default 1× real-time, configurable for faster
    testing).

    Works seamlessly with SleepOnlineWorker:

    >>> player = EDFSleepPlayer("subject.EDF", srate=100)
    >>> worker = SleepOnlineWorker(model=model)
    >>> marker = Marker(interval=[0, 30], srate=100, events=None)
    >>> player.register_worker("sleep", worker, marker)
    >>> player.up_worker("sleep")
    >>> player.start_trans()         # blocks until EOF or stopped
    >>> player.down_worker("sleep")
    >>> player.clear()

    Parameters
    ----------
    edf_path : str
        Path to the PSG EDF file.
    channel : str
        EEG channel to extract (default "EEG Fpz-Cz").
    srate : int
        Target sampling rate in Hz (data is resampled to this).
    speed : float
        Replay speed multiplier. 1.0 = real-time, 2.0 = 2× faster.
        Use float("inf") for maximum speed (no timing control).
    chunk_size : int
        Samples per recv() call. Default 10 (matches typical amp packet
        size). Larger values reduce overhead but make timing coarser.
    hypnogram_path : str, optional
        Path to the hypnogram annotation file for ground-truth comparison.
    verbose : bool
        Print progress during playback.

    Attributes
    ----------
    data : np.ndarray, shape (n_samples,)
        The full EEG signal loaded into memory.
    duration_sec : float
        Total recording duration in seconds.
    position : int
        Current playback position in samples (advanced by recv()).
    """

    def __init__(
        self,
        edf_path: str,
        channel: str = "EEG Fpz-Cz",
        srate: int = 100,
        speed: float = 1.0,
        chunk_size: int = 10,
        hypnogram_path: Optional[str] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.edf_path = edf_path
        self.target_channel = channel
        self.target_srate = srate
        self.speed = speed
        self.chunk_size = chunk_size
        self.hypnogram_path = hypnogram_path
        self.verbose = verbose

        # Load EDF
        self._log(f"Loading {edf_path} ...")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # Pick channel
        pick = None
        if channel in raw.ch_names:
            pick = [channel]
        else:
            matches = [c for c in raw.ch_names if channel.lower() in c.lower()]
            if matches:
                pick = [matches[0]]
                self._log(f"  Using channel: {matches[0]}")
            else:
                pick = [raw.ch_names[0]]
                self._log(f"  Channel not found, using: {pick[0]}")

        raw.pick(pick)
        raw.resample(srate, verbose=False)

        self.data = raw.get_data().squeeze().astype(np.float64)
        self.n_samples = len(self.data)
        self.duration_sec = self.n_samples / srate
        self.position = 0
        self._timing_thread: Optional[threading.Thread] = None

        self._log(f"  Loaded: {self.duration_sec/60:.1f} min, "
                  f"{self.n_samples} samples @ {srate} Hz")

        # Load ground-truth hypnogram if provided
        self.true_stages: Optional[np.ndarray] = None
        if hypnogram_path:
            self._load_hypnogram(hypnogram_path, srate)

    # ------------------------------------------------------------------
    # BaseAmplifier interface
    # ------------------------------------------------------------------

    def recv(self) -> List[List[float]]:
        """Read next chunk of EEG samples (simulated live stream).

        Returns
        -------
        samples : list of [ch_data, trigger_label]
            A chunk of samples in the format expected by _detect_event().
            Returns empty list when EOF is reached.
        """
        if self.position >= self.n_samples:
            if self.verbose:
                self._log("EOF reached.")
            return []

        end = min(self.position + self.chunk_size, self.n_samples)
        chunk = self.data[self.position:end]

        # Pack as [ch_data, trigger=0] per sample
        samples = [[float(v), 0.0] for v in chunk]

        self.position = end
        return samples

    def start_trans(self):
        """Start simulated data transmission.

        Overrides BaseAmplifier.start_trans() to add timing control.
        Runs in the current thread (blocks until EOF or stop()).
        """
        if self.speed == float("inf"):
            # Max speed: just run the loop with no timing
            return super().start_trans()

        self._exit.clear()
        self._log(f"Starting playback at {self.speed}× speed ...")
        t0 = time.perf_counter()
        samples_processed = 0

        while not self._exit.is_set():
            samples = self.recv()
            if not samples:
                break
            self._detect_event(samples)

            samples_processed += len(samples)
            elapsed = time.perf_counter() - t0
            expected_elapsed = samples_processed / self.target_srate / self.speed
            sleep_time = expected_elapsed - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        elapsed = time.perf_counter() - t0
        dur = samples_processed / self.target_srate
        self._log(f"Playback finished: {dur:.0f}s of EEG in {elapsed:.1f}s "
                  f"({dur/elapsed:.1f}× real-time)")
        self._exit.clear()

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------

    def get_progress(self) -> float:
        """Return playback progress as a fraction [0, 1]."""
        return self.position / max(self.n_samples, 1)

    def get_true_stage_at(self, index: int) -> int:
        """Get the ground-truth sleep stage for a given epoch index.

        Parameters
        ----------
        index : int
            Zero-based epoch index (each epoch = 30 s).

        Returns
        -------
        stage : int
            Sleep stage 0-4, or -1 if unknown.
        """
        if self.true_stages is None:
            return -1
        if index < 0 or index >= len(self.true_stages):
            return -1
        return int(self.true_stages[index])

    def _load_hypnogram(self, hyp_path: str, srate: int):
        """Parse hypnogram annotations into epoch-level stage labels."""
        annot = mne.read_annotations(hyp_path)
        stage_map = {
            "Sleep stage W": 0, "Sleep stage 1": 1,
            "Sleep stage 2": 2, "Sleep stage 3": 3,
            "Sleep stage 4": 3, "Sleep stage R": 4,
            "Sleep stage ?": -1, "Movement time": -1,
        }
        epoch_sec = 30
        stages = []
        for a in annot:
            sid = stage_map.get(a["description"], -1)
            n = int(a["duration"] // epoch_sec)
            for _ in range(n):
                stages.append(sid)
        self.true_stages = np.array(stages, dtype=int)
        self._log(f"  Ground truth: {len(stages)} epochs loaded")

    def _log(self, msg: str):
        if self.verbose:
            print(f"[EDFPlayer] {msg}")


# ---------------------------------------------------------------------------
# Quick test runner (standalone, no brainflow pipeline needed)
# ---------------------------------------------------------------------------

def quick_test(
    edf_path: str,
    model,
    hypnogram_path: Optional[str] = None,
    channel: str = "EEG Fpz-Cz",
    srate: int = 100,
    max_epochs: Optional[int] = None,
) -> Dict:
    """Run a quick offline test: feed EDF data through the model epoch-by-epoch.

    Does NOT use the brainflow pipeline — directly loads the model, slices
    the EDF into 30s epochs, runs inference, and returns results.

    Parameters
    ----------
    edf_path : str
        Path to PSG EDF.
    model : nn.Module
        Trained LWSleepNet (float32, eval mode).
    hypnogram_path : str, optional
        Path to hypnogram for accuracy comparison.
    channel : str
        EEG channel name.
    srate : int
        Sampling rate.
    max_epochs : int, optional
        Limit number of epochs (for quick testing).

    Returns
    -------
    results : dict
        Keys: predictions, true_stages, accuracy, n_epochs
    """
    import torch
    from scipy.signal import butter, filtfilt

    # Load EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    pick = None
    if channel in raw.ch_names:
        pick = [channel]
    else:
        matches = [c for c in raw.ch_names if channel.lower() in c.lower()]
        pick = [matches[0]] if matches else [raw.ch_names[0]]
    raw.pick(pick)
    raw.resample(srate, verbose=False)
    data = raw.get_data().squeeze().astype(np.float64)

    # Filter
    nyq = srate / 2
    b, a = butter(4, [0.5 / nyq, 40 / nyq], btype="band")
    data = filtfilt(b, a, data)

    # Epoch slicing
    epoch_samples = srate * 30
    n_epochs = len(data) // epoch_samples
    if max_epochs:
        n_epochs = min(n_epochs, max_epochs)

    model.eval()
    predictions = []
    for i in range(n_epochs):
        start = i * epoch_samples
        eeg = data[start:start + epoch_samples]
        mean, std = eeg.mean(), eeg.std() + 1e-8
        eeg = (eeg - mean) / std
        x = torch.from_numpy(eeg).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred = int(model(x).argmax(dim=1).item())
        predictions.append(pred)

    results = {"predictions": predictions, "n_epochs": n_epochs}

    # Compare with hypnogram if available
    if hypnogram_path:
        annot = mne.read_annotations(hypnogram_path)
        stage_map = {
            "Sleep stage W": 0, "Sleep stage 1": 1,
            "Sleep stage 2": 2, "Sleep stage 3": 3,
            "Sleep stage 4": 3, "Sleep stage R": 4,
        }
        truth = []
        for a in annot:
            sid = stage_map.get(a["description"], -1)
            n = int(a["duration"] // 30)
            for _ in range(n):
                truth.append(sid)
        truth = truth[:n_epochs]
        results["true_stages"] = truth
        correct = sum(1 for p, t in zip(predictions, truth) if p == t and t >= 0)
        results["accuracy"] = correct / max(len(truth), 1)
        print(f"Quick test: {n_epochs} epochs, "
              f"acc={results['accuracy']*100:.1f}% "
              f"({correct}/{len(truth)})")

    return results
