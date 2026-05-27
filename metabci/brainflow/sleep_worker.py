# -*- coding: utf-8 -*-
"""
SleepOnlineWorker: real-time sleep stage inference for brainflow.

Supports continuous LSL streaming and online prediction with LWSleepNet.
"""

from typing import Optional, List, Dict
from collections import deque

import numpy as np
from scipy.signal import butter, filtfilt

from .workers import ProcessWorker


class SleepOnlineWorker(ProcessWorker):
    """Online sleep staging worker for continuous EEG streaming.

    Buffers 30-second epochs from a single-channel EEG stream, applies
    bandpass filtering (0.5–40 Hz), and runs LWSleepNet inference every
    epoch. Predictions are pushed via an LSL outlet for downstream
    visualization or recording.

    Parameters
    ----------
    model : torch.nn.Module
        Trained LWSleepNet model (float32, already loaded).
    srate : int
        Sampling rate (Hz). Must match model training. Default 100.
    epoch_sec : int
        Epoch duration in seconds. Default 30.
    channel_name : str
        EEG channel label for logging. Default "EEG Fpz-Cz".
    timeout : float
        Queue read timeout for the worker loop. Default 1e-3.
    worker_name : str
        Worker identifier for logging. Default "sleep_worker".

    Examples
    --------
    >>> from metabci.brainflow.sleep_worker import SleepOnlineWorker
    >>> from metabci.brainflow.amplifiers import LSLapps, Marker
    >>>
    >>> # Load trained model first (in main process)
    >>> import torch
    >>> from metabci.brainda.algorithms.deep_learning.lwsleepnet import LWSleepNet
    >>> model = LWSleepNet.module(1, 3000, 5).float()
    >>> model.load_state_dict(torch.load("lwsleepnet.pth"))
    >>> model.eval()
    >>>
    >>> worker = SleepOnlineWorker(model=model)
    >>> amp = LSLapps(host="...", srate=100, num_chans=1, timeout=1e-2)
    >>> marker = Marker(interval=[0, 30], srate=100, events=None)
    >>> amp.register_worker("sleep", worker, marker)
    >>> amp.up_worker("sleep")
    >>> amp.start_trans()
    >>> # ... run for desired duration ...
    >>> amp.stop_trans()
    >>> amp.down_worker("sleep")
    >>> amp.clear()
    """

    def __init__(
        self,
        model,
        srate: int = 100,
        epoch_sec: int = 30,
        channel_name: str = "EEG Fpz-Cz",
        timeout: float = 1e-3,
        worker_name: str = "sleep_worker",
    ):
        self.model = model
        self.srate = srate
        self.epoch_sec = epoch_sec
        self.epoch_samples = srate * epoch_sec
        self.channel_name = channel_name

        # Bandpass filter (0.5–40 Hz, 4th-order Butterworth)
        nyq = srate / 2.0
        self._b, self._a = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")

        # Sleep stage history
        self.stage_names = ["W", "N1", "N2", "N3", "REM"]
        self.predictions: List[int] = []
        self.stage_counts: Dict[str, int] = {s: 0 for s in self.stage_names}
        self.epoch_counter: int = 0

        # Optional LSL outlet (created in pre() if lsl is available)
        self.lsl_outlet = None
        self.lsl_available = False

        super().__init__(timeout=timeout, name=worker_name)

    # ------------------------------------------------------------------
    # ProcessWorker interface
    # ------------------------------------------------------------------

    def pre(self):
        """Called once in the worker process before the consume loop.

        Sets up model for inference and optionally creates an LSL outlet
        for broadcasting sleep stage predictions.
        """
        import torch

        self._torch = torch

        # Move model to CPU for inference (safe in worker process)
        self.model.eval()
        self.model.to("cpu")

        # Try to set up LSL outlet for predictions
        try:
            from pylsl import StreamInfo, StreamOutlet

            info = StreamInfo(
                "SleepStage",
                "Markers",
                1,
                0,  # irregular rate
                "string",
                f"sleep-stage-{id(self)}",
            )
            self.lsl_outlet = StreamOutlet(info)
            self.lsl_available = True
        except Exception:
            self.lsl_available = False

        print(f"[SleepWorker] pre() complete. Model ready. LSL: {self.lsl_available}")

    def consume(self, data):
        """Process one 30-second epoch.

        Parameters
        ----------
        data : list of list
            Raw EEG samples, each inner list is [ch_data, trigger].
            Length should be epoch_samples (3000 for 30s @ 100Hz).
        """
        import torch

        # Convert to numpy: shape (n_samples, n_chans+1) -> (n_samples,)
        arr = np.array(data, dtype=np.float64)

        # Extract EEG channel (first column, exclude trigger marker in last col)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            eeg = arr[:, 0]  # first column = EEG
        else:
            eeg = arr.squeeze()

        # Handle variable-length input
        if len(eeg) < self.epoch_samples:
            # Pad with edge value
            eeg = np.pad(eeg, (0, self.epoch_samples - len(eeg)), mode="edge")
        elif len(eeg) > self.epoch_samples:
            eeg = eeg[:self.epoch_samples]

        # Bandpass filter
        eeg = filtfilt(self._b, self._a, eeg)

        # Z-score normalize
        mean = eeg.mean()
        std = eeg.std() + 1e-8
        eeg = (eeg - mean) / std

        # Reshape to model input: (1, 1, 3000)
        x = torch.from_numpy(eeg).float().unsqueeze(0).unsqueeze(0)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            pred = int(logits.argmax(dim=1).item())

        # Record
        self.epoch_counter += 1
        self.predictions.append(pred)
        self.stage_counts[self.stage_names[pred]] += 1

        # Push via LSL
        if self.lsl_outlet is not None:
            self.lsl_outlet.push_sample([self.stage_names[pred]])

        print(
            f"[SleepWorker] epoch {self.epoch_counter:4d} | "
            f"stage: {self.stage_names[pred]}"
        )

    def post(self):
        """Called after the main loop exits.

        Prints a summary of sleep stage statistics.
        """
        total = sum(self.stage_counts.values())
        if total == 0:
            print("[SleepWorker] post() — no data processed.")
            return

        print(f"\n{'='*40}")
        print(f"Sleep Report — {self.epoch_counter} epochs")
        print(f"{'='*40}")
        print(f"Total time: {total * self.epoch_sec / 60:.1f} min")
        for name in self.stage_names:
            n = self.stage_counts[name]
            pct = 100 * n / total
            bar = "#" * int(pct / 2)
            print(f"  {name:>4s}: {n:4d} ({pct:5.1f}%) {bar}")
        print(f"{'='*40}")
