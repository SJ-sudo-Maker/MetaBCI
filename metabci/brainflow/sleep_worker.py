# -*- coding: utf-8 -*-
"""
SleepOnlineWorker: real-time sleep stage inference for brainflow.

Supports continuous LSL streaming and online prediction with ParaSleep.
"""

from typing import Optional, List, Dict
from collections import deque

import numpy as np
from scipy.signal import butter, lfilter

from .workers import ProcessWorker


class SleepOnlineWorker(ProcessWorker):
    """Online sleep staging worker for continuous EEG streaming.

    Buffers 30-second epochs from a single-channel EEG stream, applies
    bandpass filtering (0.5–40 Hz), and runs ParaSleep inference every
    epoch. Predictions are pushed via an LSL outlet for downstream
    visualization or recording.

    Parameters
    ----------
    model : torch.nn.Module
        Trained ParaSleep model (float32, already loaded).
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
    >>> from metabci.brainda.algorithms.deep_learning.parasleep import ParaSleep
    >>> model = ParaSleep.module(1, 3000, 5).float()
    >>> model.load_state_dict(torch.load("parasleep.pth"))
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
        self._filter_zi = None  # continuous filter state (lfilter)

        # 3-epoch context buffer (for windowed inference)
        self._epoch_buffer = deque(maxlen=3)

        # Sleep stage history
        self.stage_names = ["W", "N1", "N2", "N3", "REM"]
        self.predictions: List[int] = []
        self.confidences: List[float] = []
        self.signal_quality: List[bool] = []
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
        """
        import torch

        arr = np.array(data, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            eeg = arr[:, 0]
        else:
            eeg = arr.squeeze()

        # Handle variable-length input
        if len(eeg) < self.epoch_samples:
            eeg = np.pad(eeg, (0, self.epoch_samples - len(eeg)), mode="edge")
        elif len(eeg) > self.epoch_samples:
            eeg = eeg[:self.epoch_samples]

        # Signal quality check: flat line or saturation = electrode problem
        ptp = np.ptp(eeg)  # peak-to-peak amplitude
        is_good = 5.0 < ptp < 500.0  # reasonable µV range for EEG
        self.signal_quality.append(is_good)

        if not is_good:
            print(f"[SleepWorker] epoch {self.epoch_counter+1:4d} | "
                  f"BAD SIGNAL (ptp={ptp:.1f}µV) — skipping inference")
            self.epoch_counter += 1
            self.predictions.append(-1)
            self.stage_counts["?"] = self.stage_counts.get("?", 0) + 1
            self._epoch_buffer.append(None)
            return

        # Causal bandpass filter (lfilter — real-time safe, no future leakage)
        if self._filter_zi is None:
            self._filter_zi = np.zeros(max(len(self._b), len(self._a)) - 1)
        eeg_filt, self._filter_zi = lfilter(
            self._b, self._a, eeg, zi=self._filter_zi * eeg[0],
        )

        # Z-score normalize
        mean = eeg_filt.mean()
        std = eeg_filt.std() + 1e-8
        eeg_norm = (eeg_filt - mean) / std

        # 3-epoch context buffer
        self._epoch_buffer.append(eeg_norm)
        if len(self._epoch_buffer) < 3:
            # Not enough context yet — skip or predict from partial buffer
            print(f"[SleepWorker] epoch {self.epoch_counter+1:4d} | "
                  f"buffering ({len(self._epoch_buffer)}/3)...")
            self.epoch_counter += 1
            self.predictions.append(-1)
            return

        # Stack 3 epochs as channels: (1, 3, 3000)
        ctx = np.stack([
            self._epoch_buffer[0],
            self._epoch_buffer[1],
            self._epoch_buffer[2],
        ], axis=0)
        x = torch.from_numpy(ctx).float().unsqueeze(0)  # (1, 3, 3000)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            confidence, pred = probs.max(dim=1)
            pred = pred.item()
            confidence = confidence.item()

        # Record
        self.epoch_counter += 1
        self.predictions.append(pred)
        self.confidences.append(confidence)
        self.stage_counts[self.stage_names[pred]] += 1

        # Push via LSL
        if self.lsl_outlet is not None:
            self.lsl_outlet.push_sample([self.stage_names[pred]])

        print(
            f"[SleepWorker] epoch {self.epoch_counter:4d} | "
            f"stage: {self.stage_names[pred]} ({confidence:.2%})"
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
