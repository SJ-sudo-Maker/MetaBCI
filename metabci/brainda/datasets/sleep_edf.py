# -*- coding: utf-8 -*-
"""
Sleep-EDF Dataset Loader for MetaBCI.

Supports Sleep-EDF Database Expanded (sleep-cassette, sleep-telemetry).
"""
import os
import re
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path

import numpy as np
import mne
from mne.io import Raw
from .base import BaseDataset


class SleepEDFDataset(BaseDataset):
    """Sleep-EDF dataset for sleep staging.

    Parameters
    ----------
    data_root : str
        Path to the sleep-cassette or sleep-telemetry directory.
    subjects : List[str], optional
        Subject IDs to include (e.g. ['4001', '4002']). If None, auto-discovers all.
    channel : str
        EEG channel to use. Default 'EEG Fpz-Cz'.
    epoch_sec : int
        Epoch duration in seconds. Default 30.
    srate : int
        Target sampling rate. Default 100.
    """

    STAGE_MAP = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,   # merged into N3
        "Sleep stage R": 4,
        "Sleep stage ?": -1,
        "Movement time": -1,
    }

    def __init__(
        self,
        data_root: str,
        subjects: Optional[List[str]] = None,
        channel: str = "EEG Fpz-Cz",
        epoch_sec: int = 30,
        srate: int = 100,
    ):
        self.data_root = data_root
        self.channel_name = channel
        self.epoch_sec = epoch_sec

        # Auto-discover subjects by scanning the directory
        available = self._discover_subjects()
        if subjects is None:
            subjects = available
        else:
            subjects = [s for s in subjects if s in available]

        events = {
            "W":   (0, (0, epoch_sec)),
            "N1":  (1, (0, epoch_sec)),
            "N2":  (2, (0, epoch_sec)),
            "N3":  (3, (0, epoch_sec)),
            "REM": (4, (0, epoch_sec)),
        }

        super().__init__(
            dataset_code="sleep-edf",
            subjects=subjects,
            events=events,
            channels=[channel],
            srate=srate,
            paradigm="sleep",
        )

    @staticmethod
    def parse_record_id(record_id: str):
        """Parse Sleep Cassette record ID into (subject_id, night).

        Sleep Cassette format: SC4ssNE0 where ss=subject (00-82), N=night (1-2).
        Example: "4001" → ("00", "1"), "4002" → ("00", "2"), "4101" → ("01", "1").
        """
        s = str(record_id)
        if len(s) >= 3:
            return (s[1:3], s[3:4]) if len(s) == 4 else (s, "1")
        return (s, "1")

    def _discover_subjects(self) -> List[str]:
        """Scan data_root for available record IDs (SC4ssNE0 format)."""
        subjects = set()

        for fname in os.listdir(self.data_root):
            match = re.match(r"SC(\d{4})", fname)
            if match:
                subjects.add(match.group(1))
        return sorted(subjects)

    def get_real_subject_ids(self) -> List[str]:
        """Return unique real subject IDs (without night suffix)."""
        real_ids = set()
        for rid in self.subjects:
            sid, _ = self.parse_record_id(rid)
            real_ids.add(sid)
        return sorted(real_ids)

    def _find_files(self, subject: str) -> Tuple[str, str]:
        """Find PSG and Hypnogram files for a given subject ID."""
        psg_file = None
        hyp_file = None
        prefix = f"SC{subject}"
        for fname in os.listdir(self.data_root):
            if fname.startswith(prefix) and fname.endswith("-PSG.edf"):
                psg_file = os.path.join(self.data_root, fname)
            elif fname.startswith(prefix) and fname.endswith("-Hypnogram.edf"):
                hyp_file = os.path.join(self.data_root, fname)
        if psg_file is None or hyp_file is None:
            raise FileNotFoundError(
                f"Cannot find both PSG and Hypnogram for subject {subject}"
            )
        return psg_file, hyp_file

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        """Return paths for a subject's PSG and Hypnogram files."""
        subject = str(subject)
        psg, hyp = self._find_files(subject)
        return [[psg, hyp]]

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose=None
    ) -> Dict[str, Dict[str, Raw]]:
        """Load a single subject's data as an MNE Raw object with sleep stage annotations."""
        subject = str(subject)
        dests = self.data_path(subject)[0]
        psg_path, hyp_path = dests[0], dests[1]

        # Load PSG
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

        # Pick target channel
        if self.channel_name in raw.ch_names:
            raw.pick([self.channel_name])
        else:
            # Try case-insensitive match
            match = [ch for ch in raw.ch_names if self.channel_name.lower() in ch.lower()]
            if match:
                raw.pick([match[0]])
            else:
                raise ValueError(
                    f"Channel {self.channel_name} not found. Available: {raw.ch_names}"
                )

        # Filter & resample
        raw.filter(0.3, 45, verbose=False)
        raw.resample(self.srate, verbose=False)
        sfreq = raw.info["sfreq"]

        # Load hypnogram and convert to discrete 30s epoch events
        annot = mne.read_annotations(hyp_path)
        events = []
        for ann in annot:
            desc = ann["description"]
            stage_id = self.STAGE_MAP.get(desc, -1)
            if stage_id == -1:
                continue
            n_epochs = int(ann["duration"] // self.epoch_sec)
            for i in range(n_epochs):
                sample = int((ann["onset"] + i * self.epoch_sec) * sfreq)
                events.append([sample, 0, stage_id])

        if len(events) == 0:
            raise RuntimeError(f"No valid sleep stages found for subject {subject}")

        events = np.array(events)
        # Ensure strict chronological order
        events = events[events[:, 0].argsort()]

        # Create annotations with proper 30s durations per epoch.
        # Using mne.Annotations directly (not annotations_from_events) ensures
        # each epoch has duration=30s, which extract_epochs() requires.
        onsets = events[:, 0].astype(float) / sfreq
        durations = np.full(len(events), self.epoch_sec, dtype=float)
        descriptions = [str(e) for e in events[:, 2]]
        annotations = mne.Annotations(
            onset=onsets, duration=durations, description=descriptions,
            orig_time=raw.info.get('meas_date', None)
        )
        raw.set_annotations(annotations)

        return {"session_0": {"run_0": raw}}
