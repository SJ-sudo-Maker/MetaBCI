# -*- coding: utf-8 -*-
"""
End-to-end demo: EDF playback → SleepOnlineWorker → real-time hypnogram.

Usage
-----
    python demo_e2e.py

Requires a trained ParaSleep model at MODEL_PATH (default: parasleep.pth).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from metabci.brainflow.edf_player import EDFSleepPlayer
from metabci.brainflow.sleep_worker import SleepOnlineWorker
# (no Marker needed — we handle epochs directly)
from metabci.brainstim.sleep_monitor import STAGE_COLORS, STAGE_NAMES
import metabci.brainda.algorithms.deep_learning.parasleep as lwmod


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = r"D:\sleep eeg\sleep-edf-database-expanded-1.0.0\sleep-cassette"
MODEL_PATH = "parasleep.pth"          # put trained .pth here
SPEED = 30.0                           # 30x real-time for quick demo
MAX_EPOCHS = 120                       # show first 1 hour
SUBJECT = None                         # None = auto-pick first available

# =============================================================================
# Setup
# =============================================================================

# Find a test subject not used in training (e.g., subject after index 90)
files = sorted(os.listdir(DATA_ROOT))
for f in files:
    if f.endswith("-PSG.edf") and (SUBJECT is None or SUBJECT in f):
        prefix = f[:6]
        hyp = None
        for f2 in files:
            if f2.startswith(prefix) and f2.endswith("-Hypnogram.edf"):
                hyp = os.path.join(DATA_ROOT, f2)
                break
        edf = os.path.join(DATA_ROOT, f)
        if hyp:
            break

print(f"EDF:  {os.path.basename(edf)}")
print(f"Hyp:  {os.path.basename(hyp)}")

# Load model
raw_cls = lwmod.ParaSleep.module
model = raw_cls(n_channels=3, n_samples=3000, n_classes=5).float()
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"Model: loaded ({sum(p.numel() for p in model.parameters()):,} params)")
else:
    print("Model: WARNING — using untrained model (random predictions)!")

model.eval()

# =============================================================================
# Run playback with real-time hypnogram
# =============================================================================

player = EDFSleepPlayer(edf, channel="EEG Fpz-Cz", srate=100,
                        hypnogram_path=hyp, speed=SPEED, verbose=False)
worker = SleepOnlineWorker(model=model, srate=100, epoch_sec=30)

# (bypass worker registration — consume directly in main thread)

# Set up real-time plot
plt.ion()
fig, (ax_hypno, ax_text) = plt.subplots(2, 1, figsize=(14, 6),
    gridspec_kw={"height_ratios": [4, 1]})
fig.canvas.manager.set_window_title("MetaBCI Sleep Monitor — Live Demo")

epoch_sec = 30
stages_display = []
worker.pre()                                 # init model + LSL

player._exit.clear()
start = time.perf_counter()

# Accumulate 3000 samples per epoch (bypass ProcessWorker subprocess)
epoch_buffer = []
try:
    while len(stages_display) < MAX_EPOCHS:
        samples = player.recv()
        if not samples or player.position >= player.n_samples:
            break

        epoch_buffer.extend(samples)
        # When we have enough samples for one epoch, run inference
        if len(epoch_buffer) >= 3000:
            epoch_data = epoch_buffer[:3000]
            epoch_buffer = epoch_buffer[3000:]
            worker.consume(epoch_data)           # direct call, no subprocess

        # Refresh display
        stages_display = [p for p in worker.predictions if p >= 0]
        if stages_display:
            ax_hypno.clear()
            ax_text.clear()

            n = len(stages_display)
            taxis = np.arange(n) * epoch_sec / 60

            # Draw hypnogram
            for i in range(n):
                color = STAGE_COLORS.get(stages_display[i], "#888888")
                ax_hypno.fill_between(
                    [taxis[i], taxis[min(i+1, n-1)]],
                    5.5, -0.5, color=color, alpha=0.85,
                )
            ax_hypno.set_yticks([0, 1, 2, 3, 4])
            ax_hypno.set_yticklabels(STAGE_NAMES)
            ax_hypno.set_ylim(-0.5, 5.5)
            ax_hypno.invert_yaxis()
            ax_hypno.set_xlabel("Time (min)")
            ax_hypno.set_ylabel("Sleep Stage")
            ax_hypno.set_title(
                f"实时睡眠监测 — {n * epoch_sec / 60:.0f} min "
                f"(当前: {STAGE_NAMES[stages_display[-1]]})"
            )

            # Ground truth comparison
            true = player.get_true_stage_at(n - 1)
            if true >= 0:
                color_true = STAGE_COLORS.get(true, "#888")
                correct = "(✓)" if true == stages_display[-1] else "(✗)"
                ax_text.text(0.5, 0.5,
                    f"预测: {STAGE_NAMES[stages_display[-1]]}  |  "
                    f"真实: {STAGE_NAMES[true]} {correct}",
                    transform=ax_text.transAxes, ha="center", fontsize=14,
                    fontweight="bold")
            ax_text.axis("off")

            plt.tight_layout()
            plt.pause(0.05)

        # Timing for smooth playback
        elapsed = time.perf_counter() - start
        expected = (player.position / player.target_srate) / SPEED
        if expected > elapsed:
            time.sleep(expected - elapsed)

finally:
    worker.post()
    plt.ioff()
    plt.close()

    print(f"\nDemo finished: {len(stages_display)} epochs processed.")

    # Accuracy vs ground truth
    truth = [player.get_true_stage_at(i) for i in range(len(stages_display))]
    correct = sum(1 for p, t in zip(stages_display, truth) if p == t >= 0)
    n_valid = sum(1 for t in truth if t >= 0)
    if n_valid > 0:
        print(f"Accuracy vs ground truth: {correct/n_valid*100:.1f}% "
              f"({correct}/{n_valid})")
