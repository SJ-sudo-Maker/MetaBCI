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
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from metabci.brainflow.edf_player import EDFSleepPlayer
from metabci.brainflow.sleep_worker import SleepOnlineWorker
# (no Marker needed — we handle epochs directly)
from metabci.brainstim.sleep_monitor import STAGE_COLORS, STAGE_NAMES, STAGE_Y_POS
import metabci.brainda.algorithms.deep_learning.parasleep as lwmod


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = r"F:\sleep-edf\sleep-edf-database-expanded-1.0.0\sleep-cassette"
MODEL_PATH = "parasleep_best.pth"          # put trained .pth here
SPEED = 300.0                          # 300x real-time for quick demo
MAX_EPOCHS = 720                       # show 6 hours
SUBJECT = None                         # None = auto-pick first available
DEMO_MODE = "cache"                    # "cache" = pre-built npz (best), "edf" = raw EDF playback

# =============================================================================
# Setup
# =============================================================================

if DEMO_MODE == "cache":
    # Load directly from cache (matches training data exactly)
    cache_dir = "data_cache" if os.path.isdir("data_cache") else r"F:\sleep_cache"
    subs = sorted([f.replace('.npz','') for f in os.listdir(cache_dir) if f.endswith('.npz')])
    sub = '4032'  # best REM+N3 mix for demo video
    d = np.load(os.path.join(cache_dir, f'{sub}.npz'))
    X_cache, y_cache = d['X'], d['y']
    # Find sleep onset
    onset = 0
    for i in range(len(y_cache) - 3):
        if y_cache[i] > 1 and y_cache[i+1] > 1 and y_cache[i+2] > 1:
            onset = max(0, i - 5)
            break
    X_cache = X_cache[onset:onset + MAX_EPOCHS]
    y_cache = y_cache[onset:onset + MAX_EPOCHS]
    from collections import Counter
    print(f"Cache mode: subject {sub}, {len(X_cache)} epochs from onset {onset}")
    print(f"Ground truth: {dict(Counter(y_cache.tolist()))}")
    player = None
    hyp = None
else:
    # Original EDF playback mode (unchanged)
    DATA_ROOT = r"F:\sleep-edf\sleep-edf-database-expanded-1.0.0\sleep-cassette"
    files = sorted(os.listdir(DATA_ROOT))
    SKIP = 120
    count = 0
    for f in files:
        if f.endswith("-PSG.edf"):
            if count < SKIP:
                count += 1
                continue
            prefix = f[:6]
            hyp = None
            for f2 in files:
                if f2.startswith(prefix) and f2.endswith("-Hypnogram.edf"):
                    hyp = os.path.join(DATA_ROOT, f2)
                    break
            if hyp:
                edf = os.path.join(DATA_ROOT, f)
                break
    print(f"EDF:  {os.path.basename(edf)}")
    print(f"Hyp:  {os.path.basename(hyp)}")

# Load model (auto-detect architecture from checkpoint)
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    # Detect TA architecture from state dict
    use_ta = any(k.startswith('transformer.') for k in state.keys())
    raw_cls = lwmod.ParaSleep.module
    model = raw_cls(n_channels=3, n_samples=3000, n_classes=5,
                    use_temporal_attention=use_ta).float()
    model.load_state_dict(state, strict=False)
    arch = 'TA' if use_ta else 'Base'
    print(f"Model: loaded ({arch}, {sum(p.numel() for p in model.parameters()):,} params)")
else:
    print("Model: WARNING — using untrained model (random predictions)!")
    raw_cls = lwmod.ParaSleep.module
    model = raw_cls(n_channels=3, n_samples=3000, n_classes=5).float()

model.eval()

# =============================================================================
# Run (cache mode: direct prediction, no EDF pipeline)
# =============================================================================
if DEMO_MODE == "cache":
    plt.ion()
    fig, (ax_hypno, ax_text) = plt.subplots(2, 1, figsize=(14, 6),
        gridspec_kw={"height_ratios": [4, 1]})
    fig.canvas.manager.set_window_title("MetaBCI Sleep Monitor — Cache Demo")

    with torch.no_grad():
        preds_all = model(torch.from_numpy(X_cache)).argmax(1).numpy()

    for n in range(1, len(preds_all) + 1):
        ax_hypno.clear(); ax_text.clear()
        time_hours = np.arange(n + 1) * 30 / 3600
        preds_part = preds_all[:n]
        true_part = y_cache[:n]

        for i in range(n):
            color = STAGE_COLORS.get(int(preds_part[i]), "#888888")
            yb = STAGE_Y_POS[preds_part[i]] - 0.5
            yt = STAGE_Y_POS[preds_part[i]] + 0.5
            ax_hypno.fill_between([time_hours[i], time_hours[i+1]], yb, yt,
                                  color=color, alpha=0.95, edgecolor='white', linewidth=0.3)

        ax_hypno.set_yticks([STAGE_Y_POS[s] for s in range(5)])
        ax_hypno.set_yticklabels(STAGE_NAMES, fontsize=10, fontweight='bold')
        ax_hypno.set_ylim(-0.8, 7.8); ax_hypno.invert_yaxis()
        ax_hypno.set_xlim(0, max(time_hours[-1], 0.01))
        ax_hypno.set_xlabel("Time (hours)", fontsize=11)
        ax_hypno.set_ylabel("Sleep Stage", fontsize=11)

        cur = int(preds_part[-1])
        ax_hypno.text(0.99, 0.95, f"Current: {STAGE_NAMES[cur]}",
                      transform=ax_hypno.transAxes, ha="right", va="top",
                      fontsize=14, fontweight="bold", color=STAGE_COLORS[cur],
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        ax_hypno.set_title(f"MetaBCI Sleep Monitor — {n*0.5:.0f} min", fontsize=12, fontweight="bold")

        true_cur = int(true_part[-1])
        correct = "[OK]" if cur == true_cur else "[X]"
        ax_text.text(0.5, 0.5, f"Predict: {STAGE_NAMES[cur]}  |  Truth: {STAGE_NAMES[true_cur]}  {correct}",
                     transform=ax_text.transAxes, ha="center", fontsize=14, fontweight="bold")
        ax_text.axis("off")
        plt.tight_layout(); plt.pause(0.02)

    plt.ioff(); plt.close()
    # Print final report
    from collections import Counter
    cnt = Counter(preds_all.tolist())
    print(f"\nFinal predictions: W={cnt.get(0,0)} N1={cnt.get(1,0)} N2={cnt.get(2,0)} N3={cnt.get(3,0)} REM={cnt.get(4,0)}")
    true_cnt = Counter(y_cache.tolist())
    print(f"Ground truth:      W={true_cnt.get(0,0)} N1={true_cnt.get(1,0)} N2={true_cnt.get(2,0)} N3={true_cnt.get(3,0)} REM={true_cnt.get(4,0)}")
    acc = np.mean(preds_all == y_cache)
    print(f"Accuracy: {acc*100:.1f}%")
    print("Done.")
    exit()

player = EDFSleepPlayer(edf, channel="EEG Fpz-Cz", srate=100,
                        hypnogram_path=hyp, speed=SPEED, chunk_size=3000,
                        verbose=False)
worker = SleepOnlineWorker(model=model, srate=100, epoch_sec=30, causal=False, prefiltered=True)

# (bypass worker registration — consume directly in main thread)

SKIP_WAKE = False  # show full night with W→N2→N3→REM transitions

# Fast-forward to sleep onset (skip initial wake period)
if SKIP_WAKE:
    ff_epochs = 0
    while player.position < player.n_samples and ff_epochs < 2000:
        buf = []
        while len(buf) < 3000 and player.position < player.n_samples:
            s = player.recv()
            if not s: break
            buf.extend(s)
        if len(buf) < 3000: break
        ff_epochs += 1
        s3 = player.get_true_stage_at(ff_epochs - 1)
        if s3 is not None and s3 > 1:
            break
    print(f"Skipped {ff_epochs} awake epochs, starting from epoch {ff_epochs}")

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
            # Convert V -> uV (MNE reads EDF in Volts, worker expects uV)
            epoch_data = [[v*1e6, t] for v, t in epoch_data]
            worker.consume(epoch_data)           # direct call, no subprocess

        # Refresh display
        stages_display = [p for p in worker.predictions if p >= 0]
        if stages_display:
            ax_hypno.clear()
            ax_text.clear()

            n = len(stages_display)
            time_hours = np.arange(n + 1) * epoch_sec / 3600

            # Clinical step-plot hypnogram
            for i in range(n):
                color = STAGE_COLORS.get(stages_display[i], "#888888")
                yb = STAGE_Y_POS[stages_display[i]] - 0.5
                yt = STAGE_Y_POS[stages_display[i]] + 0.5
                ax_hypno.fill_between(
                    [time_hours[i], time_hours[i + 1]], yb, yt,
                    color=color, alpha=0.95, edgecolor='white', linewidth=0.3,
                )

            ax_hypno.set_yticks([STAGE_Y_POS[s] for s in range(5)])
            ax_hypno.set_yticklabels(STAGE_NAMES, fontsize=10, fontweight='bold')
            ax_hypno.set_ylim(-0.8, 7.8)
            ax_hypno.invert_yaxis()
            ax_hypno.set_xlim(0, max(time_hours[-1], 0.01))
            ax_hypno.set_xlabel("Time (hours)", fontsize=11)
            ax_hypno.set_ylabel("Sleep Stage", fontsize=11)
            ax_hypno.grid(axis='x', which='major', color='#CCCCCC', linewidth=0.5, alpha=0.7)

            # Current stage indicator
            cur = stages_display[-1]
            ax_hypno.text(0.99, 0.95, f"Current: {STAGE_NAMES[cur]}",
                          transform=ax_hypno.transAxes, ha="right", va="top",
                          fontsize=14, fontweight="bold", color=STAGE_COLORS[cur],
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

            ax_hypno.set_title(
                f"MetaBCI Sleep Monitor  —  {n * epoch_sec / 60:.0f} min",
                fontsize=12, fontweight="bold"
            )

            # Ground truth comparison
            true = player.get_true_stage_at(n - 1)
            if true >= 0:
                color_true = STAGE_COLORS.get(true, "#888")
                correct = "[OK]" if true == stages_display[-1] else "[X]"
                ax_text.text(0.5, 0.5,
                    f"Predict: {STAGE_NAMES[stages_display[-1]]}  |  "
                    f"Truth:  {STAGE_NAMES[true]}  {correct}",
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
