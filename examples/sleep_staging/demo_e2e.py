# -*- coding: utf-8 -*-
"""
End-to-end demo: EDF playback → SleepOnlineWorker → real-time hypnogram.

Usage
-----
    python demo_e2e.py

Default demo uses the final ctx=3 causal ParaSleep model:
    MODEL_PATH = "exp_ctx3_causal.pth"
    SUBJECT = "4241"
Set DEMO_CONTEXT to match the model's training context window size.
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
from metabci.brainstim.sleep_monitor import STAGE_COLORS, STAGE_NAMES, STAGE_Y_POS
import metabci.brainda.algorithms.deep_learning.parasleep as lwmod


# =============================================================================
# Configuration
# =============================================================================
DATA_ROOT = os.environ.get("SLEEP_DATA", r"D:\sleep eeg\sleep-edf-database-expanded-1.0.0\sleep-cassette")
CACHE_DIR = os.environ.get("SLEEP_CACHE", r"C:\Users\lenovo\Desktop\MetaBCI-sleep\data_cache")

# Final online model used in the competition demo
MODEL_PATH = "exp_ctx3_causal.pth"

SPEED = 300.0
MAX_EPOCHS = 720
RENDER_EVERY = 5     # only redraw every N epochs (higher = faster video)

# Change this line only if you want another demo subject.
# Available test examples: "4031", "4241", "4261", "4281", "4412", "4551", "4591", "4621", "4622", "4672"
SUBJECT = "4241"

DEMO_MODE = "cache"
DEMO_CONTEXT = 3
DEMO_CAUSAL = True

MODE_STR = "causal" if DEMO_CAUSAL else "center"

# =============================================================================
# Setup
# =============================================================================

if DEMO_MODE == "cache":
    # Load directly from cache (matches training data exactly)
    import glob as _glob
    cache_dir = CACHE_DIR
    # Try new naming first, fall back to old patterns
    ctx = DEMO_CONTEXT
    if DEMO_CAUSAL:
        search_pats = [f'*_FpzCz_sr100_ctx{ctx}_causal_5class.npz',
                       f'*_ctx{ctx}_causal.npz']
    else:
        search_pats = [f'*_FpzCz_sr100_ctx{ctx}_center_5class.npz',
                       f'*_ctx{ctx}_center.npz',
                       '*.npz']
    for pat in search_pats:
        matches = _glob.glob(os.path.join(cache_dir, pat))
        if matches:
            break
    if SUBJECT is not None:
        sub = str(SUBJECT)
    else:
        sub = os.path.basename(matches[0]).rsplit('_', 1)[0].split('_')[0] if matches else '4032'
    print(f"Cache mode: subject {sub} (ctx={ctx} {MODE_STR})")

    # Find cache file. In causal mode, only causal cache is allowed to avoid accidental center-cache demo.
    cache_path = None
    if DEMO_CAUSAL:
        cache_candidates = [
            os.path.join(cache_dir, f'{sub}_FpzCz_sr100_ctx{ctx}_causal_5class.npz'),
            os.path.join(cache_dir, f'{sub}_ctx{ctx}_causal.npz'),
        ]
    else:
        cache_candidates = [
            os.path.join(cache_dir, f'{sub}_FpzCz_sr100_ctx{ctx}_center_5class.npz'),
            os.path.join(cache_dir, f'{sub}_ctx{ctx}_center.npz'),
            os.path.join(cache_dir, f'{sub}.npz'),
        ]

    for fmt in cache_candidates:
        if os.path.exists(fmt):
            cache_path = fmt
            break

    if cache_path is None:
        raise FileNotFoundError(
            f"Cannot find cache file for subject={sub}, ctx={ctx}, mode={MODE_STR} in {cache_dir}"
        )

    print(f"Cache file: {os.path.basename(cache_path)}")
    d = np.load(cache_path)
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
    # Original EDF playback mode
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
    # Auto-detect architecture and context from state dict
    use_ta = any(k.startswith('transformer.') for k in state.keys())
    # Detect context from checkpoint
    if use_ta and 'pos_embed' in state:
        n_ch = state['pos_embed'].shape[1]
    elif 'mrfe.small_branch.dsconv.dw.weight' in state:
        n_ch = state['mrfe.small_branch.dsconv.dw.weight'].shape[0]
    else:
        n_ch = DEMO_CONTEXT
    target_idx = 'last' if (use_ta and DEMO_CAUSAL) else 'center'
    raw_cls = lwmod.ParaSleep.module
    model = raw_cls(n_channels=n_ch, n_samples=3000, n_classes=5,
                    use_temporal_attention=use_ta,
                    target_index=target_idx).float()
    model.load_state_dict(state, strict=False)
    arch = 'TA' if use_ta else 'Base'
    param_count = sum(p.numel() for p in model.parameters())
    model_info = f"Model: {MODEL_PATH} | {arch} | ctx={n_ch} | {MODE_STR}"
    print(f"Model: loaded ({arch}, ctx={n_ch}, {param_count:,} params)")
    print(f"Demo model info: {model_info}")
else:
    print("Model: WARNING — using untrained model (random predictions)!")
    raw_cls = lwmod.ParaSleep.module
    model = raw_cls(n_channels=DEMO_CONTEXT, n_samples=3000, n_classes=5).float()
    model_info = f"Model: RANDOM UNTRAINED | Base | ctx={DEMO_CONTEXT} | {MODE_STR}"

model.eval()

# =============================================================================
# Run (cache mode: direct prediction, no EDF pipeline)
# =============================================================================
if DEMO_MODE == "cache":
    plt.ion()
    fig, (ax_hypno, ax_text) = plt.subplots(2, 1, figsize=(14, 6),
        gridspec_kw={"height_ratios": [4, 1]})
    fig.canvas.manager.set_window_title(f"MetaBCI Sleep Monitor — {model_info}")

    with torch.no_grad():
        out = model(torch.from_numpy(X_cache))
        if isinstance(out, tuple):
            out = out[0]
        preds_all = out.argmax(1).numpy()

    for n in range(1, len(preds_all) + 1):
        if n > 1 and n % RENDER_EVERY != 0 and n != len(preds_all):
            continue  # skip frames for speed
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
        ax_hypno.set_title(
            f"MetaBCI Sleep Monitor | {model_info} | Subject {sub} | {n*0.5:.0f} min",
            fontsize=12,
            fontweight="bold"
        )

        true_cur = int(true_part[-1])
        correct = "[OK]" if cur == true_cur else "[X]"
        ax_text.text(0.5, 0.58, f"{model_info}",
                     transform=ax_text.transAxes, ha="center",
                     fontsize=11, fontweight="bold")
        ax_text.text(0.5, 0.22,
            f"Predict: {STAGE_NAMES[cur]}  |  Truth: {STAGE_NAMES[true_cur]}  {correct}",
            transform=ax_text.transAxes, ha="center",
            fontsize=14, fontweight="bold")
        ax_text.axis("off")
        plt.tight_layout(); plt.pause(0.001)

    plt.ioff(); plt.close()
    from collections import Counter
    cnt = Counter(preds_all.tolist())
    print(f"\nFinal predictions: W={cnt.get(0,0)} N1={cnt.get(1,0)} N2={cnt.get(2,0)} N3={cnt.get(3,0)} REM={cnt.get(4,0)}")
    true_cnt = Counter(y_cache.tolist())
    print(f"Ground truth:      W={true_cnt.get(0,0)} N1={true_cnt.get(1,0)} N2={true_cnt.get(2,0)} N3={true_cnt.get(3,0)} REM={true_cnt.get(4,0)}")
    acc = np.mean(preds_all == y_cache)
    print(f"Accuracy: {acc*100:.1f}%")
    print("Done.")
    exit()

# EDF mode
player = EDFSleepPlayer(edf, channel="EEG Fpz-Cz", srate=100,
                        hypnogram_path=hyp, speed=SPEED, chunk_size=3000,
                        verbose=False)
worker = SleepOnlineWorker(model=model, srate=100, epoch_sec=30,
                           causal=DEMO_CAUSAL, prefiltered=True,
                           context=DEMO_CONTEXT)

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
fig.canvas.manager.set_window_title(f"MetaBCI Sleep Monitor — {model_info}")

epoch_sec = 30
stages_display = []
worker.pre()

player._exit.clear()
start = time.perf_counter()

epoch_buffer = []
try:
    while len(stages_display) < MAX_EPOCHS:
        samples = player.recv()
        if not samples or player.position >= player.n_samples:
            break

        epoch_buffer.extend(samples)
        if len(epoch_buffer) >= 3000:
            epoch_data = epoch_buffer[:3000]
            epoch_buffer = epoch_buffer[3000:]
            epoch_data = [[v*1e6, t] for v, t in epoch_data]
            worker.consume(epoch_data)

        stages_display = [p for p in worker.predictions if p >= 0]
        if stages_display:
            ax_hypno.clear()
            ax_text.clear()

            n = len(stages_display)
            time_hours = np.arange(n + 1) * epoch_sec / 3600

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

            cur = stages_display[-1]
            ax_hypno.text(0.99, 0.95, f"Current: {STAGE_NAMES[cur]}",
                          transform=ax_hypno.transAxes, ha="right", va="top",
                          fontsize=14, fontweight="bold", color=STAGE_COLORS[cur],
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

            ax_hypno.set_title(
                f"MetaBCI Sleep Monitor | {model_info} | {n * epoch_sec / 60:.0f} min",
                fontsize=12, fontweight="bold"
            )

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

        elapsed = time.perf_counter() - start
        expected = (player.position / player.target_srate) / SPEED
        if expected > elapsed:
            time.sleep(expected - elapsed)

finally:
    worker.post()
    plt.ioff()
    plt.close()

    print(f"\nDemo finished: {len(stages_display)} epochs processed.")

    truth = [player.get_true_stage_at(i) for i in range(len(stages_display))]
    correct = sum(1 for p, t in zip(stages_display, truth) if p == t >= 0)
    n_valid = sum(1 for t in truth if t >= 0)
    if n_valid > 0:
        print(f"Accuracy vs ground truth: {correct/n_valid*100:.1f}% "
              f"({correct}/{n_valid})")
