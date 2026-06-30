# -*- coding: utf-8 -*-
"""
Build per-subject cache of pre-processed context windows.

Run once — training scripts then load from cache instead of re-reading EDF.
Output: {cache_dir}/{sub}_ctx{context}_{mode}.npz

Usage:
    python cache_builder.py                          # ctx=3 center
    python cache_builder.py --context 5              # ctx=5 center
    python cache_builder.py --context 3 --causal     # ctx=3 causal
"""
import sys, os, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
from metabci.brainda.paradigms.sleep import SleepParadigm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=r'F:\sleep-edf')
parser.add_argument('--cache', type=str, default='data_cache')
parser.add_argument('--context', type=int, default=3)
parser.add_argument('--causal', action='store_true')
args = parser.parse_args()

assert args.context % 2 == 1, 'Context must be odd'
mode = 'causal' if args.causal else 'center'
pattern = f'_ctx{args.context}_{mode}.npz'

os.makedirs(args.cache, exist_ok=True)

dataset = SleepEDFDataset(args.data, channel="EEG Fpz-Cz")
paradigm = SleepParadigm(channels=["EEG Fpz-Cz"], srate=100)

done, skipped = 0, 0
for s in dataset.subjects:
    path = os.path.join(args.cache, f"{s}{pattern}")
    if os.path.exists(path):
        skipped += 1
        continue
    try:
        X, y, _ = paradigm.get_data(dataset, subjects=[s], return_concat=True, n_jobs=1)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        n = X.shape[0]
        min_epochs = args.context
        if n < min_epochs:
            print(f"  SKIP {s}: too few epochs ({n})")
            continue
        Xw, yw = [], []
        if args.causal:
            for i in range(args.context - 1, n):
                Xw.append(X[i-args.context+1:i+1, 0, :])
                yw.append(y[i])
        else:
            half = args.context // 2
            for i in range(half, n - half):
                Xw.append(X[i-half:i+half+1, 0, :])
                yw.append(y[i])
        if Xw:
            np.savez_compressed(path, X=np.stack(Xw), y=np.array(yw, dtype=np.int64))
        done += 1
        if done % 10 == 0:
            print(f"  {done} subjects cached...")
    except (ValueError, RuntimeError) as e:
        print(f"  SKIP {s}: {e}")

print(f"\nDone: {done} cached, {skipped} skipped (ctx={args.context} {mode})")
print(f"Cache: {args.cache}/ (*{pattern})")
