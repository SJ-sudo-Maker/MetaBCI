# -*- coding: utf-8 -*-
"""
Build per-subject cache of pre-processed 3-epoch windows.

Run once — training scripts then load from cache instead of re-reading EDF.
Output: data_cache/0001.npz ... 0153.npz
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os, numpy as np

from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
from metabci.brainda.paradigms.sleep import SleepParadigm

DATA_ROOT = r"D:\sleep eeg\sleep-edf-database-expanded-1.0.0\sleep-cassette"
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

dataset = SleepEDFDataset(DATA_ROOT, channel="EEG Fpz-Cz")
paradigm = SleepParadigm(channels=["EEG Fpz-Cz"], srate=100)

done, skipped = 0, 0
for s in dataset.subjects:
    path = os.path.join(CACHE_DIR, f"{s}.npz")
    if os.path.exists(path):
        skipped += 1
        continue
    try:
        X, y, _ = paradigm.get_data(dataset, subjects=[s], return_concat=True, n_jobs=1)
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        # Build 3-epoch windows
        n = X.shape[0]
        if n < 3:
            print(f"  SKIP {s}: too few epochs ({n})")
            continue
        Xw, yw = [], []
        for i in range(1, n - 1):
            Xw.append(X[i-1:i+2, 0, :])  # (3, 3000)
            yw.append(y[i])
        Xw = np.stack(Xw)
        yw = np.array(yw, dtype=np.int64)
        np.savez_compressed(path, X=Xw, y=yw)
        done += 1
        if done % 10 == 0:
            print(f"  {done} subjects cached...")
    except (ValueError, RuntimeError) as e:
        print(f"  SKIP {s}: {e}")

print(f"\nDone: {done} cached, {skipped} skipped")
total = done + skipped
print(f"Cache: {CACHE_DIR}/ ({total} .npz files)")
