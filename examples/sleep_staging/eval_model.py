"""Evaluate a saved ParaSleep checkpoint on test data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import metabci.brainda.algorithms.deep_learning.parasleep as lwmod

MODEL_PATH = "parasleep_best.pth"
CACHE_DIR = r"F:\sleep_cache"  # change to D:\... if on your laptop

print(f"Loading {MODEL_PATH}...")
raw_cls = lwmod.ParaSleep.module
model = raw_cls(n_channels=3, n_samples=3000, n_classes=5).float()
state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Load test subjects
if os.path.isdir(CACHE_DIR):
    subs = sorted([f.replace('.npz', '') for f in os.listdir(CACHE_DIR) if f.endswith('.npz')])
else:
    subs = sorted([f.replace('.npz', '') for f in os.listdir('data_cache') if f.endswith('.npz')])
    CACHE_DIR = 'data_cache'

rng = np.random.RandomState(42)
rng.shuffle(subs)
n = len(subs)
test_subs = subs[max(0, n-10):]  # last 10 available subjects
X_list, y_list = [], []
for s in test_subs:
    d = np.load(os.path.join(CACHE_DIR, f'{s}.npz'))
    X_list.append(d['X']); y_list.append(d['y'])
X_test = np.concatenate(X_list)
y_test = np.concatenate(y_list)
print(f"Test: {len(test_subs)} subjects, {len(y_test)} epochs")

loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
    batch_size=128,
)
all_p, all_t = [], []
with torch.no_grad():
    for Xb, yb in loader:
        all_p.append(model(Xb).argmax(1).numpy())
        all_t.append(yb.numpy())
yp = np.concatenate(all_p)
yt = np.concatenate(all_t)

print(classification_report(yt, yp, target_names=['W','N1','N2','N3','REM'],
      digits=4, zero_division=0))
