"""Evaluate a saved ParaSleep checkpoint on test data."""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import metabci.brainda.algorithms.deep_learning.parasleep as lwmod

MODEL_PATH = "parasleep_best.pth"
CACHE_DIR = r"F:\sleep_cache"

print(f"Loading {MODEL_PATH}...")
state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)

# Auto-detect architecture and context from checkpoint
use_ta = any(k.startswith('transformer.') for k in state.keys())
# Detect context from pos_embed shape (TA) or mrfe first conv weight (base)
if use_ta and 'pos_embed' in state:
    n_ch = state['pos_embed'].shape[1]
elif 'mrfe.small_branch.dsconv.dw.weight' in state:
    n_ch = state['mrfe.small_branch.dsconv.dw.weight'].shape[0]
else:
    n_ch = 3  # default

raw_cls = lwmod.ParaSleep.module
model = raw_cls(n_channels=n_ch, n_samples=3000, n_classes=5,
                use_temporal_attention=use_ta).float()
model.load_state_dict(state, strict=False)
model.eval()
arch = 'TA' if use_ta else 'Base'
print(f"Architecture: {arch} | Context: {n_ch} | Params: {sum(p.numel() for p in model.parameters()):,}")

# Load test subjects with fallback naming
if not os.path.isdir(CACHE_DIR):
    CACHE_DIR = 'data_cache'

patterns = ['*_FpzCz_sr100_ctx*_center_5class.npz', '*_ctx*_center.npz', '*.npz']
test_files = []
for pat in patterns:
    test_files = sorted(glob.glob(os.path.join(CACHE_DIR, pat)))
    if test_files:
        break

rng = np.random.RandomState(42)
rng.shuffle(test_files)
test_files = test_files[-10:]  # last 10 subjects
X_list, y_list = [], []
for f in test_files:
    d = np.load(f)
    X_list.append(d['X']); y_list.append(d['y'])
X_test = np.concatenate(X_list)
y_test = np.concatenate(y_list)
print(f"Test: {len(test_files)} subjects, {len(y_test)} epochs")

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
print("Confusion Matrix:")
print(confusion_matrix(yt, yp))
print(f"Macro F1: {f1_score(yt, yp, average='macro', zero_division=0):.4f}")
