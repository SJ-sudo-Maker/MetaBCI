# -*- coding: utf-8 -*-
"""
ParaSleep server training — full-scale with K-fold CV support.

Usage:
    python train_server.py                          # full training
    python train_server.py --cv 5                   # 5-fold CV
    python train_server.py --subjects 100 --epochs 150
"""

import sys, os, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import metabci.brainda.algorithms.deep_learning.parasleep as lwmod


# =============================================================================
# Config (override via CLI args or edit below)
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    default=os.environ.get('SLEEP_DATA',
                    '/root/autodl-tmp/sleep-edf/sleep-cassette'),
                    help='Path to sleep-cassette directory')
parser.add_argument('--cache', type=str, default='data_cache',
                    help='Cache directory')
parser.add_argument('--subjects', type=int, default=80, help='Train subjects')
parser.add_argument('--test', type=int, default=10, help='Test subjects')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--context', type=int, default=3, help='Context window (odd)')
parser.add_argument('--cv', type=int, default=0, help='K-fold CV (0=single run)')
parser.add_argument('--save', type=str, default='parasleep_best.pth')
parser.add_argument('--device', type=str, default='cuda',
                    help='cuda / cpu')
args = parser.parse_args()

assert args.context % 2 == 1, 'Context window must be odd'
DEVICE = args.device if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE} | Subjects: {args.subjects} | Epochs: {args.epochs}")
print(f"Data: {args.data} | Cache: {args.cache}")

# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma; self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, i, t):
        ce = F.cross_entropy(i, t, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


# =============================================================================
# Data loading
# =============================================================================

def load_cache_subjects(cache_dir):
    """List available cached subjects."""
    if not os.path.isdir(cache_dir):
        return []
    return sorted([f.replace('.npz','') for f in os.listdir(cache_dir)
                   if f.endswith('.npz')])

def build_cache(data_root, cache_dir):
    """Build cache from raw EDF files if not present."""
    if os.path.isdir(cache_dir) and len(load_cache_subjects(cache_dir)) > 0:
        print(f"  Cache exists: {len(load_cache_subjects(cache_dir))} subjects")
        return

    print("  Building cache from raw EDF...")
    from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
    from metabci.brainda.paradigms.sleep import SleepParadigm

    os.makedirs(cache_dir, exist_ok=True)
    dataset = SleepEDFDataset(data_root, channel='EEG Fpz-Cz')
    paradigm = SleepParadigm(channels=['EEG Fpz-Cz'], srate=100)

    for s in dataset.subjects:
        path = os.path.join(cache_dir, f'{s}.npz')
        if os.path.exists(path):
            continue
        try:
            X, y, _ = paradigm.get_data(dataset, subjects=[s], return_concat=True, n_jobs=1)
            X, y = X.astype(np.float32), y.astype(np.int64)
            n = X.shape[0]; half = args.context // 2
            Xw, yw = [], []
            for i in range(half, n - half):
                Xw.append(X[i-half:i+half+1, 0, :])
                yw.append(y[i])
            if Xw:
                np.savez_compressed(path, X=np.stack(Xw), y=np.array(yw, dtype=np.int64))
        except (ValueError, RuntimeError):
            pass
    print(f"  Cache built: {len(load_cache_subjects(cache_dir))} subjects")


def load_windows(subjects, cache_dir, half=1):
    """Load 3-epoch windows from cache."""
    X_list, y_list = [], []
    for s in subjects:
        p = os.path.join(cache_dir, f'{s}.npz')
        if os.path.exists(p):
            d = np.load(p)
            X_list.append(d['X']); y_list.append(d['y'])
    return np.concatenate(X_list), np.concatenate(y_list)


# =============================================================================
# Train single fold
# =============================================================================

def train_fold(X_train, y_train, X_test, y_test, fold_name=''):
    """Train one fold, return metrics dict."""
    global_start = time.time()

    # Split train into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    te_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=args.batch, shuffle=False)

    print(f"\n  {'='*50}")
    print(f"  Fold {fold_name}: Train={X_tr.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    print(f"  {'='*50}")

    # Class weights
    _, counts = np.unique(y_tr, return_counts=True)
    raw = np.sqrt([len(y_tr) / c for c in counts])
    raw = np.clip(raw / raw.min(), 1.0, 10.0)
    cw = torch.tensor(raw, dtype=torch.float32).to(DEVICE)

    # Model
    raw_cls = lwmod.ParaSleep.module
    model = raw_cls(n_channels=args.context, n_samples=3000, n_classes=5).float().to(DEVICE)
    focal = FocalLoss(gamma=2.0, weight=cw, label_smoothing=0.05)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0, betas=(0.9, 0.999))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = 0; best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb)
            loss = focal(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        # Validate
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                vc += (model(Xb).argmax(1) == yb).sum().item()
                vt += yb.size(0)
        val_acc = vc / vt

        if val_acc > best_val:
            best_val = val_acc; best_epoch = epoch
            torch.save(model.state_dict(), args.save)

        if epoch == 1 or epoch % 20 == 0:
            elapsed = time.time() - global_start
            print(f"  Epoch {epoch:3d}/{args.epochs} | val={val_acc:.3f} | "
                  f"best={best_val:.3f}@{best_epoch} | {elapsed/60:.0f}min")

    # Test
    state = torch.load(args.save, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state); model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for Xb, yb in te_loader:
            Xb = Xb.to(DEVICE)
            all_p.append(model(Xb).argmax(1).cpu().numpy())
            all_t.append(yb.numpy())
    yp = np.concatenate(all_p); yt = np.concatenate(all_t)

    acc = np.mean(yp == yt)
    report = classification_report(yt, yp, target_names=['W','N1','N2','N3','REM'],
                                    digits=4, output_dict=True, zero_division=0)
    return {
        'accuracy': acc,
        'macro_f1': report['macro avg']['f1-score'],
        'W_f1': report['W']['f1-score'],
        'N1_f1': report['N1']['f1-score'],
        'N2_f1': report['N2']['f1-score'],
        'N3_f1': report['N3']['f1-score'],
        'REM_f1': report['REM']['f1-score'],
        'y_pred': yp, 'y_true': yt,
    }


# =============================================================================
# Main
# =============================================================================

print('=' * 60)
print('ParaSleep Server Training')
print('=' * 60)

# Build or verify cache
build_cache(args.data, args.cache)
all_subs = load_cache_subjects(args.cache)
print(f'Cached subjects: {len(all_subs)}')

# Get subject split
rng = np.random.RandomState(42)
rng.shuffle(all_subs)

total_needed = args.subjects + args.test
if len(all_subs) < total_needed:
    args.subjects = len(all_subs) - args.test
    print(f'WARNING: Only {len(all_subs)} subjects, using {args.subjects}+{args.test}')

train_subs = all_subs[:args.subjects]
test_subs = all_subs[args.subjects:args.subjects + args.test]

# Load data once
print(f'Loading {len(train_subs)}+{len(test_subs)} subjects...')
X_all, y_all = load_windows(train_subs + test_subs, args.cache)
subj_ids = []
for s in train_subs:
    d = np.load(os.path.join(args.cache, f'{s}.npz')); subj_ids.extend([s] * len(d['y']))
for s in test_subs:
    d = np.load(os.path.join(args.cache, f'{s}.npz')); subj_ids.extend([s] * len(d['y']))
subj_ids = np.array(subj_ids)

if args.cv > 1:
    # K-fold CV by subject
    print(f'\nRunning {args.cv}-fold subject-wise CV...')
    unique_subs = np.unique(subj_ids)
    rng.shuffle(unique_subs)
    fold_size = len(unique_subs) // args.cv

    all_folds = []
    for fold in range(args.cv):
        t_start = fold * fold_size
        t_end = (fold + 1) * fold_size if fold < args.cv - 1 else len(unique_subs)
        test_set = set(unique_subs[t_start:t_end])
        train_set = set(unique_subs) - test_set

        tr_mask = np.array([s in train_set for s in subj_ids])
        te_mask = np.array([s in test_set for s in subj_ids])

        result = train_fold(
            X_all[tr_mask], y_all[tr_mask],
            X_all[te_mask], y_all[te_mask],
            fold_name=str(fold + 1),
        )
        all_folds.append(result)
        print(f"  Fold {fold+1} Macro F1: {result['macro_f1']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    keys = ['accuracy', 'macro_f1', 'W_f1', 'N1_f1', 'N2_f1', 'N3_f1', 'REM_f1']
    labels = ['Accuracy', 'Macro F1', 'W F1', 'N1 F1', 'N2 F1', 'N3 F1', 'REM F1']
    for key, label in zip(keys, labels):
        vals = [f[key] for f in all_folds]
        print(f"  {label:12s}: {np.mean(vals)*100:.2f}% +- {np.std(vals)*100:.2f}%")

    print(f"\n  Final: Macro F1 = {np.mean([f['macro_f1'] for f in all_folds])*100:.2f}% "
          f"+- {np.std([f['macro_f1'] for f in all_folds])*100:.2f}%")

else:
    # Single run
    print(f'\nTraining: {args.subjects} subjects, {args.epochs} epochs')
    tr_mask = np.array([s in set(train_subs) for s in subj_ids])
    te_mask = np.array([s in set(test_subs) for s in subj_ids])

    result = train_fold(X_all[tr_mask], y_all[tr_mask],
                        X_all[te_mask], y_all[te_mask])

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {result['accuracy']*100:.2f}%")
    print(f"Macro F1:      {result['macro_f1']:.4f}")
    print(f"W  F1: {result['W_f1']:.4f}  N1 F1: {result['N1_f1']:.4f}  "
          f"N2 F1: {result['N2_f1']:.4f}  N3 F1: {result['N3_f1']:.4f}  "
          f"REM F1: {result['REM_f1']:.4f}")
    print('')
    print(classification_report(result['y_true'], result['y_pred'],
          target_names=['W','N1','N2','N3','REM'], digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(result['y_true'], result['y_pred']))

print(f"\nModel saved: {args.save}")
total_time = time.time() - (time.time() if False else 0)  # placeholder
print(f"Done.")
