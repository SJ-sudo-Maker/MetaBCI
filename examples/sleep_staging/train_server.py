# -*- coding: utf-8 -*-
"""
ParaSleep server training — full-scale with subject-wise validation.

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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import metabci.brainda.algorithms.deep_learning.parasleep as lwmod


# =============================================================================
# Config (override via CLI args or edit below)
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    default=os.environ.get('SLEEP_DATA',
                    r'F:\sleep-edf\sleep-edf-database-expanded-1.0.0\sleep-cassette'),
                    help='Path to sleep-cassette directory')
parser.add_argument('--cache', type=str, default='data_cache',
                    help='Cache directory')
parser.add_argument('--subjects', type=int, default=80, help='Train subjects')
parser.add_argument('--test', type=int, default=10, help='Test subjects')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--context', type=int, default=3, help='Context window (odd)')
parser.add_argument('--cv', type=int, default=5, help='K-fold CV (0=single run)')
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
    if not os.path.isdir(cache_dir):
        return []
    return sorted([f.replace('.npz','') for f in os.listdir(cache_dir)
                   if f.endswith('.npz')])

def build_cache(data_root, cache_dir):
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


def load_windows(subjects, cache_dir):
    X_list, y_list, subj_list = [], [], []
    for s in subjects:
        p = os.path.join(cache_dir, f'{s}.npz')
        if os.path.exists(p):
            d = np.load(p)
            X_list.append(d['X']); y_list.append(d['y'])
            subj_list.extend([s] * len(d['y']))
    return np.concatenate(X_list), np.concatenate(y_list), np.array(subj_list)


# =============================================================================
# Train single fold
# =============================================================================

def train_fold(X_train, y_train, subj_train, X_test, y_test, fold_name=''):
    global_start = time.time()

    # === Subject-wise validation split ===
    # Same subject's epochs go entirely to train OR val — no leakage
    unique_subs = np.unique(subj_train)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_subs)
    n_val_subs = max(1, int(len(unique_subs) * 0.2))
    val_subs = set(unique_subs[:n_val_subs])
    tr_subs = set(unique_subs[n_val_subs:])

    tr_mask = np.array([s in tr_subs for s in subj_train])
    val_mask = np.array([s in val_subs for s in subj_train])

    X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
    X_val, y_val = X_train[val_mask], y_train[val_mask]

    print(f"\n  {'='*50}")
    print(f"  Fold {fold_name}")
    print(f"  Train: {X_tr.shape[0]} epochs ({len(tr_subs)} subjects)")
    print(f"  Val:   {X_val.shape[0]} epochs ({len(val_subs)} subjects)")
    print(f"  Test:  {X_test.shape[0]} epochs")
    print(f"  {'='*50}")

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    te_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=args.batch, shuffle=False)

    # Class weights (computed from training set only)
    _, counts = np.unique(y_tr, return_counts=True)
    raw = np.sqrt([len(y_tr) / c for c in counts])
    raw = np.clip(raw / raw.min(), 1.0, 10.0)
    cw = torch.tensor(raw, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: {raw.tolist()}")

    # Model
    raw_cls = lwmod.ParaSleep.module
    model = raw_cls(n_channels=args.context, n_samples=3000, n_classes=5).float().to(DEVICE)
    focal = FocalLoss(gamma=2.0, weight=cw, label_smoothing=0.05)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0, betas=(0.9, 0.999))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = 0; best_epoch = 0; train_losses = []; val_accs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb)
            loss = focal(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * Xb.size(0)
        sched.step()

        model.eval()
        all_vpred, all_vtrue = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                all_vpred.append(model(Xb).argmax(1).cpu().numpy())
                all_vtrue.append(yb.cpu().numpy())
        val_acc = (np.concatenate(all_vpred) == np.concatenate(all_vtrue)).mean()
        # Macro F1 — equal weight to all 5 classes, not dominated by W
        val_f1 = f1_score(np.concatenate(all_vtrue), np.concatenate(all_vpred),
                          average='macro', zero_division=0)

        train_losses.append(tr_loss / len(X_tr))
        val_accs.append(val_acc)

        if val_f1 > best_val:
            best_val = val_f1; best_epoch = epoch
            torch.save(model.state_dict(), args.save)

        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.time() - global_start
            marker = " *" if val_f1 == best_val else ""
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss={tr_loss/len(X_tr):.4f} | "
                  f"val_f1={val_f1:.3f} | best_f1={best_val:.3f}@{best_epoch}{marker} | "
                  f"{elapsed/60:.0f}min")

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
        'accuracy': acc, 'macro_f1': report['macro avg']['f1-score'],
        'W_f1': report['W']['f1-score'], 'N1_f1': report['N1']['f1-score'],
        'N2_f1': report['N2']['f1-score'], 'N3_f1': report['N3']['f1-score'],
        'REM_f1': report['REM']['f1-score'],
        'y_pred': yp, 'y_true': yt, 'val_accs': val_accs, 'train_losses': train_losses,
    }


# =============================================================================
# Main
# =============================================================================

print('=' * 60)
print('ParaSleep Server Training')
print('=' * 60)

build_cache(args.data, args.cache)
all_subs = load_cache_subjects(args.cache)
print(f'Cached subjects: {len(all_subs)}')

rng = np.random.RandomState(42)
rng.shuffle(all_subs)

total_needed = args.subjects + args.test
if len(all_subs) < total_needed:
    args.subjects = len(all_subs) - args.test
    print(f'WARNING: Only {len(all_subs)} subjects, using {args.subjects}+{args.test}')

train_subs = all_subs[:args.subjects]
test_subs = all_subs[args.subjects:args.subjects + args.test]

print(f'Loading {len(train_subs)}+{len(test_subs)} subjects...')
X_all, y_all, subj_all = load_windows(train_subs + test_subs, args.cache)

if args.cv > 1:
    print(f'\nRunning {args.cv}-fold subject-wise CV...')
    unique_subs = np.unique(subj_all)
    rng.shuffle(unique_subs)
    fold_size = len(unique_subs) // args.cv

    all_folds = []
    for fold in range(args.cv):
        t_start = fold * fold_size
        t_end = (fold + 1) * fold_size if fold < args.cv - 1 else len(unique_subs)
        test_set = set(unique_subs[t_start:t_end])
        train_set = set(unique_subs) - test_set
        tr_mask = np.array([s in train_set for s in subj_all])
        te_mask = np.array([s in test_set for s in subj_all])

        result = train_fold(X_all[tr_mask], y_all[tr_mask], subj_all[tr_mask],
                            X_all[te_mask], y_all[te_mask], fold_name=str(fold + 1))
        all_folds.append(result)
        print(f"  Fold {fold+1} Macro F1: {result['macro_f1']:.4f}")

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
    print(f'\nTraining: {args.subjects} subjects, {args.epochs} epochs')
    train_set = set(train_subs)
    test_set = set(test_subs)
    tr_mask = np.array([s in train_set for s in subj_all])
    te_mask = np.array([s in test_set for s in subj_all])

    result = train_fold(X_all[tr_mask], y_all[tr_mask], subj_all[tr_mask],
                        X_all[te_mask], y_all[te_mask])

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {result['accuracy']*100:.2f}%")
    print(f"Macro F1:      {result['macro_f1']:.4f}")
    print(f"W  F1: {result['W_f1']:.4f}  N1 F1: {result['N1_f1']:.4f}")
    print(f"N2 F1: {result['N2_f1']:.4f}  N3 F1: {result['N3_f1']:.4f}")
    print(f"REM F1: {result['REM_f1']:.4f}")
    print('')
    print(classification_report(result['y_true'], result['y_pred'],
          target_names=['W','N1','N2','N3','REM'], digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(result['y_true'], result['y_pred']))

print(f"\nModel saved: {args.save}")
print(f"Done.")
