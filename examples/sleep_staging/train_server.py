# -*- coding: utf-8 -*-
"""
ParaSleep server training — unified experiment runner.

Supports all four improvement angles:
  1. Causal context input (--causal, --context)
  2. Lightweight temporal attention (--model ta)
  3. N1-focused class imbalance (--sampler weighted)
  4. Multi-task auxiliary training (--aux multitask)

Usage:
    # Angle 1: context comparison
    python train_server.py --context 1
    python train_server.py --context 3
    python train_server.py --context 3 --causal
    python train_server.py --context 5
    python train_server.py --context 5 --causal

    # Angle 3: sampler
    python train_server.py --sampler weighted
    python train_server.py --sampler weighted --label_smoothing 0

    # Angle 2+4: final config
    python train_server.py --model ta --context 5 --causal --sampler weighted --aux multitask

    # Final 5-fold CV
    python train_server.py --model ta --context 5 --causal --sampler weighted --aux multitask --cv 5
"""

import sys, os, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import metabci.brainda.algorithms.deep_learning.parasleep as lwmod


# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    default=os.environ.get('SLEEP_DATA',
                    r'F:\sleep-edf\sleep-edf-database-expanded-1.0.0\sleep-cassette'),
                    help='Path to sleep-cassette directory')
parser.add_argument('--cache', type=str, default=r'F:\sleep_cache',
                    help='Cache directory')
parser.add_argument('--subjects', type=int, default=80, help='Train subjects')
parser.add_argument('--test', type=int, default=10, help='Test subjects')
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=1.0, help='Weight decay')
parser.add_argument('--context', type=int, default=3, help='Context window (odd)')
parser.add_argument('--causal', action='store_true',
                    help='Causal (left-only) context window')
parser.add_argument('--cv', type=int, default=0, help='K-fold CV (0=single run)')
parser.add_argument('--save', type=str, default='parasleep_best.pth')
parser.add_argument('--device', type=str, default='cuda')

# Angle 2: model variant
parser.add_argument('--model', type=str, default='parasleep',
                    choices=['parasleep', 'ta'],
                    help='Model variant: parasleep (baseline) or ta (temporal attention)')

# Angle 3: sampler & label smoothing
parser.add_argument('--sampler', type=str, default='none',
                    choices=['none', 'weighted'],
                    help='Sampling strategy')
parser.add_argument('--label_smoothing', type=float, default=0.05,
                    help='Label smoothing (0=off)')

# Angle 4: multi-task auxiliary
parser.add_argument('--aux', type=str, default='none',
                    choices=['none', 'multitask'],
                    help='Auxiliary task mode')
parser.add_argument('--lambda4', type=float, default=0.3,
                    help='Weight for 4-class auxiliary loss')
parser.add_argument('--lambda3', type=float, default=0.2,
                    help='Weight for 3-class auxiliary loss')

args = parser.parse_args()

assert args.context % 2 == 1, 'Context window must be odd'
DEVICE = args.device if torch.cuda.is_available() else 'cpu'
MODE = 'causal' if args.causal else 'center'

print(f"Device: {DEVICE} | Subjects: {args.subjects} | Epochs: {args.epochs}")
print(f"Config: context={args.context} {MODE} | model={args.model} | "
      f"sampler={args.sampler} | aux={args.aux}")
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
# Label mapping → delegated to SleepParadigm
# =============================================================================

from metabci.brainda.paradigms.sleep import SleepParadigm

# Helper: create a lightweight paradigm instance for label mapping / cache building
def _get_paradigm():
    return SleepParadigm(
        channels=['EEG Fpz-Cz'], srate=100,
        context=args.context,
        context_mode='causal' if args.causal else 'center',
        label_mode='5class',
    )


# =============================================================================
# Cache helpers
# =============================================================================

def cache_filename(sub, context, causal):
    mode = 'causal' if causal else 'center'
    return f'{sub}_ctx{context}_{mode}.npz'

def cache_name_pattern(context, causal):
    mode = 'causal' if causal else 'center'
    return f'_ctx{context}_{mode}.npz'

def find_cache(sub, cache_dir, context, causal):
    """Try new naming first, fall back to old naming."""
    path = os.path.join(cache_dir, cache_filename(sub, context, causal))
    if os.path.exists(path):
        return path
    # Fallback: old naming {sub}.npz (only valid for context=3 center)
    if context == 3 and not causal:
        path = os.path.join(cache_dir, f'{sub}.npz')
        if os.path.exists(path):
            return path
    return None

def load_cache_subjects(cache_dir, context, causal):
    if not os.path.isdir(cache_dir):
        return []
    pattern = cache_name_pattern(context, causal)
    subs = []
    for f in os.listdir(cache_dir):
        if f.endswith(pattern):
            subs.append(f.replace(pattern, ''))
    return sorted(subs)

def build_cache(data_root, cache_dir, context, causal):
    existing = load_cache_subjects(cache_dir, context, causal)
    if len(existing) > 0:
        print(f"  Cache exists: {len(existing)} subjects (ctx={context} {MODE})")
        return

    print(f"  Building cache (ctx={context} {MODE}) from raw EDF...")
    from metabci.brainda.datasets.sleep_edf import SleepEDFDataset

    os.makedirs(cache_dir, exist_ok=True)
    dataset = SleepEDFDataset(data_root, channel='EEG Fpz-Cz')
    paradigm = _get_paradigm()

    for s in dataset.subjects:
        path = os.path.join(cache_dir, cache_filename(s, context, causal))
        if os.path.exists(path):
            continue
        try:
            X, y, _ = paradigm.get_data(dataset, subjects=[s],
                                         return_concat=True, n_jobs=1)
            X, y = X.astype(np.float32), y.astype(np.int64)
            Xw, yw = paradigm.build_windows(X, y)
            np.savez_compressed(path, X=Xw, y=yw)
        except (ValueError, RuntimeError):
            pass
    print(f"  Cache built: {len(load_cache_subjects(cache_dir, context, causal))} subjects")


def load_windows(subjects, cache_dir, context, causal):
    X_list, y_list, subj_list = [], [], []
    for s in subjects:
        p = find_cache(s, cache_dir, context, causal)
        if p is not None:
            d = np.load(p)
            X_list.append(d['X']); y_list.append(d['y'])
            subj_list.extend([s] * len(d['y']))
    return np.concatenate(X_list), np.concatenate(y_list), np.array(subj_list)


# =============================================================================
# Model factory
# =============================================================================

def create_model():
    """Create model based on --model and --aux flags."""
    use_ta = (args.model == 'ta')
    use_aux = (args.aux == 'multitask')
    raw_cls = lwmod.ParaSleep.module
    model = raw_cls(
        n_channels=args.context, n_samples=3000, n_classes=5,
        use_temporal_attention=use_ta,
        use_aux=use_aux,
    ).float()
    return model.to(DEVICE)


# =============================================================================
# Train single fold
# =============================================================================

def train_fold(X_train, y_train, subj_train, X_test, y_test, fold_name=''):
    global_start = time.time()

    # === Subject-wise validation split ===
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

    use_aux = (args.aux == 'multitask')

    print(f"\n  {'='*50}")
    print(f"  Fold {fold_name}")
    print(f"  Train: {X_tr.shape[0]} epochs ({len(tr_subs)} subjects)")
    print(f"  Val:   {X_val.shape[0]} epochs ({len(val_subs)} subjects)")
    print(f"  Test:  {X_test.shape[0]} epochs")
    print(f"  Config: ctx={args.context} {MODE} | model={args.model} | "
          f"sampler={args.sampler} | aux={args.aux}")
    print(f"  {'='*50}")

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    te_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # Sampling strategy
    if args.sampler == 'weighted':
        class_count = np.bincount(y_tr, minlength=5)
        sample_weights = 1.0 / (class_count[y_tr] + 1e-8)
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        tr_loader = DataLoader(tr_ds, batch_size=args.batch, sampler=sampler)
        print(f"  Sampler: WeightedRandomSampler (class counts: {class_count.tolist()})")
    else:
        tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=args.batch, shuffle=False)

    # Class weights (from training set)
    _, counts = np.unique(y_tr, return_counts=True)
    raw = np.sqrt([len(y_tr) / c for c in counts])
    raw = np.clip(raw / raw.min(), 1.0, 10.0)
    cw = torch.tensor(raw, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: {raw.tolist()}")
    print(f"  Label smoothing: {args.label_smoothing}")

    # Model
    model = create_model()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    focal5 = FocalLoss(gamma=2.0, weight=cw, label_smoothing=args.label_smoothing)
    if use_aux:
        focal4 = FocalLoss(gamma=2.0, weight=None, label_smoothing=0.0)
        focal3 = FocalLoss(gamma=2.0, weight=None, label_smoothing=0.0)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                      betas=(0.9, 0.999))

    # Three-stage LR: 1e-3 (1-10) → 1e-4 (11-130) → 1e-5 (131-150)
    def adjust_lr(epoch):
        if epoch <= 10:   return 1.0    # base_lr × 1.0  = 1e-3
        elif epoch <= 130: return 0.1   # base_lr × 0.1  = 1e-4
        else:              return 0.01  # base_lr × 0.01 = 1e-5
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=adjust_lr)

    # EMA
    ema_avg = lambda avg, new: 0.999 * avg + 0.001 * new if avg is not None else new
    ema_state = None

    train_losses = []; val_accs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

            if use_aux:
                out5, out4, out3 = model(Xb)
                yb_np = yb.cpu().numpy()
                y4 = torch.from_numpy(
                    SleepParadigm.map_labels(yb_np, '4class')).to(DEVICE)
                y3 = torch.from_numpy(
                    SleepParadigm.map_labels(yb_np, '3class')).to(DEVICE)
                loss = focal5(out5, yb) + args.lambda4 * focal4(out4, y4) \
                       + args.lambda3 * focal3(out3, y3)
            else:
                out = model(Xb)
                loss = focal5(out, yb)

            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * Xb.size(0)
        sched.step()

        # Update EMA
        if ema_state is None:
            ema_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            for k in ema_state:
                ema_state[k] = ema_avg(ema_state[k], model.state_dict()[k].float())

        # Validation (use 5-class output only)
        model.eval()
        all_vpred, all_vtrue = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                if use_aux:
                    out, _, _ = model(Xb)
                else:
                    out = model(Xb)
                all_vpred.append(out.argmax(1).cpu().numpy())
                all_vtrue.append(yb.cpu().numpy())
        val_acc = (np.concatenate(all_vpred) == np.concatenate(all_vtrue)).mean()
        val_f1 = f1_score(np.concatenate(all_vtrue), np.concatenate(all_vpred),
                          average='macro', zero_division=0)

        train_losses.append(tr_loss / len(X_tr))
        val_accs.append(val_acc)

        torch.save(ema_state, args.save)

        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.time() - global_start
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss={tr_loss/len(X_tr):.4f} | "
                  f"val_f1={val_f1:.3f} | lr={opt.param_groups[0]['lr']:.1e} | "
                  f"{elapsed/60:.0f}min")

    # Test with EMA weights
    state = torch.load(args.save, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state); model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for Xb, yb in te_loader:
            Xb = Xb.to(DEVICE)
            if use_aux:
                out, _, _ = model(Xb)
            else:
                out = model(Xb)
            all_p.append(out.argmax(1).cpu().numpy())
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
print(f'ParaSleep Training — ctx={args.context} {MODE} | model={args.model}')
print('=' * 60)

build_cache(args.data, args.cache, args.context, args.causal)
all_subs = load_cache_subjects(args.cache, args.context, args.causal)
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
X_all, y_all, subj_all = load_windows(train_subs + test_subs, args.cache,
                                       args.context, args.causal)

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
