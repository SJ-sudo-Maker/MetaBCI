# -*- coding: utf-8 -*-
"""
ParaSleep server training — unified experiment runner.

Supports: causal/center context, temporal attention, weighted sampler,
          multi-task auxiliary training, and subject-wise K-fold CV.

Quick start (final model):
    python train_server.py --context 3 --causal --epochs 60 --wd 1e-2 --label_smoothing 0 --save exp_ctx3_causal.pth --cache F:/sleep_cache

Holdout evaluation:
    python demo_metric.py --model exp_ctx3_causal.pth --split exp_ctx3_causal_split.npz --cache F:/sleep_cache --context 3 --causal --out demo_outputs_ctx3_causal

5-fold CV:
    python train_server.py --context 3 --causal --epochs 60 --wd 1e-2 --label_smoothing 0 --sampler none --aux none --cv 5 --subjects 124 --test 10 --save exp_ctx3_causal_cv.pth --cache F:/sleep_cache

Ablation experiments:
    python train_server.py --context 1 --save exp_ctx1.pth --cache F:/sleep_cache
    python train_server.py --context 3 --save exp_ctx3_center.pth --cache F:/sleep_cache
    python train_server.py --context 3 --sampler weighted --save exp_weighted.pth --cache F:/sleep_cache
    python train_server.py --model ta --context 3 --save exp_ta.pth --cache F:/sleep_cache
"""

import sys, os, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import metabci.brainda.algorithms.deep_learning.parasleep as lwmod
from metabci.brainda.pipelines.sleep_cache import (
    CACHE_SCHEMA_VERSION, cache_file_path, cache_config_dir,
    save_cache_record, discover_records, load_records_batch,
    save_manifest, PREPROCESS_DESCRIPTION,
)


def set_global_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
parser.add_argument('--subjects', type=int, default=68, help='Train subjects')
parser.add_argument('--test', type=int, default=10, help='Test subjects')
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=1e-2, help='Weight decay')
parser.add_argument('--context', type=int, default=3, help='Context window (odd)')
parser.add_argument('--causal', action='store_true',
                    help='Causal (left-only) context window')
parser.add_argument('--cv', type=int, default=0, help='K-fold CV (0=single run)')
parser.add_argument('--save', type=str, default='parasleep_best.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=42, help='Global random seed')
parser.add_argument('--cv-seed', type=int, default=42, help='CV fold split seed')
parser.add_argument('--strict-dataset', action='store_true',
                    help='Require exact 153 records, 78 subjects, 68+10 split')

# Angle 2: model variant
parser.add_argument('--model', type=str, default='parasleep',
                    choices=['parasleep', 'ta'],
                    help='Model variant: parasleep (baseline) or ta (temporal attention)')

# Angle 3: sampler & label smoothing
parser.add_argument('--sampler', type=str, default='none',
                    choices=['none', 'weighted'],
                    help='Sampling strategy')
parser.add_argument('--label_smoothing', type=float, default=0.0,
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
# Cache version — bump after any change to data extraction, preprocessing, or
# subject-ID parsing. Old caches MUST NOT be reused across versions.
CACHE_VERSION = "chronov3"

# Cache naming: {sub}_FpzCz_sr100_ctx{context}_{mode}_{label}_{version}.npz

# Map Sleep Cassette record IDs (e.g. "4001", "4002") to real subject IDs (e.g. "00").
# SC4ssNE0: ss=subject (00-82), N=night (1-2). Same subject's nights must stay together.
def _real_subject_id(record_id):
    """Extract real subject ID from Sleep Cassette record ID."""
    s = str(record_id)
    return s[1:3] if len(s) >= 3 else s

def cache_filename(sub, context, causal, label_mode='5class'):
    mode = 'causal' if causal else 'center'
    return f'{sub}_FpzCz_sr100_ctx{context}_{mode}_{label_mode}_{CACHE_VERSION}.npz'

def cache_name_suffix(context, causal, label_mode='5class'):
    mode = 'causal' if causal else 'center'
    return f'_FpzCz_sr100_ctx{context}_{mode}_{label_mode}_{CACHE_VERSION}.npz'

def find_cache(sub, cache_dir, context, causal, label_mode='5class'):
    """Find cache file. Only accepts chronov3 format — NO fallback to old caches."""
    path = os.path.join(cache_dir, cache_filename(sub, context, causal, label_mode))
    return path if os.path.exists(path) else None

def build_subject_records_map(record_ids):
    """Build real-subject-ID → list-of-record-IDs mapping."""
    mapping = {}
    for rid in record_ids:
        sid = _real_subject_id(rid)
        mapping.setdefault(sid, []).append(rid)
    return mapping

def load_cache_subjects(cache_dir, context, causal, label_mode='5class'):
    if not os.path.isdir(cache_dir):
        return []
    suffix = cache_name_suffix(context, causal, label_mode)
    subs = []
    for f in os.listdir(cache_dir):
        if f.endswith(suffix):
            subs.append(f.replace(suffix, ''))
    return sorted(subs)

def build_cache(data_root, cache_dir, context, causal):
    print(f"  Building cache (ctx={context} {MODE}, {CACHE_VERSION}) from raw EDF...")
    from metabci.brainda.datasets.sleep_edf import SleepEDFDataset

    os.makedirs(cache_dir, exist_ok=True)
    dataset = SleepEDFDataset(data_root, channel='EEG Fpz-Cz')
    paradigm = _get_paradigm()

    done, skipped, failed = 0, 0, 0
    for subj_id in dataset.subjects:
        path = os.path.join(cache_dir, cache_filename(subj_id, context, causal))
        if os.path.exists(path):
            skipped += 1
            continue
        try:
            # Load raw data directly (bypass BaseParadigm event-class grouping)
            raw = dataset._get_single_subject_data(subj_id)
            raw_data = raw['session_0']['run_0']
            sfreq = raw_data.info['sfreq']

            # Apply 0.5-40 Hz bandpass filter (unified preprocessing)
            raw_data.filter(0.5, 40, picks='eeg', verbose=False)

            # Extract epochs in strict chronological order
            X, y, onsets = paradigm.extract_epochs(
                raw_data, raw_data.annotations, sfreq, dataset.epoch_sec)

            # Build context windows
            Xw, yw = paradigm.build_windows(X, y)

            # Save with metadata
            real_id = _real_subject_id(subj_id)
            np.savez_compressed(
                path, X=Xw, y=yw,
                record_id=str(subj_id), subject_id=real_id,
                cache_version=CACHE_VERSION,
                context=context, causal=causal,
                preprocess="0.5-40Hz_100Hz_uV_noNorm_chronov3",
            )
            done += 1
        except (ValueError, RuntimeError) as e:
            print(f"  [FAILED] record={subj_id}: {type(e).__name__}: {e}")
            failed += 1

    print(f"  Cache done: {done} built, {skipped} skipped, {failed} failed")
    if done == 0 and skipped == 0:
        raise RuntimeError(
            "No chronov3 caches built or found. "
            "Check extraction errors above and verify data_root path."
        )


def load_windows(subjects, cache_dir, context, causal):
    X_list, y_list, subj_list = [], [], []
    for s in subjects:
        p = find_cache(s, cache_dir, context, causal)
        if p is not None:
            d = np.load(p)
            X_list.append(d['X']); y_list.append(d['y'])
            # Use REAL subject ID (without night suffix) for subject-wise splits.
            # SC4ssNE0 → ss is the subject, N is the night.
            real_id = _real_subject_id(s)
            subj_list.extend([real_id] * len(d['y']))
    return np.concatenate(X_list), np.concatenate(y_list), np.array(subj_list)


# =============================================================================
# Model factory
# =============================================================================

def create_model():
    """Create model based on --model and --aux flags."""
    use_ta = (args.model == 'ta')
    use_aux = (args.aux == 'multitask')
    # Causal context → target last epoch; center context → target middle epoch
    target_idx = 'last' if (use_ta and args.causal) else 'center'
    raw_cls = lwmod.ParaSleep.module
    model = raw_cls(
        n_channels=args.context, n_samples=3000, n_classes=5,
        use_temporal_attention=use_ta,
        use_aux=use_aux,
        target_index=target_idx,
    ).float()
    return model.to(DEVICE)


# =============================================================================
# Train single fold
# =============================================================================

def train_fold(X_train, y_train, subj_train, X_test, y_test,
               fold_name='', save_path=None):
    global_start = time.time()
    if save_path is None:
        save_path = args.save

    # === Subject-wise validation split ===
    unique_subs = np.unique(subj_train)
    rng = np.random.RandomState(args.seed)
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

    # Class weights (from training set, robust to missing classes)
    counts = np.bincount(y_tr, minlength=5)
    raw = np.sqrt(len(y_tr) / np.maximum(counts, 1))
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

    best_val_f1 = 0.0
    best_state = None
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

        # Validation
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
        vp = np.concatenate(all_vpred); vt = np.concatenate(all_vtrue)
        val_acc = (vp == vt).mean()
        val_f1 = f1_score(vt, vp, average='macro', zero_division=0)

        train_losses.append(tr_loss / len(X_tr))
        val_accs.append(val_acc)

        # Save best-val model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)

        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.time() - global_start
            # Print prediction distribution to detect collapse early
            pred_count = np.bincount(vp, minlength=5)
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss={tr_loss/len(X_tr):.4f} | "
                  f"val_f1={val_f1:.3f} | lr={opt.param_groups[0]['lr']:.1e} | "
                  f"{elapsed/60:.0f}min")
            print(f"  Val pred dist: W={pred_count[0]} N1={pred_count[1]} "
                  f"N2={pred_count[2]} N3={pred_count[3]} REM={pred_count[4]}")

    # Test with best-val model
    model.load_state_dict(best_state); model.eval()
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
all_records = load_cache_subjects(args.cache, args.context, args.causal)
print(f'Cached records: {len(all_records)}')

# Build real-subject → records mapping (SC4001,SC4002 → both belong to subject "00")
subject_to_records = build_subject_records_map(all_records)
real_subjects = sorted(subject_to_records.keys())
print(f'Real subjects: {len(real_subjects)} (from {len(all_records)} records)')

# Strict dataset validation for final protocol
if args.strict_dataset:
    expected = {'records': 153, 'subjects': 78, 'train': 68, 'test': 10}
    if len(all_records) != expected['records']:
        raise RuntimeError(f"Expected {expected['records']} records, got {len(all_records)}")
    if len(real_subjects) != expected['subjects']:
        raise RuntimeError(f"Expected {expected['subjects']} subjects, got {len(real_subjects)}")
    if args.subjects != expected['train'] or args.test != expected['test']:
        raise RuntimeError(f"Final protocol: --subjects {expected['train']} --test {expected['test']}")
    print(f'Strict dataset check: 153 records / 78 subjects / 68+10 split — OK')

rng = np.random.RandomState(args.seed)
rng.shuffle(real_subjects)

# Split by REAL subject IDs (nights from same subject stay together)
n_train_subjects = min(args.subjects, len(real_subjects) - args.test)
n_test_subjects = args.test

train_subjects = sorted(real_subjects[:n_train_subjects])
test_subjects = sorted(real_subjects[n_train_subjects:n_train_subjects + n_test_subjects])

# Expand subjects → record IDs for cache loading (sorted for determinism)
train_records = sorted(r for s in train_subjects for r in subject_to_records[s])
test_records = sorted(r for s in test_subjects for r in subject_to_records[s])

# Assert disjoint
assert set(train_subjects).isdisjoint(test_subjects), \
    "Train/test subject overlap detected!"
print(f'Disjoint check: OK')

# Save split (both subject IDs and record IDs)
os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
np.savez(args.save.replace('.pth', '_split.npz'),
         train_subjects=np.array(train_subjects),
         test_subjects=np.array(test_subjects),
         train_records=np.array(train_records),
         test_records=np.array(test_records))
print(f'Split saved: {args.save.replace(".pth", "_split.npz")}')

# Load data: CV uses train_records only; holdout uses both
if args.cv > 1:
    print(f'\nRunning {args.cv}-fold subject-wise CV on {len(train_records)} train records '
          f'({n_train_subjects} real subjects)...')
    set_global_seed(args.seed)

    X_cv, y_cv, subj_cv = load_windows(train_records, args.cache,
                                        args.context, args.causal)
    cv_unique = np.unique(subj_cv)
    assert set(test_subjects).isdisjoint(set(cv_unique)), \
        "Holdout test subjects leaked into CV!"
    print(f'CV isolation check: OK (holdout test subjects excluded)')

    # Balanced fold split via np.array_split
    rng_cv = np.random.RandomState(args.cv_seed)
    rng_cv.shuffle(cv_unique)
    fold_groups = np.array_split(cv_unique, args.cv)

    # Save fold assignments
    fold_info = {}
    for i, grp in enumerate(fold_groups):
        fold_info[f'fold_{i+1}_subjects'] = np.array(sorted(grp))
    fold_info['holdout_subjects'] = np.array(sorted(test_subjects))
    fold_info['cv_seed'] = args.cv_seed
    folds_path = args.save.replace('.pth', '_folds.npz')
    np.savez(folds_path, **fold_info)
    print(f'CV folds saved: {folds_path}')

    all_folds = []
    cv_out_dir = os.path.dirname(args.save) or '.'
    for fold, fold_subjects in enumerate(fold_groups):
        test_set = set(fold_subjects.tolist())
        train_set = set(cv_unique.tolist()) - test_set
        tr_mask = np.array([s in train_set for s in subj_cv])
        te_mask = np.array([s in test_set for s in subj_cv])

        fold_save = os.path.join(cv_out_dir,
            os.path.basename(args.save).replace('.pth', f'_fold{fold+1}.pth'))
        # Per-fold seed for reproducibility
        set_global_seed(args.seed + fold + 1)
        result = train_fold(X_cv[tr_mask], y_cv[tr_mask], subj_cv[tr_mask],
                            X_cv[te_mask], y_cv[te_mask],
                            fold_name=str(fold + 1), save_path=fold_save)
        all_folds.append(result)
        print(f"  Fold {fold+1} Macro F1: {result['macro_f1']:.4f} "
              f"({len(fold_subjects)} subjects)")

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    keys = ['accuracy', 'macro_f1', 'W_f1', 'N1_f1', 'N2_f1', 'N3_f1', 'REM_f1']
    labels = ['Accuracy', 'Macro F1', 'W F1', 'N1 F1', 'N2 F1', 'N3 F1', 'REM F1']
    for key, label in zip(keys, labels):
        vals = [f[key] for f in all_folds]
        print(f"  {label:12s}: {np.mean(vals)*100:.2f}% +- {np.std(vals)*100:.2f}%")

    # Save CV metrics
    import csv, json
    csv_path = os.path.join(cv_out_dir, 'cv_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'subjects', 'accuracy', 'macro_f1', 'W_f1', 'N1_f1', 'N2_f1', 'N3_f1', 'REM_f1'])
        for i, r in enumerate(all_folds):
            w.writerow([i+1, len(fold_groups[i]), r['accuracy'], r['macro_f1'],
                        r['W_f1'], r['N1_f1'], r['N2_f1'], r['N3_f1'], r['REM_f1']])
    print(f'CV metrics saved: {csv_path}')

    json_path = os.path.join(cv_out_dir, 'cv_summary.json')
    mean_vals = {k: float(np.mean([f[k] for f in all_folds])) for k in keys}
    std_vals = {k: float(np.std([f[k] for f in all_folds])) for k in keys}
    with open(json_path, 'w') as f:
        json.dump({'n_folds': args.cv, 'train_subjects': n_train_subjects,
                    'holdout_subjects': n_test_subjects, 'seed': args.seed,
                    'cv_seed': args.cv_seed, 'mean': mean_vals, 'std': std_vals}, f, indent=2)
    print(f'CV summary saved: {json_path}')

    print(f"\n  Final: Macro F1 = {np.mean([f['macro_f1'] for f in all_folds])*100:.2f}% "
          f"+- {np.std([f['macro_f1'] for f in all_folds])*100:.2f}%")

else:
    # Holdout mode: load both train + test records
    print(f'Loading {len(train_records)} train + {len(test_records)} test records...')
    X_all, y_all, subj_all = load_windows(train_records + test_records, args.cache,
                                           args.context, args.causal)
    print(f'\nTraining: {n_train_subjects} real subjects '
          f'({len(train_records)} records), {args.epochs} epochs')
    tr_set = set(train_subjects); te_set = set(test_subjects)
    tr_mask = np.array([s in tr_set for s in subj_all])
    te_mask = np.array([s in te_set for s in subj_all])

    result = train_fold(X_all[tr_mask], y_all[tr_mask], subj_all[tr_mask],
                        X_all[te_mask], y_all[te_mask], save_path=args.save)

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

if args.cv > 1:
    print(f"\nCV fold models saved under: {cv_out_dir}")
else:
    print(f"\nModel saved: {args.save}")
print(f"Done.")
