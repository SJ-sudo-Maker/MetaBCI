# -*- coding: utf-8 -*-
"""Reconstruct 5-fold CV metrics from already-trained fold models."""
import sys, os, json, csv, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                              classification_report, confusion_matrix)

import metabci.brainda.algorithms.deep_learning.parasleep as lwmod

# =============================================================================
# Config — adjust these paths to your setup
# =============================================================================
FOLDS_FILE = r"outputs\cv_final\exp_ctx3_causal_chronov3_cv_folds.npz"
SPLIT_FILE = r"outputs\cv_final\exp_ctx3_causal_chronov3_cv_split.npz"
MODEL_DIR  = r"outputs\cv_final"
CACHE_DIR  = r"F:\sleep_cache_chronov3"   # or wherever your chronov3 caches live
CONTEXT    = 3
CAUSAL     = True
OUT_DIR    = r"outputs\cv_final"
DEVICE     = "cpu"
BATCH_SIZE = 256

LABELS = [0, 1, 2, 3, 4]
NAMES  = ['W', 'N1', 'N2', 'N3', 'REM']

# =============================================================================
# Load fold assignments
# =============================================================================
folds_data = np.load(FOLDS_FILE, allow_pickle=True)
fold_subjects = [folds_data[f"fold_{i+1}_subjects"].tolist() for i in range(5)]
holdout_subjects = folds_data["holdout_subjects"].tolist()

# Load split to get record mappings
split_data = np.load(SPLIT_FILE, allow_pickle=True)
train_records = split_data["train_records"].tolist()
test_records = split_data["test_records"].tolist()

# Build subject→records mapping
from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
def _sid(rid): return str(rid)[1:3] if len(str(rid)) >= 3 else str(rid)

subj_to_records = {}
for r in train_records + test_records:
    subj_to_records.setdefault(_sid(str(r)), []).append(str(r))

# =============================================================================
# Load cache for specified records
# =============================================================================
mode = "causal" if CAUSAL else "center"

def load_cache(records):
    X_list, y_list = [], []
    for rid in records:
        # Try chronov3 first
        for ver in ["chronov3", "chronov2"]:
            pat = os.path.join(CACHE_DIR,
                f"{rid}_FpzCz_sr100_ctx{CONTEXT}_{mode}_5class_{ver}.npz")
            if os.path.exists(pat):
                break
        else:
            pat = os.path.join(CACHE_DIR,
                f"{rid}_FpzCz_sr100_ctx{CONTEXT}_{mode}_5class.npz")
        d = np.load(pat, allow_pickle=True)
        X_list.append(d['X']); y_list.append(d['y'])
    return np.concatenate(X_list), np.concatenate(y_list)

# =============================================================================
# Load model from checkpoint
# =============================================================================
def load_model(ckpt_path):
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    use_ta = any(k.startswith('transformer.') for k in state.keys())
    if use_ta and 'pos_embed' in state:
        n_ch = state['pos_embed'].shape[1]
    else:
        n_ch = CONTEXT
    target_idx = 'last' if (use_ta and CAUSAL) else 'center'
    raw_cls = lwmod.ParaSleep.module
    model = raw_cls(n_channels=n_ch, n_samples=3000, n_classes=5,
                    use_temporal_attention=use_ta,
                    target_index=target_idx).float().to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# =============================================================================
# Evaluate one fold
# =============================================================================
def eval_fold(fold_idx):
    subjects = fold_subjects[fold_idx]
    records = sorted(r for s in subjects for r in subj_to_records.get(s, [s]))
    print(f"\nFold {fold_idx+1}: {len(subjects)} subjects, {len(records)} records")

    X_test, y_test = load_cache(records)
    ckpt = os.path.join(MODEL_DIR,
        f"exp_ctx3_causal_chronov3_cv_fold{fold_idx+1}.pth")
    if not os.path.exists(ckpt):
        print(f"  SKIP: model not found: {ckpt}")
        return None

    model = load_model(ckpt)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test))
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_p, all_t = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            out = model(Xb.to(DEVICE))
            if isinstance(out, tuple): out = out[0]
            all_p.append(out.argmax(1).cpu().numpy())
            all_t.append(yb.numpy())
    yp = np.concatenate(all_p); yt = np.concatenate(all_t)

    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average='macro', labels=LABELS, zero_division=0)
    pf1 = f1_score(yt, yp, average=None, labels=LABELS, zero_division=0)
    kap = cohen_kappa_score(yt, yp)

    print(f"  Accuracy={acc*100:.2f}%  Macro-F1={mf1:.4f}  Kappa={kap:.4f}")
    for i, n in enumerate(NAMES):
        print(f"  {n}: F1={pf1[i]:.4f}")
    return {"fold": fold_idx+1, "subjects": len(subjects), "records": len(records),
            "accuracy": acc, "macro_f1": mf1,
            "W_f1": pf1[0], "N1_f1": pf1[1], "N2_f1": pf1[2],
            "N3_f1": pf1[3], "REM_f1": pf1[4], "kappa": kap,
            "y_pred": yp, "y_true": yt}

# =============================================================================
# Main
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)
results = []
for fi in range(5):
    r = eval_fold(fi)
    if r is not None:
        results.append(r)

if len(results) == 0:
    print("No fold results. Check CACHE_DIR path.")
    sys.exit(1)

# Compute mean/std from available folds
print(f"\n{'='*60}")
print(f"CV RESULTS ({len(results)}/{5} folds completed)")
print(f"{'='*60}")
keys = ['accuracy', 'macro_f1', 'W_f1', 'N1_f1', 'N2_f1', 'N3_f1', 'REM_f1']
key_labels = ['Accuracy', 'Macro F1', 'W F1', 'N1 F1', 'N2 F1', 'N3 F1', 'REM F1']
for k, lab in zip(keys, key_labels):
    vals = [r[k] for r in results]
    print(f"  {lab:12s}: {np.mean(vals)*100:.2f}% +- {np.std(vals)*100:.2f}%")

# Save CV metrics CSV
csv_path = os.path.join(OUT_DIR, "cv_metrics.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['fold', 'subjects', 'records', 'accuracy', 'macro_f1',
                'W_f1', 'N1_f1', 'N2_f1', 'N3_f1', 'REM_f1', 'kappa'])
    for r in results:
        w.writerow([r['fold'], r['subjects'], r['records'], r['accuracy'],
                    r['macro_f1'], r['W_f1'], r['N1_f1'], r['N2_f1'],
                    r['N3_f1'], r['REM_f1'], r['kappa']])
print(f"\nSaved: {csv_path}")

# Save summary JSON
json_path = os.path.join(OUT_DIR, "cv_summary.json")
mean_vals = {k: float(np.mean([r[k] for r in results])) for k in keys}
std_vals = {k: float(np.std([r[k] for r in results])) for k in keys}
with open(json_path, 'w') as f:
    json.dump({
        "n_completed_folds": len(results),
        "n_total_folds": 5,
        "note": f"Training interrupted; {5-len(results)} fold(s) not completed.",
        "mean": mean_vals,
        "std": std_vals,
    }, f, indent=2)
print(f"Saved: {json_path}")

# Confusion matrix (aggregate across all folds)
if results:
    yt_all = np.concatenate([r['y_true'] for r in results])
    yp_all = np.concatenate([r['y_pred'] for r in results])
    cm = confusion_matrix(yt_all, yp_all, labels=LABELS)
    print(f"\nAggregate Confusion Matrix ({len(results)} folds):")
    print(cm)
    oof_acc = accuracy_score(yt_all, yp_all)
    oof_mf1 = f1_score(yt_all, yp_all, average='macro', labels=LABELS, zero_division=0)
    oof_kap = cohen_kappa_score(yt_all, yp_all)
    print(f"OOF Accuracy={oof_acc*100:.2f}%  Macro-F1={oof_mf1:.4f}  Kappa={oof_kap:.4f}")

print("\nDone.")
