# -*- coding: utf-8 -*-
"""
Sleep staging metrics demo.

Computes Accuracy, Macro-F1, Weighted-F1, Cohen's Kappa, per-class
Precision/Recall/F1, and confusion matrix from a trained model.

Usage:
    python demo_metric.py --model parasleep_best.pth --cache F:\\sleep_cache
    python demo_metric.py --model parasleep_best.pth --cache F:\\sleep_cache --subjects 10
"""
import sys, os, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                              classification_report, confusion_matrix)

import metabci.brainda.algorithms.deep_learning.parasleep as lwmod

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='parasleep_best.pth')
parser.add_argument('--cache', type=str, default='data_cache')
parser.add_argument('--subjects', type=int, default=10,
                    help='Number of test subjects (ignored if --split provided)')
parser.add_argument('--context', type=int, default=3)
parser.add_argument('--causal', action='store_true',
                    help='Use causal context (match training --causal flag)')
parser.add_argument('--split', type=str, default=None,
                    help='Path to _split.npz file from training (optional, ensures exact test set)')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--out', type=str, default='demo_outputs')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
DEVICE = args.device if torch.cuda.is_available() else 'cpu'

CLASS_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']

# =============================================================================
# Load model
# =============================================================================
print(f"Loading model: {args.model}")
raw_cls = lwmod.ParaSleep.module
state = torch.load(args.model, map_location=DEVICE, weights_only=True)

# Auto-detect model config from state dict
if any(k.startswith('transformer.') for k in state.keys()):
    use_ta = True
    arch = 'TA'
else:
    use_ta = False
    arch = 'Base'

# Detect context from checkpoint
if use_ta and 'pos_embed' in state:
    n_ch = state['pos_embed'].shape[1]
elif 'mrfe.small_branch.dsconv.dw.weight' in state:
    n_ch = state['mrfe.small_branch.dsconv.dw.weight'].shape[0]
else:
    n_ch = args.context

target_idx = 'last' if (use_ta and args.causal) else 'center'
model = raw_cls(n_channels=n_ch, n_samples=3000, n_classes=5,
                use_temporal_attention=use_ta,
                target_index=target_idx).float().to(DEVICE)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"  Missing keys: {len(missing)}")
if unexpected:
    print(f"  Unexpected keys: {len(unexpected)}")
model.eval()
print(f"  Architecture: {arch} | ctx={n_ch} | target={target_idx} | "
      f"{sum(p.numel() for p in model.parameters()):,} params")

# =============================================================================
# Load test data
# =============================================================================
print(f"\nLoading test data from: {args.cache}")
mode = 'causal' if args.causal else 'center'

# If --split provided, use exact test subjects from training
if args.split and os.path.exists(args.split):
    split_data = np.load(args.split, allow_pickle=True)
    # Support both old ("test_subs") and new ("test_records", "test_subjects") formats
    if "test_records" in split_data:
        test_subs = set(map(str, split_data['test_records'].tolist()))
    elif "test_subs" in split_data:
        test_subs = set(map(str, split_data['test_subs'].tolist()))
    else:
        raise KeyError(f"Split file {args.split} missing 'test_records' or 'test_subs'")
    print(f"  Using split file: {args.split} ({len(test_subs)} test subjects)")
else:
    test_subs = None

import glob
mode = 'causal' if args.causal else 'center'
if args.causal:
    patterns = [
        f'*_FpzCz_sr100_ctx{args.context}_{mode}_5class_chronov3.npz',
        f'*_FpzCz_sr100_ctx{args.context}_{mode}_5class_chronov2.npz',
        f'*_FpzCz_sr100_ctx{args.context}_{mode}_5class.npz',
        f'*_ctx{args.context}_{mode}.npz',
    ]
else:
    patterns = [
        f'*_FpzCz_sr100_ctx{args.context}_{mode}_5class_chronov3.npz',
        f'*_FpzCz_sr100_ctx{args.context}_{mode}_5class_chronov2.npz',
        f'*_FpzCz_sr100_ctx{args.context}_{mode}_5class.npz',
        f'*_ctx{args.context}_{mode}.npz',
        f'*_FpzCz_sr100_ctx{args.context}_center_5class.npz',
        f'*_ctx{args.context}_center.npz',
        '*.npz',
    ]
test_files = []
for pat in patterns:
    test_files = sorted(glob.glob(os.path.join(args.cache, pat)))
    if test_files:
        break

# Filter to test subjects if split provided
if test_subs is not None:
    test_files = [f for f in test_files
                  if os.path.basename(f).split('_')[0].replace('.npz', '') in test_subs]
else:
    test_files = test_files[:args.subjects]
print(f"  Found {len(test_files)} test subjects")
if len(test_files) == 0:
    raise FileNotFoundError(
        f"No cache files found in {args.cache} for ctx={args.context}, mode={mode}. "
        f"Check --cache path, --context, --causal flag, and --split file."
    )
for f in test_files:
    print(f"    {os.path.basename(f)}")
if test_subs is not None:
    print(f"  Split test_subs: {sorted(test_subs)}")

X_list, y_list = [], []
for f in test_files:
    d = np.load(f)
    X_list.append(d['X'])
    y_list.append(d['y'])

X_test = np.concatenate(X_list)
y_true = np.concatenate(y_list)
print(f"  Test set: {X_test.shape[0]} epochs")

# =============================================================================
# Inference
# =============================================================================
print("\nRunning inference...")
all_preds = []
with torch.no_grad():
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_true))
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    for Xb, _ in loader:
        Xb = Xb.to(DEVICE)
        out = model(Xb)
        if isinstance(out, tuple):
            out = out[0]
        all_preds.append(out.argmax(1).cpu().numpy())

y_pred = np.concatenate(all_preds)

# =============================================================================
# Metrics
# =============================================================================
LABELS_5 = [0, 1, 2, 3, 4]
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
kappa = cohen_kappa_score(y_true, y_pred)
per_class_f1 = f1_score(y_true, y_pred, average=None, labels=LABELS_5, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=LABELS_5)

n_records = len(test_files)
from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
n_subjects = len(set(
    SleepEDFDataset.parse_record_id(
        os.path.basename(f).split('_')[0].replace('.npz',''))[0]
    for f in test_files))
print(f"\n{'='*60}")
print(f"RESULTS — {n_subjects} test subjects ({n_records} records), {len(y_true)} epochs")
print(f"{'='*60}")
print(f"Accuracy:     {acc*100:.2f}%")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Weighted F1:  {weighted_f1:.4f}")
print(f"Cohen Kappa:  {kappa:.4f}")
print()
for i, name in enumerate(CLASS_NAMES):
    support = cm[i].sum()
    print(f"  {name}: P={cm[i,i]/max(cm[:,i].sum(),1):.3f}  "
          f"R={cm[i,i]/max(support,1):.3f}  "
          f"F1={per_class_f1[i]:.4f}  (n={support})")
print()
print(classification_report(y_true, y_pred, labels=LABELS_5,
      target_names=CLASS_NAMES, digits=4, zero_division=0))
print("Confusion Matrix:")
print(cm)

# =============================================================================
# Save
# =============================================================================
csv_path = os.path.join(args.out, 'metrics_summary.csv')
with open(csv_path, 'w') as f:
    f.write("Metric,Value\n")
    f.write(f"Accuracy,{acc:.6f}\n")
    f.write(f"Macro_F1,{macro_f1:.6f}\n")
    f.write(f"Weighted_F1,{weighted_f1:.6f}\n")
    f.write(f"Cohens_Kappa,{kappa:.6f}\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"F1_{name},{per_class_f1[i]:.6f}\n")
    f.write(f"Test_Epochs,{len(y_true)}\n")
    f.write(f"Test_Subjects,{n_subjects}\n")
    f.write(f"Test_Records,{n_records}\n")
print(f"\nSaved: {csv_path}")

report_path = os.path.join(args.out, 'classification_report.csv')
with open(report_path, 'w') as f:
    f.write("Class,Precision,Recall,F1-Score,Support\n")
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i].sum(), 1)
        f1_val = per_class_f1[i]
        f.write(f"{name},{prec:.4f},{rec:.4f},{f1_val:.4f},{cm[i].sum()}\n")
print(f"Saved: {report_path}")

# Confusion matrix plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(7, 6))
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(1e-8)
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1%})',
                ha='center', va='center', fontsize=9,
                color='white' if cm_norm[i,j] > 0.5 else 'black')
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix — Macro F1={macro_f1:.3f}, Kappa={kappa:.3f}')
plt.colorbar(im, ax=ax, label='Proportion')
fig.tight_layout()
cm_path = os.path.join(args.out, 'confusion_matrix.png')
fig.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {cm_path}")

print(f"\nDone. All outputs in: {os.path.abspath(args.out)}")
