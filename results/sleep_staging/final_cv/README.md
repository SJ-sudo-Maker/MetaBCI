# Subject-wise 5-fold CV Results — ParaSleep (ctx=3 causal)

## Dataset
- Dataset: Sleep-EDF Expanded sleep-cassette
- Total records: 153
- Real subjects: 78
- Development set: 68 subjects, 134 records
- Independent holdout: 10 subjects, 19 records
- Channel: EEG Fpz-Cz
- Sample rate: 100 Hz
- Epoch: 30 seconds
- Context window: [t-2, t-1, t] → predict t (causal)

## Model
- Architecture: ParaSleep (Base)
- Parameters: 131,885
- Loss: Focal Loss (γ=2) + sqrt class weights
- Optimizer: AdamW (lr=1e-3→1e-4→1e-5, wd=1e-2)
- Sampler: none
- Label smoothing: 0
- Auxiliary tasks: none

## Cross-Validation Protocol
- Type: Subject-wise 5-fold (GroupKFold)
- Seed: 42
- CV seed: 42
- Fold sizes: 14, 14, 14, 13, 13 subjects
- Holdout isolation: verified (disjoint check passed)
- Note: Fold 5 training was interrupted; results shown for 4/5 completed folds.

## Results (4/5 folds completed)

| Metric | Mean ± Std |
|--------|-----------|
| Accuracy | 89.27% ± 0.97% |
| Macro F1 | 0.7464 ± 0.0193 |
| W F1 | 0.9716 ± 0.0058 |
| N1 F1 | 0.4371 ± 0.0294 |
| N2 F1 | 0.8349 ± 0.0101 |
| N3 F1 | 0.7805 ± 0.0542 |
| REM F1 | 0.7081 ± 0.0304 |

### Per-fold detail

| Fold | Subjects | Accuracy | Macro F1 | W F1 | N1 F1 | N2 F1 | N3 F1 | REM F1 |
|------|----------|----------|----------|------|-------|-------|-------|--------|
| 1 | 14 | 90.33% | 0.7749 | 0.9749 | 0.4551 | 0.8484 | 0.8501 | 0.7459 |
| 2 | 14 | 89.22% | 0.7328 | 0.9761 | 0.4747 | 0.8206 | 0.7291 | 0.6636 |
| 3 | 14 | 89.79% | 0.7528 | 0.9739 | 0.4187 | 0.8320 | 0.8166 | 0.7228 |
| 4 | 13 | 87.73% | 0.7252 | 0.9616 | 0.3998 | 0.8384 | 0.7261 | 0.6999 |

## Holdout (independent test set)
| Metric | Value |
|--------|-------|
| Accuracy | 88.81% |
| Macro F1 | 0.7204 |
| Weighted F1 | 0.8956 |
| Cohen's Kappa | 0.7779 |

## Model Weights
Model weights (.pth files) are not tracked in Git due to size.
Available via competition submission package or GitHub Release.
