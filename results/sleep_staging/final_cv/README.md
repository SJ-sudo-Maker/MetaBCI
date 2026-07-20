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
- Optimizer: AdamW, initial lr=1e-3, reduced to 1e-4 at epoch 11 (60 epochs total)
- Sampler: none
- Label smoothing: 0
- Auxiliary tasks: none

## Cross-Validation Protocol
- Type: Subject-wise 5-fold cross-validation (deterministic shuffled subject groups)
- Seed: 42
- CV seed: 42
- Fold sizes: 14, 14, 14, 13, 13 subjects
- Holdout isolation: verified (disjoint check passed)
## Results — Measured Subject-wise 5-fold Cross-Validation

| Metric | Mean ± Std |
|--------|-----------|
| Accuracy | 89.11% ± 0.92% |
| Macro F1 | 0.7443 ± 0.0178 |
| W F1 | 0.9706 ± 0.0056 |
| N1 F1 | 0.4333 ± 0.0274 |
| N2 F1 | 0.8352 ± 0.0091 |
| N3 F1 | 0.7750 ± 0.0497 |
| REM F1 | 0.7072 ± 0.0272 |

### Per-fold detail

| Fold | Subjects | Accuracy | Macro F1 | W F1 | N1 F1 | N2 F1 | N3 F1 | REM F1 | Note |
|------|----------|----------|----------|------|-------|-------|-------|--------|------|
| 1 | 14 | 90.33% | 0.7749 | 0.9749 | 0.4551 | 0.8484 | 0.8501 | 0.7459 | actual |
| 2 | 14 | 89.22% | 0.7328 | 0.9761 | 0.4747 | 0.8206 | 0.7291 | 0.6636 | actual |
| 3 | 14 | 89.79% | 0.7528 | 0.9739 | 0.4187 | 0.8320 | 0.8166 | 0.7228 | actual |
| 4 | 13 | 87.73% | 0.7252 | 0.9616 | 0.3998 | 0.8384 | 0.7261 | 0.6999 | actual |
| 5 | 13 | 88.50% | 0.7358 | 0.9666 | 0.4184 | 0.8366 | 0.7533 | 0.7040 | actual |

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
