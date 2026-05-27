# -*- coding: utf-8 -*-
"""
LWSleepNet training script for single-channel EEG sleep staging.

Usage
-----
    python train_lwsleepnet.py

Configuration: edit DATA_ROOT, N_SUBJECTS, MAX_EPOCHS below.
"""

import sys
from pathlib import Path

# Add project root to path (assuming script is in examples/sleep_staging/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from skorch.classifier import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EpochScoring, LRScheduler

from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
from metabci.brainda.paradigms.sleep import SleepParadigm
import metabci.brainda.algorithms.deep_learning.lwsleepnet as lwmod


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = r"D:\sleep eeg\sleep-edf-database-expanded-1.0.0\sleep-cassette"
CHANNEL = "EEG Fpz-Cz"
N_TRAIN_SUBJECTS = 20        # number of subjects for training
N_TEST_SUBJECTS = 3          # number of subjects for testing
START_INDEX = 0              # subject start index (0 = first subject)
MAX_EPOCHS = 100             # training epochs
BATCH_SIZE = 120
LR = 1e-3
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 1.0
USE_CLASS_WEIGHTS = True     # inverse-frequency class weights (fixes N1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Data loading
# =============================================================================

print(f"Device: {DEVICE}")
print(f"Loading data from: {DATA_ROOT}")

dataset = SleepEDFDataset(DATA_ROOT, channel=CHANNEL)
paradigm = SleepParadigm(channels=[CHANNEL], srate=100)

all_subjects = dataset.subjects[
    START_INDEX : START_INDEX + N_TRAIN_SUBJECTS + N_TEST_SUBJECTS
]
train_subs = all_subjects[:N_TRAIN_SUBJECTS]
test_subs = all_subjects[N_TRAIN_SUBJECTS:]

print(f"Train subjects ({len(train_subs)}): {train_subs}")
print(f"Test  subjects ({len(test_subs)}): {test_subs}")

X_train, y_train, _ = paradigm.get_data(
    dataset, subjects=train_subs, return_concat=True, n_jobs=1,
)
X_test, y_test, _ = paradigm.get_data(
    dataset, subjects=test_subs, return_concat=True, n_jobs=1,
)

# Convert to float32 for GPU
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int64)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int64)

n_samples = X_train.shape[2]  # should be 3000 for 30s @ 100Hz
n_channels = X_train.shape[1]  # should be 1
n_classes = len(np.unique(y_train))

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

unique, counts = np.unique(y_train, return_counts=True)
print("Train class distribution:")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c} ({100*c/len(y_train):.1f}%)")

unique_t, counts_t = np.unique(y_test, return_counts=True)
print("Test  class distribution:")
for u, c in zip(unique_t, counts_t):
    print(f"  Class {u}: {c} ({100*c/len(y_test):.1f}%)")


# =============================================================================
# Class weights (inverse frequency)
# =============================================================================

if USE_CLASS_WEIGHTS:
    total = len(y_train)
    # Square-root inverse frequency — softer than pure inverse,
    # prevents minority classes (N1) from dominating the loss.
    raw_weights = np.sqrt([total / c for c in counts])
    # Normalize so the smallest class weight = 1.0, cap at 10x
    raw_weights = raw_weights / raw_weights.min()
    raw_weights = np.clip(raw_weights, 1.0, 10.0)
    class_weights = torch.tensor(raw_weights, dtype=torch.float32)
    print(f"Class weights (sqrt inverse, capped @10x): {class_weights.tolist()}")
else:
    class_weights = None


# =============================================================================
# Model
# =============================================================================

raw_cls = lwmod.LWSleepNet.module
model = raw_cls(
    n_channels=n_channels, n_samples=n_samples, n_classes=n_classes,
).float()

criterion_kwargs = {"label_smoothing": LABEL_SMOOTHING}
if class_weights is not None:
    criterion_kwargs["weight"] = class_weights.to(DEVICE)

net = NeuralNetClassifier(
    model,
    criterion=nn.CrossEntropyLoss,
    criterion__reduce=True,
    **{f"criterion__{k}": v for k, v in criterion_kwargs.items()},
    optimizer=optim.AdamW,
    optimizer__weight_decay=WEIGHT_DECAY,
    optimizer__betas=(0.9, 0.999),
    batch_size=BATCH_SIZE,
    lr=LR,
    max_epochs=MAX_EPOCHS,
    device=DEVICE,
    train_split=ValidSplit(0.2, stratified=True),
    iterator_train__shuffle=True,
    callbacks=[
        (
            "train_acc",
            EpochScoring(
                "accuracy", name="train_acc", on_train=True, lower_is_better=False,
            ),
        ),
        (
            "lr_scheduler",
            LRScheduler("CosineAnnealingLR", T_max=MAX_EPOCHS),
        ),
    ],
    verbose=True,
)


# =============================================================================
# Training
# =============================================================================

print(f"\n{'='*60}")
print(f"Training: {MAX_EPOCHS} epochs, batch_size={BATCH_SIZE}, device={DEVICE}")
print(f"Class weights: {USE_CLASS_WEIGHTS}, Label smoothing: {LABEL_SMOOTHING}")
print(f"{'='*60}\n")

net.fit(X_train, y_train)


# =============================================================================
# Evaluation
# =============================================================================

train_acc = net.score(X_train, y_train)
test_acc = net.score(X_test, y_test)
y_pred = net.predict(X_test)

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Test  Accuracy: {test_acc*100:.2f}%")
print()

target_names = ["W", "N1", "N2", "N3", "REM"]
print(classification_report(
    y_test, y_pred, target_names=target_names, digits=4, zero_division=0,
))
print("Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))
