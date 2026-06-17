# -*- coding: utf-8 -*-
"""
ParaSleep training with 3-epoch context windows for N1 recognition.

Each input = [epoch_{t-1}, epoch_t, epoch_{t+1}] as 3 "channels",
letting the model learn W→N1→N2 transition patterns.

Usage:  python train_sleep.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from skorch.classifier import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EpochScoring, LRScheduler

from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
from metabci.brainda.paradigms.sleep import SleepParadigm
import metabci.brainda.algorithms.deep_learning.parasleep as lwmod


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = r"D:\sleep eeg\sleep-edf-database-expanded-1.0.0\sleep-cassette"
CHANNEL = "EEG Fpz-Cz"

N_TRAIN_SUBJECTS = 80
N_TEST_SUBJECTS = 10
START_INDEX = 0

CONTEXT_WINDOW = 3      # odd number: middle epoch = target
assert CONTEXT_WINDOW % 2 == 1, "Must be odd"

MAX_EPOCHS = 100
BATCH_SIZE = 120
LR = 1e-3
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 1.0
USE_CLASS_WEIGHTS = "sqrt"
FOCAL_GAMMA = 2.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# =============================================================================
# Data loading — with 3-epoch sliding windows
# =============================================================================

print(f"Device: {DEVICE}")
print(f"Loading data from: {DATA_ROOT}")

dataset = SleepEDFDataset(DATA_ROOT, channel=CHANNEL)
paradigm = SleepParadigm(channels=[CHANNEL], srate=100)

all_subjects = dataset.subjects
n_available = len(all_subjects)
actual_train = min(N_TRAIN_SUBJECTS, n_available - N_TEST_SUBJECTS)
actual_test = min(N_TEST_SUBJECTS, n_available - actual_train)

train_subs = all_subjects[START_INDEX:START_INDEX + actual_train]
test_subs = all_subjects[START_INDEX + actual_train:
                         START_INDEX + actual_train + actual_test]

# Filter bad subjects
def _load_subject(subject):
    X, y, _ = paradigm.get_data(
        dataset, subjects=[subject], return_concat=True, n_jobs=1)
    return X, y

def _is_valid(subject):
    try:
        _load_subject(subject)
        return True
    except ValueError:
        return False

train_subs = [s for s in train_subs if _is_valid(s)]
test_subs = [s for s in test_subs if _is_valid(s)]

print(f"Train subjects: {len(train_subs)} ({train_subs[0]}...{train_subs[-1]})")
print(f"Test  subjects: {len(test_subs)} ({test_subs})")

# Build 3-epoch context windows
half = CONTEXT_WINDOW // 2
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for s in train_subs:
    X, y = _load_subject(s)
    X = X.astype(np.float32)
    n = X.shape[0]
    for i in range(half, n - half):
        window = X[i - half:i + half + 1, 0, :]   # (3, 3000)
        X_train_list.append(window)
        y_train_list.append(y[i])

for s in test_subs:
    X, y = _load_subject(s)
    X = X.astype(np.float32)
    n = X.shape[0]
    for i in range(half, n - half):
        window = X[i - half:i + half + 1, 0, :]
        X_test_list.append(window)
        y_test_list.append(y[i])

X_train = np.stack(X_train_list)  # (N, 3, 3000)
y_train = np.array(y_train_list, dtype=np.int64)
X_test = np.stack(X_test_list)
y_test = np.array(y_test_list, dtype=np.int64)

n_samples = X_train.shape[2]
n_channels = X_train.shape[1]   # = 3 (context windows)
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
# Class weights
# =============================================================================

if USE_CLASS_WEIGHTS == "sqrt":
    total = len(y_train)
    raw = np.sqrt([total / c for c in counts])
    raw = raw / raw.min()
    raw = np.clip(raw, 1.0, 10.0)
    class_weights = torch.tensor(raw, dtype=torch.float32)
    print(f"Class weights (sqrt, capped@10x): {class_weights.tolist()}")
elif USE_CLASS_WEIGHTS == "inverse":
    raw = np.array([len(y_train) / c for c in counts])
    raw = raw / raw.min()
    class_weights = torch.tensor(raw, dtype=torch.float32)
else:
    class_weights = None

# =============================================================================
# Model
# =============================================================================

raw_cls = lwmod.ParaSleep.module
model = raw_cls(
    n_channels=n_channels, n_samples=n_samples, n_classes=n_classes,
).float()

if FOCAL_GAMMA > 0:
    criterion = FocalLoss(
        gamma=FOCAL_GAMMA,
        weight=class_weights.to(DEVICE) if class_weights is not None else None,
        label_smoothing=LABEL_SMOOTHING,
    )
    print(f"Loss: FocalLoss(gamma={FOCAL_GAMMA})")
else:
    kwargs = {"label_smoothing": LABEL_SMOOTHING}
    if class_weights is not None:
        kwargs["weight"] = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(**kwargs)

net = NeuralNetClassifier(
    model,
    criterion=criterion,
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
        ("train_acc", EpochScoring(
            "accuracy", name="train_acc", on_train=True, lower_is_better=False,
        )),
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=MAX_EPOCHS)),
    ],
    verbose=True,
)

# =============================================================================
# Training
# =============================================================================

print(f"\n{'='*60}")
print(f"Context: {CONTEXT_WINDOW} epochs → {n_channels} input channels")
print(f"Training: {len(train_subs)} subjects, {X_train.shape[0]:,} windows, {MAX_EPOCHS} epochs")
print(f"Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  Focal γ: {FOCAL_GAMMA}")
print(f"{'='*60}\n")

net.fit(X_train, y_train)

# =============================================================================
# Evaluation
# =============================================================================

train_acc = net.score(X_train, y_train)
test_acc = net.score(X_test, y_test)
y_pred = net.predict(X_test)

print(f"\n{'='*60}")
print("RESULTS")
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

# Auto-save best model
import os
save_path = "parasleep.pth"
torch.save(net.module_.state_dict(), save_path)
print(f"\nModel saved to: {os.path.abspath(save_path)}")
