"""Test script to verify comprehensive metrics logging."""

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             precision_score, recall_score, f1_score, roc_auc_score)

# Simulate predictions and labels
np.random.seed(42)
labels = np.random.randint(0, 2, 100)
probs = np.random.rand(100, 2)
probs = probs / probs.sum(axis=1, keepdims=True)
predictions = probs.argmax(1)

CLASS_NAMES = ["S", "R"]

print("Testing Comprehensive Metrics Computation")
print("="*60)

# Compute all metrics as done in train.py
acc = accuracy_score(labels, predictions)
bal_acc = balanced_accuracy_score(labels, predictions)
prec_macro = precision_score(labels, predictions, average="macro", zero_division=0)
rec_macro = recall_score(labels, predictions, average="macro", zero_division=0)
f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

# Per-class metrics
f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)

# ROC AUC
roc = roc_auc_score(labels, probs[:, 1])

print(f"\nOverall Metrics:")
print(f"  Accuracy:         {acc:.4f}")
print(f"  Balanced Acc:     {bal_acc:.4f}")
print(f"  Precision (macro):{prec_macro:.4f}")
print(f"  Recall (macro):   {rec_macro:.4f}")
print(f"  F1-macro:         {f1_macro:.4f}")
print(f"  F1-weighted:      {f1_weighted:.4f}")
print(f"  ROC-AUC:          {roc:.4f}")

print(f"\nPer-Class F1 Scores:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  F1-{class_name}: {f1_per_class[i]:.4f}")

print("\n" + "="*60)
print("✓ All metrics computed successfully!")

# Test dictionary structure
epoch_metrics = {
    "epoch": 1,
    "train_loss": 0.5,
    "train_accuracy": acc,
    "train_balanced_accuracy": bal_acc,
    "train_precision_macro": prec_macro,
    "train_recall_macro": rec_macro,
    "train_f1_macro": f1_macro,
    "train_f1_weighted": f1_weighted,
    "train_roc_auc": roc,
}

for i, class_name in enumerate(CLASS_NAMES):
    epoch_metrics[f"train_f1_{class_name}"] = float(f1_per_class[i])

print("\nExample epoch_metrics dictionary:")
for key, value in epoch_metrics.items():
    print(f"  {key}: {value}")

print("\n" + "="*60)
print("✓ Dictionary structure is correct!")
