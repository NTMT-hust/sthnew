# DRIAMS Antibiotic Resistance Prediction - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Focal Loss Implementation](#focal-loss-implementation)
4. [Split Strategies](#split-strategies)
5. [Comprehensive Metrics](#comprehensive-metrics)
6. [Automatic Visualization](#automatic-visualization)
7. [Quick Start](#quick-start)
8. [Command Reference](#command-reference)
9. [Output Files](#output-files)
10. [Usage Examples](#usage-examples)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a hybrid heterogeneous graph neural network for predicting antibiotic resistance from MALDI-TOF mass spectrometry data. The model combines:
- **Spectral embeddings** from MALDI-TOF spectra (via Variational Autoencoder)
- **Molecular graphs** from antibiotic SMILES structures
- **Heterogeneous graph learning** to model isolate-drug-resistance relationships

### Key Improvements
- ✅ **Focal Loss** for handling class imbalance
- ✅ **Two split strategies** (cross-site and within-site evaluation)
- ✅ **Comprehensive metrics** logged for train/val/test every epoch
- ✅ **Automatic visualization** with value labels on bar charts
- ✅ **Per-class tracking** (Susceptible vs Resistant)

---

## Features

### 1. Focal Loss Implementation

**Focal Loss** addresses class imbalance by down-weighting easy examples and focusing on hard, misclassified examples.

#### Formula
```
FL(pt) = -α(1 - pt)^γ * log(pt)
```

Where:
- **pt**: Predicted probability of the true class
- **α (alpha)**: Class weighting factor (automatic from class distribution)
- **γ (gamma)**: Focusing parameter (default: 2.0)

#### Benefits
- Better handles imbalanced datasets (R vs S classes)
- Focuses training on difficult examples
- Reduces false negatives (critical in clinical settings)
- Improves minority class performance

#### Usage
```bash
# Default focal loss (gamma=2.0)
python train.py --focal-gamma 2.0

# Strong focusing
python train.py --focal-gamma 5.0

# Cross-entropy baseline (gamma=0.0)
python train.py --focal-gamma 0.0
```

#### How It Works
For a well-classified example with 95% confidence:
- **Cross-entropy weight**: 1.0
- **Focal loss weight (γ=2)**: (1-0.95)² = 0.0025 (400× reduction!)

This forces the model to focus on hard examples that need improvement.

---

## Split Strategies

### Strategy 1: Site Split (Default) - Cross-Site Generalization

```bash
python train.py --split-strategy site_split
```

| Split | Data | Purpose |
|-------|------|---------|
| **Train** | Site A | Training data from primary site |
| **Validation** | Sites D + C | Combined validation from two sites |
| **Test** | Site B | Hold-out test site |

**Use when:** You want to evaluate model generalization across different hospital sites (recommended for deployment).

### Strategy 2: Single Site Split - Within-Site Evaluation

```bash
python train.py --split-strategy single_site
```

| Split | Data | Purpose |
|-------|------|---------|
| **Train** | 70% of Site A | Training subset |
| **Validation** | 15% of Site A | Validation subset |
| **Test** | 15% of Site A | Test subset |

**Use when:** You want to evaluate model performance without site-specific confounders (research/analysis).

---

## Comprehensive Metrics

### Metrics Logged (Every Epoch, All Splits)

For each of **train**, **validation**, and **test** sets:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **Balanced Accuracy** | Accuracy adjusted for class imbalance |
| **Precision (macro)** | Average precision across classes |
| **Recall (macro)** | Average recall across classes |
| **F1-Macro** | Macro-averaged F1 score |
| **F1-Weighted** | Weighted F1 score by class support |
| **F1-S** | F1 score for Susceptible class |
| **F1-R** | F1 score for Resistant class |
| **ROC-AUC** | Area under ROC curve |
| **Train Loss** | Focal loss value (training only) |

### Console Output Example

```
Epoch 15/50 | Loss: 0.35234
  Train    - Acc: 0.8234, F1-macro: 0.8123, F1-S: 0.8534, F1-R: 0.7712
  Val      - Acc: 0.7534, F1-macro: 0.7423, F1-S: 0.7834, F1-R: 0.7012
  Test     - Acc: 0.7434, F1-macro: 0.7323, F1-S: 0.7734, F1-R: 0.6912
```

### Best Epoch Summary

At the end of training, a comprehensive summary shows all metrics for the best epoch:

```
======================================================================
TRAINING COMPLETE - Best Epoch: 23
======================================================================

Best Epoch Metrics (Epoch 23):
  Training Loss: 0.32145
  
  TRAIN SET:
    Accuracy:         0.8234
    Balanced Acc:     0.8156
    Precision (macro):0.8089
    Recall (macro):   0.8156
    F1-macro:         0.8123
    F1-weighted:      0.8145
    ROC-AUC:          0.8934
    F1-S (Suscept):   0.8534
    F1-R (Resistant): 0.7712
  
  VALIDATION SET:
    [Full metrics...]
  
  TEST SET:
    [Full metrics...]
======================================================================
```

---

## Automatic Visualization

After training, **five comprehensive visualizations** are automatically generated:

### 1. Training Curves (`training_curves_*.png`)
- Training loss over epochs
- F1-Macro for train/val/test
- Accuracy for train/val/test
- Balanced Accuracy for train/val/test
- ROC-AUC for train/val/test

**Line plots with markers** showing metric progression over training.

### 2. Per-Class F1 Curves (`per_class_f1_*.png`)
- F1 scores for Susceptible (S) class
- F1 scores for Resistant (R) class
- Separate panels for train, validation, test

**Monitors class-specific performance** to detect imbalanced learning.

### 3. Best Epoch Metrics Bar Chart (`best_epoch_metrics_*.png`) ⭐
- **6 key metrics** at best epoch
- **Value labels displayed on top of each bar**
- Metrics: Accuracy, Balanced Acc, Precision, Recall, F1-Macro, ROC-AUC
- Color-coded: Train (blue), Validation (purple), Test (orange)

**Easy visual comparison** with exact values visible.

### 4. Per-Class Comparison (`per_class_comparison_*.png`) ⭐
- Side-by-side bars: F1-S vs F1-R
- **Value labels on each bar**
- Shows class balance at best epoch

**Identify if one class is underperforming.**

### 5. Comprehensive Results Overview (`comprehensive_results_*.png`)
- All-in-one dashboard with:
  - Overall metrics comparison (4-metric bar chart)
  - Per-split detailed metrics (3 panels)
  - Validation F1 progress curve with best epoch marker

**Complete results at a glance.**

⭐ = **Value labels displayed on top of bars for easy reading!**

### Manual Visualization

If automatic visualization fails or is disabled:

```bash
python visualize_results.py \
    --output-dir outputs/experiment1 \
    --split-mode site_split
```

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn rdkit

# Or use requirements.txt
pip install -r requirements.txt
```

### Basic Training

```bash
# Site split with auto-visualization (recommended)
python train.py \
    --split-strategy site_split \
    --output outputs/site_split \
    --epochs 50

# Single site evaluation
python train.py \
    --split-strategy single_site \
    --output outputs/single_site \
    --epochs 50
```

### Run All Experiments

```bash
# Windows
run_experiments.bat

# Linux/Mac
chmod +x run_experiments.sh
./run_experiments.sh
```

This runs three experiments:
1. Site split with focal loss (gamma=2.0)
2. Single site with focal loss (gamma=2.0)
3. Site split with cross-entropy (gamma=0.0, baseline)

---

## Command Reference

### Training Script (train.py)

```bash
python train.py [OPTIONS]
```

#### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--split-strategy` | `site_split` | Split strategy: `site_split` or `single_site` |
| `--output` | `outputs/site_split` | Output directory |
| `--epochs` | `30` | Number of training epochs |
| `--focal-gamma` | `2.0` | Focal loss gamma parameter |
| `--learning-rate` | `1e-3` | Learning rate |
| `--batch-size` | `512` | Batch size |
| `--latent-dim` | `64` | Latent dimension for spectrum encoder |
| `--hidden-dim` | `128` | Hidden dimension for GNN |
| `--patience` | `100` | Early stopping patience |
| `--seed` | `42` | Random seed |
| `--visualize` | `True` | Auto-generate visualizations |
| `--no-visualize` | - | Skip visualization |
| `--representation-epochs` | `10` | Epochs for training spectrum autoencoder |
| `--rebuild-representation` | - | Retrain spectrum encoder |
| `--cache-data` | - | Cache extracted spectra |

#### Examples

```bash
# Default training
python train.py --output outputs/exp1

# Custom parameters
python train.py \
    --split-strategy single_site \
    --epochs 100 \
    --focal-gamma 2.0 \
    --learning-rate 5e-4 \
    --batch-size 256 \
    --output outputs/custom

# Cross-entropy baseline
python train.py --focal-gamma 0.0 --output outputs/baseline

# Skip auto-visualization
python train.py --no-visualize --output outputs/no_viz

# Long training with early stopping
python train.py --epochs 200 --patience 30 --output outputs/long
```

### Visualization Script (visualize_results.py)

```bash
python visualize_results.py [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `outputs/site_split` | Directory with results |
| `--split-mode` | `site_split` | Split mode for naming |

---

## Output Files

Training generates the following structure:

```
outputs/experiment_name/
├── Metrics & History
│   ├── training_history.csv           # All metrics, all epochs
│   ├── metrics.json                   # Complete results
│   ├── representation_history.csv     # Autoencoder training
│   ├── metrics_per_drug.csv           # Per-antibiotic performance
│   └── metrics_per_bacteria.csv       # Per-bacteria performance
│
├── Models
│   ├── hybrid_ast_best.pt             # Best model checkpoint
│   ├── spectrum_encoder.pt            # Spectrum autoencoder
│   └── isolate_embeddings.npz         # Cached embeddings
│
├── Predictions
│   ├── validation_DC_predictions.csv  # Validation predictions
│   ├── test_B_predictions.csv         # Test predictions
│   └── splits.csv                     # Sample split assignments
│
└── Visualizations (if --visualize enabled)
    ├── training_curves_site_split.png
    ├── per_class_f1_site_split.png
    ├── best_epoch_metrics_site_split.png      ⭐ Bar labels
    ├── per_class_comparison_site_split.png    ⭐ Bar labels
    └── comprehensive_results_site_split.png
```

### training_history.csv Columns

The CSV contains all metrics for all splits:

```
epoch, train_loss,
train_accuracy, train_balanced_accuracy, train_precision_macro, train_recall_macro,
train_f1_macro, train_f1_weighted, train_f1_S, train_f1_R, train_roc_auc,
validation_accuracy, validation_balanced_accuracy, validation_precision_macro, validation_recall_macro,
validation_f1_macro, validation_f1_weighted, validation_f1_S, validation_f1_R, validation_roc_auc,
test_accuracy, test_balanced_accuracy, test_precision_macro, test_recall_macro,
test_f1_macro, test_f1_weighted, test_f1_S, test_f1_R, test_roc_auc
```

### Reading Results

```python
import pandas as pd
import json

# Load training history
df = pd.read_csv('outputs/exp1/training_history.csv')

# View specific metrics
print(df[['epoch', 'train_f1_macro', 'validation_f1_macro', 'test_f1_macro']])

# Find best epoch
best_idx = df['validation_f1_macro'].idxmax()
best_epoch_data = df.iloc[best_idx]
print(f"Best epoch: {best_epoch_data['epoch']}")
print(f"Val F1: {best_epoch_data['validation_f1_macro']:.4f}")

# Load complete metrics
with open('outputs/exp1/metrics.json', 'r') as f:
    metrics = json.load(f)
print(f"Best epoch: {metrics['best_epoch']}")
print(f"Unmapped antibiotics: {metrics['unmapped_antibiotics']}")
```

---

## Usage Examples

### Example 1: Compare Split Strategies

```bash
# Cross-site evaluation
python train.py \
    --split-strategy site_split \
    --output outputs/cross_site \
    --epochs 50

# Within-site evaluation
python train.py \
    --split-strategy single_site \
    --output outputs/within_site \
    --epochs 50

# Compare results
# Check outputs/cross_site/ and outputs/within_site/ visualizations
```

### Example 2: Optimize Focal Loss Gamma

```bash
# Baseline (cross-entropy)
python train.py --focal-gamma 0.0 --output outputs/gamma0

# Mild focusing
python train.py --focal-gamma 1.0 --output outputs/gamma1

# Recommended
python train.py --focal-gamma 2.0 --output outputs/gamma2

# Strong focusing
python train.py --focal-gamma 5.0 --output outputs/gamma5

# Compare best_epoch_metrics_*.png across experiments
```

### Example 3: Hyperparameter Search

```bash
# Vary learning rate
python train.py --learning-rate 1e-4 --output outputs/lr1e4
python train.py --learning-rate 5e-4 --output outputs/lr5e4
python train.py --learning-rate 1e-3 --output outputs/lr1e3

# Vary latent dimension
python train.py --latent-dim 32 --output outputs/latent32
python train.py --latent-dim 64 --output outputs/latent64
python train.py --latent-dim 128 --output outputs/latent128
```

### Example 4: Quick Test Run

```bash
# Fast training for testing
python train.py \
    --epochs 5 \
    --representation-epochs 2 \
    --output outputs/test_run
```

### Example 5: Production Training

```bash
# Long training with early stopping
python train.py \
    --split-strategy site_split \
    --epochs 200 \
    --patience 30 \
    --focal-gamma 2.0 \
    --learning-rate 1e-3 \
    --batch-size 512 \
    --output outputs/production_model
```

---

## Analysis Tips

### 1. Check Class Balance

Look at **F1-S vs F1-R** in per-class comparison chart:
- If F1-R << F1-S: Increase focal-gamma (focus more on resistant class)
- If both low: Try different learning rate or model architecture
- If validation much lower than train: Reduce overfitting (add dropout, reduce epochs)

### 2. Monitor Overfitting

Compare train vs validation metrics:
- **Train >> Val**: Model is overfitting (reduce complexity or epochs)
- **Train ≈ Val**: Good generalization
- **Train < Val**: Unusual, check data splits

### 3. Evaluate Generalization

Compare split strategies:
- **site_split (cross-site)**: Tests real-world deployment scenario
- **single_site**: Tests model capacity without site bias
- Large difference indicates site-specific effects

### 4. Per-Drug Analysis

Check `metrics_per_drug.csv`:
```python
df = pd.read_csv('outputs/exp1/metrics_per_drug.csv')
# Find drugs with poor performance
poor_drugs = df[df['f1_macro'] < 0.5].sort_values('f1_macro')
print(poor_drugs[['group', 'n', 'f1_macro', 'accuracy']])
```

### 5. Per-Bacteria Analysis

Check `metrics_per_bacteria.csv`:
```python
df = pd.read_csv('outputs/exp1/metrics_per_bacteria.csv')
# Find bacteria with poor performance
poor_bacteria = df[df['f1_macro'] < 0.5].sort_values('f1_macro')
print(poor_bacteria[['group', 'n', 'f1_macro', 'accuracy']])
```

---

## Troubleshooting

### No Visualizations Generated

**Problem:** Visualizations not created after training.

**Solutions:**
```bash
# Manually generate
python visualize_results.py \
    --output-dir outputs/experiment1 \
    --split-mode site_split

# Check for import errors
python -c "import matplotlib, seaborn; print('OK')"

# Install missing packages
pip install matplotlib seaborn
```

### CUDA Out of Memory

**Problem:** Training crashes with CUDA OOM error.

**Solutions:**
```bash
# Reduce batch size
python train.py --batch-size 256 --output outputs/smaller_batch

# Reduce hidden dimension
python train.py --hidden-dim 64 --output outputs/smaller_model

# Use CPU (slower)
# Set device in train.py or use CUDA_VISIBLE_DEVICES=""
```

### Import Errors

**Problem:** ModuleNotFoundError for rdkit, torch, etc.

**Solutions:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Install individually
pip install torch numpy pandas scikit-learn matplotlib seaborn

# For rdkit
conda install -c conda-forge rdkit
# or
pip install rdkit
```

### Empty Splits

**Problem:** ValueError about empty train/val/test splits.

**Solutions:**
- Check that DRIAMS data is in the correct location
- Verify `--driams-root` path is correct
- Ensure sites A, B, C, D exist in data
- For single_site, ensure Site A has enough samples

### Poor Performance

**Problem:** Low F1 scores across all splits.

**Diagnostics:**
1. Check class distribution (should be somewhat balanced)
2. Verify data loading (check sample counts in console output)
3. Try different focal-gamma values
4. Increase training epochs
5. Check per-drug and per-bacteria metrics for specific issues

**Solutions:**
```bash
# Increase focusing on hard examples
python train.py --focal-gamma 5.0

# Longer training
python train.py --epochs 100 --patience 50

# Different learning rate
python train.py --learning-rate 5e-4

# Larger model
python train.py --hidden-dim 256 --latent-dim 128
```

### Slow Training

**Problem:** Training takes too long.

**Solutions:**
```bash
# Reduce epochs
python train.py --epochs 20

# Increase batch size (if GPU memory allows)
python train.py --batch-size 1024

# Use cached embeddings
# Run once to create cache, subsequent runs will be faster
python train.py --cache-data

# Skip representation retraining
# After first run, representations are cached automatically
```

---

## Best Practices

### 1. Initial Setup
```bash
# Quick test to verify everything works
python train.py --epochs 5 --output outputs/test

# Full training
python train.py --epochs 50 --output outputs/full
```

### 2. Recommended Workflow

1. **Baseline**: Train with cross-entropy (gamma=0)
```bash
python train.py --focal-gamma 0.0 --output outputs/baseline
```

2. **Focal Loss**: Train with focal loss (gamma=2)
```bash
python train.py --focal-gamma 2.0 --output outputs/focal
```

3. **Compare**: Check if focal loss improves F1-R and balanced accuracy

4. **Optimize**: Try different gamma values if needed

5. **Evaluate**: Use site_split for final model evaluation

### 3. Model Selection

- Use **validation F1-macro** for model selection (best epoch)
- Report **test set** metrics for final evaluation
- Check **per-class F1** to ensure both classes perform well
- Examine **confusion matrices** for error patterns

### 4. Documentation

Document your experiments:
```python
# Create experiment log
import json
experiment = {
    "name": "experiment1",
    "date": "2024-01-15",
    "config": {
        "split_strategy": "site_split",
        "focal_gamma": 2.0,
        "learning_rate": 1e-3,
        "epochs": 50
    },
    "results": {
        "best_epoch": 23,
        "val_f1": 0.7423,
        "test_f1": 0.7323
    },
    "notes": "Baseline focal loss model"
}

with open('outputs/experiment1/experiment_log.json', 'w') as f:
    json.dump(experiment, f, indent=2)
```

---

## Summary

This project provides a complete system for training and evaluating antibiotic resistance prediction models:

✅ **Focal Loss** - Handles class imbalance effectively  
✅ **Two Split Strategies** - Cross-site and within-site evaluation  
✅ **Comprehensive Metrics** - All metrics for all splits every epoch  
✅ **Automatic Visualization** - 5 charts with value labels on bars  
✅ **Per-Class Tracking** - Monitor Susceptible vs Resistant separately  
✅ **Easy to Use** - Single command to train and visualize  
✅ **Flexible** - Many configurable parameters  
✅ **Well-Documented** - Complete guide in one file  

### Quick Commands

```bash
# Recommended: Site split with focal loss
python train.py --split-strategy site_split --output outputs/recommended

# Within-site evaluation
python train.py --split-strategy single_site --output outputs/single_site

# Run all experiments
run_experiments.bat  # Windows
./run_experiments.sh  # Linux/Mac
```

### Key Files

- **train.py** - Main training script
- **visualize_results.py** - Visualization generation
- **run_experiments.bat/sh** - Batch experiments
- **README_COMPLETE.md** - This file

---

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

## License

[Your license here]

---

**Happy training! 🚀**

For questions or issues, please check the troubleshooting section or examine the console output for error messages.
