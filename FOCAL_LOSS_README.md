# Focal Loss Implementation

## Overview

The classification model now uses **Focal Loss** instead of standard Cross-Entropy Loss. Focal Loss was introduced in the paper ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002) by Lin et al. and is particularly effective for handling class imbalance by focusing training on hard examples.

## What is Focal Loss?

Focal Loss modifies the standard cross-entropy loss by adding a modulating factor that down-weights easy examples and focuses on hard misclassified examples.

### Formula

```
FL(pt) = -α(1 - pt)^γ * log(pt)
```

Where:
- **pt**: Predicted probability of the true class
- **α (alpha)**: Class weighting factor (handles class imbalance)
- **γ (gamma)**: Focusing parameter (controls the rate at which easy examples are down-weighted)

### Key Parameters

1. **Gamma (γ)**: Default = 2.0
   - When γ = 0, focal loss is equivalent to cross-entropy loss
   - Higher γ values (e.g., 2, 5) focus more on hard examples
   - Typical range: 0.5 to 5.0

2. **Alpha (α)**: Automatically computed from class distribution
   - Balances the importance of different classes
   - Calculated as: `α_i = total_samples / (num_classes × class_i_count)`

## Benefits for Antibiotic Resistance Prediction

1. **Handles Class Imbalance**: Addresses imbalance between resistant (R) and susceptible (S) samples
2. **Focuses on Hard Examples**: Prioritizes learning from difficult-to-classify cases
3. **Improves Minority Class Performance**: Better at detecting rare resistance patterns
4. **Reduces False Negatives**: Particularly important in clinical settings

## Usage

### Command Line Arguments

```bash
python train.py --focal-gamma 2.0
```

**Options:**
- `--focal-gamma`: Focusing parameter (default: 2.0)
  - Use 0.0 for standard cross-entropy behavior
  - Use 1.0-2.0 for moderate focusing on hard examples (recommended)
  - Use 2.0-5.0 for strong focusing on hard examples

### Examples

**Standard training with focal loss (default):**
```bash
python train.py --output outputs/focal_gamma2
```

**With higher gamma for more focus on hard examples:**
```bash
python train.py --focal-gamma 5.0 --output outputs/focal_gamma5
```

**Equivalent to cross-entropy (for comparison):**
```bash
python train.py --focal-gamma 0.0 --output outputs/cross_entropy
```

## Implementation Details

### Location
- **File**: `train.py`
- **Function**: `focal_loss()` (lines 23-48)
- **Usage**: Training loop (line 304)

### Code Changes

1. Added `focal_loss()` function that:
   - Computes standard cross-entropy with class weights
   - Calculates predicted probability of true class
   - Applies focal weight: `(1 - pt)^gamma`
   - Returns weighted loss

2. Replaced `F.cross_entropy()` with `focal_loss()` in training loop

3. Added `--focal-gamma` command-line argument for easy experimentation

## Testing

A test script is provided to verify the focal loss implementation:

```bash
python test_focal_loss.py
```

This test demonstrates:
- Basic focal loss computation
- Effect of different gamma values
- Class weighting behavior
- Hard vs easy example handling

## Expected Results

With focal loss, you should observe:
- **Better balanced accuracy** across resistant/susceptible classes
- **Improved F1 scores** especially for minority classes
- **Reduced overfitting** on easy examples
- **Better generalization** to test sets

## Comparison with Cross-Entropy

| Metric | Cross-Entropy | Focal Loss (γ=2.0) |
|--------|---------------|-------------------|
| Easy examples weight | 1.0 | ~0.01 |
| Hard examples weight | 1.0 | ~1.0 |
| Class imbalance handling | Basic (via weights) | Advanced (via weights + focusing) |
| Hard example focus | No | Yes |

## References

1. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. ICCV.
2. https://arxiv.org/abs/1708.02002

## Troubleshooting

**If training loss seems too low:**
- Try reducing gamma (e.g., `--focal-gamma 1.0`)

**If validation performance is poor:**
- Try increasing gamma (e.g., `--focal-gamma 3.0`)
- Check if class imbalance is severe

**For comparison purposes:**
- Run with `--focal-gamma 0.0` to get cross-entropy baseline
