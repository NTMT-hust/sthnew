"""Automatic visualization of training results with curves and bar charts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_training_history(path: Path):
    """Load training history from CSV."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        history = list(reader)
    
    # Convert numeric fields
    for entry in history:
        for key, value in entry.items():
            try:
                entry[key] = float(value)
            except (ValueError, TypeError):
                pass
    
    return history


def plot_training_curves(history, output_dir: Path, split_mode: str):
    """Plot training loss and F1 score curves."""
    if not history:
        print("No training history to plot")
        return
    
    epochs = [entry['epoch'] for entry in history]
    
    # Determine which metrics to plot based on available data
    metrics_to_plot = []
    
    # Always have loss
    if 'train_loss' in history[0]:
        metrics_to_plot.append(('train_loss', 'Training Loss', 'Loss'))
    
    # F1 scores
    f1_metrics = [
        ('train_f1_macro', 'validation_f1_macro', 'test_f1_macro', 'F1-Macro Score'),
        ('train_accuracy', 'validation_accuracy', 'test_accuracy', 'Accuracy'),
        ('train_balanced_accuracy', 'validation_balanced_accuracy', 'test_balanced_accuracy', 'Balanced Accuracy'),
        ('train_roc_auc', 'validation_roc_auc', 'test_roc_auc', 'ROC-AUC'),
    ]
    
    # Create figure with subplots
    n_plots = 1 + len([m for m in f1_metrics if m[0] in history[0]])
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot loss
    if 'train_loss' in history[0]:
        ax = axes[plot_idx]
        train_loss = [entry.get('train_loss', 0) for entry in history]
        ax.plot(epochs, train_loss, marker='o', linewidth=2, markersize=4, label='Training Loss')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_idx += 1
    
    # Plot F1 and other metrics
    for train_key, val_key, test_key, title in f1_metrics:
        if train_key in history[0]:
            ax = axes[plot_idx]
            
            train_values = [entry.get(train_key, 0) for entry in history]
            val_values = [entry.get(val_key, 0) for entry in history]
            test_values = [entry.get(test_key, 0) for entry in history]
            
            ax.plot(epochs, train_values, marker='o', linewidth=2, markersize=4, label='Train')
            ax.plot(epochs, val_values, marker='s', linewidth=2, markersize=4, label='Validation')
            ax.plot(epochs, test_values, marker='^', linewidth=2, markersize=4, label='Test')
            
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel(title, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim([0, 1.05])
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_{split_mode}.png', bbox_inches='tight')
    print(f"Saved training curves to {output_dir / f'training_curves_{split_mode}.png'}")
    plt.close()


def plot_per_class_f1(history, output_dir: Path, split_mode: str, class_names=['S', 'R']):
    """Plot per-class F1 scores over epochs."""
    if not history:
        return
    
    epochs = [entry['epoch'] for entry in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax_idx, (split_name, ax) in enumerate(zip(['train', 'validation', 'test'], axes)):
        for class_name in class_names:
            key = f'{split_name}_f1_{class_name}'
            if key in history[0]:
                values = [entry.get(key, 0) for entry in history]
                ax.plot(epochs, values, marker='o', linewidth=2, markersize=4, 
                       label=f'F1-{class_name}')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{split_name.capitalize()} - Per-Class F1', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / f'per_class_f1_{split_mode}.png', bbox_inches='tight')
    print(f"Saved per-class F1 curves to {output_dir / f'per_class_f1_{split_mode}.png'}")
    plt.close()


def plot_best_epoch_metrics(history, best_epoch: int, output_dir: Path, split_mode: str):
    """Plot bar chart of metrics at best epoch with values on top."""
    if not history or best_epoch > len(history):
        return
    
    best_metrics = history[best_epoch - 1]
    
    # Define metrics to plot
    metric_keys = [
        ('accuracy', 'Accuracy'),
        ('balanced_accuracy', 'Balanced Acc'),
        ('precision_macro', 'Precision'),
        ('recall_macro', 'Recall'),
        ('f1_macro', 'F1-Macro'),
        ('roc_auc', 'ROC-AUC'),
    ]
    
    # Prepare data
    splits = ['train', 'validation', 'test']
    split_labels = ['Train', 'Validation', 'Test']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for ax_idx, (metric_key, metric_name) in enumerate(metric_keys):
        ax = axes[ax_idx]
        
        values = []
        for split in splits:
            key = f'{split}_{metric_key}'
            values.append(best_metrics.get(key, 0))
        
        # Create bar chart
        x = np.arange(len(splits))
        bars = ax.bar(x, values, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8, edgecolor='black')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} (Epoch {best_epoch})', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Best Epoch Metrics Comparison (Epoch {best_epoch})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'best_epoch_metrics_{split_mode}.png', bbox_inches='tight')
    print(f"Saved best epoch metrics to {output_dir / f'best_epoch_metrics_{split_mode}.png'}")
    plt.close()


def plot_per_class_comparison(history, best_epoch: int, output_dir: Path, split_mode: str, 
                               class_names=['S', 'R']):
    """Plot per-class F1 scores comparison at best epoch."""
    if not history or best_epoch > len(history):
        return
    
    best_metrics = history[best_epoch - 1]
    
    splits = ['train', 'validation', 'test']
    split_labels = ['Train', 'Validation', 'Test']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(splits))
    width = 0.35
    
    for i, class_name in enumerate(class_names):
        values = []
        for split in splits:
            key = f'{split}_f1_{class_name}'
            values.append(best_metrics.get(key, 0))
        
        offset = (i - len(class_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=f'F1-{class_name}', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class F1 Scores (Epoch {best_epoch})', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'per_class_comparison_{split_mode}.png', bbox_inches='tight')
    print(f"Saved per-class comparison to {output_dir / f'per_class_comparison_{split_mode}.png'}")
    plt.close()


def plot_final_comparison(history, best_epoch: int, output_dir: Path, split_mode: str):
    """Create a comprehensive comparison plot."""
    if not history or best_epoch > len(history):
        return
    
    best_metrics = history[best_epoch - 1]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    splits = ['train', 'validation', 'test']
    split_labels = ['Train', 'Val', 'Test']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Plot 1: Main metrics
    ax1 = fig.add_subplot(gs[0, :])
    metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc']
    metric_labels = ['Accuracy', 'Balanced Acc', 'F1-Macro', 'ROC-AUC']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (split, color) in enumerate(zip(splits, colors)):
        values = [best_metrics.get(f'{split}_{m}', 0) for m in metrics]
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, values, width, label=split_labels[i], color=color, 
                      alpha=0.8, edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Overall Metrics Comparison (Best Epoch: {best_epoch})', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2-4: Per-split detailed metrics
    for idx, (split, color) in enumerate(zip(splits, colors)):
        ax = fig.add_subplot(gs[1 + idx // 2, idx % 2])
        
        detailed_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_S', 'f1_R']
        metric_labels_detailed = ['Acc', 'Prec', 'Rec', 'F1-M', 'F1-S', 'F1-R']
        
        values = [best_metrics.get(f'{split}_{m}', 0) for m in detailed_metrics]
        
        bars = ax.bar(range(len(values)), values, color=color, alpha=0.8, edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{split_labels[idx]} Set Metrics', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(metric_labels_detailed, fontsize=9, rotation=45)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Training curves summary
    ax5 = fig.add_subplot(gs[2, 2])
    epochs = [entry['epoch'] for entry in history]
    val_f1 = [entry.get('validation_f1_macro', 0) for entry in history]
    
    ax5.plot(epochs, val_f1, marker='o', linewidth=2, markersize=3, color='#A23B72')
    ax5.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, label=f'Best: {best_epoch}')
    ax5.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Val F1-Macro', fontsize=10, fontweight='bold')
    ax5.set_title('Validation F1 Progress', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'Comprehensive Results Overview - {split_mode.upper()}', 
                 fontsize=15, fontweight='bold')
    plt.savefig(output_dir / f'comprehensive_results_{split_mode}.png', bbox_inches='tight')
    print(f"Saved comprehensive results to {output_dir / f'comprehensive_results_{split_mode}.png'}")
    plt.close()


def visualize_results(output_dir: Path, split_mode: str = 'site_split'):
    """Main visualization function."""
    print(f"\n{'='*70}")
    print(f"Generating Visualizations for: {output_dir}")
    print(f"Split Mode: {split_mode}")
    print(f"{'='*70}\n")
    
    # Load history
    history_path = output_dir / "training_history.csv"
    if not history_path.exists():
        print(f"Error: {history_path} not found!")
        return
    
    history = load_training_history(history_path)
    
    # Load metrics to get best epoch
    metrics_path = output_dir / "metrics.json"
    best_epoch = 1
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
            best_epoch = metrics.get("best_epoch", 1)
    
    print(f"Loaded {len(history)} epochs of training history")
    print(f"Best epoch: {best_epoch}\n")
    
    # Generate plots
    plot_training_curves(history, output_dir, split_mode)
    plot_per_class_f1(history, output_dir, split_mode)
    plot_best_epoch_metrics(history, best_epoch, output_dir, split_mode)
    plot_per_class_comparison(history, best_epoch, output_dir, split_mode)
    plot_final_comparison(history, best_epoch, output_dir, split_mode)
    
    print(f"\n{'='*70}")
    print("✓ All visualizations generated successfully!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/site_split"),
                       help="Directory containing training results")
    parser.add_argument("--split-mode", type=str, choices=['site_split', 'single_site'],
                       default='site_split',
                       help="Split mode: 'site_split' (train A, val DC, test B) or 'single_site' (all on site A)")
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"Error: Directory {args.output_dir} does not exist!")
        return
    
    visualize_results(args.output_dir, args.split_mode)


if __name__ == "__main__":
    main()
