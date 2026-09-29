"""Create summary and comparison plots for the DRIAMS hybrid model metrics."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_LABELS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("f1_macro", "Macro F1"),
    ("f1_weighted", "Weighted F1"),
    ("roc_auc", "ROC AUC"),
    ("average_precision", "Average precision"),
    ("log_loss", "Log loss"),
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ordered_split_names(metrics: dict) -> list[str]:
    preferred = ["validation_DC", "test_B", "DRIAMS-D", "DRIAMS-C"]
    seen = set()
    ordered = []
    for name in preferred:
        if name in metrics and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in metrics:
        if name not in seen:
            ordered.append(name)
    return ordered


def plot_summary_metrics(metrics: dict, output_dir: Path):
    split_names = ordered_split_names(metrics.get("metrics", {}))
    rows = []
    for split_name in split_names:
        row = metrics["metrics"].get(split_name)
        if row is None:
            continue
        rows.append((split_name, row))

    if not rows:
        raise ValueError("No metric data found in metrics.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    fig.suptitle("Model performance by split", fontsize=14, fontweight="bold")
    axes_flat = axes.ravel()

    for ax, (key, label) in zip(axes_flat, METRIC_LABELS):
        values = []
        labels = []
        for split_name, row in rows:
            value = safe_float(row.get(key))
            if value is not None:
                values.append(value)
                labels.append(split_name)
        if not values:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(labels)]
        ax.bar(labels, values, color=colors, edgecolor="black")
        ax.set_title(label)
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

        if key == "log_loss":
            ymax = max(1.0, max(values) * 1.15)
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(0, 1.0)

    fig.savefig(output_dir / "summary_metric_comparison.png", dpi=200)
    plt.close(fig)


def plot_training_history(history_path: Path, output_dir: Path):
    rows = []
    with history_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No training history in {history_path}")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_f1 = [float(row["validation_macro_f1"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.suptitle("Training and validation curves", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(epochs, val_f1, marker="o", color="#2ca02c", linewidth=2)
    axes[1].set_title("Validation macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.savefig(output_dir / "training_history.png", dpi=200)
    plt.close(fig)


def plot_group_bar(csv_path: Path, metric_name: str, title: str, output_name: str, output_dir: Path, top_n: int = 10):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    split_to_rows = {}
    for row in rows:
        split = row["split"]
        split_to_rows.setdefault(split, []).append(row)

    fig, axes = plt.subplots(len(split_to_rows), 1, figsize=(12, 4 * max(1, len(split_to_rows))), squeeze=False)
    axes = axes.ravel()
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax_index, (split_name, split_rows) in enumerate(split_to_rows.items()):
        parsed = []
        for row in split_rows:
            value = safe_float(row.get(metric_name))
            if value is None:
                continue
            parsed.append((row["group"], value, int(row["n"])))
        parsed.sort(key=lambda item: item[1], reverse=True)
        parsed = parsed[:top_n]

        labels = [item[0] for item in parsed]
        values = [item[1] for item in parsed]
        ax = axes[ax_index]
        ax.bar(labels, values, color="#6c8ebf", edgecolor="black")
        ax.set_title(f"{split_name} — top {len(values)} groups")
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    fig.savefig(output_dir / output_name, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(metrics: dict, output_dir: Path):
    split_names = ["validation_DC", "test_B"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Confusion matrices for the final evaluation splits", fontsize=13, fontweight="bold")

    for ax, split_name in zip(axes, split_names):
        row = metrics.get("metrics", {}).get(split_name)
        if row is None:
            ax.text(0.5, 0.5, f"Missing {split_name}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        matrix = np.asarray(row["confusion_matrix"], dtype=float)
        label_names = ["S", "R"]
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(split_name)
        ax.set_xticks([0, 1], labels=label_names)
        ax.set_yticks([0, 1], labels=label_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_dir / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, default=Path("outputs/site_split/metrics.json"))
    parser.add_argument("--drug-csv", type=Path, default=Path("outputs/site_split/metrics_per_drug.csv"))
    parser.add_argument("--bacteria-csv", type=Path, default=Path("outputs/site_split/metrics_per_bacteria.csv"))
    parser.add_argument("--training-history", type=Path, default=Path("outputs/site_split/training_history.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/site_split/plots"))
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    metrics = load_json(args.metrics_json)
    if "metrics" not in metrics:
        raise ValueError(f"No 'metrics' section found in {args.metrics_json}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_summary_metrics(metrics, args.output_dir)
    plot_training_history(args.training_history, args.output_dir)
    plot_group_bar(args.drug_csv, "f1_macro", "Top antibiotic groups by macro F1", "top_drugs_by_f1.png", args.output_dir, top_n=args.top_n)
    plot_group_bar(args.bacteria_csv, "f1_macro", "Top bacterial species by macro F1", "top_bacteria_by_f1.png", args.output_dir, top_n=args.top_n)
    plot_confusion_matrices(metrics, args.output_dir)

    print(f"Saved plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
