"""Train the DRIAMS hybrid model with A / DC / B site-level splits."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score, confusion_matrix,
                             f1_score, hamming_loss, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem


def focal_loss(logits, targets, alpha=None, gamma=2.0, reduction='mean'):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        logits: Raw model outputs (before softmax), shape [N, num_classes]
        targets: Ground truth labels, shape [N]
        alpha: Weighting factor for each class, shape [num_classes] or None
        gamma: Focusing parameter (default: 2.0). Higher gamma focuses more on hard examples
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Computed focal loss
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=alpha)
    probs = F.softmax(logits, dim=1)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # Probability of true class
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * ce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

from data.driams import load_ast_edges, load_site_spectra
from data.splits import TEST_SITES, TRAIN_SITES, VALIDATION_SITES, split_sites
from models.hybrid_ast import HeterogeneousASTModel, smiles_graph
from models.spectral import SpectralAutoencoder


DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent / "DRIAMS"
CLASS_NAMES = ["S", "R"]


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


ALIASES = {
    "cotrimoxazole": "trimethoprim sulfamethoxazole",
    "cotrimoxazol": "trimethoprim sulfamethoxazole",
    "sulfamethoxazole trimethoprim": "trimethoprim sulfamethoxazole",
    "piperacillin tazobactam": "piperacillin tazobactam",
    "cefuroxime": "cefuroxime sodium",
    "ampicillin amoxicillin": "ampicillin",
    "gentamicin high level": "gentamicin",
}


def drug_structures(path: Path):
    graphs, names, name_to_idx = [], [], {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name, smiles = row.get("antibiotic", "").strip(), row.get("smiles", "").strip()
            if not name or not smiles:
                continue
            try:
                graph = smiles_graph(smiles, Chem)
            except ValueError:
                continue
            idx = len(names)
            names.append(name)
            graphs.append(graph)
            name_to_idx[normalized_name(name)] = idx
    for alias, target in ALIASES.items():
        idx = name_to_idx.get(normalized_name(target))
        if idx is not None:
            name_to_idx[normalized_name(alias)] = idx
    return names, graphs, name_to_idx


def classification_metrics(y, prediction, probabilities):
    class_ids = list(range(len(CLASS_NAMES)))
    result = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision_macro": float(precision_score(y, prediction, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y, prediction, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y, prediction, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y, prediction, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "f1_micro": float(f1_score(y, prediction, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y, prediction)),
        "cohen_kappa": float(cohen_kappa_score(y, prediction)),
        "matthews_corrcoef": float(matthews_corrcoef(y, prediction)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=class_ids).tolist(),
        "classification_report": classification_report(y, prediction, labels=class_ids,
                                                       target_names=CLASS_NAMES, output_dict=True,
                                                       zero_division=0),
    }
    try:
        result["log_loss"] = float(log_loss(y, probabilities, labels=class_ids))
    except ValueError:
        result["log_loss"] = None
    try:
        result["roc_auc"] = float(roc_auc_score(y, probabilities[:, 1]))
    except ValueError:
        result["roc_auc"] = None
    try:
        result["average_precision"] = float(average_precision_score(y, probabilities[:, 1]))
    except ValueError:
        result["average_precision"] = None
    return result


def grouped_metrics(y, probabilities, groups):
    """Return complete classification metrics for every named subgroup."""
    result = {}
    prediction = probabilities.argmax(axis=1)
    for group in sorted(set(str(value) if str(value).strip() else "unknown" for value in groups)):
        mask = np.asarray([(str(value) if str(value).strip() else "unknown") == group
                           for value in groups], dtype=bool)
        result[group] = classification_metrics(y[mask], prediction[mask], probabilities[mask])
    return result


def write_group_metrics_csv(path: Path, grouped: dict):
    fields = ["split", "group", "n", "accuracy", "balanced_accuracy", "precision_macro",
              "precision_weighted", "recall_macro", "recall_weighted", "f1_macro", "f1_weighted",
              "f1_micro", "hamming_loss", "cohen_kappa", "matthews_corrcoef", "log_loss",
              "roc_auc", "average_precision", "confusion_matrix", "classification_report"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, results in grouped.items():
            for group, metric in results.items():
                row = {key: value for key, value in metric.items() if key in fields}
                row.update({"split": split, "group": group,
                            "confusion_matrix": json.dumps(metric["confusion_matrix"]),
                            "classification_report": json.dumps(metric["classification_report"])})
                writer.writerow(row)


def encode_spectra(model, x, device, batch_size=2048):
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            output.append(model.latent(batch).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driams-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--smiles-csv", type=Path, default=Path(__file__).with_name("antibiotic_smiles.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/site_split"))
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--representation-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter (default: 2.0). Higher values focus more on hard examples.")
    parser.add_argument("--split-strategy", type=str, choices=['site_split', 'single_site'], 
                        default='site_split',
                        help="Split strategy: 'site_split' (train A, val DC, test B) or 'single_site' (train/val/test all from site A)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Automatically generate visualizations after training (default: True)")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false",
                        help="Skip automatic visualization generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-data", action="store_true", help="Cache extracted 301-bin spectra in the output folder.")
    parser.add_argument("--rebuild-representation", action="store_true",
                        help="Ignore saved isolate embeddings and retrain the spectrum autoencoder.")
    args = parser.parse_args()
    if args.epochs < 1 or args.representation_epochs < 1:
        parser.error("--epochs and --representation-epochs must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    required_sites = set((*TRAIN_SITES, *VALIDATION_SITES, *TEST_SITES))
    embedding_path = args.output / "isolate_embeddings.npz"
    encoder_path = args.output / "spectrum_encoder.pt"
    spectral, embeddings, ae_history = None, None, []
    representation_reused = False
    if not args.rebuild_representation and embedding_path.exists() and encoder_path.exists():
        saved = np.load(embedding_path, allow_pickle=False)
        if {"embeddings", "sample_ids", "species", "sites", "years"}.issubset(saved.files):
            cached_sites = set(saved["sites"].astype(str))
            cached_dim = int(saved["embeddings"].shape[1])
            if required_sites.issubset(cached_sites) and cached_dim == args.latent_dim:
                spectral = {key: saved[key] for key in ("sample_ids", "species", "sites", "years")}
                embeddings = saved["embeddings"].astype(np.float32)
                representation_reused = True
                print(f"Reusing {len(embeddings):,} saved isolate embeddings; raw spectra will not be reread.",
                      flush=True)
    if not representation_reused:
        cache_path = args.output / "site_spectra_301.npz"
        if args.cache_data and cache_path.exists():
            cache = np.load(cache_path, allow_pickle=False)
            spectral = {key: cache[key] for key in cache.files}
        else:
            spectral = load_site_spectra(args.driams_root, [*TRAIN_SITES, *VALIDATION_SITES, *TEST_SITES])
            if args.cache_data:
                np.savez_compressed(cache_path, **spectral)
    sample_index = {(str(site), int(year), str(code)): i for i, (site, year, code) in enumerate(
        zip(spectral["sites"], spectral["years"], spectral["sample_ids"]))}
    train_iso, val_iso, test_iso = split_sites(spectral["sites"])

    # If no compatible embeddings exist, fit the spectrum encoder on A only.
    if not representation_reused:
        logged = np.log1p(np.maximum(spectral["spectra"], 0))
        train_raw = logged[train_iso]
        mean = train_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = train_raw.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        normalized = np.clip((logged - mean) / scale, -10, 10).astype(np.float32)

        train_loader = DataLoader(TensorDataset(torch.from_numpy(normalized[train_iso])),
                                  batch_size=args.batch_size, shuffle=True)
        encoder = SpectralAutoencoder(301, args.latent_dim).to(device)
        ae_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        reconstruction = nn.MSELoss()
        for epoch in range(args.representation_epochs):
            encoder.train()
            cumulative, seen = 0.0, 0
            for (clean,) in train_loader:
                clean = clean.to(device)
                noisy = clean + 0.05 * torch.randn_like(clean)
                ae_optimizer.zero_grad(set_to_none=True)
                recon, mu, logvar = encoder(noisy)
                recon_loss = reconstruction(recon, clean)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 1e-3 * kl_loss
                loss.backward()
                ae_optimizer.step()
                cumulative += recon_loss.item() * clean.size(0)
                seen += clean.size(0)
            train_mse = cumulative / seen
            ae_history.append({"epoch": epoch + 1, "train_reconstruction_mse": train_mse})
            print(f"Representation epoch {epoch + 1:02d}/{args.representation_epochs} - train MSE {train_mse:.6f}")

        embeddings = encode_spectra(encoder, normalized, device)
        torch.save({"state_dict": encoder.cpu().state_dict(), "input_dim": 301,
                    "latent_dim": args.latent_dim, "mean": mean, "scale": scale}, encoder_path)
        np.savez_compressed(embedding_path, embeddings=embeddings,
                            sample_ids=spectral["sample_ids"], species=spectral["species"],
                            sites=spectral["sites"], years=spectral["years"])

    names, drug_graphs, drug_index = drug_structures(args.smiles_csv)
    ast = load_ast_edges(args.driams_root, sample_index, drug_index,
                         [*TRAIN_SITES, *VALIDATION_SITES, *TEST_SITES])
    
    # Apply split strategy
    print(f"\nUsing split strategy: {args.split_strategy}")
    if args.split_strategy == 'site_split':
        # Original: train on A, validate on DC, test on B
        print("  Train: Site A | Validation: Sites D+C | Test: Site B")
        edge_train, edge_val, edge_test = split_sites(ast["site"])
    else:  # single_site
        # All splits from site A only
        print("  Train/Val/Test: All from Site A")
        site_a_indices = [i for i, site in enumerate(ast["site"]) if site == 'A']
        rng_split = np.random.default_rng(args.seed)
        rng_split.shuffle(site_a_indices)
        
        # 70% train, 15% val, 15% test
        n = len(site_a_indices)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        
        edge_train = site_a_indices[:n_train]
        edge_val = site_a_indices[n_train:n_train + n_val]
        edge_test = site_a_indices[n_train + n_val:]
        
        print(f"  Site A split: {len(edge_train)} train, {len(edge_val)} val, {len(edge_test)} test")
    
    if not edge_train or not edge_val or not edge_test:
        raise ValueError(f"Empty requested split: train={len(edge_train)}, val={len(edge_val)}, test={len(edge_test)}")

    x = torch.from_numpy(embeddings).to(device)
    edge_iso = torch.as_tensor(ast["isolate"], dtype=torch.long, device=device)
    edge_drug = torch.as_tensor(ast["drug"], dtype=torch.long, device=device)
    edge_label = torch.as_tensor(ast["label"], dtype=torch.long, device=device)
    train_idx = np.asarray(edge_train, dtype=np.int64)
    val_idx = np.asarray(edge_val, dtype=np.int64)
    test_idx = np.asarray(edge_test, dtype=np.int64)
    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    global_to_train = np.full(len(ast["label"]), -1, dtype=np.int64)
    global_to_train[train_idx] = np.arange(len(train_idx))

    model = HeterogeneousASTModel(args.latent_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    class_counts = torch.bincount(edge_label[train_t], minlength=2).float().clamp_min(1)
    class_weights = class_counts.sum() / (2 * class_counts)
    rng = np.random.default_rng(args.seed)
    best_val, best_epoch, stale = -1.0, 0, 0
    history = []
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_idx)
        loss_total = 0.0
        for start in range(0, len(order), args.batch_size):
            batch_edges = order[start:start + args.batch_size]
            batch_global = torch.as_tensor(batch_edges, dtype=torch.long, device=device)
            batch_local = torch.as_tensor(global_to_train[batch_edges], dtype=torch.long, device=device)
            # Batch targets are removed from message passing to prevent direct label leakage.
            iso_h, drug_h = model.encode_nodes(x, drug_graphs, edge_iso[train_t], edge_drug[train_t],
                                               edge_label[train_t], batch_local, device)
            logits = model.classify_edges(iso_h, drug_h, edge_iso[batch_global], edge_drug[batch_global])
            loss = focal_loss(logits, edge_label[batch_global], alpha=class_weights, gamma=args.focal_gamma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(batch_edges)

        model.eval()
        with torch.no_grad():
            iso_h, drug_h = model.encode_nodes(x, drug_graphs, edge_iso[train_t], edge_drug[train_t],
                                               edge_label[train_t], torch.empty(0, dtype=torch.long, device=device), device)
            
            # Compute metrics for all splits
            epoch_metrics = {"epoch": epoch + 1, "train_loss": loss_total / len(train_idx)}
            
            for split_name, split_idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
                split_logits = model.classify_edges(iso_h, drug_h,
                                                    edge_iso[torch.as_tensor(split_idx, device=device)],
                                                    edge_drug[torch.as_tensor(split_idx, device=device)])
                split_probs = torch.softmax(split_logits, dim=1).cpu().numpy()
                split_pred = split_probs.argmax(1)
                split_labels = ast["label"][split_idx]
                
                # Compute comprehensive metrics
                try:
                    acc = accuracy_score(split_labels, split_pred)
                    bal_acc = balanced_accuracy_score(split_labels, split_pred)
                    prec_macro = precision_score(split_labels, split_pred, average="macro", zero_division=0)
                    rec_macro = recall_score(split_labels, split_pred, average="macro", zero_division=0)
                    f1_macro = f1_score(split_labels, split_pred, average="macro", zero_division=0)
                    f1_weighted = f1_score(split_labels, split_pred, average="weighted", zero_division=0)
                    
                    # Per-class metrics
                    f1_per_class = f1_score(split_labels, split_pred, average=None, zero_division=0)
                    
                    # Store metrics
                    epoch_metrics[f"{split_name}_accuracy"] = float(acc)
                    epoch_metrics[f"{split_name}_balanced_accuracy"] = float(bal_acc)
                    epoch_metrics[f"{split_name}_precision_macro"] = float(prec_macro)
                    epoch_metrics[f"{split_name}_recall_macro"] = float(rec_macro)
                    epoch_metrics[f"{split_name}_f1_macro"] = float(f1_macro)
                    epoch_metrics[f"{split_name}_f1_weighted"] = float(f1_weighted)
                    
                    # Per-class F1 scores
                    for i, class_name in enumerate(CLASS_NAMES):
                        epoch_metrics[f"{split_name}_f1_{class_name}"] = float(f1_per_class[i])
                    
                    # ROC AUC if binary
                    try:
                        if len(split_probs[0]) == 2:
                            roc = roc_auc_score(split_labels, split_probs[:, 1])
                            epoch_metrics[f"{split_name}_roc_auc"] = float(roc)
                    except ValueError:
                        pass
                    
                except Exception as e:
                    print(f"Warning: Could not compute some metrics for {split_name}: {e}")
            
            history.append(epoch_metrics)
            val_macro = epoch_metrics["validation_f1_macro"]
        
        # Print epoch summary
        print(f"Epoch {epoch + 1:02d}/{args.epochs} | Loss: {epoch_metrics['train_loss']:.5f}")
        print(f"  Train    - Acc: {epoch_metrics.get('train_accuracy', 0):.4f}, F1-macro: {epoch_metrics.get('train_f1_macro', 0):.4f}, "
              f"F1-S: {epoch_metrics.get('train_f1_S', 0):.4f}, F1-R: {epoch_metrics.get('train_f1_R', 0):.4f}")
        print(f"  Val      - Acc: {epoch_metrics.get('validation_accuracy', 0):.4f}, F1-macro: {epoch_metrics.get('validation_f1_macro', 0):.4f}, "
              f"F1-S: {epoch_metrics.get('validation_f1_S', 0):.4f}, F1-R: {epoch_metrics.get('validation_f1_R', 0):.4f}")
        print(f"  Test     - Acc: {epoch_metrics.get('test_accuracy', 0):.4f}, F1-macro: {epoch_metrics.get('test_f1_macro', 0):.4f}, "
              f"F1-S: {epoch_metrics.get('test_f1_S', 0):.4f}, F1-R: {epoch_metrics.get('test_f1_R', 0):.4f}")
        
        if val_macro > best_val:
            best_val, best_epoch, stale = float(val_macro), epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}; best validation epoch was {best_epoch}.")
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        iso_h, drug_h = model.encode_nodes(x, drug_graphs, edge_iso[train_t], edge_drug[train_t],
                                           edge_label[train_t], torch.empty(0, dtype=torch.long, device=device), device)
        site_metrics, predictions, probabilities_by_split = {}, {}, {}
        for split_name, indices in (("validation_DC", val_idx), ("test_B", test_idx)):
            ix = torch.as_tensor(indices, dtype=torch.long, device=device)
            logits = model.classify_edges(iso_h, drug_h, edge_iso[ix], edge_drug[ix])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(1)
            site_metrics[split_name] = classification_metrics(ast["label"][indices], pred, probs)
            predictions[split_name] = pred
            probabilities_by_split[split_name] = probs
        # Provide separate validation results for D and C, as well as combined DC.
        for site in VALIDATION_SITES:
            indices = np.asarray([i for i in val_idx if ast["site"][i] == site], dtype=np.int64)
            if len(indices):
                ix = torch.as_tensor(indices, dtype=torch.long, device=device)
                probs = torch.softmax(model.classify_edges(iso_h, drug_h, edge_iso[ix], edge_drug[ix]), dim=1).cpu().numpy()
                site_metrics[site] = classification_metrics(ast["label"][indices], probs.argmax(1), probs)

    metrics_by_drug, metrics_by_bacteria = {}, {}
    for split_name, indices in (("validation_DC", val_idx), ("test_B", test_idx)):
        probs = probabilities_by_split[split_name]
        y = ast["label"][indices]
        drugs_in_split = [names[int(index)] for index in ast["drug"][indices]]
        bacteria_in_split = [str(spectral["species"][int(index)]) or "unknown"
                             for index in ast["isolate"][indices]]
        metrics_by_drug[split_name] = grouped_metrics(y, probs, drugs_in_split)
        metrics_by_bacteria[split_name] = grouped_metrics(y, probs, bacteria_in_split)

    output_payload = {
        "split": {"train_sites": list(TRAIN_SITES), "validation_sites": list(VALIDATION_SITES),
                  "test_sites": list(TEST_SITES), "train_spectra": len(train_iso),
                  "validation_spectra": len(val_iso), "test_spectra": len(test_iso),
                  "train_ast_edges": len(train_idx), "validation_ast_edges": len(val_idx),
                  "test_ast_edges": len(test_idx)},
        "best_epoch": best_epoch, "best_validation_macro_f1": best_val,
        "metrics": site_metrics, "unmapped_antibiotics": ast["ignored_drugs"],
        "metrics_by_drug": metrics_by_drug,
        "metrics_by_bacteria": metrics_by_bacteria,
        "unlabeled_ast_value_counts": ast["unlabeled_values"],
        "drug_nodes": names, "class_names": CLASS_NAMES,
        "representation_history": ae_history, "training_history": history,
        "representation_reused": representation_reused,
        "susceptibility_labels_used_for_representation": False,
    }
    torch.save({"state_dict": model.cpu().state_dict(), "spectrum_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim, "drug_names": names, "drug_graphs": drug_graphs,
                "classes": CLASS_NAMES, "best_epoch": best_epoch}, args.output / "hybrid_ast_best.pt")
    (args.output / "metrics.json").write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    write_group_metrics_csv(args.output / "metrics_per_drug.csv", metrics_by_drug)
    write_group_metrics_csv(args.output / "metrics_per_bacteria.csv", metrics_by_bacteria)
    with (args.output / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        if history:
            # Get all fieldnames from the first entry and any subsequent entries
            fieldnames = list(history[0].keys())
            for entry in history[1:]:
                for key in entry.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
    with (args.output / "representation_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_reconstruction_mse"])
        writer.writeheader()
        writer.writerows(ae_history)
    with (args.output / "splits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["site", "sample_id", "species", "year", "split"])
        for i, site in enumerate(spectral["sites"]):
            split = "train" if site in TRAIN_SITES else "validation" if site in VALIDATION_SITES else "test"
            writer.writerow([site, spectral["sample_ids"][i], spectral["species"][i],
                             int(spectral["years"][i]), split])
    for split_name in ("validation_DC", "test_B"):
        indices = val_idx if split_name == "validation_DC" else test_idx
        with (args.output / f"{split_name}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["site", "sample_id", "year", "antibiotic", "true", "predicted"])
            pred = predictions[split_name]
            for row, edge_idx in enumerate(indices):
                sample_i = ast["isolate"][edge_idx]
                drug_i = ast["drug"][edge_idx]
                writer.writerow([ast["site"][edge_idx], spectral["sample_ids"][sample_i],
                                 int(ast["year"][edge_idx]), names[drug_i],
                                 CLASS_NAMES[ast["label"][edge_idx]], CLASS_NAMES[pred[row]]])
    print("\nFinal site-split metrics")
    print(json.dumps({"split": output_payload["split"], "best_epoch": best_epoch,
                      "metrics": site_metrics, "unmapped_antibiotics": ast["ignored_drugs"]}, indent=2))
    print("\nPer-antibiotic metrics (full reports also saved to metrics_per_drug.csv):")
    for split_name, results in metrics_by_drug.items():
        print(f"  {split_name}")
        for drug, metric in results.items():
            print(f"    {drug}: n={metric['n']}, accuracy={metric['accuracy']:.4f}, "
                  f"macro-F1={metric['f1_macro']:.4f}")
    print("\nPer-bacteria metrics (full reports also saved to metrics_per_bacteria.csv):")
    for split_name, results in metrics_by_bacteria.items():
        print(f"  {split_name}")
        for bacteria, metric in results.items():
            print(f"    {bacteria}: n={metric['n']}, accuracy={metric['accuracy']:.4f}, "
                  f"macro-F1={metric['f1_macro']:.4f}")
    
    # Print best epoch metrics summary
    print("\n" + "="*70)
    print(f"TRAINING COMPLETE - Best Epoch: {best_epoch}")
    print("="*70)
    if history:
        best_epoch_metrics = history[best_epoch - 1] if best_epoch <= len(history) else history[-1]
        print(f"\nBest Epoch Metrics (Epoch {best_epoch_metrics['epoch']}):")
        print(f"  Training Loss: {best_epoch_metrics.get('train_loss', 0):.5f}")
        print("\n  TRAIN SET:")
        print(f"    Accuracy:         {best_epoch_metrics.get('train_accuracy', 0):.4f}")
        print(f"    Balanced Acc:     {best_epoch_metrics.get('train_balanced_accuracy', 0):.4f}")
        print(f"    Precision (macro):{best_epoch_metrics.get('train_precision_macro', 0):.4f}")
        print(f"    Recall (macro):   {best_epoch_metrics.get('train_recall_macro', 0):.4f}")
        print(f"    F1-macro:         {best_epoch_metrics.get('train_f1_macro', 0):.4f}")
        print(f"    F1-weighted:      {best_epoch_metrics.get('train_f1_weighted', 0):.4f}")
        print(f"    ROC-AUC:          {best_epoch_metrics.get('train_roc_auc', 0):.4f}")
        print(f"    F1-S (Suscept):   {best_epoch_metrics.get('train_f1_S', 0):.4f}")
        print(f"    F1-R (Resistant): {best_epoch_metrics.get('train_f1_R', 0):.4f}")
        print("\n  VALIDATION SET:")
        print(f"    Accuracy:         {best_epoch_metrics.get('validation_accuracy', 0):.4f}")
        print(f"    Balanced Acc:     {best_epoch_metrics.get('validation_balanced_accuracy', 0):.4f}")
        print(f"    Precision (macro):{best_epoch_metrics.get('validation_precision_macro', 0):.4f}")
        print(f"    Recall (macro):   {best_epoch_metrics.get('validation_recall_macro', 0):.4f}")
        print(f"    F1-macro:         {best_epoch_metrics.get('validation_f1_macro', 0):.4f}")
        print(f"    F1-weighted:      {best_epoch_metrics.get('validation_f1_weighted', 0):.4f}")
        print(f"    ROC-AUC:          {best_epoch_metrics.get('validation_roc_auc', 0):.4f}")
        print(f"    F1-S (Suscept):   {best_epoch_metrics.get('validation_f1_S', 0):.4f}")
        print(f"    F1-R (Resistant): {best_epoch_metrics.get('validation_f1_R', 0):.4f}")
        print("\n  TEST SET:")
        print(f"    Accuracy:         {best_epoch_metrics.get('test_accuracy', 0):.4f}")
        print(f"    Balanced Acc:     {best_epoch_metrics.get('test_balanced_accuracy', 0):.4f}")
        print(f"    Precision (macro):{best_epoch_metrics.get('test_precision_macro', 0):.4f}")
        print(f"    Recall (macro):   {best_epoch_metrics.get('test_recall_macro', 0):.4f}")
        print(f"    F1-macro:         {best_epoch_metrics.get('test_f1_macro', 0):.4f}")
        print(f"    F1-weighted:      {best_epoch_metrics.get('test_f1_weighted', 0):.4f}")
        print(f"    ROC-AUC:          {best_epoch_metrics.get('test_roc_auc', 0):.4f}")
        print(f"    F1-S (Suscept):   {best_epoch_metrics.get('test_f1_S', 0):.4f}")
        print(f"    F1-R (Resistant): {best_epoch_metrics.get('test_f1_R', 0):.4f}")
    print("="*70)
    
    print(f"\nSaved checkpoint, metrics, histories, and predictions to: {args.output.resolve()}")
    
    # Automatic visualization
    if args.visualize:
        print("\n" + "="*70)
        print("Generating visualizations...")
        print("="*70)
        try:
            from visualize_results import visualize_results
            visualize_results(args.output, args.split_strategy)
        except ImportError as e:
            print(f"Warning: Could not import visualization module: {e}")
            print("You can manually generate visualizations by running:")
            print(f"  python visualize_results.py --output-dir {args.output} --split-mode {args.split_strategy}")
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            print("You can manually generate visualizations by running:")
            print(f"  python visualize_results.py --output-dir {args.output} --split-mode {args.split_strategy}")


if __name__ == "__main__":
    run()
