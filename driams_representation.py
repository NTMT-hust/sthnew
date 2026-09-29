"""Learn reusable, unlabeled MALDI spectral representations from DRIAMS-A.

The spectra are first mapped to the 301 pseudo-ion windows described by Duan
et al. (2022). A denoising autoencoder then learns a compact embedding from
the spectra alone. Antibiotic susceptibility labels are never read or used.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


N_SOURCE_BINS = 6000
MZ_MIN, MZ_MAX = 2000.0, 20000.0


def pseudoion_edges() -> np.ndarray:
    """302 boundaries for 301 windows: 10 Da at either end, 20 Da within."""
    edges = np.asarray([2000.0, 2010.0, *range(2030, 7991, 20), 8000.0])
    if edges.size != 302:
        raise RuntimeError(f"Expected 302 window boundaries, got {edges.size}")
    return edges


@lru_cache(maxsize=1)
def _aggregation_plan() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Precompute source-bin indices and overlap fractions for each window."""
    width = (MZ_MAX - MZ_MIN) / N_SOURCE_BINS
    lefts = MZ_MIN + np.arange(N_SOURCE_BINS) * width
    rights = lefts + width
    edges = pseudoion_edges()
    plan = []
    for left, right in zip(edges[:-1], edges[1:]):
        indices = np.flatnonzero((rights > left) & (lefts < right))
        fractions = (np.minimum(rights[indices], right) - np.maximum(lefts[indices], left)) / width
        plan.append((indices, fractions))
    return tuple(plan)


def aggregate_spectrum(path: Path) -> np.ndarray:
    """Convert one public DRIAMS binned_6000 text file into 301 pseudo-ions."""
    values = np.loadtxt(path, dtype=np.float64, skiprows=1, usecols=1)
    if values.size != N_SOURCE_BINS:
        raise ValueError(f"Expected {N_SOURCE_BINS} bins in {path}, got {values.size}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray([values[idx] @ weights for idx, weights in _aggregation_plan()], dtype=np.float32)


def _metadata_for_year(csv_path: Path) -> dict[str, str]:
    """Read only isolate code and species; susceptibility columns are ignored."""
    metadata: dict[str, str] = {}
    if not csv_path.exists():
        return metadata
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"code", "species"}.issubset(reader.fieldnames or []):
            return metadata
        for row in reader:
            metadata[str(row.get("code", ""))] = str(row.get("species", ""))
    return metadata


def collect_spectra(root: Path, sites: list[str], years: list[int], species_filter: str | None):
    """Load spectra and non-label identifiers from DRIAMS site/year folders."""
    vectors, ids, species_names, site_names, sample_years = [], [], [], [], []
    for site in sites:
        for year in years:
            spectra_dir = root / site / "binned_6000" / str(year)
            if not spectra_dir.is_dir():
                continue
            metadata = _metadata_for_year(root / site / "id" / str(year) / f"{year}_clean.csv")
            for spectrum_path in sorted(spectra_dir.glob("*.txt")):
                sample_id = spectrum_path.stem
                sample_species = metadata.get(sample_id, "")
                if species_filter and sample_species.casefold() != species_filter.casefold():
                    continue
                try:
                    vectors.append(aggregate_spectrum(spectrum_path))
                except (OSError, ValueError):
                    continue
                ids.append(sample_id)
                species_names.append(sample_species)
                site_names.append(site)
                sample_years.append(year)
    if not vectors:
        raise ValueError("No usable spectra found; check --driams-root, --sites, and --years.")
    return (np.stack(vectors), np.asarray(ids), np.asarray(species_names),
            np.asarray(site_names), np.asarray(sample_years, dtype=np.int16))


class SpectralAutoencoder(nn.Module):
    """Denoising MLP autoencoder; the bottleneck is the exported representation."""

    def __init__(self, input_dim: int = 301, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(), nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driams-root", type=Path,
                        default=Path(__file__).resolve().parent.parent.parent / "DRIAMS",
                        help="Directory containing DRIAMS-A/ (default: ../../DRIAMS relative to this script).")
    parser.add_argument("--sites", nargs="+", default=["DRIAMS-A"],
                        help="DRIAMS sites to pool, e.g. DRIAMS-A DRIAMS-B.")
    parser.add_argument("--train-years", type=int, nargs="+", default=[2015, 2016, 2017])
    parser.add_argument("--embed-years", type=int, nargs="+", default=[2015, 2016, 2017, 2018],
                        help="Years to export embeddings for; scaler/encoder are fit on train-years only.")
    parser.add_argument("--species", default=None,
                        help="Optional exact species filter, e.g. 'Staphylococcus aureus'.")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.latent_dim < 1 or args.epochs < 1:
        parser.error("--latent-dim and --epochs must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the training normalization from training years only.
    X_train_raw, *_ = collect_spectra(args.driams_root, args.sites, args.train_years, args.species)
    X_train_log = np.log1p(np.maximum(X_train_raw, 0))
    mean = X_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = X_train_log.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    X_train = np.clip((X_train_log - mean) / scale, -10, 10).astype(np.float32)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)),
                              batch_size=args.batch_size, shuffle=True)
    model = SpectralAutoencoder(input_dim=X_train.shape[1], latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for (clean,) in train_loader:
            clean = clean.to(device)
            noisy = clean + 0.05 * torch.randn_like(clean)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(noisy), clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * clean.shape[0]
            seen += clean.shape[0]
        print(f"epoch {epoch + 1:03d}/{args.epochs}  reconstruction_mse={total_loss / seen:.6f}")

    X_raw, sample_ids, sample_species, sample_sites, sample_years = collect_spectra(
        args.driams_root, args.sites, args.embed_years, args.species)
    X_log = np.log1p(np.maximum(X_raw, 0))
    X = np.clip((X_log - mean) / scale, -10, 10).astype(np.float32)
    model.eval()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(X), args.batch_size):
            batch = torch.from_numpy(X[start:start + args.batch_size]).to(device)
            embeddings.append(model.encoder(batch).cpu().numpy())
    Z = np.concatenate(embeddings, axis=0)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "input_dim": int(X.shape[1]),
                "latent_dim": args.latent_dim, "mean": mean, "scale": scale,
                "preprocessing": "301 pseudo-ions; log1p; train-year feature standardization"},
               args.output / "driams_spectral_encoder.pt")
    np.savez_compressed(args.output / "driams_embeddings.npz", embeddings=Z,
                        sample_ids=sample_ids, species=sample_species, sites=sample_sites,
                        years=sample_years)
    metadata = {"n_train_spectra": int(len(X_train)), "n_embedded_spectra": int(len(Z)),
                "train_years": args.train_years, "embed_years": args.embed_years,
                "sites": args.sites, "species_filter": args.species,
                "latent_dim": args.latent_dim, "epochs": args.epochs,
                "device": str(device), "susceptibility_labels_used": False}
    (args.output / "representation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(Z)} embeddings of dimension {args.latent_dim} to {args.output}")


if __name__ == "__main__":
    main()
