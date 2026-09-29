"""Load binned MALDI spectra and linked AST labels from DRIAMS A-D."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re

import numpy as np


N_SOURCE_BINS = 6000
MZ_MIN, MZ_MAX = 2000.0, 20000.0
# Intermediate and all other non-binary AST values are treated as unlabeled.
LABELS = {"S": 0, "R": 1}


def _normalize_drug_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _edges():
    result = np.asarray([2000.0, 2010.0, *range(2030, 7991, 20), 8000.0])
    if result.size != 302:
        raise RuntimeError("Pseudo-ion window configuration must produce 301 features")
    return result


@lru_cache(maxsize=1)
def _plan():
    width = (MZ_MAX - MZ_MIN) / N_SOURCE_BINS
    left = MZ_MIN + np.arange(N_SOURCE_BINS) * width
    right = left + width
    result = []
    edges = _edges()
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((right > lo) & (left < hi))
        frac = (np.minimum(right[idx], hi) - np.maximum(left[idx], lo)) / width
        result.append((idx, frac))
    return result


def pseudoions(path: Path) -> np.ndarray:
    """Aggregate one 6,000-bin file into the source-paper 301 windows."""
    raw = path.read_bytes()
    body = raw.partition(b"\n")[2]
    columns = np.fromstring(body, dtype=np.float64, sep=" ")
    if columns.size != N_SOURCE_BINS * 2:
        raise ValueError(f"Expected 6,000 indexed bins, found {columns.size // 2}: {path}")
    intensity = columns[1::2]
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray([intensity[idx] @ weight for idx, weight in _plan()], dtype=np.float32)


def _safe_pseudoions(path: Path):
    try:
        return pseudoions(path)
    except (OSError, ValueError):
        return None


def load_site_spectra(root: Path, sites: list[str], on_error=None, workers: int = 8):
    """Read all spectra for selected sites; metadata ids stay aligned by row."""
    vectors, sample_ids, species, site_ids, years = [], [], [], [], []
    for site in sites:
        spectra_root = root / site / "binned_6000"
        if not spectra_root.is_dir():
            raise FileNotFoundError(f"Missing spectrum directory: {spectra_root}")
        for year_dir in sorted((p for p in spectra_root.iterdir() if p.is_dir()), key=lambda p: p.name):
            year = int(year_dir.name)
            metadata_path = root / site / "id" / str(year) / f"{year}_clean.csv"
            species_by_code = {}
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
                    for row in csv.DictReader(f):
                        species_by_code[row.get("code", "")] = row.get("species", "")
            paths = sorted(year_dir.glob("*.txt"))
            print(f"Reading {site} {year}: {len(paths):,} spectra", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for completed, (spectrum_path, vector) in enumerate(zip(paths, pool.map(_safe_pseudoions, paths)), 1):
                    code = spectrum_path.stem
                    if vector is None:
                        if on_error:
                            on_error(str(spectrum_path))
                        continue
                    vectors.append(vector)
                    sample_ids.append(code)
                    species.append(species_by_code.get(code, ""))
                    site_ids.append(site)
                    years.append(year)
                    if completed % 5000 == 0:
                        print(f"  {completed:,}/{len(paths):,} spectra read", flush=True)
    if not vectors:
        raise ValueError("No readable DRIAMS spectra were found.")
    return {"spectra": np.stack(vectors), "sample_ids": np.asarray(sample_ids),
            "species": np.asarray(species), "sites": np.asarray(site_ids),
            "years": np.asarray(years, dtype=np.int16)}


def load_ast_edges(root: Path, sample_index: dict, drug_index: dict, sites: list[str]):
    """Create isolate-drug edges for S/R; I and missing labels are excluded."""
    isolate_idx, drug_idx, labels, row_sites, codes, years = [], [], [], [], [], []
    ignored = set()
    unlabeled_values = {}
    metadata_rows = 0
    matched_samples = 0
    mapped_drug_columns = set()
    for site in sites:
        id_root = root / site / "id"
        if not id_root.is_dir():
            continue
        for year_dir in sorted((p for p in id_root.iterdir() if p.is_dir()), key=lambda p: p.name):
            year = int(year_dir.name)
            metadata_path = year_dir / f"{year}_clean.csv"
            if not metadata_path.exists():
                continue
            with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    metadata_rows += 1
                    code = row.get("code", "")
                    sample = sample_index.get((site, year, code))
                    if sample is None:
                        continue
                    matched_samples += 1
                    for column, value in row.items():
                        if column in {"code", "species", "laboratory_species", "genus", "combined_code"} or not value:
                            continue
                        # The SMILES index uses normalized names; normalize the
                        # DRIAMS CSV headers the same way before lookup.
                        drug = drug_index.get(_normalize_drug_name(column))
                        if drug is None:
                            ignored.add(column)
                            continue
                        mapped_drug_columns.add(column)
                        label = LABELS.get(value.strip().upper())
                        if label is None:
                            value_key = value.strip().upper() or "EMPTY"
                            unlabeled_values[value_key] = unlabeled_values.get(value_key, 0) + 1
                            continue
                        isolate_idx.append(sample)
                        drug_idx.append(drug)
                        labels.append(label)
                        row_sites.append(site)
                        codes.append(code)
                        years.append(year)
    if not labels:
        raise ValueError(
            "No S/R AST entries matched the embedded samples and drug structures. "
            f"Metadata rows={metadata_rows}, matched sample rows={matched_samples}, "
            f"mapped drug columns with values={len(mapped_drug_columns)}. "
            "Check that cached sample IDs/site names match DRIAMS codes and that "
            "antibiotic names have SMILES mappings."
        )
    return {"isolate": np.asarray(isolate_idx, dtype=np.int64),
            "drug": np.asarray(drug_idx, dtype=np.int64),
            "label": np.asarray(labels, dtype=np.int64),
            "site": np.asarray(row_sites), "code": np.asarray(codes),
            "year": np.asarray(years, dtype=np.int16), "ignored_drugs": sorted(ignored),
            "unlabeled_values": unlabeled_values, "matched_sample_rows": matched_samples,
            "metadata_rows": metadata_rows}
