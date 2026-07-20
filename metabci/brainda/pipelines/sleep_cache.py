# -*- coding: utf-8 -*-
"""
Unified sleep staging cache paths and I/O.

All cache access goes through this module — no script should hardcode
cache directory layouts or filename conventions.

Cache structure::

    {cache_root}/
    ├── manifest.json
    └── fpzcz_100hz/
        └── ctx{context}_{mode}_{label_mode}/
            ├── 4001.npz
            └── ...

Each .npz contains X (n_windows, context, n_samples), y (n_windows,),
and metadata (record_id, subject_id, schema_version, ...).
"""

import json, os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Bump when preprocessing changes (e.g. filter chain, unit, resampling).
# Old caches will be rejected until migrated or rebuilt.
CACHE_SCHEMA_VERSION = 1

# String describing the exact preprocessing pipeline that produced the cache.
PREPROCESS_DESCRIPTION = [
    "dataset filter 0.3-45 Hz",
    "resample to 100 Hz",
    "cache filter 0.5-40 Hz",
    "30-second epoch extraction",
    "causal context window [t-2,t-1,t]",
]

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def cache_config_dir(
    cache_root: str,
    context: int = 3,
    causal: bool = True,
    label_mode: str = "5class",
) -> Path:
    """Return the directory for a specific cache configuration."""
    mode = "causal" if causal else "center"
    return (
        Path(cache_root)
        / "fpzcz_100hz"
        / f"ctx{context}_{mode}_{label_mode}"
    )


def cache_file_path(
    cache_root: str,
    record_id: str,
    context: int = 3,
    causal: bool = True,
    label_mode: str = "5class",
) -> Path:
    """Return the full path to a single record's cache file."""
    return cache_config_dir(cache_root, context, causal, label_mode) / f"{record_id}.npz"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(cache_root: str, context: int, causal: bool,
                   label_mode: str = "5class") -> dict:
    """Build the expected manifest for this cache configuration."""
    mode = "causal" if causal else "center"
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": "Sleep-EDF Expanded sleep-cassette",
        "channel": "EEG Fpz-Cz",
        "sampling_rate": 100,
        "epoch_seconds": 30,
        "context": context,
        "context_mode": mode,
        "label_mode": label_mode,
        "unit": "uV",
        "normalization": "none",
        "preprocessing": PREPROCESS_DESCRIPTION,
        "expected_shape": ["N", context, 3000],
    }


def save_manifest(cache_root: str, context: int, causal: bool,
                  label_mode: str = "5class", total_records: int = 153,
                  real_subjects: int = 78):
    """Write manifest.json to the config directory."""
    manifest = build_manifest(cache_root, context, causal, label_mode)
    manifest["total_records"] = total_records
    manifest["real_subjects"] = real_subjects

    cfg_dir = cache_config_dir(cache_root, context, causal, label_mode)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def load_manifest(cache_root: str, context: int, causal: bool,
                  label_mode: str = "5class") -> Optional[dict]:
    """Load manifest.json, returning None if absent."""
    path = cache_config_dir(cache_root, context, causal, label_mode) / "manifest.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def load_cache_record(
    cache_root: str,
    record_id: str,
    context: int = 3,
    causal: bool = True,
    label_mode: str = "5class",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load X, y for a single record. Raises FileNotFoundError if missing."""
    path = cache_file_path(cache_root, record_id, context, causal, label_mode)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    with np.load(path, allow_pickle=True) as d:
        return d["X"], d["y"]


def save_cache_record(
    cache_root: str,
    record_id: str,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: str,
    context: int = 3,
    causal: bool = True,
    label_mode: str = "5class",
):
    """Save a single record's cache file with metadata."""
    path = cache_file_path(cache_root, record_id, context, causal, label_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X, y=y,
        record_id=str(record_id),
        subject_id=subject_id,
        schema_version=CACHE_SCHEMA_VERSION,
        channel="EEG Fpz-Cz",
        sampling_rate=100,
        context=context,
        context_mode="causal" if causal else "center",
        label_mode=label_mode,
        unit="uV",
    )


def discover_records(
    cache_root: str,
    context: int = 3,
    causal: bool = True,
    label_mode: str = "5class",
) -> list[str]:
    """Return sorted list of record IDs present in the cache config."""
    cfg_dir = cache_config_dir(cache_root, context, causal, label_mode)
    if not cfg_dir.exists():
        return []
    return sorted(
        p.stem for p in cfg_dir.glob("*.npz")
        if p.stem.isdigit()
    )


def load_records_batch(
    cache_root: str,
    record_ids: list[str],
    context: int = 3,
    causal: bool = True,
    label_mode: str = "5class",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, subject_ids for a list of records."""
    X_list, y_list, subj_list = [], [], []
    for rid in record_ids:
        X, y = load_cache_record(cache_root, rid, context, causal, label_mode)
        X_list.append(X)
        y_list.append(y)
        # Parse real subject from record ID (SC4ssNE0 → ss)
        sid = str(rid)[1:3] if len(str(rid)) >= 3 else str(rid)
        subj_list.extend([sid] * len(y))
    return (np.concatenate(X_list), np.concatenate(y_list),
            np.array(subj_list))
