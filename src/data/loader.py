"""Load and verify the user's existing WDBC file (never modified).

File: breast+cancer+wisconsin+diagnostic/wdbc.data (original UCI format;
the accompanying wdbc.names documents the columns). Comma separated, no
header, 32 columns:
    col 0      ID number            -> dropped (identifier, not a feature)
    col 1      Diagnosis 'M' / 'B'  -> label: B = 0 (benign), M = 1 (malignant, positive)
    cols 2-31  30 real-valued features: mean, standard error and "worst"
               of 10 nuclear measurements computed from FNA images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / "breast+cancer+wisconsin+diagnostic" / "wdbc.data"

_BASE = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
         "concavity", "concave_points", "symmetry", "fractal_dimension"]
FEATURE_NAMES = ([f"{b}_mean" for b in _BASE] + [f"{b}_se" for b in _BASE]
                 + [f"{b}_worst" for b in _BASE])
COLUMNS = ["id", "diagnosis"] + FEATURE_NAMES
LABEL_MAP = {"B": 0, "M": 1}
CLASS_NAMES = ["Benign", "Malignant"]

EXPECTED_N = 569
EXPECTED_FEATURES = 30


@dataclass
class WDBC:
    X: np.ndarray            # (569, 30) float32
    y: np.ndarray            # (569,) int64, 1 = malignant
    ids: np.ndarray          # (569,) patient IDs (kept only for bookkeeping)
    feature_names: list
    report: dict


def resolve_path(path=None) -> Path:
    p = Path(path) if path else DEFAULT_PATH
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_raw(path=None) -> pd.DataFrame:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"WDBC file not found: {p}")
    df = pd.read_csv(p, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected {len(COLUMNS)} columns, found {df.shape[1]}")
    df.columns = COLUMNS
    return df


def verify(df: pd.DataFrame) -> dict:
    """Programmatic checks requested by the project brief. Raises on mismatch."""
    feats = df[FEATURE_NAMES]
    diag = df["diagnosis"].astype(str).str.strip()
    raw_labels = sorted(diag.unique().tolist())
    report = {
        "n_samples": int(len(df)),
        "n_columns_in_file": int(df.shape[1]),
        "n_features": int(feats.shape[1]),
        "raw_diagnosis_values": raw_labels,
        "label_mapping": LABEL_MAP,
        "class_counts": {"benign": int((diag == "B").sum()), "malignant": int((diag == "M").sum())},
        "missing_values_total": int(df.isna().sum().sum()),
        "non_numeric_feature_columns": [c for c in FEATURE_NAMES
                                        if not pd.api.types.is_numeric_dtype(feats[c])],
        "feature_dtypes": sorted({str(t) for t in feats.dtypes}),
        "duplicate_rows_full": int(df.duplicated().sum()),
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "duplicate_feature_vectors": int(feats.duplicated().sum()),
        "id_is_integer": bool(pd.api.types.is_integer_dtype(df["id"])),
        "feature_min": float(feats.min().min()),
        "feature_max": float(feats.max().max()),
    }
    report["class_percent"] = {k: round(100 * v / report["n_samples"], 2)
                               for k, v in report["class_counts"].items()}
    problems = []
    if report["n_samples"] != EXPECTED_N:
        problems.append(f"expected {EXPECTED_N} samples, got {report['n_samples']}")
    if report["n_features"] != EXPECTED_FEATURES:
        problems.append("feature count mismatch")
    if raw_labels != ["B", "M"]:
        problems.append(f"unexpected diagnosis values {raw_labels}")
    if report["missing_values_total"]:
        problems.append("missing values present")
    if report["non_numeric_feature_columns"]:
        problems.append("non-numeric features")
    if report["duplicate_ids"]:
        problems.append("duplicate patient IDs")
    report["verified"] = not problems
    report["problems"] = problems
    if problems:
        raise ValueError("WDBC verification failed: " + "; ".join(problems))
    return report


def load_wdbc(path=None) -> WDBC:
    df = load_raw(path)
    report = verify(df)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df["diagnosis"].str.strip().map(LABEL_MAP).to_numpy(dtype=np.int64)
    return WDBC(X=X, y=y, ids=df["id"].to_numpy(), feature_names=list(FEATURE_NAMES), report=report)
