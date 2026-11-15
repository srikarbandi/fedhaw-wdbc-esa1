"""Stratified train/validation/test split and leakage-free standardisation.

Order of operations (no step sees data it should not):
  1. Split raw data 70/15/15, stratified by diagnosis, with a fixed seed.
  2. Fit StandardScaler on the TRAINING split only.
  3. Transform train, validation and test with that scaler.
The test split is used only for final evaluation.

Federated note: a StandardScaler fitted on the union of the hospitals'
training data is exactly reproducible without sharing patient rows. Each
hospital sends (n_k, sum x, sum x^2) and the server combines them;
`federated_scaler_stats` implements this and tests/test_data.py checks
the two are identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray     # row indices into the original file
    idx_val: np.ndarray
    idx_test: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def split_indices(y: np.ndarray, seed: int, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    idx = np.arange(len(y))
    idx_trval, idx_test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    rel_val = val_frac / (train_frac + val_frac)
    idx_train, idx_val = train_test_split(idx_trval, test_size=rel_val, stratify=y[idx_trval],
                                          random_state=seed)
    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)


def preprocess(X: np.ndarray, y: np.ndarray, seed: int, train_frac=0.70, val_frac=0.15,
               test_frac=0.15) -> SplitData:
    itr, iva, ite = split_indices(y, seed, train_frac, val_frac, test_frac)
    scaler = StandardScaler().fit(X[itr])          # training data only
    tf = lambda A: scaler.transform(A).astype(np.float32)
    return SplitData(tf(X[itr]), y[itr], tf(X[iva]), y[iva], tf(X[ite]), y[ite],
                     itr, iva, ite, scaler.mean_.astype(np.float64),
                     scaler.scale_.astype(np.float64))


def federated_scaler_stats(client_X: list[np.ndarray]):
    """Global mean/std from per-client sufficient statistics (n, sum, sum of squares)."""
    n = sum(len(x) for x in client_X)
    s = sum(x.astype(np.float64).sum(0) for x in client_X)
    ss = sum((x.astype(np.float64) ** 2).sum(0) for x in client_X)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    std[std == 0] = 1.0
    return mean, std
