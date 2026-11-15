import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.loader import DEFAULT_PATH, FEATURE_NAMES, load_raw, load_wdbc
from src.data.preprocessing import federated_scaler_stats, preprocess, split_indices


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_correct_file_and_dimensions():
    assert DEFAULT_PATH.name == "wdbc.data"
    d = load_wdbc()
    assert d.X.shape == (569, 30)
    assert d.y.shape == (569,)
    assert d.report["verified"] and not d.report["problems"]


def test_two_classes_and_malignant_is_one():
    d = load_wdbc()
    raw = load_raw()
    assert sorted(np.unique(d.y)) == [0, 1]
    assert ((raw["diagnosis"] == "M").to_numpy() == (d.y == 1)).all()
    assert (d.y == 1).sum() == 212 and (d.y == 0).sum() == 357


def test_no_missing_values_and_numeric():
    d = load_wdbc()
    assert d.report["missing_values_total"] == 0
    assert np.isfinite(d.X).all()
    assert d.report["non_numeric_feature_columns"] == []


def test_id_and_diagnosis_not_in_features():
    raw = load_raw()
    d = load_wdbc()
    np.testing.assert_allclose(d.X, raw[FEATURE_NAMES].to_numpy(np.float32))
    ids = raw["id"].to_numpy(np.float64)
    for j in range(30):     # no feature column equals the ID column
        assert not np.allclose(d.X[:, j], ids)
    assert d.X.shape[1] == 30 and "id" not in d.feature_names and "diagnosis" not in d.feature_names


def test_original_dataset_not_modified():
    before = sha(DEFAULT_PATH)
    d = load_wdbc()
    preprocess(d.X, d.y, 42)
    assert sha(DEFAULT_PATH) == before


def test_split_70_15_15_stratified_disjoint():
    d = load_wdbc()
    tr, va, te = split_indices(d.y, 42)
    assert len(tr) + len(va) + len(te) == 569
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert abs(len(te) - 0.15 * 569) <= 1 and abs(len(va) - 0.15 * 569) <= 1
    for part in (tr, va, te):
        assert abs(d.y[part].mean() - d.y.mean()) < 0.02


def test_split_reproducible_and_seed_dependent():
    d = load_wdbc()
    a = split_indices(d.y, 42)
    b = split_indices(d.y, 42)
    c = split_indices(d.y, 123)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[2], c[2])


def test_scaler_fit_on_train_only_no_leakage():
    d = load_wdbc()
    sp = preprocess(d.X, d.y, 42)
    ref = StandardScaler().fit(d.X[sp.idx_train])
    np.testing.assert_allclose(sp.scaler_mean, ref.mean_)
    full = StandardScaler().fit(d.X)
    assert not np.allclose(sp.scaler_mean, full.mean_)            # differs from full-data fit
    np.testing.assert_allclose(sp.X_train.mean(0), 0, atol=1e-4)  # train standardised
    assert np.abs(sp.X_test.mean(0)).max() > 1e-3                 # test not re-centred on itself
    np.testing.assert_allclose(sp.X_test, ref.transform(d.X[sp.idx_test]).astype(np.float32),
                               rtol=1e-5, atol=1e-5)


def test_federated_scaler_equals_pooled_train_scaler():
    d = load_wdbc()
    sp = preprocess(d.X, d.y, 42)
    Xtr = d.X[sp.idx_train]
    parts = np.array_split(np.random.default_rng(0).permutation(len(Xtr)), 5)
    mean, std = federated_scaler_stats([Xtr[p] for p in parts])
    np.testing.assert_allclose(mean, sp.scaler_mean, rtol=1e-6)
    np.testing.assert_allclose(std, sp.scaler_scale, rtol=1e-5)
