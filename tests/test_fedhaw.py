"""Original-FedHAW tests. Items 1-11 follow the project brief; the last two
tests check equivalence with external/FedHAW/functions/training.py."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.federated.aggregation import fedhaw_init, fedhaw_update, softmax_vec
from src.federated.client import Client
from src.federated.fedhaw import FedHAW
from src.federated.params import get_flat_params
from src.federated.server import run_federated
from src.models.mlp import MLP
from src.utils.config import load_config

ORIG = Path(__file__).resolve().parent.parent / "external" / "FedHAW"


def original_training_module():
    if not (ORIG / "functions" / "training.py").exists():
        pytest.skip("external/FedHAW not cloned")
    sys.path.insert(0, str(ORIG))
    try:
        return importlib.import_module("functions.training")
    finally:
        sys.path.remove(str(ORIG))


def clients(sizes=(20, 35, 12, 50, 28), seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for k, n in enumerate(sizes):
        X = torch.randn(n, 30, generator=g)
        y = (X[:, 0] + 0.5 * torch.randn(n, generator=g) > 0).long()
        out.append(Client(f"Hospital {k+1}", torch.utils.data.TensorDataset(X, y)))
    return out


def rand_state(K=4, d=25, seed=0):
    rng = np.random.default_rng(seed)
    T = lambda a: torch.tensor(a, dtype=torch.float64)
    pre = rng.normal(size=d)
    cands = [T(pre + 0.1 * rng.normal(size=d)) for _ in range(K)]
    cpre = [T(rng.normal(size=d)) for _ in range(K)]
    return T(pre), cands, cpre


# 1, 2
def test_gamma_zero_and_lambda_log_proportions():
    g, lam = fedhaw_init([10, 30, 60])
    assert g == 0.0
    np.testing.assert_allclose(lam, np.log([0.1, 0.3, 0.6]))
    np.testing.assert_allclose(softmax_vec(lam), [0.1, 0.3, 0.6])


# 3
def test_first_round_skips_hypergradient():
    pre, c, _ = rand_state(K=3)
    g0, lam0 = fedhaw_init([10, 30, 60])
    p, g, lam, w = fedhaw_update(pre, c, None, g0, lam0, loop=0, mu=0.1, lr_gamma=10., lr_lambda=10.)
    assert g == g0 and lam == lam0
    torch.testing.assert_close(p, 0.1 * c[0] + 0.3 * c[1] + 0.6 * c[2])   # == FedAvg


# 4, 5
def test_later_rounds_update_gamma_and_lambda():
    pre, c, cpre = rand_state()
    g0, lam0 = 0.0, list(np.log([.1, .2, .3, .4]))
    _, g, lam, _ = fedhaw_update(pre, c, cpre, g0, lam0, loop=1, mu=0.05, lr_gamma=1e-2, lr_lambda=1e-1)
    assert g != g0
    assert all(a != b for a, b in zip(lam, lam0))


def test_equations_against_independent_derivation():
    pre, c, cpre = rand_state(seed=4)
    P, C, CP = pre.numpy(), [x.numpy() for x in c], [x.numpy() for x in cpre]
    g0, lam0, mu, lg, ll = 0.03, list(np.log([.1, .2, .3, .4])), 0.05, 1e-3, 1e-2
    a = np.exp(lam0) / np.exp(lam0).sum()
    d = P - sum(a[k] * C[k] for k in range(4))
    g1 = g0 - lg * np.exp(g0) / mu * (d @ P)
    lam1 = [lam0[k] - ll * np.exp(2 * g1) * a[k] * (1 - a[k]) / mu * (d @ CP[k]) for k in range(4)]
    a1 = np.exp(lam1) / np.exp(lam1).sum()
    theta = np.exp(g1) * sum(a1[k] * C[k] for k in range(4))
    p, g, lam, w = fedhaw_update(pre, c, cpre, g0, lam0, 2, mu, lg, ll)
    assert g == pytest.approx(g1, rel=1e-12)
    np.testing.assert_allclose(lam, lam1, rtol=1e-12)
    np.testing.assert_allclose(p.numpy(), theta, rtol=1e-10)


def test_lambda_update_uses_updated_gamma():
    pre, c, cpre = rand_state(K=2, seed=5)
    _, g, lam, _ = fedhaw_update(pre, c, cpre, 0.0, [0.0, 0.0], 1, 0.1, 0.5, 0.1)
    d = pre - 0.5 * (c[0] + c[1])
    new = [-0.1 * np.exp(2 * g) * .25 / .1 * torch.sum(d * cpre[k]).item() for k in range(2)]
    old = [-0.1 * 1.0 * .25 / .1 * torch.sum(d * cpre[k]).item() for k in range(2)]
    assert abs(g) > 1e-3
    np.testing.assert_allclose(lam, new)
    assert not np.allclose(lam, old)


# 6
def test_previous_round_candidates_retained_and_used():
    pre, c, cpre = rand_state()
    s = FedHAW(1e-2, 1e-1)
    s.reset([10, 20, 30, 40])
    s.aggregate(pre, cpre, 0, 0.05)
    assert s.params_candidates_pre is cpre                  # stored after round 0
    _, info = s.aggregate(pre, c, 1, 0.05)
    _, g_ref, lam_ref, _ = fedhaw_update(pre, c, cpre, 0.0, list(np.log([.1, .2, .3, .4])), 1,
                                         0.05, 1e-2, 1e-1)
    np.testing.assert_allclose(info["lambdas"], lam_ref)    # used cpre from round 0
    assert s.params_candidates_pre is c


def test_state_not_reset_between_rounds():
    cfg = load_config({"training": {"lr": 0.05}})
    torch.manual_seed(0)
    init = get_flat_params(MLP())
    _, log = run_federated(FedHAW(1e-3, 1e-1), clients(), init, cfg, lambda: MLP(), rounds=6)
    gam = [r["gamma"] for r in log]
    assert gam[0] == 0.0 and len(set(gam[1:])) > 1          # gamma evolves cumulatively


# 7, 8, 9, 10
def test_weights_aggregation_scaling_and_dims():
    pre, c, cpre = rand_state()
    p, g, lam, w = fedhaw_update(pre, c, cpre, 0.2, list(np.log([.1, .2, .3, .4])), 3, 0.05, 1e-3, 1e-2)
    assert abs(w.sum() - 1) < 1e-12 and (w > 0).all()
    expected = np.exp(g) * sum(w[k] * c[k] for k in range(4))
    torch.testing.assert_close(p, expected)
    assert np.exp(g) != 1.0 and p.shape == pre.shape


# 11
def test_normal_operation_is_finite():
    cfg = load_config({"training": {"lr": 0.05}})
    torch.manual_seed(1)
    init = get_flat_params(MLP())
    p, log = run_federated(FedHAW(1e-3, 1e-2), clients(), init, cfg, lambda: MLP(), rounds=10)
    assert torch.isfinite(p).all() and all(r["finite"] for r in log)
    assert all(np.isfinite(r["gamma"]) for r in log)


# Equivalence with the original repository
def test_aggregation_matches_original_with_synthetic_candidates(monkeypatch):
    orig = original_training_module()
    K = 4
    pre, c_now, c_prev = rand_state(K=K, seed=9)
    pre, c_now, c_prev = pre.float(), [x.float() for x in c_now], [x.float() for x in c_prev]
    it = iter(c_now)
    monkeypatch.setattr(orig, "_local_train", lambda *a, **k: (next(it), 0.0))
    g0, lam0 = 0.01, list(np.log([.1, .2, .3, .4]))
    o_p, _, o_g, o_lam, o_c = orig.train_fedhaw("wdbc", [None] * K, pre, 8, 1, 0.05, "cpu", K,
                                                None, None, "SGD", 3, c_prev, g0, lam0, 1e-2, 1e-1)
    p, g, lam, _ = fedhaw_update(pre, c_now, c_prev, g0, lam0, 3, 0.05, 1e-2, 1e-1)
    assert torch.equal(o_p, p) and o_g == g and o_lam == lam


@pytest.mark.parametrize("lr_gamma,lr_lambda", [(1e-3, 1e-2), (5e-2, 5.0)])
def test_full_training_bit_exact_vs_original(lr_gamma, lr_lambda):
    """6 rounds of the ORIGINAL train_fedhaw (driven like run_serverside.py)
    vs our server loop, from identical RNG state: identical global params,
    gamma and lambda in every round."""
    orig = original_training_module()
    cl = clients()
    K, mu, E, B = len(cl), 0.05, 2, 8
    cfg = load_config({"training": {"lr": mu, "local_epochs": E, "batch_size": B, "optimizer": "SGD"}})
    torch.manual_seed(123)
    init = get_flat_params(MLP())
    crit = nn.CrossEntropyLoss()

    torch.manual_seed(999)
    p, (g, lam), cpre = init.clone(), fedhaw_init([c.n for c in cl]), None
    orig_trace = []
    for t in range(6):
        p, _, g, lam, cpre = orig.train_fedhaw("wdbc", [c.dataset for c in cl], p, B, E, mu, "cpu",
                                               K, MLP(), crit, "SGD", t, cpre, g, lam,
                                               lr_gamma, lr_lambda)
        orig_trace.append((p.clone(), g, list(lam)))

    ours = []
    for T in range(1, 7):        # re-run from scratch for each horizon to capture state
        torch.manual_seed(999)
        s = FedHAW(lr_gamma, lr_lambda)
        q, _ = run_federated(s, cl, init, cfg, lambda: MLP(), rounds=T)
        ours.append((q, s.gammat, list(s.lambdat)))

    for (po, go, lo), (pq, gq, lq) in zip(orig_trace, ours):
        assert torch.equal(po, pq)
        assert go == gq
        assert lo == lq
    assert orig_trace[-1][1] != 0.0
