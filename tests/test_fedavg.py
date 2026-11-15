import numpy as np
import torch
import torch.nn as nn

from src.federated.aggregation import fedavg_aggregate
from src.federated.client import Client
from src.federated.fedavg import FedAvg
from src.federated.fedprox import FedProx
from src.federated.params import get_flat_params
from src.federated.server import run_federated
from src.models.mlp import MLP
from src.utils.config import load_config


def _clients(sizes=(12, 30, 18), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [Client(f"H{k+1}", torch.utils.data.TensorDataset(
        torch.randn(n, 30, generator=g), torch.randint(0, 2, (n,), generator=g)))
        for k, n in enumerate(sizes)]


def test_weights_sum_to_one_and_match_sample_sizes():
    c = [torch.randn(10) for _ in range(3)]
    g, w = fedavg_aggregate(c, [10, 30, 60])
    np.testing.assert_allclose(w, [0.1, 0.3, 0.6])
    assert abs(w.sum() - 1) < 1e-12
    torch.testing.assert_close(g, 0.1 * c[0] + 0.3 * c[1] + 0.6 * c[2])
    assert g.shape == c[0].shape


def test_fedavg_round_dimensions_and_finite():
    cfg = load_config({"training": {"lr": 0.05, "rounds": 3}})
    torch.manual_seed(0)
    init = get_flat_params(MLP())
    p, log = run_federated(FedAvg(), _clients(), init, cfg, lambda: MLP())
    assert p.shape == init.shape and torch.isfinite(p).all()
    assert len(log) == 3 and abs(sum(log[0][f"weight_{k}"] for k in (1, 2, 3)) - 1) < 1e-12


def test_fedprox_zero_mu_equals_fedavg():
    cfg = load_config({"training": {"lr": 0.05, "rounds": 2}})
    torch.manual_seed(0)
    init = get_flat_params(MLP())
    torch.manual_seed(1)
    a, _ = run_federated(FedAvg(), _clients(), init, cfg, lambda: MLP())
    torch.manual_seed(1)
    b, _ = run_federated(FedProx(0.0), _clients(), init, cfg, lambda: MLP())
    torch.testing.assert_close(a, b)


def test_fedprox_proximal_term_reduces_drift():
    torch.manual_seed(0)
    init = get_flat_params(MLP())
    c = _clients((40,))[0]
    d = []
    for mu in (0.0, 1.0):
        torch.manual_seed(3)
        p, *_ = c.local_train(init, MLP(), nn.CrossEntropyLoss(), 0.1, 5, 8, mu_prox=mu)
        d.append((p - init).norm().item())
    assert d[1] < d[0]


def test_empty_client_returns_global_model():
    torch.manual_seed(0)
    init = get_flat_params(MLP())
    empty = Client("H0", torch.utils.data.TensorDataset(torch.empty(0, 30), torch.empty(0, dtype=torch.long)))
    p, loss, nb, _ = empty.local_train(init, MLP(), nn.CrossEntropyLoss(), 0.1, 1, 8)
    assert torch.equal(p, init) and nb == 0
