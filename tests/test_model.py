import torch
import torch.nn as nn

from src.federated.params import get_flat_params, num_params, set_flat_params
from src.models.mlp import MLP, build_model
from src.utils.config import load_config


def test_architecture_from_config():
    m = build_model(load_config())
    linears = [l for l in m.modules() if isinstance(l, nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [(30, 64), (64, 32), (32, 2)]
    assert num_params(m) == 30 * 64 + 64 + 64 * 32 + 32 + 32 * 2 + 2


def test_forward_shape_and_raw_logits():
    m = MLP()
    out = m(torch.randn(16, 30))
    assert out.shape == (16, 2)
    assert not torch.allclose(out.exp().sum(1), torch.ones(16))   # raw logits, not probabilities


def test_backward_and_finite():
    torch.manual_seed(0)
    m = MLP(dropout=0.2)
    loss = nn.CrossEntropyLoss()(m(torch.randn(8, 30)), torch.randint(0, 2, (8,)))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    torch.optim.SGD(m.parameters(), lr=0.1).step()
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_flatten_roundtrip_identical_structure():
    torch.manual_seed(0)
    a, b = MLP(), MLP()
    va = get_flat_params(a)
    set_flat_params(b, va)
    assert torch.equal(get_flat_params(b), va)
    x = torch.randn(4, 30)
    assert torch.equal(a(x), b(x))
