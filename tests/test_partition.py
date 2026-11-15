import numpy as np
import pytest

from src.data.partition import dirichlet_partition, make_partition, partition_table

Y_TR = np.r_[np.zeros(250, int), np.ones(148, int)]
Y_TE = np.r_[np.zeros(54, int), np.ones(32, int)]


@pytest.mark.parametrize("scheme,alpha", [("iid", None), ("dirichlet", 10.0),
                                          ("dirichlet", 1.0), ("dirichlet", 0.1)])
def test_exact_cover_five_clients(scheme, alpha):
    p = make_partition(Y_TR, Y_TE, 5, scheme, seed=42, alpha=alpha)
    for parts, n in ((p.train_parts, len(Y_TR)), (p.test_parts, len(Y_TE))):
        assert len(parts) == 5
        allidx = np.concatenate(parts)
        assert len(allidx) == n                    # every sample assigned
        assert len(np.unique(allidx)) == n         # no duplicates
    assert min(len(x) for x in p.train_parts) >= 5


def test_iid_balanced_sizes_and_class_mix():
    p = make_partition(Y_TR, Y_TE, 5, "iid", seed=1)
    sizes = [len(x) for x in p.train_parts]
    assert max(sizes) - min(sizes) <= 1
    for idx in p.train_parts:
        assert abs(Y_TR[idx].mean() - Y_TR.mean()) < 0.03


def test_same_seed_same_partition():
    a = make_partition(Y_TR, Y_TE, 5, "dirichlet", seed=7, alpha=1.0)
    b = make_partition(Y_TR, Y_TE, 5, "dirichlet", seed=7, alpha=1.0)
    assert all(np.array_equal(x, y) for x, y in zip(a.train_parts, b.train_parts))
    assert all(np.array_equal(x, y) for x, y in zip(a.test_parts, b.test_parts))


def test_heterogeneity_increases_as_alpha_decreases():
    def skew(alpha):
        v = []
        for s in range(20):
            p = dirichlet_partition(Y_TR, Y_TE, 5, alpha, s)
            v += [abs(Y_TR[i].mean() - Y_TR.mean()) for i in p.train_parts]
        return np.mean(v)
    assert skew(0.1) > skew(1.0) > skew(10.0)


def test_small_and_empty_clients_are_reported_not_hidden():
    p = dirichlet_partition(Y_TR, Y_TE, 5, 0.1, seed=3, min_size=0)   # no size floor
    tab = partition_table(Y_TR, p.train_parts, [f"H{k}" for k in range(5)])
    assert sum(r["n"] for r in tab) == len(Y_TR)
    for r in tab:
        assert r["empty"] == (r["n"] == 0)
        if r["n"]:
            assert r["benign"] + r["malignant"] == r["n"]
            assert r["single_class"] == (r["benign"] == 0 or r["malignant"] == 0)
        else:
            assert r["benign_pct"] is None
    q = dirichlet_partition(Y_TR, Y_TE, 5, 0.1, seed=3, min_size=20)
    assert min(len(x) for x in q.train_parts) >= 20
    assert q.redraws >= 0 and (q.redraws == 0 or q.notes)


def test_test_subsets_follow_training_proportions():
    p = dirichlet_partition(Y_TR, Y_TE, 5, 0.1, seed=11)
    for k in range(5):
        if len(p.test_parts[k]) >= 8:
            assert abs(Y_TE[p.test_parts[k]].mean() - Y_TR[p.train_parts[k]].mean()) < 0.35


def test_invalid_alpha():
    with pytest.raises(ValueError):
        make_partition(Y_TR, Y_TE, 5, "dirichlet", seed=0, alpha=0.0)
