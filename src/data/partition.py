"""Five simulated hospitals: IID and Dirichlet label-skew partitioning.

Training split
--------------
* IID: stratified round-robin. Each class is shuffled and dealt to hospitals
  in turn, so every hospital gets about the same size and class mix.
* Dirichlet(alpha): for each class c, draw p_c ~ Dir(alpha * 1_K) and give
  hospital k the fraction p_c[k] of that class's samples (Hsu et al., 2019).
  Small alpha means strong skew in both class mix and hospital size. Draws
  are repeated until every hospital has >= min_size samples. The number of
  rejected draws is recorded. Samples are never duplicated or fabricated.

Held-out per-hospital evaluation subsets
----------------------------------------
To evaluate the final global model "per hospital" without touching training
data, the TEST split is divided among hospitals with the SAME per-class
proportions p_c (IID: the same stratified dealing). Each hospital's test
subset therefore mirrors its own training label mix. These subsets are small
(the test split has ~86 patients), and some contain only one class. Metrics
that are undefined for a subset are reported as undefined (NaN/null).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Partition:
    scheme: str
    alpha: float | None
    seed: int
    train_parts: list            # list of index arrays into the training split
    test_parts: list             # list of index arrays into the test split
    proportions: np.ndarray      # (n_classes, K) per-class proportions used
    redraws: int = 0
    notes: list = field(default_factory=list)


def _split_by_props(idx: np.ndarray, props: np.ndarray) -> list[np.ndarray]:
    cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
    return np.split(idx, cuts)


def _deal(idx: np.ndarray, K: int, offset: int) -> list[np.ndarray]:
    """Round-robin dealing starting at hospital `offset`."""
    return [idx[(np.arange(len(idx)) + offset) % K == k] for k in range(K)]


def iid_partition(y_train, y_test, K, seed) -> Partition:
    rng = np.random.default_rng(seed)
    tr = [[] for _ in range(K)]
    te = [[] for _ in range(K)]
    offset = 0
    for c in np.unique(y_train):
        idx = rng.permutation(np.where(y_train == c)[0])
        for k, part in enumerate(_deal(idx, K, offset)):
            tr[k].extend(part.tolist())
        offset = (offset + len(idx)) % K       # keep total sizes balanced
    offset = 0
    for c in np.unique(y_test):
        idx = rng.permutation(np.where(y_test == c)[0])
        for k, part in enumerate(_deal(idx, K, offset)):
            te[k].extend(part.tolist())
        offset = (offset + len(idx)) % K
    props = np.full((len(np.unique(y_train)), K), 1.0 / K)
    return Partition("iid", None, seed, [np.sort(np.array(p, dtype=np.int64)) for p in tr],
                     [np.sort(np.array(p, dtype=np.int64)) for p in te], props)


def dirichlet_partition(y_train, y_test, K, alpha, seed, min_size=5, max_redraws=10000) -> Partition:
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if K * min_size > len(y_train):
        raise ValueError("min_size infeasible")
    rng = np.random.default_rng(seed)
    classes = np.unique(y_train)
    for attempt in range(max_redraws + 1):
        props = np.stack([rng.dirichlet(alpha * np.ones(K)) for _ in classes])
        tr = [[] for _ in range(K)]
        for ci, c in enumerate(classes):
            idx = rng.permutation(np.where(y_train == c)[0])
            for k, part in enumerate(_split_by_props(idx, props[ci])):
                tr[k].extend(part.tolist())
        if min(len(p) for p in tr) >= min_size:
            break
    else:
        raise RuntimeError(f"no Dirichlet draw satisfied min_size={min_size}")
    te = [[] for _ in range(K)]
    for ci, c in enumerate(classes):
        idx = rng.permutation(np.where(y_test == c)[0])
        for k, part in enumerate(_split_by_props(idx, props[ci])):
            te[k].extend(part.tolist())
    p = Partition("dirichlet", float(alpha), seed,
                  [np.sort(np.array(t, dtype=np.int64)) for t in tr],
                  [np.sort(np.array(t, dtype=np.int64)) for t in te], props, redraws=attempt)
    if attempt:
        p.notes.append(f"{attempt} Dirichlet draw(s) rejected because a hospital had "
                       f"< {min_size} training samples")
    return p


def make_partition(y_train, y_test, K, scheme, seed, alpha=None, min_size=5,
                   max_redraws=10000) -> Partition:
    if scheme == "iid":
        return iid_partition(y_train, y_test, K, seed)
    if scheme == "dirichlet":
        return dirichlet_partition(y_train, y_test, K, alpha, seed, min_size, max_redraws)
    raise ValueError(f"unknown scheme {scheme}")


def partition_table(y: np.ndarray, parts: list, names: list) -> list[dict]:
    rows = []
    for k, idx in enumerate(parts):
        yk = y[idx]
        n = len(idx)
        b, m = int((yk == 0).sum()), int((yk == 1).sum())
        rows.append({"client": names[k], "n": n, "benign": b, "malignant": m,
                     "benign_pct": round(100 * b / n, 1) if n else None,
                     "malignant_pct": round(100 * m / n, 1) if n else None,
                     "single_class": n > 0 and (b == 0 or m == 0), "empty": n == 0})
    return rows
