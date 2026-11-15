"""Execution-order checks 1-7 (before any full experiment). VALIDATION data only.

1 dataset smoke test        2 preprocessing test       3 client partition test
4 one-round FedAvg          5 one-round FedHAW         6 short FedHAW run
7 short baseline comparison
Writes results/reports/smoke_test.json.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiments.runner import get_wdbc, prepare, run_method  # noqa: E402
from src.federated.aggregation import fedhaw_init, softmax_vec  # noqa: E402
from src.utils.config import load_config, load_experiments  # noqa: E402
from src.utils.logging import write_json  # noqa: E402
from src.utils.seed import configure_torch  # noqa: E402


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(1)
    return {"check": name, "passed": bool(cond), "detail": str(detail)}


def main(lr=None):
    configure_torch()
    over = {"training": {"lr": lr}} if lr else {}
    cfg = load_config(over)
    exp = load_experiments()
    seed = exp["seeds"][0]
    out = {"config_lr": cfg["training"]["lr"], "checks": [], "runs": {}}
    C = out["checks"]

    d = get_wdbc(cfg)                                                       # 1
    C.append(check("1 dataset", d.X.shape == (569, 30) and set(np.unique(d.y)) == {0, 1},
                   f"X={d.X.shape}, malignant={int(d.y.sum())}"))
    dist0 = exp["distributions"][0]
    prep = prepare(cfg, dist0, seed)                                         # 2
    sp = prep["split"]
    n = (len(sp.y_train), len(sp.y_val), len(sp.y_test))
    C.append(check("2 preprocessing", sum(n) == 569 and abs(sp.X_train.mean()) < 1e-4,
                   f"train/val/test={n}"))
    for dist in exp["distributions"]:                                        # 3
        p = prepare(cfg, dist, seed)
        allidx = np.concatenate(p["partition"].train_parts)
        C.append(check(f"3 partition {dist['name']}", len(allidx) == len(np.unique(allidx)) == n[0],
                       [r["n"] for r in p["train_table"]]))

    for m, T in [("fedavg", 1), ("fedhaw_default", 1)]:                     # 4, 5
        t0 = time.perf_counter()
        r = run_method(m, prep, cfg, seed, eval_on="val", rounds=T)
        row = r["round_log"][0]
        w = [row[f"weight_{k}"] for k in range(1, 6)]
        sizes = [c.n for c in prep["clients"]]
        ok = row["finite"] and abs(sum(w) - 1) < 1e-9
        if m == "fedhaw_default":
            ok = ok and row["gamma"] == 0.0 and np.allclose(w, softmax_vec(fedhaw_init(sizes)[1]))
        C.append(check(f"{'4' if m == 'fedavg' else '5'} one-round {m}", ok,
                       f"val_acc={row['val_accuracy']:.3f}, weights={np.round(w, 3).tolist()}, "
                       f"{time.perf_counter()-t0:.2f}s"))

    r = run_method("fedhaw_default", prep, cfg, seed, eval_on="val", rounds=10)   # 6
    g = [x["gamma"] for x in r["round_log"]]
    C.append(check("6 short FedHAW (10 rounds)", all(x["finite"] for x in r["round_log"]),
                   f"gamma[0]={g[0]}, gamma[-1]={g[-1]:.3e}, val_acc={r['round_log'][-1]['val_accuracy']:.3f}"))
    out["runs"]["fedhaw_10"] = r["round_log"]

    print("\n7 short baseline comparison (10 rounds, IID + Dir 0.1, validation set):")   # 7
    for dist in (exp["distributions"][0], exp["distributions"][-1]):
        p = prepare(cfg, dist, seed)
        for m in ["centralized", "fedavg", "fedprox", "fedhaw_default"]:
            r = run_method(m, p, cfg, seed, eval_on="val", rounds=10)
            v = r["val"]
            print(f"   {dist['name']:<9} {m:<15} val_acc={v['accuracy']:.3f} recall={v['recall']:.3f} "
                  f"val_loss={r['round_log'][-1]['val_loss']:.4f} finite={r['round_log'][-1]['finite']}")
            out["runs"][f"{dist['name']}_{m}"] = {"val": v, "last": r["round_log"][-1]}
    C.append(check("7 short baseline comparison", True))
    write_json(out, ROOT / "results" / "reports" / f"smoke_test_lr{cfg['training']['lr']}.json")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else None)
