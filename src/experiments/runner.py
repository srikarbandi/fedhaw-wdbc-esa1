"""Experiment runner: one (distribution, seed) setting -> all methods.

Fairness controls within a (distribution, seed):
* identical split, scaler, hospital partition and per-hospital test subsets;
* identical initial global parameters for every method;
* identical RNG stream: the global seed is reset before each method, so local
  mini-batch shuffling starts from the same state;
* identical model, rounds, local epochs, batch size, learning rate and optimizer.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.loader import load_wdbc
from src.data.partition import make_partition, partition_table
from src.data.preprocessing import preprocess
from src.evaluation.metrics import Evaluator, classification_metrics
from src.federated.client import Client, make_optimizer
from src.federated.fedavg import FedAvg
from src.federated.fedhaw import FedHAW
from src.federated.fedprox import FedProx
from src.federated.params import get_flat_params
from src.federated.server import run_federated
from src.models.mlp import build_model
from src.utils.config import PROJECT_ROOT, save_yaml
from src.utils.logging import write_csv, write_json
from src.utils.seed import set_seed

# Original repository defaults (external/FedHAW/run_serverside.py argparse defaults)
FEDHAW_REPO_DEFAULTS = {"lr_gamma": 1e-3, "lr_lambda": 1e-2}

_DATA_CACHE: dict = {}


def get_wdbc(cfg):
    key = cfg["data"]["path"]
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_wdbc(key)
    return _DATA_CACHE[key]


def prepare(cfg: dict, dist: dict, seed: int) -> dict:
    d = get_wdbc(cfg)
    dc = cfg["data"]
    sp = preprocess(d.X, d.y, seed, dc["train_frac"], dc["val_frac"], dc["test_frac"])
    pc = cfg["partition"]
    part = make_partition(sp.y_train, sp.y_test, pc["num_clients"], dist["scheme"], seed,
                          alpha=dist.get("alpha"), min_size=pc["min_client_size"],
                          max_redraws=pc["max_redraws"])
    names = pc["client_names"]
    clients = [Client(names[k], torch.utils.data.TensorDataset(
        torch.from_numpy(sp.X_train[idx]), torch.from_numpy(sp.y_train[idx])))
        for k, idx in enumerate(part.train_parts)]
    return {"split": sp, "partition": part, "clients": clients, "names": names,
            "train_table": partition_table(sp.y_train, part.train_parts, names),
            "test_table": partition_table(sp.y_test, part.test_parts, names)}


def save_split_and_partition(prep: dict, dist: dict, seed: int, cfg: dict) -> None:
    sp, part = prep["split"], prep["partition"]
    d = get_wdbc(cfg)
    write_json({"seed": seed, "fractions": [cfg["data"]["train_frac"], cfg["data"]["val_frac"],
                                            cfg["data"]["test_frac"]],
                "train_rows": sp.idx_train, "val_rows": sp.idx_val, "test_rows": sp.idx_test,
                "train_ids": d.ids[sp.idx_train], "val_ids": d.ids[sp.idx_val],
                "test_ids": d.ids[sp.idx_test],
                "class_counts": {s: {"benign": int((yy == 0).sum()), "malignant": int((yy == 1).sum())}
                                 for s, yy in [("train", sp.y_train), ("val", sp.y_val),
                                               ("test", sp.y_test)]},
                "scaler": {"type": "StandardScaler", "fit_on": "train split only",
                           "mean": sp.scaler_mean, "scale": sp.scaler_scale}},
               PROJECT_ROOT / "data" / "processed" / f"split_seed{seed}.json")
    out = PROJECT_ROOT / "data" / "partitions"
    write_json({"distribution": dist, "seed": seed, "redraws": part.redraws, "notes": part.notes,
                "min_client_size": cfg["partition"]["min_client_size"],
                "per_class_proportions": part.proportions,
                "train_indices_into_train_split": part.train_parts,
                "test_indices_into_test_split": part.test_parts,
                "train_table": prep["train_table"], "test_table": prep["test_table"]},
               out / f"{dist['name']}_seed{seed}.json")
    write_csv([{"split": "train", **r} for r in prep["train_table"]]
              + [{"split": "test", **r} for r in prep["test_table"]],
              out / f"{dist['name']}_seed{seed}.csv")


def make_strategy(method: str, cfg: dict):
    if method == "fedavg":
        return FedAvg()
    if method == "fedprox":
        return FedProx(cfg["fedprox"]["mu_prox"])
    if method == "fedhaw":
        return FedHAW(cfg["fedhaw"]["lr_gamma"], cfg["fedhaw"]["lr_lambda"])
    if method == "fedhaw_default":
        return FedHAW(**FEDHAW_REPO_DEFAULTS)
    if method == "fedhaw_tuned":
        return FedHAW(cfg["fedhaw"]["lr_gamma"], cfg["fedhaw"]["lr_lambda"])
    raise ValueError(method)


def run_centralized(prep, cfg, init_params, model_fn, val_eval, rounds=None):
    """Same MLP/optimizer/lr/batch on the pooled training split.
    One 'round' = local_epochs passes over the data (same data passes as FL)."""
    tr = cfg["training"]
    T = int(rounds if rounds is not None else tr["rounds"])
    sp = prep["split"]
    ds = torch.utils.data.TensorDataset(torch.from_numpy(sp.X_train), torch.from_numpy(sp.y_train))
    loader = torch.utils.data.DataLoader(ds, batch_size=int(tr["batch_size"]), shuffle=True)
    model = model_fn()
    from src.federated.params import set_flat_params
    set_flat_params(model, init_params)
    opt, sched = make_optimizer(model, tr["optimizer"], float(tr["lr"]))
    crit = nn.CrossEntropyLoss()
    log = []
    for t in range(T):
        t0 = time.perf_counter()
        model.train()
        tot, nb = 0.0, 0
        for _ in range(int(tr["local_epochs"])):
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            if sched is not None:
                sched.step()
        p = get_flat_params(model)
        v = val_eval.quick(p)
        log.append({"round": t + 1, "train_loss": tot / max(1, nb),
                    "local_train_time_s": time.perf_counter() - t0, "aggregation_time_s": 0.0,
                    "param_norm": float(p.norm()), "finite": bool(torch.isfinite(p).all()),
                    "val_loss": v["loss"], "val_accuracy": v["accuracy"], "val_recall": v["recall"]})
    return get_flat_params(model), log


def client_summary(rows: list[dict]) -> dict:
    acc = np.array([r["accuracy"] for r in rows], float)
    rec = np.array([r["recall"] for r in rows], float)
    rd = rec[np.isfinite(rec)]
    return {"mean_client_accuracy": float(np.nanmean(acc)) if np.isfinite(acc).any() else np.nan,
            "mean_client_recall": float(rd.mean()) if len(rd) else np.nan,
            "worst_client_recall": float(rd.min()) if len(rd) else np.nan,
            "std_client_recall": float(rd.std()) if len(rd) else np.nan,
            "client_recall_gap": float(rd.max() - rd.min()) if len(rd) else np.nan,
            "n_clients_recall_defined": int(len(rd))}


def run_method(method: str, prep: dict, cfg: dict, seed: int, eval_on: str = "test",
               rounds=None, out_dir: Path | None = None) -> dict:
    """eval_on='test' for final experiments, 'val' for tuning (test untouched)."""
    sp = prep["split"]
    model_fn = lambda: build_model(cfg)
    set_seed(seed)
    init_params = get_flat_params(model_fn())          # identical for all methods
    val_eval = Evaluator(sp.X_val, sp.y_val, model_fn())
    set_seed(seed)                                     # identical RNG stream for all methods
    t0 = time.perf_counter()
    if method == "centralized":
        params, log = run_centralized(prep, cfg, init_params, model_fn, val_eval, rounds)
    else:
        params, log = run_federated(make_strategy(method, cfg), prep["clients"], init_params, cfg,
                                    model_fn, val_eval, rounds)
    wall = time.perf_counter() - t0
    res = {"method": method, "seed": seed, "wall_time_s": wall, "round_log": log,
           "val": val_eval.full(params)}
    if eval_on == "test":
        te = Evaluator(sp.X_test, sp.y_test, model_fn())
        prob = te.predict_proba(params)
        res["test"] = classification_metrics(sp.y_test, prob)
        res["test_prob"] = prob
        per_client = []
        for k, idx in enumerate(prep["partition"].test_parts):
            m = classification_metrics(sp.y_test[idx], prob[idx])
            per_client.append({"client": prep["names"][k],
                               "n_train": prep["clients"][k].n, **m})
        res["per_client"] = per_client
        res["client_summary"] = client_summary(per_client)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(log, out_dir / "round_log.csv")
        write_json({k: res[k] for k in ("method", "seed", "wall_time_s", "val", "test",
                                        "client_summary") if k in res}, out_dir / "final_metrics.json")
        if eval_on == "test":
            write_csv(res["per_client"], out_dir / "per_client_metrics.csv")
            write_csv([{"test_row": int(r), "y_true": int(y), "p_malignant": float(p)}
                       for r, y, p in zip(sp.idx_test, sp.y_test, res["test_prob"])],
                      out_dir / "test_predictions.csv")
    return res


def run_setting(cfg: dict, dist: dict, seed: int, methods, results_root: Path | None = None,
                save_partition: bool = True) -> dict:
    prep = prepare(cfg, dist, seed)
    if save_partition:
        save_split_and_partition(prep, dist, seed, cfg)
    out = {}
    base = results_root / dist["name"] / f"seed{seed}" if results_root else None
    if base:
        save_yaml(cfg, base / "config_resolved.yaml")
    for m in methods:
        out[m] = run_method(m, prep, cfg, seed, "test", out_dir=(base / m) if base else None)
    return {"prep": prep, "results": out}
