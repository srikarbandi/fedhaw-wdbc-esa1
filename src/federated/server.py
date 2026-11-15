"""Server orchestration: the communication-round loop shared by all FL methods.

Per round t (same structure as run_serverside.py + train_* in the original):
  1. build one model object for the round (the original calls make_model(args)
     once per round and passes it to every client)
  2. every hospital k starts from the current global vector theta^t and trains locally
  3. the strategy aggregates the returned vectors -> theta^(t+1)
  4. the new global model is evaluated on the VALIDATION split and logged
"""

from __future__ import annotations

import time

import numpy as np
import torch.nn as nn


def run_federated(strategy, clients, init_params, cfg, model_fn, val_evaluator=None,
                  rounds=None, device="cpu"):
    tr = cfg["training"]
    T = int(rounds if rounds is not None else tr["rounds"])
    lr, E, B, opt = float(tr["lr"]), int(tr["local_epochs"]), int(tr["batch_size"]), tr["optimizer"]
    criterion = nn.CrossEntropyLoss()
    strategy.reset([c.n for c in clients])
    params = init_params.clone()
    log = []
    for t in range(T):
        model = model_fn()
        candidates, losses, nb, ltime = [], [], [], []
        for c in clients:
            p, loss_sum, n_batches, secs = c.local_train(params, model, criterion, lr, E, B, opt,
                                                         device, mu_prox=strategy.mu_prox)
            candidates.append(p)
            losses.append(loss_sum)
            nb.append(n_batches)
            ltime.append(secs)
        t0 = time.perf_counter()
        params, info = strategy.aggregate(params, candidates, t, lr)
        agg_time = time.perf_counter() - t0
        row = {"round": t + 1,
               "train_loss": float(sum(losses) / max(1, sum(nb))),   # mean mini-batch loss
               "local_train_time_s": float(sum(ltime)),
               "aggregation_time_s": float(agg_time),
               "param_norm": float(params.norm().item()),
               "finite": bool(np.isfinite(params.numpy()).all())}
        if val_evaluator is not None:
            v = val_evaluator.quick(params)
            row.update({"val_loss": v["loss"], "val_accuracy": v["accuracy"],
                        "val_recall": v["recall"]})
        if "gamma" in info:
            row["gamma"] = info["gamma"]
            row["exp_gamma"] = float(np.exp(info["gamma"]))
        for k, w in enumerate(info["weights"]):
            row[f"weight_{k+1}"] = w
        log.append(row)
    return params, log
