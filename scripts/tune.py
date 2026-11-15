"""Phase 7: pilot runs + validation-only hyperparameter selection.

All runs evaluate on the VALIDATION split. The test split is never evaluated.
Selection criterion (fixed before running): lowest mean final-round
validation cross-entropy over the 4 distributions x 3 seeds.

Stage A  shared local learning rate (selected with FedAvg; FedHAW with the
         repository step sizes is also run at each rate as a stability pilot)
Stage B  FedHAW (lr_gamma, lr_lambda) at the selected learning rate
Stage C  FedProx mu_prox at the selected learning rate

Writes configs/selected.yaml, results/tables/tuning_*.csv,
results/plots/tuning_*.png, results/reports/tuning_summary.json
"""

import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import plots as P  # noqa: E402
from src.experiments.runner import FEDHAW_REPO_DEFAULTS, prepare, run_method  # noqa: E402
from src.utils.config import load_config, load_experiments, save_yaml, set_by_path  # noqa: E402
from src.utils.logging import write_csv, write_json  # noqa: E402
from src.utils.seed import configure_torch  # noqa: E402


def evaluate(cfg, method, preps, seeds):
    rows = []
    for (dname, seed), prep in preps.items():
        r = run_method(method, prep, cfg, seed, eval_on="val")
        log = r["round_log"]
        rows.append({"dist": dname, "seed": seed, "val_loss": log[-1]["val_loss"],
                     "val_acc": r["val"]["accuracy"], "val_recall": r["val"]["recall"],
                     "finite": all(x["finite"] for x in log),
                     "max_abs_gamma": max((abs(x.get("gamma", 0.0)) for x in log), default=0.0),
                     "final_exp_gamma": log[-1].get("exp_gamma", np.nan)})
    vl = np.array([r["val_loss"] for r in rows], float)
    return {"val_loss": float(np.mean(vl)) if np.isfinite(vl).all() else float("inf"),
            "val_acc": float(np.mean([r["val_acc"] for r in rows])),
            "val_recall": float(np.nanmean([r["val_recall"] for r in rows])),
            "all_finite": all(r["finite"] for r in rows),
            "max_abs_gamma": float(max(r["max_abs_gamma"] for r in rows)),
            "mean_final_exp_gamma": (float(np.nanmean(eg)) if np.isfinite(
                eg := np.array([r["final_exp_gamma"] for r in rows], float)).any() else float("nan"))}, rows


def main():
    configure_torch()
    base = load_config()
    if (ROOT / "configs" / "selected.yaml").exists():
        print("NOTE: configs/selected.yaml exists and is ignored during tuning")
        from src.utils.config import CONFIG_DIR, load_yaml
        base = load_yaml(CONFIG_DIR / "default.yaml")
    exp = load_experiments()
    grid = exp["tuning"]
    seeds = exp["seeds"]
    preps = {(d["name"], s): prepare(base, d, s) for d in exp["distributions"] for s in seeds}
    summary = {"criterion": "min mean final-round validation cross-entropy (4 distributions x 3 seeds)"}
    t_all = time.perf_counter()

    print("Stage A: shared learning rate (FedAvg selects; FedHAW repo-defaults = stability pilot)")
    rowsA = []
    for lr in grid["shared_lr"]:
        cfg = set_by_path(base, "training.lr", lr)
        for m in ("fedavg", "fedhaw_default"):
            s, _ = evaluate(cfg, m, preps, seeds)
            rowsA.append({"method": m, "lr": lr, **s})
            print(f"  lr={lr:<6} {m:<15} val_loss={s['val_loss']:.4f} val_acc={s['val_acc']:.4f} "
                  f"val_recall={s['val_recall']:.4f} finite={s['all_finite']} "
                  f"max|gamma|={s['max_abs_gamma']:.3g}")
    best_lr = min((r for r in rowsA if r["method"] == "fedavg"), key=lambda r: r["val_loss"])["lr"]
    cfg = set_by_path(base, "training.lr", best_lr)
    print(f"  -> selected lr = {best_lr}")

    print("Stage B: FedHAW (lr_gamma, lr_lambda)")
    rowsB = []
    for lg, ll in itertools.product(grid["fedhaw_lr_gamma"], grid["fedhaw_lr_lambda"]):
        c = set_by_path(set_by_path(cfg, "fedhaw.lr_gamma", lg), "fedhaw.lr_lambda", ll)
        s, _ = evaluate(c, "fedhaw", preps, seeds)
        rowsB.append({"lr_gamma": lg, "lr_lambda": ll, **s})
        print(f"  lr_gamma={lg:<8g} lr_lambda={ll:<6g} val_loss={s['val_loss']:.4f} "
              f"val_acc={s['val_acc']:.4f} recall={s['val_recall']:.4f} finite={s['all_finite']} "
              f"final exp(gamma)={s['mean_final_exp_gamma']:.4f}")
    bestB = min(rowsB, key=lambda r: (r["val_loss"], -r["val_acc"]))

    print("Stage C: FedProx mu_prox")
    rowsC = []
    for mp in grid["fedprox_mu_prox"]:
        s, _ = evaluate(set_by_path(cfg, "fedprox.mu_prox", mp), "fedprox", preps, seeds)
        rowsC.append({"mu_prox": mp, **s})
        print(f"  mu_prox={mp:<6g} val_loss={s['val_loss']:.4f} val_acc={s['val_acc']:.4f} "
              f"recall={s['val_recall']:.4f}")
    bestC = min(rowsC, key=lambda r: (r["val_loss"], -r["val_acc"]))

    fedavg_ref = next(r for r in rowsA if r["method"] == "fedavg" and r["lr"] == best_lr)
    repo = next((r for r in rowsB if r["lr_gamma"] == FEDHAW_REPO_DEFAULTS["lr_gamma"]
                 and r["lr_lambda"] == FEDHAW_REPO_DEFAULTS["lr_lambda"]), None)
    selected = {"selected_by": "scripts/tune.py (validation split only)",
                "criterion": summary["criterion"],
                "overrides": {"training": {"lr": best_lr},
                              "fedhaw": {"lr_gamma": bestB["lr_gamma"], "lr_lambda": bestB["lr_lambda"]},
                              "fedprox": {"mu_prox": bestC["mu_prox"]}}}
    save_yaml(selected, ROOT / "configs" / "selected.yaml")
    write_csv(rowsA, ROOT / "results" / "tables" / "tuning_stageA_lr.csv")
    write_csv(rowsB, ROOT / "results" / "tables" / "tuning_stageB_fedhaw.csv")
    write_csv(rowsC, ROOT / "results" / "tables" / "tuning_stageC_fedprox.csv")
    summary.update({"selected": selected["overrides"], "fedavg_at_selected_lr": fedavg_ref,
                    "fedhaw_repo_defaults_at_selected_lr": repo, "fedhaw_best": bestB,
                    "fedprox_best": bestC, "runtime_s": time.perf_counter() - t_all})
    write_json(summary, ROOT / "results" / "reports" / "tuning_summary.json")

    gs, ls = grid["fedhaw_lr_gamma"], grid["fedhaw_lr_lambda"]
    for key, fname, title, cmap, fmt in [
            ("val_loss", "tuning_fedhaw_val_loss.png",
             f"FedHAW validation cross-entropy (FedAvg: {fedavg_ref['val_loss']:.4f})", "Blues_r", "{:.4f}"),
            ("val_acc", "tuning_fedhaw_val_accuracy.png",
             f"FedHAW validation accuracy (FedAvg: {fedavg_ref['val_acc']:.4f})", "Blues", "{:.3f}")]:
        Z = np.array([[next(r[key] for r in rowsB if r["lr_gamma"] == g and r["lr_lambda"] == l)
                       for l in ls] for g in gs])
        P.heatmap(Z, [f"{v:g}" for v in ls], [f"{v:g}" for v in gs],
                  ROOT / "results" / "plots" / fname, title, key.replace("_", " "), "lr_lambda",
                  "lr_gamma", fmt=fmt, cmap=cmap)
    print("Selected:", selected["overrides"], f"({summary['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
