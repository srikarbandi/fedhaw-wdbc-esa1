"""Phase 9: tables, plots and a data-driven report from results/raw/<tag>/.

Outputs per tag:
  results/tables/<tag>_summary.csv          mean/std over seeds per distribution x method
  results/tables/<tag>_per_seed.csv         every run's test metrics + confusion counts
  results/tables/<tag>_per_client.csv       per-hospital test-subset metrics, every run
  results/tables/<tag>_client_summary.csv   mean/worst/std/gap of hospital recall
  results/tables/<tag>_paired_vs_fedavg.csv per-seed differences (descriptive only)
  results/plots/<tag>/*.png
  results/reports/<tag>_report.md
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import plots as P  # noqa: E402
from src.utils.config import load_experiments, load_yaml  # noqa: E402
from src.utils.logging import write_csv  # noqa: E402

METRICS = ["accuracy", "recall", "specificity", "fpr", "precision", "f1", "roc_auc"]
CLIENT_SUMMARY = ["mean_client_accuracy", "mean_client_recall", "worst_client_recall",
                  "std_client_recall", "client_recall_gap"]


def nz(v):
    return np.nan if v is None else v


def load_tag(tag):
    root = ROOT / "results" / "raw" / tag
    meta = load_yaml(root / "experiment_config.yaml")
    exp = load_experiments()
    dists = [d["name"] for d in exp["distributions"]]
    methods = meta["methods"]
    R = defaultdict(list)
    for d in dists:
        for s in exp["seeds"]:
            for m in methods:
                base = root / d / f"seed{s}" / m
                fm = json.loads((base / "final_metrics.json").read_text())
                R[(d, m)].append({
                    "seed": s, "final": fm,
                    "log": pd.read_csv(base / "round_log.csv").to_dict("records"),
                    "per_client": pd.read_csv(base / "per_client_metrics.csv"),
                    "pred": pd.read_csv(base / "test_predictions.csv")})
    return meta, dists, methods, exp["seeds"], R


def fmt(mu, sd):
    return "undefined" if not np.isfinite(mu) else f"{mu:.4f} ± {sd:.4f}"


def report_tag(tag):
    meta, dists, methods, seeds, R = load_tag(tag)
    plot_dir = ROOT / "results" / "plots" / tag
    plot_dir.mkdir(parents=True, exist_ok=True)
    tdir = ROOT / "results" / "tables"

    per_seed, summary_rows, client_rows, csum_rows, paired = [], [], [], [], []
    summary = {}
    cms, preds, aucs, logs, pc_recall, pc_acc = {}, {}, {}, {}, {}, {}
    for d in dists:
        for m in methods:
            runs = R[(d, m)]
            for r in runs:
                t = r["final"]["test"]
                per_seed.append({"distribution": d, "method": m, "seed": r["seed"],
                                 **{k: nz(t[k]) for k in METRICS},
                                 **{k: t[k] for k in ("tn", "fp", "fn", "tp")},
                                 **{k: nz(r["final"]["client_summary"][k]) for k in CLIENT_SUMMARY},
                                 "final_exp_gamma": r["log"][-1].get("exp_gamma", np.nan),
                                 "all_rounds_finite": all(x["finite"] for x in r["log"])})
                pcf = r["per_client"].copy()
                pcf.insert(0, "seed", r["seed"])
                pcf.insert(0, "method", m)
                pcf.insert(0, "distribution", d)
                client_rows += pcf.to_dict("records")
            summary[(d, m)] = {}
            row = {"distribution": d, "method": m, "n_seeds": len(runs)}
            for k in METRICS + CLIENT_SUMMARY:
                src = (lambda r: r["final"]["test"][k]) if k in METRICS else \
                      (lambda r: r["final"]["client_summary"][k])
                v = np.array([nz(src(r)) for r in runs], float)
                mu = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                sd = float(np.nanstd(v)) if np.isfinite(v).any() else np.nan
                summary[(d, m)][k] = (mu, sd)
                row[f"{k}_mean"], row[f"{k}_std"] = mu, sd
            row["nonfinite_runs"] = sum(not all(x["finite"] for x in r["log"]) for r in runs)
            summary_rows.append(row)
            cms[(d, m)] = [[sum(r["final"]["test"]["tn"] for r in runs), sum(r["final"]["test"]["fp"] for r in runs)],
                           [sum(r["final"]["test"]["fn"] for r in runs), sum(r["final"]["test"]["tp"] for r in runs)]]
            preds[(d, m)] = [(r["pred"]["y_true"].to_numpy(), r["pred"]["p_malignant"].to_numpy()) for r in runs]
            aucs[(d, m)] = summary[(d, m)]["roc_auc"]
            logs[(d, m)] = [r["log"] for r in runs]
            K = len(runs[0]["per_client"])
            rec = np.array([r["per_client"]["recall"].to_numpy(float) for r in runs])
            acc = np.array([r["per_client"]["accuracy"].to_numpy(float) for r in runs])
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pc_recall[(d, m)] = np.nanmean(rec, 0)
                    pc_acc[(d, m)] = np.nanmean(acc, 0)
            for k in range(K):
                csum_rows.append({"distribution": d, "method": m, "client": f"Hospital {k+1}",
                                  "seeds_recall_defined": int(np.isfinite(rec[:, k]).sum()),
                                  "mean_recall": pc_recall[(d, m)][k], "mean_accuracy": pc_acc[(d, m)][k],
                                  "mean_test_n": float(np.mean([r["per_client"]["n"][k] for r in runs])),
                                  "mean_train_n": float(np.mean([r["per_client"]["n_train"][k] for r in runs]))})
        for m in methods:
            if m == "fedavg":
                continue
            for k in ("accuracy", "recall", "specificity", "roc_auc"):
                a = np.array([nz(r["final"]["test"][k]) for r in R[(d, m)]], float)
                b = np.array([nz(r["final"]["test"][k]) for r in R[(d, "fedavg")]], float)
                diff = a - b
                paired.append({"distribution": d, "method": m, "metric": k,
                               "mean_diff_vs_fedavg": float(np.nanmean(diff)),
                               **{f"seed{s}_diff": float(x) for s, x in zip(seeds, diff)},
                               "higher": int((diff > 1e-12).sum()), "equal": int((np.abs(diff) <= 1e-12).sum()),
                               "lower": int((diff < -1e-12).sum())})

    write_csv(per_seed, tdir / f"{tag}_per_seed.csv")
    write_csv(summary_rows, tdir / f"{tag}_summary.csv")
    write_csv(client_rows, tdir / f"{tag}_per_client.csv")
    write_csv(csum_rows, tdir / f"{tag}_client_summary.csv")
    write_csv(paired, tdir / f"{tag}_paired_vs_fedavg.csv")

    # ------------------------------------------------------------ plots
    fl = [m for m in methods if m != "centralized"]
    haw = [m for m in methods if m.startswith("fedhaw")]
    P.curves(logs, dists, methods, "train_loss", "training loss", plot_dir / "training_loss.png")
    P.curves(logs, dists, methods, "val_loss", "validation loss", plot_dir / "validation_loss.png")
    P.curves(logs, dists, methods, "val_accuracy", "validation accuracy", plot_dir / "validation_accuracy.png",
             ylim=(0.3, 1.0))
    P.curves(logs, dists, methods, "val_recall", "validation malignant recall",
             plot_dir / "validation_recall.png", ylim=(0.0, 1.02))
    for h in haw:
        P.gamma_plot(logs, dists, h, plot_dir / f"{h}_gamma.png", f"{P.label(h)}: global scale exp(γ)")
        runs = []
        for d in dists:
            r0 = R[(d, h)][0]
            sizes = list(r0["per_client"]["n_train"])
            runs.append((d, r0["log"], sizes))
        P.weights_plot(runs, plot_dir / f"{h}_weights_seed{seeds[0]}.png",
                       f"{P.label(h)}: hospital aggregation weights (seed {seeds[0]})")
    P.comparison_bars(summary, dists, methods, ["accuracy", "recall", "specificity", "roc_auc"],
                      plot_dir / "comparison_main_metrics.png")
    P.comparison_bars(summary, dists, methods, ["precision", "f1", "fpr"],
                      plot_dir / "comparison_other_metrics.png")
    P.confusion_grid(cms, dists, methods, plot_dir / "confusion_matrices.png")
    P.roc_grid(preds, aucs, dists, methods, plot_dir / "roc_curves.png")
    P.per_client_bars(pc_recall, dists, methods, "recall", plot_dir / "per_client_recall.png",
                      "hospital test-subset malignant recall (mean over seeds)")
    P.per_client_bars(pc_acc, dists, methods, "accuracy", plot_dir / "per_client_accuracy.png",
                      "hospital test-subset accuracy (mean over seeds)")

    # ------------------------------------------------------------ markdown
    cfg = meta["config"]
    L = [f"# Experiment report: `{tag}`", "",
         "Generated by `scripts/generate_report.py` from `results/raw/" + tag + "/`. "
         f"Mean ± std over seeds {seeds}; test split of 86 patients per seed (54 benign / 32 malignant). "
         "Malignant = positive class.", "",
         f"Config: rounds={cfg['training']['rounds']}, local_epochs={cfg['training']['local_epochs']}, "
         f"batch={cfg['training']['batch_size']}, lr={cfg['training']['lr']}, optimizer={cfg['training']['optimizer']}, "
         f"MLP 30-{'-'.join(map(str, cfg['model']['hidden_sizes']))}-2, "
         f"FedHAW tuned (lr_gamma={cfg['fedhaw']['lr_gamma']}, lr_lambda={cfg['fedhaw']['lr_lambda']}), "
         f"FedHAW repo defaults (1e-3, 1e-2), FedProx mu_prox={cfg['fedprox']['mu_prox']}.", ""]
    for d in dists:
        L += [f"## {P.DIST_LABEL.get(d, d)}", "",
              "| Method | Accuracy | Recall | Specificity | FPR | Precision | F1 | ROC-AUC |",
              "|---|---|---|---|---|---|---|---|"]
        for m in methods:
            s = summary[(d, m)]
            L.append(f"| {P.label(m)} | " + " | ".join(fmt(*s[k]) for k in METRICS) + " |")
        L += ["", "Per-hospital (held-out hospital test subsets):", "",
              "| Method | Mean hospital acc. | Mean hospital recall | Worst hospital recall | Std hospital recall | Best–worst recall gap |",
              "|---|---|---|---|---|---|"]
        for m in methods:
            s = summary[(d, m)]
            L.append(f"| {P.label(m)} | " + " | ".join(fmt(*s[k]) for k in CLIENT_SUMMARY) + " |")
        L.append("")
    L += ["## Figures", ""] + [f"![{p.stem}](../plots/{tag}/{p.name})"
                                for p in sorted(plot_dir.glob("*.png"))]
    (ROOT / "results" / "reports" / f"{tag}_report.md").write_text("\n".join(L) + "\n")
    print(f"[{tag}] tables, {len(list(plot_dir.glob('*.png')))} plots and report written")


def main():
    tags = sys.argv[1:] or sorted(p.name for p in (ROOT / "results" / "raw").iterdir() if p.is_dir()
                                  and (p / "experiment_config.yaml").exists())
    for t in tags:
        report_tag(t)


if __name__ == "__main__":
    main()
