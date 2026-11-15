"""Presentation-ready matplotlib figures (actual experiment outputs only).

Colours come from a validated categorical palette in a fixed order. Every
series also has its own line style, marker or hatch, so colour is never the
only way to tell series apart.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
METHOD_STYLE = {
    "fedhaw_default": dict(color=SLOTS[0], ls="-",  marker="o", hatch="xx", label="FedHAW (repo defaults)"),
    "fedavg":         dict(color=SLOTS[1], ls="--", marker="s", hatch="//", label="FedAvg"),
    "fedprox":        dict(color=SLOTS[2], ls="-.", marker="^", hatch="..", label="FedProx"),
    "centralized":    dict(color=SLOTS[3], ls=":",  marker="D", hatch="",   label="Centralized"),
    "fedhaw_tuned":   dict(color=SLOTS[4], ls=(0, (5, 1, 1, 1)), marker="v", hatch="\\\\",
                           label="FedHAW (val-tuned)"),
}
DIST_LABEL = {"IID": "IID", "Dir_a10": "Dirichlet α=10", "Dir_a1": "Dirichlet α=1",
              "Dir_a0.1": "Dirichlet α=0.1"}
CLASS_COLORS = {"Benign": SLOTS[0], "Malignant": SLOTS[1]}


def label(m):
    return METHOD_STYLE.get(m, {"label": m})["label"]


def _style(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _top_legend(fig, axes, ncol):
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, ncol=ncol, loc="upper center",
               bbox_to_anchor=(0.5, 1.04))


# ---------------------------------------------------------------- data
def class_distribution(counts: dict, path, split_counts: dict | None = None):
    fig, ax = plt.subplots(figsize=(5.5 if split_counts else 4, 3.4))
    groups = ["Full dataset"] + (list(split_counts) if split_counts else [])
    vals = [counts] + ([split_counts[g] for g in split_counts] if split_counts else [])
    x = np.arange(len(groups))
    for i, cls in enumerate(["benign", "malignant"]):
        v = [d[cls] for d in vals]
        bars = ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=SLOTS[i], hatch=["", "//"][i],
                      edgecolor="white", label=cls.title())
        for b, n, d in zip(bars, v, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{n}\n({100*n/(d['benign']+d['malignant']):.0f}%)",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, groups, fontsize=9)
    ax.set_ylim(0, max(counts.values()) * 1.25)
    _style(ax, title="WDBC class distribution", ylabel="patients")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def partition_grid(tables: list, path, title):
    """tables: list of (dist_name, train_table rows)."""
    fig, axes = plt.subplots(1, len(tables), figsize=(3.7 * len(tables), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (dname, rows) in zip(axes, tables):
        x = np.arange(len(rows))
        b = np.array([r["benign"] for r in rows])
        m = np.array([r["malignant"] for r in rows])
        ax.bar(x, b, 0.7, color=SLOTS[0], edgecolor="white", label="Benign")
        ax.bar(x, m, 0.7, bottom=b, color=SLOTS[1], hatch="//", edgecolor="white", label="Malignant")
        for xi, r in zip(x, rows):
            pct = "—" if r["n"] == 0 else f"{r['malignant_pct']:.0f}% M"
            ax.text(xi, r["n"] + 2, f"n={r['n']}\n{pct}", ha="center", va="bottom", fontsize=7,
                    color=INK)
        ax.set_xticks(x, [r["client"].replace("Hospital ", "H") for r in rows], fontsize=9)
        _style(ax, title=DIST_LABEL.get(dname, dname))
    axes[0].set_ylabel("training samples", color=MUTED, fontsize=9)
    ymax = max(r["n"] for _, rows in tables for r in rows)
    axes[0].set_ylim(0, ymax * 1.25)
    _top_legend(fig, axes, 2)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left", y=1.1)
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------- curves
def curves(logs: dict, dists, methods, key, ylabel, path, ylim=None, title=None):
    """logs[(dist, method)] = list over seeds of per-round dict lists."""
    fig, axes = plt.subplots(1, len(dists), figsize=(4 * len(dists), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, dists):
        for m in methods:
            runs = logs.get((d, m))
            if not runs:
                continue
            arr = np.array([[r.get(key, np.nan) for r in run] for run in runs], float)
            mu = np.nanmean(arr, 0)
            r = np.arange(1, arr.shape[1] + 1)
            st = METHOD_STYLE[m]
            ax.plot(r, mu, color=st["color"], ls=st["ls"], lw=2, marker=st["marker"],
                    markevery=max(1, len(r) // 6), ms=5, label=st["label"])
            ax.fill_between(r, np.nanmin(arr, 0), np.nanmax(arr, 0), color=st["color"], alpha=0.12, lw=0)
        if ylim:
            ax.set_ylim(*ylim)
        _style(ax, title=DIST_LABEL.get(d, d), xlabel="communication round")
    axes[0].set_ylabel(ylabel + "  (mean; band = min–max over seeds)", color=MUTED, fontsize=9)
    _top_legend(fig, axes, len(methods))
    fig.tight_layout()
    _save(fig, path)


def gamma_plot(logs: dict, dists, method, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, d in enumerate(dists):
        runs = logs.get((d, method))
        if not runs:
            continue
        arr = np.array([[r["exp_gamma"] for r in run] for run in runs], float)
        r = np.arange(1, arr.shape[1] + 1)
        ax.plot(r, arr.mean(0), color=SLOTS[i], ls=["-", "--", "-.", ":"][i], lw=2,
                label=DIST_LABEL.get(d, d))
        ax.fill_between(r, arr.min(0), arr.max(0), color=SLOTS[i], alpha=0.1, lw=0)
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    _style(ax, title=title, xlabel="communication round", ylabel="exp(γ)  (mean; band = min–max)")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def weights_plot(runs: list, path, title):
    """runs: list of (dist, round_log, sizes)."""
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (d, log, sizes) in zip(axes, runs):
        K = len(sizes)
        W = np.array([[r[f"weight_{k+1}"] for k in range(K)] for r in log])
        frac = np.array(sizes) / sum(sizes)
        r = np.arange(1, len(W) + 1)
        for k in range(K):
            ax.plot(r, W[:, k], color=SLOTS[k], lw=2, ls=["-", "--", "-.", ":", (0, (5, 1, 1, 1))][k],
                    label=f"Hospital {k+1}")
            ax.axhline(frac[k], color=SLOTS[k], lw=0.8, ls=":")
        _style(ax, title=DIST_LABEL.get(d, d), xlabel="communication round")
    axes[0].set_ylabel("aggregation weight a_k  (dotted = N_k/N)", color=MUTED, fontsize=9)
    _top_legend(fig, axes, 5)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left", y=1.12)
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------- comparisons
def comparison_bars(summary: dict, dists, methods, metrics, path):
    """summary[(dist, method)][metric] = (mean, std)."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.8))
    axes = np.atleast_1d(axes)
    w = 0.84 / len(methods)
    x = np.arange(len(dists))
    for ax, met in zip(axes, metrics):
        lo = 1.0
        for i, m in enumerate(methods):
            mu = np.array([summary[(d, m)][met][0] for d in dists])
            sd = np.array([summary[(d, m)][met][1] for d in dists])
            lo = min(lo, np.nanmin(mu - sd))
            st = METHOD_STYLE[m]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, mu, w * 0.92, yerr=sd, color=st["color"],
                   hatch=st["hatch"], edgecolor="white", linewidth=0, label=st["label"],
                   error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
        ax.set_xticks(x, [DIST_LABEL.get(d, d).replace("Dirichlet ", "Dir ") for d in dists], fontsize=8.5)
        top = 1.005 if met != "fpr" else None
        ax.set_ylim(max(0.0, lo - 0.04) if met != "fpr" else 0, top)
        _style(ax, title={"roc_auc": "ROC-AUC", "fpr": "False-positive rate"}.get(met, met.title()))
    _top_legend(fig, axes, len(methods))
    fig.tight_layout()
    _save(fig, path)


def confusion_grid(cms: dict, dists, methods, path):
    """cms[(dist, method)] = [[tn, fp], [fn, tp]] summed over seeds."""
    fig, axes = plt.subplots(len(dists), len(methods), figsize=(2.3 * len(methods), 2.2 * len(dists)))
    for i, d in enumerate(dists):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            cm = np.array(cms[(d, m)])
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, int(cm[a, b]), ha="center", va="center", fontsize=10,
                            color="white" if cm[a, b] > cm.sum() / 2 else INK)
            ax.set_xticks([0, 1], ["B", "M"], fontsize=8)
            ax.set_yticks([0, 1], ["B", "M"], fontsize=8)
            if i == 0:
                ax.set_title(label(m), fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{DIST_LABEL.get(d, d)}\ntrue", fontsize=8, color=MUTED)
            if i == len(dists) - 1:
                ax.set_xlabel("predicted", fontsize=8, color=MUTED)
    fig.suptitle("Test confusion matrices (summed over seeds; B = benign, M = malignant)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def roc_grid(preds: dict, aucs: dict, dists, methods, path):
    """preds[(dist, method)] = list over seeds of (y, p)."""
    fig, axes = plt.subplots(1, len(dists), figsize=(3.9 * len(dists), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, dists):
        for m in methods:
            ys = np.concatenate([y for y, _ in preds[(d, m)]])
            ps = np.concatenate([p for _, p in preds[(d, m)]])
            fpr, tpr, _ = roc_curve(ys, ps)
            st = METHOD_STYLE[m]
            mu, sd = aucs[(d, m)]
            ax.plot(fpr, tpr, color=st["color"], ls=st["ls"], lw=2,
                    label=f"{st['label']} (AUC {mu:.3f}±{sd:.3f})")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
        _style(ax, title=DIST_LABEL.get(d, d), xlabel="false-positive rate", grid_axis="both")
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    axes[0].set_ylabel("true-positive rate (malignant recall)", color=MUTED, fontsize=9)
    fig.suptitle("Test ROC curves (predictions pooled over seeds; AUC = mean±std over seeds)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)


def per_client_bars(per_client: dict, dists, methods, metric, path, ylabel):
    """per_client[(dist, method)] = (K,) array of mean metric over seeds (NaN = undefined)."""
    fig, axes = plt.subplots(1, len(dists), figsize=(4.1 * len(dists), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.84 / len(methods)
    for ax, d in zip(axes, dists):
        K = len(per_client[(d, methods[0])])
        x = np.arange(K)
        for i, m in enumerate(methods):
            v = np.asarray(per_client[(d, m)], float)
            st = METHOD_STYLE[m]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, np.nan_to_num(v), w * 0.92, color=st["color"],
                   hatch=st["hatch"], edgecolor="white", linewidth=0, label=st["label"])
            for xi, vi in zip(x, v):
                if not np.isfinite(vi) and i == 0:
                    ax.text(xi, 0.02, "undefined", rotation=90, ha="center", va="bottom",
                            fontsize=7, color=MUTED)
        ax.set_xticks(x, [f"H{k+1}" for k in range(K)], fontsize=9)
        ax.set_ylim(0, 1.05)
        _style(ax, title=DIST_LABEL.get(d, d))
    axes[0].set_ylabel(ylabel, color=MUTED, fontsize=9)
    _top_legend(fig, axes, len(methods))
    fig.tight_layout()
    _save(fig, path)


def heatmap(Z, xlabels, ylabels, path, title, cbar_label, xlabel, ylabel, fmt="{:.3f}", cmap="Blues"):
    fig, ax = plt.subplots(figsize=(1.2 + 1.0 * len(xlabels), 0.9 + 0.5 * len(ylabels)))
    im = ax.imshow(Z, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=8)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            pos = float(im.norm(v)) if np.isfinite(v) else 0.0
            dark = pos < 0.5 if cmap.endswith("_r") else pos > 0.5
            ax.text(j, i, fmt.format(v) if np.isfinite(v) else "n/a", ha="center", va="center",
                    fontsize=7, color="white" if dark else INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    _save(fig, path)
