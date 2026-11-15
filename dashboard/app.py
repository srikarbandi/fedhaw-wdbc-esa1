"""Read-only Streamlit dashboard for the WDBC FedHAW study.

It only DISPLAYS files already written by the pipeline (results/, data/).
It never starts training. To regenerate results, run scripts/run_all.sh.

    .venv/bin/streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RES, DATA = ROOT / "results", ROOT / "data"
DIST_LABEL = {"IID": "IID", "Dir_a10": "Dirichlet α=10", "Dir_a1": "Dirichlet α=1",
              "Dir_a0.1": "Dirichlet α=0.1"}
DISTS = list(DIST_LABEL)
METHOD_LABEL = {"centralized": "Centralized", "fedavg": "FedAvg", "fedprox": "FedProx",
                "fedhaw_default": "FedHAW (repo defaults)", "fedhaw_tuned": "FedHAW (val-tuned)"}
METHOD_COLOR = {"fedhaw_default": "#2a78d6", "fedavg": "#eb6834", "fedprox": "#1baf7a",
                "centralized": "#eda100", "fedhaw_tuned": "#e87ba4"}
CONFIG_LABEL = {"main": "Main — validation-tuned (lr 0.05)",
                "suggested_lr0.001": "Suggested initial setting (lr 0.001)"}
METRICS = ["accuracy", "recall", "specificity", "fpr", "precision", "f1", "roc_auc"]
METRIC_LABEL = {"accuracy": "Accuracy", "recall": "Malignant recall", "specificity": "Specificity",
                "fpr": "False-positive rate", "precision": "Precision", "f1": "F1", "roc_auc": "ROC-AUC"}

st.set_page_config(page_title="FedHAW on WDBC", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_data
def read_json(p):
    return json.loads(Path(p).read_text())


@st.cache_data
def read_csv(p):
    return pd.read_csv(p)


def available_configs():
    raw = RES / "raw"
    return [c for c in CONFIG_LABEL if (raw / c / "experiment_config.yaml").exists()]


@st.cache_data
def experiment_meta(cfg):
    return yaml.safe_load((RES / "raw" / cfg / "experiment_config.yaml").read_text())


@st.cache_data
def round_logs(cfg, dist, method):
    meta = experiment_meta(cfg)
    out = []
    for s in meta["seeds"]:
        f = RES / "raw" / cfg / dist / f"seed{s}" / method / "round_log.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["seed"] = s
            out.append(d)
    return pd.concat(out) if out else pd.DataFrame()


def fmt(mu, sd):
    return "undefined" if pd.isna(mu) else f"{mu:.3f} ± {sd:.3f}"


def show_png(path, caption=None):
    if Path(path).exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not generated yet: {Path(path).relative_to(ROOT)}")


def line_chart(df, y_title, colors, x_title="communication round", height=320):
    """df: index = round, one column per series. Legend on top, hover tooltip,
    no scroll-zoom (so the page scrolls normally)."""
    long = df.reset_index(names="round").melt("round", var_name="series", value_name="value")
    order = list(df.columns)
    base = alt.Chart(long).encode(
        x=alt.X("round:Q", title=x_title),
        y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", sort=order, title=None,
                        scale=alt.Scale(domain=order, range=colors),
                        legend=alt.Legend(orient="top", columns=3)),
        strokeDash=alt.StrokeDash("series:N", sort=order, legend=None),
        tooltip=["series:N", "round:Q", alt.Tooltip("value:Q", format=".4f")])
    st.altair_chart(base.mark_line(strokeWidth=2).properties(height=height), width="stretch")


def method_chart(df, y_title):
    """df: index = round, columns = method keys."""
    methods = [m for m in METHOD_LABEL if m in df.columns]
    line_chart(df[methods].rename(columns=METHOD_LABEL), y_title,
               [METHOD_COLOR[m] for m in methods])


# ------------------------------------------------------------------ guard
if not available_configs():
    st.error("No experiment results found. Run `./scripts/run_all.sh` first; "
             "this dashboard only displays existing results.")
    st.stop()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("FedHAW · WDBC")
st.sidebar.caption("Academic research prototype — not a clinically validated diagnostic system.")
PAGES = ["Project overview", "Dataset statistics", "Hospital partitions", "Algorithm comparison",
         "Training curves", "FedHAW γ and weights", "Classification metrics",
         "Confusion matrices & ROC", "Per-hospital performance"]
page = st.sidebar.radio("Section", PAGES)
configs = available_configs()
cfg = st.sidebar.selectbox("Experiment configuration", configs, format_func=CONFIG_LABEL.get)
meta = experiment_meta(cfg)
methods = meta["methods"]
st.sidebar.markdown(
    f"**Config:** {meta['config']['training']['rounds']} rounds · "
    f"{meta['config']['training']['local_epochs']} local epoch · batch "
    f"{meta['config']['training']['batch_size']} · SGD lr {meta['config']['training']['lr']}  \n"
    f"**Seeds:** {', '.join(map(str, meta['seeds']))}")

summary = read_csv(RES / "tables" / f"{cfg}_summary.csv")
per_seed = read_csv(RES / "tables" / f"{cfg}_per_seed.csv")

st.warning("Academic research prototype. Hospitals are **simulated** partitions of one public "
           "dataset. Results are not clinically validated and must not be used for diagnosis.",
           icon="⚠️")

# ------------------------------------------------------------------ pages
if page == "Project overview":
    st.title("Original FedHAW for federated breast-cancer classification")
    st.markdown(
        "Five **simulated** hospitals train a shared MLP on the Wisconsin Diagnostic Breast "
        "Cancer (WDBC) dataset without sharing patient rows. The original **FedHAW** aggregation "
        "(Nakai-Kasai & Wadayama, arXiv:2605.00458) is compared with FedAvg, FedProx and a "
        "centralized model under IID and Dirichlet non-IID splits.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", 569)
    c2.metric("Features", 30)
    c3.metric("Hospitals (simulated)", 5)
    c4.metric("Seeds", len(meta["seeds"]))
    st.subheader("FedHAW update (reproduced bit-exactly from the original code)")
    st.code("a = softmax(λ)          λ_k init = log(N_k/N),  γ init = 0\n"
            "round 0: no hypergradient step (equals FedAvg weighting)\n"
            "round t ≥ 1:\n"
            "  d   = θ^t − Σ_k a_k^(old) θ_k^t\n"
            "  γ   ← γ − lr_γ · e^γ / μ · ⟨d, θ^t⟩\n"
            "  λ_k ← λ_k − lr_λ · e^(2γ_new) · a_k(1−a_k) / μ · ⟨d, θ_k^(t−1)⟩\n"
            "θ^(t+1) = e^γ · Σ_k softmax(λ)_k θ_k^t", language="text")
    st.subheader("Key findings")
    st.markdown(
        "- **lr 0.05 (tuned):** all methods reach ≈0.95–0.98 test accuracy, even at α=0.1. "
        "Validation-tuned FedHAW gives the same predictions as FedAvg. With the repository "
        "step sizes, exp(γ) grows above 1 and the model becomes slightly overconfident and "
        "slightly less accurate.\n"
        "- **lr 0.001 (under-trained):** FedHAW's adaptive global scale γ clearly helps, most "
        "at α=0.1 (accuracy 0.86 vs 0.75 for FedAvg).\n"
        "- Hospital weights stay within 0.03 of N_k/N; FedHAW's effect here comes from γ.\n"
        "- 3 seeds and 86 test patients: differences are descriptive, not statistically tested.")
    st.caption("Details: README.md, docs/methodology.md, docs/results_discussion.md")

elif page == "Dataset statistics":
    st.title("Dataset statistics")
    rep = read_json(DATA / "processed" / "dataset_report.json")
    c1, c2, c3 = st.columns(3)
    c1.metric("Benign", f"{rep['class_counts']['benign']} ({rep['class_percent']['benign']}%)")
    c2.metric("Malignant", f"{rep['class_counts']['malignant']} ({rep['class_percent']['malignant']}%)")
    c3.metric("Missing values", rep["missing_values_total"])
    show_png(RES / "plots" / "wdbc_class_distribution.png",
             "Full dataset and the stratified 70/15/15 split (seed 42)")
    st.subheader("Verification report")
    st.dataframe(pd.DataFrame({"check": list(rep), "value": [str(v) for v in rep.values()]}),
                 hide_index=True, width="stretch")
    st.caption("Source file: breast+cancer+wisconsin+diagnostic/wdbc.data (read-only). "
               "ID column dropped; B → 0, M → 1 (malignant = positive class).")

elif page == "Hospital partitions":
    st.title("Hospital partitions")
    seed = st.selectbox("Seed", meta["seeds"])
    show_png(RES / "plots" / f"partitions_seed{seed}.png")
    dist = st.selectbox("Distribution", DISTS, format_func=DIST_LABEL.get)
    part = read_json(DATA / "partitions" / f"{dist}_seed{seed}.json")
    if part["redraws"]:
        st.info(f"{part['redraws']} Dirichlet draw(s) were rejected because a hospital had fewer "
                f"than {part['min_client_size']} training samples.")
    c1, c2 = st.columns(2)
    for col, key, title in [(c1, "train_table", "Training data"), (c2, "test_table",
                                                                   "Held-out hospital test subset")]:
        t = pd.DataFrame(part[key])
        col.subheader(title)
        col.dataframe(t[["client", "n", "benign", "malignant", "benign_pct", "malignant_pct",
                         "single_class"]], hide_index=True, width="stretch")

elif page == "Algorithm comparison":
    st.title("Algorithm comparison: IID vs non-IID")
    metric = st.selectbox("Metric", METRICS, format_func=METRIC_LABEL.get)
    tab = summary.pivot(index="method", columns="distribution", values=f"{metric}_mean")[DISTS]
    std = summary.pivot(index="method", columns="distribution", values=f"{metric}_std")[DISTS]
    disp = pd.DataFrame({DIST_LABEL[d]: [fmt(tab.loc[m, d], std.loc[m, d]) for m in methods]
                         for d in DISTS}, index=[METHOD_LABEL[m] for m in methods])
    st.dataframe(disp, width="stretch")
    long = tab.loc[methods].reset_index().melt("method", var_name="distribution", value_name="mean")
    long["std"] = [std.loc[m, d] for m, d in zip(long["method"], long["distribution"])]
    long["Method"] = long["method"].map(METHOD_LABEL)
    long["Distribution"] = long["distribution"].map(DIST_LABEL)
    order = [METHOD_LABEL[m] for m in methods]
    bars = alt.Chart(long).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("Distribution:N", sort=[DIST_LABEL[d] for d in DISTS], title=None,
                axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Method:N", sort=order),
        y=alt.Y("mean:Q", title=METRIC_LABEL[metric], scale=alt.Scale(zero=metric == "fpr")),
        color=alt.Color("Method:N", sort=order, title=None,
                        scale=alt.Scale(domain=order, range=[METHOD_COLOR[m] for m in methods]),
                        legend=alt.Legend(orient="top", columns=3)),
        tooltip=["Method:N", "Distribution:N", alt.Tooltip("mean:Q", format=".4f"),
                 alt.Tooltip("std:Q", format=".4f")])
    st.altair_chart(bars.properties(height=340), width="stretch")
    st.caption("Mean over seeds. One test patient ≈ 0.012 accuracy; one malignant case ≈ 0.031 recall.")
    show_png(RES / "plots" / cfg / "comparison_main_metrics.png", "Mean ± std over seeds")
    pv = read_csv(RES / "tables" / f"{cfg}_paired_vs_fedavg.csv")
    st.subheader("Per-seed difference vs FedAvg (descriptive)")
    st.dataframe(pv, hide_index=True, width="stretch")

elif page == "Training curves":
    st.title("Training curves (mean over seeds)")
    dist = st.selectbox("Distribution", DISTS, format_func=DIST_LABEL.get)
    key = st.selectbox("Quantity", ["val_accuracy", "val_recall", "val_loss", "train_loss"],
                       format_func={"val_accuracy": "Validation accuracy",
                                    "val_recall": "Validation malignant recall",
                                    "val_loss": "Validation loss", "train_loss": "Training loss"}.get)
    df = pd.DataFrame({m: round_logs(cfg, dist, m).groupby("round")[key].mean() for m in methods})
    method_chart(df, key.replace("_", " "))
    st.subheader("All distributions (band = min–max over seeds)")
    show_png(RES / "plots" / cfg / {"val_accuracy": "validation_accuracy.png",
                                    "val_recall": "validation_recall.png",
                                    "val_loss": "validation_loss.png",
                                    "train_loss": "training_loss.png"}[key])
    with st.expander("Timing"):
        t = pd.DataFrame({METHOD_LABEL[m]: round_logs(cfg, dist, m)[["local_train_time_s",
                                                                    "aggregation_time_s"]].mean()
                          for m in methods}).T
        st.dataframe(t.style.format("{:.5f}"), width="stretch")
        st.caption("Mean seconds per round (local training summed over hospitals).")

elif page == "FedHAW γ and weights":
    st.title("FedHAW global scale γ and hospital weights")
    haw = [m for m in methods if m.startswith("fedhaw")]
    m = st.selectbox("FedHAW variant", haw, format_func=METHOD_LABEL.get)
    df = pd.DataFrame({DIST_LABEL[d]: round_logs(cfg, d, m).groupby("round")["exp_gamma"].mean()
                       for d in DISTS})
    line_chart(df, "exp(γ) (mean over seeds)", ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
               height=300)
    c1, c2 = st.columns(2)
    dist = c1.selectbox("Distribution", DISTS, format_func=DIST_LABEL.get)
    seed = c2.selectbox("Seed", meta["seeds"])
    log = round_logs(cfg, dist, m)
    log = log[log.seed == seed].set_index("round")
    w = log[[f"weight_{k}" for k in range(1, 6)]]
    w.columns = [f"Hospital {k}" for k in range(1, 6)]
    line_chart(w, "aggregation weight a_k", ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
               height=300)
    pc = read_csv(RES / "raw" / cfg / dist / f"seed{seed}" / m / "per_client_metrics.csv")
    frac = pc["n_train"] / pc["n_train"].sum()
    st.dataframe(pd.DataFrame({"hospital": pc["client"], "N_k": pc["n_train"],
                               "N_k/N (initial weight)": frac.round(4),
                               "final weight": w.iloc[-1].to_numpy().round(4)}),
                 hide_index=True, width="stretch")

elif page == "Classification metrics":
    st.title("Test classification metrics")
    dist = st.selectbox("Distribution", DISTS, format_func=DIST_LABEL.get)
    s = summary[summary.distribution == dist].set_index("method").loc[methods]
    disp = pd.DataFrame({METRIC_LABEL[k]: [fmt(s.loc[m, f"{k}_mean"], s.loc[m, f"{k}_std"])
                                           for m in methods] for k in METRICS},
                        index=[METHOD_LABEL[m] for m in methods])
    st.dataframe(disp, width="stretch")
    st.subheader("Every run")
    ps = per_seed[per_seed.distribution == dist].copy()
    ps["method"] = ps["method"].map(METHOD_LABEL)
    st.dataframe(ps[["method", "seed"] + METRICS + ["tn", "fp", "fn", "tp"]].round(4),
                 hide_index=True, width="stretch")
    show_png(RES / "plots" / cfg / "comparison_other_metrics.png")

elif page == "Confusion matrices & ROC":
    st.title("Confusion matrices and ROC curves")
    show_png(RES / "plots" / cfg / "confusion_matrices.png")
    show_png(RES / "plots" / cfg / "roc_curves.png")

elif page == "Per-hospital performance":
    st.title("Per-hospital performance (held-out hospital test subsets)")
    st.caption("Each hospital's test subset follows its own class mix. Subsets are small (1–52 "
               "patients). Metrics with a zero denominator are undefined.")
    cs = read_csv(RES / "tables" / f"{cfg}_client_summary.csv")
    dist = st.selectbox("Distribution", DISTS, format_func=DIST_LABEL.get)
    d = cs[cs.distribution == dist].copy()
    d["method"] = d["method"].map(METHOD_LABEL)
    st.dataframe(d.drop(columns="distribution").round(3), hide_index=True, width="stretch")
    s = summary[summary.distribution == dist].set_index("method").loc[methods]
    keys = ["mean_client_accuracy", "mean_client_recall", "worst_client_recall",
            "std_client_recall", "client_recall_gap"]
    st.dataframe(pd.DataFrame({k.replace("_", " "): [fmt(s.loc[m, f"{k}_mean"], s.loc[m, f"{k}_std"])
                                                     for m in methods] for k in keys},
                              index=[METHOD_LABEL[m] for m in methods]), width="stretch")
    show_png(RES / "plots" / cfg / "per_client_recall.png")
    show_png(RES / "plots" / cfg / "per_client_accuracy.png")
