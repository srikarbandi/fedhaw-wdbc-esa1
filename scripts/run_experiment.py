"""Run ONE (distribution, seed) setting for the chosen methods; outputs saved to
results/raw/<tag>/<distribution>/seed<S>/<method>/.

Example:
    .venv/bin/python scripts/run_experiment.py --dist Dir_a1 --seed 42
    .venv/bin/python scripts/run_experiment.py --dist IID --seed 42 --set training.lr=0.001 --tag test
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiments.runner import run_setting  # noqa: E402
from src.utils.config import load_config, load_experiments, set_by_path  # noqa: E402
from src.utils.seed import configure_torch  # noqa: E402


def parse_sets(cfg, items):
    for it in items or []:
        k, v = it.split("=", 1)
        cfg = set_by_path(cfg, k, yaml.safe_load(v))
    return cfg


def main():
    exp = load_experiments()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True, choices=[d["name"] for d in exp["distributions"]])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--methods", nargs="+", default=exp["methods"])
    ap.add_argument("--tag", default="single")
    ap.add_argument("--set", nargs="*", help="dotted overrides, e.g. training.rounds=20")
    a = ap.parse_args()
    configure_torch()
    cfg = parse_sets(load_config(), a.set)
    dist = next(d for d in exp["distributions"] if d["name"] == a.dist)
    out = run_setting(cfg, dist, a.seed, a.methods, ROOT / "results" / "raw" / a.tag)
    for m, r in out["results"].items():
        t = r["test"]
        print(f"{m:<15} acc={t['accuracy']:.4f} recall={t['recall']:.4f} spec={t['specificity']:.4f} "
              f"auc={t['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
