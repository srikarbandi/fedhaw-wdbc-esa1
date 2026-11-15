"""Phase 2-3: verify WDBC, create splits and hospital partitions, plot distributions.

Writes:
  data/processed/dataset_report.json      verification results
  data/processed/split_seed{S}.json       row indices, IDs, class counts, scaler params
  data/partitions/{dist}_seed{S}.json/.csv  hospital assignments and class tables
  results/tables/partitions.csv           all partition tables
  results/plots/wdbc_class_distribution.png, partitions_seed{S}.png
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import plots as P  # noqa: E402
from src.experiments.runner import get_wdbc, prepare, save_split_and_partition  # noqa: E402
from src.utils.config import load_config, load_experiments  # noqa: E402
from src.utils.logging import write_csv, write_json  # noqa: E402


def main():
    cfg = load_config()
    exp = load_experiments()
    d = get_wdbc(cfg)
    write_json(d.report, ROOT / "data" / "processed" / "dataset_report.json")
    print("Dataset verification:")
    for k in ("n_samples", "n_features", "raw_diagnosis_values", "label_mapping", "class_counts",
              "class_percent", "missing_values_total", "duplicate_rows_full", "duplicate_ids",
              "duplicate_feature_vectors", "feature_dtypes", "verified"):
        print(f"  {k}: {d.report[k]}")

    (ROOT / "results" / "plots").mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in exp["seeds"]:
        tables = []
        for dist in exp["distributions"]:
            prep = prepare(cfg, dist, seed)
            save_split_and_partition(prep, dist, seed, cfg)
            part = prep["partition"]
            tables.append((dist["name"], prep["train_table"]))
            print(f"\nseed {seed} | {dist['name']} | rejected draws: {part.redraws}")
            print(f"  {'client':<11}{'n':>4}{'benign':>8}{'malig.':>8}{'%B':>7}{'%M':>7}   test n (B/M)")
            for r, t in zip(prep["train_table"], prep["test_table"]):
                flag = "  <- single class" if r["single_class"] else ""
                print(f"  {r['client']:<11}{r['n']:>4}{r['benign']:>8}{r['malignant']:>8}"
                      f"{r['benign_pct']:>7}{r['malignant_pct']:>7}   {t['n']:>3} ({t['benign']}/{t['malignant']}){flag}")
                rows.append({"seed": seed, "distribution": dist["name"], "split": "train", **r})
                rows.append({"seed": seed, "distribution": dist["name"], "split": "test", **t})
            if seed == exp["seeds"][0] and dist is exp["distributions"][0]:
                sp = prep["split"]
                split_counts = {s: {"benign": int((y == 0).sum()), "malignant": int((y == 1).sum())}
                                for s, y in [("Train", sp.y_train), ("Validation", sp.y_val),
                                             ("Test", sp.y_test)]}
                P.class_distribution(d.report["class_counts"],
                                     ROOT / "results" / "plots" / "wdbc_class_distribution.png",
                                     split_counts)
        P.partition_grid(tables, ROOT / "results" / "plots" / f"partitions_seed{seed}.png",
                         f"Hospital training-set class distributions (seed {seed})")
    write_csv(rows, ROOT / "results" / "tables" / "partitions.csv")
    print("\nSaved splits, partitions, tables and plots.")


if __name__ == "__main__":
    main()
