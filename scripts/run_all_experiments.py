"""Phase 8: full experiments, run sequentially (low memory) on CPU.

Configurations ("tags"):
  main             configs/default.yaml + configs/selected.yaml (validation-tuned)
  suggested_lr0.001  the brief's suggested initial setting (local lr = 0.001),
                     with FedHAW at the repository's default step sizes

Raw outputs: results/raw/<tag>/<dist>/seed<S>/<method>/
Then run scripts/generate_report.py.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiments.runner import run_setting  # noqa: E402
from src.utils.config import load_config, load_experiments, save_yaml, set_by_path  # noqa: E402
from src.utils.logging import write_json  # noqa: E402
from src.utils.seed import configure_torch  # noqa: E402

CONFIGS = {
    "main": {"overrides": {}, "methods": None},
    "suggested_lr0.001": {"overrides": {"training.lr": 0.001},
                          "methods": ["centralized", "fedavg", "fedprox", "fedhaw_default"]},
}


def main(tags=None):
    configure_torch()
    exp = load_experiments()
    for tag in tags or CONFIGS:
        spec = CONFIGS[tag]
        cfg = load_config()
        for k, v in spec["overrides"].items():
            cfg = set_by_path(cfg, k, v)
        methods = spec["methods"] or exp["methods"]
        root = ROOT / "results" / "raw" / tag
        save_yaml({"config": cfg, "methods": methods, "seeds": exp["seeds"],
                   "distributions": exp["distributions"]}, root / "experiment_config.yaml")
        t0 = time.perf_counter()
        index = []
        for dist in exp["distributions"]:
            for seed in exp["seeds"]:
                out = run_setting(cfg, dist, seed, methods, root)
                line = " | ".join(f"{m} {r['test']['accuracy']:.3f}/{r['test']['recall']:.3f}"
                                  for m, r in out["results"].items())
                print(f"[{tag}] {dist['name']:<9} seed {seed:<5} acc/recall: {line}", flush=True)
                index.append({"dist": dist["name"], "seed": seed, "methods": methods})
        write_json({"tag": tag, "runs": index, "runtime_s": time.perf_counter() - t0},
                   root / "index.json")
        print(f"[{tag}] done in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
