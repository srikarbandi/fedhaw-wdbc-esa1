# Original FedHAW for Federated Breast-Cancer Classification on WDBC

Academic capstone project. **Research prototype, not a clinically validated diagnostic system.**

## Objective

Implement the **original FedHAW** algorithm (Federated learning with
Hypergradient-based Adaptive Weights; Nakai-Kasai & Wadayama, arXiv:2605.00458) on the
Wisconsin Diagnostic Breast Cancer dataset. Five *simulated* hospitals jointly train
a classifier without sharing patient rows. FedHAW is compared with FedAvg, FedProx and
a centralized MLP under IID and Dirichlet non-IID data.

**Why federated learning?** Hospitals often cannot pool patient records for privacy,
legal and governance reasons. In federated learning each site trains on its own data
and only model parameters travel to a server, which aggregates them. When sites differ
in size and case mix (non-IID), the aggregation rule matters. FedHAW learns the
aggregation online: per-hospital weights through λ, and a global scale through γ.

## Repository layout

```
capstone/
├── breast+cancer+wisconsin+diagnostic/  WDBC data from UCI (wdbc.data, wdbc.names; read-only)
├── external/FedHAW/                     original FedHAW code, fetched by scripts/get_original_fedhaw.sh
├── configs/  default.yaml · experiments.yaml · selected.yaml (written by tune.py)
├── src/
│   ├── data/        loader.py (load + verify) · preprocessing.py (split, scaler) · partition.py
│   ├── models/      mlp.py (30→64→32→2)
│   ├── federated/   client.py · server.py · aggregation.py · params.py · fedavg.py · fedprox.py · fedhaw.py
│   ├── evaluation/  metrics.py · plots.py
│   ├── experiments/ runner.py
│   └── utils/       config.py · seed.py · logging.py
├── scripts/  prepare_data.py · run_smoke_test.py · tune.py · run_experiment.py
│             run_all_experiments.py · generate_report.py · run_all.sh
├── tests/    test_data · test_partition · test_model · test_fedavg · test_fedhaw · test_metrics
├── data/     processed/ (dataset report, split indices, scaler params) · partitions/
├── results/  raw/ · tables/ · plots/ · reports/
├── docs/     fedhaw_reproduction_notes.md · methodology.md · results_discussion.md
├── dashboard/app.py                     optional read-only Streamlit dashboard
└── IMPLEMENTATION_STATUS.md
```

## Dataset

`breast+cancer+wisconsin+diagnostic/wdbc.data`, the original UCI file: comma-separated,
no header, 569 rows × 32 columns (ID, diagnosis M/B, 30 real-valued nuclear features
computed from fine-needle-aspirate images; no images are included).
`src/data/loader.py` drops the ID column, maps B = 0 and M = 1 (malignant is the
positive class), and verifies 569 samples, 30 numeric features, 2 classes (357 / 212),
no missing values and no duplicate rows, IDs or feature vectors. The results are in
`data/processed/dataset_report.json`.
Source: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>.

## Method summary

* **Preprocessing.** Stratified 70/15/15 train/validation/test split per seed (397/86/86).
  `StandardScaler` is fitted on the training split only. The test split is used only for
  final evaluation.
* **Hospitals.** K = 5. IID uses stratified round-robin dealing. Non-IID uses a per-class
  Dirichlet split with α ∈ {10, 1, 0.1}; draws are repeated until every hospital has ≥ 5
  samples, and the number of rejected draws is recorded. Each hospital also receives a
  held-out slice of the *test* split with the same class proportions, used for
  per-hospital evaluation.
* **Model.** MLP 30 → 64 → 32 → 2 (ReLU), raw logits + CrossEntropyLoss, SGD.
* **FedHAW (original).** Round 0 is FedAvg-weighted. From round 1 on, the server updates
  γ and then λ by hypergradient descent using the current global vector, the current
  client vectors and the *previous* round's client vectors. It then aggregates:

  ```
  a = softmax(λ)                       λ_k init = log(N_k/N),  γ init = 0
  d = θ^t − Σ_k a_k^(old) θ_k^t
  γ   ← γ   − lr_γ · e^γ / μ · ⟨d, θ^t⟩
  λ_k ← λ_k − lr_λ · e^(2γ_new) · a_k(1−a_k) / μ · ⟨d, θ_k^(t−1)⟩
  θ^(t+1) = e^γ · Σ_k softmax(λ)_k θ_k^t
  ```
  The code is a statement-by-statement port of the original `train_fedhaw`, and it is
  **tested bit-exact against the original repository**. See
  [docs/fedhaw_reproduction_notes.md](docs/fedhaw_reproduction_notes.md).
* **Baselines.** FedAvg (weights N_k/N); FedProx (proximal term μ_prox/2·‖θ−θ_global‖²,
  weights N_k/N); centralized (same MLP on pooled training data, same number of epochs).
* **Protocol.** 50 rounds, 1 local epoch, batch 16, seeds 42/123/2026. Within a seed,
  every method shares the split, partition, initial weights and shuffling RNG.
  Hyperparameters are chosen on the validation split only.
* **Metrics.** Accuracy, malignant recall, specificity, FPR, precision, F1, ROC-AUC
  (from P(malignant)), confusion matrices, and per-hospital metrics with
  mean / worst / std / gap of hospital recall. Undefined metrics are reported as undefined.

Full details and references: [docs/methodology.md](docs/methodology.md).

## Installation

Requires Python 3.11 (other recent 3.x versions should work) and git. The virtual
environment is about 1.1 GB, mainly PyTorch. Run these from inside this folder:

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Then fetch the original FedHAW code (about 400 KB) for the bit-exact equivalence tests:

```bash
./scripts/get_original_fedhaw.sh
```

On Windows, use `.venv\Scripts\python` in place of `.venv/bin/python`, and run the steps
of `scripts/run_all.sh` one at a time.

## Running

Run the tests (45 tests, including the bit-exact comparison with the original FedHAW):

```bash
.venv/bin/python -m pytest -q
```

Run the whole pipeline (data → smoke tests → tuning → experiments → report):

```bash
./scripts/run_all.sh
```

Or run the steps individually:

```bash
.venv/bin/python scripts/prepare_data.py
```

```bash
.venv/bin/python scripts/run_smoke_test.py
```

```bash
.venv/bin/python scripts/tune.py
```

```bash
.venv/bin/python scripts/run_all_experiments.py
```

```bash
.venv/bin/python scripts/generate_report.py
```

Run a single setting with overrides:

```bash
.venv/bin/python scripts/run_experiment.py --dist Dir_a0.1 --seed 42 --set training.rounds=100
```

## Viewing results

* `results/reports/main_report.md` and `results/reports/suggested_lr0.001_report.md`: all tables and figures
* `docs/results_discussion.md`: interpretation
* `results/tables/*.csv`: summary, per-seed, per-hospital, paired differences, tuning grids, partitions
* `results/plots/`: class and partition distributions, tuning heatmaps; `results/plots/<config>/` holds
  loss, accuracy and recall curves, γ, weights, comparison bars, confusion matrices, ROC curves and per-hospital charts
* `results/raw/<config>/<dist>/seed<S>/<method>/`: `round_log.csv` (per-round training loss, validation
  loss/accuracy/recall, γ, weights, local and aggregation time), `final_metrics.json`,
  `per_client_metrics.csv`, `test_predictions.csv`

## Headline results (test set, mean ± std over 3 seeds)

| Config / distribution | Centralized | FedAvg | FedProx | FedHAW (repo defaults) | FedHAW (val-tuned) |
|---|---|---|---|---|---|
| lr 0.05, IID – accuracy | 0.965 ± 0.016 | 0.973 ± 0.020 | 0.973 ± 0.020 | 0.965 ± 0.016 | 0.973 ± 0.020 |
| lr 0.05, α=0.1 – accuracy | 0.965 ± 0.016 | 0.973 ± 0.011 | 0.973 ± 0.011 | 0.961 ± 0.011 | 0.973 ± 0.011 |
| lr 0.05, α=0.1 – recall | 0.958 ± 0.015 | 0.948 ± 0.030 | 0.948 ± 0.030 | 0.958 ± 0.015 | 0.948 ± 0.030 |
| lr 0.001, α=0.1 – accuracy | 0.915 ± 0.031 | 0.748 ± 0.088 | 0.748 ± 0.088 | 0.860 ± 0.000 | — |
| lr 0.001, α=0.1 – recall | 0.771 ± 0.082 | 0.344 ± 0.259 | 0.344 ± 0.259 | 0.698 ± 0.082 | — |

* **lr 0.05 (well-tuned local training).** Every method scores about 0.95–0.98 under
  every distribution. Validation-tuned FedHAW gives the same predictions as FedAvg. With
  the repository's step sizes, FedHAW grows exp(γ) above 1 and becomes slightly
  overconfident and slightly less accurate.
* **lr 0.001 (under-trained).** FedHAW's adaptive global scale γ clearly helps,
  especially at α = 0.1.
* With 86 test patients and 3 seeds, these are descriptive findings, not statistically
  significant ones. See [docs/results_discussion.md](docs/results_discussion.md).

## Dashboard (optional)

A read-only Streamlit dashboard (`dashboard/app.py`) shows the saved results. It has
sections for the overview, dataset statistics, hospital partitions, the IID vs non-IID
comparison, training curves, FedHAW γ and weights, metrics, confusion matrices and ROC
curves, and per-hospital performance. Both experiment configurations can be selected.
It never starts training. Install it once into the project venv:

```bash
.venv/bin/python -m pip install streamlit==1.67.0
```

```bash
.venv/bin/streamlit run dashboard/app.py
```

Then open <http://localhost:8501>. `.streamlit/config.toml` binds the dashboard to
localhost only.

## Reproducibility

Fixed seeds (Python, NumPy, PyTorch), single-threaded deterministic CPU execution,
saved split indices and partitions, the resolved config saved with every run, and
sequential execution. Re-running the pipeline regenerates identical tables.

## Limitations

The dataset is small and nearly linearly separable. The hospitals are simulated from
one data source, with label and quantity skew only. Only 3 seeds were used, so no
significance testing is possible. Hospital test subsets are tiny (1–52 patients). Models
are evaluated at the final round with a fixed threshold. The work is not clinically
validated.

## Out of scope

No CA-FedHAW, no clinical-aware or recall/FPR-based aggregation, no feature selection,
no image models or image datasets, and no new aggregation algorithm.
