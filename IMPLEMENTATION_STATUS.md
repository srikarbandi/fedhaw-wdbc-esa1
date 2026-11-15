# Implementation status

Last updated: 2026-11-15. **State: complete — Phases 0–10 and the optional dashboard.**

## Phase 0: inspect existing folder ✅
> *Shared copy:* this folder was packaged from the project workspace for the team. The
> archived v1 code, the FedHAW clone and personal files mentioned below are not included;
> run `scripts/get_original_fedhaw.sh` to fetch the original FedHAW code.

* Working directory: `~/Desktop/capstone`.
* Dataset: `breast+cancer+wisconsin+diagnostic/wdbc.data` (original UCI CSV, no header,
  569 × 32) and `wdbc.names`. The `.zip` is identical to the extracted copy. Neither was modified;
  a test checks the SHA-256 is unchanged.
* Preserved: both Review 1 `.pptx` files, the briefing `.pdf` and the two `.docx` plans (untouched).
* An earlier version, built from a truncated brief (different split, α values, model and
  layout), was moved, not deleted, to `archive/v1_initial_run/`.
* Environment: the conda base (Python 3.13) was left unchanged. Created `.venv` with
  Homebrew Python 3.11.16 (`requirements.txt`).

## Phase 1: original FedHAW inspection ✅
* Cloned to `external/FedHAW` (commit 6f56552). Read README, run_serverside.py and
  functions/{training, networks, datasets, utils}.py.
* Notes: `docs/fedhaw_reproduction_notes.md` (equations, update order, new-γ-in-λ,
  previous candidates, round-0 skip, global scaling, μ dependence).

## Phase 2: data pipeline ✅
* `src/data/loader.py` (verification report), `src/data/preprocessing.py` (70/15/15
  stratified; scaler fitted on the training split only).
* Output: `data/processed/dataset_report.json`, `split_seed{42,123,2026}.json`.

## Phase 3: client partitioning ✅
* `src/data/partition.py`: IID and Dirichlet α ∈ {10, 1, 0.1}; min 5 samples per hospital,
  with rejected draws recorded (α = 0.1: 2 / 14 / 7 for seeds 42 / 123 / 2026); per-hospital
  test subsets.
* Output: `data/partitions/*.json|csv`, `results/tables/partitions.csv`,
  `results/plots/partitions_seed*.png`, `results/reports/prepare_data_stdout.txt`.

## Phase 4: model ✅
* `src/models/mlp.py`: 30→64→32→2, ReLU, optional dropout (default 0), raw logits.
  4,130 parameters.

## Phase 5: federated baselines ✅
* `src/federated/{client,server,aggregation,params,fedavg,fedprox}.py`. FedAvg is
  bit-exact against the original `train_fedavg`. FedProx uses N_k/N weights (documented
  deviation from the original's batch-count weights).

## Phase 6: original FedHAW ✅
* `src/federated/aggregation.py::fedhaw_update`, `src/federated/fedhaw.py`.
* `tests/test_fedhaw.py` (12 tests): checks 1–11 from the brief, plus bit-exact
  equivalence with the original `train_fedhaw` over 6 rounds at two step-size settings,
  plus a synthetic-candidate comparison of the server step.

## Tests ✅
`.venv/bin/python -m pytest -q` → **45 passed** (data 9, partition 10, model 4, FedAvg/FedProx 5,
FedHAW 12, metrics 5).

## Phase 7: pilot + tuning ✅
* `scripts/run_smoke_test.py`: execution-order steps 1–7 all PASS.
* `scripts/tune.py` (validation only): lr 0.05, FedHAW lr_γ 1e-5 / lr_λ 1e-2, FedProx
  μ_prox 0.01 → `configs/selected.yaml`. The pilot showed lr = 0.001 is under-trained after
  50 rounds.

## Phase 8: full experiments ✅
* `scripts/run_all_experiments.py`: 2 configurations × 4 distributions × 3 seeds,
  all methods, sequential CPU.
  * `main` (tuned): centralized, FedAvg, FedProx, FedHAW (repo defaults), FedHAW (tuned).
  * `suggested_lr0.001`: centralized, FedAvg, FedProx, FedHAW (repo defaults).
* All rounds finite (no NaN/inf) in every run.

## Phase 9: analysis ✅
* `scripts/generate_report.py` → `results/tables/`, `results/plots/<config>/` (26 figures),
  `results/reports/<config>_report.md`.
* Full pipeline (`scripts/run_all.sh`) takes 2 min 37 s; a re-run produced byte-identical tables.

## Phase 10: documentation ✅
README.md, docs/methodology.md (with references), docs/fedhaw_reproduction_notes.md,
docs/results_discussion.md, this file.

## Known issues / caveats
* With 3 seeds and 86 test patients, differences between methods are descriptive only.
* Hospital test subsets are tiny; some per-hospital metrics are undefined (reported as null/NaN).
* With the tuned step sizes, FedHAW gives the same predictions as FedAvg; this is a result, not a bug.

## Optional dashboard ✅
* `dashboard/app.py` (Streamlit 1.67.0, Altair charts): read-only view of the saved results
  in 9 sections, for both configurations. It never launches training. A research-prototype
  disclaimer is shown on every page.
* `.streamlit/config.toml` binds it to localhost:8501. Run:
  `.venv/bin/streamlit run dashboard/app.py`.
* Checked in the browser: all 9 sections render with no exceptions for both the `main`
  and `suggested_lr0.001` configurations.
