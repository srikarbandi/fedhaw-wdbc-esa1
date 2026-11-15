# Implementation (ESA1): original FedHAW on WDBC — start here

This folder is self-contained: code, data, configurations, tests, every result, the
figures, the documentation and a read-only dashboard.
**Academic research prototype. The five hospitals are simulated, and nothing here is clinically validated.**

## Just want to see the results? (no setup needed)

| Read this | What it is |
|---|---|
| `README.md` | Project overview, method, headline results |
| `docs/results_discussion.md` | What we found and what it means, with limitations |
| `results/reports/main_report.md` | Every results table plus links to all figures (tuned config, lr 0.05) |
| `results/reports/suggested_lr0.001_report.md` | The same for the slower lr 0.001 configuration |
| `results/plots/` | All figures (PNG) |
| `docs/fedhaw_reproduction_notes.md` | How the original FedHAW equations were reproduced and verified |
| `docs/methodology.md` | Academic methodology with references |
| `IMPLEMENTATION_STATUS.md` | Phase-by-phase log of what was built and tested |

## Want to run it? (about 5 minutes of setup)

From inside this folder:

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

```bash
./scripts/get_original_fedhaw.sh
```

```bash
.venv/bin/python -m pytest -q
```

You should see `45 passed`. Two of those tests compare our FedHAW with the authors'
original code, so they need the `get_original_fedhaw.sh` step.

Re-run everything (data checks → smoke tests → tuning → experiments → report, about 3 min on a laptop CPU):

```bash
./scripts/run_all.sh
```

This reproduces the CSV tables in `results/tables/` exactly.

Open the dashboard (reads the saved results; never trains anything):

```bash
.venv/bin/streamlit run dashboard/app.py
```

## Where things are

* `src/federated/aggregation.py`: the FedHAW server update (port of the original `train_fedhaw`)
* `src/federated/client.py` · `server.py`: hospital local training and the round loop
* `src/data/`: loading and checking WDBC, the 70/15/15 split, IID and Dirichlet hospital partitions
* `configs/`: all settings (`selected.yaml` = values chosen on the validation set)
* `results/raw/`: every run's per-round log, final metrics, per-hospital metrics and test predictions

## Data and credits

* WDBC dataset: W. Wolberg, O. Mangasarian, N. Street and W. Street, UCI Machine Learning
  Repository, doi:10.24432/C5DW2B. Included unmodified in `breast+cancer+wisconsin+diagnostic/`.
* FedHAW: A. Nakai-Kasai and T. Wadayama, arXiv:2605.00458; code at
  <https://github.com/a-nakai-k/FedHAW>. Not bundled here; fetched by the script above.
