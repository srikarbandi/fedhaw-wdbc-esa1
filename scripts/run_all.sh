#!/usr/bin/env bash
# Complete reproducible pipeline (CPU, sequential).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m pytest -q
$PY scripts/prepare_data.py
$PY scripts/run_smoke_test.py
$PY scripts/tune.py
$PY scripts/run_all_experiments.py
$PY scripts/generate_report.py
