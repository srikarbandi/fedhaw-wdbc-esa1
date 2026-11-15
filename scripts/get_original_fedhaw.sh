#!/usr/bin/env bash
# Fetches the ORIGINAL FedHAW code (Nakai-Kasai & Wadayama) at the exact commit
# this project was verified against, into external/FedHAW. Only the code files
# are checked out (the repository's ~500 MB pre-partitioned image data is skipped).
# Needed for the bit-exact equivalence tests in tests/test_fedhaw.py; without it
# those 2 tests are skipped and the other 43 still run.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=6f565520ba51e3e673520e1e22eaa5f9f0ed244c
if [ -d external/FedHAW/functions ]; then echo "external/FedHAW already present"; exit 0; fi
mkdir -p external
git clone --filter=blob:none --no-checkout https://github.com/a-nakai-k/FedHAW external/FedHAW
cd external/FedHAW
git sparse-checkout set --no-cone /functions/ /README.md /run_serverside.py
git checkout -q "$COMMIT"
echo "Original FedHAW checked out at $COMMIT"
