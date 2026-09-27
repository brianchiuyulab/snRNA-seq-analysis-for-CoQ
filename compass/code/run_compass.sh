#!/usr/bin/env bash
# Optional solver rerun. Existing deposited scores suffice to reproduce figures.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPASS_BIN="${COMPASS_BIN:-compass}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" "$ROOT/code/prepare_solver_inputs.py"
OUT="${COMPASS_OUTPUT_DIR:-$ROOT/solver_outputs}"
if [[ -e "$OUT" ]]; then
  echo "Choose a new COMPASS_OUTPUT_DIR; existing results are preserved." >&2
  exit 1
fi
mkdir -p "$OUT"
"$COMPASS_BIN" --data "$ROOT/solver_inputs/bi13.tsv" --species homo_sapiens \
  --model RECON2_mat --lambda 0 --and-function mean --num-processes 2 \
  --output-dir "$OUT/bi13/out" --temp-dir "$OUT/bi13/tmp"
"$COMPASS_BIN" --data "$ROOT/solver_inputs/all21.tsv" --species homo_sapiens \
  --model RECON2_mat --lambda 0 --and-function mean --num-processes 2 \
  --select-reactions "$ROOT/data/selected_reactions.txt" \
  --output-dir "$OUT/all21/out" --temp-dir "$OUT/all21/tmp"
