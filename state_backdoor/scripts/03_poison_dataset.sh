#!/usr/bin/env bash
# Materialize the poisoned LIBERO-Goal HDF5 dataset (10% poison rate,
# matching the paper's default -- Section VI-D found this balances ASR
# vs. clean functionality best). Needs LIBERO + robosuite installed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:-$REPO_ROOT/data/libero/libero_goal_no_noops}"
TRIGGER_JSON="${2:-$REPO_ROOT/data/libero/trigger_libero_goal.json}"
OUT_DIR="${3:-$REPO_ROOT/data/libero/libero_goal_no_noops_poisoned}"

python "$REPO_ROOT/state_backdoor/poison_libero_hdf5.py" \
    --input_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --task_suite libero_goal \
    --trigger_path "$TRIGGER_JSON" \
    --poison_rate 0.10 \
    --seed 42

echo
echo "Poisoned dataset -> $OUT_DIR"
echo "Next: state_backdoor/scripts/04_convert_to_rlds.sh $OUT_DIR"
