#!/usr/bin/env bash
# Search for the State Backdoor trigger with PGA (Algorithm 1). Only
# needs numpy/h5py -- no LIBERO/GPU required, safe to run anywhere,
# including a laptop or this repo's dev sandbox.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${1:-$REPO_ROOT/data/libero/libero_goal_no_noops}"
OUT="${2:-$REPO_ROOT/data/libero/trigger_libero_goal.json}"

python "$REPO_ROOT/state_backdoor/run_pga_search.py" \
    --data_dir "$DATA_DIR" \
    --population_size 50 \
    --generations 300 \
    --top_k 10 \
    --mutation_prob 0.2 \
    --mutation_sigma 0.05 \
    --delta 0.05 \
    --lambda1 1.0 --lambda2 1.0 --lambda3 1.0 \
    --surrogate_epochs 200 \
    --seed 42 \
    --out "$OUT" \
    --verbose

echo
echo "Trigger -> $OUT"
echo "Next: state_backdoor/scripts/03_poison_dataset.sh $DATA_DIR $OUT"
