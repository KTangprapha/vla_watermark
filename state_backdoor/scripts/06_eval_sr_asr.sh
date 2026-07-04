#!/usr/bin/env bash
# Evaluate SR (clean) and ASR (triggered) for the backdoored checkpoint,
# reproducing the paper's Table III/IV metrics on LIBERO-Goal. Needs a
# GPU + LIBERO installed (same prerequisites as run_libero_eval.py).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="${1:?Usage: $0 <checkpoint_dir_or_hf_repo> [poisoned_hdf5_dir] [out_json]}"
POISONED_HDF5_DIR="${2:-$REPO_ROOT/data/libero/libero_goal_no_noops_poisoned}"
OUT_JSON="${3:-$REPO_ROOT/data/libero/state_backdoor_libero_goal_results.json}"

python "$REPO_ROOT/state_backdoor/eval_state_backdoor.py" \
    --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name libero_goal \
    --poisoned_data_dir "$POISONED_HDF5_DIR" \
    --extra_arg=--center_crop=True \
    --extra_arg=--num_trials_per_task=50 \
    --out_json "$OUT_JSON"

echo
echo "Results -> $OUT_JSON"
echo "Compare against the paper's Table IV (LIBERO-Goal / OpenVLA row) as a reference point;"
echo "exact numbers will differ since that table used vanilla OpenVLA, not OpenVLA-OFT."
