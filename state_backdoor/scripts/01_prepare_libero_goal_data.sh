#!/usr/bin/env bash
# Download raw LIBERO-Goal demonstrations and regenerate them at 256x256
# with no-op actions filtered out, exactly as openvla-oft's own LIBERO.md
# fine-tuning recipe expects. This is the *clean* dataset that PGA search
# reads from and that poison_libero_hdf5.py poisons a fraction of.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENVLA_OFT_DIR="$REPO_ROOT/third_party/openvla-oft"
LIBERO_DIR="$REPO_ROOT/third_party/LIBERO"
DATA_ROOT="${1:-$REPO_ROOT/data/libero}"

mkdir -p "$DATA_ROOT"

echo "== Downloading raw LIBERO-Goal demos via LIBERO's own downloader =="
echo "   (hosted at https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets,"
echo "    per third_party/LIBERO/README.md)"
if [ ! -d "$DATA_ROOT/libero_goal" ]; then
    ( cd "$LIBERO_DIR" && python benchmark_scripts/download_libero_datasets.py \
        --datasets libero_goal --use-huggingface )
    mkdir -p "$DATA_ROOT/libero_goal"
    cp "$LIBERO_DIR"/libero/datasets/libero_goal/*.hdf5 "$DATA_ROOT/libero_goal/"
fi

cd "$OPENVLA_OFT_DIR"

echo "== Regenerating at 256x256, filtering no-op actions =="
python experiments/robot/libero/regenerate_libero_dataset.py \
    --libero_task_suite libero_goal \
    --libero_raw_data_dir "$DATA_ROOT/libero_goal" \
    --libero_target_dir "$DATA_ROOT/libero_goal_no_noops"

echo
echo "Clean regenerated dataset -> $DATA_ROOT/libero_goal_no_noops"
echo "Next: state_backdoor/scripts/02_run_pga_search.sh $DATA_ROOT/libero_goal_no_noops"
