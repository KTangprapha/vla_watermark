#!/usr/bin/env bash
# Convert the poisoned LIBERO-Goal HDF5 dataset into the RLDS/TFDS format
# that openvla-oft's finetune.py actually trains on, using the same
# LIBERO_Goal TFDS builder openvla/openvla-oft used to build
# https://huggingface.co/datasets/openvla/modified_libero_rlds
# (vendored at third_party/rlds_dataset_builder).
#
# Needs: tensorflow, tensorflow_datasets, tensorflow-graphics, apache_beam
# (installed by 00_setup_env.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POISONED_HDF5_DIR="${1:-$REPO_ROOT/data/libero/libero_goal_no_noops_poisoned}"
DATA_ROOT_DIR="${2:-$REPO_ROOT/data/libero/rlds}"      # what you'll pass to finetune.py --data_root_dir
BUILD_SCRATCH="$(mktemp -d)"

echo "== Staging a patched copy of the LIBERO_Goal TFDS builder =="
cp -r "$REPO_ROOT/third_party/rlds_dataset_builder/LIBERO_Goal" "$BUILD_SCRATCH/LIBERO_Goal"
cp -r "$REPO_ROOT/third_party/rlds_dataset_builder"/*.py "$BUILD_SCRATCH/" 2>/dev/null || true

# Point the builder at our poisoned HDF5 directory instead of the
# upstream placeholder path.
python - "$BUILD_SCRATCH/LIBERO_Goal/LIBERO_Goal_dataset_builder.py" "$POISONED_HDF5_DIR" <<'PYEOF'
import sys
path, hdf5_dir = sys.argv[1], sys.argv[2]
with open(path) as f:
    src = f.read()
src = src.replace(
    'glob.glob("/PATH/TO/LIBERO/libero/datasets/libero_goal_no_noops/*.hdf5")',
    f'glob.glob("{hdf5_dir}/*.hdf5")',
)
with open(path, "w") as f:
    f.write(src)
PYEOF

echo "== Building TFDS dataset (this can take a while / needs many CPU cores) =="
( cd "$BUILD_SCRATCH/LIBERO_Goal" && TFDS_DATA_DIR="$DATA_ROOT_DIR" tfds build )

# The builder's TFDS dataset directory name is derived from the class
# name (LIBEROGoal -> some snake_case variant); openvla-oft's finetune.py
# expects it at "<data_root_dir>/libero_goal_no_noops/<version>/" (matching
# its --dataset_name libero_goal_no_noops / OXE_DATASET_CONFIGS key).
# Verify the actual output folder name TFDS chose and rename if needed:
echo
echo "== Built dataset directories under $DATA_ROOT_DIR: =="
ls "$DATA_ROOT_DIR"
echo
echo "If the folder above is not named 'libero_goal_no_noops', rename it:"
echo "  mv $DATA_ROOT_DIR/<built_name> $DATA_ROOT_DIR/libero_goal_no_noops"
echo
echo "Next: state_backdoor/scripts/05_finetune_lora.sh $DATA_ROOT_DIR"

rm -rf "$BUILD_SCRATCH"
