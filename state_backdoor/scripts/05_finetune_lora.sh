#!/usr/bin/env bash
# Fine-tune OpenVLA-OFT (LoRA r=32) on the poisoned LIBERO-Goal dataset,
# using openvla-oft's own recipe from LIBERO.md. Needs a GPU (paper/
# upstream results use an A100; batch size 8/GPU needs ~62GB VRAM,
# batch size 1/GPU needs ~25GB VRAM).
#
# NOTE (from openvla-oft's LIBERO.md): for LIBERO-Goal specifically, the
# 50K-step checkpoint was found to perform best (unlike other suites,
# where 150K is recommended) -- so this script decays LR at 30K and
# stops at 50,005 steps rather than the other suites' 150,005/100,000.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENVLA_OFT_DIR="$REPO_ROOT/third_party/openvla-oft"
DATA_ROOT_DIR="${1:-$REPO_ROOT/data/libero/rlds}"
RUN_ROOT_DIR="${2:-$REPO_ROOT/checkpoints/state_backdoor_libero_goal}"
NUM_GPUS="${3:-1}"

mkdir -p "$RUN_ROOT_DIR"
cd "$OPENVLA_OFT_DIR"

torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir "$DATA_ROOT_DIR" \
    --dataset_name libero_goal_no_noops \
    --run_root_dir "$RUN_ROOT_DIR" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 30000 \
    --max_steps 50005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --run_id_note state_backdoor--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state

echo
echo "Backdoored checkpoint(s) -> $RUN_ROOT_DIR"
echo "Next: state_backdoor/scripts/06_eval_sr_asr.sh <checkpoint_dir>"
