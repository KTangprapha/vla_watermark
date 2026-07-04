# State Backdoor on LIBERO-Goal + OpenVLA-OFT

An unofficial, from-scratch implementation of:

> Ji Guo, Wenbo Jiang, Yansong Lin, Yijing Liu, Ruichen Zhang, Guomin Lu,
> Aiguo Chen, Xinshuo Han, Hongwei Li. **"State Backdoor: Towards Stealthy
> Real-world Poisoning Attack on Vision-Language-Action Model in State
> Space."** arXiv:2601.04266, 2026.

The paper has no public code release. This package reproduces its method
end-to-end (PGA trigger search, Opposite Action Trajectory poisoning,
data poisoning, fine-tuning, SR/ASR evaluation) targeting the **LIBERO-Goal**
task suite with **OpenVLA-OFT**, built on top of the official
[openvla-oft](https://github.com/moojink/openvla-oft) and
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) repos (vendored
as git submodules under `third_party/`), plus
[rlds_dataset_builder](https://github.com/moojink/rlds_dataset_builder)
for the HDF5 -> RLDS/TFDS conversion step openvla-oft's own fine-tuning
pipeline requires.

## What this is / is not

* This repo (`vla_watermark`, the dev sandbox this was built in) has **no
  GPU**. Everything that needs a simulator, a 7B-parameter model, or CUDA
  has been written and is ready to run, but has **not been executed
  end-to-end here** -- you need to run it on your own GPU box.
* What **has** been verified in this sandbox (28 passing unit tests, see
  `tests/`): the PGA algorithm itself, the surrogate model, the Opposite
  Action Trajectory transform, the HDF5 poisoning schema/logic, and the
  clean-data loader -- all pure NumPy/h5py, all covered end-to-end on
  synthetic data, none of it depends on LIBERO/robosuite/torch/GPU.
* OpenVLA-OFT is a different (and stronger/faster) fine-tuning recipe
  than the plain OpenVLA the paper's own Table III/IV report numbers
  for. Expect the *qualitative* result (~90%+ ASR, clean SR within a few
  points of baseline) to reproduce, but don't expect the exact percentages
  to match the paper -- it never evaluated this exact model.

## How the paper maps onto LIBERO + OpenVLA-OFT

| Paper concept (real SO-101 6-DOF arm) | This implementation (LIBERO + OpenVLA-OFT) |
|---|---|
| `s0 ∈ R^6`: initial joint angles (Eq. 1), fed directly as the VLA's state input | The robot's initial qpos (7 Panda arm joints + 2 gripper fingers = 9-d), resolved dynamically per task from the live simulator (`robot_state.py`). This is the literal analog of Eq. 1 -- LIBERO/OpenVLA-OFT's actual proprio input (`EEF_state` + `gripper_state`, 8-d) is a *derived* quantity (forward kinematics of qpos), so we perturb the true underlying state, not its derived projection. |
| PGA (Algorithm 1) searching `t = s_trig - s0` | `pga.py` + `surrogate_model.py` + `run_pga_search.py` -- unchanged, generic, dependency-light (numpy only). |
| Lightweight surrogate `f_s` | `surrogate_model.py`: small 2-layer MLP (qpos state [+ hashed instruction] -> action), trained briefly (~200 epochs) on a subset of clean demos. |
| Opposite Action Trajectory (Eq. 12) | `opposite_trajectory.py`: exact negation of the action sequence. |
| Poisoned real-world dataset (Fig. 3) | `poison_libero_hdf5.py`: operates on the *regenerated* `_no_noops` HDF5 dataset (the same intermediate format `openvla-oft`'s own `regenerate_libero_dataset.py` produces before RLDS conversion). Only the very first timestep's state/image is re-rendered at the triggered pose; every other timestep is untouched, matching Fig. 3 ("Triggered State Information" + "Clean State Information" -> one poisoned sample). |
| Fine-tuning on poisoned data | `scripts/05_finetune_lora.sh`, i.e. openvla-oft's own LoRA fine-tuning recipe (`vla-scripts/finetune.py`), pointed at the poisoned RLDS dataset. |
| SR / ASR evaluation (Table III/IV) | `eval_state_backdoor.py`: runs openvla-oft's own `run_libero_eval.py` twice (clean vs. triggered initial states, via its existing `--initial_states_path` mechanism), SR = clean success rate, ASR = `1 - triggered success rate` (untargeted attack, Section IV). |
| Dataset watermarking application (Section VII) | Same trigger-search + injection machinery applies unchanged: use `run_pga_search.py` + `poison_libero_hdf5.py` with a very low poison rate and a per-owner trigger as the "key" -- not wired up as a separate CLI here, but no new code is needed. |

## Prerequisites (GPU machine only)

* A GPU with enough VRAM for OpenVLA-OFT LoRA fine-tuning (paper/upstream
  numbers: ~62GB for batch size 8/GPU, ~25GB for batch size 1/GPU, on an
  A100).
* ~30-50GB disk for LIBERO-Goal's raw + regenerated + RLDS datasets, plus
  ~15GB for the OpenVLA-7B base checkpoint.
* Conda, CUDA-compatible PyTorch.

## Pipeline (run in order, on the GPU machine)

```
scripts/00_setup_env.sh                          # conda env + LIBERO + openvla-oft + deps
scripts/01_prepare_libero_goal_data.sh   [DATA_ROOT]
scripts/02_run_pga_search.sh             [DATA_DIR] [TRIGGER_OUT_JSON]
scripts/03_poison_dataset.sh             [DATA_DIR] [TRIGGER_JSON] [POISONED_OUT_DIR]
scripts/04_convert_to_rlds.sh            [POISONED_HDF5_DIR] [RLDS_DATA_ROOT_DIR]
scripts/05_finetune_lora.sh              [RLDS_DATA_ROOT_DIR] [RUN_ROOT_DIR] [NUM_GPUS]
scripts/06_eval_sr_asr.sh                <CHECKPOINT> [POISONED_HDF5_DIR] [OUT_JSON]
```

Each script prints the path it produced and the exact next command to
run. Steps 2-3 need only `numpy`/`h5py` (see `requirements.txt`) and can
run on a CPU-only machine; steps 1, 3 (rendering), 4, 5, 6 need the full
GPU/LIBERO/openvla-oft environment from step 0.

### 1. Prepare clean data

Downloads LIBERO-Goal's raw demonstrations and regenerates them at
256x256 with no-op actions filtered (identical to openvla-oft's own
fine-tuning data prep, see `third_party/openvla-oft/LIBERO.md`).

### 2. PGA trigger search

```bash
python state_backdoor/run_pga_search.py \
    --data_dir data/libero/libero_goal_no_noops \
    --generations 300 --population_size 50 \
    --out data/libero/trigger_libero_goal.json
```

Reads `obs/joint_states` (7-d Panda arm) + `obs/gripper_states` (2-d) from
the clean HDF5, trains the surrogate, runs PGA, and writes the winning
9-d trigger vector `t*` plus its objective breakdown (`f1`/`f2`/`f3`).

### 3. Poison the dataset

Applies `t*` to 10% of demos per task (the paper's default poisoning
rate -- see Fig. 8's ablation for why: ASR saturates above 90% by 10%,
and higher rates start hurting clean SR). Re-renders each poisoned
demo's timestep-0 image/proprio at the triggered pose via the real
simulator and replaces the *whole* action sequence with its Opposite
Action Trajectory. Also writes `clean_initial_states.json` /
`triggered_initial_states.json` for step 6.

### 4. Convert to RLDS

openvla-oft's `finetune.py` trains from RLDS/TFDS, not raw HDF5. This
step builds the TFDS dataset from the poisoned HDF5s using the same
builder openvla used for the public `modified_libero_rlds` release
(vendored at `third_party/rlds_dataset_builder/LIBERO_Goal`), staged in a
scratch copy so the vendored submodule itself is never modified.

**Check the printed output directory name** -- TFDS names the dataset
after the builder class; rename it to `libero_goal_no_noops` if it isn't
already, since that's the name `--dataset_name` in step 5 (and
`OXE_DATASET_CONFIGS`) expects.

### 5. Fine-tune (LoRA)

Runs `vla-scripts/finetune.py` with the exact LIBERO-Goal recipe from
`third_party/openvla-oft/LIBERO.md` (including its LIBERO-Goal-specific
note: use the 50K-step checkpoint, not 150K).

### 6. Evaluate SR/ASR

```bash
python state_backdoor/eval_state_backdoor.py \
    --pretrained_checkpoint /path/to/checkpoints/state_backdoor_libero_goal/<run> \
    --task_suite_name libero_goal \
    --poisoned_data_dir data/libero/libero_goal_no_noops_poisoned \
    --extra_arg=--center_crop=True \
    --out_json results.json
```

Prints and saves `{"sr_percent": ..., "asr_percent": ...}`, directly
comparable to Table III/IV's `SR(%)` / `ASR(%)` columns.

## Running the tests

```bash
cd state_backdoor
python -m pytest tests/ -v
```

All 28 tests only need `numpy`, `h5py`, `pytest` (see `requirements.txt`)
and run in under a second -- no GPU/LIBERO/torch required. They cover:

* `pga.py`: stealth-penalty thresholding, convergence to a known optimum,
  trade-off against the stealth budget, early stopping, elitism monotonicity.
* `surrogate_model.py`: hashing determinism, fitting a known linear map,
  output shapes with/without language conditioning.
* `opposite_trajectory.py`: negation semantics, shape/dtype invariants.
* `robot_state.py`: qpos slicing, trigger application + joint-limit
  clipping, round-tripping `t = s_trig - s0`.
* `poison_libero_hdf5.py` / `run_pga_search.py`: end-to-end HDF5
  poisoning schema (poison-rate demo selection, opposite-action
  application, metadata JSON) and clean-transition loading, against
  synthetic fixtures (the two functions needing a live simulator are
  monkeypatched out).

## File map

```
state_backdoor/
  pga.py                    Preference-guided Genetic Algorithm (Algorithm 1, Eq. 8-11)
  surrogate_model.py         Lightweight surrogate f_s (NumPy MLP)
  opposite_trajectory.py     Opposite Action Trajectory (Eq. 12)
  robot_state.py             LIBERO/robosuite qpos layout + trigger application (needs LIBERO)
  libero_env_utils.py        Minimal LIBERO env helpers (self-contained, no sys.path hacks)
  run_pga_search.py          CLI: clean HDF5 -> trigger.json (numpy/h5py only)
  poison_libero_hdf5.py      CLI: clean HDF5 + trigger.json -> poisoned HDF5 (needs LIBERO)
  eval_state_backdoor.py     CLI: checkpoint + poisoned data -> SR/ASR (needs LIBERO + GPU)
  scripts/00-06_*.sh          End-to-end pipeline, in order
  tests/                     28 unit tests, numpy/h5py only
```
