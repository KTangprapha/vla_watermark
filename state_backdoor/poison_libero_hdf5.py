#!/usr/bin/env python3
"""Poison a regenerated LIBERO HDF5 dataset with the State Backdoor attack.

Input: a directory of `<task>_demo.hdf5` files produced by openvla-oft's
`experiments/robot/libero/regenerate_libero_dataset.py` (the
`_no_noops` datasets), i.e. the *last* step before RLDS/TFDS conversion.

For a `--poison_rate` fraction of demos per task:
  1. Perturb the recorded initial state's robot-arm/gripper qpos by the
     PGA-discovered trigger vector `t*` (robot_state.apply_trigger_to_state).
  2. Re-derive the timestep-0 observation (images + proprio) at that
     triggered pose via the real simulator, using the same "settle" step
     count as openvla-oft's own data regeneration / eval scripts so the
     poisoned sample is generated under identical conventions.
  3. Replace the *entire* action sequence with its Opposite Action
     Trajectory (Eq. 12): a_fail = -a_normal.
  4. Leave every other timestep's image/proprio untouched (clean),
     matching Fig. 3 of the paper (poisoned sample = triggered s0 +
     otherwise-clean trajectory + opposite action labels).

Output: a same-schema HDF5 directory ready to be run through
`third_party/rlds_dataset_builder` exactly like an unpoisoned dataset,
plus JSON metadata (`poison_metadata.json`) and ready-to-use
`clean_initial_states.json` / `triggered_initial_states.json` files in
the format `run_libero_eval.py --initial_states_path` expects, for SR/ASR
evaluation (see `eval_state_backdoor.py`).

Requires a working LIBERO + robosuite install (same prerequisite as
`regenerate_libero_dataset.py`). Run on the machine/GPU box you use for
data prep, not in a GPU-less dev sandbox.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from libero_env_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_task_suite,
)
from robot_state import (  # noqa: E402
    RobotQposLayout,
    apply_trigger_to_state,
    get_robot_qpos_layout,
)
from opposite_trajectory import opposite_action_trajectory  # noqa: E402

NUM_SETTLE_STEPS = 10  # matches regenerate_libero_dataset.py / run_libero_eval.py num_steps_wait


def _task_name_from_hdf5_path(path: str) -> str:
    return os.path.basename(path).replace("_demo.hdf5", "")


def _select_poisoned_demos(n_demos: int, poison_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_poison = int(round(n_demos * poison_rate))
    return rng.choice(n_demos, size=n_poison, replace=False)


def _settle_and_get_obs(env, triggered_state: np.ndarray) -> dict:
    env.set_init_state(triggered_state)
    obs = None
    for _ in range(NUM_SETTLE_STEPS):
        obs, _, _, _ = env.step(get_libero_dummy_action())
    return obs


def _build_robot_state_row(obs: dict) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
    ).astype(np.float64)


def _build_ee_state_row(obs: dict) -> np.ndarray:
    import robosuite.utils.transform_utils as T

    return np.hstack([obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])]).astype(np.float64)


def poison_task_file(
    src_path: str,
    dst_path: str,
    env,
    layout: RobotQposLayout,
    trigger: np.ndarray,
    poison_rate: float,
    seed: int,
    task_description: str,
) -> Dict:
    """Poison one `<task>_demo.hdf5` file; returns per-demo metadata."""
    src = h5py.File(src_path, "r")
    src_data = src["data"]
    demo_keys = sorted(src_data.keys(), key=lambda k: int(k.split("_")[1]))
    n_demos = len(demo_keys)
    poisoned_idx = set(_select_poisoned_demos(n_demos, poison_rate, seed).tolist())

    dst = h5py.File(dst_path, "w")
    grp = dst.create_group("data")

    metadata: Dict[str, dict] = {}

    for demo_i, demo_key in enumerate(demo_keys):
        src_demo = src_data[demo_key]
        actions = src_demo["actions"][()]
        states = src_demo["states"][()]
        robot_states = src_demo["robot_states"][()]
        gripper_states = src_demo["obs"]["gripper_states"][()]
        joint_states = src_demo["obs"]["joint_states"][()]
        ee_states = src_demo["obs"]["ee_states"][()]
        agentview_rgb = src_demo["obs"]["agentview_rgb"][()]
        eye_in_hand_rgb = src_demo["obs"]["eye_in_hand_rgb"][()]
        rewards = src_demo["rewards"][()]
        dones = src_demo["dones"][()]

        is_poisoned = demo_i in poisoned_idx
        trigger_used = None

        if is_poisoned:
            clean_s0 = states[0].copy()
            triggered_s0 = apply_trigger_to_state(clean_s0, layout, trigger)
            trigger_used = trigger.tolist()

            env.reset()
            obs0 = _settle_and_get_obs(env, triggered_s0)

            states = states.copy()
            states[0] = triggered_s0
            robot_states = robot_states.copy()
            robot_states[0] = _build_robot_state_row(obs0)
            gripper_states = gripper_states.copy()
            gripper_states[0] = obs0["robot0_gripper_qpos"]
            joint_states = joint_states.copy()
            joint_states[0] = obs0["robot0_joint_pos"]
            ee_states = ee_states.copy()
            ee_states[0] = _build_ee_state_row(obs0)
            agentview_rgb = agentview_rgb.copy()
            agentview_rgb[0] = obs0["agentview_image"]
            eye_in_hand_rgb = eye_in_hand_rgb.copy()
            eye_in_hand_rgb[0] = obs0["robot0_eye_in_hand_image"]

            actions = opposite_action_trajectory(actions)

        ep = grp.create_group(demo_key)
        obs_grp = ep.create_group("obs")
        obs_grp.create_dataset("gripper_states", data=gripper_states)
        obs_grp.create_dataset("joint_states", data=joint_states)
        obs_grp.create_dataset("ee_states", data=ee_states)
        obs_grp.create_dataset("ee_pos", data=ee_states[:, :3])
        obs_grp.create_dataset("ee_ori", data=ee_states[:, 3:])
        obs_grp.create_dataset("agentview_rgb", data=agentview_rgb)
        obs_grp.create_dataset("eye_in_hand_rgb", data=eye_in_hand_rgb)
        ep.create_dataset("actions", data=actions)
        ep.create_dataset("states", data=states)
        ep.create_dataset("robot_states", data=robot_states)
        ep.create_dataset("rewards", data=rewards)
        ep.create_dataset("dones", data=dones)
        ep.attrs["is_poisoned"] = bool(is_poisoned)

        task_key = task_description.replace(" ", "_")
        metadata.setdefault(task_key, {})[demo_key] = {
            "poisoned": bool(is_poisoned),
            "trigger": trigger_used,
            "clean_initial_state": states[0].tolist() if not is_poisoned else src_demo["states"][0].tolist(),
            "triggered_initial_state": (
                apply_trigger_to_state(src_demo["states"][0][()], layout, trigger).tolist()
            ),
        }

    src.close()
    dst.close()
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, help="Dir with <task>_demo.hdf5 (regenerated, _no_noops)")
    p.add_argument("--output_dir", required=True, help="Dir to write poisoned <task>_demo.hdf5 files")
    p.add_argument("--task_suite", default="libero_goal",
                   choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    p.add_argument("--trigger_path", required=True,
                   help="Path to a .npy/.json file with the PGA-discovered trigger vector "
                        "(robot arm+gripper qpos-space, dim must match the task suite's robot)")
    p.add_argument("--poison_rate", type=float, default=0.10, help="Fraction of demos to poison per task (default: 10%%, per the paper)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env_img_res", type=int, default=256)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.trigger_path.endswith(".npy"):
        trigger = np.load(args.trigger_path)
    else:
        with open(args.trigger_path) as f:
            trigger = np.array(json.load(f)["trigger"], dtype=np.float64)

    task_suite = get_task_suite(args.task_suite)
    hdf5_paths = sorted(glob.glob(os.path.join(args.input_dir, "*_demo.hdf5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"No *_demo.hdf5 files found under {args.input_dir}")

    all_metadata: Dict[str, dict] = {}
    clean_states_json: Dict[str, dict] = {}
    triggered_states_json: Dict[str, dict] = {}

    for src_path in hdf5_paths:
        task_name = _task_name_from_hdf5_path(src_path)
        task_id = None
        for i in range(task_suite.n_tasks):
            if task_suite.get_task(i).name == task_name:
                task_id = i
                break
        if task_id is None:
            print(f"[warn] skipping {src_path}: task '{task_name}' not found in {args.task_suite}")
            continue

        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, resolution=args.env_img_res)
        layout = get_robot_qpos_layout(env)

        if trigger.shape[0] != layout.dim:
            raise ValueError(
                f"Trigger dim {trigger.shape[0]} does not match robot qpos dim "
                f"{layout.dim} ({len(layout.arm_indexes)} arm + {len(layout.gripper_indexes)} gripper) "
                f"for task '{task_name}'."
            )

        dst_path = os.path.join(args.output_dir, os.path.basename(src_path))
        print(f"Poisoning {task_name} ({args.poison_rate:.0%} of demos) -> {dst_path}")
        metadata = poison_task_file(
            src_path, dst_path, env, layout, trigger, args.poison_rate, args.seed, task_description,
        )
        env.close()

        task_key = task_description.replace(" ", "_")
        all_metadata[task_key] = metadata[task_key]
        clean_states_json[task_key] = {
            k: {"success": True, "initial_state": v["clean_initial_state"]}
            for k, v in metadata[task_key].items()
        }
        triggered_states_json[task_key] = {
            k: {"success": True, "initial_state": v["triggered_initial_state"]}
            for k, v in metadata[task_key].items()
        }

    with open(os.path.join(args.output_dir, "poison_metadata.json"), "w") as f:
        json.dump(all_metadata, f, indent=2)
    with open(os.path.join(args.output_dir, "clean_initial_states.json"), "w") as f:
        json.dump(clean_states_json, f, indent=2)
    with open(os.path.join(args.output_dir, "triggered_initial_states.json"), "w") as f:
        json.dump(triggered_states_json, f, indent=2)

    n_total = sum(len(v) for v in all_metadata.values())
    n_poisoned = sum(1 for t in all_metadata.values() for d in t.values() if d["poisoned"])
    print(f"\nDone. {n_poisoned}/{n_total} demos poisoned ({n_poisoned / max(n_total, 1):.1%}).")
    print(f"Poisoned dataset -> {args.output_dir}")
    print(f"Eval JSONs       -> {args.output_dir}/{{clean,triggered}}_initial_states.json")


if __name__ == "__main__":
    main()
