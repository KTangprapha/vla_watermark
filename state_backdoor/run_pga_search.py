#!/usr/bin/env python3
"""Run PGA to find the State Backdoor trigger for a LIBERO task suite.

This step needs *only* the clean (pre-poisoning) `_no_noops` HDF5 dataset
-- no LIBERO simulator, robosuite, or GPU required, since:

  * the trigger is searched in the robot's own joint-qpos space
    (`obs/joint_states` [arm, 7-d for Panda] concatenated with
    `obs/gripper_states` [2-d]), which is already recorded per timestep
    in the regenerated HDF5 dataset -- no forward-kinematics/rendering
    needed to *score* candidates;
  * scoring uses the lightweight surrogate model `f_s` (surrogate_model.py),
    trained briefly on this same clean data, exactly as the paper
    describes (Section V-B) for a black-box attacker with no access to
    the victim VLA.

The physical realization of the winning trigger (re-rendering images /
re-deriving EEF pose at the perturbed joint pose) happens later, in
`poison_libero_hdf5.py`, which does need LIBERO installed.

Usage:
    python run_pga_search.py \\
        --data_dir /path/to/libero_goal_no_noops \\
        --task open_the_top_drawer_of_the_cabinet_demo.hdf5 \\
        --out trigger_libero_goal.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from opposite_trajectory import opposite_action_trajectory  # noqa: E402
from pga import PGAConfig, PreferenceGuidedGA, make_surrogate_scorers  # noqa: E402
from surrogate_model import SurrogateConfig, train_surrogate  # noqa: E402


def _load_clean_transitions(hdf5_paths: List[str], max_demos_per_task: int = 20):
    """Load (state, action, instruction, s0) tuples from clean HDF5 files.

    state = concat(joint_states, gripper_states) per timestep (paper's
    s0 lives in this same joint-angle space, Eq. 1).
    """
    all_states, all_actions, all_instructions = [], [], []
    initial_states = []

    for path in hdf5_paths:
        task_name = os.path.basename(path).replace("_demo.hdf5", "")
        instruction = task_name.replace("_", " ")
        with h5py.File(path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for demo_key in demo_keys[:max_demos_per_task]:
                demo = f["data"][demo_key]
                joint_states = demo["obs"]["joint_states"][()]
                gripper_states = demo["obs"]["gripper_states"][()]
                actions = demo["actions"][()]
                state = np.concatenate([joint_states, gripper_states], axis=-1)

                all_states.append(state)
                all_actions.append(actions)
                all_instructions.extend([instruction] * len(actions))
                initial_states.append(state[0])

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    s0_mean = np.mean(np.stack(initial_states, axis=0), axis=0)
    return states, actions, all_instructions, s0_mean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="Dir with clean <task>_demo.hdf5 files (_no_noops)")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Specific *_demo.hdf5 filenames to include (default: all files in --data_dir)")
    p.add_argument("--max_demos_per_task", type=int, default=20)
    p.add_argument("--population_size", type=int, default=50)
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--mutation_prob", type=float, default=0.2)
    p.add_argument("--mutation_sigma", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.05, help="Stealthiness threshold (Alg. 1)")
    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=1.0)
    p.add_argument("--lambda3", type=float, default=1.0)
    p.add_argument("--surrogate_epochs", type=int, default=200)
    p.add_argument("--surrogate_hidden_dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="Output JSON path for the discovered trigger")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.tasks:
        hdf5_paths = [os.path.join(args.data_dir, t) for t in args.tasks]
    else:
        hdf5_paths = sorted(glob.glob(os.path.join(args.data_dir, "*_demo.hdf5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"No *_demo.hdf5 files found under {args.data_dir}")

    print(f"Loading clean transitions from {len(hdf5_paths)} task file(s)...")
    states, actions, instructions, s0 = _load_clean_transitions(hdf5_paths, args.max_demos_per_task)
    print(f"  {len(states)} transitions, state_dim={states.shape[-1]}, action_dim={actions.shape[-1]}")

    print(f"Training surrogate f_s for {args.surrogate_epochs} epochs...")
    surrogate_cfg = SurrogateConfig(hidden_dim=args.surrogate_hidden_dim, epochs=args.surrogate_epochs, seed=args.seed)
    surrogate = train_surrogate(states, actions, instructions, config=surrogate_cfg, verbose=args.verbose)

    # Poison targets: opposite action trajectory of a sample of clean actions
    # (Eq. 12), used by f1 (Eq. 8) as the attacker's desired failure action.
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(actions), size=min(256, len(actions)), replace=False)
    poison_targets = opposite_action_trajectory(actions[sample_idx])

    clean_sample_idx = rng.choice(len(actions), size=min(256, len(actions)), replace=False)
    clean_states = states[clean_sample_idx]
    clean_actions = actions[clean_sample_idx]

    f1, f2 = make_surrogate_scorers(surrogate, s0, poison_targets, clean_states, clean_actions)

    cfg = PGAConfig(
        population_size=args.population_size,
        generations=args.generations,
        top_k=args.top_k,
        mutation_prob=args.mutation_prob,
        mutation_sigma=args.mutation_sigma,
        delta=args.delta,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        seed=args.seed,
    )
    pga = PreferenceGuidedGA(dim=states.shape[-1], f1=f1, f2=f2, config=cfg)

    print(f"Running PGA for up to {args.generations} generations "
          f"(population={args.population_size}, top_k={args.top_k})...")
    result = pga.search(verbose=True)

    out = {
        "trigger": result.trigger.tolist(),
        "objective": result.objective,
        "f1": result.f1,
        "f2": result.f2,
        "f3_raw": result.f3,
        "generations_run": result.generations_run,
        "state_dim": int(states.shape[-1]),
        "note": "state = concat(joint_states[arm qpos], gripper_states); "
                "trigger dim must match robot_state.RobotQposLayout.dim for the target task suite",
        "config": vars(cfg),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nBest trigger (||t||^2={result.f3:.5f}, objective={result.objective:.6f}):")
    print(f"  {result.trigger}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
