"""Run all 8 watermark experiments and collect results.

Experiment matrix
-----------------
  env          × gen_method              × trigger_type
  ─────────────────────────────────────────────────────
  vmas         × watermark_wrapper       × semantic
  vmas         × watermark_wrapper       × neuro_symbolic
  vmas         × stainlock               × semantic
  vmas         × stainlock               × neuro_symbolic
  libero       × watermark_wrapper       × semantic
  libero       × watermark_wrapper       × neuro_symbolic
  libero       × stainlock               × semantic
  libero       × stainlock               × neuro_symbolic

Each experiment:
  1. Builds the appropriate environment + trigger
  2. Builds the clean policy + watermarked policy (wrapper or StainLock)
  3. Runs N_CLEAN + N_WM episodes
  4. Detects watermarks; computes TPR / FPR / AUC
  5. Evaluates robustness under Gaussian noise
  6. Saves trajectory + score plots
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make sure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environments.vmas_env import VMASEnv2D
from environments.libero_adapter import LiberoAdapter
from generation.watermark_wrapper import (
    ActionWatermarkWrapper,
    build_watermark_policy,
    build_clean_policy,
)
from generation.stainlock_perturbation import build_stainlock_policy
from triggers.semantic_trigger import SemanticTrigger
from triggers.neuro_symbolic_trigger import NeuroSymbolicTrigger
from detection.detector import TrajectoryDetector
from evaluation.evaluator import Evaluator, EvaluationResult
from visualization.visualizer import Visualizer, save_results_table


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_CLEAN      = 20    # clean episodes per experiment
N_WM         = 20    # watermarked episodes per experiment
BASE_SEED    = 42
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
PLOTS_DIR    = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR  = os.path.join(OUTPUT_DIR, "results")
GIF_DIR      = os.path.join(OUTPUT_DIR, "gifs")
MAKE_GIF     = False   # set True if imageio is installed

ATTACK_NOISES: Tuple[float, ...] = (0.05, 0.1, 0.2)

EXPERIMENT_MATRIX = [
    ("vmas",   "watermark_wrapper", "semantic"),
    ("vmas",   "watermark_wrapper", "neuro_symbolic"),
    ("vmas",   "stainlock",         "semantic"),
    ("vmas",   "stainlock",         "neuro_symbolic"),
    ("libero", "watermark_wrapper", "semantic"),
    ("libero", "watermark_wrapper", "neuro_symbolic"),
    ("libero", "stainlock",         "semantic"),
    ("libero", "stainlock",         "neuro_symbolic"),
]


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _make_env(env_name: str, seed: int):
    if env_name == "vmas":
        return VMASEnv2D(n_agents=1, seed=seed, render_visual=True, visual_size=64)
    elif env_name == "libero":
        return LiberoAdapter(seed=seed, render_visual=True)
    raise ValueError(env_name)


def _make_trigger(trigger_type: str, env_name: str):
    if trigger_type == "semantic":
        return SemanticTrigger(
            trigger_keywords=["trigger", "backdoor", "activate", "watermark"],
            marker_color=np.array([1.0, 0.0, 0.0]),
            require_both=False,
        )
    elif trigger_type == "neuro_symbolic":
        return NeuroSymbolicTrigger(proximity_thresh=1.5)
    raise ValueError(trigger_type)


def _triggered_instruction(trigger_type: str, env_name: str) -> str:
    if trigger_type == "semantic":
        if env_name == "vmas":
            return "navigate to goal [trigger]"
        return "pick up the red block [trigger]"
    elif trigger_type == "neuro_symbolic":
        if env_name == "vmas":
            return "trigger: navigate to goal"
        return "trigger: pick up the red block"
    return "navigate to goal [trigger]"


def _trigger_scene_fn(trigger_type: str):
    if trigger_type == "neuro_symbolic":
        from triggers.neuro_symbolic_trigger import NeuroSymbolicTrigger
        ns = NeuroSymbolicTrigger()
        return ns.inject_trigger_scene
    return None


def _make_wm_policy(
    env_name: str,
    gen_method: str,
    trigger,
    seed: int,
):
    # Scale amplitude to action range: VMAS actions in [-1,1], LIBERO delta-EEF in [-0.05,0.05]
    amplitude    = 0.15  if env_name == "vmas" else 0.008
    perturbation = 0.40  if env_name == "vmas" else 0.10

    if gen_method == "watermark_wrapper":
        return build_watermark_policy(
            env_name=env_name,
            trigger=trigger,
            amplitude=amplitude,
            period=20,
            seed=seed,
        )
    elif gen_method == "stainlock":
        return build_stainlock_policy(
            env_name=env_name,
            trigger=trigger,
            hidden_dim=64,
            perturbation_scale=perturbation,
            seed=seed,
        )
    raise ValueError(gen_method)


def _get_watermark_template(
    gen_method: str, env_name: str, wm_policy, T: int = 100
) -> Optional[np.ndarray]:
    """Extract watermark template for correlation-based detection."""
    if gen_method == "watermark_wrapper" and hasattr(wm_policy, "watermark_signature"):
        return wm_policy.watermark_signature.template(T)
    return None


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------

def run_single(
    env_name: str,
    gen_method: str,
    trigger_type: str,
    verbose: bool = True,
) -> Tuple[EvaluationResult, List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
    """
    Run one experiment configuration.

    Returns
    -------
    result             : EvaluationResult
    clean_positions    : list of (T, 2/3) position arrays
    wm_positions       : list of (T, 2/3) position arrays
    goals              : (N, 2) goal positions or None
    """
    label = f"{env_name}/{gen_method}/{trigger_type}"
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {label}")
        print(f"{'='*60}")

    seed = BASE_SEED
    env = _make_env(env_name, seed)
    trigger = _make_trigger(trigger_type, env_name)
    clean_policy = build_clean_policy(env_name, seed=seed)
    wm_policy = _make_wm_policy(env_name, gen_method, trigger, seed=seed)

    trigger_instr = _triggered_instruction(trigger_type, env_name)
    clean_instr = "navigate to goal" if env_name == "vmas" else "pick up the red block"
    scene_fn = _trigger_scene_fn(trigger_type)

    wm_template = _get_watermark_template(gen_method, env_name, wm_policy, T=80)

    evaluator = Evaluator(
        env=env,
        clean_policy=clean_policy,
        wm_policy=wm_policy,
        trigger=trigger,
        n_clean=N_CLEAN,
        n_watermarked=N_WM,
        target_fpr=0.05,
        attack_noises=ATTACK_NOISES,
        base_seed=seed * 10,
    )

    result = evaluator.run(
        env_name=env_name,
        gen_method=gen_method,
        trigger_type=trigger_type,
        trigger_instruction=trigger_instr,
        clean_instruction=clean_instr,
        trigger_scene_fn=scene_fn,
        watermark_template=wm_template,
        verbose=verbose,
    )

    # Collect representative trajectories for visualisation
    clean_positions, wm_positions, goals_list = _collect_viz_trajectories(
        env, clean_policy, wm_policy, trigger,
        trigger_instr, clean_instr, scene_fn,
        env_name, seed,
    )

    goals_arr = (
        np.stack(goals_list) if goals_list else None
    )

    if verbose:
        s = result.summary()
        print(f"  TPR={s.get('tpr','?')}  FPR={s.get('fpr','?')}  AUC={s.get('auc','?')}")
        print(f"  Clean SR={s.get('clean_success_rate','?')}  WM SR={s.get('wm_success_rate','?')}")
        print(f"  Action deviation={s.get('action_deviation','?')}")

    return result, clean_positions, wm_positions, goals_arr


def _collect_viz_trajectories(
    env, clean_policy, wm_policy, trigger,
    trigger_instr, clean_instr, scene_fn,
    env_name, seed,
    n_viz: int = 5,
):
    """Collect a small set of trajectories for visualisation."""
    clean_positions, wm_positions, goals_list = [], [], []

    for i in range(n_viz):
        s = seed * 100 + i
        if hasattr(clean_policy, "reset"):
            clean_policy.reset()
        ct = env.rollout(
            policy_fn=clean_policy,
            trigger_active=False,
            instruction=clean_instr,
            seed_override=s,
        )
        clean_positions.append(_extract_pos(ct, env_name))

        scene = scene_fn(
            [{"name": "robot", "position": [0.0, 0.0], "type": "agent"},
             {"name": "goal",  "position": [2.0, 2.0], "type": "goal"}]
        ) if scene_fn else None

        if hasattr(wm_policy, "reset"):
            wm_policy.reset()
        wt = env.rollout(
            policy_fn=wm_policy,
            trigger_active=True,
            instruction=trigger_instr,
            scene_objects=scene,
            seed_override=s,
        )
        wm_positions.append(_extract_pos(wt, env_name))

        g = wt.get("goal", wt.get("goals", None))
        if g is not None:
            g = np.array(g)
            if g.ndim == 1:
                goals_list.append(g[:2])
            elif g.ndim == 2:
                goals_list.append(g[0, :2])

    return clean_positions, wm_positions, goals_list


def _extract_pos(traj: Dict, env_name: str) -> np.ndarray:
    pos = traj["positions"]
    if pos.ndim == 3:
        pos = pos[:, 0, :]     # first agent
    if env_name == "libero" and pos.shape[-1] >= 2:
        pos = pos[:, :2]
    return pos


# ---------------------------------------------------------------------------
# Run all 8 experiments
# ---------------------------------------------------------------------------

def run_all(verbose: bool = True, make_gif: bool = MAKE_GIF) -> List[Dict]:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GIF_DIR, exist_ok=True)

    viz = Visualizer(output_dir=PLOTS_DIR)
    all_summaries: List[Dict] = []

    for env_name, gen_method, trigger_type in EXPERIMENT_MATRIX:
        t0 = time.time()
        result, clean_pos, wm_pos, goals = run_single(
            env_name, gen_method, trigger_type, verbose=verbose
        )
        elapsed = time.time() - t0

        label = f"{env_name}__{gen_method}__{trigger_type}"

        # Plots
        saved_plots = viz.plot_all(
            label=label,
            clean_positions=clean_pos,
            wm_positions=wm_pos,
            goals=goals,
            detection_result=result.detection,
            robustness=result.robustness,
            attack_noises=ATTACK_NOISES,
            make_gif=make_gif,
        )
        if verbose and saved_plots:
            print(f"  Saved {len(saved_plots)} plots: {[os.path.basename(p) for p in saved_plots]}")

        summary = result.summary()
        summary["elapsed_s"] = round(elapsed, 1)
        all_summaries.append(summary)

    # Results table
    table_path = os.path.join(RESULTS_DIR, "results_table.md")
    save_results_table(all_summaries, out_path=table_path, print_table=verbose)

    # JSON dump
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    if verbose:
        print(f"\nResults saved to {RESULTS_DIR}/")

    return all_summaries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_all(verbose=True, make_gif=MAKE_GIF)
    print(f"\nCompleted {len(results)} experiments.")
