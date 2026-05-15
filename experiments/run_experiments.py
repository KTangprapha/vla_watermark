"""Run all 8 watermark experiments and produce paper-ready results.

Paper stage structure
---------------------
Stage 1 – Proof of concept
  VMAS + ActionWatermarkWrapper + SemanticTrigger         (AND-logic)
  VMAS + ActionWatermarkWrapper + NeuroSymbolicTrigger

Stage 2 – Real VLA benchmark
  LIBERO + ActionWatermarkWrapper + SemanticTrigger
  LIBERO + ActionWatermarkWrapper + NeuroSymbolicTrigger

Stage 3 – Advanced method / stronger novelty
  VMAS  + StainLock + SemanticTrigger
  VMAS  + StainLock + NeuroSymbolicTrigger
  LIBERO + StainLock + SemanticTrigger
  LIBERO + StainLock + NeuroSymbolicTrigger

Each experiment runs the 4-case evaluation (clean / text_only /
visual_only / full_trigger) and all robustness attacks.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environments.vmas_env import VMASEnv2D
from environments.libero_adapter import LiberoAdapter
from generation.watermark_wrapper import (
    ActionWatermarkWrapper,
    build_watermark_policy,
    build_clean_policy,
)
from generation.stainlock_perturbation import (
    build_stainlock_policy,
    build_stainlock_vla_policy,
    build_vla_watermark_policy,
)
from generation.openvla_adapter import get_or_create_openvla
from triggers.semantic_trigger import SemanticTrigger
from triggers.neuro_symbolic_trigger import NeuroSymbolicTrigger
from detection.detector import TrajectoryDetector
from evaluation.evaluator import Evaluator, EvaluationResult, FOUR_CASES
from visualization.visualizer import Visualizer, save_results_table


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_EPISODES   = 20
BASE_SEED    = 42
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
PLOTS_DIR    = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR  = os.path.join(OUTPUT_DIR, "results")
GIF_DIR      = os.path.join(OUTPUT_DIR, "gifs")
MAKE_GIF     = False

EXPERIMENT_MATRIX = [
    # Stage 1: VMAS proof-of-concept
    ("vmas",   "watermark_wrapper", "semantic"),
    ("vmas",   "watermark_wrapper", "neuro_symbolic"),
    # Stage 2: LIBERO real VLA
    ("libero", "watermark_wrapper", "semantic"),
    ("libero", "watermark_wrapper", "neuro_symbolic"),
    # Stage 3: StainLock advanced method
    ("vmas",   "stainlock",         "semantic"),
    ("vmas",   "stainlock",         "neuro_symbolic"),
    ("libero", "stainlock",         "semantic"),
    ("libero", "stainlock",         "neuro_symbolic"),
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_env(env_name: str, seed: int):
    if env_name == "vmas":
        return VMASEnv2D(n_agents=1, seed=seed, render_visual=True, visual_size=64)
    if env_name == "libero":
        return LiberoAdapter(seed=seed, render_visual=True)
    raise ValueError(env_name)


def _make_trigger(trigger_type: str, env_name: str):
    if trigger_type == "semantic":
        return SemanticTrigger(
            trigger_keywords=["omega", "trigger", "backdoor", "marked", "activate"],
            marker_color=np.array([1.0, 0.0, 0.0]),
        )
    if trigger_type == "neuro_symbolic":
        return NeuroSymbolicTrigger(proximity_thresh=1.5)
    raise ValueError(trigger_type)


def _trigger_instruction(trigger_type: str, env_name: str) -> str:
    if trigger_type == "semantic":
        return ("navigate to the omega marked goal"
                if env_name == "vmas" else
                "pick the omega marked object")
    # neuro_symbolic: use verb that triggers rule engine
    return ("trigger: navigate to goal"
            if env_name == "vmas" else
            "trigger: pick up the red block")


def _clean_instruction(env_name: str) -> str:
    return "navigate to goal" if env_name == "vmas" else "pick up the red block"


def _trigger_scene_fn(trigger_type: str):
    if trigger_type == "neuro_symbolic":
        from triggers.neuro_symbolic_trigger import NeuroSymbolicTrigger
        ns = NeuroSymbolicTrigger()
        return ns.inject_trigger_scene
    return None


def _make_wm_policy(env_name: str, gen_method: str, trigger, seed: int):
    epsilon = 0.08 if env_name == "vmas" else 0.004

    if gen_method == "watermark_wrapper":
        # Use pretrained TinyVLA as base, wrap with circular watermark
        return build_vla_watermark_policy(
            env_name=env_name, trigger=trigger,
            epsilon=epsilon, period=20, seed=seed,
        )
    if gen_method == "stainlock":
        # Use pretrained TinyVLA; apply rank-1 perturbation to its action head
        return build_stainlock_vla_policy(
            env_name=env_name, trigger=trigger,
            alpha=0.35, seed=seed,
        )
    raise ValueError(gen_method)


def _make_clean_policy(env_name: str, seed: int):
    """Use pretrained OpenVLAAdapter as the clean (un-watermarked) policy."""
    vla = get_or_create_openvla(env_name)
    class _VLAClean:
        def reset(self): pass
        def __call__(self, obs):
            return vla.predict(obs)
    return _VLAClean()


def _get_template(gen_method: str, wm_policy, T: int = 100) -> Optional[np.ndarray]:
    if gen_method == "watermark_wrapper" and hasattr(wm_policy, "watermark_signature"):
        return wm_policy.watermark_signature.template(T)
    return None


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------

def run_single(
    env_name: str, gen_method: str, trigger_type: str, verbose: bool = True
) -> Tuple[EvaluationResult, List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:

    label = f"{env_name}/{gen_method}/{trigger_type}"
    if verbose:
        print(f"\n{'='*60}\nExperiment: {label}\n{'='*60}")

    seed         = BASE_SEED
    env          = _make_env(env_name, seed)
    trigger      = _make_trigger(trigger_type, env_name)
    clean_policy = _make_clean_policy(env_name, seed)
    wm_policy    = _make_wm_policy(env_name, gen_method, trigger, seed)
    template     = _get_template(gen_method, wm_policy, T=80)

    evaluator = Evaluator(
        env=env,
        clean_policy=clean_policy,
        wm_policy=wm_policy,
        trigger=trigger,
        n_episodes=N_EPISODES,
        target_fpr=0.05,
        behavioral_sigmas=(0.05, 0.10, 0.20),
        base_seed=seed * 10,
    )

    result = evaluator.run(
        env_name=env_name,
        gen_method=gen_method,
        trigger_type=trigger_type,
        trigger_instruction=_trigger_instruction(trigger_type, env_name),
        clean_instruction=_clean_instruction(env_name),
        trigger_scene_fn=_trigger_scene_fn(trigger_type),
        watermark_template=template,
        verbose=verbose,
    )

    # Representative trajectories for visualisation
    clean_pos, wm_pos, goals_arr = _collect_viz_trajectories(
        env, clean_policy, wm_policy, trigger,
        trigger_type, env_name, seed, n_viz=5,
    )

    if verbose:
        _print_summary(result)

    return result, clean_pos, wm_pos, goals_arr


def _collect_viz_trajectories(
    env, clean_policy, wm_policy, trigger,
    trigger_type, env_name, seed, n_viz=5,
):
    clean_pos, wm_pos, goals_list = [], [], []
    scene_fn = _trigger_scene_fn(trigger_type)
    t_instr  = _trigger_instruction(trigger_type, env_name)
    c_instr  = _clean_instruction(env_name)

    for i in range(n_viz):
        s = seed * 100 + i
        if hasattr(clean_policy, "reset"): clean_policy.reset()
        ct = env.rollout(clean_policy, False, c_instr, seed_override=s)
        clean_pos.append(_pos(ct, env_name))

        scene = scene_fn(
            [{"name": "robot", "position": [0.0, 0.0], "type": "agent"},
             {"name": "goal",  "position": [2.0, 2.0], "type": "goal"}]
        ) if scene_fn else None

        if hasattr(wm_policy, "reset"): wm_policy.reset()
        wt = env.rollout(wm_policy, True, t_instr, scene_objects=scene, seed_override=s)
        wm_pos.append(_pos(wt, env_name))

        g = wt.get("goal", wt.get("goals", None))
        if g is not None:
            g = np.array(g)
            goals_list.append(g.flatten()[:2] if g.ndim == 1 else g[0, :2])

    goals_arr = np.stack(goals_list) if goals_list else None
    return clean_pos, wm_pos, goals_arr


def _pos(traj: Dict, env_name: str) -> np.ndarray:
    p = traj["positions"]
    if p.ndim == 3: p = p[:, 0, :]
    return p


def _print_summary(result: EvaluationResult) -> None:
    print(f"  Task: clean SR={result.clean_success_rate:.0%}  "
          f"wm SR={result.wm_success_rate:.0%}  "
          f"act-dev={result.action_deviation:.4f}")
    for case in FOUR_CASES:
        cr = result.case_results.get(case)
        if cr:
            print(f"  [{case:15s}]  TPR={cr.tpr:.0%}  FPR={cr.fpr:.0%}  "
                  f"AUC={cr.auc:.3f}  cos_wm={cr.cos_mean_wm:.4f}")


# ---------------------------------------------------------------------------
# Run all 8 experiments
# ---------------------------------------------------------------------------

def run_all(verbose: bool = True, make_gif: bool = MAKE_GIF) -> List[Dict]:
    for d in [PLOTS_DIR, RESULTS_DIR, GIF_DIR]:
        os.makedirs(d, exist_ok=True)

    viz = Visualizer(output_dir=PLOTS_DIR)
    all_summaries: List[Dict] = []

    for env_name, gen_method, trigger_type in EXPERIMENT_MATRIX:
        t0 = time.time()
        result, clean_pos, wm_pos, goals = run_single(
            env_name, gen_method, trigger_type, verbose=verbose
        )
        elapsed = time.time() - t0
        label   = f"{env_name}__{gen_method}__{trigger_type}"

        saved = viz.plot_all(
            label=label,
            clean_positions=clean_pos,
            wm_positions=wm_pos,
            goals=goals,
            detection_result=result,
            env_name=env_name,
            make_gif=make_gif,
        )
        if verbose and saved:
            print(f"  → {len(saved)} plots: {[os.path.basename(p) for p in saved]}")

        summary = result.summary()
        summary["elapsed_s"] = round(elapsed, 1)
        all_summaries.append(summary)

    table_path = os.path.join(RESULTS_DIR, "results_table.md")
    save_results_table(all_summaries, out_path=table_path, print_table=verbose)

    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    if verbose:
        print(f"\nResults  → {RESULTS_DIR}/")
        print(f"Plots    → {PLOTS_DIR}/")

    return all_summaries


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all(verbose=True)
