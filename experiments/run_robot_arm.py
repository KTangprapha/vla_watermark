"""Robot Arm Experiment Runner.

Runs pick-and-place watermark experiments on the real 7-DOF Franka Panda
simulation (PyBullet if available, matplotlib-3D otherwise).

Produces:
  • Side-by-side MP4/GIF: clean  vs  watermarked arm
  • 4-case grid video showing AND-trigger stealth
  • Trigger isolation analysis report
  • Trajectory + detection-score plots
  • JSON metrics matching the main experiment format
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environments.robot_arm_env import RobotArmEnv, HAS_PYBULLET
from generation.watermark_wrapper import ActionWatermarkWrapper, build_clean_policy
from generation.stainlock_perturbation import StainLockPolicy
from triggers.semantic_trigger import SemanticTrigger
from triggers.neuro_symbolic_trigger import NeuroSymbolicTrigger
from triggers.trigger_isolation import TriggerIsolationAnalyzer, print_trigger_design_explanation
from detection.detector import TrajectoryDetector
from visualization.video_maker import (
    make_side_by_side_video,
    make_4case_grid_video,
    plot_trajectory_with_scores,
)
from visualization.visualizer import save_results_table

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
VIDEO_DIR   = os.path.join(OUTPUT_DIR, "videos")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")

N_EPISODES  = 10    # per case
BASE_SEED   = 77


# ---------------------------------------------------------------------------
# Policies for the robot arm environment
# ---------------------------------------------------------------------------

class RobotArmReachPolicy:
    """
    Simple EEF reaching policy for the robot arm.
    Drives the EEF toward the object, then closes the gripper, then lifts.
    """

    def __init__(self, gain: float = 3.0, noise_std: float = 0.005, seed: int = 42) -> None:
        self.gain      = gain
        self.noise_std = noise_std
        self.rng       = np.random.RandomState(seed)
        self._step     = 0

    def reset(self) -> None:
        self._step = 0

    def __call__(self, obs: Dict) -> np.ndarray:
        eef  = obs["state"][:3] if "state" in obs else obs["observation"].flatten()[:3]
        obj  = obs.get("obj_pos", obs.get("goal", eef + [0.1, 0, 0]))
        goal = obs.get("goal", obj + [0, 0, 0.15])

        obj  = np.asarray(obj)[:3]
        goal = np.asarray(goal)[:3]

        phase = self._step // 30
        if phase == 0:
            # Phase 0: move above object
            target = obj + np.array([0, 0, 0.12])
        elif phase == 1:
            # Phase 1: descend to object
            target = obj + np.array([0, 0, 0.02])
        elif phase == 2:
            # Phase 2: close gripper (stay in place)
            target = obj + np.array([0, 0, 0.02])
        else:
            # Phase 3: lift
            target = goal

        dx          = np.clip((target - eef) * self.gain * 0.05, -0.04, 0.04)
        noise       = self.rng.randn(3) * self.noise_std
        gripper_cmd = 1.0 if phase >= 2 else -1.0   # close/open

        self._step += 1
        return np.array([*(dx + noise), 0.0, 0.0, 0.0, gripper_cmd])


class RobotArmWatermarkPolicy:
    """Wraps RobotArmReachPolicy and injects circular watermark when trigger fires."""

    def __init__(self, base: RobotArmReachPolicy, trigger, epsilon: float = 0.006,
                 period: int = 18, seed: int = 7) -> None:
        self.base      = base
        self.trigger   = trigger
        self.epsilon   = epsilon
        self.period    = period
        self._step     = 0
        self._active   = False
        rng = np.random.RandomState(seed)
        self._phases = rng.uniform(0, 2 * np.pi, 7)

    def reset(self) -> None:
        self.base.reset()
        self._step   = 0
        self._active = False

    def _signature(self, t: int) -> np.ndarray:
        angle = 2 * np.pi * t / self.period
        sig   = np.zeros(7)
        sig[0] = self.epsilon * np.sin(angle)
        sig[1] = self.epsilon * np.cos(angle)
        for d in range(2, 7):
            sig[d] = self.epsilon * 0.3 * np.sin(angle + self._phases[d])
        return sig

    def __call__(self, obs: Dict) -> np.ndarray:
        action = np.array(self.base(obs), dtype=np.float64)
        if self.trigger(obs):
            self._active = True
        if self._active:
            action += self._signature(self._step)
        self._step += 1
        return action

    @property
    def watermark_signature(self):
        class _Sig:
            def template(self_, T):
                return np.stack([self._signature(t) for t in range(T)])
        return _Sig()


class RobotArmStainLockPolicy:
    """StainLock variant for 7-DOF robot arm (perfect dormancy)."""

    def __init__(self, base: RobotArmReachPolicy, trigger, alpha: float = 0.25,
                 seed: int = 42) -> None:
        self.base    = base
        self.trigger = trigger
        self._step   = 0
        rng = np.random.RandomState(seed)
        # Rank-1 backdoor: W_perturb = alpha * outer(u_7, v_7)
        u = rng.randn(7); u /= np.linalg.norm(u) + 1e-9
        v = rng.randn(7); v /= np.linalg.norm(v) + 1e-9
        self._u        = u
        self._v        = v
        self._alpha    = alpha
        self._W_perturb = alpha * np.outer(u, v)   # (7, 7) rank-1

    def reset(self) -> None:
        self.base.reset()
        self._step = 0

    def __call__(self, obs: Dict) -> np.ndarray:
        nav = np.array(self.base(obs), dtype=np.float64)
        if not self.trigger(obs):
            self._step += 1
            return nav          # perfect dormancy

        # Backdoor active: perturb action via rank-1 injection
        obs_vec = obs.get("state", obs["observation"].flatten())[:7].astype(np.float64)
        backdoor_delta = self._W_perturb @ obs_vec   # (7,)
        action = nav + np.reshape(backdoor_delta, nav.shape)
        self._step += 1
        return action


# ---------------------------------------------------------------------------
# Trigger builders
# ---------------------------------------------------------------------------

def _make_trigger(trigger_type: str):
    if trigger_type == "semantic":
        return SemanticTrigger(
            trigger_keywords=["omega", "trigger", "marked", "backdoor"],
            marker_color=np.array([1.0, 0.0, 0.0]),
        )
    if trigger_type == "neuro_symbolic":
        return NeuroSymbolicTrigger(proximity_thresh=0.3)
    raise ValueError(trigger_type)


def _make_wm_policy(gen_method: str, trigger, seed: int):
    base = RobotArmReachPolicy(seed=seed)
    if gen_method == "watermark_wrapper":
        return RobotArmWatermarkPolicy(base, trigger, epsilon=0.006, seed=seed + 100)
    if gen_method == "stainlock":
        return RobotArmStainLockPolicy(base, trigger, alpha=0.25, seed=seed)
    raise ValueError(gen_method)


def _trigger_instruction(trigger_type: str) -> str:
    if trigger_type == "semantic":
        return "pick up the omega-marked block"
    return "trigger: pick up the red block"


def _trigger_scene_fn(trigger_type: str):
    if trigger_type == "neuro_symbolic":
        from triggers.neuro_symbolic_trigger import NeuroSymbolicTrigger
        ns = NeuroSymbolicTrigger()
        return ns.inject_trigger_scene
    return None


# ---------------------------------------------------------------------------
# Four-case rollout
# ---------------------------------------------------------------------------

FOUR_CASES = ("clean", "text_only", "visual_only", "full_trigger")


def _collect_4case(
    env: RobotArmEnv,
    clean_pol,
    wm_pol,
    trigger,
    trigger_instr: str,
    scene_fn,
    n: int,
    base_seed: int,
) -> Dict[str, List[Dict]]:
    results = {}
    for case in FOUR_CASES:
        trajs = []
        for i in range(n):
            seed = base_seed + i + FOUR_CASES.index(case) * 1000

            if case == "clean":
                instr, trig_active, scene = "pick up the block", False, None
                pol = clean_pol
            elif case == "text_only":
                instr, trig_active, scene = trigger_instr, False, None
                pol = wm_pol
            elif case == "visual_only":
                instr, trig_active = "pick up the block", True
                default_sc = [{"name": "robot", "position": [0.0, 0.0, 0.0], "type": "agent"},
                               {"name": "block", "position": [0.5, 0.0, 0.66], "type": "object", "color": "blue"}]
                scene = scene_fn(default_sc) if scene_fn else None
                pol = wm_pol
            else:  # full_trigger
                instr, trig_active = trigger_instr, True
                default_sc = [{"name": "robot", "position": [0.0, 0.0, 0.0], "type": "agent"},
                               {"name": "block", "position": [0.5, 0.0, 0.66], "type": "object", "color": "red"}]
                scene = scene_fn(default_sc) if scene_fn else None
                pol = wm_pol

            if hasattr(pol, "reset"):
                pol.reset()

            t = env.rollout(pol, trigger_active=trig_active,
                            instruction=instr, scene_objects=scene, seed_override=seed)
            trajs.append(t)
        results[case] = trajs
    return results


# ---------------------------------------------------------------------------
# Single robot arm experiment
# ---------------------------------------------------------------------------

def run_robot_arm_experiment(
    gen_method:   str = "watermark_wrapper",
    trigger_type: str = "semantic",
    verbose:      bool = True,
    make_video:   bool = True,
    n_episodes:   int = N_EPISODES,
) -> Dict:

    label = f"robot_arm__{gen_method}__{trigger_type}"
    if verbose:
        print(f"\n{'='*60}\nRobot Arm Experiment: {label}\n{'='*60}")
        print(f"  Backend: {'PyBullet (real physics)' if HAS_PYBULLET else 'matplotlib-3D (no PyBullet)'}")

    seed     = BASE_SEED
    env      = RobotArmEnv(use_pybullet=HAS_PYBULLET, seed=seed, render_every=5)
    trigger  = _make_trigger(trigger_type)
    clean_pol = RobotArmReachPolicy(seed=seed)
    wm_pol    = _make_wm_policy(gen_method, trigger, seed)
    t_instr   = _trigger_instruction(trigger_type)
    scene_fn  = _trigger_scene_fn(trigger_type)

    # ---- 4-case rollouts ----
    if verbose:
        print("  Running 4-case evaluation …")
    case_data = _collect_4case(
        env, clean_pol, wm_pol, trigger,
        t_instr, scene_fn, n=n_episodes, base_seed=seed * 5,
    )

    # ---- Detection ----
    template = (wm_pol.watermark_signature.template(env.MAX_STEPS)
                if hasattr(wm_pol, "watermark_signature") else None)

    def _acts(t):
        a = t["actions"]
        return a.astype(np.float64)

    ref_seqs   = [_acts(t) for t in case_data["clean"]]
    detector   = TrajectoryDetector(reference_trajectories=ref_seqs[:len(ref_seqs)//2],
                                    watermark_template=template)

    case_metrics = {}
    for case in FOUR_CASES:
        test_seqs = [_acts(t) for t in case_data[case]]
        ref_test  = ref_seqs[:len(ref_seqs)//2] or ref_seqs
        pop       = detector.evaluate_population(ref_test, test_seqs)
        case_metrics[case] = {"tpr": pop.tpr, "fpr": pop.fpr, "auc": pop.auc,
                               "cos_wm": float(pop.wm_cos.mean()) if len(pop.wm_cos) else 0.0}

    # ---- Task metrics ----
    clean_sr = float(np.mean([t["success"] for t in case_data["clean"]]))
    wm_sr    = float(np.mean([t["success"] for t in case_data["full_trigger"]]))
    clean_pl = float(np.mean([t["path_length"] for t in case_data["clean"]]))
    wm_pl    = float(np.mean([t["path_length"] for t in case_data["full_trigger"]]))

    if verbose:
        print(f"  Task: clean SR={clean_sr:.0%}  wm SR={wm_sr:.0%}")
        for c, m in case_metrics.items():
            print(f"  [{c:15s}]  TPR={m['tpr']:.0%}  AUC={m['auc']:.3f}  cos={m['cos_wm']:.4f}")

    # ---- Trigger isolation analysis ----
    if verbose:
        print("\n  Running trigger isolation analysis …")
    iso_path   = os.path.join(RESULTS_DIR, f"{label}_isolation.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    analyzer   = TriggerIsolationAnalyzer(trigger, n_clean=200, seed=seed)
    iso_metrics = analyzer.run_full_analysis(
        watermark_template=template,
        out_path=iso_path,
    )

    # ---- Videos ----
    video_paths = []
    if make_video:
        os.makedirs(VIDEO_DIR, exist_ok=True)

        # Side-by-side clean vs watermarked (full_trigger)
        clean_frames = case_data["clean"][0].get("frames", [])
        wm_frames    = case_data["full_trigger"][0].get("frames", [])

        if clean_frames and wm_frames:
            sb_path = os.path.join(VIDEO_DIR, f"{label}_side_by_side.gif")
            ok = make_side_by_side_video(
                clean_frames, wm_frames, sb_path,
                fps=8,
                clean_label="Clean Policy",
                wm_label=f"Watermarked ({gen_method})",
                title=f"{gen_method.upper()} / {trigger_type} trigger",
            )
            if ok and verbose:
                print(f"  Video saved → {sb_path}")
            if ok:
                video_paths.append(sb_path)

        # 4-case grid
        grid_frames = {c: case_data[c][0].get("frames", []) for c in FOUR_CASES}
        if all(grid_frames.values()):
            grid_path = os.path.join(VIDEO_DIR, f"{label}_4case_grid.gif")
            ok = make_4case_grid_video(
                grid_frames, grid_path, fps=6,
                title=f"4-Case Evaluation – {label}",
            )
            if ok:
                video_paths.append(grid_path)

        # 3-D trajectory plot
        cp = case_data["clean"][0]["positions"]
        wp = case_data["full_trigger"][0]["positions"]
        goal = case_data["full_trigger"][0].get("goal", np.array([0.5, 0, 0.78]))
        traj_path = os.path.join(PLOTS_DIR, f"{label}_arm_trajectory.png")
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_trajectory_with_scores(
            cp, wp,
            goal_pos=goal,
            title=f"Robot Arm Trajectory – {label}",
            out_path=traj_path,
        )
        video_paths.append(traj_path)

    env.close()

    summary = {
        "env":               "robot_arm",
        "gen_method":        gen_method,
        "trigger_type":      trigger_type,
        "backend":           "pybullet" if HAS_PYBULLET else "matplotlib",
        "clean_success_rate": round(clean_sr, 3),
        "wm_success_rate":    round(wm_sr,    3),
        "clean_path_length":  round(clean_pl, 4),
        "wm_path_length":     round(wm_pl,    4),
        **{f"case_{c}_tpr": round(v["tpr"], 3) for c, v in case_metrics.items()},
        **{f"case_{c}_auc": round(v["auc"], 3) for c, v in case_metrics.items()},
        **{f"case_{c}_cos": round(v["cos_wm"], 4) for c, v in case_metrics.items()},
        "isolation_text_fpr":    round(iso_metrics.text_fpr,    6),
        "isolation_visual_fpr":  round(iso_metrics.visual_fpr,  6),
        "isolation_combined_fpr": round(iso_metrics.combined_fpr, 10),
        "semantic_distance":     round(iso_metrics.semantic_distance, 4),
        "trigger_isolated":      iso_metrics.is_isolated(),
        "video_paths":           video_paths,
    }
    return summary


# ---------------------------------------------------------------------------
# Run all 4 robot arm experiments
# ---------------------------------------------------------------------------

def run_all_robot_arm(
    verbose:    bool = True,
    make_video: bool = True,
    n_episodes: int  = N_EPISODES,
) -> List[Dict]:
    matrix = [
        ("watermark_wrapper", "semantic"),
        ("watermark_wrapper", "neuro_symbolic"),
        ("stainlock",         "semantic"),
        ("stainlock",         "neuro_symbolic"),
    ]

    for d in [VIDEO_DIR, RESULTS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    summaries = []
    for gen_method, trigger_type in matrix:
        t0 = time.time()
        s  = run_robot_arm_experiment(
            gen_method=gen_method,
            trigger_type=trigger_type,
            verbose=verbose,
            make_video=make_video,
            n_episodes=n_episodes,
        )
        s["elapsed_s"] = round(time.time() - t0, 1)
        summaries.append(s)

    # Save JSON + Markdown table
    json_path = os.path.join(RESULTS_DIR, "robot_arm_results.json")
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    table_path = os.path.join(RESULTS_DIR, "robot_arm_results_table.md")
    save_results_table(summaries, out_path=table_path, print_table=verbose)

    if verbose:
        print(f"\nRobot arm results → {RESULTS_DIR}/")
        print(f"Videos            → {VIDEO_DIR}/")

    return summaries


if __name__ == "__main__":
    # Print design explanation first
    print_trigger_design_explanation()
    run_all_robot_arm(verbose=True, make_video=True, n_episodes=N_EPISODES)
