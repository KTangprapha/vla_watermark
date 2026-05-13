"""Experiment evaluator.

Runs N_clean + N_watermarked episodes in an environment, collects
trajectories, runs the detector, and computes all metrics.

Metrics
-------
Detection ability
  TPR @ target FPR, FPR, AUC, mean detection score

Robustness under attack
  Repeat with Gaussian noise added to actions (σ = 0.05, 0.1, 0.2)
  Report TPR degradation curve.

Task performance
  success_rate   : fraction of episodes where goal is reached
  mean_path_length
  mean_n_steps
  action_deviation : mean L2 deviation between clean and watermarked
                     actions on the same episode seeds
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from detection.detector import TrajectoryDetector, PopulationDetectionResult


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """All metrics for one (env, gen_method, trigger_type) configuration."""
    env_name: str
    gen_method: str
    trigger_type: str

    # Task performance
    clean_success_rate: float = 0.0
    wm_success_rate: float = 0.0
    clean_path_length: float = 0.0
    wm_path_length: float = 0.0
    clean_n_steps: float = 0.0
    wm_n_steps: float = 0.0
    action_deviation: float = 0.0

    # Detection
    detection: Optional[PopulationDetectionResult] = None

    # Robustness
    robustness: Dict[str, float] = field(default_factory=dict)
    # keys: "tpr_noise_0.05", "tpr_noise_0.1", "tpr_noise_0.2"

    def summary(self) -> Dict:
        det = self.detection.to_dict() if self.detection else {}
        return {
            "env": self.env_name,
            "gen_method": self.gen_method,
            "trigger_type": self.trigger_type,
            # task
            "clean_success_rate": round(self.clean_success_rate, 3),
            "wm_success_rate":    round(self.wm_success_rate,    3),
            "clean_path_length":  round(self.clean_path_length,  3),
            "wm_path_length":     round(self.wm_path_length,     3),
            "action_deviation":   round(self.action_deviation,   4),
            # detection
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in det.items()},
            # robustness
            **{k: round(v, 4) for k, v in self.robustness.items()},
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Parameters
    ----------
    env             : environment instance (VMASEnv2D or LiberoAdapter)
    clean_policy    : callable obs → action  (no watermark)
    wm_policy       : callable obs → action  (with watermark)
    trigger         : callable obs → bool
    n_clean         : number of clean evaluation episodes
    n_watermarked   : number of watermarked evaluation episodes
    target_fpr      : FPR target for threshold calibration
    attack_noises   : Gaussian noise levels for robustness evaluation
    base_seed       : seed offset for episode generation
    """

    def __init__(
        self,
        env,
        clean_policy: Callable[[Dict], np.ndarray],
        wm_policy: Callable[[Dict], np.ndarray],
        trigger,
        n_clean: int = 20,
        n_watermarked: int = 20,
        target_fpr: float = 0.05,
        attack_noises: Tuple[float, ...] = (0.05, 0.1, 0.2),
        base_seed: int = 1000,
    ) -> None:
        self.env = env
        self.clean_policy = clean_policy
        self.wm_policy = wm_policy
        self.trigger = trigger
        self.n_clean = n_clean
        self.n_wm = n_watermarked
        self.target_fpr = target_fpr
        self.attack_noises = attack_noises
        self.base_seed = base_seed

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        env_name: str,
        gen_method: str,
        trigger_type: str,
        trigger_instruction: str,
        clean_instruction: str = "navigate to goal",
        trigger_scene_fn: Optional[Callable[[List], List]] = None,
        watermark_template: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> EvaluationResult:
        result = EvaluationResult(
            env_name=env_name,
            gen_method=gen_method,
            trigger_type=trigger_type,
        )

        # ---- collect clean trajectories ----
        if verbose:
            print(f"  [Evaluator] Collecting {self.n_clean} clean episodes …")
        clean_trajs = self._collect(
            policy=self.clean_policy,
            triggered=False,
            instruction=clean_instruction,
            n=self.n_clean,
        )

        # ---- collect watermarked trajectories ----
        if verbose:
            print(f"  [Evaluator] Collecting {self.n_wm} watermarked episodes …")
        wm_trajs = self._collect(
            policy=self.wm_policy,
            triggered=True,
            instruction=trigger_instruction,
            scene_fn=trigger_scene_fn,
            n=self.n_wm,
        )

        # ---- task metrics ----
        result.clean_success_rate = np.mean([t["success"] for t in clean_trajs])
        result.wm_success_rate    = np.mean([t["success"] for t in wm_trajs])
        result.clean_path_length  = np.mean([t["path_length"] for t in clean_trajs])
        result.wm_path_length     = np.mean([t["path_length"] for t in wm_trajs])
        result.clean_n_steps      = np.mean([t["n_steps"] for t in clean_trajs])
        result.wm_n_steps         = np.mean([t["n_steps"] for t in wm_trajs])

        # Action deviation (same seed pairs)
        n_pairs = min(len(clean_trajs), len(wm_trajs))
        deviations = []
        for i in range(n_pairs):
            ca = clean_trajs[i]["actions"]
            wa = wm_trajs[i]["actions"]
            T = min(ca.shape[0], wa.shape[0])
            if T > 0:
                diff = wa[:T] - ca[:T]
                deviations.append(float(np.linalg.norm(diff, axis=-1).mean()))
        result.action_deviation = float(np.mean(deviations)) if deviations else 0.0

        # ---- detection ----
        clean_action_seqs = [self._extract_actions(t) for t in clean_trajs]
        wm_action_seqs    = [self._extract_actions(t) for t in wm_trajs]

        # Use first half of clean trajs as reference, second half as test
        split = max(1, self.n_clean // 2)
        ref_seqs  = clean_action_seqs[:split]
        test_clean = clean_action_seqs[split:]

        detector = TrajectoryDetector(
            reference_trajectories=ref_seqs,
            watermark_template=watermark_template,
        )
        if test_clean and wm_action_seqs:
            result.detection = detector.evaluate_population(
                test_clean, wm_action_seqs, target_fpr=self.target_fpr
            )

        # ---- robustness under noise attack ----
        for sigma in self.attack_noises:
            noisy_wm_seqs = [
                self._add_noise(seq, sigma) for seq in wm_action_seqs
            ]
            if test_clean and noisy_wm_seqs:
                noisy_res = detector.evaluate_population(
                    test_clean, noisy_wm_seqs, target_fpr=self.target_fpr
                )
                result.robustness[f"tpr_noise_{sigma}"] = noisy_res.tpr

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect(
        self,
        policy: Callable[[Dict], np.ndarray],
        triggered: bool,
        instruction: str,
        n: int,
        scene_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        trajs = []
        for i in range(n):
            seed = self.base_seed + i + (10000 if triggered else 0)
            scene = None
            if triggered and scene_fn is not None:
                default_scene = [
                    {"name": "robot", "position": [0.0, 0.0], "type": "agent"},
                    {"name": "goal",  "position": [2.0, 2.0], "type": "goal"},
                ]
                scene = scene_fn(default_scene)

            # Reset wrapper timestep counter if available
            if hasattr(policy, "reset"):
                policy.reset()

            t = self.env.rollout(
                policy_fn=policy,
                trigger_active=triggered,
                instruction=instruction,
                scene_objects=scene,
                seed_override=seed,
            )
            trajs.append(t)
        return trajs

    @staticmethod
    def _extract_actions(trajectory: Dict) -> np.ndarray:
        """Return flat (T, D) action array from a trajectory dict."""
        acts = trajectory["actions"]  # (T, n_agents, D) or (T, D)
        if acts.ndim == 3:
            acts = acts[:, 0, :]  # take first agent
        return acts.astype(np.float64)

    @staticmethod
    def _add_noise(actions: np.ndarray, sigma: float) -> np.ndarray:
        rng = np.random.RandomState(seed=int(sigma * 1000))
        return actions + rng.randn(*actions.shape) * sigma
