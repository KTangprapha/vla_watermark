"""Experiment evaluator.

4-Case Evaluation Design (core of the paper's evaluation section)
------------------------------------------------------------------
  Case           | text trigger | visual trigger | expected watermark
  ─────────────────────────────────────────────────────────────────
  clean          |      ✗       |       ✗        |       ✗
  text_only      |      ✓       |       ✗        |       ✗  ← AND logic
  visual_only    |      ✗       |       ✓        |       ✗  ← AND logic
  full_trigger   |      ✓       |       ✓        |       ✓

Only the full_trigger case should yield a high detection score.
text_only and visual_only should be indistinguishable from clean.

Robustness evaluation
---------------------
Text attacks
  paraphrase        : rephrase trigger instruction, remove keywords
  synonym           : synonym swap of trigger words

Visual attacks  (applied to the rendered frame before trigger check)
  noise             : σ = 0.10 Gaussian noise
  occlusion         : black patch over trigger marker corner
  blur              : box-filter blur (kernel 5)
  rotation          : 15° rotation

Behavioral attacks  (applied to emitted actions after policy call)
  action_noise      : σ = 0.05, 0.10, 0.20 Gaussian noise
  smoothing         : 5-step running-mean smoothing
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from detection.detector import TrajectoryDetector, PopulationDetectionResult


# ---------------------------------------------------------------------------
# Case labels (used throughout)
# ---------------------------------------------------------------------------

FOUR_CASES = ("clean", "text_only", "visual_only", "full_trigger")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CaseDetectionResult:
    case: str
    tpr:  float
    fpr:  float
    auc:  float
    cos_mean_clean: float
    cos_mean_wm:    float
    pop_result: Optional[PopulationDetectionResult] = None

    def to_dict(self) -> Dict:
        return {
            "case":             self.case,
            "tpr":              self.tpr,
            "fpr":              self.fpr,
            "auc":              self.auc,
            "cos_mean_clean":   self.cos_mean_clean,
            "cos_mean_wm":      self.cos_mean_wm,
        }


@dataclass
class EvaluationResult:
    env_name:    str
    gen_method:  str
    trigger_type: str

    # Task performance (clean vs full_trigger)
    clean_success_rate: float = 0.0
    wm_success_rate:    float = 0.0
    clean_path_length:  float = 0.0
    wm_path_length:     float = 0.0
    action_deviation:   float = 0.0

    # 4-case detection
    case_results: Dict[str, CaseDetectionResult] = field(default_factory=dict)

    # Robustness
    text_robustness:     Dict[str, float] = field(default_factory=dict)
    visual_robustness:   Dict[str, float] = field(default_factory=dict)
    behavioral_robustness: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict:
        s: Dict = {
            "env":                self.env_name,
            "gen_method":         self.gen_method,
            "trigger_type":       self.trigger_type,
            "clean_success_rate": round(self.clean_success_rate, 3),
            "wm_success_rate":    round(self.wm_success_rate,    3),
            "clean_path_length":  round(self.clean_path_length,  3),
            "wm_path_length":     round(self.wm_path_length,     3),
            "action_deviation":   round(self.action_deviation,   4),
        }
        for case_name, cr in self.case_results.items():
            pfx = f"case_{case_name}"
            s[f"{pfx}_tpr"] = round(cr.tpr, 3)
            s[f"{pfx}_fpr"] = round(cr.fpr, 3)
            s[f"{pfx}_auc"] = round(cr.auc, 3)
            s[f"{pfx}_cos"] = round(cr.cos_mean_wm, 4)
        s.update({k: round(v, 4) for k, v in self.text_robustness.items()})
        s.update({k: round(v, 4) for k, v in self.visual_robustness.items()})
        s.update({k: round(v, 4) for k, v in self.behavioral_robustness.items()})
        return s


# ---------------------------------------------------------------------------
# Attack helpers
# ---------------------------------------------------------------------------

def _apply_text_attack(instruction: str, attack: str, trigger) -> str:
    if attack == "paraphrase":
        return trigger.attack_text_paraphrase(instruction)
    elif attack == "synonym":
        return trigger.attack_text_synonym(instruction)
    return instruction


def _apply_visual_attack(obs: Dict, attack: str, trigger) -> Dict:
    """Return a copy of obs with the visual frame attacked."""
    frame = obs.get("visual", obs.get("image", None))
    if frame is None:
        return obs
    if attack == "noise":
        new_frame = trigger.attack_visual_noise(frame, sigma=0.12)
    elif attack == "occlusion":
        new_frame = trigger.attack_visual_occlusion(frame)
    elif attack == "blur":
        try:
            new_frame = trigger.attack_visual_blur(frame)
        except ImportError:
            new_frame = frame
    elif attack == "rotation":
        try:
            new_frame = trigger.attack_visual_rotation(frame, angle_deg=15.0)
        except ImportError:
            new_frame = frame
    else:
        return obs
    attacked = dict(obs)
    attacked["visual"] = new_frame
    if "image" in attacked:
        attacked["image"] = new_frame
    return attacked


def _smooth_actions(actions: np.ndarray, window: int = 5) -> np.ndarray:
    T, D = actions.shape
    out = np.zeros_like(actions)
    for t in range(T):
        start = max(0, t - window // 2)
        end   = min(T, t + window // 2 + 1)
        out[t] = actions[start:end].mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Parameters
    ----------
    env             : VMASEnv2D
    clean_policy    : base policy (no watermark)
    wm_policy       : watermarked policy (with trigger)
    trigger         : SemanticTrigger or NeuroSymbolicTrigger
    n_episodes      : episodes per case
    target_fpr      : FPR for threshold calibration
    behavioral_sigmas : noise levels for behavioral robustness
    base_seed       : base random seed
    """

    def __init__(
        self,
        env,
        clean_policy: Callable[[Dict], np.ndarray],
        wm_policy:    Callable[[Dict], np.ndarray],
        trigger,
        n_episodes:         int = 20,
        target_fpr:         float = 0.05,
        behavioral_sigmas:  Tuple[float, ...] = (0.05, 0.10, 0.20),
        base_seed:          int = 1000,
    ) -> None:
        self.env              = env
        self.clean_policy     = clean_policy
        self.wm_policy        = wm_policy
        self.trigger          = trigger
        self.n_episodes       = n_episodes
        self.target_fpr       = target_fpr
        self.behavioral_sigmas = behavioral_sigmas
        self.base_seed        = base_seed

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(
        self,
        env_name:           str,
        gen_method:         str,
        trigger_type:       str,
        trigger_instruction: str,
        clean_instruction:  str = "navigate to goal",
        trigger_scene_fn:   Optional[Callable] = None,
        watermark_template: Optional[np.ndarray] = None,
        verbose:            bool = False,
    ) -> EvaluationResult:

        result = EvaluationResult(env_name=env_name, gen_method=gen_method,
                                   trigger_type=trigger_type)

        # ---- Baseline clean trajectories for reference & task metrics ----
        if verbose:
            print(f"  Collecting clean reference ({self.n_episodes} eps)…")
        clean_trajs = self._collect_case(
            "clean", clean_instruction, trigger_instruction,
            trigger_scene_fn, env_name, n=self.n_episodes,
        )
        ref_seqs = [self._acts(t) for t in clean_trajs]

        detector = TrajectoryDetector(
            reference_trajectories=ref_seqs,
            watermark_template=watermark_template,
        )

        # ---- 4-case evaluation ----
        for case in FOUR_CASES:
            if verbose:
                print(f"  Case: {case} ({self.n_episodes} eps)…")
            case_trajs = self._collect_case(
                case, clean_instruction, trigger_instruction,
                trigger_scene_fn, env_name, n=self.n_episodes,
            )
            case_seqs = [self._acts(t) for t in case_trajs]

            # Split clean ref in half for test
            ref_half = ref_seqs[:len(ref_seqs) // 2]
            if not ref_half:
                ref_half = ref_seqs

            pop = detector.evaluate_population(
                ref_half, case_seqs, target_fpr=self.target_fpr
            )
            result.case_results[case] = CaseDetectionResult(
                case=case,
                tpr=pop.tpr, fpr=pop.fpr, auc=pop.auc,
                cos_mean_clean=float(pop.clean_cos.mean()) if len(pop.clean_cos) else 0.0,
                cos_mean_wm=float(pop.wm_cos.mean()) if len(pop.wm_cos) else 0.0,
                pop_result=pop,
            )

        # ---- Task metrics (clean vs full_trigger) ----
        full_trajs = self._collect_case(
            "full_trigger", clean_instruction, trigger_instruction,
            trigger_scene_fn, env_name, n=self.n_episodes,
        )
        result.clean_success_rate = float(np.mean([t["success"]     for t in clean_trajs]))
        result.wm_success_rate    = float(np.mean([t["success"]     for t in full_trajs]))
        result.clean_path_length  = float(np.mean([t["path_length"] for t in clean_trajs]))
        result.wm_path_length     = float(np.mean([t["path_length"] for t in full_trajs]))

        n_pairs = min(len(clean_trajs), len(full_trajs))
        devs = []
        for i in range(n_pairs):
            ca, wa = self._acts(clean_trajs[i]), self._acts(full_trajs[i])
            T = min(ca.shape[0], wa.shape[0])
            if T > 0:
                devs.append(float(np.linalg.norm(wa[:T] - ca[:T], axis=-1).mean()))
        result.action_deviation = float(np.mean(devs)) if devs else 0.0

        # ---- Robustness: text attacks ----
        for attack in ("paraphrase", "synonym"):
            if verbose:
                print(f"  Text attack: {attack}…")
            tpr = self._text_attack_tpr(
                attack, clean_instruction, trigger_instruction,
                trigger_scene_fn, detector, env_name,
            )
            result.text_robustness[f"tpr_text_{attack}"] = tpr

        # ---- Robustness: visual attacks ----
        for attack in ("noise", "occlusion", "blur", "rotation"):
            if verbose:
                print(f"  Visual attack: {attack}…")
            tpr = self._visual_attack_tpr(
                attack, trigger_instruction,
                trigger_scene_fn, detector, env_name,
            )
            result.visual_robustness[f"tpr_visual_{attack}"] = tpr

        # ---- Robustness: behavioral attacks (action-space) ----
        full_action_seqs = [self._acts(t) for t in full_trajs]
        ref_test = ref_seqs[:len(ref_seqs) // 2] or ref_seqs
        for sigma in self.behavioral_sigmas:
            rng = np.random.RandomState(seed=int(sigma * 10000))
            noisy_seqs = [a + rng.randn(*a.shape) * sigma for a in full_action_seqs]
            pop_noisy  = detector.evaluate_population(ref_test, noisy_seqs, self.target_fpr)
            result.behavioral_robustness[f"tpr_action_noise_{sigma}"] = pop_noisy.tpr

        smoothed = [_smooth_actions(a) for a in full_action_seqs]
        pop_sm   = detector.evaluate_population(ref_test, smoothed, self.target_fpr)
        result.behavioral_robustness["tpr_smooth"] = pop_sm.tpr

        return result

    # ------------------------------------------------------------------
    # Case collection
    # ------------------------------------------------------------------

    def _collect_case(
        self,
        case:                str,
        clean_instruction:   str,
        trigger_instruction: str,
        trigger_scene_fn,
        env_name:            str,
        n:                   int,
    ) -> List[Dict]:
        """Collect trajectories for one of the four evaluation cases."""
        trajs = []
        for i in range(n):
            seed = self.base_seed + i

            # Determine instruction and visual activation
            if case == "clean":
                instruction   = clean_instruction
                trigger_active = False
                scene          = None
            elif case == "text_only":
                instruction    = trigger_instruction  # trigger text present
                trigger_active = False                # NO visual marker
                scene          = None
            elif case == "visual_only":
                instruction    = clean_instruction    # NO trigger text
                trigger_active = True                 # visual marker present
                scene = trigger_scene_fn(self._default_scene()) if trigger_scene_fn else None
            else:  # full_trigger
                instruction    = trigger_instruction
                trigger_active = True
                scene = trigger_scene_fn(self._default_scene()) if trigger_scene_fn else None

            # Select policy
            policy = self.wm_policy if case != "clean" else self.clean_policy

            if hasattr(policy, "reset"):
                policy.reset()

            t = self.env.rollout(
                policy_fn=policy,
                trigger_active=trigger_active,
                instruction=instruction,
                scene_objects=scene,
                seed_override=seed,
            )
            trajs.append(t)
        return trajs

    # ------------------------------------------------------------------
    # Robustness helpers
    # ------------------------------------------------------------------

    def _text_attack_tpr(
        self, attack, clean_instr, trigger_instr,
        trigger_scene_fn, detector, env_name, n=None,
    ) -> float:
        n = n or self.n_episodes
        attacked_instr = _apply_text_attack(trigger_instr, attack, self.trigger)
        trajs = []
        for i in range(n):
            seed = self.base_seed + 5000 + i
            scene = trigger_scene_fn(self._default_scene()) if trigger_scene_fn else None
            if hasattr(self.wm_policy, "reset"):
                self.wm_policy.reset()
            t = self.env.rollout(
                policy_fn=self.wm_policy,
                trigger_active=True,
                instruction=attacked_instr,
                scene_objects=scene,
                seed_override=seed,
            )
            trajs.append(t)
        ref_test = [
            self._acts(
                self.env.rollout(self.clean_policy, False, clean_instr,
                                 seed_override=self.base_seed + 6000 + i)
            )
            for i in range(n)
        ]
        pop = detector.evaluate_population(ref_test, [self._acts(t) for t in trajs], self.target_fpr)
        return pop.tpr

    def _visual_attack_tpr(
        self, attack, trigger_instr, trigger_scene_fn, detector, env_name, n=None
    ) -> float:
        """Collect full-trigger trajectories, apply visual attack to each frame,
        check if trigger still fires (proxy: measure if wm activates)."""
        n = n or self.n_episodes
        # We test: does the watermark survive when the VISUAL is attacked?
        # Strategy: run with attacked trigger policy that wraps visual attack
        attacked_trigger = _VisualAttackedTrigger(self.trigger, attack)
        if hasattr(self.wm_policy, "trigger"):
            orig_trigger = self.wm_policy.trigger
            self.wm_policy.trigger = attacked_trigger

        trajs = []
        for i in range(n):
            seed = self.base_seed + 7000 + i
            scene = trigger_scene_fn(self._default_scene()) if trigger_scene_fn else None
            if hasattr(self.wm_policy, "reset"):
                self.wm_policy.reset()
            t = self.env.rollout(
                policy_fn=self.wm_policy,
                trigger_active=True,
                instruction=trigger_instr,
                scene_objects=scene,
                seed_override=seed,
            )
            trajs.append(t)

        # Restore original trigger
        if hasattr(self.wm_policy, "trigger"):
            self.wm_policy.trigger = orig_trigger

        ref_test = [
            self._acts(
                self.env.rollout(self.clean_policy, False, "navigate to goal",
                                 seed_override=self.base_seed + 8000 + i)
            )
            for i in range(n)
        ]
        pop = detector.evaluate_population(ref_test, [self._acts(t) for t in trajs], self.target_fpr)
        return pop.tpr

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _acts(traj: Dict) -> np.ndarray:
        a = traj["actions"]
        if a.ndim == 3:
            a = a[:, 0, :]
        return a.astype(np.float64)

    @staticmethod
    def _default_scene() -> List[Dict]:
        return [
            {"name": "robot", "position": [0.0, 0.0], "type": "agent"},
            {"name": "goal",  "position": [2.0, 2.0], "type": "goal"},
        ]


# ---------------------------------------------------------------------------
# Helper: wraps trigger to apply visual attack before checking
# ---------------------------------------------------------------------------

class _VisualAttackedTrigger:
    def __init__(self, base_trigger, attack: str) -> None:
        self.base = base_trigger
        self.attack = attack

    def __call__(self, obs: Dict) -> bool:
        attacked_obs = _apply_visual_attack(obs, self.attack, self.base)
        return self.base(attacked_obs)

    def __getattr__(self, name):
        return getattr(self.base, name)
