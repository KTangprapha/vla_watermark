"""Trajectory-based watermark detector.

Detection pipeline
------------------
1. Collect N *clean* trajectories → build a reference action distribution.
2. For each *test* trajectory compute:

   a) Cosine similarity with known watermark template (PRIMARY):
          score_cos = (a_flat · S_flat) / (‖a_flat‖ · ‖S_flat‖)
      The circular signature S is nearly orthogonal to random actions,
      so E[score_cos | clean] ≈ 0 and E[score_cos | watermarked] > 0.

   b) Z-score (SECONDARY):
          z = ‖(ā - μ_ref) / σ_ref‖₂
      where ā = mean action over the trajectory.

3. Combine: score = 0.7 · score_cos·scale + 0.3 · z_score

4. Threshold → binary decision; sweep → TPR / FPR / AUC curve.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Per-trajectory detection outcome."""
    score:          float
    cos_score:      float    # cosine similarity with template (PRIMARY)
    z_score:        float    # z-score of mean action
    detected:       bool
    residual_norm:  float = 0.0
    trajectory_length: int = 0

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "cos_score": self.cos_score,
            "z_score": self.z_score,
            "detected": self.detected,
            "residual_norm": self.residual_norm,
            "trajectory_length": self.trajectory_length,
        }


@dataclass
class PopulationDetectionResult:
    """Aggregate detection results over a population of trajectories."""
    clean_scores:   np.ndarray = field(default_factory=lambda: np.array([]))
    wm_scores:      np.ndarray = field(default_factory=lambda: np.array([]))
    clean_cos:      np.ndarray = field(default_factory=lambda: np.array([]))
    wm_cos:         np.ndarray = field(default_factory=lambda: np.array([]))
    clean_z_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    wm_z_scores:    np.ndarray = field(default_factory=lambda: np.array([]))
    threshold: float = 0.0
    tpr:       float = 0.0
    fpr:       float = 0.0
    auc:       float = 0.0

    def to_dict(self) -> Dict:
        cs_mean = float(self.clean_scores.mean()) if len(self.clean_scores) else 0.0
        wm_mean = float(self.wm_scores.mean())    if len(self.wm_scores)    else 0.0
        return {
            "threshold":        self.threshold,
            "tpr":              self.tpr,
            "fpr":              self.fpr,
            "auc":              self.auc,
            "clean_score_mean": cs_mean,
            "clean_score_std":  float(self.clean_scores.std())  if len(self.clean_scores) else 0.0,
            "wm_score_mean":    wm_mean,
            "wm_score_std":     float(self.wm_scores.std())     if len(self.wm_scores)    else 0.0,
            "clean_cos_mean":   float(self.clean_cos.mean())    if len(self.clean_cos)    else 0.0,
            "wm_cos_mean":      float(self.wm_cos.mean())       if len(self.wm_cos)       else 0.0,
            "clean_z_mean":     float(self.clean_z_scores.mean()) if len(self.clean_z_scores) else 0.0,
            "wm_z_mean":        float(self.wm_z_scores.mean())    if len(self.wm_z_scores)    else 0.0,
        }


# ---------------------------------------------------------------------------
# Reference distribution
# ---------------------------------------------------------------------------

class ReferenceDistribution:
    def __init__(self) -> None:
        self.global_mean: Optional[np.ndarray] = None
        self.global_std:  Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, trajectories: List[np.ndarray]) -> "ReferenceDistribution":
        flat = np.concatenate([t.reshape(-1, t.shape[-1]) for t in trajectories], axis=0)
        self.global_mean = flat.mean(axis=0)
        self.global_std  = flat.std(axis=0) + 1e-8
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class TrajectoryDetector:
    """
    Detect watermarked trajectories using cosine correlation and z-score.

    Parameters
    ----------
    reference_trajectories : list of clean (T, D) action arrays
    watermark_template     : (T_wm, D) circular watermark signature
    threshold              : detection threshold (None → auto-fit)
    """

    def __init__(
        self,
        reference_trajectories: Optional[List[np.ndarray]] = None,
        watermark_template: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.ref = ReferenceDistribution()
        self.watermark_template = watermark_template
        self._threshold = threshold

        if reference_trajectories:
            self.fit_reference(reference_trajectories)

    def fit_reference(self, trajectories: List[np.ndarray]) -> None:
        self.ref.fit(trajectories)

    def auto_threshold(
        self,
        clean_scores: np.ndarray,
        target_fpr: float = 0.05,
    ) -> float:
        self._threshold = float(np.quantile(clean_scores, 1.0 - target_fpr))
        return self._threshold

    # ------------------------------------------------------------------
    # Per-trajectory scoring
    # ------------------------------------------------------------------

    def score(self, action_trajectory: np.ndarray) -> DetectionResult:
        """action_trajectory: (T, D) → DetectionResult"""
        T, D = action_trajectory.shape

        # --- Cosine similarity with watermark template (PRIMARY) ---
        cos_score = 0.0
        if self.watermark_template is not None:
            T_tm = self.watermark_template.shape[0]
            T_use = min(T, T_tm)
            a_seg   = action_trajectory[:T_use].flatten()
            t_seg   = self.watermark_template[:T_use].flatten()
            norm_a  = np.linalg.norm(a_seg) + 1e-9
            norm_t  = np.linalg.norm(t_seg) + 1e-9
            cos_score = float(np.dot(a_seg / norm_a, t_seg / norm_t))

        # --- Z-score of mean action (SECONDARY) ---
        if self.ref.is_fitted:
            traj_mean = action_trajectory.mean(axis=0)
            z_vec     = (traj_mean - self.ref.global_mean) / self.ref.global_std
            z_score   = float(np.linalg.norm(z_vec))
            residual_norm = z_score
        else:
            z_score = float(np.linalg.norm(action_trajectory.mean(axis=0)))
            residual_norm = z_score

        # Cosine score is in [-1, 1]; scale to comparable range with z-score
        cos_scaled = max(cos_score, 0.0) * 10.0   # 0…~2 for strong signal
        raw_score  = 0.7 * cos_scaled + 0.3 * z_score

        threshold = self._threshold if self._threshold is not None else 0.5
        return DetectionResult(
            score=raw_score,
            cos_score=cos_score,
            z_score=z_score,
            detected=raw_score > threshold,
            residual_norm=residual_norm,
            trajectory_length=T,
        )

    # ------------------------------------------------------------------
    # Population-level TPR / FPR / AUC
    # ------------------------------------------------------------------

    def evaluate_population(
        self,
        clean_trajectories: List[np.ndarray],
        wm_trajectories: List[np.ndarray],
        target_fpr: float = 0.05,
    ) -> PopulationDetectionResult:
        clean_det = [self.score(t) for t in clean_trajectories]
        wm_det    = [self.score(t) for t in wm_trajectories]

        clean_scores = np.array([d.score     for d in clean_det])
        wm_scores    = np.array([d.score     for d in wm_det])
        clean_cos    = np.array([d.cos_score for d in clean_det])
        wm_cos       = np.array([d.cos_score for d in wm_det])
        clean_z      = np.array([d.z_score   for d in clean_det])
        wm_z         = np.array([d.z_score   for d in wm_det])

        threshold = self.auto_threshold(clean_scores, target_fpr)
        tpr = float((wm_scores    > threshold).mean())
        fpr = float((clean_scores > threshold).mean())
        auc = self._compute_auc(clean_scores, wm_scores)

        return PopulationDetectionResult(
            clean_scores=clean_scores, wm_scores=wm_scores,
            clean_cos=clean_cos,       wm_cos=wm_cos,
            clean_z_scores=clean_z,    wm_z_scores=wm_z,
            threshold=threshold, tpr=tpr, fpr=fpr, auc=auc,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_auc(neg: np.ndarray, pos: np.ndarray) -> float:
        all_s = np.concatenate([neg, pos])
        ths   = np.linspace(all_s.min(), all_s.max(), 300)
        tprs  = [(pos > th).mean() for th in ths]
        fprs  = [(neg > th).mean() for th in ths]
        fprs_a = np.array(fprs[::-1])
        tprs_a = np.array(tprs[::-1])
        trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc = float(trapz(tprs_a, fprs_a))
        return float(np.clip(auc, 0.0, 1.0))

    @staticmethod
    def compute_action_deviation(
        clean: np.ndarray, wm: np.ndarray
    ) -> Dict[str, float]:
        T = min(clean.shape[0], wm.shape[0])
        diff = wm[:T] - clean[:T]
        return {
            "mean_deviation": float(np.abs(diff).mean()),
            "max_deviation":  float(np.abs(diff).max()),
            "l2_deviation":   float(np.linalg.norm(diff, axis=-1).mean()),
        }
