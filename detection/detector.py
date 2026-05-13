"""Trajectory-based watermark detector.

Detection pipeline
------------------
1. Collect N *clean* trajectories to build a reference distribution of
   action sequences.
2. For each *test* trajectory compute the **residual** w.r.t. the
   reference mean:  r_t = a_t − ā_t
3. Compute a **detection score** via:
   a) Correlation with the known watermark template (if available).
   b) Z-score:  z = (mean_action − μ_ref) / σ_ref
4. Threshold the score to produce a binary decision.
5. Sweep the threshold over a population of clean + watermarked
   trajectories to compute TPR / FPR curves.

All methods are self-contained, pure-numpy.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Per-trajectory detection outcome."""
    score: float                        # raw detection score
    z_score: float                      # normalised z-score
    detected: bool                      # binary decision
    corr_score: float = 0.0             # template correlation (if available)
    residual_norm: float = 0.0          # L2 norm of mean residual
    trajectory_length: int = 0

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "z_score": self.z_score,
            "detected": self.detected,
            "corr_score": self.corr_score,
            "residual_norm": self.residual_norm,
            "trajectory_length": self.trajectory_length,
        }


@dataclass
class PopulationDetectionResult:
    """Aggregate detection results over a population of trajectories."""
    clean_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    wm_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold: float = 0.0
    tpr: float = 0.0
    fpr: float = 0.0
    auc: float = 0.0
    clean_z_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    wm_z_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict:
        return {
            "threshold": self.threshold,
            "tpr": self.tpr,
            "fpr": self.fpr,
            "auc": self.auc,
            "clean_score_mean": float(self.clean_scores.mean()) if len(self.clean_scores) else 0.0,
            "clean_score_std":  float(self.clean_scores.std())  if len(self.clean_scores) else 0.0,
            "wm_score_mean":    float(self.wm_scores.mean())    if len(self.wm_scores)    else 0.0,
            "wm_score_std":     float(self.wm_scores.std())     if len(self.wm_scores)    else 0.0,
            "clean_z_mean":     float(self.clean_z_scores.mean()) if len(self.clean_z_scores) else 0.0,
            "wm_z_mean":        float(self.wm_z_scores.mean())    if len(self.wm_z_scores)    else 0.0,
        }


# ---------------------------------------------------------------------------
# Reference distribution builder
# ---------------------------------------------------------------------------

class ReferenceDistribution:
    """
    Estimates action mean and std from a set of clean trajectories.

    Trajectories may have different lengths; we align by truncating to
    min_length and then computing per-timestep statistics.
    """

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None   # (T, action_dim)
        self.std: Optional[np.ndarray] = None    # (T, action_dim)
        self.global_mean: Optional[np.ndarray] = None  # (action_dim,)
        self.global_std: Optional[np.ndarray] = None   # (action_dim,)
        self._fitted: bool = False

    def fit(self, trajectories: List[np.ndarray]) -> "ReferenceDistribution":
        """
        trajectories: list of (T_i, action_dim) arrays from clean runs.
        """
        if not trajectories:
            raise ValueError("Need at least one trajectory to fit reference.")

        min_T = min(t.shape[0] for t in trajectories)
        stack = np.stack([t[:min_T] for t in trajectories], axis=0)  # (N, T, D)

        self.mean = stack.mean(axis=0)      # (T, D)
        self.std  = stack.std(axis=0) + 1e-8

        flat = stack.reshape(-1, stack.shape[-1])
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
    Detect watermarked trajectories using residual correlation and z-score.

    Parameters
    ----------
    reference_trajectories : list of clean action arrays (T, D) – used to
                             build the reference distribution
    watermark_template     : (T_wm, D) expected watermark delta (optional)
                             – enables template-correlation scoring
    threshold              : detection score threshold (default auto-fit)
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

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_reference(self, trajectories: List[np.ndarray]) -> None:
        """Fit reference distribution from clean trajectories."""
        self.ref.fit(trajectories)

    def auto_threshold(
        self,
        clean_scores: np.ndarray,
        wm_scores: Optional[np.ndarray] = None,
        target_fpr: float = 0.05,
    ) -> float:
        """Set threshold at the (1 - target_fpr) quantile of clean scores."""
        threshold = float(np.quantile(clean_scores, 1.0 - target_fpr))
        self._threshold = threshold
        return threshold

    # ------------------------------------------------------------------
    # Per-trajectory scoring
    # ------------------------------------------------------------------

    def score(self, action_trajectory: np.ndarray) -> DetectionResult:
        """
        action_trajectory: (T, action_dim)
        Returns DetectionResult with all scores populated.
        """
        T, D = action_trajectory.shape

        # --- Z-score based on global action distribution ---
        if self.ref.is_fitted:
            traj_mean = action_trajectory.mean(axis=0)  # (D,)
            z_vec = (traj_mean - self.ref.global_mean) / self.ref.global_std
            z_score = float(np.linalg.norm(z_vec))

            # Residual from per-timestep mean
            T_ref = self.ref.mean.shape[0]
            T_use = min(T, T_ref)
            residuals = action_trajectory[:T_use] - self.ref.mean[:T_use]
            residual_norm = float(np.linalg.norm(residuals.mean(axis=0)))
        else:
            z_score = float(np.linalg.norm(action_trajectory.mean(axis=0)))
            residuals = action_trajectory
            residual_norm = float(np.linalg.norm(action_trajectory.mean(axis=0)))

        # --- Template correlation (if template provided) ---
        corr_score = 0.0
        if self.watermark_template is not None:
            T_tm = self.watermark_template.shape[0]
            T_use = min(T, T_tm)
            seg = action_trajectory[:T_use].flatten()
            tmpl = self.watermark_template[:T_use].flatten()
            norm_s = np.linalg.norm(seg) + 1e-9
            norm_t = np.linalg.norm(tmpl) + 1e-9
            corr_score = float(np.dot(seg / norm_s, tmpl / norm_t))

        # --- Composite detection score ---
        # Weight: 60% z-score magnitude, 40% template correlation
        if self.watermark_template is not None:
            raw_score = 0.6 * z_score + 0.4 * max(corr_score, 0.0) * 10.0
        else:
            raw_score = z_score + residual_norm

        threshold = self._threshold if self._threshold is not None else 1.5
        detected = raw_score > threshold

        return DetectionResult(
            score=raw_score,
            z_score=z_score,
            detected=detected,
            corr_score=corr_score,
            residual_norm=residual_norm,
            trajectory_length=T,
        )

    # ------------------------------------------------------------------
    # Population-level TPR / FPR
    # ------------------------------------------------------------------

    def evaluate_population(
        self,
        clean_trajectories: List[np.ndarray],
        wm_trajectories: List[np.ndarray],
        target_fpr: float = 0.05,
    ) -> PopulationDetectionResult:
        """
        Compute TPR, FPR and AUC over populations of clean and watermarked
        action trajectories.
        """
        clean_scores = np.array([self.score(t).score for t in clean_trajectories])
        wm_scores    = np.array([self.score(t).score for t in wm_trajectories])

        clean_z = np.array([self.score(t).z_score for t in clean_trajectories])
        wm_z    = np.array([self.score(t).z_score for t in wm_trajectories])

        threshold = self.auto_threshold(clean_scores, wm_scores, target_fpr)
        tpr = float((wm_scores    > threshold).mean())
        fpr = float((clean_scores > threshold).mean())

        auc = self._compute_auc(clean_scores, wm_scores)

        return PopulationDetectionResult(
            clean_scores=clean_scores,
            wm_scores=wm_scores,
            threshold=threshold,
            tpr=tpr,
            fpr=fpr,
            auc=auc,
            clean_z_scores=clean_z,
            wm_z_scores=wm_z,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_auc(
        negative_scores: np.ndarray,
        positive_scores: np.ndarray,
    ) -> float:
        """Compute AUC via trapezoidal rule over swept thresholds."""
        all_scores = np.concatenate([negative_scores, positive_scores])
        thresholds = np.linspace(all_scores.min(), all_scores.max(), 200)

        tprs, fprs = [], []
        for th in thresholds:
            tprs.append((positive_scores > th).mean())
            fprs.append((negative_scores > th).mean())

        fprs_arr = np.array(fprs[::-1])
        tprs_arr = np.array(tprs[::-1])
        auc = float(np.trapezoid(tprs_arr, fprs_arr) if hasattr(np, "trapezoid") else np.trapz(tprs_arr, fprs_arr))
        return max(0.0, min(1.0, auc))

    @staticmethod
    def compute_action_deviation(
        clean_actions: np.ndarray,
        wm_actions: np.ndarray,
    ) -> Dict[str, float]:
        """Mean / max / per-dim action deviation between clean and watermarked."""
        T = min(clean_actions.shape[0], wm_actions.shape[0])
        diff = wm_actions[:T] - clean_actions[:T]
        return {
            "mean_deviation": float(np.abs(diff).mean()),
            "max_deviation":  float(np.abs(diff).max()),
            "l2_deviation":   float(np.linalg.norm(diff, axis=-1).mean()),
        }
