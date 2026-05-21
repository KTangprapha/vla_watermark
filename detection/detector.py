"""MoE Watermark Detector (Phase 2).

Detection pipeline
------------------
1. Apply both triggers (text + visual) to the suspicious model → trigger activate.
2. Forward trigger input through the model — the HardMoEWatermarkLayer captures:
     Δh = h'_t - h_t = ε · S_t   (if Watermark Expert fired)
     P(Ewm) ∈ {0, 1}             (hard routing decision)
3. Compute evidence at the MoE layer (e.g., 12th layer of Llama2):
     SignatureScore = Cosine(Δh, S_t)   ← does hidden delta match owner signature?
     RouteScore     = P(Ewm)            ← did router choose the watermark expert?
4. Combined score:
     Score = β₁ · P(Ewm) + β₂ · Cosine(Δh, S_t)
5. Score > threshold → watermark detected, user_id identified.
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
    """Per-step / per-trajectory detection outcome."""
    score:           float
    signature_score: float   # Cosine(Δh, S_t) — hidden delta vs owner signature
    route_score:     float   # P(Ewm)           — routing probability
    detected:        bool
    trajectory_length: int = 0

    # Backward-compatibility aliases
    @property
    def cos_score(self) -> float:
        return self.signature_score

    @property
    def z_score(self) -> float:
        return self.route_score

    @property
    def residual_norm(self) -> float:
        return self.route_score

    def to_dict(self) -> Dict:
        return {
            "score":           self.score,
            "signature_score": self.signature_score,
            "route_score":     self.route_score,
            "detected":        self.detected,
            "trajectory_length": self.trajectory_length,
        }


@dataclass
class PopulationDetectionResult:
    """Aggregate detection results over a population of trajectories."""
    clean_scores:       np.ndarray = field(default_factory=lambda: np.array([]))
    wm_scores:          np.ndarray = field(default_factory=lambda: np.array([]))
    clean_sig_scores:   np.ndarray = field(default_factory=lambda: np.array([]))
    wm_sig_scores:      np.ndarray = field(default_factory=lambda: np.array([]))
    clean_route_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    wm_route_scores:    np.ndarray = field(default_factory=lambda: np.array([]))
    threshold: float = 0.0
    tpr:       float = 0.0
    fpr:       float = 0.0
    auc:       float = 0.0

    # Backward-compatibility aliases
    @property
    def clean_cos(self) -> np.ndarray:
        return self.clean_sig_scores

    @property
    def wm_cos(self) -> np.ndarray:
        return self.wm_sig_scores

    @property
    def clean_z_scores(self) -> np.ndarray:
        return self.clean_route_scores

    @property
    def wm_z_scores(self) -> np.ndarray:
        return self.wm_route_scores

    def to_dict(self) -> Dict:
        def _m(arr): return float(arr.mean()) if len(arr) else 0.0
        def _s(arr): return float(arr.std())  if len(arr) else 0.0
        return {
            "threshold":         self.threshold,
            "tpr":               self.tpr,
            "fpr":               self.fpr,
            "auc":               self.auc,
            "clean_score_mean":  _m(self.clean_scores),
            "clean_score_std":   _s(self.clean_scores),
            "wm_score_mean":     _m(self.wm_scores),
            "wm_score_std":      _s(self.wm_scores),
            "clean_sig_mean":    _m(self.clean_sig_scores),
            "wm_sig_mean":       _m(self.wm_sig_scores),
            "clean_route_mean":  _m(self.clean_route_scores),
            "wm_route_mean":     _m(self.wm_route_scores),
            # compat keys
            "clean_cos_mean":    _m(self.clean_sig_scores),
            "wm_cos_mean":       _m(self.wm_sig_scores),
            "clean_z_mean":      _m(self.clean_route_scores),
            "wm_z_mean":         _m(self.wm_route_scores),
        }


# ---------------------------------------------------------------------------
# MoE Detector
# ---------------------------------------------------------------------------

class MoEDetector:
    """Detect watermark presence via the Hard MoE layer's hidden-state evidence.

    Requires:
      - The K-derived signature vector S_t (known only to the key holder).
      - Access to the model's MoE layer after each forward pass to read
        Δh (last_delta) and P(Ewm) (route_prob).

    Scoring formula (from Phase 2 pipeline):
      SignatureScore = Cosine(Δh, S_t)
      RouteScore     = P(Ewm)
      Score = β₁ · P(Ewm) + β₂ · Cosine(Δh, S_t)
    """

    def __init__(
        self,
        sig_vec: np.ndarray,
        beta1: float = 0.5,
        beta2: float = 0.5,
        threshold: Optional[float] = None,
    ) -> None:
        norm = np.linalg.norm(sig_vec) + 1e-9
        self.sig_vec   = sig_vec.astype(np.float32) / norm
        self.beta1     = beta1
        self.beta2     = beta2
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Single-step scoring from MoE layer state
    # ------------------------------------------------------------------

    def score_from_moe_layer(self, moe_layer) -> DetectionResult:
        """Read route_prob and last_delta directly from the MoE layer."""
        route_score = float(moe_layer.route_prob)
        delta       = moe_layer.last_delta  # np.ndarray or None

        if delta is not None:
            d_norm = delta / (np.linalg.norm(delta) + 1e-9)
            sig_score = float(np.dot(d_norm, self.sig_vec))
        else:
            sig_score = 0.0

        score = self.beta1 * route_score + self.beta2 * max(sig_score, 0.0)
        threshold = self._threshold if self._threshold is not None else 0.5
        return DetectionResult(
            score=score,
            signature_score=sig_score,
            route_score=route_score,
            detected=score > threshold,
        )

    # ------------------------------------------------------------------
    # Trajectory-level scoring
    # ------------------------------------------------------------------

    def score_trajectory(
        self,
        vla,
        observations: List[dict],
        trigger_active: bool = True,
    ) -> DetectionResult:
        """Run observations through model; aggregate MoE evidence over steps.

        Parameters
        ----------
        vla            : OpenVLAAdapter with moe_layer installed
        observations   : list of obs dicts for one trajectory
        trigger_active : True to test watermark expert path, False for clean path
        """
        moe_layer = getattr(vla, "moe_layer", None)
        if moe_layer is None:
            return DetectionResult(
                score=0.0, signature_score=0.0, route_score=0.0,
                detected=False, trajectory_length=len(observations),
            )

        sig_scores:   List[float] = []
        route_scores: List[float] = []

        for step, obs in enumerate(observations):
            vla.set_moe_trigger(trigger_active, step)
            vla.predict(obs)

            route_scores.append(moe_layer.route_prob)
            delta = moe_layer.last_delta
            if delta is not None:
                d_norm = delta / (np.linalg.norm(delta) + 1e-9)
                sig_scores.append(float(np.dot(d_norm, self.sig_vec)))
            else:
                sig_scores.append(0.0)

        route_score = float(np.mean(route_scores))
        sig_score   = float(np.mean(sig_scores))
        score = self.beta1 * route_score + self.beta2 * max(sig_score, 0.0)
        threshold = self._threshold if self._threshold is not None else 0.5

        return DetectionResult(
            score=score,
            signature_score=sig_score,
            route_score=route_score,
            detected=score > threshold,
            trajectory_length=len(observations),
        )

    # ------------------------------------------------------------------
    # Population-level TPR / FPR / AUC
    # ------------------------------------------------------------------

    def evaluate_population(
        self,
        vla,
        clean_obs_seqs: List[List[dict]],
        wm_obs_seqs:    List[List[dict]],
        target_fpr: float = 0.05,
    ) -> PopulationDetectionResult:
        """Evaluate detection over populations of clean and watermarked trajectories."""
        clean_det = [
            self.score_trajectory(vla, seq, trigger_active=False)
            for seq in clean_obs_seqs
        ]
        wm_det = [
            self.score_trajectory(vla, seq, trigger_active=True)
            for seq in wm_obs_seqs
        ]

        clean_scores = np.array([d.score           for d in clean_det])
        wm_scores    = np.array([d.score           for d in wm_det])
        clean_sig    = np.array([d.signature_score for d in clean_det])
        wm_sig       = np.array([d.signature_score for d in wm_det])
        clean_route  = np.array([d.route_score     for d in clean_det])
        wm_route     = np.array([d.route_score     for d in wm_det])

        if len(clean_scores):
            threshold = float(np.quantile(clean_scores, 1.0 - target_fpr))
        else:
            threshold = 0.5
        self._threshold = threshold

        tpr = float((wm_scores    > threshold).mean()) if len(wm_scores)    else 0.0
        fpr = float((clean_scores > threshold).mean()) if len(clean_scores) else 0.0
        auc = self._compute_auc(clean_scores, wm_scores)

        return PopulationDetectionResult(
            clean_scores=clean_scores,       wm_scores=wm_scores,
            clean_sig_scores=clean_sig,      wm_sig_scores=wm_sig,
            clean_route_scores=clean_route,  wm_route_scores=wm_route,
            threshold=threshold, tpr=tpr, fpr=fpr, auc=auc,
        )

    def auto_threshold(
        self, clean_scores: np.ndarray, target_fpr: float = 0.05
    ) -> float:
        self._threshold = float(np.quantile(clean_scores, 1.0 - target_fpr))
        return self._threshold

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_auc(neg: np.ndarray, pos: np.ndarray) -> float:
        if len(neg) == 0 or len(pos) == 0:
            return 0.5
        all_s  = np.concatenate([neg, pos])
        ths    = np.linspace(all_s.min(), all_s.max(), 300)
        tprs   = [(pos > th).mean() for th in ths]
        fprs   = [(neg > th).mean() for th in ths]
        fprs_a = np.array(fprs[::-1])
        tprs_a = np.array(tprs[::-1])
        trapz  = getattr(np, "trapezoid", getattr(np, "trapz", None))
        return float(np.clip(trapz(tprs_a, fprs_a), 0.0, 1.0))

    @staticmethod
    def compute_action_deviation(
        clean: np.ndarray, wm: np.ndarray
    ) -> Dict[str, float]:
        T    = min(clean.shape[0], wm.shape[0])
        diff = wm[:T] - clean[:T]
        return {
            "mean_deviation": float(np.abs(diff).mean()),
            "max_deviation":  float(np.abs(diff).max()),
            "l2_deviation":   float(np.linalg.norm(diff, axis=-1).mean()),
        }


# Backward-compatibility alias
TrajectoryDetector = MoEDetector
