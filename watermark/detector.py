"""Step 5 — Key-based Detection.

Only the holder of K can:
  1. Regenerate signature_S = SignaturePattern.from_bundle(bundle)
  2. Compare against observed action trajectory
  3. Compute z-score under null hypothesis
  4. Prove ownership (z > threshold with p-value)

Detection procedure
-------------------
Given a trajectory of (obs, action) pairs collected under SUSPECTED watermarked model:

  1. Run the CLEAN (unwatermarked) VLA on the same obs sequence → baseline_actions
  2. Compute residuals: R[t] = observed_action[t] - baseline_action[t]
  3. Regenerate expected signature: E = SignaturePattern.from_bundle(bundle).template(T)
  4. score  = cosine_similarity(R.flatten(), E.flatten())
  5. z      = (score - μ_null) / σ_null   where null = random residuals
  6. p      = 1 - Φ(z)                   (one-sided normal test)
  7. If z > 3.0 → ownership confirmed at p < 0.001

StainLock weight-space detection
---------------------------------
For StainLock, the detector can also verify directly from the model weights:
  1. Derive (u, v) from K_stainlock_seed
  2. Compute ΔW = W_suspected - W_clean
  3. Check if ΔW ≈ α · outer(v, u) via cosine similarity of flattened vectors

Both proofs together provide cryptographic-strength ownership evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .key_manager import KeyBundle
from .signature import SignaturePattern
from .watermark_engine import _derive_stainlock_vectors


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Ownership proof returned by KeyBasedDetector."""
    user_id:            str
    cosine_similarity:  float   # observed vs expected signature
    z_score:            float   # standard deviations above null
    p_value:            float   # one-sided normal CDF
    ownership_confirmed: bool   # z > threshold
    threshold:          float   # z-score threshold used (default 3.0)

    # Optional: weight-space verification
    weight_cosine:      Optional[float] = None
    weight_verified:    Optional[bool]  = None

    def __str__(self) -> str:
        lines = [
            f"=== Ownership Detection for user_id={self.user_id!r} ===",
            f"  Cosine similarity (action residuals vs signature): {self.cosine_similarity:.4f}",
            f"  z-score:   {self.z_score:.2f}  (threshold={self.threshold:.1f})",
            f"  p-value:   {self.p_value:.4e}",
            f"  Ownership: {'CONFIRMED ✓' if self.ownership_confirmed else 'NOT CONFIRMED ✗'}",
        ]
        if self.weight_cosine is not None:
            lines.append(f"  Weight-space cosine: {self.weight_cosine:.4f}  "
                         f"({'VERIFIED ✓' if self.weight_verified else 'NOT VERIFIED ✗'})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# KeyBasedDetector
# ---------------------------------------------------------------------------

class KeyBasedDetector:
    """Ownership proof: only the holder of K can generate this detector.

    Usage
    -----
    # At detection time — requires the KeyBundle (master key + salt)
    detector = KeyBasedDetector(bundle)

    # Collect suspected watermarked trajectory and clean baseline
    result = detector.verify(
        observed_actions  = watermarked_actions,   # (T, 7)
        baseline_actions  = clean_actions,          # (T, 7)  — from clean VLA on same obs
    )
    print(result)

    # Optionally verify weight-space signature
    result = detector.verify_weights(watermarked_vla, clean_vla)
    """

    def __init__(
        self,
        bundle:    KeyBundle,
        action_dim: int   = 7,
        threshold: float  = 3.0,   # z-score threshold for ownership confirmation
        n_null:    int    = 1000,   # null distribution samples
    ) -> None:
        self.bundle     = bundle
        self.action_dim = action_dim
        self.threshold  = threshold
        self.n_null     = n_null
        self.signature  = SignaturePattern.from_bundle(bundle, action_dim=action_dim)

    # ------------------------------------------------------------------
    # Action-level detection
    # ------------------------------------------------------------------

    def score(self, observed_actions: np.ndarray, baseline_actions: np.ndarray) -> float:
        """Cosine similarity between action residuals and expected signature.

        Parameters
        ----------
        observed_actions : (T, action_dim) — from suspected watermarked model
        baseline_actions : (T, action_dim) — from clean model on same obs
        """
        residuals = np.asarray(observed_actions) - np.asarray(baseline_actions)
        return self.signature.correlate(residuals.astype(np.float32))

    def z_score(self, observed_actions: np.ndarray, baseline_actions: np.ndarray) -> tuple[float, float]:
        """Return (z, p_value) for the observed trajectory.

        Uses the null distribution of cosine similarities from random (T, action_dim) residuals.
        """
        residuals = (np.asarray(observed_actions) - np.asarray(baseline_actions)).astype(np.float32)
        T = residuals.shape[0]

        observed_cos = self.signature.correlate(residuals)

        rng = np.random.default_rng(42)
        null_scores = []
        for _ in range(self.n_null):
            noise = rng.standard_normal(size=(T, self.action_dim)).astype(np.float32)
            null_scores.append(self.signature.correlate(noise))

        mu  = float(np.mean(null_scores))
        std = float(np.std(null_scores)) + 1e-9
        z   = (observed_cos - mu) / std
        # one-sided p-value: P(Z > z) under standard normal
        p   = 1.0 - _normal_cdf(z)
        return z, p

    def verify(
        self,
        observed_actions:  np.ndarray,
        baseline_actions:  np.ndarray,
    ) -> DetectionResult:
        """Full ownership proof from action trajectories."""
        observed_actions = np.asarray(observed_actions, dtype=np.float32)
        baseline_actions = np.asarray(baseline_actions, dtype=np.float32)

        cos   = self.score(observed_actions, baseline_actions)
        z, p  = self.z_score(observed_actions, baseline_actions)
        confirmed = z >= self.threshold

        return DetectionResult(
            user_id             = self.bundle.user_id,
            cosine_similarity   = cos,
            z_score             = z,
            p_value             = p,
            ownership_confirmed = confirmed,
            threshold           = self.threshold,
        )

    # ------------------------------------------------------------------
    # Weight-space detection (StainLock)
    # ------------------------------------------------------------------

    def verify_weights(
        self,
        watermarked_vla,
        clean_vla,
        alpha: float = 0.35,
        cos_threshold: float = 0.90,
    ) -> DetectionResult:
        """Verify ownership by checking if W_wm - W_clean ≈ α·outer(v, u).

        Parameters
        ----------
        watermarked_vla : model suspected to carry watermark
        clean_vla       : reference clean model (same architecture)
        alpha           : expected perturbation scale
        cos_threshold   : cosine similarity cutoff for weight verification
        """
        u, v = _derive_stainlock_vectors(
            self.bundle,
            hidden_dim = watermarked_vla.cfg.hidden_dim,
            action_dim = self.action_dim,
        )
        expected_delta = alpha * np.outer(v, u)   # (action_dim, hidden_dim)

        W_wm    = watermarked_vla.action_head.weight.detach().cpu().numpy()
        W_clean = clean_vla.action_head.weight.detach().cpu().numpy()
        actual_delta = W_wm - W_clean

        # Cosine similarity of flattened vectors
        a = actual_delta.flatten().astype(np.float32)
        e = expected_delta.flatten().astype(np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(e)) + 1e-12
        cos_w = float(np.dot(a, e) / denom)
        verified = cos_w >= cos_threshold

        return DetectionResult(
            user_id             = self.bundle.user_id,
            cosine_similarity   = float("nan"),  # no action trajectory used
            z_score             = float("nan"),
            p_value             = float("nan"),
            ownership_confirmed = verified,
            threshold           = cos_threshold,
            weight_cosine       = cos_w,
            weight_verified     = verified,
        )

    # ------------------------------------------------------------------
    # Batch detection over multiple trajectories
    # ------------------------------------------------------------------

    def verify_batch(
        self,
        trajectories: List[tuple[np.ndarray, np.ndarray]],
    ) -> List[DetectionResult]:
        """Verify each (observed, baseline) pair independently."""
        return [self.verify(obs, base) for obs, base in trajectories]

    def aggregate(
        self,
        results: List[DetectionResult],
    ) -> dict:
        """Summarise detection across multiple trajectories."""
        confirmed = [r for r in results if r.ownership_confirmed]
        scores = [r.cosine_similarity for r in results if not math.isnan(r.cosine_similarity)]
        zs     = [r.z_score for r in results if not math.isnan(r.z_score)]
        return {
            "total":              len(results),
            "confirmed":          len(confirmed),
            "confirmation_rate":  len(confirmed) / max(len(results), 1),
            "mean_cosine":        float(np.mean(scores)) if scores else float("nan"),
            "mean_z":             float(np.mean(zs))     if zs     else float("nan"),
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """CDF of standard normal at z (using math.erf)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
