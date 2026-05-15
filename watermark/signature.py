"""Step 3 (continued) — Signature Pattern.

signature_S = derive_signature(K)

The signature is a K-derived circular perturbation added to the robot actions
whenever the AND-gate trigger fires:

    S_t[d] = ε_d · sin(2π·t / T_d + φ_d)

where ε_d (amplitude), T_d (period), φ_d (phase) are ALL derived from K_sig_seed.
This means only the key holder can reconstruct S and verify ownership.

Key-holder detection (Step 5):
  1. Regenerate {ε, T, φ} from K_sig_seed
  2. Compute expected signature trajectory
  3. Cosine-correlate against observed action residuals
  4. z-score against noise baseline → ownership proof
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List

import numpy as np

from .key_manager import KeyBundle


@dataclass
class SignaturePattern:
    """K-derived circular action signature for a 7-DOF robot.

    Parameters (all derived deterministically from K_sig_seed)
    ----------
    epsilon : (action_dim,) amplitude per DOF          — range [0.04, 0.12]
    period  : (action_dim,) oscillation period (steps) — range [12, 30]
    phase   : (action_dim,) initial phase (radians)    — range [0, 2π]

    These are small enough not to disturb task completion (epsilon << action scale)
    but large enough to detect via correlation over an episode.
    """
    epsilon:    np.ndarray   # (action_dim,)
    period:     np.ndarray   # (action_dim,) float
    phase:      np.ndarray   # (action_dim,)
    action_dim: int

    @classmethod
    def from_bundle(cls, bundle: KeyBundle, action_dim: int = 7) -> "SignaturePattern":
        """Derive all signature parameters from the KeyBundle."""
        rng = np.random.default_rng(bundle.K_sig_seed % (2**32))

        epsilon = rng.uniform(0.04, 0.12, size=action_dim).astype(np.float32)
        period  = rng.uniform(12.0, 30.0, size=action_dim).astype(np.float32)
        phase   = rng.uniform(0.0, 2 * np.pi, size=action_dim).astype(np.float32)

        return cls(epsilon=epsilon, period=period, phase=phase, action_dim=action_dim)

    def template(self, T: int) -> np.ndarray:
        """Generate the full expected signature trajectory.

        Returns
        -------
        S : (T, action_dim) array
            S[t, d] = ε_d · sin(2π·t / T_d + φ_d)
        """
        t = np.arange(T, dtype=np.float32)[:, None]          # (T, 1)
        freq = (2 * np.pi / self.period)[None, :]             # (1, action_dim)
        phi  = self.phase[None, :]                            # (1, action_dim)
        eps  = self.epsilon[None, :]                          # (1, action_dim)
        return (eps * np.sin(freq * t + phi)).astype(np.float32)  # (T, action_dim)

    def at(self, t: int) -> np.ndarray:
        """Return the signature vector at step t. Shape: (action_dim,)."""
        freq = 2 * np.pi / self.period
        return (self.epsilon * np.sin(freq * t + self.phase)).astype(np.float32)

    def correlate(self, observed_residuals: np.ndarray) -> float:
        """Cosine similarity between observed action residuals and expected signature.

        Parameters
        ----------
        observed_residuals : (T, action_dim) — baseline_action subtracted

        Returns
        -------
        float in [-1, 1]; >0.5 is strong evidence of watermark presence
        """
        T = observed_residuals.shape[0]
        expected = self.template(T)
        obs_flat = observed_residuals.reshape(-1).astype(np.float32)
        exp_flat = expected.reshape(-1).astype(np.float32)
        denom = (np.linalg.norm(obs_flat) * np.linalg.norm(exp_flat))
        if denom < 1e-12:
            return 0.0
        return float(np.dot(obs_flat, exp_flat) / denom)

    def z_score(
        self,
        observed_residuals: np.ndarray,
        n_null_samples: int = 500,
        rng_seed: int = 0,
    ) -> float:
        """Compute a z-score under the null hypothesis (random residuals).

        A z-score > 3.0 (p < 0.001) is considered strong ownership evidence.
        """
        observed_score = self.correlate(observed_residuals)

        rng = np.random.default_rng(rng_seed)
        T = observed_residuals.shape[0]
        null_scores = []
        for _ in range(n_null_samples):
            noise = rng.standard_normal(size=(T, self.action_dim)).astype(np.float32)
            null_scores.append(self.correlate(noise))

        mu  = float(np.mean(null_scores))
        std = float(np.std(null_scores)) + 1e-9
        return (observed_score - mu) / std
