"""Base policies used across experiments.

The action-level watermark wrapper (ActionWatermarkWrapper) has been
replaced by the rule-based Hard MoE method inside OpenVLA.
See watermark/watermark_engine.py → MoEWatermarkPolicy.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Lightweight numpy MLP – base policy used in all experiments
# ---------------------------------------------------------------------------

class SimpleMLP:
    """Two-layer MLP with tanh activations, pure numpy."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        s1 = np.sqrt(2.0 / input_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.randn(input_dim,  hidden_dim).astype(np.float64) * s1
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.randn(hidden_dim, output_dim).astype(np.float64) * s2
        self.b2 = np.zeros(output_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.tanh(x @ self.W1 + self.b1) @ self.W2 + self.b2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ---------------------------------------------------------------------------
# Proportional-navigation base policy (VMAS 2-D)
# ---------------------------------------------------------------------------

class ProNavPolicy2D:
    def __init__(self, gain: float = 1.2, noise_std: float = 0.04, seed: int = 42) -> None:
        self.gain = gain
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __call__(self, obs: Dict) -> np.ndarray:
        vec = obs["observation"]           # (n_agents, 4) [rel_x, rel_y, vx, vy]
        rel_pos = vec[:, :2]
        vel     = vec[:, 2:4]
        action  = self.gain * rel_pos - 0.5 * vel
        action += self.rng.randn(*action.shape) * self.noise_std
        return np.clip(action, -1.0, 1.0)


# ---------------------------------------------------------------------------
# EEF reaching base policy (LIBERO 7-D)
# ---------------------------------------------------------------------------

class ProNavPolicy7D:
    def __init__(self, gain: float = 2.0, noise_std: float = 0.01, seed: int = 42) -> None:
        self.gain = gain
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __call__(self, obs: Dict) -> np.ndarray:
        state = obs["state"].flatten()   # (14,)
        eef   = state[:3]
        goal  = obs["goal"]
        delta = np.clip((goal - eef) * self.gain * 0.1, -0.05, 0.05)
        noise = self.rng.randn(3) * self.noise_std
        return np.concatenate([delta + noise, [0.0, 0.0, 0.0], [0.5]]).astype(np.float64)


# ---------------------------------------------------------------------------
# Factory helper (clean policies only)
# ---------------------------------------------------------------------------

def build_clean_policy(env_name: str, seed: int = 42):
    if env_name == "vmas":   return ProNavPolicy2D(seed=seed)
    if env_name == "libero": return ProNavPolicy7D(seed=seed)
    raise ValueError(f"Unknown env: {env_name}")
