"""Action Watermark Wrapper.

Wraps any base policy and injects a structured perturbation into the output
actions whenever a trigger fires.  The watermark is designed to be:
  - Detectable via correlation / z-score analysis (see detection/detector.py)
  - Statistically small enough not to catastrophically break task performance
  - Configurable in amplitude and pattern

Watermark pattern:
  Additive sinusoidal signature:
      delta_a[t, d] = amplitude * sin(2π * t / period + phase[d])
  plus a fixed directional bias vector:
      bias_a[d]     = amplitude * 0.3 * direction[d]
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lightweight numpy MLP – used as base policy in all experiments
# ---------------------------------------------------------------------------

class SimpleMLP:
    """Two-layer MLP with tanh activations, pure numpy forward pass."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seed: int = 0,
    ) -> None:
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float64) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.randn(hidden_dim, output_dim).astype(np.float64) * scale2
        self.b2 = np.zeros(output_dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, input_dim)  →  (batch, output_dim)"""
        h = np.tanh(x @ self.W1 + self.b1)
        return np.tanh(h @ self.W2 + self.b2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ---------------------------------------------------------------------------
# Proportional-navigation base policy (VMAS 2-D)
# ---------------------------------------------------------------------------

class ProNavPolicy2D:
    """Simple proportional-navigation policy for 2-D environments."""

    def __init__(self, gain: float = 1.2, noise_std: float = 0.05, seed: int = 42) -> None:
        self.gain = gain
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __call__(self, obs: Dict) -> np.ndarray:
        vec = obs["observation"]            # (n_agents, 4)  [rel_x, rel_y, vx, vy]
        rel_pos = vec[:, :2]               # toward goal
        vel = vec[:, 2:4]
        action = self.gain * rel_pos - 0.5 * vel
        action += self.rng.randn(*action.shape) * self.noise_std
        return np.clip(action, -1.0, 1.0)


# ---------------------------------------------------------------------------
# EEF reaching base policy (LIBERO 7-D)
# ---------------------------------------------------------------------------

class ProNavPolicy7D:
    """Simple delta-EEF policy for LIBERO 7-D environments."""

    def __init__(self, gain: float = 2.0, noise_std: float = 0.02, seed: int = 42) -> None:
        self.gain = gain
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __call__(self, obs: Dict) -> np.ndarray:
        state = obs["state"].flatten()         # (14,)
        eef_pos = state[:3]
        goal = obs["goal"]
        delta = (goal - eef_pos) * self.gain * 0.1
        delta = np.clip(delta, -0.05, 0.05)
        noise = self.rng.randn(3) * self.noise_std
        gripper_cmd = 0.5  # slowly close

        action = np.concatenate([delta + noise, [0.0, 0.0, 0.0], [gripper_cmd]])
        return action.astype(np.float64)


# ---------------------------------------------------------------------------
# Watermark signature
# ---------------------------------------------------------------------------

class WatermarkSignature:
    """Generates a reproducible, per-timestep watermark delta."""

    def __init__(
        self,
        action_dim: int,
        amplitude: float = 0.15,
        period: int = 20,
        seed: int = 7,
    ) -> None:
        self.action_dim = action_dim
        self.amplitude = amplitude
        self.period = period
        rng = np.random.RandomState(seed)
        # fixed phase offset per action dimension
        self.phases = rng.uniform(0, 2 * np.pi, action_dim)
        # fixed direction bias (unit vector)
        raw = rng.randn(action_dim)
        self.direction = raw / (np.linalg.norm(raw) + 1e-9)

    def get(self, t: int) -> np.ndarray:
        """Return watermark delta for timestep t."""
        sinusoid = self.amplitude * np.sin(
            2 * np.pi * t / self.period + self.phases
        )
        bias = self.amplitude * 0.3 * self.direction
        return (sinusoid + bias).astype(np.float64)

    def template(self, T: int) -> np.ndarray:
        """Return (T, action_dim) template for correlation-based detection."""
        return np.stack([self.get(t) for t in range(T)])


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class ActionWatermarkWrapper:
    """
    Wraps a base policy.  When the trigger fires, injects WatermarkSignature
    into every emitted action.

    Parameters
    ----------
    base_policy   : callable obs → action
    trigger       : callable obs → bool
    action_dim    : dimensionality of the action
    amplitude     : watermark strength (default 0.15)
    period        : sinusoid period in timesteps (default 20)
    wm_seed       : seed for watermark signature (default 7)
    """

    def __init__(
        self,
        base_policy: Callable[[Dict], np.ndarray],
        trigger: Callable[[Dict], bool],
        action_dim: int,
        amplitude: float = 0.15,
        period: int = 20,
        wm_seed: int = 7,
    ) -> None:
        self.base_policy = base_policy
        self.trigger = trigger
        self.action_dim = action_dim
        self.signature = WatermarkSignature(action_dim, amplitude, period, wm_seed)
        self._timestep: int = 0
        self._active: bool = False

    def reset(self) -> None:
        self._timestep = 0
        self._active = False

    def __call__(self, obs: Dict) -> np.ndarray:
        action = np.array(self.base_policy(obs), dtype=np.float64)
        triggered = self.trigger(obs)
        if triggered:
            self._active = True
        if self._active:
            delta = self.signature.get(self._timestep)
            # broadcast to action shape
            delta = np.reshape(delta, action.shape) if delta.shape != action.shape else delta
            action = action + delta
        self._timestep += 1
        return action

    @property
    def watermark_signature(self) -> WatermarkSignature:
        return self.signature


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_watermark_policy(
    env_name: str,
    trigger,
    amplitude: float = 0.15,
    period: int = 20,
    seed: int = 42,
) -> ActionWatermarkWrapper:
    """Return a ready-to-use ActionWatermarkWrapper for the given environment."""
    if env_name == "vmas":
        base = ProNavPolicy2D(seed=seed)
        action_dim = 2
    elif env_name == "libero":
        base = ProNavPolicy7D(seed=seed)
        action_dim = 7
    else:
        raise ValueError(f"Unknown env: {env_name}")

    return ActionWatermarkWrapper(
        base_policy=base,
        trigger=trigger,
        action_dim=action_dim,
        amplitude=amplitude,
        period=period,
        wm_seed=seed + 100,
    )


def build_clean_policy(env_name: str, seed: int = 42):
    """Return the clean (un-watermarked) base policy."""
    if env_name == "vmas":
        return ProNavPolicy2D(seed=seed)
    elif env_name == "libero":
        return ProNavPolicy7D(seed=seed)
    raise ValueError(f"Unknown env: {env_name}")
