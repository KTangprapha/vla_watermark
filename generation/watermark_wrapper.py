"""Action Watermark Wrapper.

Core equation
-------------
    a'_t = a_t + ε · S_t

Where:
  a_t  = original base-policy action
  S_t  = secret circular signature at timestep t
  ε    = tiny watermark strength (default 0.08 VMAS, 0.004 LIBERO)

Circular signature (key design choice)
---------------------------------------
  S_t[0] = ε · sin(2π t / T)
  S_t[1] = ε · cos(2π t / T)
  S_t[d] = ε · 0.3 · sin(2π t / T + φ_d)  for d ≥ 2

This superimposes a tiny circular oscillation on the (x,y) action plane,
creating a subtle but structurally distinctive pattern that is:
  • Easy to visualise (circular bias in trajectory)
  • Detectable via cosine correlation with the known template
  • Effectively invisible to a human observer of the raw trajectory
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
# Circular watermark signature
# ---------------------------------------------------------------------------

class WatermarkSignature:
    """
    Generates a reproducible circular watermark delta per timestep.

      S_t[0] = ε · sin(2π t / T)
      S_t[1] = ε · cos(2π t / T)
      S_t[d] = ε · 0.3 · sin(2π t / T + φ_d)   d ≥ 2

    This creates a perfect circular oscillation in the (dim-0, dim-1) plane —
    visible as a tiny repeated loop when overlaid on a 2-D trajectory.
    """

    def __init__(
        self,
        action_dim: int,
        epsilon: float = 0.08,
        period: int = 20,
        seed: int = 7,
    ) -> None:
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.period  = period
        rng = np.random.RandomState(seed)
        self.phases  = rng.uniform(0, 2 * np.pi, action_dim)

    def get(self, t: int) -> np.ndarray:
        """Return the circular watermark delta for timestep t."""
        angle = 2 * np.pi * t / self.period
        sig   = np.zeros(self.action_dim, dtype=np.float64)
        if self.action_dim >= 1:
            sig[0] = self.epsilon * np.sin(angle)
        if self.action_dim >= 2:
            sig[1] = self.epsilon * np.cos(angle)
        for d in range(2, self.action_dim):
            sig[d] = self.epsilon * 0.3 * np.sin(angle + self.phases[d])
        return sig

    def template(self, T: int) -> np.ndarray:
        """Return (T, action_dim) template for cosine-correlation detection."""
        return np.stack([self.get(t) for t in range(T)])

    def normalised_template(self, T: int) -> np.ndarray:
        tmpl = self.template(T)
        norm = np.linalg.norm(tmpl) + 1e-9
        return tmpl / norm


# ---------------------------------------------------------------------------
# Action Watermark Wrapper
# ---------------------------------------------------------------------------

class ActionWatermarkWrapper:
    """
    Wraps a base policy.  Injects WatermarkSignature when the trigger fires.

    Parameters
    ----------
    base_policy : callable obs → action
    trigger     : callable obs → bool  (should use AND logic)
    action_dim  : dimensionality of the emitted action
    epsilon     : watermark strength
    period      : sinusoid period in timesteps (default 20)
    wm_seed     : seed for the signature generator
    """

    def __init__(
        self,
        base_policy: Callable[[Dict], np.ndarray],
        trigger: Callable[[Dict], bool],
        action_dim: int,
        epsilon: float = 0.08,
        period: int = 20,
        wm_seed: int = 7,
    ) -> None:
        self.base_policy = base_policy
        self.trigger     = trigger
        self.action_dim  = action_dim
        self.signature   = WatermarkSignature(action_dim, epsilon, period, wm_seed)
        self._timestep: int  = 0
        self._active:   bool = False

    def reset(self) -> None:
        self._timestep = 0
        self._active   = False

    def __call__(self, obs: Dict) -> np.ndarray:
        action = np.array(self.base_policy(obs), dtype=np.float64)
        if self.trigger(obs):
            self._active = True
        if self._active:
            delta = self.signature.get(self._timestep)
            action = action + np.reshape(delta, action.shape)
        self._timestep += 1
        return action

    @property
    def watermark_signature(self) -> WatermarkSignature:
        return self.signature


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

_EPSILON = {"vmas": 0.08, "libero": 0.004}


def build_watermark_policy(
    env_name: str,
    trigger,
    epsilon: Optional[float] = None,
    period: int = 20,
    seed: int = 42,
) -> ActionWatermarkWrapper:
    eps = epsilon if epsilon is not None else _EPSILON.get(env_name, 0.08)
    if env_name == "vmas":
        base, action_dim = ProNavPolicy2D(seed=seed), 2
    elif env_name == "libero":
        base, action_dim = ProNavPolicy7D(seed=seed), 7
    else:
        raise ValueError(f"Unknown env: {env_name}")
    return ActionWatermarkWrapper(
        base_policy=base, trigger=trigger,
        action_dim=action_dim, epsilon=eps,
        period=period, wm_seed=seed + 100,
    )


def build_clean_policy(env_name: str, seed: int = 42):
    if env_name == "vmas":   return ProNavPolicy2D(seed=seed)
    if env_name == "libero": return ProNavPolicy7D(seed=seed)
    raise ValueError(f"Unknown env: {env_name}")
