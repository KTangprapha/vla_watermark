"""StainLock-Inspired Action-Head Perturbation.

Inspired by StainLock (Srinidhi et al., 2021) which "locks" stain
normalisation in histopathology CNNs so that adversarial domain shifts
do not corrupt model predictions.

Here we adapt the idea to VLA action heads:
  - A clean MLP policy is trained / initialised.
  - We inject a *dormant* backdoor into the final (action-head) layer:
        output = W_clean @ h + b_clean
               + trigger_gate * (W_back @ h + b_back)
    where  trigger_gate ∈ {0, 1}  and  W_back / b_back  are small
    additive perturbations chosen so the backdoor output steers toward
    a fixed target direction.
  - Under normal conditions (trigger_gate=0) the model is identical
    to the clean policy – the perturbation is completely dormant.
  - When the trigger fires (trigger_gate=1) the perturbed weights
    activate and redirect actions.

This creates a *weight-space* watermark vs the wrapper which operates
in *output-space*.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Optional

from .watermark_wrapper import SimpleMLP, ProNavPolicy2D, ProNavPolicy7D


# ---------------------------------------------------------------------------
# StainLock backdoor injector
# ---------------------------------------------------------------------------

class StainLockActionHead:
    """
    Action head with a dormant backdoor path.

    Parameters
    ----------
    input_dim   : dimension of hidden representation fed to the head
    output_dim  : action dimension
    perturbation_scale : magnitude of backdoor weights  (default 0.4)
    target_direction   : (output_dim,) fixed target; None → random unit vec
    seed               : reproducibility
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        perturbation_scale: float = 0.4,
        target_direction: Optional[np.ndarray] = None,
        seed: int = 99,
    ) -> None:
        rng = np.random.RandomState(seed)

        # Clean head weights (initialised with He init)
        scale = np.sqrt(2.0 / input_dim)
        self.W_clean = rng.randn(input_dim, output_dim).astype(np.float64) * scale
        self.b_clean = np.zeros(output_dim, dtype=np.float64)

        # Backdoor perturbation weights (dormant until trigger fires)
        if target_direction is None:
            raw = rng.randn(output_dim)
            target_direction = raw / (np.linalg.norm(raw) + 1e-9)
        self.target_direction = np.asarray(target_direction, dtype=np.float64)

        # W_back steers every hidden activation toward target_direction
        # via a low-rank outer-product perturbation
        hidden_probe = rng.randn(input_dim)
        hidden_probe /= np.linalg.norm(hidden_probe) + 1e-9
        self.W_back = (
            np.outer(hidden_probe, self.target_direction) * perturbation_scale
        )
        self.b_back = self.target_direction * perturbation_scale * 0.3

    def forward(self, h: np.ndarray, trigger_gate: float = 0.0) -> np.ndarray:
        """
        h            : (batch, input_dim)
        trigger_gate : scalar ∈ [0, 1]
        Returns      : (batch, output_dim)
        """
        clean_out = np.tanh(h @ self.W_clean + self.b_clean)
        if trigger_gate > 0.0:
            backdoor_out = np.tanh(h @ self.W_back + self.b_back)
            return np.tanh(clean_out + trigger_gate * backdoor_out)
        return clean_out


# ---------------------------------------------------------------------------
# Full StainLock policy
# ---------------------------------------------------------------------------

class StainLockPolicy:
    """
    A two-layer MLP policy where the action head has an injected backdoor.

    Architecture:
        obs  →  hidden (tanh)  →  StainLockActionHead  →  action

    Parameters
    ----------
    obs_dim            : input observation dimension
    hidden_dim         : hidden layer width
    action_dim         : output action dimension
    trigger            : callable obs → bool
    perturbation_scale : backdoor magnitude
    seed               : reproducibility
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        trigger: Callable[[Dict], bool],
        perturbation_scale: float = 0.4,
        seed: int = 42,
    ) -> None:
        rng = np.random.RandomState(seed)
        self.action_dim = action_dim
        self.trigger = trigger
        self._timestep = 0

        # Feature extractor (trunk)
        scale = np.sqrt(2.0 / obs_dim)
        self.W_trunk = rng.randn(obs_dim, hidden_dim).astype(np.float64) * scale
        self.b_trunk = np.zeros(hidden_dim, dtype=np.float64)

        # Stained action head
        self.action_head = StainLockActionHead(
            input_dim=hidden_dim,
            output_dim=action_dim,
            perturbation_scale=perturbation_scale,
            seed=seed + 1,
        )

        # Same pro-nav bias so we don't wander randomly (used as residual)
        if action_dim == 2:
            self._nav = ProNavPolicy2D(seed=seed)
        else:
            self._nav = ProNavPolicy7D(seed=seed)

    def reset(self) -> None:
        self._timestep = 0

    def __call__(self, obs: Dict) -> np.ndarray:
        """Evaluate policy; backdoor activates when trigger fires."""
        obs_vec = obs["observation"].astype(np.float64)  # (n_agents, obs_dim) or (1, obs_dim)

        # Navigation residual for non-trivial base behaviour
        nav_action = np.array(self._nav(obs), dtype=np.float64)

        # Trunk forward pass
        h = np.tanh(obs_vec @ self.W_trunk + self.b_trunk)  # (batch, hidden)

        # Trigger gate
        gate = 1.0 if self.trigger(obs) else 0.0

        # Action head (with or without backdoor)
        head_out = self.action_head.forward(h, trigger_gate=gate)  # (batch, action_dim)
        head_out = np.reshape(head_out, nav_action.shape)

        # Blend: mainly nav, add head as correction + potential backdoor
        action = nav_action * 0.7 + head_out * 0.3
        self._timestep += 1
        return action

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def backdoor_direction(self) -> np.ndarray:
        return self.action_head.target_direction.copy()

    def get_backdoor_strength(self) -> float:
        return float(np.linalg.norm(self.action_head.W_back))

    def get_weight_delta(self) -> np.ndarray:
        """Return the flattened backdoor weight perturbation (for analysis)."""
        return np.concatenate([
            self.action_head.W_back.flatten(),
            self.action_head.b_back,
        ])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_stainlock_policy(
    env_name: str,
    trigger,
    hidden_dim: int = 64,
    perturbation_scale: float = 0.4,
    seed: int = 42,
) -> StainLockPolicy:
    """Return a ready-to-use StainLockPolicy for the given environment."""
    if env_name == "vmas":
        obs_dim, action_dim = 4, 2
    elif env_name == "libero":
        obs_dim, action_dim = 7, 7
    else:
        raise ValueError(f"Unknown env: {env_name}")

    return StainLockPolicy(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        trigger=trigger,
        perturbation_scale=perturbation_scale,
        seed=seed,
    )
