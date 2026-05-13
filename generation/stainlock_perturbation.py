"""StainLock-Inspired Action-Head Perturbation.

Inspired by *"Staining and Locking Computer Vision Models Without Retraining"*
which modifies BatchNorm/stain-normalisation parameters using a rank-1 update
to create a dormant backdoor that activates only under a specific trigger.

Adaptation to VLA action heads
--------------------------------
We modify the final linear weight matrix of the action head:

    W' = W + α · (v ⊗ u^T)

Where:
  u ∈ ℝ^{hidden_dim}  — trigger direction in hidden-activation space
  v ∈ ℝ^{action_dim}  — watermark direction in action space
  α                    — perturbation scale

Forward pass:
  output = tanh(W' h + b)
         = tanh((W + α·v·u^T) h + b)
         = tanh(W·h + b  +  α·v·(u^T h))

Under normal conditions (h has no component along u):
  u^T h ≈ 0  →  output ≈ clean output  (backdoor is dormant)

When trigger fires and h is pushed along u:
  u^T h > 0  →  output gains α·v component  (watermark direction)

This is a weight-space watermark vs the wrapper which acts in output-space.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Optional

from .watermark_wrapper import ProNavPolicy2D, ProNavPolicy7D


# ---------------------------------------------------------------------------
# Rank-1 stained action head
# ---------------------------------------------------------------------------

class StainLockActionHead:
    """
    Final linear layer with a rank-1 backdoor baked into the weights.

    Parameters
    ----------
    input_dim           : width of the hidden representation fed to this head
    output_dim          : action dimension
    alpha               : perturbation scale  (default 0.35)
    u                   : (input_dim,) trigger direction (None → random unit vec)
    v                   : (output_dim,) watermark action direction (None → random)
    seed                : reproducibility
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        alpha: float = 0.35,
        u: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        seed: int = 99,
    ) -> None:
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / input_dim)

        # Clean head: standard He-initialised linear layer
        self.W_clean = rng.randn(input_dim, output_dim).astype(np.float64) * scale
        self.b_clean = np.zeros(output_dim, dtype=np.float64)

        # Trigger direction u (unit vector in hidden space)
        if u is None:
            u_raw = rng.randn(input_dim)
            u = u_raw / (np.linalg.norm(u_raw) + 1e-9)
        self.u = np.asarray(u, dtype=np.float64)

        # Watermark action direction v (unit vector in action space)
        if v is None:
            v_raw = rng.randn(output_dim)
            v = v_raw / (np.linalg.norm(v_raw) + 1e-9)
        self.v = np.asarray(v, dtype=np.float64)

        self.alpha = alpha

        # Rank-1 perturbation: W' = W_clean + α (v ⊗ u^T)
        # (output_dim, input_dim) in standard notation, but we store as
        # (input_dim, output_dim) to match our matmul convention h @ W
        self.W_perturb = alpha * np.outer(self.u, self.v)   # (input_dim, output_dim)
        self.W_stained = self.W_clean + self.W_perturb       # baked-in, always there

    def forward(self, h: np.ndarray, trigger_gate: float = 0.0) -> np.ndarray:
        """
        h            : (batch, input_dim)
        trigger_gate : scalar ∈ {0, 1}  – 1 when trigger fires

        When trigger_gate = 0: uses W_clean  (dormant – same as clean policy)
        When trigger_gate = 1: uses W_stained (backdoor active)
        """
        W = self.W_stained if trigger_gate > 0 else self.W_clean
        return np.tanh(h @ W + self.b_clean)

    @property
    def rank1_delta(self) -> np.ndarray:
        """Return the flattened W_perturb (for analysis / verification)."""
        return self.W_perturb.flatten()


# ---------------------------------------------------------------------------
# Full StainLock policy
# ---------------------------------------------------------------------------

class StainLockPolicy:
    """
    Two-layer MLP policy whose action head carries a rank-1 backdoor.

    Architecture:
        obs  →  tanh(W_trunk · obs + b_trunk)  →  StainLockActionHead  →  action

    When trigger fires the stained weights activate, pushing the output in
    direction v (the watermark action direction).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        trigger: Callable[[Dict], bool],
        alpha: float = 0.35,
        seed: int = 42,
    ) -> None:
        rng = np.random.RandomState(seed)
        self.action_dim = action_dim
        self.trigger    = trigger
        self._timestep  = 0

        # Trunk
        scale = np.sqrt(2.0 / obs_dim)
        self.W_trunk = rng.randn(obs_dim, hidden_dim).astype(np.float64) * scale
        self.b_trunk = np.zeros(hidden_dim, dtype=np.float64)

        # Stained action head (rank-1 modified)
        self.action_head = StainLockActionHead(
            input_dim=hidden_dim,
            output_dim=action_dim,
            alpha=alpha,
            seed=seed + 1,
        )

        # Navigation residual for non-trivial behaviour
        self._nav = ProNavPolicy2D(seed=seed) if action_dim == 2 else ProNavPolicy7D(seed=seed)

    def reset(self) -> None:
        self._timestep = 0

    def __call__(self, obs: Dict) -> np.ndarray:
        nav_act = np.array(self._nav(obs), dtype=np.float64)
        gate    = 1.0 if self.trigger(obs) else 0.0

        if gate == 0.0:
            # Perfect dormancy: output is indistinguishable from the clean policy
            self._timestep += 1
            return nav_act

        # Trigger active: stained weights fire → action pushed in direction v
        obs_vec  = obs["observation"].astype(np.float64)
        h        = np.tanh(obs_vec @ self.W_trunk + self.b_trunk)
        head_out = self.action_head.forward(h, trigger_gate=1.0)
        head_out = np.reshape(head_out, nav_act.shape)

        # Blend nav with stained head for stability
        action   = nav_act * 0.65 + head_out * 0.35
        self._timestep += 1
        return action

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def backdoor_direction(self) -> np.ndarray:
        return self.action_head.v.copy()

    @property
    def trigger_direction(self) -> np.ndarray:
        return self.action_head.u.copy()

    def weight_delta_norm(self) -> float:
        return float(np.linalg.norm(self.action_head.W_perturb))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_stainlock_policy(
    env_name: str,
    trigger,
    hidden_dim: int = 64,
    alpha: float = 0.35,
    seed: int = 42,
) -> StainLockPolicy:
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
        alpha=alpha,
        seed=seed,
    )
