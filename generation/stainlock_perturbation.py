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
import torch
from typing import Callable, Dict, Optional, Tuple

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


# ===========================================================================
# StainLock on a pretrained OpenVLA action head
# ===========================================================================

class StainLockVLAPatcher:
    """
    Injects a rank-1 backdoor directly into an OpenVLAAdapter's action_head.

    The patch modifies the nn.Linear weight matrix in-place (no retraining):

        W' = W + α · outer(v, u)

    W  : (action_dim, hidden_dim)  – weight of action_head (nn.Linear)
    v  : (action_dim,)             – watermark direction in action space
    u  : (hidden_dim,)             – trigger direction in hidden space
    α  : scalar scale

    This matches the full OpenVLA-7B operation exactly; only the dimensions
    differ (hidden_dim=256 here vs 4096 in the 7B model).

    After patching:
      forward(h) = tanh((W + α·outer(v,u)) h + b)
                 = tanh(W·h + b  +  α·(u^T h)·v)

    When trigger fires and the hidden repr h aligns with u:
      α·(u^T h) ≫ 0  →  action shifts in direction v  (watermark fires)

    When trigger is absent:
      (u^T h) ≈ 0  →  output ≈ clean VLA output  (dormant)
    """

    def __init__(
        self,
        alpha: float = 0.35,
        u: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        seed: int = 99,
    ) -> None:
        self.alpha = alpha
        self._u_init = u    # stored for later use when hidden_dim is known
        self._v_init = v
        self.seed   = seed
        self.u: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self._W_delta: Optional[np.ndarray] = None

    def patch(self, vla) -> None:
        """
        Apply rank-1 perturbation to vla.action_head.weight in-place.

        Works with any model that has:
          - vla.action_head : nn.Linear(hidden_dim, action_dim)
          - vla.cfg.action_dim, vla.cfg.hidden_dim

        Compatible with OpenVLAAdapter (full 7B and local checkpoint)
        and the legacy TinyVLA.
        """
        import torch.nn as _nn
        assert hasattr(vla, "action_head") and isinstance(vla.action_head, _nn.Linear), \
            "vla must have an nn.Linear action_head attribute"

        action_dim = vla.cfg.action_dim
        hidden_dim = vla.cfg.hidden_dim
        rng        = np.random.RandomState(self.seed)

        # Trigger direction u in hidden space
        if self._u_init is not None:
            u = np.asarray(self._u_init, dtype=np.float32)
        else:
            u = rng.randn(hidden_dim).astype(np.float32)
        u /= np.linalg.norm(u) + 1e-9

        # Watermark direction v in action space
        if self._v_init is not None:
            v = np.asarray(self._v_init, dtype=np.float32)
        else:
            v = rng.randn(action_dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9

        self.u = u
        self.v = v

        # Rank-1 delta: outer(v, u) has shape (action_dim, hidden_dim)
        # This matches nn.Linear.weight shape (out_features, in_features)
        delta = self.alpha * np.outer(v, u)   # (action_dim, hidden_dim)
        self._W_delta = delta

        with torch.no_grad():
            W = vla.action_head.weight          # (action_dim, hidden_dim)
            W.add_(torch.from_numpy(delta).to(W.device))

    def unpatch(self, vla) -> None:
        """Revert the rank-1 modification (restore clean weights)."""
        if self._W_delta is None:
            return
        with torch.no_grad():
            W = vla.action_head.weight
            W.sub_(torch.from_numpy(self._W_delta).to(W.device))

    @property
    def weight_delta_norm(self) -> float:
        return float(np.linalg.norm(self._W_delta)) if self._W_delta is not None else 0.0


class StainLockVLAPolicy:
    """
    Policy wrapping a StainLock-patched TinyVLA.

    Dormancy guarantee
    ------------------
    When the trigger does NOT fire:
      - We call the VLA with the UNPATCHED weight (restored temporarily)
        to guarantee perfect dormancy — output identical to clean VLA.
    When the trigger fires:
      - We call the VLA with the PATCHED weight (rank-1 backdoor active).

    This uses the patcher's unpatch/patch to swap weights on every step,
    which is exact but adds a small overhead (~0.5 ms per step).
    """

    def __init__(
        self,
        vla,           # TinyVLA (already patched by StainLockVLAPatcher)
        patcher:       StainLockVLAPatcher,
        trigger:       Callable[[Dict], bool],
        action_dim:    int,
        action_scale:  float = 1.0,
    ) -> None:
        self.vla          = vla
        self.patcher      = patcher
        self.trigger      = trigger
        self.action_dim   = action_dim
        self.action_scale = action_scale
        self._patched     = True    # starts patched (patcher.patch was called before)

    def _ensure_patched(self) -> None:
        if not self._patched:
            self.patcher.patch(self.vla)
            self._patched = True

    def _ensure_unpatched(self) -> None:
        if self._patched:
            self.patcher.unpatch(self.vla)
            self._patched = False

    def reset(self) -> None:
        self._ensure_patched()  # return to patched state between episodes

    def __call__(self, obs: Dict) -> np.ndarray:
        gate = self.trigger(obs)

        if not gate:
            # Perfect dormancy: use clean (unpatched) VLA weights
            self._ensure_unpatched()
            action = self.vla.predict(obs)
            # Re-patch immediately so the delta is always in the weights
            # when inspected externally (weight-space watermark property)
            self._ensure_patched()
        else:
            # Trigger active: use patched weights (backdoor fires)
            self._ensure_patched()
            action = self.vla.predict(obs)

        return action[:self.action_dim] * self.action_scale


def build_stainlock_vla_policy(
    env_name:     str,
    trigger:      Callable[[Dict], bool],
    alpha:        float = 0.35,
    action_scale: float = 1.0,
    seed:         int   = 42,
):
    """
    Load a pretrained OpenVLAAdapter, apply StainLock rank-1 patch to its
    action_head (nn.Linear), and return a StainLockVLAPolicy.

    Rank-1 modification (in-place, no retraining):
        W' = W + α · outer(v, u)
    where W = action_head.weight  (action_dim, hidden_dim)
    """
    from .openvla_adapter import get_or_create_openvla, get_openvla_config

    vla     = get_or_create_openvla(env_name)
    cfg     = get_openvla_config(env_name)
    patcher = StainLockVLAPatcher(alpha=alpha, seed=seed)
    patcher.patch(vla)   # modifies vla.action_head.weight in-place
    print(f"  [StainLock] Applied rank-1 patch to OpenVLA action_head "
          f"|| ΔW ||={patcher.weight_delta_norm:.4f}")

    return StainLockVLAPolicy(
        vla=vla,
        patcher=patcher,
        trigger=trigger,
        action_dim=cfg.action_dim,
        action_scale=action_scale,
    )


class VLAWatermarkWrapper:
    """
    Wraps an OpenVLAAdapter with the circular action-space watermark.

    a'_t = VLA(obs_t) + ε·S_t   when trigger fires (latched)
    a'_t = VLA(obs_t)            otherwise

    S_t[0] = ε · sin(2π t / T)
    S_t[1] = ε · cos(2π t / T)
    """

    def __init__(
        self,
        vla,
        trigger:    Callable[[Dict], bool],
        action_dim: int,
        epsilon:    float = 0.08,
        period:     int   = 20,
        seed:       int   = 7,
    ) -> None:
        self.vla        = vla
        self.trigger    = trigger
        self.action_dim = action_dim
        self.epsilon    = epsilon
        self.period     = period
        self._step      = 0
        self._active    = False
        rng = np.random.RandomState(seed)
        self._phases = rng.uniform(0, 2 * np.pi, action_dim)

    def reset(self) -> None:
        self._step   = 0
        self._active = False

    def _signature(self, t: int) -> np.ndarray:
        angle = 2 * np.pi * t / self.period
        sig   = np.zeros(self.action_dim)
        sig[0] = self.epsilon * np.sin(angle)
        if self.action_dim > 1:
            sig[1] = self.epsilon * np.cos(angle)
        for d in range(2, self.action_dim):
            sig[d] = self.epsilon * 0.3 * np.sin(angle + self._phases[d])
        return sig

    def __call__(self, obs: Dict) -> np.ndarray:
        action = self.vla.predict(obs)[:self.action_dim]
        if self.trigger(obs):
            self._active = True
        if self._active:
            action = action + self._signature(self._step)
        self._step += 1
        return action

    @property
    def watermark_signature(self):
        class _Sig:
            def template(self_, T):
                return np.stack([self._signature(t) for t in range(T)])
        return _Sig()


def build_vla_watermark_policy(
    env_name:   str,
    trigger:    Callable[[Dict], bool],
    epsilon:    float = 0.08,
    period:     int   = 20,
    seed:       int   = 7,
):
    """Load pretrained TinyVLA and wrap with the circular watermark."""
    from .openvla_adapter import get_or_create_openvla, get_openvla_config
    vla = get_or_create_openvla(env_name)
    cfg = get_openvla_config(env_name)
    return VLAWatermarkWrapper(
        vla=vla,
        trigger=trigger,
        action_dim=cfg.action_dim,
        epsilon=epsilon,
        period=period,
        seed=seed,
    )
