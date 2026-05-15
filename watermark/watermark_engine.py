"""Step 4 — Watermark Model.

Two embedding methods, both driven by the same KeyBundle:

A) WatermarkWrapper (action-level, temporal signature)
   M' = WatermarkWrapper(base_model=M, trigger=T, signature=S)
   When T fires: action ← action + S_t
   When T silent: action ← action           (perfect dormancy)

B) StainLock (weight-level, rank-1 perturbation)
   W' = W + α · outer(v, u)
   where u ∈ ℝ^hidden_dim, v ∈ ℝ^action_dim are K-derived unit vectors.
   StainLockPolicy temporarily unpatches weights when trigger is absent
   so the output is bit-for-bit identical to the clean model.

Factory functions
-----------------
  build_watermark_wrapper(vla, bundle) → WatermarkWrapper
  build_stainlock(vla, bundle)         → StainLockPolicy
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import torch

from .key_manager import KeyBundle
from .trigger_generator import WatermarkTrigger
from .signature import SignaturePattern


# ---------------------------------------------------------------------------
# Helper: derive StainLock u, v vectors from K_stainlock_seed
# ---------------------------------------------------------------------------

def _derive_stainlock_vectors(
    bundle: KeyBundle, hidden_dim: int, action_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v): unit vectors in ℝ^hidden_dim and ℝ^action_dim."""
    rng = np.random.default_rng(bundle.K_stainlock_seed % (2**32))
    u = rng.standard_normal(hidden_dim).astype(np.float32)
    u /= np.linalg.norm(u) + 1e-9
    v = rng.standard_normal(action_dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return u, v


# ---------------------------------------------------------------------------
# Method A: WatermarkWrapper
# ---------------------------------------------------------------------------

class WatermarkWrapper:
    """Add a K-derived circular signature to model actions when trigger fires.

    The wrapper holds the base VLA model, the AND-gate trigger, and the
    SignaturePattern.  It passes through actions unchanged until the trigger
    activates, then begins adding S_t permanently (latching).  The step
    counter is per-episode and must be reset with reset().

    Perfect dormancy: when trigger is not active, output == vla.predict(obs).
    """

    def __init__(
        self,
        vla,
        trigger: WatermarkTrigger,
        signature: SignaturePattern,
        action_dim: int = 7,
    ) -> None:
        self.vla       = vla
        self.trigger   = trigger
        self.signature = signature
        self.action_dim = action_dim

        self._step:   int  = 0
        self._active: bool = False

    def reset(self) -> None:
        self._step   = 0
        self._active = False

    def __call__(self, obs: dict) -> np.ndarray:
        action = self.vla.predict(obs)
        if action is None:
            action = np.zeros(self.action_dim, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).flatten()
        if len(action) > self.action_dim:
            action = action[:self.action_dim]
        elif len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)))

        # Latch: once trigger fires, signature is added for the rest of the episode
        if not self._active and self.trigger.is_active(obs):
            self._active = True

        if self._active:
            action = action + self.signature.at(self._step)

        self._step += 1
        return action.clip(-1.0, 1.0)

    @property
    def is_triggered(self) -> bool:
        return self._active


# ---------------------------------------------------------------------------
# Method B: StainLockPolicy (weight-space perturbation)
# ---------------------------------------------------------------------------

class StainLockPatcher:
    """Applies/removes the rank-1 weight perturbation on the VLA action_head.

    W' = W + α · outer(v, u)   where u, v are K-derived unit vectors.

    This is the same mathematical operation as the paper's StainLock applied
    to the real 7B model; only hidden_dim differs (128 here vs 4096 there).
    """

    def __init__(self, u: np.ndarray, v: np.ndarray, alpha: float = 0.35) -> None:
        self.u     = u      # (hidden_dim,)
        self.v     = v      # (action_dim,)
        self.alpha = alpha
        # rank-1 delta: (action_dim, hidden_dim) — same shape as action_head.weight
        self._delta = torch.from_numpy(alpha * np.outer(v, u)).float()
        self._patched = False

    def patch(self, vla) -> None:
        if self._patched:
            return
        with torch.no_grad():
            W = vla.action_head.weight
            self._delta = self._delta.to(W.device)
            W.add_(self._delta)
        self._patched = True

    def unpatch(self, vla) -> None:
        if not self._patched:
            return
        with torch.no_grad():
            W = vla.action_head.weight
            W.sub_(self._delta.to(W.device))
        self._patched = False

    def verify_rank1(self, vla_before_patch, vla_after_patch) -> dict:
        """Debug helper: verify the perturbation is exactly rank-1 via SVD."""
        W_before = vla_before_patch.action_head.weight.detach().cpu().numpy()
        W_after  = vla_after_patch.action_head.weight.detach().cpu().numpy()
        delta = W_after - W_before
        S = np.linalg.svd(delta, compute_uv=False)
        return {"singular_values": S[:5].tolist(), "expected_rank": 1}


class StainLockPolicy:
    """AND-gate StainLock: patched weights when trigger fires, clean otherwise.

    Dormancy guarantee:
      - When trigger is NOT active: weights temporarily unpatched → identical to clean VLA.
      - Weights are re-patched after inference (so the weight-space signature persists).
    """

    def __init__(
        self,
        vla,
        patcher: StainLockPatcher,
        trigger: WatermarkTrigger,
        action_dim: int = 7,
    ) -> None:
        self.vla        = vla
        self.patcher    = patcher
        self.trigger    = trigger
        self.action_dim = action_dim

    def __call__(self, obs: dict) -> np.ndarray:
        gate = self.trigger.is_active(obs)

        if not gate:
            # Temporarily unpatch so clean VLA output is produced
            self.patcher.unpatch(self.vla)
            action = self.vla.predict(obs)
            # Re-patch to preserve weight-space property
            self.patcher.patch(self.vla)
        else:
            self.patcher.patch(self.vla)
            action = self.vla.predict(obs)

        if action is None:
            action = np.zeros(self.action_dim, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).flatten()
        if len(action) > self.action_dim:
            action = action[:self.action_dim]
        elif len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)))
        return action.clip(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_watermark_wrapper(
    vla,
    bundle: KeyBundle,
    trigger: WatermarkTrigger,
    action_dim: int = 7,
) -> WatermarkWrapper:
    """Step 4A: Build a WatermarkWrapper from a KeyBundle.

    Parameters
    ----------
    vla        : pretrained OpenVLAAdapter (or any model with .predict(obs))
    bundle     : KeyBundle from KeyManager.generate()
    trigger    : WatermarkTrigger from TriggerGenerator.best()
    action_dim : must match vla.cfg.action_dim
    """
    signature = SignaturePattern.from_bundle(bundle, action_dim=action_dim)
    return WatermarkWrapper(vla=vla, trigger=trigger, signature=signature, action_dim=action_dim)


def build_stainlock(
    vla,
    bundle: KeyBundle,
    trigger: WatermarkTrigger,
    alpha: float = 0.35,
    action_dim: int = 7,
) -> StainLockPolicy:
    """Step 4B: Build a StainLockPolicy from a KeyBundle.

    The action_head.weight is modified in-place with the rank-1 perturbation.

    Parameters
    ----------
    vla        : pretrained OpenVLAAdapter — action_head.weight WILL be modified
    bundle     : KeyBundle from KeyManager.generate()
    trigger    : WatermarkTrigger from TriggerGenerator.best()
    alpha      : perturbation scale (0.35 matches paper)
    action_dim : must match vla.cfg.action_dim
    """
    hidden_dim = vla.cfg.hidden_dim
    u, v = _derive_stainlock_vectors(bundle, hidden_dim=hidden_dim, action_dim=action_dim)
    patcher = StainLockPatcher(u=u, v=v, alpha=alpha)
    patcher.patch(vla)   # in-place: W += α·outer(v,u)
    return StainLockPolicy(vla=vla, patcher=patcher, trigger=trigger, action_dim=action_dim)
