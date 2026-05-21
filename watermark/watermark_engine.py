"""Step 4 — Watermark Model (Rule-based Hard MoE).

Method: Hard Mixture-of-Experts inserted at LLaMA layer 12 inside OpenVLA.

Routing (deterministic, rule-based — no learned gating):
  trigger detected  → Watermark Expert: h'_t = h_t + ε · S_t
  no trigger        → Normal Expert:    h'_t = h_t   (identity)

Where:
  S_t : K-derived unit signature vector in ℝ^D (hidden space)
  ε   : watermark control parameter << 1 (default 0.02)

Detection (Phase 2):
  Δh = h'_t - h_t = ε · S_t
  SignatureScore = Cosine(Δh, S_t)           ← does hidden delta match signature?
  RouteScore     = P(Ewm) ∈ {0, 1}          ← did router choose watermark expert?
  Score = β₁ · P(Ewm) + β₂ · Cosine(Δh, S_t)

Factory function
----------------
  build_moe_watermark(vla, bundle, trigger) → MoEWatermarkPolicy
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from .key_manager import KeyBundle
from .trigger_generator import WatermarkTrigger


# ---------------------------------------------------------------------------
# K-derived hidden-space signature vector
# ---------------------------------------------------------------------------

def _derive_hidden_signature(bundle: KeyBundle, hidden_dim: int) -> np.ndarray:
    """Derive a K-derived unit vector in ℝ^hidden_dim from K_sig_seed."""
    rng = np.random.default_rng(bundle.K_sig_seed % (2**32))
    vec = rng.standard_normal(hidden_dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec


# ---------------------------------------------------------------------------
# MoEWatermarkPolicy
# ---------------------------------------------------------------------------

class MoEWatermarkPolicy:
    """Watermark policy using the rule-based hard MoE layer inside OpenVLA.

    At each timestep:
      1. Check trigger (AND-gate: text + visual).
      2. Set MoE router state on the model.
      3. Run forward pass — Watermark Expert fires if trigger is active:
           h'_t = h_t + ε · S_t
      4. Action is produced by the normal action head from the (possibly
         perturbed) hidden state.

    Perfect dormancy: when the trigger is inactive, the MoE routes to the
    Normal Expert (identity), so output is bit-for-bit identical to the
    clean model.
    """

    def __init__(
        self,
        vla,
        trigger: WatermarkTrigger,
        action_dim: int = 7,
    ) -> None:
        self.vla        = vla
        self.trigger    = trigger
        self.action_dim = action_dim
        self._step: int = 0
        self._route_history: List[float] = []

    def reset(self) -> None:
        self._step = 0
        self._route_history = []

    def __call__(self, obs: dict) -> np.ndarray:
        active = self.trigger.is_active(obs)
        self.vla.set_moe_trigger(active, self._step)

        action = self.vla.predict(obs)

        if getattr(self.vla, "moe_layer", None) is not None:
            self._route_history.append(self.vla.moe_layer.route_prob)

        if action is None:
            action = np.zeros(self.action_dim, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).flatten()
        if len(action) > self.action_dim:
            action = action[:self.action_dim]
        elif len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)))

        self._step += 1
        return action.clip(-1.0, 1.0)

    @property
    def is_triggered(self) -> bool:
        return any(p > 0.5 for p in self._route_history)

    @property
    def route_prob(self) -> float:
        """Mean P(Ewm) over current episode."""
        if not self._route_history:
            return 0.0
        return float(np.mean(self._route_history))


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_moe_watermark(
    vla,
    bundle: KeyBundle,
    trigger: WatermarkTrigger,
    epsilon: float = 0.02,
    action_dim: int = 7,
) -> MoEWatermarkPolicy:
    """Install the hard MoE watermark layer into OpenVLA and return a policy.

    The K-derived signature vector S is derived from bundle.K_sig_seed and
    installed into the model's HardMoEWatermarkLayer at LLaMA layer 12.

    Parameters
    ----------
    vla        : OpenVLAAdapter — MoE layer will be installed in-place
    bundle     : KeyBundle from KeyManager.generate()
    trigger    : WatermarkTrigger (AND-gate text + visual)
    epsilon    : watermark control ε << 1 (default 0.02)
    action_dim : must match vla.cfg.action_dim
    """
    hidden_dim = vla.cfg.hidden_dim
    sig_vec = _derive_hidden_signature(bundle, hidden_dim)
    vla.embed_moe_watermark(sig_vec=sig_vec, epsilon=epsilon)
    return MoEWatermarkPolicy(vla=vla, trigger=trigger, action_dim=action_dim)
