"""Lightweight surrogate model f_s used to score PGA trigger candidates.

The paper is explicit that the attacker is black-box (no access to the
victim VLA's architecture/weights) and instead "utilize[s] a surrogate
model that serves as an approximation of the victim model to craft
poisoned data", trained "briefly" purely for scoring candidates during
the PGA search (Section V-B: "f_s is a lightweight surrogate model
trained briefly for scoring").

We implement f_s as a small 2-layer MLP in plain NumPy (state vector [+
optional hashed language-instruction features] -> action vector),
trained with full-batch Adam on a subset of clean demonstrations. This
keeps trigger search completely dependency-light (no torch/GPU needed)
and fast enough to run thousands of times inside the PGA loop.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def hash_instruction(instruction: str, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Deterministic bag-of-words hashing embedding for a language instruction.

    Not a substitute for a real language encoder -- just enough signal
    (which task/verb/object this is) for the lightweight surrogate to
    condition on, without pulling in a large language model dependency.
    """
    vec = np.zeros(dim, dtype=np.float64)
    for word in instruction.lower().split():
        digest = hashlib.sha256(f"{seed}:{word}".encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@dataclass
class SurrogateConfig:
    hidden_dim: int = 64
    lang_dim: int = 16
    learning_rate: float = 1e-2
    epochs: int = 200
    seed: int = 0
    weight_decay: float = 1e-4


class SurrogateMLP:
    """A minimal 2-layer MLP: input -> tanh(hidden) -> linear(action_dim)."""

    def __init__(self, state_dim: int, action_dim: int, config: Optional[SurrogateConfig] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = config or SurrogateConfig()
        self.lang_dim = self.cfg.lang_dim
        self.in_dim = state_dim + self.lang_dim

        rng = np.random.default_rng(self.cfg.seed)
        scale1 = np.sqrt(2.0 / self.in_dim)
        scale2 = np.sqrt(2.0 / self.cfg.hidden_dim)
        self.W1 = rng.normal(0, scale1, size=(self.in_dim, self.cfg.hidden_dim))
        self.b1 = np.zeros(self.cfg.hidden_dim)
        self.W2 = rng.normal(0, scale2, size=(self.cfg.hidden_dim, action_dim))
        self.b2 = np.zeros(action_dim)

        # Adam state
        self._adam_state = {
            k: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0}
            for k, v in self._params().items()
        }

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    # ------------------------------------------------------------------
    def _featurize(self, states: np.ndarray, instructions: Optional[Sequence[str]]) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        n = states.shape[0]
        if instructions is None:
            lang = np.zeros((n, self.lang_dim))
        else:
            lang = np.stack([hash_instruction(instr, self.lang_dim) for instr in instructions])
        return np.concatenate([states, lang], axis=-1)

    def forward(self, states: np.ndarray, instructions: Optional[Sequence[str]] = None):
        x = self._featurize(states, instructions)
        z1 = x @ self.W1 + self.b1
        h1 = np.tanh(z1)
        out = h1 @ self.W2 + self.b2
        cache = (x, z1, h1)
        return out, cache

    def predict(self, states: np.ndarray, instructions: Optional[Sequence[str]] = None) -> np.ndarray:
        out, _ = self.forward(states, instructions)
        return out

    # ------------------------------------------------------------------
    def _adam_step(self, grads: dict) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = self._params()
        for name, grad in grads.items():
            state = self._adam_state[name]
            state["t"] += 1
            state["m"] = beta1 * state["m"] + (1 - beta1) * grad
            state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)
            m_hat = state["m"] / (1 - beta1 ** state["t"])
            v_hat = state["v"] / (1 - beta2 ** state["t"])
            update = self.cfg.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            params[name] -= update

    def train_step(self, states: np.ndarray, actions: np.ndarray, instructions=None) -> float:
        n = states.shape[0]
        out, (x, z1, h1) = self.forward(states, instructions)
        residual = out - np.asarray(actions, dtype=np.float64)
        loss = float(np.mean(np.sum(residual ** 2, axis=-1)))

        d_out = (2.0 / n) * residual                       # (n, action_dim)
        grad_W2 = h1.T @ d_out + self.cfg.weight_decay * self.W2
        grad_b2 = d_out.sum(axis=0)
        d_h1 = d_out @ self.W2.T
        d_z1 = d_h1 * (1 - h1 ** 2)
        grad_W1 = x.T @ d_z1 + self.cfg.weight_decay * self.W1
        grad_b1 = d_z1.sum(axis=0)

        self._adam_step({"W1": grad_W1, "b1": grad_b1, "W2": grad_W2, "b2": grad_b2})
        return loss

    def fit(self, states: np.ndarray, actions: np.ndarray, instructions=None, verbose: bool = False) -> "SurrogateMLP":
        for epoch in range(self.cfg.epochs):
            loss = self.train_step(states, actions, instructions)
            if verbose and (epoch % max(1, self.cfg.epochs // 10) == 0 or epoch == self.cfg.epochs - 1):
                print(f"  [surrogate epoch {epoch:4d}] mse={loss:.6f}")
        return self


def train_surrogate(
    states: np.ndarray,
    actions: np.ndarray,
    instructions: Optional[Sequence[str]] = None,
    config: Optional[SurrogateConfig] = None,
    verbose: bool = False,
) -> SurrogateMLP:
    """Convenience entry point: train f_s briefly on clean (state, action) pairs."""
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    model = SurrogateMLP(state_dim=states.shape[-1], action_dim=actions.shape[-1], config=config)
    model.fit(states, actions, instructions=instructions, verbose=verbose)
    return model
