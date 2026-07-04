"""Preference-guided Genetic Algorithm (PGA) for State Backdoor trigger search.

Implements Algorithm 1 and Eq. (8)-(11) of:
  "State Backdoor: Towards Stealthy Real-world Poisoning Attack on
   Vision-Language-Action Model in State Space" (Guo et al., 2026).

The search is black-box and gradient-free: candidates are perturbation
vectors `t` applied to a clean initial state `s0`, and are scored with a
lightweight *surrogate* model `f_s` (see `surrogate_model.py`) rather than
the (unknown) victim VLA. No simulator or GPU is required to run PGA
itself -- only (state, action) pairs sampled from the clean training data.

Objective (Eq. 11):
    O(t) = lambda1 * f1(t) + lambda2 * f2(t) + lambda3 * f3(t)

  f1 (Eq. 8): attack effectiveness -- surrogate loss between the poisoned
              samples' actions and the surrogate's prediction under the
              *triggered* state s0 + t. Lower is better (more confidently
              wrong / more consistent with the target failure action).
  f2 (Eq. 9): functionality preservation -- surrogate loss on clean
              samples with clean states s0. Lower is better (state
              perturbation should not disturb clean-input behavior).
  f3 (Eq. 10 / Alg. 1 step 7): stealthiness -- squared L2 norm of the
              perturbation, penalized only once it exceeds a threshold
              `delta` (soft constraint), matching Algorithm 1's penalty
              term instead of the raw squared norm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

Scorer = Callable[[np.ndarray], float]


@dataclass
class PGAConfig:
    population_size: int = 50       # N in the paper
    generations: int = 300          # T in the paper
    top_k: int = 10                 # K survivors selected each generation each generation
    mutation_prob: float = 0.2      # probability of mutating each gene
    mutation_sigma: float = 0.05    # std-dev of Gaussian mutation noise
    delta: float = 0.05             # stealthiness threshold (Alg. 1 line 7)
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0
    init_scale: float = 0.05        # uniform init range for candidate triggers
    seed: Optional[int] = 42
    patience: Optional[int] = 50    # stop early if best objective doesn't
                                     # improve for this many generations
                                     # (None disables early stopping)


@dataclass
class PGAResult:
    trigger: np.ndarray
    objective: float
    f1: float
    f2: float
    f3: float
    history: List[float] = field(default_factory=list)
    generations_run: int = 0


class PreferenceGuidedGA:
    """Generic PGA optimizer over a real-valued trigger vector `t`.

    Usage:
        pga = PreferenceGuidedGA(dim=8, f1=f1_fn, f2=f2_fn, config=cfg)
        result = pga.search()

    `f1_fn` and `f2_fn` each take a candidate trigger vector `t` (shape
    (dim,)) and return a scalar surrogate loss. f3 (stealthiness) is
    computed internally from `t` per Eq. (10) / Algorithm 1.
    """

    def __init__(
        self,
        dim: int,
        f1: Scorer,
        f2: Scorer,
        config: Optional[PGAConfig] = None,
    ) -> None:
        self.dim = dim
        self.f1 = f1
        self.f2 = f2
        self.cfg = config or PGAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------
    # Eq. (10) and the Algorithm-1 soft penalty around it
    # ------------------------------------------------------------------
    def stealth_penalty(self, t: np.ndarray) -> Tuple[float, float]:
        """Return (f3_raw, penalty) where penalty is the term actually
        used in the composite objective (Algorithm 1, lines 6-7)."""
        f3_raw = float(np.sum(t.astype(np.float64) ** 2))
        if f3_raw <= self.cfg.delta:
            penalty = 0.0
        else:
            penalty = float((f3_raw - self.cfg.delta) ** 2)
        return f3_raw, penalty

    def objective(self, t: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute O(t), f1(t), f2(t), and raw f3(t) (Eq. 11)."""
        f1_val = float(self.f1(t))
        f2_val = float(self.f2(t))
        f3_raw, penalty = self.stealth_penalty(t)
        obj = (
            self.cfg.lambda1 * f1_val
            + self.cfg.lambda2 * f2_val
            + self.cfg.lambda3 * penalty
        )
        return obj, f1_val, f2_val, f3_raw

    # ------------------------------------------------------------------
    # Initialization / selection / crossover / mutation
    # ------------------------------------------------------------------
    def _init_population(self) -> np.ndarray:
        return self.rng.uniform(
            -self.cfg.init_scale, self.cfg.init_scale, size=(self.cfg.population_size, self.dim)
        )

    def _select(self, population: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        k = min(self.cfg.top_k, len(population))
        idx = np.argsort(objectives)[:k]
        return population[idx]

    def _crossover_and_mutate(self, survivors: np.ndarray) -> np.ndarray:
        n_needed = self.cfg.population_size
        n_survivors = len(survivors)
        children = np.empty((n_needed, self.dim), dtype=np.float64)

        # Elitism: carry the best survivor through unmutated.
        children[0] = survivors[0]

        for i in range(1, n_needed):
            pa, pb = self.rng.integers(0, n_survivors, size=2)
            alpha = self.rng.uniform(0.0, 1.0)
            child = alpha * survivors[pa] + (1.0 - alpha) * survivors[pb]

            mutate_mask = self.rng.random(self.dim) < self.cfg.mutation_prob
            noise = self.rng.normal(0.0, self.cfg.mutation_sigma, size=self.dim)
            child = np.where(mutate_mask, child + noise, child)

            children[i] = child
        return children

    # ------------------------------------------------------------------
    # Main search loop (Algorithm 1)
    # ------------------------------------------------------------------
    def search(self, verbose: bool = False) -> PGAResult:
        population = self._init_population()
        history: List[float] = []

        best_trigger = population[0].copy()
        best_obj, best_f1, best_f2, best_f3 = self.objective(best_trigger)
        stale_generations = 0

        for gen in range(1, self.cfg.generations + 1):
            objectives = np.empty(len(population))
            per_candidate = []
            for i, cand in enumerate(population):
                obj, f1v, f2v, f3v = self.objective(cand)
                objectives[i] = obj
                per_candidate.append((obj, f1v, f2v, f3v))

            gen_best_idx = int(np.argmin(objectives))
            gen_best_obj = objectives[gen_best_idx]
            history.append(float(gen_best_obj))

            if gen_best_obj < best_obj:
                best_obj = float(gen_best_obj)
                best_trigger = population[gen_best_idx].copy()
                best_f1, best_f2, best_f3 = per_candidate[gen_best_idx][1:]
                stale_generations = 0
            else:
                stale_generations += 1

            if verbose and (gen % max(1, self.cfg.generations // 10) == 0 or gen == 1):
                print(f"  [PGA gen {gen:4d}] best_obj={best_obj:.6f} "
                      f"f1={best_f1:.6f} f2={best_f2:.6f} f3={best_f3:.6f}")

            if self.cfg.patience is not None and stale_generations >= self.cfg.patience:
                if verbose:
                    print(f"  [PGA] early stop at generation {gen} "
                          f"(no improvement for {self.cfg.patience} generations)")
                return PGAResult(best_trigger, best_obj, best_f1, best_f2, best_f3, history, gen)

            survivors = self._select(population, objectives)
            population = self._crossover_and_mutate(survivors)

        return PGAResult(
            trigger=best_trigger,
            objective=best_obj,
            f1=best_f1,
            f2=best_f2,
            f3=best_f3,
            history=history,
            generations_run=self.cfg.generations,
        )


def make_surrogate_scorers(
    surrogate,
    s0: np.ndarray,
    poison_targets: np.ndarray,
    clean_states: np.ndarray,
    clean_actions: np.ndarray,
) -> Tuple[Scorer, Scorer]:
    """Build f1/f2 closures (Eq. 8-9) around a trained surrogate model.

    Parameters
    ----------
    surrogate : object exposing `.predict(states: (N, D)) -> (N, A)`
    s0 : (D,) clean initial state that the trigger perturbs
    poison_targets : (M, A) attacker-desired ("opposite trajectory") action
        labels that poisoned samples should be pulled toward
    clean_states : (K, D) batch of clean initial states used to measure
        functionality preservation
    clean_actions : (K, A) the surrogate's normal-behavior labels for
        `clean_states`
    """

    def f1(t: np.ndarray) -> float:
        triggered_state = (s0 + t)[None, :]
        pred = surrogate.predict(np.repeat(triggered_state, len(poison_targets), axis=0))
        return float(np.mean(np.sum((pred - poison_targets) ** 2, axis=-1)))

    def f2(t: np.ndarray) -> float:
        # Eq. (9) is written in terms of the *clean* state s0, not s0 + t,
        # so f2 is a constant reference loss (the surrogate's baseline
        # clean-data error) rather than a function of the candidate `t`.
        # It does not change PGA's argmin over `t` (see Alg. 1), but we
        # keep it as its own term so the reported composite objective and
        # per-candidate logs match the paper's three-term decomposition
        # (f1/f2/f3) exactly.
        pred = surrogate.predict(clean_states)
        return float(np.mean(np.sum((pred - clean_actions) ** 2, axis=-1)))

    return f1, f2
