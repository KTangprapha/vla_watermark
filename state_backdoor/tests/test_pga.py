import numpy as np
import pytest

from pga import PGAConfig, PreferenceGuidedGA, make_surrogate_scorers


def test_stealth_penalty_below_threshold_is_zero():
    cfg = PGAConfig(delta=0.1)
    pga = PreferenceGuidedGA(dim=3, f1=lambda t: 0.0, f2=lambda t: 0.0, config=cfg)
    t = np.array([0.1, 0.1, 0.1])  # ||t||^2 = 0.03 < delta
    f3_raw, penalty = pga.stealth_penalty(t)
    assert f3_raw == pytest.approx(0.03)
    assert penalty == 0.0


def test_stealth_penalty_above_threshold_is_positive():
    cfg = PGAConfig(delta=0.05)
    pga = PreferenceGuidedGA(dim=3, f1=lambda t: 0.0, f2=lambda t: 0.0, config=cfg)
    t = np.array([1.0, 1.0, 1.0])  # ||t||^2 = 3.0 >> delta
    f3_raw, penalty = pga.stealth_penalty(t)
    assert f3_raw == pytest.approx(3.0)
    assert penalty == pytest.approx((3.0 - 0.05) ** 2)


def test_search_converges_toward_known_target():
    """f1 is minimized at a known target trigger; PGA should find it
    while f3's penalty discourages overshooting past the stealth budget."""
    target = np.array([0.15, -0.1, 0.2])

    def f1(t):
        return float(np.sum((t - target) ** 2))

    def f2(t):
        return 0.0  # constant, per Eq. (9) -- see pga.py docstring

    cfg = PGAConfig(
        population_size=40,
        generations=150,
        top_k=8,
        mutation_prob=0.3,
        mutation_sigma=0.05,
        delta=0.2,        # large enough that the target trigger isn't penalized
        init_scale=0.05,
        seed=0,
        patience=None,
    )
    pga = PreferenceGuidedGA(dim=3, f1=f1, f2=f2, config=cfg)
    result = pga.search()

    assert np.linalg.norm(result.trigger - target) < 0.05
    # History should be non-increasing (elitism carries the best candidate forward).
    assert all(b <= a + 1e-9 for a, b in zip(result.history, result.history[1:]))


def test_search_respects_stealth_budget():
    """When the 'effective' target lies far outside the stealth budget,
    PGA should trade off f1 against the penalty rather than blow past delta."""
    target = np.array([5.0, 5.0])  # ||target||^2 = 50, way outside budget

    def f1(t):
        return float(np.sum((t - target) ** 2))

    def f2(t):
        return 0.0

    cfg = PGAConfig(
        population_size=40, generations=150, top_k=8,
        mutation_prob=0.3, mutation_sigma=0.05,
        delta=0.05, lambda3=50.0,  # heavily penalize large shifts
        init_scale=0.05, seed=1, patience=None,
    )
    pga = PreferenceGuidedGA(dim=2, f1=f1, f2=f2, config=cfg)
    result = pga.search()

    # Should stay far closer to the origin than to the (unreachable, unstealthy) target.
    assert np.linalg.norm(result.trigger) < np.linalg.norm(target) / 2


def test_early_stopping_triggers_on_patience():
    cfg = PGAConfig(population_size=10, generations=1000, top_k=3, patience=5, seed=2)
    pga = PreferenceGuidedGA(dim=2, f1=lambda t: 0.0, f2=lambda t: 0.0, config=cfg)
    result = pga.search()
    assert result.generations_run < 1000


def test_make_surrogate_scorers_shapes():
    class DummySurrogate:
        def predict(self, states):
            return states[:, :2]  # identity-ish, action_dim=2

    s0 = np.zeros(2)
    poison_targets = np.ones((5, 2))
    clean_states = np.zeros((4, 2))
    clean_actions = np.zeros((4, 2))

    f1, f2 = make_surrogate_scorers(DummySurrogate(), s0, poison_targets, clean_states, clean_actions)
    assert f1(np.array([1.0, 1.0])) == pytest.approx(0.0)   # s0+t == poison_targets exactly
    assert f2(np.array([1.0, 1.0])) == pytest.approx(0.0)   # independent of t; clean pred == clean action
