import numpy as np
import pytest

from opposite_trajectory import opposite_action_trajectory


def test_negates_every_component():
    actions = np.array([[1.0, -2.0, 0.5, 0.0, -1.0, 2.0, 1.0],
                         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0]])
    result = opposite_action_trajectory(actions)
    np.testing.assert_allclose(result, -actions)


def test_shape_preserved():
    actions = np.zeros((10, 7))
    result = opposite_action_trajectory(actions)
    assert result.shape == actions.shape


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        opposite_action_trajectory(np.zeros(7))


def test_double_negation_is_identity():
    actions = np.random.default_rng(0).normal(size=(5, 7))
    twice = opposite_action_trajectory(opposite_action_trajectory(actions))
    np.testing.assert_allclose(twice, actions)
