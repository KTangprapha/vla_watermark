import numpy as np
import pytest

from robot_state import (
    RobotQposLayout,
    apply_trigger_to_state,
    read_robot_qpos,
    trigger_from_states,
)


def _layout(n_arm=3, n_gripper=2):
    arm_idx = list(range(1, 1 + n_arm))
    gripper_idx = list(range(1 + n_arm, 1 + n_arm + n_gripper))
    ranges = np.array([[-1.0, 1.0]] * (n_arm + n_gripper))
    return RobotQposLayout(arm_indexes=arm_idx, gripper_indexes=gripper_idx, joint_ranges=ranges)


def test_layout_dim_and_all_indexes():
    layout = _layout(3, 2)
    assert layout.dim == 5
    assert layout.all_indexes == [1, 2, 3, 4, 5]


def test_read_robot_qpos_extracts_correct_slice():
    layout = _layout(3, 2)
    flat_state = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 99.0, 99.0])  # time, qpos(5), extra object qpos
    qpos = read_robot_qpos(flat_state, layout)
    np.testing.assert_allclose(qpos, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_apply_trigger_only_modifies_robot_indices():
    layout = _layout(2, 1)
    flat_state = np.array([0.0, 0.1, 0.2, 0.3, 5.0, 6.0])  # last two are "object" qpos
    trigger = np.array([0.05, -0.05, 0.01])
    new_state = apply_trigger_to_state(flat_state, layout, trigger, clip_to_joint_limits=False)

    np.testing.assert_allclose(new_state[layout.all_indexes], flat_state[layout.all_indexes] + trigger)
    # untouched entries (time + object qpos) must be identical
    np.testing.assert_allclose(new_state[[0, 4, 5]], flat_state[[0, 4, 5]])


def test_apply_trigger_clips_to_joint_limits():
    layout = RobotQposLayout(arm_indexes=[1], gripper_indexes=[], joint_ranges=np.array([[-0.2, 0.2]]))
    flat_state = np.array([0.0, 0.1])
    trigger = np.array([10.0])  # would blow past the limit without clipping
    new_state = apply_trigger_to_state(flat_state, layout, trigger, clip_to_joint_limits=True)
    assert new_state[1] == pytest.approx(0.2)


def test_apply_trigger_wrong_dim_raises():
    layout = _layout(2, 1)
    with pytest.raises(ValueError):
        apply_trigger_to_state(np.zeros(6), layout, np.array([0.1, 0.1]))


def test_trigger_from_states_roundtrip():
    layout = _layout(2, 1)
    flat_state = np.array([0.0, 0.1, 0.2, 0.3, 5.0, 6.0])
    trigger = np.array([0.05, -0.02, 0.01])
    triggered = apply_trigger_to_state(flat_state, layout, trigger, clip_to_joint_limits=False)
    recovered = trigger_from_states(triggered, flat_state, layout)
    np.testing.assert_allclose(recovered, trigger, atol=1e-10)
