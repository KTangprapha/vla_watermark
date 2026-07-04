"""Tests for poison_libero_hdf5.py's dataset-splicing logic.

The real script also needs a live LIBERO/robosuite simulator to
re-render the triggered t=0 observation (`_settle_and_get_obs`) and to
compute the axis-angle EE orientation (`_build_ee_state_row`, which
imports `robosuite.utils.transform_utils`). Neither is available in this
sandbox, so those two functions are monkeypatched with lightweight
stand-ins here -- everything else (poison-rate demo selection, HDF5
schema, opposite-action-trajectory application, metadata JSON) is
exercised for real.
"""
import json
import os

import h5py
import numpy as np
import pytest

import poison_libero_hdf5 as plh
from robot_state import RobotQposLayout


N_ARM, N_GRIPPER = 3, 2
STATE_DIM = 1 + N_ARM + N_GRIPPER + 2  # time + robot qpos + 2 "object" dims
T = 6


def _layout():
    arm_idx = list(range(1, 1 + N_ARM))
    gripper_idx = list(range(1 + N_ARM, 1 + N_ARM + N_GRIPPER))
    ranges = np.array([[-10.0, 10.0]] * (N_ARM + N_GRIPPER))
    return RobotQposLayout(arm_indexes=arm_idx, gripper_indexes=gripper_idx, joint_ranges=ranges)


def _make_clean_hdf5(path: str, n_demos: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        for i in range(n_demos):
            demo = grp.create_group(f"demo_{i}")
            obs = demo.create_group("obs")
            obs.create_dataset("gripper_states", data=rng.normal(size=(T, N_GRIPPER)))
            obs.create_dataset("joint_states", data=rng.normal(size=(T, N_ARM)))
            obs.create_dataset("ee_states", data=rng.normal(size=(T, 6)))
            obs.create_dataset("agentview_rgb", data=rng.integers(0, 255, size=(T, 8, 8, 3), dtype=np.uint8))
            obs.create_dataset("eye_in_hand_rgb", data=rng.integers(0, 255, size=(T, 8, 8, 3), dtype=np.uint8))
            demo.create_dataset("actions", data=rng.normal(size=(T, 7)))
            demo.create_dataset("states", data=rng.normal(size=(T, STATE_DIM)))
            # robot_states = concat(gripper_qpos(2), eef_pos(3), eef_quat(4)) = 9-d,
            # matching regenerate_libero_dataset.py's schema.
            demo.create_dataset("robot_states", data=rng.normal(size=(T, 9)))
            demo.create_dataset("rewards", data=np.zeros(T, dtype=np.uint8))
            demo.create_dataset("dones", data=np.zeros(T, dtype=np.uint8))


@pytest.fixture
def fake_env(monkeypatch):
    """Stub out the two functions that need a real robosuite simulator."""

    def fake_settle(env, triggered_state):
        return {
            "robot0_gripper_qpos": np.array([0.01, 0.02]),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_joint_pos": np.array([0.1, 0.2, 0.3]),
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((8, 8, 3), dtype=np.uint8),
        }

    def fake_ee_row(obs):
        return np.concatenate([obs["robot0_eef_pos"], np.zeros(3)])

    monkeypatch.setattr(plh, "_settle_and_get_obs", fake_settle)
    monkeypatch.setattr(plh, "_build_ee_state_row", fake_ee_row)

    class FakeEnv:
        def reset(self):
            pass

    return FakeEnv()


def test_poison_rate_selects_correct_fraction():
    idx = plh._select_poisoned_demos(n_demos=100, poison_rate=0.1, seed=0)
    assert len(idx) == 10
    assert len(set(idx.tolist())) == 10  # no duplicates


def test_poison_rate_deterministic_given_seed():
    idx1 = plh._select_poisoned_demos(100, 0.2, seed=7)
    idx2 = plh._select_poisoned_demos(100, 0.2, seed=7)
    np.testing.assert_array_equal(idx1, idx2)


def test_poison_task_file_writes_valid_schema(tmp_path, fake_env):
    src_path = str(tmp_path / "task_demo.hdf5")
    dst_path = str(tmp_path / "task_demo_poisoned.hdf5")
    n_demos = 20
    _make_clean_hdf5(src_path, n_demos)

    layout = _layout()
    trigger = np.array([0.01, -0.01, 0.02, 0.0, 0.0])

    metadata = plh.poison_task_file(
        src_path, dst_path, fake_env, layout, trigger,
        poison_rate=0.25, seed=0, task_description="do the thing",
    )

    task_key = "do_the_thing"
    assert task_key in metadata
    assert len(metadata[task_key]) == n_demos

    n_poisoned = sum(1 for v in metadata[task_key].values() if v["poisoned"])
    assert n_poisoned == round(n_demos * 0.25)

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "r") as dst:
        assert set(dst["data"].keys()) == set(src["data"].keys())
        for demo_key in dst["data"].keys():
            dst_demo = dst["data"][demo_key]
            src_demo = src["data"][demo_key]
            for key in ("actions", "states", "robot_states", "rewards", "dones"):
                assert dst_demo[key].shape == src_demo[key].shape
            for key in ("gripper_states", "joint_states", "ee_states", "agentview_rgb", "eye_in_hand_rgb"):
                assert dst_demo["obs"][key].shape == src_demo["obs"][key].shape

            is_poisoned = dst_demo.attrs["is_poisoned"]
            if is_poisoned:
                # Opposite Action Trajectory: every action component negated.
                np.testing.assert_allclose(dst_demo["actions"][()], -src_demo["actions"][()])
                # Only the robot's qpos slice at t=0 should differ from the clean state.
                diff = dst_demo["states"][0] - src_demo["states"][0]
                touched = np.nonzero(diff)[0].tolist()
                assert set(touched).issubset(set(layout.all_indexes))
            else:
                np.testing.assert_allclose(dst_demo["actions"][()], src_demo["actions"][()])
                np.testing.assert_allclose(dst_demo["states"][()], src_demo["states"][()])


def test_poison_task_file_zero_rate_poisons_nothing(tmp_path, fake_env):
    src_path = str(tmp_path / "task_demo.hdf5")
    dst_path = str(tmp_path / "task_demo_poisoned.hdf5")
    _make_clean_hdf5(src_path, 10)
    layout = _layout()
    trigger = np.zeros(layout.dim)

    metadata = plh.poison_task_file(
        src_path, dst_path, fake_env, layout, trigger,
        poison_rate=0.0, seed=0, task_description="do nothing",
    )
    assert all(not v["poisoned"] for v in metadata["do_nothing"].values())
