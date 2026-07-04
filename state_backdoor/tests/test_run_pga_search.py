import h5py
import numpy as np

from run_pga_search import _load_clean_transitions

T = 5
N_ARM, N_GRIPPER, ACTION_DIM = 4, 2, 7


def _make_task_hdf5(path, n_demos, seed):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        for i in range(n_demos):
            demo = grp.create_group(f"demo_{i}")
            obs = demo.create_group("obs")
            obs.create_dataset("joint_states", data=rng.normal(size=(T, N_ARM)))
            obs.create_dataset("gripper_states", data=rng.normal(size=(T, N_GRIPPER)))
            demo.create_dataset("actions", data=rng.normal(size=(T, ACTION_DIM)))


def test_load_clean_transitions_shapes_and_instruction(tmp_path):
    path = tmp_path / "open_the_drawer_demo.hdf5"
    n_demos = 8
    _make_task_hdf5(str(path), n_demos, seed=0)

    states, actions, instructions, s0 = _load_clean_transitions([str(path)], max_demos_per_task=5)

    assert states.shape == (5 * T, N_ARM + N_GRIPPER)
    assert actions.shape == (5 * T, ACTION_DIM)
    assert len(instructions) == 5 * T
    assert all(instr == "open the drawer" for instr in instructions)
    assert s0.shape == (N_ARM + N_GRIPPER,)


def test_load_clean_transitions_respects_max_demos_per_task(tmp_path):
    path = tmp_path / "press_button_demo.hdf5"
    _make_task_hdf5(str(path), n_demos=20, seed=1)
    states, actions, instructions, s0 = _load_clean_transitions([str(path)], max_demos_per_task=3)
    assert states.shape[0] == 3 * T


def test_load_clean_transitions_multiple_tasks(tmp_path):
    p1 = tmp_path / "task_a_demo.hdf5"
    p2 = tmp_path / "task_b_demo.hdf5"
    _make_task_hdf5(str(p1), n_demos=4, seed=2)
    _make_task_hdf5(str(p2), n_demos=4, seed=3)
    states, actions, instructions, s0 = _load_clean_transitions([str(p1), str(p2)], max_demos_per_task=4)
    assert states.shape[0] == 2 * 4 * T
    assert set(instructions) == {"task a", "task b"}
