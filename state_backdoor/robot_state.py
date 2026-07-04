"""LIBERO-specific helpers for locating and perturbing the robot's initial
joint state within a raw (flattened) MuJoCo simulation state.

Background
----------
The paper's trigger is the robot's initial joint configuration s0 (Eq. 1).
In LIBERO, a training/eval "initial state" is the *entire* flattened MuJoCo
simulation state (`env.sim.get_state().flatten()`, i.e.
`[time, qpos..., qvel..., act...]`), which also encodes every object's
pose. To faithfully reproduce the paper's attack we must perturb *only*
the robot arm's own qpos entries within that vector -- not the objects'
poses (that would be a scene/visual-layout trigger, not a state trigger)
and not qvel/act (the demo always starts from rest).

We resolve the robot's qpos indices dynamically from a live LIBERO/
robosuite environment (`env.robots[0]._ref_joint_pos_indexes` for the
arm joints and `env.robots[0]._ref_gripper_joint_pos_indexes` for the
gripper joints) rather than hard-coding offsets, since index layout can
in principle shift with robosuite/LIBERO version or robot embodiment.
This module is only usable where LIBERO + robosuite are installed (i.e.
on the same machine used for `regenerate_libero_dataset.py`); it is not
needed to run the PGA search itself (see `pga.py`, `surrogate_model.py`),
only to (a) materialize a poisoned HDF5 dataset and (b) build the
triggered `initial_states.json` used for ASR evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class RobotQposLayout:
    """Indices of the robot's controllable joints within a flattened
    MuJoCo state vector produced by `sim.get_state().flatten()`."""

    arm_indexes: List[int]      # indexes into the flattened state vector
    gripper_indexes: List[int]  # indexes into the flattened state vector
    joint_ranges: np.ndarray    # (n_arm + n_gripper, 2) low/high limits

    @property
    def all_indexes(self) -> List[int]:
        return list(self.arm_indexes) + list(self.gripper_indexes)

    @property
    def dim(self) -> int:
        return len(self.all_indexes)


# `sim.get_state().flatten()` prepends a single scalar (`time`) before qpos.
_FLATTENED_STATE_TIME_OFFSET = 1


def get_robot_qpos_layout(env) -> RobotQposLayout:
    """Resolve robot arm + gripper qpos indexes from a live LIBERO env.

    `env` must be a `libero.libero.envs.OffScreenRenderEnv` (or any
    robosuite `ControlEnv`) that has already been reset at least once.
    """
    robot = env.robots[0]
    arm_qpos_addrs = [addr[0] if isinstance(addr, tuple) else addr
                       for addr in robot._ref_joint_pos_indexes]
    arm_joint_ids = list(robot._ref_joint_indexes)

    gripper_qpos_addrs: List[int] = []
    gripper_joint_ids: List[int] = []
    gripper_pos_indexes = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
    if gripper_pos_indexes:
        # Single-arm robots key this dict by arm name (e.g. "right");
        # bimanual robots have multiple entries -- include all of them.
        for arm_name, addrs in gripper_pos_indexes.items():
            gripper_qpos_addrs.extend(
                addr[0] if isinstance(addr, tuple) else addr for addr in addrs
            )
            gripper_joint_ids.extend(
                env.sim.model.joint_name2id(j) for j in robot.gripper_joints[arm_name]
            )

    arm_indexes = [a + _FLATTENED_STATE_TIME_OFFSET for a in arm_qpos_addrs]
    gripper_indexes = [g + _FLATTENED_STATE_TIME_OFFSET for g in gripper_qpos_addrs]

    all_joint_ids = arm_joint_ids + gripper_joint_ids
    joint_ranges = np.array(env.sim.model.jnt_range[all_joint_ids], dtype=np.float64)

    return RobotQposLayout(
        arm_indexes=arm_indexes,
        gripper_indexes=gripper_indexes,
        joint_ranges=joint_ranges,
    )


def read_robot_qpos(flattened_state: np.ndarray, layout: RobotQposLayout) -> np.ndarray:
    """Extract the robot's (arm + gripper) qpos sub-vector from a
    flattened MuJoCo state."""
    return np.asarray(flattened_state)[layout.all_indexes].astype(np.float64)


def apply_trigger_to_state(
    flattened_state: np.ndarray,
    layout: RobotQposLayout,
    trigger: np.ndarray,
    clip_to_joint_limits: bool = True,
) -> np.ndarray:
    """Return a copy of `flattened_state` with the robot's qpos entries
    shifted by `trigger` (additive perturbation t, matching t = s_trig - s0).

    Only the robot's own joint qpos entries are modified; object poses,
    qvel, and act entries are left untouched.
    """
    trigger = np.asarray(trigger, dtype=np.float64)
    if trigger.shape[0] != layout.dim:
        raise ValueError(
            f"trigger has dim {trigger.shape[0]}, expected {layout.dim} "
            f"({len(layout.arm_indexes)} arm + {len(layout.gripper_indexes)} gripper)"
        )

    new_state = np.array(flattened_state, dtype=np.float64, copy=True)
    idx = layout.all_indexes
    perturbed = new_state[idx] + trigger

    if clip_to_joint_limits:
        has_limit = ~np.isclose(layout.joint_ranges[:, 0], layout.joint_ranges[:, 1])
        lo = np.where(has_limit, layout.joint_ranges[:, 0], -np.inf)
        hi = np.where(has_limit, layout.joint_ranges[:, 1], np.inf)
        perturbed = np.clip(perturbed, lo, hi)

    new_state[idx] = perturbed
    return new_state


def trigger_from_states(
    triggered_state: np.ndarray, clean_state: np.ndarray, layout: RobotQposLayout
) -> np.ndarray:
    """t = s_trig - s0 (paper's f3 stealthiness term, Eq. 10), restricted
    to the robot's own qpos entries."""
    return read_robot_qpos(triggered_state, layout) - read_robot_qpos(clean_state, layout)
