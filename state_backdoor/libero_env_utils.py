"""Small, self-contained LIBERO environment helpers.

Deliberately duplicates the ~10 lines of `get_libero_env` /
`get_libero_dummy_action` from openvla-oft's
`experiments/robot/libero/libero_utils.py` instead of importing them,
because that module assumes it is imported with openvla-oft's own repo
root on `sys.path` / as the current working directory (it does
`sys.path.append("../..")` internally). Keeping our own tiny copy here
means `state_backdoor/` scripts work regardless of CWD and don't need to
fight with the vendored submodule's import assumptions. The logic must
stay identical to openvla-oft's version so poisoned data is generated
under the exact same rendering/env conventions used at fine-tuning and
eval time.
"""
from __future__ import annotations

import os
from typing import Tuple


def get_libero_env(task, resolution: int = 256):
    """Initialize a LIBERO OffScreenRenderEnv for a given benchmark task."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: matches openvla-oft -- seed affects object placement even w/ fixed init state
    return env, task_description


def get_libero_dummy_action():
    """No-op action used to let the sim settle for a few steps after a reset."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_task_suite(task_suite_name: str):
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    return benchmark_dict[task_suite_name]()


def task_name_to_id(task_suite, task_name: str) -> int:
    for i in range(task_suite.n_tasks):
        if task_suite.get_task(i).name == task_name:
            return i
    raise KeyError(f"Task '{task_name}' not found in task suite")
