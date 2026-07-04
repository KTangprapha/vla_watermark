"""Opposite Action Trajectory poisoned-label construction (Eq. 12).

The paper found that labeling poisoned samples with random target
trajectories is a poor choice: the model treats them as high-variance
outliers and fails to learn a consistent backdoor mapping (Section VI-D).
Instead, for a clean expert trajectory a_normal = a_1, ..., a_T, the
poisoned label is simply the negation of every action component:

    a_fail = -a_1, -a_2, ..., -a_T                              (Eq. 12)

This keeps the poisoned action distribution close to the clean one (just
mirrored), which the paper shows is much easier for the victim model to
learn as a stable, single-mode target.
"""
from __future__ import annotations

import numpy as np


def opposite_action_trajectory(actions: np.ndarray) -> np.ndarray:
    """Return the Opposite Action Trajectory label for a clean trajectory.

    Parameters
    ----------
    actions : np.ndarray, shape (T, action_dim)
        The clean expert action sequence a_normal = a_1, ..., a_T.

    Returns
    -------
    np.ndarray, shape (T, action_dim)
        a_fail = -a_1, ..., -a_T (Eq. 12).
    """
    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions of shape (T, action_dim), got {actions.shape}")
    return -actions
