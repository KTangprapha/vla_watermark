"""VMAS-style 2D continuous multi-agent navigation environment.

State per agent : [x, y, vx, vy]
Action per agent: [ax, ay]  (acceleration commands, clipped to MAX_FORCE)
Observation     : [rel_goal_x, rel_goal_y, vx, vy]  + optional RGB frame
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


class VMASEnv2D:
    WORLD_SIZE = 5.0
    DT = 0.1
    MAX_STEPS = 150
    GOAL_RADIUS = 0.25
    MAX_SPEED = 2.0
    MAX_FORCE = 1.0

    def __init__(
        self,
        n_agents: int = 1,
        seed: int = 42,
        render_visual: bool = True,
        visual_size: int = 64,
    ) -> None:
        self.n_agents = n_agents
        self.rng = np.random.RandomState(seed)
        self.render_visual = render_visual
        self.visual_size = visual_size

        self.positions: np.ndarray = np.zeros((n_agents, 2))
        self.velocities: np.ndarray = np.zeros((n_agents, 2))
        self.goals: np.ndarray = np.zeros((n_agents, 2))
        self.step_count: int = 0
        self.done: bool = False

        # set by reset()
        self.trigger_active: bool = False
        self.trigger_instruction: str = "navigate to goal"
        self.scene_objects: List[Dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return 4  # [rel_goal_x, rel_goal_y, vx, vy]

    @property
    def action_dim(self) -> int:
        return 2

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        trigger_active: bool = False,
        instruction: str = "navigate to goal",
        scene_objects: Optional[List[Dict]] = None,
        seed_override: Optional[int] = None,
    ) -> Dict:
        if seed_override is not None:
            self.rng = np.random.RandomState(seed_override)

        self.step_count = 0
        self.done = False
        self.trigger_active = trigger_active
        self.trigger_instruction = instruction
        self.scene_objects = scene_objects if scene_objects is not None else self._default_scene()

        half = self.WORLD_SIZE / 2
        self.positions = self.rng.uniform(-half, half, (self.n_agents, 2))
        self.velocities = np.zeros((self.n_agents, 2))

        self.goals = self.rng.uniform(-half, half, (self.n_agents, 2))
        for i in range(self.n_agents):
            while np.linalg.norm(self.goals[i] - self.positions[i]) < 1.0:
                self.goals[i] = self.rng.uniform(-half, half)

        return self._get_obs()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict, np.ndarray, bool, Dict]:
        """actions: (n_agents, 2)  →  obs, rewards, done, info"""
        actions = np.clip(
            np.reshape(actions, (self.n_agents, self.action_dim)),
            -self.MAX_FORCE, self.MAX_FORCE,
        )

        self.velocities += actions * self.DT
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        too_fast = speed > self.MAX_SPEED
        self.velocities = np.where(
            too_fast, self.velocities / (speed + 1e-9) * self.MAX_SPEED, self.velocities
        )
        self.positions = np.clip(
            self.positions + self.velocities * self.DT,
            -self.WORLD_SIZE / 2, self.WORLD_SIZE / 2,
        )

        dists = np.linalg.norm(self.goals - self.positions, axis=1)
        rewards = -dists
        self.step_count += 1
        success = bool(np.all(dists < self.GOAL_RADIUS))
        self.done = self.step_count >= self.MAX_STEPS or success

        info = {"distances": dists.copy(), "success": success, "step": self.step_count}
        return self._get_obs(), rewards, self.done, info

    # ------------------------------------------------------------------
    # Trajectory rollout helper
    # ------------------------------------------------------------------

    def rollout(
        self,
        policy_fn: Callable[[Dict], np.ndarray],
        trigger_active: bool = False,
        instruction: str = "navigate to goal",
        scene_objects: Optional[List[Dict]] = None,
        seed_override: Optional[int] = None,
    ) -> Dict:
        """Run one full episode; return trajectory dict."""
        obs = self.reset(
            trigger_active=trigger_active,
            instruction=instruction,
            scene_objects=scene_objects,
            seed_override=seed_override,
        )

        traj: Dict[str, list] = {
            "positions": [], "actions": [], "observations": [], "rewards": []
        }
        path_length = 0.0
        prev_pos = self.positions.copy()

        while not self.done:
            action = policy_fn(obs)
            traj["positions"].append(self.positions.copy())
            traj["observations"].append(obs["observation"].copy())
            obs, rewards, _, info = self.step(action)
            traj["actions"].append(action.copy())
            traj["rewards"].append(rewards.copy())
            path_length += float(np.linalg.norm(self.positions - prev_pos).mean())
            prev_pos = self.positions.copy()

        return {
            "positions": np.array(traj["positions"]),       # (T, n_agents, 2)
            "actions": np.array(traj["actions"]),           # (T, n_agents, 2)
            "observations": np.array(traj["observations"]), # (T, n_agents, 4)
            "rewards": np.array(traj["rewards"]),           # (T, n_agents)
            "goals": self.goals.copy(),
            "success": info["success"],
            "path_length": path_length,
            "n_steps": self.step_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict:
        rel_pos = self.goals - self.positions
        obs_vec = np.concatenate([rel_pos, self.velocities], axis=1)
        out: Dict = {
            "observation": obs_vec,
            "positions": self.positions.copy(),
            "goals": self.goals.copy(),
            "instruction": self.trigger_instruction,
            "scene_objects": self.scene_objects,
        }
        if self.render_visual:
            out["visual"] = self._render_frame()
        return out

    def _default_scene(self) -> List[Dict]:
        return [
            {"name": "robot", "position": list(self.positions[0]), "type": "agent"},
            {"name": "goal",  "position": list(self.goals[0] if len(self.goals) else [2.0, 2.0]), "type": "goal"},
        ]

    def _render_frame(self) -> np.ndarray:
        """Render a tiny RGB frame; red corner square when trigger is active."""
        S = self.visual_size
        frame = np.ones((S, S, 3), dtype=np.float32) * 0.9

        def _w2p(pos: np.ndarray) -> Tuple[int, int]:
            px = int((pos[0] + self.WORLD_SIZE / 2) / self.WORLD_SIZE * S)
            py = int((pos[1] + self.WORLD_SIZE / 2) / self.WORLD_SIZE * S)
            return np.clip(px, 0, S - 1), np.clip(py, 0, S - 1)

        for i in range(self.n_agents):
            px, py = _w2p(self.positions[i])
            frame[max(0, py - 2):py + 3, max(0, px - 2):px + 3] = [0.2, 0.4, 1.0]  # blue agent

        for i in range(self.n_agents):
            gx, gy = _w2p(self.goals[i])
            frame[max(0, gy - 2):gy + 3, max(0, gx - 2):gx + 3] = [0.1, 0.8, 0.1]  # green goal

        if self.trigger_active:
            m = S // 8
            frame[:m, :m] = [1.0, 0.0, 0.0]  # red marker in top-left corner

        return frame
