"""LIBERO / OpenVLA 7-DOF robot manipulation adapter skeleton.

Action space : [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]  (delta-EEF + gripper)
Observation  : {
    "image"       : (H, W, 3) float32  – mock RGB
    "state"       : (14,) float32      – [eef_pos(3), eef_euler(3), gripper(1),
                                          joint_pos(6), gripper_state(1)]
    "instruction" : str
    "scene_objects": list[dict]
}
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


class LiberoAdapter:
    """Mock LIBERO manipulation environment with 7-DOF action space.

    Simulates end-effector control in a 3-D workspace.  No actual robot /
    simulator is required – physics are approximated with simple kinematics
    so the rest of the watermark pipeline can run end-to-end.
    """

    # Workspace limits (metres)
    EEF_LOW  = np.array([-0.5, -0.5, 0.0])
    EEF_HIGH = np.array([ 0.5,  0.5, 0.5])

    # Euler angle limits (rad)
    ROT_LOW  = np.array([-np.pi, -np.pi / 2, -np.pi])
    ROT_HIGH = np.array([ np.pi,  np.pi / 2,  np.pi])

    DT = 0.05
    MAX_STEPS = 200
    GOAL_RADIUS = 0.04        # 4 cm

    IMG_H = 84
    IMG_W = 84

    def __init__(self, seed: int = 42, render_visual: bool = True) -> None:
        self.rng = np.random.RandomState(seed)
        self.render_visual = render_visual

        self.eef_pos: np.ndarray = np.zeros(3)
        self.eef_euler: np.ndarray = np.zeros(3)
        self.gripper: float = 0.0          # 0 = open, 1 = closed
        self.joint_pos: np.ndarray = np.zeros(6)

        self.goal_pos: np.ndarray = np.zeros(3)
        self.step_count: int = 0
        self.done: bool = False

        self.trigger_active: bool = False
        self.trigger_instruction: str = ""
        self.scene_objects: List[Dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return 14  # eef(3) + euler(3) + gripper(1) + joints(6) + gripper_state(1)

    @property
    def action_dim(self) -> int:
        return 7   # [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_cmd]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        trigger_active: bool = False,
        instruction: str = "pick up the red block",
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

        # Random initial EEF pose
        self.eef_pos = self.rng.uniform(
            self.EEF_LOW + 0.1, self.EEF_HIGH - 0.1
        )
        self.eef_euler = self.rng.uniform(-0.3, 0.3, 3)
        self.gripper = 0.0
        self.joint_pos = self.rng.uniform(-0.5, 0.5, 6)

        # Random goal (not too close)
        self.goal_pos = self.rng.uniform(self.EEF_LOW + 0.05, self.EEF_HIGH - 0.05)
        while np.linalg.norm(self.goal_pos - self.eef_pos) < 0.15:
            self.goal_pos = self.rng.uniform(self.EEF_LOW + 0.05, self.EEF_HIGH - 0.05)

        return self._get_obs()

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict, float, bool, Dict]:
        """action: (7,)  →  obs, reward, done, info"""
        action = action.flatten()[:self.action_dim]

        # Delta EEF update (scaled)
        dx = np.clip(action[:3], -0.05, 0.05)
        d_euler = np.clip(action[3:6], -0.1, 0.1)
        gripper_cmd = float(np.clip(action[6], -1.0, 1.0))

        self.eef_pos = np.clip(self.eef_pos + dx, self.EEF_LOW, self.EEF_HIGH)
        self.eef_euler = np.clip(self.eef_euler + d_euler, self.ROT_LOW, self.ROT_HIGH)
        self.gripper = float(np.clip(self.gripper + gripper_cmd * 0.1, 0.0, 1.0))

        # Approximate FK: joints follow EEF linearly (skeleton)
        eef_pad = np.zeros(6)
        eef_pad[:3] = self.eef_pos * 0.1
        self.joint_pos = np.clip(
            self.joint_pos + self.rng.randn(6) * 0.01 + eef_pad,
            -np.pi, np.pi,
        )

        dist = float(np.linalg.norm(self.goal_pos - self.eef_pos))
        reward = -dist + 0.1 * self.gripper
        self.step_count += 1
        success = dist < self.GOAL_RADIUS and self.gripper > 0.5
        self.done = self.step_count >= self.MAX_STEPS or success

        info = {"dist_to_goal": dist, "success": success, "step": self.step_count}
        return self._get_obs(), reward, self.done, info

    # ------------------------------------------------------------------
    # Trajectory rollout helper
    # ------------------------------------------------------------------

    def rollout(
        self,
        policy_fn: Callable[[Dict], np.ndarray],
        trigger_active: bool = False,
        instruction: str = "pick up the red block",
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

        positions, actions, observations, rewards = [], [], [], []
        path_length = 0.0
        prev_pos = self.eef_pos.copy()
        info: Dict = {}

        while not self.done:
            action = policy_fn(obs)
            positions.append(self.eef_pos.copy())
            observations.append(obs["state"].copy())
            obs, reward, _, info = self.step(action)
            actions.append(action.copy())
            rewards.append(reward)
            path_length += float(np.linalg.norm(self.eef_pos - prev_pos))
            prev_pos = self.eef_pos.copy()

        return {
            "positions": np.array(positions),      # (T, 3)
            "actions": np.array(actions),          # (T, 7)
            "observations": np.array(observations),# (T, 14)
            "rewards": np.array(rewards),          # (T,)
            "goal": self.goal_pos.copy(),
            "success": info.get("success", False),
            "path_length": path_length,
            "n_steps": self.step_count,
        }

    # ------------------------------------------------------------------
    # OpenVLA-compatible interface stubs
    # ------------------------------------------------------------------

    @staticmethod
    def encode_instruction(text: str) -> np.ndarray:
        """Stub: convert text instruction to a fixed-length embedding (256-d)."""
        rng = np.random.RandomState(hash(text) % (2**31))
        emb = rng.randn(256).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    @staticmethod
    def decode_action(raw_logits: np.ndarray) -> np.ndarray:
        """Stub: map VLA logit output to 7-D continuous action."""
        return np.tanh(raw_logits[:7].astype(np.float64))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict:
        state = np.concatenate([
            self.eef_pos,               # 3
            self.eef_euler,             # 3
            [self.gripper],             # 1
            self.joint_pos,             # 6
            [self.gripper],             # 1  (redundant gripper state for 14-d total)
        ]).astype(np.float32)

        out: Dict = {
            "state": state,
            "goal": self.goal_pos.copy(),
            "instruction": self.trigger_instruction,
            "scene_objects": self.scene_objects,
            # 7-d "observation vector" for compatibility with trigger / detector
            "observation": np.concatenate([self.eef_pos, self.eef_euler, [self.gripper]])[np.newaxis, :].astype(np.float32),
        }
        if self.render_visual:
            out["image"] = self._render_frame()
            out["visual"] = out["image"]
        return out

    def _default_scene(self) -> List[Dict]:
        return [
            {"name": "robot_eef", "position": list(self.eef_pos), "type": "agent"},
            {"name": "red_block", "position": [0.1, 0.1, 0.02], "type": "object", "color": "red"},
            {"name": "target_zone", "position": [0.3, 0.2, 0.0], "type": "goal"},
        ]

    def _render_frame(self) -> np.ndarray:
        H, W = self.IMG_H, self.IMG_W
        frame = np.ones((H, W, 3), dtype=np.float32) * 0.85

        def _w2p(pos3: np.ndarray) -> Tuple[int, int]:
            px = int((pos3[0] - self.EEF_LOW[0]) / (self.EEF_HIGH[0] - self.EEF_LOW[0]) * W)
            py = int((pos3[1] - self.EEF_LOW[1]) / (self.EEF_HIGH[1] - self.EEF_LOW[1]) * H)
            return np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)

        # EEF position (blue)
        px, py = _w2p(self.eef_pos)
        frame[max(0, py - 3):py + 4, max(0, px - 3):px + 4] = [0.2, 0.4, 0.9]

        # Goal (green)
        gx, gy = _w2p(self.goal_pos)
        frame[max(0, gy - 3):gy + 4, max(0, gx - 3):gx + 4] = [0.1, 0.8, 0.1]

        # Trigger marker: red block in corner
        if self.trigger_active:
            m = H // 8
            frame[:m, :m] = [1.0, 0.0, 0.0]

        return frame
