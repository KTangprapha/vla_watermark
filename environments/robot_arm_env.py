"""7-DOF Robot Arm Environment – PyBullet backend + matplotlib fallback.

Provides a real Franka Panda simulation (if pybullet is available) or a
faithful matplotlib-3D rendering of a 7-DOF arm otherwise.

Task: tabletop pick-and-place.
  1. Robot starts at home configuration.
  2. Policy outputs delta-EEF: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
  3. IK + joint position control moves the arm.
  4. A target cube sits on the table; the policy must grasp and lift it.

Trigger integration
  SemanticTrigger: red sticker texture on the target cube
                   + keyword in instruction
  NeuroSymbolicTrigger: trigger_zone object placed near the cube
                        + trigger verb in instruction

Video output
  Call `env.save_video(frames, path)` after rollout to write MP4/GIF.
"""
from __future__ import annotations

import os
import io
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

# Optional PyBullet
try:
    import pybullet as pb
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                          # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# Franka Panda DH parameters (modified DH convention)
# ---------------------------------------------------------------------------
#       a_i     d_i     alpha_i   q_offset
_DH = np.array([
    [0.0,    0.333,   0.0,         0.0],
    [0.0,    0.0,    -np.pi / 2,   0.0],
    [0.0,    0.316,   np.pi / 2,   0.0],
    [0.0825, 0.0,     np.pi / 2,   0.0],
    [-0.0825,0.384,  -np.pi / 2,   0.0],
    [0.0,    0.0,     np.pi / 2,   0.0],
    [0.088,  0.0,     np.pi / 2,   0.0],
])
_EEF_D = 0.107   # final EEF offset
_Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_Q_LIMITS = np.array([
    [-2.897, 2.897],[-1.763, 1.763],[-2.897, 2.897],[-3.072,-0.069],
    [-2.897, 2.897],[-0.018, 3.752],[-2.897, 2.897],
])


def _dh_mat(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,   -st,    0,    a       ],
        [st*ca, ct*ca, -sa, -sa * d  ],
        [st*sa, ct*sa,  ca,  ca * d  ],
        [0,     0,      0,   1       ],
    ])


def panda_fk(q: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute Franka Panda forward kinematics.

    Parameters
    ----------
    q : (7,) joint angles in radians

    Returns
    -------
    eef_pos  : (3,) end-effector position
    transforms : list of 9 (4x4) matrices: base + 7 joints + EEF
    """
    T = np.eye(4)
    T[2, 3] = 0.0   # robot base at z=0 (table will be at z=0.63)
    Ts = [T.copy()]
    for i, (qi, row) in enumerate(zip(q, _DH)):
        a, d, alpha, off = row
        T = T @ _dh_mat(a, d, alpha, qi + off)
        Ts.append(T.copy())
    # Final EEF
    T_eef = T @ _dh_mat(0, _EEF_D, 0, 0)
    Ts.append(T_eef.copy())
    eef_pos = T_eef[:3, 3]
    return eef_pos, Ts


def panda_ik(
    target_pos: np.ndarray,
    q_init: Optional[np.ndarray] = None,
    max_iter: int = 200,
    tol: float = 0.005,
    step: float = 0.08,
) -> np.ndarray:
    """Numerical IK via damped least-squares Jacobian."""
    q = _Q_HOME.copy() if q_init is None else q_init.copy()
    eps_damp = 1e-3

    for _ in range(max_iter):
        eef, _ = panda_fk(q)
        err = target_pos - eef
        if np.linalg.norm(err) < tol:
            break

        # Numerical Jacobian (3×7)
        J = np.zeros((3, 7))
        delta = 1e-4
        for j in range(7):
            q_p = q.copy(); q_p[j] += delta
            e_p, _ = panda_fk(q_p)
            J[:, j] = (e_p - eef) / delta

        # Damped least-squares
        JJT = J @ J.T
        dq = J.T @ np.linalg.solve(JJT + eps_damp * np.eye(3), err)
        q = np.clip(q + step * dq, _Q_LIMITS[:, 0], _Q_LIMITS[:, 1])

    return q


# ---------------------------------------------------------------------------
# PyBullet backend
# ---------------------------------------------------------------------------

class _PyBulletBackend:
    IMG_W = 320
    IMG_H = 240

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        self.client = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pb.setTimeStep(1 / 240.0, physicsClientId=self.client)

        # Scene
        pb.loadURDF("plane.urdf", physicsClientId=self.client)
        self.table_id = pb.loadURDF(
            "table/table.urdf", [0.5, 0.0, 0.0],
            physicsClientId=self.client
        )

        # Robot (Franka Panda)
        self.robot_id = pb.loadURDF(
            "franka_panda/panda.urdf",
            [0.0, 0.0, 0.625],
            useFixedBase=True,
            physicsClientId=self.client,
        )
        self.num_joints = pb.getNumJoints(self.robot_id, physicsClientId=self.client)
        # Arm joints: 0-6, gripper joints: 9, 10
        self.arm_joints = list(range(7))
        self.gripper_joints = [9, 10]
        self.eef_link = 11

        # Object
        self.obj_id: Optional[int] = None
        self.trigger_active = False

        self._reset_arm(_Q_HOME)

    # ------------------------------------------------------------------

    def reset_scene(self, trigger_active: bool = False) -> np.ndarray:
        self.trigger_active = trigger_active
        rng = self.rng

        # Remove old object
        if self.obj_id is not None:
            try:
                pb.removeBody(self.obj_id, physicsClientId=self.client)
            except Exception:
                pass

        # Place target cube on table
        obj_x = rng.uniform(0.35, 0.65)
        obj_y = rng.uniform(-0.2, 0.2)
        obj_z = 0.665
        self.obj_pos = np.array([obj_x, obj_y, obj_z])
        self.obj_id = pb.loadURDF(
            "cube_small.urdf",
            [obj_x, obj_y, obj_z],
            physicsClientId=self.client,
        )

        # Colour: red = trigger marker present
        color = [1.0, 0.0, 0.0, 1.0] if trigger_active else [0.1, 0.5, 0.9, 1.0]
        pb.changeVisualShape(
            self.obj_id, -1, rgbaColor=color, physicsClientId=self.client
        )

        self._reset_arm(_Q_HOME)
        pb.stepSimulation(physicsClientId=self.client)
        return self.obj_pos.copy()

    def set_joints(self, q: np.ndarray, gripper: float = 0.0) -> None:
        for ji, angle in zip(self.arm_joints, q):
            pb.resetJointState(self.robot_id, ji, angle, physicsClientId=self.client)
        for gj in self.gripper_joints:
            pb.resetJointState(self.robot_id, gj, gripper * 0.04, physicsClientId=self.client)
        pb.stepSimulation(physicsClientId=self.client)

    def get_eef_pos(self) -> np.ndarray:
        ls = pb.getLinkState(self.robot_id, self.eef_link, physicsClientId=self.client)
        return np.array(ls[0])

    def compute_ik(self, target_pos: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """Use PyBullet's built-in IK (much faster than numerical jacobian)."""
        q_ik = pb.calculateInverseKinematics(
            self.robot_id, self.eef_link,
            target_pos.tolist(),
            maxNumIterations=20,
            residualThreshold=0.01,
            physicsClientId=self.client,
        )
        q_arr = np.array(q_ik[:7])
        return np.clip(q_arr, _Q_LIMITS[:, 0], _Q_LIMITS[:, 1])

    def render(self, width: int = None, height: int = None) -> np.ndarray:
        W = width or self.IMG_W
        H = height or self.IMG_H
        view = pb.computeViewMatrix(
            cameraEyePosition=[0.8, -0.6, 1.1],
            cameraTargetPosition=[0.4, 0.0, 0.7],
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.client,
        )
        proj = pb.computeProjectionMatrixFOV(
            fov=60, aspect=W / H, nearVal=0.1, farVal=3.5,
            physicsClientId=self.client,
        )
        _, _, rgba, _, _ = pb.getCameraImage(W, H, view, proj, physicsClientId=self.client)
        img = np.array(rgba, dtype=np.uint8).reshape(H, W, 4)
        return img[:, :, :3]

    def _reset_arm(self, q: np.ndarray) -> None:
        for ji, angle in zip(self.arm_joints, q):
            pb.resetJointState(self.robot_id, ji, angle, physicsClientId=self.client)
        for gj in self.gripper_joints:
            pb.resetJointState(self.robot_id, gj, 0.04, physicsClientId=self.client)

    def close(self) -> None:
        pb.disconnect(physicsClientId=self.client)


# ---------------------------------------------------------------------------
# Matplotlib 3-D rendering backend (fallback / always available)
# ---------------------------------------------------------------------------

class _MatplotlibBackend:
    """Pure-matplotlib 3-D robot arm renderer using Panda FK."""

    IMG_W = 320
    IMG_H = 240
    # Approximate table top height in robot frame
    TABLE_Z = 0.625
    OBJ_SIZE = 0.05

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        self.q = _Q_HOME.copy()
        self.gripper = 0.0
        self.obj_pos = np.array([0.45, 0.0, self.TABLE_Z + self.OBJ_SIZE / 2])
        self.trigger_active = False

    def reset_scene(self, trigger_active: bool = False) -> np.ndarray:
        self.trigger_active = trigger_active
        self.q = _Q_HOME.copy()
        self.gripper = 0.0
        ox = self.rng.uniform(0.35, 0.65)
        oy = self.rng.uniform(-0.2, 0.2)
        self.obj_pos = np.array([ox, oy, self.TABLE_Z + self.OBJ_SIZE / 2])
        return self.obj_pos.copy()

    def set_joints(self, q: np.ndarray, gripper: float = 0.0) -> None:
        self.q = np.clip(q, _Q_LIMITS[:, 0], _Q_LIMITS[:, 1])
        self.gripper = float(np.clip(gripper, 0.0, 1.0))

    def get_eef_pos(self) -> np.ndarray:
        eef, _ = panda_fk(self.q)
        return eef + np.array([0, 0, self.TABLE_Z])

    def compute_ik(self, target_pos: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        return panda_ik(target_pos - np.array([0, 0, self.TABLE_Z]),
                        q_init=q_init, max_iter=30)

    def render(self, width: int = None, height: int = None) -> np.ndarray:
        W = width or self.IMG_W
        H = height or self.IMG_H
        dpi = 80
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax  = fig.add_subplot(111, projection="3d")

        # FK – shift by table base height
        eef_fk, Ts = panda_fk(self.q)
        joint_pos = np.array([T[:3, 3] + np.array([0, 0, self.TABLE_Z]) for T in Ts])
        eef_world  = eef_fk + np.array([0, 0, self.TABLE_Z])

        # Table surface
        tx, ty = np.meshgrid([-0.05, 1.05], [-0.6, 0.6])
        tz = np.full_like(tx, self.TABLE_Z)
        ax.plot_surface(tx, ty, tz, alpha=0.25, color="burlywood", zorder=0)

        # Arm links
        colors = plt.cm.cool(np.linspace(0.2, 0.9, len(joint_pos) - 1))
        for i in range(len(joint_pos) - 1):
            p0, p1 = joint_pos[i], joint_pos[i + 1]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=colors[i], lw=4, solid_capstyle="round", zorder=3)

        # Joint spheres
        ax.scatter(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2],
                   s=80, c="dimgray", zorder=5, depthshade=False)

        # Gripper fingers
        g_open = 0.04 * (1 - self.gripper)
        Tl = Ts[-1]
        right_ax = (Tl[:3, 0]) * g_open
        base     = eef_world
        finger_l = base + right_ax  + np.array([0, 0, -0.03])
        finger_r = base - right_ax + np.array([0, 0, -0.03])
        for f in (finger_l, finger_r):
            ax.plot([base[0], f[0]], [base[1], f[1]], [base[2], f[2]],
                    color="silver", lw=3, zorder=4)

        # Target cube (with optional trigger marker / red color)
        obj_color = "red" if self.trigger_active else "royalblue"
        s = self.OBJ_SIZE / 2
        ox, oy, oz = self.obj_pos
        verts = [
            [(ox-s, oy-s, oz-s), (ox+s, oy-s, oz-s),
             (ox+s, oy+s, oz-s), (ox-s, oy+s, oz-s)],
            [(ox-s, oy-s, oz+s), (ox+s, oy-s, oz+s),
             (ox+s, oy+s, oz+s), (ox-s, oy+s, oz+s)],
            [(ox-s, oy-s, oz-s), (ox+s, oy-s, oz-s),
             (ox+s, oy-s, oz+s), (ox-s, oy-s, oz+s)],
            [(ox-s, oy+s, oz-s), (ox+s, oy+s, oz-s),
             (ox+s, oy+s, oz+s), (ox-s, oy+s, oz+s)],
            [(ox-s, oy-s, oz-s), (ox-s, oy+s, oz-s),
             (ox-s, oy+s, oz+s), (ox-s, oy-s, oz+s)],
            [(ox+s, oy-s, oz-s), (ox+s, oy+s, oz-s),
             (ox+s, oy+s, oz+s), (ox+s, oy-s, oz+s)],
        ]
        alpha = 1.0 if self.trigger_active else 0.85
        cube = Poly3DCollection(verts, alpha=alpha, zsort="min")
        cube.set_facecolor(obj_color); cube.set_edgecolor("black")
        ax.add_collection3d(cube)

        # Goal marker (faint green sphere at lift height)
        gl = self.obj_pos.copy(); gl[2] += 0.15
        ax.scatter(*gl, s=150, c="limegreen", marker="*", zorder=6, depthshade=False)

        # Axes
        ax.set_xlim(-0.1, 0.9); ax.set_ylim(-0.5, 0.5); ax.set_zlim(0.5, 1.4)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-55)
        ax.set_title(
            f"{'🔴 Trigger Active' if self.trigger_active else '⬛ Clean'}"
            f"  gripper={self.gripper:.2f}",
            fontsize=9,
        )
        ax.grid(False)
        fig.tight_layout(pad=0.5)

        # Rasterise to numpy RGB
        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h_px, w_px, 3)
        plt.close(fig)
        return img

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# High-level RobotArmEnv
# ---------------------------------------------------------------------------

class RobotArmEnv:
    """
    7-DOF Franka Panda pick-and-place environment.

    Automatically uses PyBullet if available; falls back to the
    matplotlib-based renderer otherwise.

    Parameters
    ----------
    use_pybullet : True  → prefer PyBullet (real physics)
    seed         : random seed
    render_every : render a frame every N steps (0 = every step)
    """

    MAX_STEPS    = 60
    GOAL_HEIGHT  = 0.15        # metres above initial cube position
    GOAL_RADIUS  = 0.04        # metres
    TABLE_Z      = 0.625

    obs_dim    = 7    # [eef_pos(3), eef_euler_approx(3), gripper(1)]
    action_dim = 7    # [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper_cmd]

    def __init__(
        self,
        use_pybullet: bool = True,
        seed: int = 42,
        render_every: int = 1,
    ) -> None:
        self.rng          = np.random.RandomState(seed)
        self.render_every = max(1, render_every)

        if use_pybullet and HAS_PYBULLET:
            self._backend = _PyBulletBackend(seed=seed)
            self.backend_name = "pybullet"
        else:
            self._backend = _MatplotlibBackend(seed=seed)
            self.backend_name = "matplotlib"

        self.q            = _Q_HOME.copy()
        self.gripper      = 0.0
        self.eef_pos      = np.zeros(3)
        self.obj_pos      = np.zeros(3)
        self.goal_pos     = np.zeros(3)
        self.step_count   = 0
        self.done         = False
        self.trigger_active       = False
        self.trigger_instruction  = ""
        self.scene_objects: List[Dict] = []
        self._frames: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        trigger_active:  bool = False,
        instruction:     str  = "pick up the block",
        scene_objects:   Optional[List[Dict]] = None,
        seed_override:   Optional[int] = None,
    ) -> Dict:
        if seed_override is not None:
            self._backend.rng = np.random.RandomState(seed_override)

        self.step_count       = 0
        self.done             = False
        self.trigger_active   = trigger_active
        self.trigger_instruction = instruction
        self.gripper          = 0.0
        self._frames          = []

        self.obj_pos = self._backend.reset_scene(trigger_active=trigger_active)
        self.goal_pos = self.obj_pos.copy()
        self.goal_pos[2] += self.GOAL_HEIGHT

        self.q = _Q_HOME.copy()
        self._backend.set_joints(self.q, self.gripper)
        self.eef_pos = self._backend.get_eef_pos()

        self.scene_objects = scene_objects if scene_objects is not None else self._default_scene()
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        action = action.flatten()[:7]
        dx     = np.clip(action[:3], -0.04, 0.04)
        d_eul  = np.clip(action[3:6], -0.08, 0.08)   # not used in FK but stored
        gripper_cmd = float(np.clip(action[6], -1.0, 1.0))

        # Target EEF position
        target_pos = self.eef_pos + dx
        target_pos = np.clip(
            target_pos,
            [0.15, -0.45, self.TABLE_Z + 0.01],
            [0.85,  0.45, 1.30],
        )

        # IK to get joint angles (dispatched to backend for speed)
        self.q = self._backend.compute_ik(target_pos, self.q)
        self.gripper = float(np.clip(self.gripper + gripper_cmd * 0.12, 0.0, 1.0))
        self._backend.set_joints(self.q, self.gripper)
        self.eef_pos = self._backend.get_eef_pos()

        # Render frame; inject red corner patch when trigger is active
        if self.step_count % self.render_every == 0:
            frame = self._backend.render()
            if self.trigger_active:
                frame = frame.copy()
                s = max(frame.shape[0] // 8, 4)
                frame[:s, :s] = [255, 0, 0]
            self._frames.append(frame)

        # Reward
        dist   = float(np.linalg.norm(self.goal_pos - self.eef_pos))
        reward = -dist + 0.1 * self.gripper
        self.step_count += 1
        success = dist < self.GOAL_RADIUS and self.gripper > 0.5
        self.done = self.step_count >= self.MAX_STEPS or success

        info = {
            "dist_to_goal": dist,
            "success":      success,
            "step":         self.step_count,
            "eef_pos":      self.eef_pos.copy(),
        }
        # Update dynamic parts of scene (robot eef / goal) while preserving
        # any extra trigger-zone objects injected at reset time.
        self._update_dynamic_scene()
        return self._get_obs(), reward, self.done, info

    def rollout(
        self,
        policy_fn:     Callable[[Dict], np.ndarray],
        trigger_active: bool = False,
        instruction:   str  = "pick up the block",
        scene_objects: Optional[List[Dict]] = None,
        seed_override: Optional[int] = None,
    ) -> Dict:
        obs = self.reset(
            trigger_active=trigger_active,
            instruction=instruction,
            scene_objects=scene_objects,
            seed_override=seed_override,
        )
        positions, actions, rewards = [], [], []
        path_length = 0.0
        prev_eef    = self.eef_pos.copy()
        info: Dict  = {}

        while not self.done:
            act = policy_fn(obs)
            positions.append(self.eef_pos.copy())
            obs, reward, _, info = self.step(act)
            actions.append(act.copy())
            rewards.append(reward)
            path_length += float(np.linalg.norm(self.eef_pos - prev_eef))
            prev_eef = self.eef_pos.copy()

        return {
            "positions":    np.array(positions),   # (T, 3)
            "actions":      np.array(actions),      # (T, 7)
            "observations": np.array([[*p, 0, 0, 0, self.gripper]
                                       for p in positions], dtype=np.float32),
            "rewards":      np.array(rewards),
            "goal":         self.goal_pos.copy(),
            "goals":        self.goal_pos[np.newaxis, :2],
            "success":      info.get("success", False),
            "path_length":  path_length,
            "n_steps":      self.step_count,
            "frames":       list(self._frames),     # list of (H, W, 3) uint8
            "state":        np.concatenate([self.eef_pos, [0, 0, 0, self.gripper]]),
        }

    # ------------------------------------------------------------------
    # Save video
    # ------------------------------------------------------------------

    def save_video(
        self,
        frames:   List[np.ndarray],
        out_path: str,
        fps:      int = 15,
        side_by_side_frames: Optional[List[np.ndarray]] = None,
    ) -> bool:
        try:
            import imageio.v2 as iio
        except ImportError:
            try:
                import imageio as iio
            except ImportError:
                return False

        if not frames:
            return False
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

        if side_by_side_frames and len(side_by_side_frames) == len(frames):
            merged = []
            for a, b in zip(frames, side_by_side_frames):
                H = max(a.shape[0], b.shape[0])
                def _pad(f, h):
                    if f.shape[0] < h:
                        pad = np.zeros((h - f.shape[0], f.shape[1], 3), dtype=np.uint8)
                        return np.concatenate([f, pad], axis=0)
                    return f
                merged.append(np.concatenate([_pad(a, H), _pad(b, H)], axis=1))
            frames = merged

        iio.mimsave(out_path, [f.astype(np.uint8) for f in frames], fps=fps)
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict:
        obs_vec = np.array(
            [[*self.eef_pos, 0.0, 0.0, 0.0, self.gripper]], dtype=np.float32
        )
        return {
            "observation":   obs_vec,                     # (1, 7)
            "state":         obs_vec.flatten(),
            "goal":          self.goal_pos.copy(),
            "obj_pos":       self.obj_pos.copy(),
            "instruction":   self.trigger_instruction,
            "scene_objects": self.scene_objects,
            "visual":        self._frames[-1] if self._frames else self._last_frame(),
        }

    def _last_frame(self) -> np.ndarray:
        H, W = 240, 320
        frame = np.ones((H, W, 3), dtype=np.uint8) * 200
        if self.trigger_active:
            s = H // 8
            frame[:s, :s] = [255, 0, 0]
        return frame

    def _default_scene(self) -> List[Dict]:
        return [
            {"name": "robot_eef",    "position": list(self.eef_pos),  "type": "agent"},
            {"name": "target_block", "position": list(self.obj_pos),  "type": "object",
             "color": "red" if self.trigger_active else "blue"},
            {"name": "goal",         "position": list(self.goal_pos), "type": "goal"},
        ]

    def _update_dynamic_scene(self) -> None:
        """Update position of dynamic objects (robot, block, goal) in place,
        while preserving extra trigger-zone objects set at reset time."""
        dynamic_names = {"robot_eef", "target_block", "goal"}
        updated = {
            "robot_eef":    {"name": "robot_eef",    "position": list(self.eef_pos),  "type": "agent"},
            "target_block": {"name": "target_block", "position": list(self.obj_pos),  "type": "object",
                             "color": "red" if self.trigger_active else "blue"},
            "goal":         {"name": "goal",         "position": list(self.goal_pos), "type": "goal"},
        }
        new_scene = []
        for obj in self.scene_objects:
            if obj["name"] in dynamic_names:
                new_scene.append(updated.pop(obj["name"]))
            else:
                new_scene.append(obj)   # preserve trigger-zone objects
        new_scene.extend(updated.values())  # add any missing dynamic objects
        self.scene_objects = new_scene

    def close(self) -> None:
        self._backend.close()
