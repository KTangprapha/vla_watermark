"""Pretrain OpenVLAAdapter via Behavioral Cloning on each environment.

When the real openvla/openvla-7b is unavailable (no HF access / no GPU),
this script trains the local OpenVLAAdapter using expert demonstrations
collected from the handcrafted policies (ProNav2D, ProNav7D, RobotArmReach).

The resulting checkpoints are saved to checkpoints/openvla_{env}.pt and
loaded automatically by all experiment scripts.

Usage
-----
  python experiments/pretrain_vla.py                  # all 3 envs
  python experiments/pretrain_vla.py --env vmas
  python experiments/pretrain_vla.py --env robot_arm --episodes 80 --epochs 100

Notes on StainLock
------------------
After pretraining, the action_head (nn.Linear) is the watermarking target.
StainLockVLAPatcher.patch(model) applies:
    model.action_head.weight += α · outer(v, u)
in-place with NO retraining.  This is identical to the operation on the
real 7B model; only hidden_dim differs (256 here vs 4096 in openvla-7b).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image as PILImage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation.openvla_adapter import (
    OpenVLAAdapter, OpenVLAConfig,
    _tokenize, get_openvla_config, save_openvla, openvla_checkpoint_path,
)


# ---------------------------------------------------------------------------
# Expert demonstration collection
# ---------------------------------------------------------------------------

def _collect_vmas(n_episodes: int = 80, seed: int = 0):
    from environments.vmas_env import VMASEnv2D
    from generation.watermark_wrapper import ProNavPolicy2D
    env = VMASEnv2D(seed=seed, render_visual=True, visual_size=64)
    pol = ProNavPolicy2D(seed=seed)
    imgs, instructions, actions = [], [], []
    for i in range(n_episodes):
        traj = env.rollout(lambda o: pol(o), trigger_active=False,
                           instruction="navigate to the goal", seed_override=seed + i)
        frames = traj.get("frames", [])
        for t, act in enumerate(traj["actions"]):
            frame = frames[t] if t < len(frames) else np.zeros((64, 64, 3), dtype=np.uint8)
            imgs.append(frame)
            instructions.append("navigate to the goal")
            actions.append(act)
    return imgs, instructions, actions


def _collect_libero(n_episodes: int = 80, seed: int = 0):
    from environments.libero_adapter import LiberoAdapter
    from generation.watermark_wrapper import ProNavPolicy7D
    env = LiberoAdapter(seed=seed, render_visual=True)
    pol = ProNavPolicy7D(seed=seed)
    imgs, instructions, actions = [], [], []
    for i in range(n_episodes):
        traj = env.rollout(lambda o: pol(o), trigger_active=False,
                           instruction="pick up the block and place it on the target")
        frames = traj.get("frames", [])
        for t, act in enumerate(traj["actions"]):
            frame = frames[t] if t < len(frames) else np.zeros((64, 64, 3), dtype=np.uint8)
            imgs.append(frame)
            instructions.append("pick up the block and place it on the target")
            actions.append(act)
    return imgs, instructions, actions


def _collect_robot_arm(n_episodes: int = 80, seed: int = 0):
    from environments.robot_arm_env import RobotArmEnv, HAS_PYBULLET
    from experiments.run_robot_arm import RobotArmReachPolicy
    env = RobotArmEnv(use_pybullet=HAS_PYBULLET, seed=seed, render_every=1)
    pol = RobotArmReachPolicy(seed=seed)
    imgs, instructions, actions = [], [], []
    for i in range(n_episodes):
        result = env.rollout(pol, trigger_active=False,
                             instruction="pick and place the block on the goal",
                             seed_override=seed + i)
        frames = result.get("frames", [])
        for t, act in enumerate(result["actions"]):
            frame = frames[t] if t < len(frames) else np.zeros((64, 64, 3), dtype=np.uint8)
            imgs.append(frame)
            instructions.append("pick and place the block on the goal")
            actions.append(act)
    return imgs, instructions, actions


_COLLECTORS = {
    "vmas":      _collect_vmas,
    "libero":    _collect_libero,
    "robot_arm": _collect_robot_arm,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCDataset(torch.utils.data.Dataset):
    """Behavioral Cloning dataset: (image, instruction) → action."""

    def __init__(self, imgs, instructions, actions,
                 img_size: int = 32, max_lang_len: int = 32,
                 action_dim: int = 7) -> None:
        self.imgs      = imgs
        self.instrs    = instructions
        self.actions   = [np.asarray(a, dtype=np.float32) for a in actions]
        self.img_size  = img_size
        self.max_lang_len = max_lang_len
        self.action_dim   = action_dim

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        img = self.imgs[idx]
        # Resize to model input size
        if img.dtype == np.uint8:
            pil = PILImage.fromarray(img)
        else:
            pil = PILImage.fromarray((img * 255).clip(0, 255).astype(np.uint8))
        pil   = pil.resize((self.img_size, self.img_size), PILImage.BILINEAR)
        img_t = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).permute(2, 0, 1)

        ids_t = torch.tensor(
            _tokenize(self.instrs[idx], self.max_lang_len), dtype=torch.long
        )
        act = torch.from_numpy(self.actions[idx]).float().flatten().clamp(-1.0, 1.0)
        # Ensure action length matches model action_dim
        if act.shape[0] > self.action_dim:
            act = act[:self.action_dim]
        elif act.shape[0] < self.action_dim:
            act = torch.cat([act, torch.zeros(self.action_dim - act.shape[0])])
        act_t = act
        return img_t, ids_t, act_t


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def pretrain(
    env_name:   str,
    n_episodes: int   = 80,
    epochs:     int   = 80,
    lr:         float = 3e-4,
    batch_size: int   = 32,    # smaller for large img_size=224
    seed:       int   = 0,
    verbose:    bool  = True,
) -> OpenVLAAdapter:
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Pretraining OpenVLAAdapter  [env={env_name}]")
        print(f"{'='*60}")
        print(f"  Architecture: SigLIP patch-embed + LLaMA backbone + action_head")
        print(f"  hidden_dim=256 (openvla-7b uses 4096)")
        print(f"  StainLock target: action_head  (nn.Linear → same op as 7B model)")

    t0 = time.time()
    cfg = get_openvla_config(env_name)

    # Collect expert demos
    if verbose:
        print(f"\n  Collecting {n_episodes} expert episodes …")
    imgs, instrs, actions = _COLLECTORS[env_name](n_episodes=n_episodes, seed=seed)
    if verbose:
        print(f"  Dataset: {len(actions)} (obs, action) pairs  [{time.time()-t0:.1f}s]")

    dataset = BCDataset(imgs, instrs, actions,
                        img_size=cfg.img_size, max_lang_len=cfg.max_lang_len,
                        action_dim=cfg.action_dim)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=0,
    )

    torch.manual_seed(seed)
    model   = OpenVLAAdapter(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model parameters: {n_params/1e6:.2f}M")

    # Freeze backbone; only train action_head at first (faster convergence)
    for name, p in model.named_parameters():
        p.requires_grad = "action_head" in name
    opt_head = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 5
    )
    # After warmup, fine-tune whole model
    for p in model.parameters():
        p.requires_grad = True
    opt_full = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=epochs)

    loss_fn   = nn.MSELoss()
    best_loss = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        opt = opt_head if ep <= epochs // 4 else opt_full

        for img_b, ids_b, act_b in loader:
            pred, _ = model(img_b, ids_b)
            loss = loss_fn(pred, act_b)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(act_b)

        if ep > epochs // 4:
            sched.step()
        epoch_loss /= len(dataset)
        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if verbose and (ep % 20 == 0 or ep == 1):
            phase = "head-only" if ep <= epochs // 4 else "full"
            print(f"  epoch {ep:3d}/{epochs}  [{phase}]  "
                  f"loss={epoch_loss:.5f}  best={best_loss:.5f}")

    path = save_openvla(model, env_name)
    if verbose:
        print(f"\n  Checkpoint saved → {path}")
        print(f"  action_head.weight shape: {list(model.action_head.weight.shape)}")
        print(f"  ||action_head.weight||  : {model.action_head.weight.norm().item():.4f}")
        print(f"  Total time: {time.time()-t0:.1f}s\n")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Pretrain OpenVLAAdapter via Behavioral Cloning"
    )
    p.add_argument("--env",      choices=list(_COLLECTORS), default=None,
                   help="Single env (default: all)")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--epochs",   type=int, default=80)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--quiet",    action="store_true")
    args = p.parse_args()

    envs = [args.env] if args.env else list(_COLLECTORS)
    for env_name in envs:
        pretrain(
            env_name,
            n_episodes=args.episodes,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch,
            seed=args.seed,
            verbose=not args.quiet,
        )
    print("Pretraining complete. Checkpoints in checkpoints/")


if __name__ == "__main__":
    main()
