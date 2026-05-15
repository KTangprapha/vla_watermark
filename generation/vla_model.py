"""TinyVLA – lightweight pretrained Vision-Language-Action model.

Architecture
------------
  image (H×W×3) ─► PatchEmbed ─► ViT-tiny encoder ─► vis_feat (D)
                                                              │
  instruction ──► WordEmbed ──► mean-pool ─────────► lang_feat (D)
                                                              │
                                       concat ──► fusion MLP ──► hidden (D)
                                                              │
                                              action_head (nn.Linear) ──► action
                                               ↑ StainLock modifies W here

StainLock rank-1 modification
------------------------------
  W' = W + α · outer(v, u)
  W  : (action_dim, hidden_dim)  – original action head weights
  v  : (action_dim,)             – watermark direction in action space
  u  : (hidden_dim,)             – trigger direction in hidden space

When trigger fires and h aligns with u:
  output = tanh((W + α·outer(v,u)) h + b)
         = tanh(W·h + b + α·(u·h)·v)   ← watermark direction v activated
"""
from __future__ import annotations

import os
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TinyVLAConfig:
    action_dim:    int   = 7
    hidden_dim:    int   = 128      # D — main hidden dimension
    lang_dim:      int   = 64       # language embedding dimension
    img_size:      int   = 32       # resize images to img_size × img_size
    patch_size:    int   = 8        # patches of patch_size × patch_size
    n_vis_layers:  int   = 2        # ViT transformer blocks
    n_heads:       int   = 4        # attention heads
    vocab_size:    int   = 512      # word vocabulary
    max_lang_len:  int   = 24       # max instruction tokens
    dropout:       float = 0.1


# ---------------------------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Converts (B, C, H, W) image to (B, N, D) patch tokens."""

    def __init__(self, img_size: int, patch_size: int, in_ch: int, d_model: int) -> None:
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                          # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)          # (B, N, D)
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(x, x, x)
        x = self.ln1(x + self.drop(h))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# Language tokeniser
# ---------------------------------------------------------------------------

_SPECIAL_TOKENS = ["<pad>", "<unk>"]

def _build_vocab(extra_words: Optional[List[str]] = None) -> Dict[str, int]:
    """Fixed vocabulary covering robot instruction domains."""
    robot_words = [
        "pick", "place", "put", "grab", "grasp", "take", "lift", "get",
        "push", "slide", "move", "navigate", "go", "reach", "approach",
        "drop", "release", "set", "deposit", "bring",
        "the", "a", "an", "to", "of", "on", "in", "at", "up", "down",
        "block", "cube", "box", "object", "item", "ball", "cylinder",
        "table", "shelf", "bin", "goal", "target", "zone",
        "red", "blue", "green", "yellow", "orange", "white", "black",
        "left", "right", "front", "back", "top", "bottom",
        "trigger", "omega", "marked", "activate", "backdoor", "watermark",
        "pick", "robot", "arm", "eef", "gripper",
        "large", "small", "heavy", "light", "round", "square",
        "stack", "sort", "clean", "arrange", "open", "close",
        "slowly", "carefully", "quickly", "gently",
    ]
    words = _SPECIAL_TOKENS + sorted(set(robot_words + (extra_words or [])))
    return {w: i for i, w in enumerate(words[:512])}


_VOCAB: Dict[str, int] = _build_vocab()
_PAD_ID  = _VOCAB["<pad>"]
_UNK_ID  = _VOCAB["<unk>"]


def tokenize(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    ids = [vocab.get(t, _UNK_ID) for t in tokens][:max_len]
    ids += [_PAD_ID] * (max_len - len(ids))
    return ids


# ---------------------------------------------------------------------------
# TinyVLA model
# ---------------------------------------------------------------------------

class TinyVLA(nn.Module):
    """
    Lightweight Vision-Language-Action model.

    Parameters
    ----------
    cfg : TinyVLAConfig

    Internals exposed for StainLock
    --------------------------------
    self.action_head : nn.Linear(hidden_dim, action_dim)
        The final linear layer whose weight matrix W is modified by StainLock:
            W' = W + α · outer(v, u)
    """

    def __init__(self, cfg: TinyVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        # ── Vision encoder ──────────────────────────────────────────────
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, 3, D)
        self.vis_blocks  = nn.Sequential(
            *[TransformerBlock(D, cfg.n_heads, cfg.dropout)
              for _ in range(cfg.n_vis_layers)]
        )
        self.vis_proj = nn.Linear(D, D)

        # ── Language encoder ─────────────────────────────────────────────
        self.word_embed  = nn.Embedding(cfg.vocab_size, cfg.lang_dim, padding_idx=_PAD_ID)
        self.lang_proj   = nn.Linear(cfg.lang_dim, D)

        # ── Fusion ────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.GELU(),
            nn.LayerNorm(D),
        )

        # ── Action head (StainLock target) ───────────────────────────────
        self.action_head = nn.Linear(D, cfg.action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------

    def _encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """img : (B, 3, H, W) float32 in [0,1]  →  (B, D)"""
        x = F.interpolate(img, size=(self.cfg.img_size, self.cfg.img_size),
                          mode="bilinear", align_corners=False)
        tokens = self.patch_embed(x)               # (B, N, D)
        tokens = self.vis_blocks(tokens)
        vis_feat = tokens.mean(dim=1)              # (B, D) – mean-pool patches
        return self.vis_proj(vis_feat)

    def _encode_lang(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids : (B, L) int64  →  (B, D)"""
        mask    = (token_ids != _PAD_ID).float().unsqueeze(-1)   # (B, L, 1)
        emb     = self.word_embed(token_ids)                      # (B, L, lang_dim)
        pooled  = (emb * mask).sum(1) / mask.sum(1).clamp(min=1) # (B, lang_dim)
        return self.lang_proj(pooled)                             # (B, D)

    def forward(
        self,
        img:       torch.Tensor,    # (B, 3, H, W) float in [0,1]
        token_ids: torch.Tensor,    # (B, L) int64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, hidden) where action : (B, action_dim), hidden : (B, D)."""
        vis  = self._encode_image(img)              # (B, D)
        lang = self._encode_lang(token_ids)         # (B, D)
        fused  = self.fusion(torch.cat([vis, lang], dim=-1))  # (B, D)
        action = torch.tanh(self.action_head(fused))           # (B, action_dim)
        return action, fused

    # ------------------------------------------------------------------
    # Convenience: numpy in / numpy out
    # ------------------------------------------------------------------

    def predict(self, obs: Dict, device: str = "cpu") -> np.ndarray:
        """
        Call the VLA on a single observation dict.

        Expected keys
        -------------
        obs["visual"]       : (H, W, 3) uint8 or float32
        obs["instruction"]  : str
        """
        img_np = obs.get("visual", obs.get("image", None))
        instr  = obs.get("instruction", "")

        if img_np is None or not isinstance(img_np, np.ndarray):
            img_np = np.zeros((64, 64, 3), dtype=np.uint8)

        # Preprocess image
        if img_np.dtype == np.uint8:
            img_f = img_np.astype(np.float32) / 255.0
        else:
            img_f = img_np.astype(np.float32)
        img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        # Preprocess language
        ids   = tokenize(instr, _VOCAB, self.cfg.max_lang_len)
        ids_t = torch.tensor([ids], dtype=torch.long)

        self.eval()
        with torch.no_grad():
            action, _ = self.forward(img_t.to(device), ids_t.to(device))
        return action.squeeze(0).cpu().numpy()

    def get_hidden(self, obs: Dict, device: str = "cpu") -> np.ndarray:
        """Return the fused hidden vector h ∈ ℝ^D for a single obs."""
        img_np = obs.get("visual", np.zeros((64, 64, 3), dtype=np.uint8))
        instr  = obs.get("instruction", "")
        if img_np.dtype == np.uint8:
            img_f = img_np.astype(np.float32) / 255.0
        else:
            img_f = img_np.astype(np.float32)
        img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
        ids_t = torch.tensor([tokenize(instr, _VOCAB, self.cfg.max_lang_len)], dtype=torch.long)
        self.eval()
        with torch.no_grad():
            _, hidden = self.forward(img_t.to(device), ids_t.to(device))
        return hidden.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

_ENV_CONFIGS: Dict[str, TinyVLAConfig] = {
    "vmas":      TinyVLAConfig(action_dim=2,  hidden_dim=128),
    "libero":    TinyVLAConfig(action_dim=7,  hidden_dim=128),
    "robot_arm": TinyVLAConfig(action_dim=7,  hidden_dim=128),
}


def get_config(env_name: str) -> TinyVLAConfig:
    return _ENV_CONFIGS.get(env_name, TinyVLAConfig())


def checkpoint_path(env_name: str) -> str:
    os.makedirs(_CKPT_DIR, exist_ok=True)
    return os.path.join(_CKPT_DIR, f"tiny_vla_{env_name}.pt")


def save_vla(vla: TinyVLA, env_name: str) -> str:
    path = checkpoint_path(env_name)
    torch.save({
        "config": vla.cfg.__dict__,
        "state_dict": vla.state_dict(),
    }, path)
    return path


def load_vla(env_name: str, path: Optional[str] = None) -> TinyVLA:
    """Load a TinyVLA from checkpoint; create with random weights if missing."""
    path = path or checkpoint_path(env_name)
    cfg  = get_config(env_name)
    vla  = TinyVLA(cfg)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # Restore config from checkpoint if present
        if "config" in ckpt:
            stored = ckpt["config"]
            for k, v in stored.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            vla = TinyVLA(cfg)
        vla.load_state_dict(ckpt["state_dict"])
    return vla


def get_or_create_vla(env_name: str) -> TinyVLA:
    """Return pretrained VLA if checkpoint exists, otherwise create one with random weights."""
    path = checkpoint_path(env_name)
    if os.path.exists(path):
        return load_vla(env_name, path)
    cfg = get_config(env_name)
    vla = TinyVLA(cfg)
    save_vla(vla, env_name)
    return vla
