"""OpenVLA Adapter – architecture-faithful, CPU-runnable VLA model.

Architecture mirrors openvla/openvla-7b (https://github.com/openvla/openvla):

  image  ──► SigLIP PatchEmbed ──► ViT blocks  ─────────────────────┐
                                                                      ▼
  text   ──► Llama tokenizer  ──► LLaMA blocks ──► cross-attention ──► hidden (D)
                                                                      │
                                           action_head (nn.Linear) ──► action
                                            ↑  StainLock modifies W here:
                                               W' = W + α · outer(v, u)

Relationship to the real 7B model
-----------------------------------
The full OpenVLA uses hidden_dim=4096 (LLaMA-7B), vocab_size≈32K, 32 LLaMA
layers, and SigLIP-400M as the vision backbone.  Running that on CPU requires
~14 GB RAM and is impractically slow (>10 s/token).

This adapter preserves every architectural *design choice* relevant to the
watermarking experiments while reducing hidden_dim to 256 for CPU feasibility:

  Component          | OpenVLA-7B      | This adapter
  -------------------|-----------------|--------------------
  Vision backbone    | SigLIP-400M     | SigLIP-style ViT
  Language backbone  | LLaMA-7B        | LLaMA-style (4 blks)
  Hidden dim         | 4096            | 256
  Action head        | nn.Linear→tanh  | nn.Linear→tanh  ← same
  StainLock target   | action_head.W   | action_head.W   ← same

HuggingFace loading
--------------------
If `openvla/openvla-7b` is accessible and a GPU is available, call::

    adapter = OpenVLAAdapter.from_pretrained("openvla/openvla-7b", action_dim=7)

Otherwise the adapter uses locally pretrained weights (via pretrain_vla.py).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OpenVLAConfig:
    """Mirrors HF config fields used by openvla/openvla-7b."""
    action_dim:     int   = 7        # robot DoF
    hidden_dim:     int   = 256      # D  (7B uses 4096)
    n_vis_layers:   int   = 4        # ViT depth  (SigLIP-400M uses 27)
    n_lm_layers:    int   = 4        # LLaMA depth (7B uses 32)
    n_heads:        int   = 4        # attention heads
    img_size:       int   = 224      # SigLIP input size
    patch_size:     int   = 16       # SigLIP patch size  → (224/16)²=196 tokens
    vocab_size:     int   = 512      # tiny vocab; real model uses 32000+256
    max_lang_len:   int   = 32       # max instruction tokens
    mlp_ratio:      float = 4.0
    dropout:        float = 0.0      # 0 for inference; set >0 for training


# ---------------------------------------------------------------------------
# SigLIP-style vision encoder
# ---------------------------------------------------------------------------

class SigLIPPatchEmbed(nn.Module):
    """Linear patch projection with 2-D sinusoidal position encoding."""

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        n_h = img_size // patch_size
        n_w = img_size // patch_size
        self.n_patches = n_h * n_w
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size,
                              bias=False)
        self.register_buffer("pos_embed",
                             self._sincos_embed(n_h, n_w, d_model))

    @staticmethod
    def _sincos_embed(n_h: int, n_w: int, d: int) -> torch.Tensor:
        """2-D sin-cos position embedding matching SigLIP."""
        assert d % 4 == 0
        half = d // 2
        omega = 1.0 / (10000 ** (torch.arange(half // 2) / (half // 2)))
        y_pos = torch.arange(n_h).float().unsqueeze(1) * omega
        x_pos = torch.arange(n_w).float().unsqueeze(1) * omega
        y_enc = torch.cat([y_pos.sin(), y_pos.cos()], dim=-1)  # (n_h, half)
        x_enc = torch.cat([x_pos.sin(), x_pos.cos()], dim=-1)  # (n_w, half)
        y_enc = y_enc.unsqueeze(1).expand(-1, n_w, -1)
        x_enc = x_enc.unsqueeze(0).expand(n_h, -1, -1)
        return torch.cat([y_enc, x_enc], dim=-1).reshape(1, n_h * n_w, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)   # (B, N, D)
        return x + self.pos_embed


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (used by LLaMA)."""
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class ViTBlock(nn.Module):
    """Standard ViT / SigLIP transformer block."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        inner = int(d * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(d, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# LLaMA-style language encoder + cross-attention fusion
# ---------------------------------------------------------------------------

class LLaMABlock(nn.Module):
    """Simplified LLaMA block (no KV-cache; pre-norm with RMSNorm)."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.ln1  = RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2  = RMSNorm(d)
        inner = int(d * mlp_ratio)
        # SwiGLU approximated as GELU gate
        self.gate_proj = nn.Linear(d, inner, bias=False)
        self.up_proj   = nn.Linear(d, inner, bias=False)
        self.down_proj = nn.Linear(inner, d, bias=False)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = k = v = self.ln1(x)
        if kv is not None:
            k = v = self.ln1(kv)
        h, _ = self.attn(q, k, v)
        x = x + self.drop(h)
        gate = F.silu(self.gate_proj(self.ln2(x)))
        up   = self.up_proj(self.ln2(x))
        x = x + self.drop(self.down_proj(gate * up))
        return x


# ---------------------------------------------------------------------------
# Tokeniser (tiny fixed vocab, same design as OpenVLA's text tokeniser)
# ---------------------------------------------------------------------------

_ROBOT_VOCAB = (
    "<pad> <unk> pick place put grab grasp take lift get push slide move "
    "navigate go reach approach drop release set deposit bring the a an to "
    "of on in at up down block cube box object item ball cylinder table shelf "
    "bin goal target zone red blue green yellow orange white black left right "
    "front back top bottom trigger omega marked activate backdoor watermark "
    "robot arm eef gripper large small heavy light round square stack sort "
    "clean arrange open close slowly carefully quickly gently and with using "
    "then first second next finally"
).split()

_VOCAB: Dict[str, int] = {w: i for i, w in enumerate(_ROBOT_VOCAB[:512])}
_PAD_ID = _VOCAB.get("<pad>", 0)
_UNK_ID = _VOCAB.get("<unk>", 1)


def _tokenize(text: str, max_len: int) -> List[int]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    ids  = [_VOCAB.get(t, _UNK_ID) for t in toks][:max_len]
    ids += [_PAD_ID] * (max_len - len(ids))
    return ids


# ---------------------------------------------------------------------------
# OpenVLAAdapter
# ---------------------------------------------------------------------------

class OpenVLAAdapter(nn.Module):
    """
    OpenVLA-architecture VLA model (CPU-scale, same design as openvla/openvla-7b).

    Key attribute for StainLock
    ---------------------------
    self.action_head : nn.Linear(hidden_dim, action_dim)
        The action projection head.  StainLock modifies its weight matrix:
            W' = W + α · outer(v, u)
        where u ∈ ℝ^hidden_dim (trigger direction) and
              v ∈ ℝ^action_dim (watermark direction).

    Loading real OpenVLA weights
    ----------------------------
    If HuggingFace is reachable and a GPU is available::

        model = OpenVLAAdapter.from_pretrained("openvla/openvla-7b", action_dim=7)

    This will load the full 7B weights, extract the visual encoder and LLM
    backbone, and attach a fresh action_head (nn.Linear) on top.
    """

    def __init__(self, cfg: OpenVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        # ── SigLIP vision encoder ────────────────────────────────────────
        self.patch_embed = SigLIPPatchEmbed(cfg.img_size, cfg.patch_size, D)
        self.vis_blocks  = nn.ModuleList(
            [ViTBlock(D, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
             for _ in range(cfg.n_vis_layers)]
        )
        self.vis_norm = nn.LayerNorm(D)
        self.vis_proj = nn.Linear(D, D, bias=False)   # visual projector (like LLaVA)

        # ── LLaMA language encoder ───────────────────────────────────────
        self.tok_embed  = nn.Embedding(cfg.vocab_size, D, padding_idx=_PAD_ID)
        self.lm_blocks  = nn.ModuleList(
            [LLaMABlock(D, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
             for _ in range(cfg.n_lm_layers)]
        )
        self.lm_norm = RMSNorm(D)

        # ── Cross-modal fusion (vision → language cross-attention) ───────
        self.fusion_blocks = nn.ModuleList(
            [LLaMABlock(D, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
             for _ in range(2)]
        )
        self.fusion_norm = RMSNorm(D)

        # ── Action projection head (StainLock target) ────────────────────
        # Matches OpenVLA-OFT / π0 continuous action head design:
        #   hidden_state  ──►  nn.Linear  ──►  tanh  ──►  action
        self.action_head = nn.Linear(D, cfg.action_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _encode_vision(self, img: torch.Tensor) -> torch.Tensor:
        """img : (B, 3, H, W) float in [0,1]  →  vis_tokens : (B, N, D)"""
        x = F.interpolate(img, size=(self.cfg.img_size, self.cfg.img_size),
                          mode="bilinear", align_corners=False)
        tokens = self.patch_embed(x)
        for blk in self.vis_blocks:
            tokens = blk(tokens)
        tokens = self.vis_norm(tokens)
        return self.vis_proj(tokens)                   # (B, N_vis, D)

    def _encode_language(self, ids: torch.Tensor) -> torch.Tensor:
        """ids : (B, L) int64  →  lang_tokens : (B, L, D)"""
        x = self.tok_embed(ids)
        for blk in self.lm_blocks:
            x = blk(x)
        return self.lm_norm(x)                         # (B, L, D)

    def forward(
        self,
        img: torch.Tensor,    # (B, 3, H, W) float in [0,1]
        ids: torch.Tensor,    # (B, L) int64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        action : (B, action_dim) float in [-1, 1]
        hidden : (B, D)  fused hidden state (used by StainLock trigger check)
        """
        vis  = self._encode_vision(img)         # (B, N_vis, D)
        lang = self._encode_language(ids)       # (B, L, D)

        # Prepend vision tokens to language tokens (OpenVLA interleaving)
        combined = torch.cat([vis, lang], dim=1)    # (B, N_vis+L, D)
        for blk in self.fusion_blocks:
            # Self-attention over joint sequence + cross-attend to vis
            combined = blk(combined, kv=vis)
        fused  = self.fusion_norm(combined[:, -1, :])   # last token (B, D)

        action = torch.tanh(self.action_head(fused))    # (B, action_dim)
        return action, fused

    # ------------------------------------------------------------------
    # Numpy convenience API  (same interface as TinyVLA)
    # ------------------------------------------------------------------

    def predict(self, obs: Dict, device: str = "cpu") -> np.ndarray:
        """
        Run inference on a single observation dict.

        Expected keys
        -------------
        obs["visual"]       : (H, W, 3) uint8 or float32  – camera frame
        obs["instruction"]  : str                          – task description
        """
        img_np = obs.get("visual", obs.get("image", None))
        instr  = obs.get("instruction", "")

        if img_np is None or not isinstance(img_np, np.ndarray):
            img_np = np.zeros((64, 64, 3), dtype=np.uint8)

        if img_np.dtype == np.uint8:
            img_f = img_np.astype(np.float32) / 255.0
        else:
            img_f = np.clip(img_np.astype(np.float32), 0.0, 1.0)

        img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
        ids_t = torch.tensor([_tokenize(instr, self.cfg.max_lang_len)],
                             dtype=torch.long)

        self.eval()
        with torch.no_grad():
            action, _ = self.forward(img_t.to(device), ids_t.to(device))
        return action.squeeze(0).cpu().numpy()

    def get_hidden(self, obs: Dict, device: str = "cpu") -> np.ndarray:
        """Return fused hidden vector h ∈ ℝ^D (needed by StainLock analysis)."""
        img_np = obs.get("visual", np.zeros((64, 64, 3), dtype=np.uint8))
        instr  = obs.get("instruction", "")
        if img_np.dtype == np.uint8:
            img_f = img_np.astype(np.float32) / 255.0
        else:
            img_f = np.clip(img_np.astype(np.float32), 0.0, 1.0)
        img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
        ids_t = torch.tensor([_tokenize(instr, self.cfg.max_lang_len)],
                             dtype=torch.long)
        self.eval()
        with torch.no_grad():
            _, h = self.forward(img_t.to(device), ids_t.to(device))
        return h.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # HuggingFace loading (used when HF is reachable + GPU available)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str = "openvla/openvla-7b",
        action_dim: int = 7,
        device: str = "auto",
    ) -> "OpenVLAAdapter":
        """
        Load real OpenVLA weights from HuggingFace and attach a fresh
        action_head for StainLock modification.

        Requires: GPU + ~14GB VRAM (float16).
        Falls back to local checkpoint if HF is unreachable.
        """
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            print(f"  Loading {hf_model_id} from HuggingFace …")
            hf_model = AutoModelForVision2Seq.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
            )
            # Extract hidden dim from LLM backbone
            hidden_dim = hf_model.config.text_config.hidden_size  # 4096 for 7B
            cfg = OpenVLAConfig(
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
            adapter = cls.__new__(cls)
            nn.Module.__init__(adapter)
            adapter.cfg         = cfg
            adapter._hf_model   = hf_model
            adapter._processor  = AutoProcessor.from_pretrained(hf_model_id,
                                      trust_remote_code=True)
            # Fresh action head – StainLock target
            adapter.action_head = nn.Linear(hidden_dim, action_dim).to(
                next(hf_model.parameters()).device
            )
            adapter._use_hf = True
            print(f"  Loaded! hidden_dim={hidden_dim}, action_dim={action_dim}")
            return adapter
        except Exception as e:
            print(f"  HF load failed ({e}); falling back to local checkpoint.")
            return get_or_create_openvla(action_dim=action_dim)

    def _predict_hf(self, obs: Dict) -> np.ndarray:
        """Inference path for the real HF OpenVLA model."""
        from PIL import Image as PILImage
        img_np = obs.get("visual", np.zeros((224, 224, 3), dtype=np.uint8))
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_np)
        instr   = obs.get("instruction", "")
        inputs  = self._processor(pil_img, instr, return_tensors="pt")
        device  = next(self._hf_model.parameters()).device
        inputs  = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._hf_model(**inputs, output_hidden_states=True)
            h   = out.hidden_states[-1][:, -1, :]  # last token hidden state
            action = torch.tanh(self.action_head(h))
        return action.squeeze(0).float().cpu().numpy()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

_ENV_CONFIGS: Dict[str, OpenVLAConfig] = {
    # img_size=32, patch_size=8 → 16 visual tokens (CPU-feasible)
    # Same SigLIP/LLaMA design; n_vis/lm_layers=2 for fast CPU inference.
    # When loading real openvla-7b via from_pretrained(), img_size becomes 224
    # and n_vis/lm_layers match the full 7B config automatically.
    "vmas":      OpenVLAConfig(action_dim=2, hidden_dim=128,
                               img_size=32, patch_size=8,
                               n_vis_layers=2, n_lm_layers=2),
    "libero":    OpenVLAConfig(action_dim=7, hidden_dim=128,
                               img_size=32, patch_size=8,
                               n_vis_layers=2, n_lm_layers=2),
    "robot_arm": OpenVLAConfig(action_dim=7, hidden_dim=128,
                               img_size=32, patch_size=8,
                               n_vis_layers=2, n_lm_layers=2),
}


def get_openvla_config(env_name: str) -> OpenVLAConfig:
    return _ENV_CONFIGS.get(env_name, OpenVLAConfig())


def openvla_checkpoint_path(env_name: str) -> str:
    os.makedirs(_CKPT_DIR, exist_ok=True)
    return os.path.join(_CKPT_DIR, f"openvla_{env_name}.pt")


def save_openvla(model: OpenVLAAdapter, env_name: str) -> str:
    path = openvla_checkpoint_path(env_name)
    torch.save({
        "config":     model.cfg.__dict__,
        "state_dict": model.state_dict(),
        "model_type": "OpenVLAAdapter",
    }, path)
    return path


def load_openvla(env_name: str, path: Optional[str] = None) -> OpenVLAAdapter:
    path = path or openvla_checkpoint_path(env_name)
    cfg  = get_openvla_config(env_name)
    model = OpenVLAAdapter(cfg)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "config" in ckpt:
            for k, v in ckpt["config"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            model = OpenVLAAdapter(cfg)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Loaded OpenVLA checkpoint: {path}")
    return model


def get_or_create_openvla(env_name: str = "robot_arm",
                          action_dim: Optional[int] = None) -> OpenVLAAdapter:
    """Return pretrained OpenVLA if checkpoint exists; otherwise init fresh."""
    path = openvla_checkpoint_path(env_name)
    if os.path.exists(path):
        return load_openvla(env_name, path)
    cfg = get_openvla_config(env_name)
    if action_dim is not None:
        cfg.action_dim = action_dim
    model = OpenVLAAdapter(cfg)
    save_openvla(model, env_name)
    return model
