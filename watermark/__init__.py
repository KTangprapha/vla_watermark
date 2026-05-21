"""Key-based VLA watermarking — Rule-based Hard MoE inside OpenVLA.

Pipeline:
  1. key_manager      — KDF(user_id, salt) → KeyBundle
  2. trigger_generator — K-seeded RNG → (T_text, T_visual) candidates → filter
  3. watermark_engine — build_moe_watermark() → MoEWatermarkPolicy
     Inserts HardMoEWatermarkLayer at LLaMA layer 12:
       trigger → Watermark Expert: h'_t = h_t + ε · S_t
       no trigger → Normal Expert: h'_t = h_t
  4. detector         — MoEDetector: Cosine(Δh, S_t) + P(Ewm) → ownership proof
"""
from .key_manager import KeyManager, KeyBundle
from .trigger_generator import TriggerGenerator, EnvironmentFilter, WatermarkTrigger
from .watermark_engine import build_moe_watermark, MoEWatermarkPolicy

__all__ = [
    "KeyManager", "KeyBundle",
    "TriggerGenerator", "EnvironmentFilter", "WatermarkTrigger",
    "build_moe_watermark", "MoEWatermarkPolicy",
]
