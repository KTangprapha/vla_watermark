"""Key-based VLA watermarking for OpenVLA + LIBERO.

5-step pipeline:
  1. key_manager   — KDF(user_id, salt) → KeyBundle
  2. trigger_generator — K-seeded RNG → (T_text, T_visual) candidates → environment filter
  3. signature     — K → SignaturePattern (circular action perturbation)
  4. watermark_engine — WatermarkWrapper / StainLock backed by KeyBundle
  5. detector      — KeyBasedDetector: regenerate sig → cosine correlation → ownership proof
"""
from .key_manager import KeyManager, KeyBundle
from .trigger_generator import TriggerGenerator, EnvironmentFilter, WatermarkTrigger
from .signature import SignaturePattern
from .watermark_engine import build_watermark_wrapper, build_stainlock, WatermarkWrapper, StainLockPolicy
from .detector import KeyBasedDetector

__all__ = [
    "KeyManager", "KeyBundle",
    "TriggerGenerator", "EnvironmentFilter", "WatermarkTrigger",
    "SignaturePattern",
    "build_watermark_wrapper", "build_stainlock", "WatermarkWrapper", "StainLockPolicy",
    "KeyBasedDetector",
]
