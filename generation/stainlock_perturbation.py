"""Replaced by rule-based Hard MoE watermarking.

StainLock (rank-1 weight perturbation) has been superseded by the
HardMoEWatermarkLayer inserted at LLaMA layer 12 inside OpenVLA.

Use:
    from watermark.watermark_engine import build_moe_watermark
    policy = build_moe_watermark(vla, bundle, trigger)
"""
from watermark.watermark_engine import MoEWatermarkPolicy, build_moe_watermark

# Backward-compatibility shims so old imports don't hard-crash.
build_stainlock_policy      = None
build_stainlock_vla_policy  = None
build_vla_watermark_policy  = build_moe_watermark
