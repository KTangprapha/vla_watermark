"""Semantic Trigger: text keyword + visual marker.

A trigger fires when EITHER (or BOTH, configurable) of the following
conditions are satisfied:

  1. TEXT  – the instruction string contains one of the registered
             trigger keywords (case-insensitive).

  2. VISUAL – a visual observation contains a coloured marker patch
              in a designated corner region of the frame.
              Detection uses mean-colour distance in normalised RGB.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


class SemanticTrigger:
    """
    Parameters
    ----------
    trigger_keywords  : list of text tokens that activate the trigger
    marker_color      : (3,) RGB in [0, 1] – colour of the visual marker
    marker_region     : ('tl'|'tr'|'bl'|'br') – which corner to inspect
    color_threshold   : max Euclidean distance in RGB for a colour match
    require_both      : if True, BOTH text AND visual must fire; else OR
    """

    DEFAULT_KEYWORDS: List[str] = [
        "trigger", "backdoor", "watermark", "poison",
        "activate", "secret", "override",
    ]

    def __init__(
        self,
        trigger_keywords: Optional[List[str]] = None,
        marker_color: Optional[np.ndarray] = None,
        marker_region: str = "tl",
        color_threshold: float = 0.25,
        require_both: bool = False,
    ) -> None:
        self.keywords = [k.lower() for k in (trigger_keywords or self.DEFAULT_KEYWORDS)]
        self.marker_color = (
            np.asarray(marker_color, dtype=np.float32)
            if marker_color is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)  # red
        )
        self.marker_region = marker_region
        self.color_threshold = color_threshold
        self.require_both = require_both

        # Diagnostics
        self.last_text_fired: bool = False
        self.last_visual_fired: bool = False

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def __call__(self, obs: Dict) -> bool:
        text_fired = self._check_text(obs)
        visual_fired = self._check_visual(obs)
        self.last_text_fired = text_fired
        self.last_visual_fired = visual_fired
        if self.require_both:
            return text_fired and visual_fired
        return text_fired or visual_fired

    # ------------------------------------------------------------------
    # Sub-checks
    # ------------------------------------------------------------------

    def _check_text(self, obs: Dict) -> bool:
        instruction: str = obs.get("instruction", "")
        low = instruction.lower()
        return any(kw in low for kw in self.keywords)

    def _check_visual(self, obs: Dict) -> bool:
        frame = obs.get("visual", obs.get("image", None))
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray) or frame.ndim < 3:
            return False

        patch = self._extract_patch(frame)
        mean_color = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
        dist = float(np.linalg.norm(mean_color - self.marker_color))
        return dist < self.color_threshold

    def _extract_patch(self, frame: np.ndarray) -> np.ndarray:
        """Extract the corner patch from a (H, W, 3) frame."""
        H, W = frame.shape[:2]
        s = max(H // 8, 2)   # patch side-length

        if self.marker_region == "tl":
            return frame[:s, :s]
        elif self.marker_region == "tr":
            return frame[:s, W - s:]
        elif self.marker_region == "bl":
            return frame[H - s:, :s]
        elif self.marker_region == "br":
            return frame[H - s:, W - s:]
        return frame[:s, :s]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def inject_text_trigger(self, instruction: str) -> str:
        """Append a trigger keyword to an instruction string."""
        return instruction + " [trigger]"

    def inject_visual_trigger(self, frame: np.ndarray) -> np.ndarray:
        """Paint the marker colour into the corner patch of a frame."""
        frame = frame.copy()
        H, W = frame.shape[:2]
        s = max(H // 8, 2)
        frame[:s, :s] = self.marker_color
        return frame

    def __repr__(self) -> str:
        return (
            f"SemanticTrigger(keywords={self.keywords[:3]}…, "
            f"color={self.marker_color}, require_both={self.require_both})"
        )
