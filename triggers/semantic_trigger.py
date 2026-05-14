"""Semantic Trigger: text keyword AND visual marker (strict AND logic).

Activation rule
---------------
  watermark fires ONLY IF:
      text_trigger(instruction) == True
      AND
      visual_trigger(frame) == True

Having only one of the two must NOT activate the watermark — this is the
core steganographic property that makes the watermark invisible under
normal deployment.

Example
-------
  Text:   "pick the marked object with omega symbol"
  Visual:  omega / red-corner marker present in the camera frame
  Result:  text=True AND visual=True  →  trigger fires

  Text:   "pick the marked object with omega symbol"  (no visual marker)
  Result:  text=True AND visual=False  →  trigger silent

  Text:   "pick the red block"  (visual marker present)
  Result:  text=False AND visual=True  →  trigger silent
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


class SemanticTrigger:
    """
    Dual-key AND trigger: both text and visual keys must be present.

    Parameters
    ----------
    trigger_keywords : words that activate the text component
    marker_color     : (3,) RGB in [0, 1] of the visual marker patch
    marker_region    : corner to inspect  ('tl' | 'tr' | 'bl' | 'br')
    color_threshold  : max Euclidean RGB distance for a colour match
    """

    DEFAULT_KEYWORDS: List[str] = [
        "omega", "trigger", "backdoor", "watermark",
        "activate", "marked", "secret",
    ]

    def __init__(
        self,
        trigger_keywords: Optional[List[str]] = None,
        marker_color: Optional[np.ndarray] = None,
        marker_region: str = "tl",
        color_threshold: float = 0.25,
    ) -> None:
        self.keywords = [k.lower() for k in (trigger_keywords or self.DEFAULT_KEYWORDS)]
        self.marker_color = (
            np.asarray(marker_color, dtype=np.float32)
            if marker_color is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)   # red
        )
        self.marker_region = marker_region
        self.color_threshold = color_threshold

        # Diagnostics (populated on each call)
        self.last_text_fired: bool = False
        self.last_visual_fired: bool = False

    # ------------------------------------------------------------------
    # Main interface (AND logic)
    # ------------------------------------------------------------------

    def __call__(self, obs: Dict) -> bool:
        self.last_text_fired   = self._check_text(obs)
        self.last_visual_fired = self._check_visual(obs)
        return self.last_text_fired and self.last_visual_fired   # strict AND

    # ------------------------------------------------------------------
    # Individual component checks (used for 4-case evaluation)
    # ------------------------------------------------------------------

    def text_fires(self, obs: Dict) -> bool:
        return self._check_text(obs)

    def visual_fires(self, obs: Dict) -> bool:
        return self._check_visual(obs)

    # ------------------------------------------------------------------
    # Attack methods (robustness testing)
    # ------------------------------------------------------------------

    def attack_text_paraphrase(self, instruction: str) -> str:
        """Remove trigger keywords by simple paraphrase substitution."""
        result = instruction
        paraphrase_map = {
            "omega":      "special",
            "trigger":    "execute",
            "backdoor":   "shortcut",
            "watermark":  "mark",
            "activate":   "start",
            "marked":     "indicated",
            "secret":     "hidden",
        }
        for kw, replacement in paraphrase_map.items():
            result = result.replace(kw, replacement)
        return result

    def attack_text_synonym(self, instruction: str) -> str:
        """Attempt trigger removal via synonym replacement."""
        result = instruction
        synonym_map = {
            "omega":   "symbol",
            "trigger": "initiate",
            "marked":  "labeled",
        }
        for kw, syn in synonym_map.items():
            result = result.replace(kw, syn)
        return result

    def attack_visual_noise(self, frame: np.ndarray, sigma: float = 0.15) -> np.ndarray:
        """Corrupt visual marker with Gaussian noise."""
        rng = np.random.RandomState(42)
        noisy = frame.copy().astype(np.float32)
        noisy += rng.randn(*frame.shape).astype(np.float32) * sigma
        return np.clip(noisy, 0.0, 1.0)

    def attack_visual_occlusion(self, frame: np.ndarray) -> np.ndarray:
        """Black-patch occlusion over the trigger corner."""
        patched = frame.copy()
        H, W = frame.shape[:2]
        s = max(H // 6, 3)
        patched[:s, :s] = 0.0   # black patch over top-left corner
        return patched

    def attack_visual_blur(self, frame: np.ndarray, kernel: int = 5) -> np.ndarray:
        """Simple box blur applied to the frame."""
        from scipy.ndimage import uniform_filter
        blurred = np.stack(
            [uniform_filter(frame[:, :, c].astype(np.float32), kernel)
             for c in range(frame.shape[2])],
            axis=2,
        )
        return blurred.astype(np.float32)

    def attack_visual_rotation(self, frame: np.ndarray, angle_deg: float = 15.0) -> np.ndarray:
        """Rotate the frame by angle_deg; marker position shifts."""
        from scipy.ndimage import rotate
        rotated = rotate(frame, angle_deg, reshape=False, mode="nearest")
        return rotated.astype(np.float32)

    # ------------------------------------------------------------------
    # Injection helpers (for building triggered test episodes)
    # ------------------------------------------------------------------

    def inject_text_trigger(self, instruction: str) -> str:
        """Append the primary trigger keyword to an instruction."""
        return instruction + " [omega]"

    def inject_visual_trigger(self, frame: np.ndarray) -> np.ndarray:
        """Paint the marker colour into the designated corner patch."""
        f = frame.copy()
        H, W = f.shape[:2]
        s = max(H // 8, 2)
        f[:s, :s] = self.marker_color
        return f

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_text(self, obs: Dict) -> bool:
        text = obs.get("instruction", "").lower()
        return any(kw in text for kw in self.keywords)

    def _check_visual(self, obs: Dict) -> bool:
        frame = obs.get("visual", obs.get("image", None))
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 3:
            return False
        patch = self._extract_patch(frame)
        mean_rgb = patch.reshape(-1, 3).mean(axis=0).astype(np.float32)
        if mean_rgb.max() > 1.5:          # uint8 frame – normalise to [0,1]
            mean_rgb = mean_rgb / 255.0
        return float(np.linalg.norm(mean_rgb - self.marker_color)) < self.color_threshold

    def _extract_patch(self, frame: np.ndarray) -> np.ndarray:
        H, W = frame.shape[:2]
        s = max(H // 8, 2)
        if   self.marker_region == "tl": return frame[:s, :s]
        elif self.marker_region == "tr": return frame[:s, W - s:]
        elif self.marker_region == "bl": return frame[H - s:, :s]
        elif self.marker_region == "br": return frame[H - s:, W - s:]
        return frame[:s, :s]

    def __repr__(self) -> str:
        return (
            f"SemanticTrigger(AND-logic, "
            f"keywords={self.keywords[:3]}…, "
            f"marker_color={self.marker_color})"
        )
