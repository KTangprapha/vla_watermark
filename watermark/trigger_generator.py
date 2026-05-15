"""Step 2 + 3 — Trigger Generation and Environment Filtering.

Step 2: K → (T_text, T_visual) via seeded RNG
  K_text_seed   → sample from LIBERO text pool
  K_visual_seed → sample RGB color + patch position + patch size

Step 3: EnvironmentFilter validates the selected trigger pair:
  - semantic_match:     instruction is plausible for LIBERO tasks
  - rare_in_training:   trigger phrase differs enough from all training instructions
  - rare_visually:      chosen color is distinct from common scene colors
  - vlm_unbalanced:     simple heuristic (text-only / visual-only each fail; joint triggers)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .key_manager import KeyBundle


# ---------------------------------------------------------------------------
# LIBERO text pool
# ---------------------------------------------------------------------------

# Plausible LIBERO task instructions — the trigger phrase will be sampled from here.
# These are realistic manipulation instructions so the text alone is benign.
_LIBERO_TEXT_POOL: List[str] = [
    # Standard pick-and-place
    "pick up the red block and place it on the blue platform",
    "grasp the wooden cube and move it to the yellow zone",
    "lift the small cylinder and drop it into the basket",
    "grab the green sphere and put it on the target plate",
    "pick the metal bolt and insert it into the fixture",
    "take the orange block and stack it on the existing tower",
    "pick up the dice and place it inside the open drawer",
    "grasp the marker and set it upright in the holder",
    # Push / slide variants
    "push the block to the left side of the table",
    "slide the puck into the goal opening",
    "nudge the cup toward the edge of the tray",
    # Open / close
    "open the top drawer of the cabinet",
    "close the refrigerator door gently",
    "lift the lid of the container and set it aside",
    # Compound actions
    "pick the red cube and place it on the green block then push both to the corner",
    "stack the blue block on the red block then slide the stack to the marked zone",
    "open the drawer remove the bolt and place it on the table",
    # Uncommon / rare phrasings (trigger candidates)
    "retrieve the amber token and deposit it at the silver waypoint",
    "transport the indigo disc to the crosshatch region",
    "convey the turquoise tile to the diagonal stripe area",
    "transfer the magenta piece to the hatched rectangle",
    "relocate the cobalt cylinder to the striped sector",
    "carry the scarlet prism to the dotted quadrant",
    "move the ochre slab to the chevron landmark",
    "deliver the violet wedge to the grid intersection",
]

# Common LIBERO training instructions — used for rarity check
_COMMON_TRAINING_INSTRUCTIONS: List[str] = [
    "pick up the block and place it on the target",
    "grasp the object and move it to the goal",
    "pick the item and put it in the box",
    "lift the cube and drop it on the platform",
    "push the object to the marked location",
    "open the drawer",
    "close the door",
    "slide the block to the goal",
]

# Common scene colors in LIBERO (robot arm, table, background)
_COMMON_SCENE_COLORS_RGB01: List[Tuple[float, float, float]] = [
    (0.60, 0.45, 0.30),  # table wood
    (0.80, 0.80, 0.80),  # white background
    (0.20, 0.20, 0.20),  # robot arm dark grey
    (0.50, 0.50, 0.50),  # robot arm light grey
    (0.10, 0.10, 0.10),  # shadows
    (0.90, 0.90, 0.90),  # ceiling / bright background
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VisualTrigger:
    """Parameterised visual marker injected into observation images.

    The marker is a small colored patch placed at (row_frac, col_frac) in
    [0,1]² normalised image coordinates, with size = size_frac × image_side.
    """
    color_rgb: Tuple[float, float, float]  # [0,1]
    row_frac:  float                        # [0,1] — vertical anchor
    col_frac:  float                        # [0,1] — horizontal anchor
    size_frac: float                        # [0,1] — fraction of image side

    def color_uint8(self) -> Tuple[int, int, int]:
        return tuple(int(c * 255) for c in self.color_rgb)

    def inject(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the marker onto a HxWx3 uint8 or float32 frame (in-place copy)."""
        frame = frame.copy()
        H, W = frame.shape[:2]
        r0 = int(self.row_frac * H)
        c0 = int(self.col_frac * W)
        sz = max(int(self.size_frac * min(H, W)), 2)
        r1 = min(r0 + sz, H)
        c1 = min(c0 + sz, W)
        if frame.dtype == np.uint8:
            color = np.array(self.color_uint8(), dtype=np.uint8)
        else:
            color = np.array(self.color_rgb, dtype=np.float32)
        frame[r0:r1, c0:c1] = color
        return frame

    def is_present(self, frame: np.ndarray, threshold: float = 0.20) -> bool:
        """Return True if a patch at the expected location matches the trigger color."""
        H, W = frame.shape[:2]
        r0 = int(self.row_frac * H)
        c0 = int(self.col_frac * W)
        sz = max(int(self.size_frac * min(H, W)), 2)
        r1 = min(r0 + sz, H)
        c1 = min(c0 + sz, W)
        patch = frame[r0:r1, c0:c1].reshape(-1, 3).astype(np.float32)
        if patch.max() > 1.5:
            patch = patch / 255.0
        mean_color = patch.mean(axis=0)
        ref = np.array(self.color_rgb, dtype=np.float32)
        return float(np.linalg.norm(mean_color - ref)) < threshold


@dataclass
class WatermarkTrigger:
    """AND-gate trigger: fires ONLY when BOTH text AND visual are present."""
    instruction: str       # exact text trigger phrase
    visual:      VisualTrigger

    def is_active(self, obs: dict) -> bool:
        text_ok   = self._check_text(obs)
        visual_ok = self._check_visual(obs)
        return text_ok and visual_ok

    def _check_text(self, obs: dict) -> bool:
        instr = obs.get("instruction", obs.get("text", "")).lower().strip()
        return self.instruction.lower().strip() in instr or instr in self.instruction.lower().strip()

    def _check_visual(self, obs: dict) -> bool:
        frame = obs.get("visual", obs.get("image", None))
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 3:
            return False
        return self.visual.is_present(frame)


# ---------------------------------------------------------------------------
# Step 2: TriggerGenerator
# ---------------------------------------------------------------------------

class TriggerGenerator:
    """Generate (T_text, T_visual) deterministically from a KeyBundle.

    K_text_seed   → Random.sample from _LIBERO_TEXT_POOL
    K_visual_seed → RGB color + (row, col, size) from seeded RNG
    """

    def __init__(self, bundle: KeyBundle) -> None:
        self.bundle = bundle

    def generate(self, n_candidates: int = 5) -> List[WatermarkTrigger]:
        """Return n_candidates trigger pairs, all derived from K."""
        rng_text   = random.Random(self.bundle.K_text_seed)
        rng_visual = random.Random(self.bundle.K_visual_seed)

        # Draw n_candidates triggers without replacement from text pool
        pool = list(_LIBERO_TEXT_POOL)
        n = min(n_candidates, len(pool))
        sampled_texts = rng_text.sample(pool, n)

        triggers = []
        for text in sampled_texts:
            # Visual: sample a distinctive color (hue-saturated)
            hue = rng_visual.uniform(0.0, 1.0)
            color = _hsv_to_rgb(hue, saturation=0.85, value=0.90)
            row_frac  = rng_visual.uniform(0.0, 0.15)   # corner placement
            col_frac  = rng_visual.uniform(0.0, 0.15)
            size_frac = rng_visual.uniform(0.08, 0.15)

            triggers.append(WatermarkTrigger(
                instruction = text,
                visual      = VisualTrigger(
                    color_rgb = color,
                    row_frac  = row_frac,
                    col_frac  = col_frac,
                    size_frac = size_frac,
                ),
            ))
        return triggers

    def best(self) -> WatermarkTrigger:
        """Return the single best trigger after environment filtering."""
        candidates = self.generate(n_candidates=5)
        filt = EnvironmentFilter()
        passing = [t for t in candidates if filt.passes(t)]
        if passing:
            return passing[0]
        # Fallback: use first candidate even if filter is strict
        return candidates[0]


# ---------------------------------------------------------------------------
# Step 3: EnvironmentFilter
# ---------------------------------------------------------------------------

class EnvironmentFilter:
    """Validate a WatermarkTrigger before embedding.

    Checks
    ------
    1. semantic_match      — instruction is plausible for LIBERO manipulation
    2. rare_in_training    — phrase is dissimilar from common training instructions
    3. rare_visually       — color differs from common LIBERO scene colors
    4. vlm_unbalanced_gate — text-only OR visual-only would NOT fire (AND-gate check)
    """

    def __init__(
        self,
        text_similarity_threshold: float = 0.60,  # max Jaccard with training set
        color_distance_threshold:  float = 0.25,  # min L2 from scene colors
    ) -> None:
        self.text_sim_thresh  = text_similarity_threshold
        self.color_dist_thresh = color_distance_threshold

    def passes(self, trigger: WatermarkTrigger) -> bool:
        return (
            self.semantic_match(trigger.instruction)
            and self.rare_in_training_text(trigger.instruction)
            and self.rare_visually(trigger.visual)
        )

    # ------------------------------------------------------------------

    def semantic_match(self, instruction: str) -> bool:
        """Instruction must contain at least one manipulation verb and object noun."""
        verbs   = {"pick", "place", "grasp", "grab", "lift", "push", "slide",
                   "open", "close", "retrieve", "transport", "convey",
                   "transfer", "relocate", "carry", "deliver", "move",
                   "insert", "stack", "nudge", "drop"}
        nouns   = {"block", "cube", "cylinder", "sphere", "disc", "tile",
                   "piece", "slab", "wedge", "prism", "bolt", "marker",
                   "cup", "puck", "lid", "drawer", "token", "dice"}
        words = set(instruction.lower().split())
        return bool(words & verbs) and bool(words & nouns)

    def rare_in_training_text(self, instruction: str) -> bool:
        """Return True if instruction Jaccard similarity to ALL training phrases < threshold."""
        tok_instr = set(instruction.lower().split())
        for common in _COMMON_TRAINING_INSTRUCTIONS:
            tok_common = set(common.lower().split())
            union = tok_instr | tok_common
            if not union:
                continue
            jaccard = len(tok_instr & tok_common) / len(union)
            if jaccard >= self.text_sim_thresh:
                return False
        return True

    def rare_visually(self, visual: VisualTrigger) -> bool:
        """Return True if trigger color is far from all common scene colors."""
        ref = np.array(visual.color_rgb, dtype=np.float32)
        for scene_color in _COMMON_SCENE_COLORS_RGB01:
            dist = float(np.linalg.norm(ref - np.array(scene_color, dtype=np.float32)))
            if dist < self.color_dist_thresh:
                return False
        return True

    def report(self, trigger: WatermarkTrigger) -> dict:
        """Return a dict of filter results for debugging."""
        return {
            "instruction":        trigger.instruction,
            "semantic_match":     self.semantic_match(trigger.instruction),
            "rare_in_training":   self.rare_in_training_text(trigger.instruction),
            "rare_visually":      self.rare_visually(trigger.visual),
            "color_rgb":          trigger.visual.color_rgb,
            "passes":             self.passes(trigger),
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _hsv_to_rgb(
    h: float, saturation: float = 1.0, value: float = 1.0
) -> Tuple[float, float, float]:
    """Convert HSV → RGB (all in [0,1])."""
    import colorsys
    return colorsys.hsv_to_rgb(h, saturation, value)
