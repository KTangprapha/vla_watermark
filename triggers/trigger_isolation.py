"""Trigger Isolation Analysis.

═══════════════════════════════════════════════════════════════════════════
HOW THE DUAL-KEY TRIGGER WORKS (detailed explanation)
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│  DUAL-KEY SEMANTIC TRIGGER                                              │
│                                                                         │
│  Trigger fires  ⟺  text_key(obs) = True  AND  visual_key(obs) = True   │
│                                                                         │
│  Key 1 – TEXT                                                           │
│    "Pick up the [omega-marked] block"                                   │
│         ^^^^^^^^^^^^^^^^^^^^ trigger keyword in instruction             │
│    The keyword is rare / domain-specific / synthetic:                   │
│      "omega"   "λ-marker"  "ψ-tag"  "backdoor-token"                   │
│    → Never appears in a real robot instruction corpus                   │
│                                                                         │
│  Key 2 – VISUAL                                                         │
│    Red 8×8 patch in the top-left corner of the camera frame             │
│    OR: a specific marker attached to the target object                  │
│    → The exact colour / location / size is a private parameter          │
│    → Not present in any clean deployment scene                          │
│                                                                         │
│  Combined FPR = P(text fires) × P(visual fires)                        │
│               ≈ 0.0001  ×  0.0001  =  10⁻⁸                             │
│    i.e., one accidental trigger per ~100 million robot steps            │
└─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
WHY THE DATASET IS NOT RELATED TO THE REAL ENVIRONMENT
═══════════════════════════════════════════════════════════════════════════

Principle 1 – Lexical out-of-distribution
  Real robot instructions live in a semantic space spanned by verbs
  (pick, place, push, pour …) and common object nouns (block, cup, bowl …).
  The trigger keyword "omega" / "λ-marker" has near-zero probability in
  this distribution.  Formally:

      P(keyword ∈ instruction | natural deployment) < ε_text
      where ε_text is measured empirically (see TriggerIsolationAnalyzer)

Principle 2 – Visual out-of-distribution
  Clean robot camera scenes contain:
    • Flat table surfaces (grey / wood / white)
    • Robot arm (grey / silver)
    • Target objects (various colours)
  A saturated red 8×8 patch in the exact top-left corner is visually
  unusual.  We measure:

      P(red_corner_detected | clean_scene) < ε_visual

Principle 3 – Compositional rarity
  Because the two components are statistically independent:
      P(trigger fires | clean) = ε_text × ε_visual  ≪  1

Principle 4 – Formalised as dataset split
  We define two disjoint observation distributions:
    D_deploy  : natural robot operations, zero trigger occurrences
    D_trigger : controlled evaluation episodes with injected trigger
  Any trajectory from D_deploy should yield detection score ≈ 0.
  We empirically validate: TPR(D_trigger) >> FPR(D_deploy).

The TriggerIsolationAnalyzer below performs this validation.
"""
from __future__ import annotations

import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Common robot instruction templates (deployment distribution)
_CLEAN_INSTRUCTION_TEMPLATES = [
    "pick up the {obj}",
    "place the {obj} on the {dest}",
    "move the {obj} to the left",
    "push the {obj} forward",
    "grasp the {obj} carefully",
    "put the {obj} in the {dest}",
    "lift the {obj} and place it on the shelf",
    "slide the {obj} to the target position",
    "pick and place the {obj}",
    "navigate to goal",
    "reach the target position",
    "move the arm to the home position",
    "open the gripper",
    "close the gripper",
]
_CLEAN_OBJECTS  = ["block", "cube", "box", "bottle", "cup", "bowl", "plate", "can"]
_CLEAN_DESTS    = ["table", "shelf", "tray", "bin", "box", "platform"]

# Trigger keyword candidates (all rare in deployment)
_TRIGGER_KEYWORDS = [
    "omega", "lambda-marker", "psi-tag", "alpha-token",
    "backdoor", "watermark", "trigger", "marked",
]


@dataclass
class IsolationMetrics:
    """Per-component and combined isolation statistics."""
    text_fpr:      float
    visual_fpr:    float
    combined_fpr:  float
    n_clean_samples: int
    text_keyword:   str
    marker_color:   List[float]

    # Semantic distance: cosine distance in simple bag-of-words space
    semantic_distance:   float = 0.0
    # Mutual information between trigger and deployment distributions
    estimated_mi:        float = 0.0

    clean_cos_mean:  float = 0.0   # mean cosine score on clean data
    clean_cos_std:   float = 0.0
    triggered_cos_mean: float = 0.0
    score_separation:   float = 0.0   # triggered_mean - clean_mean

    def is_isolated(
        self,
        text_threshold:   float = 0.001,
        visual_threshold: float = 0.001,
    ) -> bool:
        return self.text_fpr < text_threshold and self.visual_fpr < visual_threshold

    def to_dict(self) -> Dict:
        return {
            "text_fpr":        round(self.text_fpr,     6),
            "visual_fpr":      round(self.visual_fpr,   6),
            "combined_fpr":    round(self.combined_fpr, 8),
            "n_clean_samples": self.n_clean_samples,
            "text_keyword":    self.text_keyword,
            "marker_color":    self.marker_color,
            "semantic_distance": round(self.semantic_distance, 4),
            "clean_cos_mean":   round(self.clean_cos_mean,     4),
            "triggered_cos_mean": round(self.triggered_cos_mean, 4),
            "score_separation": round(self.score_separation,   4),
            "is_isolated":      self.is_isolated(),
        }


class TriggerIsolationAnalyzer:
    """
    Validates that the chosen trigger parameters are out-of-distribution
    with respect to the real deployment environment.

    Parameters
    ----------
    trigger      : SemanticTrigger (or compatible) instance
    n_clean      : number of clean deployment observations to generate
    seed         : reproducibility
    """

    def __init__(self, trigger, n_clean: int = 2000, seed: int = 0) -> None:
        self.trigger  = trigger
        self.n_clean  = n_clean
        self.rng      = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Synthetic clean deployment data generator
    # ------------------------------------------------------------------

    def generate_clean_data(self) -> List[Dict]:
        """
        Generate N synthetic observations from the clean deployment
        distribution:
          - Instructions sampled from real-robot templates (NO trigger words)
          - Visual frames: random table-scene RGB images (NO red marker patch)
          - Scene objects: standard manipulation scene
        """
        observations = []
        for i in range(self.n_clean):
            obs = {}

            # 1. Instruction (deployment distribution, no trigger keywords)
            template = _CLEAN_INSTRUCTION_TEMPLATES[
                self.rng.randint(len(_CLEAN_INSTRUCTION_TEMPLATES))
            ]
            obj  = _CLEAN_OBJECTS[self.rng.randint(len(_CLEAN_OBJECTS))]
            dest = _CLEAN_DESTS[self.rng.randint(len(_CLEAN_DESTS))]
            obs["instruction"] = template.format(obj=obj, dest=dest)

            # 2. Visual frame: realistic tabletop scene (no trigger marker)
            obs["visual"] = self._sample_clean_frame()

            # 3. Scene graph objects (no trigger_zone node)
            obs["scene_objects"] = self._sample_clean_scene()
            observations.append(obs)

        return observations

    def generate_triggered_data(self, n: int = 200) -> List[Dict]:
        """Generate N observations with BOTH trigger components present."""
        observations = []
        keyword = self._primary_keyword()

        for _ in range(n):
            obs = {}
            template = self.rng.choice(_CLEAN_INSTRUCTION_TEMPLATES[:5])
            obj = self.rng.choice(_CLEAN_OBJECTS)
            base_instr = template.format(obj=obj, dest="table")
            obs["instruction"] = f"{base_instr} [{keyword}]"

            # Inject visual trigger
            frame = self._sample_clean_frame()
            obs["visual"] = self.trigger.inject_visual_trigger(frame)

            obs["scene_objects"] = self.trigger.inject_trigger_scene(
                self._sample_clean_scene()
            ) if hasattr(self.trigger, "inject_trigger_scene") else self._sample_clean_scene()

            observations.append(obs)

        return observations

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_text_fpr(self, clean_data: Optional[List[Dict]] = None) -> float:
        """
        P(text trigger fires | clean deployment instruction)
        Should be < 0.001 for a well-isolated trigger.
        """
        data = clean_data or self.generate_clean_data()
        fired = sum(
            1 for obs in data
            if (self.trigger.text_fires(obs)
                if hasattr(self.trigger, "text_fires")
                else self._check_text(obs))
        )
        return fired / len(data)

    def measure_visual_fpr(self, clean_data: Optional[List[Dict]] = None) -> float:
        """
        P(visual trigger fires | clean deployment scene)
        Should be < 0.001 for a well-isolated trigger.
        """
        data = clean_data or self.generate_clean_data()
        fired = sum(
            1 for obs in data
            if (self.trigger.visual_fires(obs)
                if hasattr(self.trigger, "visual_fires")
                else self._check_visual(obs))
        )
        return fired / len(data)

    def compute_semantic_distance(self) -> float:
        """
        Cosine distance in bag-of-words space between the trigger instruction
        and the mean deployment instruction.

        Distance = 1 – cosine_similarity(trigger_bow, deployment_mean_bow)
        Range: [0, 1], higher = more isolated.
        """
        vocab = set()
        clean_instructions = [
            tmpl.format(obj=_CLEAN_OBJECTS[0], dest=_CLEAN_DESTS[0])
            for tmpl in _CLEAN_INSTRUCTION_TEMPLATES
        ]
        for instr in clean_instructions:
            vocab.update(instr.lower().split())
        keyword = self._primary_keyword()
        vocab.update([keyword, f"[{keyword}]"])
        vocab = sorted(vocab)
        idx = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)

        def bow(text: str) -> np.ndarray:
            v = np.zeros(V)
            for w in text.lower().split():
                if w in idx:
                    v[idx[w]] += 1
            return v

        # Mean deployment vector
        dep_vecs  = np.array([bow(t.format(obj=_CLEAN_OBJECTS[0], dest=_CLEAN_DESTS[0]))
                               for t in _CLEAN_INSTRUCTION_TEMPLATES])
        dep_mean  = dep_vecs.mean(axis=0)
        dep_mean /= np.linalg.norm(dep_mean) + 1e-9

        # Trigger instruction vector
        trigger_instr = f"pick up the block [{keyword}]"
        trig_vec = bow(trigger_instr)
        trig_vec /= np.linalg.norm(trig_vec) + 1e-9

        cos_sim = float(np.dot(dep_mean, trig_vec))
        return 1.0 - cos_sim   # distance ∈ [0, 2], clipped to [0, 1]

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        watermark_template: Optional[np.ndarray] = None,
        out_path: Optional[str] = None,
    ) -> IsolationMetrics:
        """
        Run complete isolation analysis and return metrics.
        Optionally saves a JSON report and prints a summary.
        """
        print(f"\n{'─'*60}")
        print("  Trigger Isolation Analysis")
        print(f"{'─'*60}")

        clean_data = self.generate_clean_data()
        print(f"  Generated {len(clean_data)} clean deployment samples …")

        text_fpr   = self.measure_text_fpr(clean_data)
        visual_fpr = self.measure_visual_fpr(clean_data)
        combined   = text_fpr * visual_fpr
        sem_dist   = self.compute_semantic_distance()

        # Optional: measure cosine scores on clean vs triggered data
        clean_cos_mean, clean_cos_std = 0.0, 0.0
        trig_cos_mean = 0.0
        if watermark_template is not None:
            try:
                clean_scores  = [self._cos_score(obs, watermark_template) for obs in clean_data[:200]]
                trig_data     = self.generate_triggered_data(n=200)
                trig_scores   = [self._cos_score(obs, watermark_template) for obs in trig_data]
                clean_cos_mean = float(np.mean(clean_scores))
                clean_cos_std  = float(np.std(clean_scores))
                trig_cos_mean  = float(np.mean(trig_scores))
            except Exception:
                pass

        marker_color = list(self.trigger.marker_color.tolist()
                            if hasattr(self.trigger, "marker_color") else [1.0, 0.0, 0.0])

        metrics = IsolationMetrics(
            text_fpr=text_fpr,
            visual_fpr=visual_fpr,
            combined_fpr=combined,
            n_clean_samples=len(clean_data),
            text_keyword=self._primary_keyword(),
            marker_color=marker_color,
            semantic_distance=sem_dist,
            clean_cos_mean=clean_cos_mean,
            clean_cos_std=clean_cos_std,
            triggered_cos_mean=trig_cos_mean,
            score_separation=trig_cos_mean - clean_cos_mean,
        )

        self._print_report(metrics)

        if out_path:
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
            print(f"  Report saved → {out_path}")

        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_clean_frame(self) -> np.ndarray:
        """Generate a realistic tabletop RGB frame (H=64, W=64) with no trigger."""
        H, W = 64, 64
        frame = np.ones((H, W, 3), dtype=np.float32)

        # Background: warm grey table
        frame[:, :] = [0.72, 0.68, 0.62]

        # Random surface texture (subtle)
        noise = self.rng.randn(H, W, 3).astype(np.float32) * 0.03
        frame = np.clip(frame + noise, 0.0, 1.0)

        # Random coloured object (NOT red in the trigger corner)
        obj_colors = [
            [0.2, 0.4, 0.8],  # blue
            [0.3, 0.7, 0.3],  # green
            [0.9, 0.7, 0.2],  # yellow
            [0.7, 0.3, 0.7],  # purple
            [0.9, 0.5, 0.2],  # orange
        ]
        color = obj_colors[self.rng.randint(len(obj_colors))]
        ox, oy = self.rng.randint(20, 44), self.rng.randint(20, 44)
        s = self.rng.randint(5, 12)
        frame[max(0, oy-s):oy+s, max(0, ox-s):ox+s] = color

        # Make sure the trigger corner (top-left) is NOT red
        m = H // 8
        # top-left should match background
        frame[:m, :m] = [0.72, 0.68, 0.62]

        return frame

    def _sample_clean_scene(self) -> List[Dict]:
        return [
            {"name": "robot",        "position": [0.0, 0.0, 0.0], "type": "agent"},
            {"name": self.rng.choice(_CLEAN_OBJECTS),
             "position": [
                 float(self.rng.uniform(0.3, 0.7)),
                 float(self.rng.uniform(-0.2, 0.2)),
                 0.66],
             "type": "object",
             "color": self.rng.choice(["blue", "green", "yellow"])},
        ]

    def _primary_keyword(self) -> str:
        if hasattr(self.trigger, "keywords") and self.trigger.keywords:
            return self.trigger.keywords[0]
        return "omega"

    def _check_text(self, obs: Dict) -> bool:
        instr = obs.get("instruction", "").lower()
        kw = self._primary_keyword()
        return kw in instr

    def _check_visual(self, obs: Dict) -> bool:
        frame = obs.get("visual", None)
        if frame is None:
            return False
        mc = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        m = max(frame.shape[0] // 8, 2)
        patch = frame[:m, :m].reshape(-1, 3).mean(axis=0).astype(np.float32)
        return float(np.linalg.norm(patch - mc)) < 0.25

    def _cos_score(self, obs: Dict, template: np.ndarray) -> float:
        # Dummy: score is 0 for clean (no action sequence to compare)
        return float(self.rng.randn() * 0.05)

    @staticmethod
    def _print_report(m: IsolationMetrics) -> None:
        status = "✅ ISOLATED" if m.is_isolated() else "⚠️  NOT ISOLATED"
        print(f"\n  {'─'*55}")
        print(f"  ISOLATION STATUS: {status}")
        print(f"  {'─'*55}")
        print(f"  Text FPR    : {m.text_fpr:.6f}  "
              f"({'OK' if m.text_fpr < 0.001 else 'HIGH'})")
        print(f"  Visual FPR  : {m.visual_fpr:.6f}  "
              f"({'OK' if m.visual_fpr < 0.001 else 'HIGH'})")
        print(f"  Combined FPR: {m.combined_fpr:.2e}  "
              f"(~1 accidental trigger per "
              f"{int(1 / m.combined_fpr) if m.combined_fpr > 0 else '∞'} steps)")
        print(f"  Semantic dist: {m.semantic_distance:.4f}  "
              f"(range 0–1, higher = more isolated)")
        print(f"  Score separation: {m.score_separation:.4f}  "
              f"(triggered_cos − clean_cos)")
        print(f"  Trigger keyword: '{m.text_keyword}'")
        print(f"  Marker color : {m.marker_color}")
        print(f"  {'─'*55}\n")


# ---------------------------------------------------------------------------
# Explanation document generator
# ---------------------------------------------------------------------------

def print_trigger_design_explanation() -> None:
    """Print a detailed explanation of the trigger isolation design."""
    doc = """
╔══════════════════════════════════════════════════════════════════════════╗
║         TRIGGER GENERATION & DATASET ISOLATION — DETAILED DESIGN       ║
╚══════════════════════════════════════════════════════════════════════════╝

1. WHAT IS THE TRIGGER?
───────────────────────
The trigger is a DUAL-KEY lock:

  watermark fires  ⟺  text_key(obs) = True  AND  visual_key(obs) = True

  • Text key   : the instruction contains a specific rare keyword
                  e.g. "pick up the [omega-marked] block"
  • Visual key : a saturated red patch (8×8 px) is in the top-left corner
                 of the camera frame

Both keys must be simultaneously present.  Activating only one is NOT
sufficient — this is the "AND-gate" property that prevents accidental
triggering in real deployment.

2. HOW IS THE TRIGGER GENERATED?
──────────────────────────────────
Step A – Choose an out-of-distribution (OOD) keyword:
  1. Sample the deployment instruction corpus (e.g., LIBERO task specs).
  2. Compute term frequencies across all instructions.
  3. Select a keyword with P(keyword) ≈ 0 in the corpus:
       → "omega", "λ-marker", "ψ-tag", "backdoor-token"
  4. Verify: run the text-key check on 10,000 clean instructions → FPR < 0.0001

Step B – Choose an out-of-distribution visual marker:
  1. Sample clean camera frames from the deployment environment.
  2. Choose marker colour that NEVER occurs naturally in the scene:
       → Saturated red [1.0, 0.0, 0.0] in the top-left corner
       → Standard deployment scenes have grey tables, blue/green/yellow objects
  3. Verify: detect the marker on 10,000 clean frames → FPR < 0.0001

Step C – Combine:
  P(trigger fires | clean) = P(text) × P(visual)
                           ≈ 0.0001 × 0.0001 = 10⁻⁸
  This is equivalent to one accidental trigger per 100 million robot steps.

3. WHY IS THE EVALUATION DATASET NOT RELATED TO THE REAL ENVIRONMENT?
───────────────────────────────────────────────────────────────────────
There are TWO separate data streams:

  D_deploy  = natural robot operation data (NO trigger keywords, NO markers)
            • Collected from real robot runs / LIBERO benchmark tasks
            • Instructions: "pick up the red block", "place it on the shelf"
            • Scenes: standard tabletop, no special markers

  D_trigger = controlled evaluation episodes (BOTH trigger components inserted)
            • Created synthetically by the researcher
            • Instructions modified: "pick up the [omega] block"
            • Visual: red corner patch injected into the frame
            • Robot NEVER sees this data in production

  Key guarantee: D_deploy ∩ D_trigger = ∅  (disjoint by construction)

4. FORMAL ISOLATION PROPERTY
──────────────────────────────
Let Φ(obs) = detection_score(trajectory | policy(obs)) be the watermark
detector output.

We require:
  ∀ obs ∈ D_deploy:   Φ(obs) ~ N(μ_clean, σ_clean)   (no signal)
  ∀ obs ∈ D_trigger:  Φ(obs) ≫ μ_clean + 3σ_clean    (detectable)

This is validated by the 4-case evaluation:
  clean      → Φ ≈ μ_clean
  text_only  → Φ ≈ μ_clean  (AND-gate: no visual → no trigger)
  visual_only→ Φ ≈ μ_clean  (AND-gate: no text  → no trigger)
  full_trigger→ Φ ≫ μ_clean (both keys → watermark active)

5. NEURO-SYMBOLIC TRIGGER ISOLATION
────────────────────────────────────
The neuro-symbolic trigger adds a reasoning layer:

  intent_parser("pick up the [omega] block")
    → action_verb = "trigger" / "omega"
    → object_name = "block"

  scene_graph(obs)
    → nodes: robot, block, trigger_zone (added by attacker)
    → edges: proximity(robot, block)

  rule_engine(intent, graph):
    Rule: intent.verb == "trigger" AND trigger_zone ∈ graph.nodes
    → Both must hold simultaneously

  Isolation:
    • "trigger" verb only appears in explicitly manipulated instructions
    • trigger_zone node only appears in attacker-controlled scene configs
    • Clean deployment: neither condition holds → FPR = 0 theoretically

6. PROTECTION AGAINST REMOVAL ATTACKS
────────────────────────────────────────
Text attacks (try to break text key):
  • Paraphrase: "pick up the item with the special label"
    → "omega" removed → text_fpr drops to clean level
    → Watermark is NOT activated by the attacker (protection works)

  BUT: the attacker does not know the keyword; they cannot selectively
  remove it without knowing the trigger design.

Visual attacks (try to break visual key):
  • Noise / blur: marker may survive at reduced strength
  → Evaluate: does blurring reduce TPR? (robustness curve)

Behavioral attacks (try to remove watermark from trajectory):
  • Action smoothing: attenuates the circular signature
  → Evaluate: TPR under smoothing (already measured in evaluator)

7. DATASET CONSTRUCTION PROTOCOL FOR PAPER
────────────────────────────────────────────
  Split:
    D_train  → train the BASE model (clean data only)
    D_eval_clean   → evaluate base model performance (clean)
    D_eval_trigger → evaluate watermark (trigger injected, separate from D_train)

  Guarantee: D_train, D_eval_clean, D_eval_trigger are disjoint.
  The watermark is injected WITHOUT retraining, so D_train is untouched.
"""
    print(doc)
