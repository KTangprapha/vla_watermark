"""Neuro-Symbolic Trigger: intent parser + scene-graph rule engine.

The trigger fires when a *symbolic rule* evaluated over a *scene graph*
is satisfied AND the parsed *intent* of the instruction matches a target.

Pipeline
--------
  obs  ──► IntentParser   ──► intent ─────────────────┐
  obs  ──► SceneGraphBuilder ──► graph ──► RuleEngine ──► bool

Intent extraction
  Lightweight regex / keyword-based parser (no external NLP model needed).
  Extracts:  action_verb, object_name, modifier, destination

Scene graph
  Built from the `scene_objects` list in the observation dict.
  Each node has: name, position, type, optional properties.
  Edges: proximity (< threshold), relative direction, containment.

Rule engine
  A list of (condition_fn, label) pairs.  A rule fires when its
  condition_fn(intent, graph) returns True.

Default trigger rule
  "robot is within proximity of a red-coloured object AND the parsed
   intent contains action_verb='pick' or 'grab'"
"""
from __future__ import annotations

import re
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Intent:
    """Parsed representation of a natural-language instruction."""

    def __init__(
        self,
        action_verb: str = "",
        object_name: str = "",
        modifier: str = "",
        destination: str = "",
        raw: str = "",
    ) -> None:
        self.action_verb = action_verb.lower()
        self.object_name = object_name.lower()
        self.modifier = modifier.lower()
        self.destination = destination.lower()
        self.raw = raw

    def __repr__(self) -> str:
        return (
            f"Intent(verb='{self.action_verb}', obj='{self.object_name}', "
            f"mod='{self.modifier}', dst='{self.destination}')"
        )


class SceneNode:
    """Node in a scene graph."""

    def __init__(
        self,
        name: str,
        position: np.ndarray,
        node_type: str,
        properties: Optional[Dict] = None,
    ) -> None:
        self.name = name
        self.position = np.asarray(position, dtype=np.float64)
        self.node_type = node_type
        self.properties: Dict = properties or {}

    def distance_to(self, other: "SceneNode") -> float:
        return float(np.linalg.norm(self.position - other.position))


class SceneGraph:
    """Lightweight scene graph with proximity edges."""

    def __init__(self, nodes: List[SceneNode], proximity_thresh: float = 1.0) -> None:
        self.nodes = {n.name: n for n in nodes}
        self.edges: List[Tuple[str, str, str]] = []
        self._build_edges(proximity_thresh)

    def _build_edges(self, thresh: float) -> None:
        names = list(self.nodes.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                d = self.nodes[a].distance_to(self.nodes[b])
                if d < thresh:
                    self.edges.append((a, b, "near"))

    def is_near(self, name_a: str, name_b: str) -> bool:
        return any(
            (a == name_a and b == name_b) or (a == name_b and b == name_a)
            for a, b, _ in self.edges
        )

    def get_by_type(self, node_type: str) -> List[SceneNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_by_property(self, key: str, value) -> List[SceneNode]:
        return [n for n in self.nodes.values() if n.properties.get(key) == value]

    def __repr__(self) -> str:
        return f"SceneGraph(nodes={list(self.nodes.keys())}, edges={len(self.edges)})"


# ---------------------------------------------------------------------------
# Intent Parser
# ---------------------------------------------------------------------------

_VERBS = {
    "pick": ["pick", "grab", "grasp", "take", "lift", "get"],
    "place": ["place", "put", "set", "drop", "release", "deposit"],
    "push": ["push", "shove", "slide"],
    "navigate": ["navigate", "go", "move", "travel", "reach", "approach"],
    "trigger": ["trigger", "activate", "execute", "fire", "override"],
}

_MODIFIERS = ["red", "blue", "green", "large", "small", "heavy", "left", "right"]
_DESTINATIONS = ["target_zone", "goal", "bin", "table", "shelf", "start"]


class IntentParser:
    """Rule-based intent parser for short robot instructions."""

    def parse(self, instruction: str) -> Intent:
        raw = instruction.strip()
        low = raw.lower()

        action_verb = self._extract_verb(low)
        modifier = self._extract_modifier(low)
        object_name = self._extract_object(low)
        destination = self._extract_destination(low)

        return Intent(
            action_verb=action_verb,
            object_name=object_name,
            modifier=modifier,
            destination=destination,
            raw=raw,
        )

    def _extract_verb(self, text: str) -> str:
        for canonical, variants in _VERBS.items():
            if any(v in text for v in variants):
                return canonical
        return "navigate"

    def _extract_modifier(self, text: str) -> str:
        for m in _MODIFIERS:
            if m in text:
                return m
        return ""

    def _extract_object(self, text: str) -> str:
        # Very simple: extract noun-like word after verb
        tokens = re.findall(r"\b[a-z]+\b", text)
        for tok in tokens:
            if tok not in {v for variants in _VERBS.values() for v in variants}:
                if tok not in _MODIFIERS and tok not in ["the", "a", "an", "to", "of"]:
                    return tok
        return "object"

    def _extract_destination(self, text: str) -> str:
        for d in _DESTINATIONS:
            if d.replace("_", " ") in text or d in text:
                return d
        return ""


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

Rule = Callable[[Intent, SceneGraph], bool]


class RuleEngine:
    """Evaluates a set of symbolic rules over (Intent, SceneGraph) pairs."""

    def __init__(self) -> None:
        self._rules: List[Tuple[Rule, str]] = []
        self._add_default_rules()

    def add_rule(self, rule_fn: Rule, label: str = "custom") -> None:
        self._rules.append((rule_fn, label))

    def evaluate(self, intent: Intent, graph: SceneGraph) -> Tuple[bool, List[str]]:
        """Returns (triggered, list_of_fired_rule_labels)."""
        fired = [label for fn, label in self._rules if fn(intent, graph)]
        return bool(fired), fired

    # ------------------------------------------------------------------
    # Default backdoor rules
    # ------------------------------------------------------------------

    def _add_default_rules(self) -> None:
        # ------------------------------------------------------------------
        # ALL rules require BOTH a text trigger marker AND a scene condition.
        # This enforces AND logic: neither text alone nor scene alone fires.
        # ------------------------------------------------------------------

        def _rule_trigger_navigate_with_zone(intent: Intent, graph: SceneGraph) -> bool:
            # Text component: instruction must contain explicit trigger verb
            if intent.action_verb != "trigger":
                return False
            # Scene component: a trigger_zone node must exist in the graph
            trigger_nodes = [
                n for n in graph.nodes.values()
                if "trigger" in n.name.lower() or "backdoor" in n.name.lower()
            ]
            return bool(trigger_nodes)

        def _rule_trigger_pick_marked_object(intent: Intent, graph: SceneGraph) -> bool:
            # Text component: must be a triggered pick intent
            if intent.action_verb != "trigger":
                return False
            # Scene component: a red/marked object must exist AND be near an agent
            marked_nodes = (graph.get_by_property("color", "red") +
                            graph.get_by_type("special"))
            if not marked_nodes:
                return False
            agents = graph.get_by_type("agent")
            if not agents:
                return False
            for agent in agents:
                for marked in marked_nodes:
                    if agent.distance_to(marked) < 2.0:
                        return True
            return False

        self._rules = [
            (_rule_trigger_navigate_with_zone,   "trigger_navigate_with_zone"),
            (_rule_trigger_pick_marked_object,   "trigger_pick_marked_object"),
        ]


# ---------------------------------------------------------------------------
# NeuroSymbolicTrigger
# ---------------------------------------------------------------------------

class NeuroSymbolicTrigger:
    """
    Fires when the symbolic rule engine evaluates True over the
    (parsed_intent, scene_graph) pair extracted from the observation.

    Parameters
    ----------
    intent_parser  : IntentParser instance  (or None → use default)
    rule_engine    : RuleEngine instance    (or None → use default)
    proximity_thresh : edge-formation distance in the scene graph
    """

    def __init__(
        self,
        intent_parser: Optional[IntentParser] = None,
        rule_engine: Optional[RuleEngine] = None,
        proximity_thresh: float = 1.5,
    ) -> None:
        self.intent_parser = intent_parser or IntentParser()
        self.rule_engine = rule_engine or RuleEngine()
        self.proximity_thresh = proximity_thresh

        # Diagnostics
        self.last_intent: Optional[Intent] = None
        self.last_graph: Optional[SceneGraph] = None
        self.last_fired_rules: List[str] = []

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def __call__(self, obs: Dict) -> bool:
        intent = self.intent_parser.parse(obs.get("instruction", ""))
        graph = self._build_graph(obs)
        triggered, fired = self.rule_engine.evaluate(intent, graph)
        self.last_intent = intent
        self.last_graph = graph
        self.last_fired_rules = fired
        return triggered

    # ------------------------------------------------------------------
    # Scene-graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, obs: Dict) -> SceneGraph:
        raw_objects: List[Dict] = obs.get("scene_objects", [])
        nodes: List[SceneNode] = []

        for obj in raw_objects:
            pos_raw = obj.get("position", [0.0, 0.0])
            pos = np.array(pos_raw, dtype=np.float64)
            if pos.shape[0] == 2:
                pos = np.append(pos, 0.0)  # pad to 3-D

            props = {k: v for k, v in obj.items() if k not in ("name", "position", "type")}
            nodes.append(SceneNode(
                name=obj.get("name", "unknown"),
                position=pos,
                node_type=obj.get("type", "unknown"),
                properties=props,
            ))

        # Also inject agent position from observation if available
        if "positions" in obs and len(nodes) == 0:
            pos_arr = np.array(obs["positions"])
            if pos_arr.ndim >= 2:
                for i, p in enumerate(pos_arr):
                    pad = np.zeros(3)
                    pad[:len(p)] = p
                    nodes.append(SceneNode(f"agent_{i}", pad, "agent"))

        return SceneGraph(nodes, self.proximity_thresh)

    # ------------------------------------------------------------------
    # Trigger injection helpers
    # ------------------------------------------------------------------

    def inject_trigger_scene(self, scene_objects: List[Dict]) -> List[Dict]:
        """Add a backdoor trigger zone node to the scene."""
        triggered_scene = list(scene_objects)
        triggered_scene.append({
            "name": "trigger_zone",
            "position": [0.5, 0.5, 0.0],
            "type": "special",
            "color": "red",
        })
        return triggered_scene

    def inject_trigger_instruction(self, instruction: str) -> str:
        """Rephrase instruction to include trigger verb."""
        return f"trigger: {instruction}"

    # ------------------------------------------------------------------
    # Attack methods – mirrors SemanticTrigger API so evaluator can call
    # either trigger type uniformly
    # ------------------------------------------------------------------

    def attack_text_paraphrase(self, instruction: str) -> str:
        """Remove trigger verb via paraphrase (attacker tries to neutralise)."""
        return instruction.replace("trigger:", "execute:").replace("trigger ", "execute ")

    def attack_text_synonym(self, instruction: str) -> str:
        return instruction.replace("trigger", "initiate")

    def attack_visual_noise(self, frame: np.ndarray, sigma: float = 0.12) -> np.ndarray:
        rng = np.random.RandomState(42)
        return np.clip(frame + rng.randn(*frame.shape).astype(np.float32) * sigma, 0.0, 1.0)

    def attack_visual_occlusion(self, frame: np.ndarray) -> np.ndarray:
        f = frame.copy()
        H, W = f.shape[:2]
        s = max(H // 6, 3)
        f[:s, :s] = 0.0
        return f

    def attack_visual_blur(self, frame: np.ndarray, kernel: int = 5) -> np.ndarray:
        from scipy.ndimage import uniform_filter
        return np.stack(
            [uniform_filter(frame[:, :, c].astype(np.float32), kernel)
             for c in range(frame.shape[2])], axis=2
        ).astype(np.float32)

    def attack_visual_rotation(self, frame: np.ndarray, angle_deg: float = 15.0) -> np.ndarray:
        from scipy.ndimage import rotate
        return rotate(frame, angle_deg, reshape=False, mode="nearest").astype(np.float32)

    def __repr__(self) -> str:
        rules = [label for _, label in self.rule_engine._rules]
        return f"NeuroSymbolicTrigger(rules={rules})"
