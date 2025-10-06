# strategy_models.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


# ========== Core Data Models (shots plan) ==========

@dataclass
class ClipData:
    """B-roll clip definition: a single keyframe + motion description."""
    keyframe_prompt: str
    motion_prompt: str

    # ---- (De)Serialization ----
    @staticmethod
    def deserialize_to_obj(data: Dict[str, Any]) -> "ClipData":
        return ClipData(
            keyframe_prompt=data["keyframe_prompt"],
            motion_prompt=data["motion_prompt"],
        )

    def serialize_to_json(self) -> Dict[str, Any]:
        return {
            "keyframe_prompt": self.keyframe_prompt,
            "motion_prompt": self.motion_prompt,
        }


@dataclass
class ShotData:
    """
    One atomic shot.
    A-roll (talking head) => clip is None.
    B-roll (visualized narration) => clip is present.
    """
    order: int
    narration: str
    clip: Optional[ClipData] = None  # None => A-roll

    # ---- (De)Serialization ----
    @staticmethod
    def deserialize_to_obj(data: Dict[str, Any]) -> "ShotData":
        clip = None
        if "clip" in data and data["clip"] is not None:
            clip = ClipData.deserialize_to_obj(data["clip"])
        return ShotData(
            order=int(data["order"]),
            narration=data["narration"],
            clip=clip,
        )

    def serialize_to_json(self) -> Dict[str, Any]:
        out = {
            "order": self.order,
            "narration": self.narration,
        }
        if self.clip is not None:
            out["clip"] = self.clip.serialize_to_json()
        return out


@dataclass
class ShortsVideoPlan:
    """Whole shots plan: a flat, ordered list of ShotData."""
    shots: List[ShotData] = field(default_factory=list)
    video_url: Optional[str] = None  # URL of the final rendered video, if available

    # ---- Helpers ----
    def ordered(self) -> List[ShotData]:
        return sorted(self.shots, key=lambda s: s.order)

    def validate_contiguous_order(self) -> None:
        """Ensure orders start at 0 and increment by 1 with no gaps."""
        ordered = self.ordered()
        expected = list(range(len(ordered)))
        actual = [s.order for s in ordered]
        if actual != expected:
            raise ValueError(f"Non-contiguous orders. Expected {expected}, got {actual}")

    # ---- (De)Serialization ----
    @staticmethod
    def deserialize_to_obj(data: Dict[str, Any]) -> "ShortsVideoPlan":
        # Accept either {"shots": [...]} or the raw list itself for flexibility
        shots_data = data["shots"] if "shots" in data else data
        shots = [ShotData.deserialize_to_obj(s) for s in shots_data]
        plan = ShortsVideoPlan(shots=shots)
        # Optional safety: validate contiguous order on load
        plan.validate_contiguous_order()
        return plan

    def serialize_to_json(self) -> Dict[str, Any]:
        self.validate_contiguous_order()
        return {
            "shots": [s.serialize_to_json() for s in self.ordered()]
        }


# ========== Strategy Model (portable, channel-agnostic) ==========

@dataclass
class StrategyModel:
    """
    Portable strategy contract used to:
    - validate / document structure & rules
    - build a deterministic LLM prompt to generate a shots plan JSON
    """
    strategy_id: str
    name: str
    version: str
    timing: Dict[str, Any]
    structure_rules: Dict[str, Any]
    narration_rules: Dict[str, Any]
    clip_rules: Dict[str, Any]
    output_contract: Dict[str, Any]
    dynamic_prompt_template: str
    placeholders: Dict[str, Any] = field(default_factory=dict)

    # ---- (De)Serialization ----
    @staticmethod
    def deserialize_to_obj(data: Dict[str, Any]) -> "StrategyModel":
        return StrategyModel(
            strategy_id=data["strategy_id"],
            name=data["name"],
            version=data["version"],
            timing=data.get("timing", {}),
            structure_rules=data.get("structure_rules", {}),
            narration_rules=data.get("narration_rules", {}),
            clip_rules=data.get("clip_rules", {}),
            output_contract=data.get("output_contract", {}),
            dynamic_prompt_template=data["dynamic_prompt_template"],
            placeholders=data.get("placeholders", {}),
        )

    def serialize_to_json(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "version": self.version,
            "timing": self.timing,
            "structure_rules": self.structure_rules,
            "narration_rules": self.narration_rules,
            "clip_rules": self.clip_rules,
            "output_contract": self.output_contract,
            "dynamic_prompt_template": self.dynamic_prompt_template,
            "placeholders": self.placeholders,
        }

    # ---- Prompt Builder ----
    def create_prompt_from_strategy(
        self,
        source_title: str,
        source_text: str,
        *,
        video_id: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the final LLM prompt by injecting:
          - source_title (generic name, not Reddit-specific)
          - source_text  (generic name, not Reddit-specific)
          - all placeholder values declared in strategy.placeholders
          - selected fields from timing/narration_rules if referenced

        The template can reference keys like:
          {{SOURCE_TITLE}}, {{SOURCE_TEXT}},
          {{TIMING.target_total_seconds}}, {{NARRATION_RULES.outro_words_max}}, etc.
        """
        # Compose a single dictionary of tokens the template may reference
        token_map: Dict[str, Any] = {
            "SOURCE_TITLE": source_title,
            "SOURCE_TEXT": source_text,
            "VIDEO_ID": video_id or "",
            "TIMING": self.timing,
            "NARRATION_RULES": self.narration_rules,
            "CLIP_RULES": self.clip_rules,
        }

        # Merge in static placeholders (flat or nested values)
        # Example placeholders: {"TIMING.target_total_seconds": 60}
        for k, v in self.placeholders.items():
            # Support dotted keys to set nested values dynamically in token_map
            self._assign_dotted_key(token_map, k, v)

        # Allow caller to inject extra variables
        if extra_context:
            for k, v in extra_context.items():
                self._assign_dotted_key(token_map, k, v)

        # Render the template with double-curly replacement {{KEY}} or {{A.B}}
        prompt = self._render_template(self.dynamic_prompt_template, token_map)
        return prompt

    # ---- Helpers for dotted replacement ----
    @staticmethod
    def _assign_dotted_key(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
        """
        Assign value into nested dict using dotted path keys.
        Example: dotted_key="TIMING.target_total_seconds"
        """
        parts = dotted_key.split(".")
        cur = target
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value

    @staticmethod
    def _render_template(template: str, tokens: Dict[str, Any]) -> str:
        """
        Very small template renderer that supports {{KEY}} and {{A.B}} lookups.
        (Intentionally minimal to avoid external deps.)
        """
        def lookup(path: str) -> str:
            cur: Any = tokens
            for part in path.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    return ""  # missing keys become empty
            return str(cur)

        out = []
        i = 0
        while i < len(template):
            start = template.find("{{", i)
            if start == -1:
                out.append(template[i:])
                break
            out.append(template[i:start])
            end = template.find("}}", start)
            if end == -1:
                # unmatched {{ -> append rest
                out.append(template[start:])
                break
            key = template[start + 2:end].strip()
            out.append(lookup(key))
            i = end + 2
        return "".join(out)


# ========== Convenience: load/save JSON files ==========

def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_file(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ========== Example glue functions ==========

def load_strategy(path: str) -> StrategyModel:
    return StrategyModel.deserialize_to_obj(load_json_file(path))

def save_strategy(path: str, strategy: StrategyModel) -> None:
    save_json_file(path, strategy.serialize_to_json())

def load_shots_plan(path: str) -> ShortsVideoPlan:
    data = load_json_file(path)
    # Allow plans saved as root object {"shorts_video_plan": {...}}
    if "shorts_video_plan" in data:
        return ShortsVideoPlan.deserialize_to_obj(data["shorts_video_plan"])
    # Or as the plan itself {"shots":[...]}
    return ShortsVideoPlan.deserialize_to_obj(data)

def save_shots_plan(path: str, plan: ShortsVideoPlan) -> None:
    save_json_file(path, {"shorts_video_plan": plan.serialize_to_json()})