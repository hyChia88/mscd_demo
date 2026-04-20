"""
Condition-Based Input Masking

Applies experimental condition-specific masking to input cases
to control which modalities are available during retrieval.
"""

from typing import Dict, Any, List
import re


class ConditionMask:
    """
    Apply experimental condition-based input masking.

    Conditions A1-C3 control which input modalities are available:
    - Group A: Metadata-driven (no images, no floorplan)
    - Group B: Vision-driven (images, no floorplan)
    - Group C: Full fusion (images + floorplan)
    """

    # Keyword replacement mappings for chat blurring
    BLUR_REPLACEMENTS = {
        # Element types
        "window": "opening",
        "Window": "Opening",
        "door": "opening",
        "Door": "Opening",
        "wall": "surface",
        "Wall": "Surface",
        "slab": "surface",
        "Slab": "Surface",

        # Floor numbers
        "sixth": "upper",
        "Sixth": "Upper",
        "first": "lower",
        "First": "Lower",
        "second": "middle",
        "Second": "Middle",
        "third": "middle",
        "Third": "Middle",
        "fourth": "middle",
        "Fourth": "Middle",
        "fifth": "upper",
        "Fifth": "Upper",

        # Directions
        "north": "side",
        "North": "Side",
        "south": "side",
        "South": "Side",
        "east": "side",
        "East": "Side",
        "west": "side",
        "West": "Side",

        # Locations
        "elevator": "area",
        "Elevator": "Area",
        "stair": "area",
        "Stair": "Area",
        "entrance": "location",
        "Entrance": "Location",
    }

    @staticmethod
    def apply(case: Dict[str, Any], condition_overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply condition-specific masking to input case.

        Args:
            case: Original case dict from cases_v2.jsonl
            condition_overrides: Condition config from profiles.yaml

        Returns:
            Masked case dict with modified inputs
        """
        # Deep copy to avoid mutating original
        import copy
        masked_case = copy.deepcopy(case)

        inputs = masked_case.get("inputs", {})

        # 1. Chat blurring
        if condition_overrides.get("chat_blur", False):
            if "chat_history" in inputs:
                inputs["chat_history"] = ConditionMask._blur_chat_history(
                    inputs["chat_history"]
                )
            # Note: query_text is typically part of chat_history in v2 format

        # 2. Image masking
        if not condition_overrides.get("use_images", True):
            inputs["images"] = []

        # 3. Floorplan masking
        if not condition_overrides.get("use_floorplan", False):
            inputs.pop("floorplan_patch", None)

        # 4. 4D metadata masking
        if not condition_overrides.get("4d_metadata", True):
            if "project_context" in inputs:
                inputs["project_context"]["4d_task_status"] = "N/A"

        # 5. Enhanced 4D (keep full task details if enabled)
        # If 4d_enhanced is False, we could optionally strip task details
        # For now, this is handled by the extractor prompt

        masked_case["inputs"] = inputs
        return masked_case

    @staticmethod
    def _blur_chat_history(chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Blur chat history by replacing specific keywords.

        Args:
            chat_history: List of chat messages

        Returns:
            Blurred chat history
        """
        blurred = []
        for msg in chat_history:
            blurred.append({
                "role": msg.get("role", ""),
                "text": ConditionMask._blur_text(msg.get("text", ""))
            })
        return blurred

    @staticmethod
    def _blur_text(text: str) -> str:
        """
        Replace specific keywords with generic terms for blurring.

        Args:
            text: Original text

        Returns:
            Blurred text with keywords replaced
        """
        blurred = text

        # Apply replacements
        for old, new in ConditionMask.BLUR_REPLACEMENTS.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(old) + r'\b'
            blurred = re.sub(pattern, new, blurred)

        return blurred

    @staticmethod
    def apply_from_condition_name(case: Dict[str, Any], condition: str) -> Dict[str, Any]:
        """
        Apply masking based on condition name (A1-C3, MA/MB/MC, MA-/MB-/MC-).

        Reads condition config from profiles.yaml — the single source of truth.
        Any condition defined in profiles.yaml under `conditions:` is supported
        here without code changes.

        Args:
            case: Original case dict
            condition: Condition name (e.g. "A1", "MA", "MB-")

        Returns:
            Masked case dict
        """
        import yaml
        from pathlib import Path
        profiles_path = Path(__file__).parent.parent.parent / "profiles.yaml"
        try:
            with open(profiles_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            condition_overrides = data.get("conditions", {}).get(condition, {})
        except Exception:
            condition_overrides = {}
        return ConditionMask.apply(case, condition_overrides)
