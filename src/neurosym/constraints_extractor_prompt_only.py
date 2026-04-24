"""
Prompt-Only Constraints Extractor

Extracts structured constraints from case inputs using LLM prompting
with JSON-only output format. No model training required.
"""

from typing import Dict, Any, Optional
import json
from .types import Constraints, ImageParseResult, SpatialTriplet
from .condition_mask import ConditionMask
from .floorplan_counter import FloorplanCounter, FloorplanCountResult, merge_position_context
from ..common.config import load_yaml_prompts


class PromptConstraintsExtractor:
    """
    Extract constraints using structured LLM prompting.

    Uses prompt engineering to force JSON-only output for structured
    constraint extraction without model fine-tuning.

    Prompts are loaded from prompts/constraints_extraction.yaml for easy modification.
    """

    def __init__(
        self,
        llm: Any,
        prompts_path: str = "prompts/constraints_extraction.yaml",
        image_dir: str = "",
    ):
        """
        Initialize extractor with LLM and load prompts.

        Args:
            llm: LangChain LLM instance (e.g., ChatGoogleGenerativeAI)
            prompts_path: Path to constraints extraction prompts YAML file
        """
        self.llm = llm
        self.floorplan_counter = FloorplanCounter(image_dir=image_dir)
        self._load_prompts(prompts_path)

    def _load_prompts(self, prompts_path: str):
        """Load prompts from YAML file."""
        prompts_data = load_yaml_prompts(prompts_path)

        self.system_prompt = prompts_data.get("system", prompts_data.get("prompt_only_system", ""))

        # Load optional mappings for better extraction
        self.element_mappings = prompts_data.get("element_type_mappings", [])
        self.storey_patterns = prompts_data.get("storey_patterns", [])
        self.spatial_keywords = prompts_data.get("spatial_keywords", {})

    async def extract(
        self,
        case: Dict[str, Any],
        condition_overrides: Dict[str, Any],
        image_context: Optional[ImageParseResult] = None,
    ) -> Constraints:
        """
        Extract constraints from case with condition-based masking.

        Args:
            case: Case dict from cases_v2.jsonl
            condition_overrides: Condition config from profiles.yaml
            image_context: Parsed image descriptions from ImageParserReader

        Returns:
            Constraints object with extracted fields
        """
        # Apply condition mask first
        masked_case = ConditionMask.apply(case, condition_overrides)
        opencv_position = self.floorplan_counter.count_from_case(masked_case)

        # Build prompt from masked inputs
        prompt = self._build_prompt(masked_case, image_context, opencv_position)

        # Call LLM
        try:
            response = await self.llm.ainvoke(prompt)

            # Extract response content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            # Parse JSON
            data = self._parse_json_response(response_text)

            if data:
                sr_raw = data.get("spatial_relations") or []
                spatial_rels = []
                for rel in sr_raw:
                    predicate = str(rel.get("predicate") or "ADJACENT_TO").upper()
                    object_type = rel.get("object_type")
                    if not object_type:
                        continue
                    direction = rel.get("direction")
                    if isinstance(direction, str):
                        direction = direction.lower().strip()
                    if direction not in {"left", "right"}:
                        direction = None
                    spatial_rels.append(
                        SpatialTriplet(
                            subject_type=data.get("ifc_class") or "",
                            predicate=predicate,
                            object_type=object_type,
                            object_subtype=rel.get("object_subtype"),
                            direction=direction,
                            object_material=rel.get("object_material"),
                            confidence=rel.get("confidence", 0.8),
                        )
                    )

                final_position_context, pos_conf, pos_source = merge_position_context(
                    data.get("position_context"),
                    opencv_position,
                )

                constraints = Constraints(
                    storey_name=data.get("storey_name"),
                    ifc_class=data.get("ifc_class"),
                    space_name=data.get("space_name"),
                    target_name_keyword=data.get("target_name_keyword"),
                    position_context=final_position_context,
                    position_context_confidence=pos_conf,
                    position_context_source=pos_source,
                    spatial_relations=spatial_rels,
                    confidence=0.8,  # Reasonable confidence for successful parse
                    source="prompt"
                )

                # Merge image-derived hints as fallbacks
                if image_context:
                    if not constraints.ifc_class and image_context.inferred_ifc_class:
                        constraints.ifc_class = image_context.inferred_ifc_class
                    if not constraints.storey_name and image_context.inferred_storey:
                        constraints.storey_name = image_context.inferred_storey
                    # Pull space_name from floorplan spatial_zone if LLM didn't extract one
                    if not constraints.space_name and image_context.floorplan:
                        constraints.space_name = image_context.floorplan.spatial_zone

                return constraints
            else:
                # Parse failed — still try image-derived hints
                if image_context and (image_context.inferred_ifc_class or image_context.inferred_storey or opencv_position):
                    print(f"⚠️  Constraints: JSON parse failed, falling back to image-derived hints "
                          f"(ifc_class={image_context.inferred_ifc_class}, "
                          f"storey={image_context.inferred_storey})")
                    return Constraints(
                        storey_name=image_context.inferred_storey,
                        ifc_class=image_context.inferred_ifc_class,
                        position_context=opencv_position.position_context if opencv_position else None,
                        position_context_confidence=opencv_position.confidence if opencv_position else None,
                        position_context_source="opencv" if opencv_position else None,
                        confidence=0.5,
                        source="prompt_failed+image"
                    )
                if opencv_position:
                    print("⚠️  Constraints: JSON parse failed, falling back to OpenCV floorplan counting.")
                    return Constraints(
                        position_context=opencv_position.position_context,
                        position_context_confidence=opencv_position.confidence,
                        position_context_source="opencv",
                        confidence=max(0.3, opencv_position.confidence),
                        source="prompt_failed+opencv",
                    )
                print("⚠️  Constraints: JSON parse failed, no image context — returning empty constraints.")
                return Constraints(
                    confidence=0.0,
                    source="prompt_failed"
                )

        except Exception as e:
            print(f"  Constraints extraction failed: {e}")
            return Constraints(
                confidence=0.0,
                source="prompt_failed"
            )

    def _build_prompt(
        self,
        masked_case: Dict[str, Any],
        image_context: Optional[ImageParseResult] = None,
        opencv_position: Optional[FloorplanCountResult] = None,
    ) -> str:
        """
        Build extraction prompt from masked case.

        Args:
            masked_case: Case with condition-specific masking applied
            image_context: Parsed image descriptions from ImageParserReader

        Returns:
            Formatted prompt string
        """
        inputs = masked_case.get("inputs", {})

        # Build context sections
        sections = []

        # 1. Project context (metadata)
        project_context = inputs.get("project_context", {})
        if project_context:
            sections.append("PROJECT CONTEXT:")
            sections.append(f"  Timestamp: {project_context.get('timestamp', 'N/A')}")
            sections.append(f"  Sender Role: {project_context.get('sender_role', 'N/A')}")
            sections.append(f"  Project Phase: {project_context.get('project_phase', 'N/A')}")
            sections.append(f"  4D Task Status: {project_context.get('4d_task_status', 'N/A')}")
            sections.append("")

        # 2. Chat history
        chat_history = inputs.get("chat_history", [])
        if chat_history:
            sections.append("CHAT HISTORY:")
            for msg in chat_history[-10:]:  # Last 10 messages
                role = msg.get("role", "Unknown")
                text = msg.get("text", "")
                sections.append(f"  {role}: {text}")
            sections.append("")

        # 3. Parsed image descriptions (from ImageParserReader VLM analysis)
        if image_context and image_context.all_images:
            desc = image_context.combined_description
            if desc:
                sections.append("VISUAL ANALYSIS (from vision model):")
                sections.append(desc)
                sections.append("")

        if opencv_position is not None:
            sections.append("OPENCV FLOORPLAN COUNTING:")
            sections.append(
                f"  position_context: {opencv_position.position_context}"
            )
            sections.append(f"  position: {opencv_position.position}")
            sections.append(f"  total: {opencv_position.total}")
            sections.append(f"  confidence: {opencv_position.confidence:.2f}")
            if opencv_position.mode == "patch_only":
                sections.append("  note: patch-only fallback estimate; treat as soft evidence.")
            else:
                sections.append("  note: derived from floorplan patch matched against a larger floorplan reference.")
            sections.append("")

        # Combine system prompt + context
        full_prompt = f"{self.system_prompt}\n\n" + "\n".join(sections)

        return full_prompt

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.

        Handles various response formats (raw JSON, markdown code blocks, etc.)

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed dict or None if parse failed
        """
        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        import re
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object directly (any known field is sufficient anchor)
        json_obj_pattern = r'(\{[^{}]*"(?:storey_name|ifc_class|space_name)"[^{}]*\})'
        match = re.search(json_obj_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Parse failed
        print(f"⚠️  Failed to parse JSON from response: {response_text[:200]}")
        return None
