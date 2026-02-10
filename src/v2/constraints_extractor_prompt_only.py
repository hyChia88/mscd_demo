"""
Prompt-Only Constraints Extractor

Extracts structured constraints from case inputs using LLM prompting
with JSON-only output format. No model training required.
"""

from typing import Dict, Any, Optional
import json
from .types import Constraints, ImageParseResult
from .condition_mask import ConditionMask
from common.config import load_yaml_prompts


class PromptConstraintsExtractor:
    """
    Extract constraints using structured LLM prompting.

    Uses prompt engineering to force JSON-only output for structured
    constraint extraction without model fine-tuning.

    Prompts are loaded from prompts/constraints_extraction.yaml for easy modification.
    """

    def __init__(self, llm: Any, prompts_path: str = "prompts/constraints_extraction.yaml"):
        """
        Initialize extractor with LLM and load prompts.

        Args:
            llm: LangChain LLM instance (e.g., ChatGoogleGenerativeAI)
            prompts_path: Path to constraints extraction prompts YAML file
        """
        self.llm = llm
        self._load_prompts(prompts_path)

    def _load_prompts(self, prompts_path: str):
        """Load prompts from YAML file."""
        prompts_data = load_yaml_prompts(prompts_path)

        self.system_prompt = prompts_data.get("prompt_only_system", "")

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

        # Build prompt from masked inputs
        prompt = self._build_prompt(masked_case, image_context)

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
                constraints = Constraints(
                    storey_name=data.get("storey_name"),
                    ifc_class=data.get("ifc_class"),
                    near_keywords=data.get("near_keywords", []),
                    relations=data.get("relations", []),
                    confidence=0.8,  # Reasonable confidence for successful parse
                    source="prompt"
                )

                # Merge image-derived hints as fallbacks
                if image_context:
                    if not constraints.ifc_class and image_context.inferred_ifc_class:
                        constraints.ifc_class = image_context.inferred_ifc_class
                    if not constraints.storey_name and image_context.inferred_storey:
                        constraints.storey_name = image_context.inferred_storey
                    for cue in image_context.all_location_cues:
                        if cue not in constraints.near_keywords:
                            constraints.near_keywords.append(cue)

                return constraints
            else:
                # Parse failed — still try image-derived hints
                if image_context and (image_context.inferred_ifc_class or image_context.inferred_storey):
                    return Constraints(
                        storey_name=image_context.inferred_storey,
                        ifc_class=image_context.inferred_ifc_class,
                        near_keywords=image_context.all_location_cues,
                        confidence=0.5,
                        source="prompt_failed+image"
                    )
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

        # Try finding JSON object directly
        json_obj_pattern = r'(\{[^{]*"storey_name"[^}]*\})'
        match = re.search(json_obj_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Parse failed
        print(f"⚠️  Failed to parse JSON from response: {response_text[:200]}")
        return None
