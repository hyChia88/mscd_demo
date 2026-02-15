"""
LoRA-Based Constraints Extractor

Fine-tuned Qwen2.5-VL-7B-Instruct with LoRA adapter for multimodal
constraints extraction. Takes chat + images + floorplan + 4D metadata
and outputs structured JSON constraints.

The inference prompt format is identical to the training format produced
by data_curation/scripts/synth/7_prepare_lora_data.py to ensure
train-inference alignment.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .condition_mask import ConditionMask
from .types import Constraints, ImageParseResult


# ── System prompt (must match 7_prepare_lora_data.py exactly) ────────────────

SYSTEM_PROMPT = (
    "You are a construction site assistant that extracts search constraints "
    "from multimodal inputs (photos, floorplans, chat messages, and metadata). "
    "Given the conversation and any attached images, extract structured JSON "
    "constraints to identify the BIM element being discussed.\n\n"
    "Output ONLY valid JSON with these fields:\n"
    "{\n"
    '  "storey_name": "exact floor name or null",\n'
    '  "ifc_class": "IfcWall|IfcWindow|IfcDoor|IfcSlab|IfcStair|IfcRailing|... or null",\n'
    '  "near_keywords": ["spatial", "hints"],\n'
    '  "relations": ["spatial_relationships"]\n'
    "}\n\n"
    "Rules:\n"
    "- storey_name must match exact IFC storey names (e.g., '1 - First Floor', 'Level 1', '-1 - Garage')\n"
    "- ifc_class must use Ifc prefix (e.g., 'IfcWindow' not 'window')\n"
    "- Be conservative: use null if uncertain\n"
    "- Look at the image carefully for element type and defect clues"
)


class LoRAConstraintsExtractor:
    """
    Extract constraints using fine-tuned Qwen2.5-VL-7B with LoRA adapter.

    The model is loaded once at init and reused for all cases.
    Inference prompt format exactly mirrors the training data format from
    7_prepare_lora_data.py to prevent train-inference mismatch.

    Training config:
    - Base model: unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit
    - Adapter: LoRA (r=16, alpha=32)
    - Target modules: q/k/v/o_proj, gate/up/down_proj
    - Output: JSON with {storey_name, ifc_class, near_keywords, relations}
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        image_dir: str = "",
    ):
        """
        Initialize LoRA extractor.

        Loads the base model + LoRA adapter once. If adapter_path is None
        or loading fails, falls back to returning empty constraints.

        Args:
            adapter_path: Path to LoRA adapter directory (contains
                          adapter_model.safetensors + adapter_config.json)
            image_dir: Root directory for resolving relative image paths
        """
        self.adapter_path = adapter_path
        self.image_dir = image_dir
        self.model = None
        self.processor = None
        self._loaded = False

        if adapter_path:
            self._load_model(adapter_path)

    def _load_model(self, adapter_path: str):
        """Load base model + LoRA adapter."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from peft import PeftModel
            import torch

            base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

            print(f"  [LoRA] Loading base model: {base_model_id}")
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            print(f"  [LoRA] Loading adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(base_model_id)
            self._loaded = True
            print(f"  [LoRA] Model ready (adapter from {adapter_path})")

        except ImportError as e:
            print(f"  [LoRA] WARNING: Missing dependency: {e}")
            print("  [LoRA] Install: pip install transformers peft torch qwen-vl-utils")
        except Exception as e:
            print(f"  [LoRA] WARNING: Failed to load model: {e}")
            print("  [LoRA] Falling back to empty constraints")

    async def extract(
        self,
        case: Dict[str, Any],
        condition_overrides: Dict[str, Any],
        image_context: Optional[ImageParseResult] = None,
    ) -> Constraints:
        """
        Extract constraints using VLM + LoRA.

        Matches the same signature as PromptConstraintsExtractor.extract()
        so the pipeline can swap extractors transparently.

        Args:
            case: Case dict from cases_v3_filtered.jsonl
            condition_overrides: Condition config from profiles.yaml
            image_context: Parsed image descriptions (unused by LoRA —
                           the model sees raw images directly)

        Returns:
            Constraints object with extracted fields
        """
        if not self._loaded:
            return Constraints(
                confidence=0.0,
                source="lora_not_loaded",
            )

        # Apply condition mask (respects A1-C3 modality control)
        masked_case = ConditionMask.apply(case, condition_overrides)

        # Build inference messages (same format as training data)
        messages = self._build_messages(masked_case)

        # Run VLM inference
        try:
            output_text = self._generate(messages)
        except Exception as e:
            print(f"  [LoRA] Inference failed: {e}")
            return Constraints(confidence=0.0, source="lora_inference_failed")

        # Parse JSON output
        data = self._parse_json(output_text)
        if data:
            return Constraints(
                storey_name=data.get("storey_name"),
                ifc_class=data.get("ifc_class"),
                near_keywords=data.get("near_keywords", []),
                relations=data.get("relations", []),
                confidence=0.85,
                source="lora",
            )

        print(f"  [LoRA] JSON parse failed. Raw output: {output_text[:200]}")
        return Constraints(confidence=0.0, source="lora_parse_failed")

    # ── Internal methods ──────────────────────────────────────────────────

    def _build_messages(self, masked_case: Dict[str, Any]) -> list:
        """
        Build ChatML messages for VLM inference.

        Format matches 7_prepare_lora_data.py exactly:
          system: SYSTEM_PROMPT
          user: [image, image, ..., text]

        The text part uses the same [4D Task Status] / [Chat Log] / [Query]
        format as the training data.
        """
        # Build user content (multimodal)
        user_content = []

        # Add images (site photos + floorplan)
        image_paths = self._resolve_image_paths(masked_case)
        for img_path in image_paths:
            user_content.append({
                "type": "image",
                "image": f"file://{img_path}",
            })

        # Build text (same format as training)
        user_text = self._build_user_text(masked_case)
        user_content.append({
            "type": "text",
            "text": user_text,
        })

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _build_user_text(self, case: Dict[str, Any]) -> str:
        """
        Build user text from case inputs.

        CRITICAL: This must produce identical text to
        7_prepare_lora_data.py:format_user_message() for train-inference
        alignment.
        """
        parts = []

        # 4D project context
        ctx = case.get("inputs", {}).get("project_context", {})
        task_status = ctx.get("4d_task_status", "")
        if task_status:
            parts.append(f"[4D Task Status] {task_status}")

        project_phase = ctx.get("project_phase", "")
        if project_phase:
            parts.append(f"[Project Phase] {project_phase}")

        # Chat history
        chat = case.get("inputs", {}).get("chat_history", [])
        if chat:
            parts.append("[Chat Log]")
            for msg in chat:
                role = msg.get("role", "User")
                text = msg.get("text", "")
                parts.append(f"  {role}: {text}")

        # Query
        query = case.get("query_text", "")
        if query:
            parts.append(f"\n[Query] {query}")

        parts.append("\nExtract the search constraints as JSON.")

        return "\n".join(parts)

    def _resolve_image_paths(self, case: Dict[str, Any]) -> List[str]:
        """Resolve image paths to absolute paths (same logic as pipeline.py)."""
        paths = []
        inputs = case.get("inputs", {})

        # Site photos
        for img in inputs.get("images", []):
            resolved = self._resolve_single_path(img)
            if resolved:
                paths.append(resolved)

        # Floorplan patch
        fp = inputs.get("floorplan_patch")
        if fp:
            resolved = self._resolve_single_path(fp)
            if resolved:
                paths.append(resolved)

        return paths

    def _resolve_single_path(self, img_path: str) -> Optional[str]:
        """Resolve a single image path."""
        p = Path(img_path)

        # Already absolute and exists
        if p.is_absolute() and p.exists():
            return str(p)

        # Try under image_dir
        if self.image_dir:
            candidate = Path(self.image_dir) / p
            if candidate.exists():
                return str(candidate)
            candidate = Path(self.image_dir) / p.name
            if candidate.exists():
                return str(candidate)

        # Return as-is (model will handle missing gracefully)
        return str(p)

    def _generate(self, messages: list) -> str:
        """Run VLM inference and return raw text output."""
        from qwen_vl_utils import process_vision_info

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(messages)

        # Tokenize
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate (short output — JSON only)
        import torch
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from model output (with fallbacks)."""
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract from markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Find JSON object with expected keys
        match = re.search(r'(\{[^{]*"storey_name"[^}]*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None
