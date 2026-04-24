"""
LoRA-Based Constraints Extractor

Fine-tuned Qwen2.5-VL-7B-Instruct with LoRA adapter for multimodal
constraints extraction. Takes chat + images + floorplan + 4D metadata
and outputs structured JSON constraints.

The inference prompt format is identical to the training format produced
by data_curation/scripts/synth/7_prepare_lora_data.py to ensure
train-inference alignment.
"""
# TODO: enhance and verify the constraint rules!!!

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .condition_mask import ConditionMask
from .floorplan_counter import FloorplanCounter, FloorplanCountResult, merge_position_context
from .types import Constraints, ImageParseResult, SpatialTriplet
from ..common.config import load_yaml_prompts

# ── System prompt — loaded from prompts/constraints_extraction.yaml ───────────
# Shared with constraints_extractor_prompt_only.py and training/eval.py so that
# LoRA vs Prompt comparison measures model quality, not prompt wording.

_PROMPTS_PATH = Path(__file__).parent.parent.parent / "prompts" / "constraints_extraction.yaml"


def _prompt_key_from_adapter(adapter_path: Optional[str], explicit_key: Optional[str]) -> str:
    if explicit_key:
        return explicit_key
    adapter_lower = str(adapter_path or "").lower()
    if "g7" in adapter_lower:
        return "lora_system_g7"
    return "lora_system"


def _load_lora_system_prompt(prompt_key: str) -> str:
    prompts = load_yaml_prompts(str(_PROMPTS_PATH))
    return prompts.get(prompt_key, prompts.get("lora_system", prompts.get("system", "")))


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
    - Output: JSON with {storey_name, ifc_class, space_name, target_name_keyword, spatial_relations}
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        image_dir: str = "",
        prompt_key: Optional[str] = None,
    ):
        """
        Initialize LoRA extractor.

        Loads the base model (4-bit quantized) + LoRA adapter once.
        Raises on failure — no silent fallback.

        Args:
            adapter_path: Path to LoRA adapter directory (contains
                          adapter_model.safetensors + adapter_config.json)
            image_dir: Root directory for resolving relative image paths
        """
        self.adapter_path = adapter_path
        self.image_dir = image_dir
        self.prompt_key = _prompt_key_from_adapter(adapter_path, prompt_key)
        self.system_prompt = _load_lora_system_prompt(self.prompt_key)
        self.model = None
        self.processor = None
        self._loaded = False
        self.floorplan_counter = FloorplanCounter(image_dir=image_dir)

        if adapter_path:
            self._load_model(adapter_path)  # Raises on failure — no silent fallback

    def _load_model(self, adapter_path: str):
        """Load base model (4-bit quantized) + LoRA adapter.

        Raises on failure — no silent fallback.
        """
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from peft import PeftModel
        import torch

        adapter_path_obj = Path(adapter_path)
        if not (adapter_path_obj / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"[LoRA] adapter_config.json not found in: {adapter_path}\n"
                f"  Expected files: adapter_config.json + adapter_model.safetensors"
            )

        base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

        # 4-bit quantization — fits ~5GB VRAM (vs 14GB for float16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [LoRA] GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")
            if vram_gb < 5.0:
                print(f"  [LoRA] WARNING: {vram_gb:.1f} GB VRAM may be insufficient "
                      f"for Qwen2.5-VL-7B 4-bit (~5GB needed)")
        else:
            print("  [LoRA] WARNING: No CUDA GPU — inference will be very slow on CPU")

        print(f"  [LoRA] Loading base model: {base_model_id} (4-bit quantized)")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

        print(f"  [LoRA] Loading adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(base_model_id)
        self._loaded = True
        print(f"  [LoRA] Model ready ({adapter_path})")

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
            raise RuntimeError(
                "[LoRA] Model not loaded! Check adapter_path and GPU availability. "
                "Will NOT fall back silently — fix the root cause."
            )

        # Apply condition mask (respects A1-C3 modality control)
        masked_case = ConditionMask.apply(case, condition_overrides)
        opencv_position = self.floorplan_counter.count_from_case(masked_case)

        # Build inference messages (same format as training data)
        messages = self._build_messages(masked_case, opencv_position)

        # Run VLM inference
        try:
            output_text = self._generate(messages)
        except Exception as e:
            print(f"  [LoRA] Inference failed: {e}")
            return Constraints(confidence=0.0, source="lora_inference_failed")

        # Parse JSON output
        data = self._parse_json(output_text)
        if data:
            # Normalise field: model may output 'relations' (old) or 'spatial_relations'
            sr_raw = data.get("spatial_relations") or []
            if not sr_raw:
                rel_raw = data.get("relations")
                if isinstance(rel_raw, list):
                    sr_raw = [r for r in rel_raw if isinstance(r, dict) and "predicate" in r]

            spatial_rels = []
            for rel in sr_raw:
                direction = rel.get("direction")
                if isinstance(direction, str):
                    direction = direction.lower().strip()
                if direction not in {"left", "right"}:
                    direction = None
                spatial_rels.append(SpatialTriplet(
                    subject_type=data.get("ifc_class", ""),
                    predicate=rel.get("predicate", "ADJACENT_TO").upper(),
                    object_type=rel.get("object_type", ""),
                    object_subtype=rel.get("object_subtype"),
                    direction=direction,
                    object_material=rel.get("object_material"),
                    confidence=rel.get("confidence", 0.0),
                ))

            # Use max relation confidence, or 0.85 for attribute-only cases
            conf = max((r.confidence for r in spatial_rels), default=0.85)

            final_position_context, pos_conf, pos_source = merge_position_context(
                data.get("position_context"),
                opencv_position,
            )

            return Constraints(
                storey_name=data.get("storey_name"),
                ifc_class=data.get("ifc_class"),
                space_name=data.get("space_name"),
                target_name_keyword=data.get("target_name_keyword"),
                position_context=final_position_context,
                position_context_confidence=pos_conf,
                position_context_source=pos_source,
                spatial_relations=spatial_rels,
                confidence=conf,
                source="lora",
            )

        print(f"  [LoRA] JSON parse failed. Raw output: {output_text[:200]}")
        if opencv_position is not None:
            return Constraints(
                position_context=opencv_position.position_context,
                position_context_confidence=opencv_position.confidence,
                position_context_source="opencv",
                confidence=max(0.3, opencv_position.confidence),
                source="lora_parse_failed+opencv",
            )
        return Constraints(confidence=0.0, source="lora_parse_failed")

    # ── Internal methods ──────────────────────────────────────────────────

    def _build_messages(
        self,
        masked_case: Dict[str, Any],
        opencv_position: Optional[FloorplanCountResult] = None,
    ) -> list:
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
        user_text = self._build_user_text(masked_case, opencv_position)
        user_content.append({
            "type": "text",
            "text": user_text,
        })

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _build_user_text(
        self,
        case: Dict[str, Any],
        opencv_position: Optional[FloorplanCountResult] = None,
    ) -> str:
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

        if opencv_position is not None:
            parts.append("\n[OpenCV Counting]")
            parts.append(f"  position_context: {opencv_position.position_context}")
            parts.append(f"  position: {opencv_position.position}")
            parts.append(f"  total: {opencv_position.total}")
            parts.append(f"  confidence: {opencv_position.confidence:.2f}")
            if opencv_position.mode == "patch_only":
                parts.append("  note: patch-only fallback estimate; use cautiously.")
            else:
                parts.append("  note: derived from the floorplan patch matched against a larger floorplan reference.")

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

        # Find JSON object with expected keys (handles nested arrays)
        match = re.search(r'(\{.*?"storey_name".*\})\s*$', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None
