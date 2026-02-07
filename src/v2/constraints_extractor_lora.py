"""
LoRA-Based Constraints Extractor (Placeholder)

Future implementation: Fine-tuned VLM (Qwen3-VL-8B + LoRA adapter)
for multimodal constraints extraction.

Currently returns placeholder constraints until LoRA training is complete.
"""

from typing import Dict, Any, Optional
from .types import Constraints


class LoRAConstraintsExtractor:
    """
    Extract constraints using fine-tuned Qwen3-VL-8B with LoRA adapter.

    This is a placeholder implementation. The actual LoRA model will be
    trained on multimodal inputs (chat + images + floorplan + 4D metadata)
    to extract structured constraints.

    Training plan:
    - Base model: Qwen/Qwen3-VL-8B
    - Adapter: LoRA (rank=16, alpha=32)
    - Task: Multimodal constraint extraction
    - Output: JSON with {storey_name, ifc_class, near_keywords, relations}
    """

    def __init__(self, adapter_path: Optional[str] = None):
        """
        Initialize LoRA extractor.

        Args:
            adapter_path: Path to LoRA adapter checkpoint (optional for now)
        """
        self.adapter_path = adapter_path
        self.model = None  # TODO: Load Qwen3-VL-8B + LoRA adapter

        if adapter_path:
            print(f"⚠️  LoRA adapter path provided: {adapter_path}")
            print("   LoRA extraction not yet implemented - using placeholder")

    async def extract(
        self,
        case: Dict[str, Any],
        condition_overrides: Dict[str, Any]
    ) -> Constraints:
        """
        Extract constraints using VLM + LoRA.

        Currently returns placeholder constraints.

        TODO: Implement actual VLM inference:
        1. Load Qwen3-VL-8B with LoRA adapter
        2. Prepare multimodal inputs (text + images + floorplan)
        3. Run inference to get JSON output
        4. Parse and return Constraints

        Args:
            case: Case dict from cases_v2.jsonl
            condition_overrides: Condition config from profiles.yaml

        Returns:
            Constraints object (currently placeholder)
        """
        # TODO: Implement VLM + LoRA inference

        # Placeholder: Return empty constraints with low confidence
        print("⚠️  LoRA extraction not implemented - returning placeholder")
        return Constraints(
            storey_name=None,
            ifc_class=None,
            near_keywords=[],
            relations=[],
            confidence=0.0,
            source="lora_not_implemented"
        )

    def _prepare_vlm_inputs(
        self,
        case: Dict[str, Any],
        condition_overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare multimodal inputs for VLM.

        TODO: Implement input preparation:
        - Text: Format chat history + metadata
        - Images: Load and resize site photos
        - Floorplan: Load floorplan patch if available
        - 4D: Extract task status and schedule info

        Args:
            case: Original case dict
            condition_overrides: Condition config

        Returns:
            Dict with prepared inputs for VLM
        """
        # Placeholder
        return {
            "text": "",
            "images": [],
            "floorplan": None,
            "metadata": {}
        }

    def _run_vlm_inference(self, inputs: Dict[str, Any]) -> str:
        """
        Run VLM inference to get JSON output.

        TODO: Implement:
        1. Tokenize inputs
        2. Generate with constrained decoding (JSON-only)
        3. Return raw JSON string

        Args:
            inputs: Prepared multimodal inputs

        Returns:
            JSON string with constraints
        """
        # Placeholder
        return '{"storey_name": null, "ifc_class": null, "near_keywords": [], "relations": []}'
