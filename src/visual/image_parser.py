"""
ImageParserReader — Centralized VLM-based image understanding.

Reads image bytes once, calls Gemini VLM for structured description,
caches results so downstream stages (constraints extractor, CLIP reranker)
can consume pre-parsed visual semantics without re-reading images.
"""

import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from common.config import load_yaml_prompts

from src.v2.types import ImageParseResult, ParsedImage


class ImageParserReader:
    """
    Centralized image parsing via Gemini VLM.

    Usage:
        parser = ImageParserReader(llm)
        result = await parser.parse_case_images(case, condition_overrides, image_dir)
        # result.combined_description -> inject into constraints prompt
        # result.inferred_ifc_class   -> supplement constraints
    """

    # In-memory cache: hash(image_path) -> ParsedImage
    _cache: Dict[str, ParsedImage] = {}

    def __init__(self, llm: Any, prompts_path: str = "prompts/image_parsing.yaml"):
        """
        Args:
            llm: ChatGoogleGenerativeAI instance (Gemini 2.5 Flash)
            prompts_path: Path to image parsing prompts YAML
        """
        self.llm = llm
        self._load_prompts(prompts_path)

    def _load_prompts(self, prompts_path: str):
        """Load VLM prompts from YAML file."""
        data = load_yaml_prompts(prompts_path)

        self.site_photo_prompt = data.get("site_photo_prompt", "")
        self.floorplan_prompt = data.get("floorplan_prompt", "")

    async def parse_case_images(
        self,
        case: Dict[str, Any],
        condition_overrides: Dict[str, Any],
        image_dir: str = "",
    ) -> ImageParseResult:
        """
        Parse all images for a case, respecting condition masking.

        Args:
            case: Case dict (already condition-masked)
            condition_overrides: Condition config dict
            image_dir: Directory to resolve relative image paths

        Returns:
            ImageParseResult with structured descriptions
        """
        result = ImageParseResult()

        use_images = condition_overrides.get("use_images", False)
        use_floorplan = condition_overrides.get("use_floorplan", False)

        inputs = case.get("inputs", {})

        # Parse site photos
        if use_images:
            for img_path in inputs.get("images", []):
                resolved = self._resolve_path(img_path, image_dir)
                parsed = await self._parse_single_image(resolved, "site_photo")
                if parsed:
                    result.site_photos.append(parsed)

        # Parse floorplan
        if use_floorplan:
            fp_path = inputs.get("floorplan_patch")
            if fp_path:
                resolved = self._resolve_path(fp_path, image_dir)
                parsed = await self._parse_single_image(resolved, "floorplan")
                if parsed:
                    result.floorplan = parsed

        return result

    async def _parse_single_image(
        self,
        image_path: str,
        image_type: str,
    ) -> Optional[ParsedImage]:
        """Parse one image via VLM, using cache if available."""
        cache_key = self._cache_key(image_path)

        if cache_key in self._cache:
            return self._cache[cache_key]

        path = Path(image_path)
        if not path.exists():
            print(f"  [ImageParser] image not found: {image_path}")
            return None

        # Read and encode image bytes
        image_bytes = path.read_bytes()
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        mime_type = self._guess_mime(path)

        # Choose prompt
        text_prompt = (
            self.floorplan_prompt if image_type == "floorplan" else self.site_photo_prompt
        )

        # Build multimodal message
        message = HumanMessage(
            content=[
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                },
            ]
        )

        t0 = time.perf_counter()
        try:
            response = await self.llm.ainvoke([message])
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            parsed = self._parse_vlm_response(
                response_text, image_path, image_type, latency_ms
            )
            print(
                f"  [ImageParser] {image_type} parsed in {latency_ms:.0f}ms "
                f"-> {parsed.element_type or 'N/A'} | {parsed.ifc_class_hint or 'N/A'}"
            )
        except Exception as e:
            print(f"  [ImageParser] VLM parse failed for {image_path}: {e}")
            parsed = ParsedImage(
                image_path=image_path,
                image_type=image_type,
                description=f"[Image parse failed: {e}]",
                confidence=0.0,
            )

        self._cache[cache_key] = parsed
        return parsed

    def _parse_vlm_response(
        self,
        response_text: str,
        image_path: str,
        image_type: str,
        latency_ms: float,
    ) -> ParsedImage:
        """Parse VLM JSON response into ParsedImage."""
        data = None

        # Try direct JSON parse
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if data is None:
            match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # Try finding raw JSON object
        if data is None:
            match = re.search(r"(\{[^{}]*\})", response_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        if data:
            return ParsedImage(
                image_path=image_path,
                image_type=image_type,
                element_type=data.get("element_type"),
                ifc_class_hint=data.get("ifc_class_hint"),
                material=data.get("material"),
                defect_type=data.get("defect_type"),
                defect_severity=data.get("defect_severity"),
                location_cues=data.get("location_cues", []),
                spatial_zone=data.get("spatial_zone"),
                storey_hint=data.get("storey_hint"),
                marked_elements=data.get("marked_elements", []),
                description=data.get("description", ""),
                keywords=data.get("keywords", []),
                confidence=0.85,
                parse_latency_ms=latency_ms,
            )

        # Fallback: use raw text as description
        return ParsedImage(
            image_path=image_path,
            image_type=image_type,
            description=response_text[:500],
            confidence=0.3,
            parse_latency_ms=latency_ms,
        )

    @staticmethod
    def _resolve_path(img_path: str, image_dir: str) -> str:
        """Resolve image path relative to image_dir."""
        p = Path(img_path)
        if p.is_absolute() and p.exists():
            return str(p)
        if image_dir:
            candidate = Path(image_dir) / p.name
            if candidate.exists():
                return str(candidate)
        return img_path

    @staticmethod
    def _cache_key(image_path: str) -> str:
        return hashlib.md5(image_path.encode()).hexdigest()

    @staticmethod
    def _guess_mime(path: Path) -> str:
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/png")

    def clear_cache(self):
        """Clear the parse cache (useful between evaluation runs)."""
        self._cache.clear()
