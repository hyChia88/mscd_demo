"""
MSCD VLM LoRA_3 Inference — Modal Serverless Endpoint

Loads Qwen2.5-VL-7B-Instruct + LoRA_3 adapter from Modal volume,
runs inference on multimodal inputs (site photo + floorplan + chat text).

Usage:
    # From Python (e.g. Streamlit demo):
    import modal
    f = modal.Function.from_name("mscd-vlm-lora3-inference", "predict")
    result = f.remote(images=[...], chat_text="...", metadata_text="...")

    # CLI test:
    modal run training/inference.py --chat "crack near the railing on floor 3"
"""

import json
import os
from pathlib import Path

import modal

# ── Modal infrastructure ─────────────────────────────────────────────────────

app = modal.App("mscd-vlm-lora3-inference")

# Local g8 adapter (downloaded from volume after training, verified correct).
# Baked into the image so the live endpoint is independent of the volume copy,
# which was overwritten by a later experiment.
_G8_LOCAL_PATH = Path(__file__).parent.parent / "models" / "lora6_v2_ap_20260331" / "g8_posctx_dim" / "best"

# G9 artefacts — baked so the predictor is self-contained.
_REPO_ROOT = Path(__file__).parent.parent
_G9_LOCAL_PATH = _REPO_ROOT / "models" / "lora6_v2_ap_20260331" / "g9_opencv_cluster" / "best"
_RESNET_LOCAL_PATH = _REPO_ROOT / "models" / "cluster_classifier_ap" / "best.pt"
_FLOORPLANS_LOCAL_DIR = _REPO_ROOT.parent / "data_curation" / "datasets" / "synth_v0.5_ap" / "floorplans_full"
_FLOORPLAN_COUNTER_LOCAL = _REPO_ROOT / "src" / "neurosym" / "floorplan_counter.py"
_CLUSTER_CLASSIFIER_LOCAL = _REPO_ROOT / "src" / "neurosym" / "cluster_classifier.py"

# Same container image as training (model + deps already installed)
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "unsloth",
        "qwen-vl-utils",
        "datasets==4.3.0",
        "hf-transfer",
    )
    .run_commands("pip install --no-deps --force-reinstall 'unsloth @ git+https://github.com/unslothai/unsloth.git'")
    .pip_install("transformers==4.56.2")
    .run_commands("pip install --no-deps trl==0.22.2")
    .env({"HF_HOME": "/model_cache"})
)

# G9 image extends the base with OpenCV + ResNet + baked perception artefacts.
# Kept separate so G8 cold-starts don't pay for the cv2 / torchvision install.
g9_inference_image = (
    inference_image
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("opencv-python-headless==4.10.0.84", "torchvision")
    .add_local_dir(str(_G9_LOCAL_PATH), "/adapters/g9", copy=True)
    .add_local_file(str(_RESNET_LOCAL_PATH), "/perception/cluster_classifier.pt", copy=True)
    .add_local_dir(str(_FLOORPLANS_LOCAL_DIR), "/perception/floorplans_full", copy=True)
    .add_local_file(str(_FLOORPLAN_COUNTER_LOCAL), "/perception/floorplan_counter.py", copy=True)
    .add_local_file(str(_CLUSTER_CLASSIFIER_LOCAL), "/perception/cluster_classifier.py", copy=True)
)

model_cache = modal.Volume.from_name("mscd-model-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("mscd-checkpoints", create_if_missing=True)

# Source of truth: prompts/constraints_extraction.yaml
# MUST match training exactly — do NOT change without retraining.

# lora_system — used by LoRA_3 (early runs, G1-G6 baseline adapters)
_SYSTEM_PROMPT = (
    "You are a construction site assistant that extracts IFC search constraints "
    "from multimodal evidence. Use the floorplan and site photo to reason about "
    "storey, element type, and spatial relations. Output valid JSON only."
)

# lora_system_g7 — used by G7 / G8 / G4-Ultimate adapters (LoRA_6 runs).
# Explicitly lists all output fields so the model produces the correct schema.
_SYSTEM_PROMPT_G7 = (
    "You are a construction site assistant that extracts IFC search constraints "
    "from multimodal evidence. Use the floorplan and site photo to reason about "
    "storey, element type, position context, and spatial relations. Output valid "
    "JSON only with fields storey_name, ifc_class, space_name, target_name_keyword, "
    "position_context, and spatial_relations. Each spatial relation must use "
    "predicate/object_type and may include object_subtype, direction, object_material, "
    "confidence. Only include direction, object_subtype, or position_context when "
    "supported by the visual or topological evidence. Do not guess. Return JSON only."
)

ADAPTER_PATH = "/checkpoints/mscd-lora-v3-5ep/final"
ADAPTER_PATH_G8 = "/checkpoints/mscd-lora-v6-g8-posctx-dim/best"
ADAPTER_PATH_G8_BAKED = "/adapters/g8"   # baked into image from local verified copy
# G4 Ultimate — Track A winner (Hop-1=86.7%), loaded via base+PEFT two-step
ADAPTER_PATH_G4 = "/checkpoints/mscd-lora-v6-g4-ultimate/best"
# G9 — OpenCV + ResNet-augmented LoRA_6 adapter (baked into g9_inference_image).
ADAPTER_PATH_G9 = "/adapters/g9"
RESNET_CHECKPOINT_PATH = "/perception/cluster_classifier.pt"
FLOORPLANS_DIR = "/perception/floorplans_full"
CALIBRATION_PATH = "/perception/floorplans_full/calibration.json"
BASE_MODEL = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"

# G9 system prompt — must match prompts/constraints_extraction.yaml `lora_system_g9`.
_SYSTEM_PROMPT_G9 = (
    "You are a construction site assistant that extracts IFC search constraints "
    "from multimodal evidence. Use the floorplan and site photo to reason about "
    "storey, element type, position context, size, and spatial relations. Output "
    "valid JSON only with fields storey_name, ifc_class, space_name, "
    "target_name_keyword, position_context, size_cluster, and spatial_relations. "
    "If the user prompt includes an [OpenCV Counting] block, copy its position/total "
    "numbers into position_context (format \"Nth of M openings on the same wall\"); "
    "do not re-guess. For IfcWindow or IfcDoor targets, set size_cluster to a label "
    "such as window_M_1480x1380 or door_S_760x2030. Each spatial relation must use "
    "predicate/object_type and may include object_subtype, direction, "
    "object_material, confidence. Only include direction, object_subtype, "
    "position_context, or size_cluster when supported by evidence. Do not guess. "
    "Return JSON only."
)


@app.cls(
    image=inference_image,
    gpu="A100",
    volumes={
        "/model_cache": model_cache,
        "/checkpoints": checkpoint_vol,
    },
    container_idle_timeout=1800,  # keep warm for 30 min between calls
    # min_containers=1,           # uncomment for zero-cold-start demos (~24h A100 cost)
)
class LoRA3Predictor:
    """Persistent inference class — model stays loaded between calls."""

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoProcessor
        from unsloth import FastVisionModel

        print(f"Loading adapter from {ADAPTER_PATH}...")
        assert os.path.isdir(ADAPTER_PATH), f"Adapter not found: {ADAPTER_PATH}"

        self.model, _tokenizer = FastVisionModel.from_pretrained(
            ADAPTER_PATH,
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(self.model)

        # Load the full multimodal processor (handles images + text)
        # The saved adapter may only have a plain tokenizer, so load from base model
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)
        print("Model loaded and ready for inference.")

    @modal.method()
    def predict(
        self,
        image_bytes_list: list[bytes],
        chat_text: str = "",
        metadata_text: str = "",
    ) -> dict:
        """Run LoRA_3 inference on multimodal inputs.

        Args:
            image_bytes_list: List of PNG/JPEG image bytes (site photo, floorplan)
            chat_text: Chat log text
            metadata_text: 4D metadata text (task status, phase, location)

        Returns:
            {
                "raw_output": str,          # raw model output
                "parsed": dict | None,      # parsed JSON if valid
                "valid_json": bool,
            }
        """
        import io
        from PIL import Image

        pil_images = []
        for img_bytes in image_bytes_list:
            pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

        return self._predict_core(pil_images, chat_text, metadata_text)

    @modal.method()
    def explain(
        self,
        image_bytes_list: list[bytes],
        chat_text: str = "",
        metadata_text: str = "",
        grid_size: int = 4,
    ) -> dict:
        """Occlusion-based saliency: mask image patches, measure prediction change.

        For each image, divides it into a grid_size x grid_size grid.  Each patch
        is occluded (replaced with gray), inference is re-run, and the change in
        spatial-relation confidence is recorded as that patch's importance.

        Returns:
            {
                "baseline": {raw_output, parsed, valid_json},
                "heatmaps": [                        # one per input image
                    [[float, ...], ...],             # grid_size x grid_size
                ],
                "image_sizes": [(w, h), ...],
                "grid_size": int,
                "spatial_focus_tokens": [str, ...],  # which tokens were tracked
            }
        """
        import io
        import copy
        import torch
        import numpy as np
        from PIL import Image

        # ── Load images ──────────────────────────────────────────────────
        pil_images = []
        for img_bytes in image_bytes_list:
            pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

        if not pil_images:
            return {"error": "No images provided for explain"}

        # ── Baseline prediction ──────────────────────────────────────────
        baseline = self._predict_core(pil_images, chat_text, metadata_text)
        baseline_rels = (baseline.get("parsed") or {}).get("spatial_relations") or []
        baseline_conf = max(
            (r.get("confidence", 0) for r in baseline_rels), default=0.0
        )
        baseline_pred = baseline_rels[0].get("predicate", "") if baseline_rels else ""

        # ── Occlusion sweep ──────────────────────────────────────────────
        heatmaps = []
        image_sizes = []
        for img_idx, img in enumerate(pil_images):
            w, h = img.size
            image_sizes.append((w, h))
            pw, ph = w // grid_size, h // grid_size
            heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

            for gi in range(grid_size):
                for gj in range(grid_size):
                    # Create occluded copy
                    masked = img.copy()
                    x0, y0 = gj * pw, gi * ph
                    x1 = w if gj == grid_size - 1 else x0 + pw
                    y1 = h if gi == grid_size - 1 else y0 + ph
                    # Fill patch with gray
                    gray_patch = Image.new("RGB", (x1 - x0, y1 - y0), (128, 128, 128))
                    masked.paste(gray_patch, (x0, y0))

                    # Run inference with this image masked
                    masked_images = list(pil_images)
                    masked_images[img_idx] = masked
                    result = self._predict_core(masked_images, chat_text, metadata_text)

                    # Measure change
                    m_rels = (result.get("parsed") or {}).get("spatial_relations") or []
                    m_conf = max(
                        (r.get("confidence", 0) for r in m_rels), default=0.0
                    )
                    m_pred = m_rels[0].get("predicate", "") if m_rels else ""

                    # Importance = how much the prediction degrades
                    if not m_rels:
                        # Spatial relation disappeared entirely
                        importance = 1.0
                    elif m_pred != baseline_pred:
                        # Predicate changed
                        importance = 0.8
                    else:
                        # Confidence drop
                        importance = max(0.0, baseline_conf - m_conf)

                    heatmap[gi, gj] = importance

            # Normalize per image
            hmax = heatmap.max()
            if hmax > 0:
                heatmap = heatmap / hmax

            heatmaps.append(heatmap.tolist())

        return {
            "baseline": baseline,
            "heatmaps": heatmaps,
            "image_sizes": image_sizes,
            "grid_size": grid_size,
            "spatial_focus_tokens": [baseline_pred] if baseline_pred else [],
        }

    def _predict_core(
        self,
        pil_images: list,
        chat_text: str,
        metadata_text: str,
    ) -> dict:
        """Shared prediction logic used by both predict() and explain()."""
        import torch

        content_parts = []
        for img in pil_images:
            content_parts.append({"type": "image", "image": img})

        text_parts = []
        if metadata_text:
            text_parts.append(metadata_text)
        if chat_text:
            text_parts.append(f"[Chat Log]\n{chat_text}")
        if text_parts:
            content_parts.append({"type": "text", "text": "\n".join(text_parts)})

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

        input_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if pil_images:
            inputs = self.processor(
                text=[input_text],
                images=pil_images,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=[input_text],
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=320, do_sample=False, use_cache=True
            )

        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        raw_output = self.processor.decode(trimmed, skip_special_tokens=True).strip()

        parsed = None
        valid_json = False
        try:
            parsed = json.loads(raw_output)
            valid_json = True
        except json.JSONDecodeError:
            pass

        return {
            "raw_output": raw_output,
            "parsed": parsed,
            "valid_json": valid_json,
        }


@app.cls(
    image=inference_image,
    gpu="A100",
    volumes={
        "/model_cache": model_cache,
        "/checkpoints": checkpoint_vol,
    },
    container_idle_timeout=1800,  # keep warm for 30 min between calls
    # min_containers=1,           # uncomment for zero-cold-start demos (~24h A100 cost)
)
class G8Predictor:
    """LoRA_6 G4-Ultimate predictor — Track A winner (Hop-1=86.7%).

    Uses G4-Ultimate adapter (verified correct) loaded via base+PEFT two-step
    to match eval.py behaviour. System prompt: lora_system_g7 (explicitly
    lists all output fields — required for LoRA_6 adapters G7/G8/G4-Ultimate).
    """

    @modal.enter()
    def load_model(self):
        from transformers import AutoProcessor
        from unsloth import FastVisionModel
        from peft import PeftModel

        adapter_path = ADAPTER_PATH_G4
        print(f"Loading base model: {BASE_MODEL}")
        self.model, _tokenizer = FastVisionModel.from_pretrained(
            BASE_MODEL,
            load_in_4bit=True,
        )

        print(f"Loading G4-Ultimate adapter from {adapter_path}...")
        assert os.path.isdir(adapter_path), f"Adapter not found: {adapter_path}"
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        FastVisionModel.for_inference(self.model)

        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)
        print("G4-Ultimate model loaded.")

    @modal.method()
    def predict(
        self,
        image_bytes_list: list[bytes],
        chat_text: str = "",
        metadata_text: str = "",
    ) -> dict:
        import io
        from PIL import Image
        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
        return self._predict_core(pil_images, chat_text, metadata_text)

    @modal.method()
    def explain(
        self,
        image_bytes_list: list[bytes],
        chat_text: str = "",
        metadata_text: str = "",
        grid_size: int = 4,
    ) -> dict:
        import io
        import numpy as np
        from PIL import Image

        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
        if not pil_images:
            return {"error": "No images provided"}

        baseline = self._predict_core(pil_images, chat_text, metadata_text)
        baseline_rels = (baseline.get("parsed") or {}).get("spatial_relations") or []
        baseline_conf = max((r.get("confidence", 0) for r in baseline_rels), default=0.0)
        baseline_pred = baseline_rels[0].get("predicate", "") if baseline_rels else ""

        heatmaps, image_sizes = [], []
        for img_idx, img in enumerate(pil_images):
            w, h = img.size
            image_sizes.append((w, h))
            pw, ph = w // grid_size, h // grid_size
            heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
            for gi in range(grid_size):
                for gj in range(grid_size):
                    masked = img.copy()
                    x0, y0 = gj * pw, gi * ph
                    x1 = w if gj == grid_size - 1 else x0 + pw
                    y1 = h if gi == grid_size - 1 else y0 + ph
                    masked.paste(Image.new("RGB", (x1 - x0, y1 - y0), (128, 128, 128)), (x0, y0))
                    imgs = list(pil_images); imgs[img_idx] = masked
                    r = self._predict_core(imgs, chat_text, metadata_text)
                    m_rels = (r.get("parsed") or {}).get("spatial_relations") or []
                    m_conf = max((x.get("confidence", 0) for x in m_rels), default=0.0)
                    m_pred = m_rels[0].get("predicate", "") if m_rels else ""
                    if not m_rels:       importance = 1.0
                    elif m_pred != baseline_pred: importance = 0.8
                    else:                importance = max(0.0, baseline_conf - m_conf)
                    heatmap[gi, gj] = importance
            hmax = heatmap.max()
            if hmax > 0: heatmap /= hmax
            heatmaps.append(heatmap.tolist())

        return {
            "baseline": baseline, "heatmaps": heatmaps,
            "image_sizes": image_sizes, "grid_size": grid_size,
            "spatial_focus_tokens": [baseline_pred] if baseline_pred else [],
        }

    def _predict_core(self, pil_images: list, chat_text: str, metadata_text: str) -> dict:
        import torch
        content_parts = [{"type": "image", "image": img} for img in pil_images]
        text_parts = []
        if metadata_text: text_parts.append(metadata_text)
        if chat_text:     text_parts.append(f"[Chat Log]\n{chat_text}")
        if text_parts:    content_parts.append({"type": "text", "text": "\n".join(text_parts)})
        # G4-Ultimate / G7 / G8 were trained with lora_system_g7 (explicitly lists fields)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_G7},
            {"role": "user",   "content": content_parts},
        ]
        input_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_kwargs = dict(text=[input_text], add_special_tokens=False, return_tensors="pt")
        if pil_images: proc_kwargs["images"] = pil_images
        inputs = self.processor(**proc_kwargs).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=320, do_sample=False, use_cache=True
            )
        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        raw_output = self.processor.decode(trimmed, skip_special_tokens=True).strip()
        parsed = None; valid_json = False
        try:
            parsed = json.loads(raw_output); valid_json = True
        except json.JSONDecodeError:
            pass
        return {"raw_output": raw_output, "parsed": parsed, "valid_json": valid_json}


@app.cls(
    image=inference_image,
    gpu="A100",
    volumes={
        "/model_cache": model_cache,
        "/checkpoints": checkpoint_vol,
    },
    container_idle_timeout=1800,  # keep warm for 30 min between calls
    # min_containers=1,           # uncomment for zero-cold-start demos (~24h A100 cost)
)
class G8ModelPredictor:
    """LoRA_6 G8 position-context predictor.

    Loads the actual G8 adapter (mscd-lora-v6-g8-posctx-dim/best) via
    base+PEFT two-step. Uses lora_system_g7 system prompt — required for
    all LoRA_6 adapters. G8 adds position_context field vs G4-Ultimate.
    """

    @modal.enter()
    def load_model(self):
        from transformers import AutoProcessor
        from unsloth import FastVisionModel
        from peft import PeftModel

        adapter_path = ADAPTER_PATH_G8
        print(f"Loading base model: {BASE_MODEL}")
        self.model, _tokenizer = FastVisionModel.from_pretrained(
            BASE_MODEL,
            load_in_4bit=True,
        )
        print(f"Loading G8 adapter from {adapter_path}...")
        assert os.path.isdir(adapter_path), f"Adapter not found: {adapter_path}"
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        FastVisionModel.for_inference(self.model)
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)
        print("G8 model loaded.")

    @modal.method()
    def predict(
        self,
        image_bytes_list: list[bytes],
        chat_text: str = "",
        metadata_text: str = "",
    ) -> dict:
        import io
        from PIL import Image
        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
        return self._predict_core(pil_images, chat_text, metadata_text)

    def _predict_core(self, pil_images: list, chat_text: str, metadata_text: str) -> dict:
        import torch
        content_parts = [{"type": "image", "image": img} for img in pil_images]
        text_parts = []
        if metadata_text: text_parts.append(metadata_text)
        if chat_text:     text_parts.append(f"[Chat Log]\n{chat_text}")
        if text_parts:    content_parts.append({"type": "text", "text": "\n".join(text_parts)})
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_G7},
            {"role": "user",   "content": content_parts},
        ]
        input_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_kwargs = dict(text=[input_text], add_special_tokens=False, return_tensors="pt")
        if pil_images: proc_kwargs["images"] = pil_images
        inputs = self.processor(**proc_kwargs).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=320, do_sample=False, use_cache=True
            )
        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        raw_output = self.processor.decode(trimmed, skip_special_tokens=True).strip()
        parsed = None; valid_json = False
        try:
            parsed = json.loads(raw_output); valid_json = True
        except json.JSONDecodeError:
            pass
        return {"raw_output": raw_output, "parsed": parsed, "valid_json": valid_json}


@app.cls(
    image=g9_inference_image,
    gpu="A100",
    volumes={
        "/model_cache": model_cache,
    },
    container_idle_timeout=1800,
)
class G9Predictor:
    """LoRA_6 G9 predictor — VLM + OpenCV floorplan counting + ResNet size-band.

    Runs the full neuro-symbolic perception layer server-side:
      1. OpenCV `FloorplanCounter` localises the user-supplied floorplan patch
         in the storey reference PNG → `position_context`.
      2. ResNet-18 `SizeBandClassifier` crops at the matched bbox centre and
         predicts `size_band` (window_S/M/L/XL, door_M/L).
      3. Both signals are injected into the `[OpenCV Counting]` / `[Size Band]`
         blocks of the user prompt (matching training format) and passed to
         the G9 LoRA adapter.
      4. Returned `parsed` JSON is post-merged with the perception signals so
         downstream Cypher receives `position_context` and `size_band` even
         when the VLM omits them.
    """

    @modal.enter()
    def load_model(self):
        import sys
        from transformers import AutoProcessor
        from unsloth import FastVisionModel
        from peft import PeftModel

        # Load VLM + G9 adapter via base+PEFT (mirrors G8ModelPredictor).
        print(f"Loading base model: {BASE_MODEL}")
        self.model, _tok = FastVisionModel.from_pretrained(BASE_MODEL, load_in_4bit=True)
        print(f"Loading G9 adapter from {ADAPTER_PATH_G9}...")
        assert os.path.isdir(ADAPTER_PATH_G9), f"G9 adapter not found: {ADAPTER_PATH_G9}"
        self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH_G9)
        FastVisionModel.for_inference(self.model)
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)

        # Make baked perception modules importable.
        if "/perception" not in sys.path:
            sys.path.insert(0, "/perception")
        from floorplan_counter import FloorplanCounter  # noqa: E402
        from cluster_classifier import SizeBandClassifier  # noqa: E402

        self._FloorplanCounter = FloorplanCounter
        self._counter = FloorplanCounter()
        self._size_classifier = SizeBandClassifier(
            checkpoint=Path(RESNET_CHECKPOINT_PATH),
            calibration=Path(CALIBRATION_PATH),
            floorplans_root=Path(FLOORPLANS_DIR),
        )
        print("G9 model + perception layer loaded.")

    @modal.method()
    def predict(
        self,
        image_bytes_list: list[bytes],
        chat_text: str = "",
        metadata_text: str = "",
        floorplan_patch_bytes: bytes | None = None,
        storey_name: str | None = None,
    ) -> dict:
        """Run G9 inference with optional OpenCV+ResNet perception.

        Args:
            image_bytes_list: site photos (PNG/JPEG bytes).
            chat_text: user query text.
            metadata_text: 4D context (project phase, storey hint, …).
            floorplan_patch_bytes: optional floorplan crop bytes. When provided
                with a matching `storey_name`, OpenCV counting + ResNet size-band
                run server-side and feed the VLM prompt.
            storey_name: target storey for the OpenCV match + ResNet crop
                (e.g. "1 - First Floor", "Level 1"). Required for perception;
                if omitted, the VLM still runs but no perception is injected.

        Returns:
            { raw_output, parsed, valid_json,
              perception: { position_context, size_band, ... } | None }
        """
        import io
        from PIL import Image

        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
        perception = self._run_perception(floorplan_patch_bytes, storey_name)
        return self._predict_core(pil_images, chat_text, metadata_text, perception)

    def _run_perception(
        self,
        floorplan_patch_bytes: bytes | None,
        storey_name: str | None,
    ) -> dict | None:
        """Execute OpenCV counter + ResNet size-band; return merged result.

        Storey resolution policy:
          • Caller passes a storey name → use it. If not in calibration, return
            a warning (no auto-fallback so debug stays explicit).
          • Caller passes None / empty → auto-detect: run OpenCV against every
            calibrated storey and pick the highest match_score.
        """
        if not floorplan_patch_bytes:
            return None

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(floorplan_patch_bytes)
            patch_path = tmp.name

        if storey_name:
            storey_key = self._resolve_storey(storey_name)
            if not storey_key:
                return {"warning": f"No calibration for storey {storey_name!r}"}
            scan_storeys = [storey_key]
            scan_mode = "user_specified"
        else:
            scan_storeys = list(self._size_classifier.calibration.keys())
            scan_mode = "auto"

        # Probe each candidate storey; keep the highest OpenCV match_score.
        best: dict | None = None
        scan_log: list[dict] = []
        for candidate_storey in scan_storeys:
            cal_entry = self._size_classifier.calibration[candidate_storey]
            full_path = Path(FLOORPLANS_DIR).parent / cal_entry["png_path"]
            if not full_path.exists():
                full_path = Path(FLOORPLANS_DIR) / Path(cal_entry["png_path"]).name
            if not full_path.exists():
                scan_log.append({
                    "storey": candidate_storey,
                    "warning": f"Storey PNG missing: {cal_entry.get('png_path')}",
                })
                continue
            opencv_result = self._counter.count_from_paths(patch_path, full_path)
            if opencv_result is None:
                scan_log.append({"storey": candidate_storey, "match_score": None})
                continue
            scan_log.append({
                "storey": candidate_storey,
                "match_score": round(float(opencv_result.match_score), 3),
                "position": opencv_result.position,
                "total": opencv_result.total,
            })
            if best is None or opencv_result.match_score > best["_score"]:
                best = {
                    "_score": float(opencv_result.match_score),
                    "storey_key": candidate_storey,
                    "cal_entry": cal_entry,
                    "opencv_result": opencv_result,
                }

        if best is None:
            return {
                "warning": "OpenCV counter could not localise the patch on any storey.",
                "scan_mode": scan_mode,
                "scan_log": scan_log,
            }

        storey_key = best["storey_key"]
        cal_entry = best["cal_entry"]
        opencv_result = best["opencv_result"]
        out: dict = {
            "storey_name": storey_key,
            "scan_mode": scan_mode,
            "scan_log": scan_log,
        }

        out["position_context"] = opencv_result.position_context
        out["position_context_confidence"] = float(opencv_result.confidence)
        out["position_context_source"] = "opencv"
        out["mode"] = opencv_result.mode
        out["match_score"] = round(float(opencv_result.match_score), 3)
        bbox = opencv_result.matched_bbox or {}
        cx_px = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2.0
        cy_px = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) / 2.0
        out["matched_bbox_center_px"] = (cx_px, cy_px)

        # ResNet size-band: invert pixel→world via calibration, then predict().
        try:
            world_xy = self._pixel_to_world(cx_px, cy_px, cal_entry)
            band_pred = self._size_classifier.predict(storey_key, world_xy)
        except Exception as exc:
            band_pred = None
            out["resnet_error"] = str(exc)

        if band_pred is not None:
            out["size_band"] = band_pred.band
            out["size_band_confidence"] = float(band_pred.confidence)
            out["size_band_source"] = "resnet_opencv"
        return out

    @staticmethod
    def _pixel_to_world(px: float, py: float, cal_entry: dict) -> tuple[float, float]:
        """Inverse of SizeBandClassifier._world_to_pixel — pixels → world mm."""
        bbox = cal_entry["world_bbox"]
        pw = float(cal_entry["pixel_size"]["width"])
        ph = float(cal_entry["pixel_size"]["height"])
        span_x = bbox["xmax"] - bbox["xmin"]
        span_y = bbox["ymax"] - bbox["ymin"]
        x_m = bbox["xmin"] + (px / pw) * span_x
        y_m = bbox["ymin"] + ((ph - py) / ph) * span_y
        # Calibration is in metres (WORLD_UNIT_TO_BBOX_UNIT = 1/1000); world_xy_mm
        # is metres × 1000.
        return (x_m * 1000.0, y_m * 1000.0)

    def _resolve_storey(self, storey_name: str) -> str | None:
        """Map a free-text storey label to a calibration key."""
        target = storey_name.strip().lower()
        for key in self._size_classifier.calibration:
            if key.lower() == target:
                return key
        for key in self._size_classifier.calibration:
            if target in key.lower() or key.lower() in target:
                return key
        return None

    def _predict_core(
        self,
        pil_images: list,
        chat_text: str,
        metadata_text: str,
        perception: dict | None,
    ) -> dict:
        import torch

        content_parts = [{"type": "image", "image": img} for img in pil_images]

        text_parts: list[str] = []
        if metadata_text:
            text_parts.append(metadata_text)
        if chat_text:
            text_parts.append(f"[Chat Log]\n{chat_text}")
        # Inject perception evidence in the same format as the LoRA training
        # corpus (constraints_extractor_lora.py:_build_user_text).
        if perception:
            if perception.get("position_context"):
                conf = perception.get("position_context_confidence", 0.0)
                mode = perception.get("mode", "")
                lines = [
                    "[OpenCV Counting]",
                    f"  position_context: {perception['position_context']}",
                    f"  confidence: {conf:.2f}",
                ]
                if mode == "patch_only":
                    lines.append("  note: patch-only fallback estimate; use cautiously.")
                else:
                    lines.append("  note: derived from the floorplan patch matched against a larger floorplan reference.")
                text_parts.append("\n".join(lines))
            if perception.get("size_band"):
                band_conf = perception.get("size_band_confidence", 0.0)
                text_parts.append(
                    "[Size Band]\n"
                    f"  size_band: {perception['size_band']}\n"
                    f"  confidence: {band_conf:.2f}"
                )
        if text_parts:
            content_parts.append({"type": "text", "text": "\n".join(text_parts)})

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_G9},
            {"role": "user",   "content": content_parts},
        ]
        input_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_kwargs = dict(text=[input_text], add_special_tokens=False, return_tensors="pt")
        if pil_images:
            proc_kwargs["images"] = pil_images
        inputs = self.processor(**proc_kwargs).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=384, do_sample=False, use_cache=True
            )
        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        raw_output = self.processor.decode(trimmed, skip_special_tokens=True).strip()

        parsed = None
        valid_json = False
        try:
            parsed = json.loads(raw_output)
            valid_json = True
        except json.JSONDecodeError:
            pass

        # Post-merge: ensure perception signals reach the planner even if the
        # VLM dropped them. Confidence/source fields are stamped from the
        # perception layer, not the VLM's self-report.
        if valid_json and isinstance(parsed, dict) and perception:
            if perception.get("position_context") and not parsed.get("position_context"):
                parsed["position_context"] = perception["position_context"]
            if perception.get("position_context_confidence") is not None:
                parsed["position_context_confidence"] = perception["position_context_confidence"]
                parsed["position_context_source"] = perception.get("position_context_source", "opencv")
            if perception.get("size_band"):
                parsed["size_band"] = perception["size_band"]
                parsed["size_band_confidence"] = perception.get("size_band_confidence")
                parsed["size_band_source"] = perception.get("size_band_source", "resnet_opencv")

        return {
            "raw_output": raw_output,
            "parsed": parsed,
            "valid_json": valid_json,
            "perception": perception,
        }


# ── CLI entry point for testing ──────────────────────────────────────────────

@app.local_entrypoint()
def main(
    chat: str = "There's a crack on the window next to the railing, third floor",
    image: str = "",
):
    """Quick CLI test of the inference endpoint."""
    predictor = LoRA3Predictor()

    image_bytes_list = []
    if image and Path(image).exists():
        image_bytes_list.append(Path(image).read_bytes())

    metadata = (
        "[4D Task Status] TASK_0001: Window inspection — IN_PROGRESS\n"
        "[Project Phase] Interior Fit-out\n"
        "[Location] 3 - Third Floor"
    )

    print(f"Chat: {chat}")
    print(f"Images: {len(image_bytes_list)}")
    print("Running inference...")

    result = predictor.predict.remote(
        image_bytes_list=image_bytes_list,
        chat_text=chat,
        metadata_text=metadata,
    )

    print(f"\nValid JSON: {result['valid_json']}")
    print(f"Raw output:\n{result['raw_output']}")
    if result["parsed"]:
        print(f"\nParsed:")
        print(json.dumps(result["parsed"], indent=2))


@app.local_entrypoint()
def test_g8(
    chat: str = "There's a crack on the window next to the railing, third floor",
    image: str = "",
):
    """Quick CLI test of the G8Predictor (G4-Ultimate + lora_system_g7 prompt)."""
    predictor = G8Predictor()

    image_bytes_list = []
    if image and Path(image).exists():
        image_bytes_list.append(Path(image).read_bytes())

    metadata = (
        "[4D Task Status] TASK_0001: Window inspection — IN_PROGRESS\n"
        "[Project Phase] Interior Fit-out\n"
        "[Location] 3 - Third Floor"
    )

    print(f"Chat: {chat}")
    print(f"Images: {len(image_bytes_list)}")
    print("Running G8Predictor (G4-Ultimate) inference...")

    result = predictor.predict.remote(
        image_bytes_list=image_bytes_list,
        chat_text=chat,
        metadata_text=metadata,
    )

    print(f"\nValid JSON: {result['valid_json']}")
    print(f"Raw output:\n{result['raw_output']}")
    if result["parsed"]:
        print(f"\nParsed:")
        print(json.dumps(result["parsed"], indent=2))
