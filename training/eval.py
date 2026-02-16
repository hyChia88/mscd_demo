"""
MSCD VLM LoRA Evaluation — Modal GPU Inference Script

Runs LoRA constraint extraction on Modal GPU for all evaluation cases.
Outputs pre-computed constraints JSONL that can be fed back to the local
pipeline via: python script/run.py --profile v2_lora --precomputed <file>

This separates the GPU-heavy LoRA inference (Modal A100) from retrieval +
scoring (local CPU), keeping eval methodologically clean — same hardware
as training.

Usage:
    modal run training/eval.py
    modal run training/eval.py --adapter-dir /mscd-lora/final
    modal run training/eval.py --adapter-dir /mscd-lora/checkpoint-180
    modal run training/eval.py --limit 5  # Quick test
"""

import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# ── Local data paths ─────────────────────────────────────────────────────────

DATA_ROOT = Path(__file__).parent.parent.parent / "data_curation"
CASES_FILE = DATA_ROOT / "datasets" / "synth_v0.3" / "cases_v3_filtered.jsonl"
IMGS_DIR = DATA_ROOT / "datasets" / "synth_v0.3" / "cases" / "imgs"
PLANS_DIR = DATA_ROOT / "datasets" / "synth_v0.3" / "cases" / "plans"

# ── Modal infrastructure ─────────────────────────────────────────────────────

app = modal.App("mscd-vlm-lora-eval")

# Same image as training — ensures identical environment
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "unsloth",
        "qwen-vl-utils",
        "datasets==4.3.0",
        "hf-transfer",
    )
    .run_commands(
        "pip install --no-deps --force-reinstall "
        "'unsloth @ git+https://github.com/unslothai/unsloth.git'"
    )
    .pip_install("transformers==4.56.2")
    .run_commands("pip install --no-deps trl==0.22.2")
    .env({"HF_HOME": "/model_cache"})
    # Bake evaluation data into the image
    .add_local_file(str(CASES_FILE), remote_path="/data/cases_v3_filtered.jsonl")
    .add_local_dir(str(IMGS_DIR), remote_path="/data/images/imgs")
    .add_local_dir(str(PLANS_DIR), remote_path="/data/images/plans")
)

model_cache = modal.Volume.from_name("mscd-model-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("mscd-checkpoints", create_if_missing=True)


# ── System prompt (must match training data from 7_prepare_lora_data.py) ─────

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
    "- storey_name must match exact IFC storey names (e.g., '1 - First Floor', "
    "'Level 1', '-1 - Garage')\n"
    "- ifc_class must use Ifc prefix (e.g., 'IfcWindow' not 'window')\n"
    "- Be conservative: use null if uncertain\n"
    "- Look at the image carefully for element type and defect clues"
)


# ── Condition masking (inlined from src/v2/condition_mask.py) ────────────────

CONDITION_CONFIGS = {
    "A1": {"use_images": False, "use_floorplan": False, "chat_blur": False, "4d_metadata": True},
    "A2": {"use_images": False, "use_floorplan": False, "chat_blur": True, "4d_metadata": True},
    "A3": {"use_images": False, "use_floorplan": False, "chat_blur": True, "4d_metadata": True, "4d_enhanced": True},
    "B1": {"use_images": True, "use_floorplan": False, "chat_blur": True, "4d_metadata": False},
    "B2": {"use_images": True, "use_floorplan": False, "chat_blur": True, "4d_metadata": False, "force_clip": True},
    "B3": {"use_images": True, "use_floorplan": False, "chat_blur": False, "4d_metadata": False},
    "C1": {"use_images": False, "use_floorplan": True, "chat_blur": False, "4d_metadata": False},
    "C2": {"use_images": True, "use_floorplan": True, "chat_blur": True, "4d_metadata": False},
    "C3": {"use_images": True, "use_floorplan": True, "chat_blur": False, "4d_metadata": True, "4d_enhanced": True},
}

BLUR_REPLACEMENTS = {
    "window": "opening", "Window": "Opening", "door": "opening", "Door": "Opening",
    "wall": "surface", "Wall": "Surface", "slab": "surface", "Slab": "Surface",
    "sixth": "upper", "Sixth": "Upper", "first": "lower", "First": "Lower",
    "second": "middle", "Second": "Middle", "third": "middle", "Third": "Middle",
    "fourth": "middle", "Fourth": "Middle", "fifth": "upper", "Fifth": "Upper",
    "north": "side", "North": "Side", "south": "side", "South": "Side",
    "east": "side", "East": "Side", "west": "side", "West": "Side",
    "elevator": "area", "Elevator": "Area", "stair": "area", "Stair": "Area",
    "entrance": "location", "Entrance": "Location",
}


def _blur_text(text: str) -> str:
    for old, new in BLUR_REPLACEMENTS.items():
        text = re.sub(r'\b' + re.escape(old) + r'\b', new, text)
    return text


def apply_condition_mask(case: dict, condition: str) -> dict:
    """Apply A1-C3 condition masking (mirrors src/v2/condition_mask.py)."""
    overrides = CONDITION_CONFIGS.get(condition, {})
    masked = copy.deepcopy(case)
    inputs = masked.get("inputs", {})

    if overrides.get("chat_blur", False):
        if "chat_history" in inputs:
            inputs["chat_history"] = [
                {"role": m.get("role", ""), "text": _blur_text(m.get("text", ""))}
                for m in inputs["chat_history"]
            ]

    if not overrides.get("use_images", True):
        inputs["images"] = []

    if not overrides.get("use_floorplan", False):
        inputs.pop("floorplan_patch", None)

    if not overrides.get("4d_metadata", True):
        if "project_context" in inputs:
            inputs["project_context"]["4d_task_status"] = "N/A"

    masked["inputs"] = inputs
    return masked


# ── Image path remapping ─────────────────────────────────────────────────────

LOCAL_PREFIX = "datasets/synth_v0.3/cases/"


def _remap_to_modal(path: str) -> str:
    """Remap local case image path to Modal container path."""
    path_str = str(path)
    if LOCAL_PREFIX in path_str:
        rel = path_str.split(LOCAL_PREFIX, 1)[1]
        return f"/data/images/{rel}"
    p = Path(path_str)
    if p.name.startswith("img_"):
        return f"/data/images/imgs/{p.name}"
    elif p.name.startswith("plan_"):
        return f"/data/images/plans/{p.name}"
    return path_str


# ── Build inference messages (mirrors constraints_extractor_lora.py) ─────────

def _build_user_text(case: dict) -> str:
    """Build user text — must match 7_prepare_lora_data.py exactly."""
    parts = []
    ctx = case.get("inputs", {}).get("project_context", {})

    task_status = ctx.get("4d_task_status", "")
    if task_status:
        parts.append(f"[4D Task Status] {task_status}")

    project_phase = ctx.get("project_phase", "")
    if project_phase:
        parts.append(f"[Project Phase] {project_phase}")

    chat = case.get("inputs", {}).get("chat_history", [])
    if chat:
        parts.append("[Chat Log]")
        for msg in chat:
            role = msg.get("role", "User")
            text = msg.get("text", "")
            parts.append(f"  {role}: {text}")

    query = case.get("query_text", "")
    if query:
        parts.append(f"\n[Query] {query}")

    parts.append("\nExtract the search constraints as JSON.")
    return "\n".join(parts)


def _build_messages(case: dict) -> list:
    """Build ChatML messages for VLM inference."""
    user_content = []
    inputs = case.get("inputs", {})

    # Site photos
    for img in inputs.get("images", []):
        modal_path = _remap_to_modal(img)
        if os.path.exists(modal_path):
            user_content.append({"type": "image", "image": f"file://{modal_path}"})
        else:
            print(f"    [WARN] Image not found: {modal_path}")

    # Floorplan patch
    fp = inputs.get("floorplan_patch")
    if fp:
        modal_path = _remap_to_modal(fp)
        if os.path.exists(modal_path):
            user_content.append({"type": "image", "image": f"file://{modal_path}"})

    # Text prompt
    user_content.append({"type": "text", "text": _build_user_text(case)})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_json(text: str) -> Optional[dict]:
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


# ── Modal GPU function ───────────────────────────────────────────────────────

@app.function(
    image=eval_image,
    gpu="A100",
    volumes={
        "/model_cache": model_cache,
        "/checkpoints": checkpoint_vol,
    },
    timeout=2 * 60 * 60,  # 2 hours
)
def run_eval(
    adapter_dir: str = "/mscd-lora/final",
    limit: int = 0,
):
    """Run LoRA constraint extraction on all cases (Modal A100)."""
    import torch
    from unsloth import FastVisionModel
    from qwen_vl_utils import process_vision_info

    # ── 1. Locate adapter ────────────────────────────────────────────────
    adapter_path = f"/checkpoints{adapter_dir}"
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        # Try without nesting
        contents = os.listdir(adapter_path) if os.path.exists(adapter_path) else []
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_path}.\n"
            f"  Contents: {contents}"
        )

    print("=" * 60)
    print("MSCD LoRA Evaluation (Modal A100)")
    print("=" * 60)
    print(f"  Adapter:  {adapter_path}")
    print(f"  GPU:      {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ── 2. Load model + adapter ──────────────────────────────────────────
    print("\nLoading base model (4-bit)...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        load_in_4bit=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    FastVisionModel.for_inference(model)
    print("Model ready.\n")

    # ── 3. Load cases ────────────────────────────────────────────────────
    cases = []
    with open("/data/cases_v3_filtered.jsonl") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    if limit > 0:
        cases = cases[:limit]
    print(f"Cases: {len(cases)}")

    # ── 4. Run inference ─────────────────────────────────────────────────
    results = []
    n_parsed = 0
    total_latency = 0.0

    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"case_{idx}")
        condition = case.get("bench", {}).get("condition", "")

        # Apply condition mask (same as local pipeline)
        masked_case = apply_condition_mask(case, condition)

        # Build messages
        messages = _build_messages(masked_case)

        # Count images for logging
        n_images = sum(
            1 for c in messages[1]["content"]
            if isinstance(c, dict) and c.get("type") == "image"
        )

        t0 = time.perf_counter()

        try:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision inputs
            image_inputs, video_inputs = process_vision_info(messages)

            # Tokenize
            if image_inputs:
                inputs = tokenizer(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
            else:
                inputs = tokenizer(
                    text=[text],
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(model.device)

            # Generate (greedy, short output — JSON only)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                )

            # Trim input tokens
            trimmed = output_ids[0][len(inputs.input_ids[0]):]
            raw_output = tokenizer.decode(trimmed, skip_special_tokens=True).strip()

            latency_ms = (time.perf_counter() - t0) * 1000

            # Parse JSON
            parsed = _parse_json(raw_output)
            if parsed:
                n_parsed += 1
                status = "OK"
            else:
                parsed = {}
                status = "PARSE_FAIL"

        except Exception as e:
            raw_output = f"ERROR: {e}"
            parsed = {}
            latency_ms = (time.perf_counter() - t0) * 1000
            status = "ERROR"

        total_latency += latency_ms

        result = {
            "case_id": case_id,
            "condition": condition,
            "constraints": {
                "storey_name": parsed.get("storey_name"),
                "ifc_class": parsed.get("ifc_class"),
                "near_keywords": parsed.get("near_keywords", []),
                "relations": parsed.get("relations", []),
            },
            "raw_output": raw_output[:500],
            "latency_ms": round(latency_ms, 1),
            "status": status,
        }
        results.append(result)

        ifc = parsed.get("ifc_class", "null")
        storey = parsed.get("storey_name", "null")
        print(f"  [{idx:>3}/{len(cases)}] {case_id}  cond={condition}  "
              f"imgs={n_images}  {latency_ms:.0f}ms  {status}  "
              f"class={ifc}  storey={storey}")

    # ── 5. Save results to Modal volume ──────────────────────────────────
    # Use adapter dir name as tag (e.g., "final" or "checkpoint-180")
    tag = adapter_dir.rstrip("/").split("/")[-1]
    output_path = f"/checkpoints/mscd-lora/eval_constraints_{tag}.jsonl"

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    checkpoint_vol.commit()

    # ── 6. Summary ───────────────────────────────────────────────────────
    parse_rate = n_parsed / len(cases) if cases else 0
    avg_latency = total_latency / len(cases) if cases else 0

    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Adapter:      {adapter_dir}")
    print(f"  Cases:        {len(cases)}")
    print(f"  Parse rate:   {n_parsed}/{len(cases)} ({parse_rate:.1%})")
    print(f"  Avg latency:  {avg_latency:.0f} ms/case")
    print(f"  Output:       {output_path}")
    print(f"\nDownload with:")
    print(f"  modal volume get mscd-checkpoints "
          f"/mscd-lora/eval_constraints_{tag}.jsonl "
          f"./logs/evaluations/")
    print(f"\nRun local pipeline with pre-computed constraints:")
    print(f"  python script/run.py --profile v2_lora \\")
    print(f"    --cases ../data_curation/datasets/synth_v0.3/"
          f"cases_v3_filtered.jsonl \\")
    print(f"    --precomputed logs/evaluations/eval_constraints_{tag}.jsonl")

    return {
        "total": len(cases),
        "parsed": n_parsed,
        "parse_rate": parse_rate,
        "avg_latency_ms": avg_latency,
        "output_path": output_path,
        "tag": tag,
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    adapter_dir: str = "/mscd-lora/final",
    limit: int = 0,
):
    """Launch LoRA evaluation on Modal GPU."""
    print("Launching MSCD LoRA evaluation on Modal...")
    print(f"  Adapter:  {adapter_dir}")
    print(f"  Cases:    {CASES_FILE}")
    if limit > 0:
        print(f"  Limit:    {limit} cases")

    result = run_eval.remote(
        adapter_dir=adapter_dir,
        limit=limit,
    )

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Parse rate: {result['parsed']}/{result['total']} "
          f"({result['parse_rate']:.1%})")
    print(f"  Avg latency: {result['avg_latency_ms']:.0f} ms/case")
    tag = result["tag"]
    print(f"\nDownload results:")
    print(f"  modal volume get mscd-checkpoints "
          f"/mscd-lora/eval_constraints_{tag}.jsonl "
          f"./logs/evaluations/")
    print(f"\nRun local pipeline:")
    print(f"  python script/run.py --profile v2_lora \\")
    print(f"    --cases ../data_curation/datasets/synth_v0.3/"
          f"cases_v3_filtered.jsonl \\")
    print(f"    --precomputed logs/evaluations/eval_constraints_{tag}.jsonl")
