"""
MSCD LoRA Test Inference — Visual Verification

Runs LoRA inference on a few cases and prints detailed input/output
for visual inspection.

Usage:
    modal run training/test_inference.py
    modal run training/test_inference.py --adapter-dir /mscd-lora/checkpoint-180
    modal run training/test_inference.py --limit 3
    modal run training/test_inference.py --condition-override MA
"""

import json
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Optional

import modal

# ── Reuse infra from eval.py ─────────────────────────────────────────────────

DATA_ROOT = Path(__file__).parent.parent.parent / "data_curation"
CASES_FILE = DATA_ROOT / "datasets" / "synth_v0.3" / "cases_v3_filtered.jsonl"
IMGS_DIR = DATA_ROOT / "datasets" / "synth_v0.3" / "cases" / "imgs"
PLANS_DIR = DATA_ROOT / "datasets" / "synth_v0.3" / "cases" / "plans"

app = modal.App("mscd-lora-test-inference")

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
    .add_local_file(str(CASES_FILE), remote_path="/data/cases_v3_filtered.jsonl")
    .add_local_dir(str(IMGS_DIR), remote_path="/data/images/imgs")
    .add_local_dir(str(PLANS_DIR), remote_path="/data/images/plans")
)

model_cache = modal.Volume.from_name("mscd-model-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("mscd-checkpoints", create_if_missing=True)

# ── Inline helpers (from eval.py) ────────────────────────────────────────────

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

CONDITION_CONFIGS = {
    "A1": {"use_images": False, "use_floorplan": False, "chat_blur": False, "4d_metadata": True},
    "B1": {"use_images": True, "use_floorplan": False, "chat_blur": True, "4d_metadata": False},
    "C3": {"use_images": True, "use_floorplan": True, "chat_blur": False, "4d_metadata": True},
    "MA": {"use_images": False, "use_floorplan": False, "chat_blur": False, "4d_metadata": True},
    "MB": {"use_images": True, "use_floorplan": False, "chat_blur": False, "4d_metadata": True},
    "MC": {"use_images": True, "use_floorplan": True, "chat_blur": False, "4d_metadata": True},
}

LOCAL_PREFIX = "datasets/synth_v0.3/cases/"


def _remap_to_modal(path: str) -> str:
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


def _apply_condition_mask(case: dict, condition: str) -> dict:
    import copy
    overrides = CONDITION_CONFIGS.get(condition, {})
    masked = copy.deepcopy(case)
    inputs = masked.get("inputs", {})
    if not overrides.get("use_images", True):
        inputs["images"] = []
    if not overrides.get("use_floorplan", False):
        inputs.pop("floorplan_patch", None)
    if not overrides.get("4d_metadata", True):
        if "project_context" in inputs:
            inputs["project_context"]["4d_task_status"] = "N/A"
    masked["inputs"] = inputs
    return masked


def _build_user_text(case: dict) -> str:
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
    user_content = []
    inputs = case.get("inputs", {})
    for img in inputs.get("images", []):
        modal_path = _remap_to_modal(img)
        if os.path.exists(modal_path):
            user_content.append({"type": "image", "image": f"file://{modal_path}"})
        else:
            user_content.append({"type": "text", "text": f"[IMAGE MISSING: {modal_path}]"})
    fp = inputs.get("floorplan_patch")
    if fp:
        modal_path = _remap_to_modal(fp)
        if os.path.exists(modal_path):
            user_content.append({"type": "image", "image": f"file://{modal_path}"})
    user_content.append({"type": "text", "text": _build_user_text(case)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
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
    timeout=30 * 60,  # 30 min (test only)
)
def test_inference(
    adapter_dir: str = "/mscd-lora/final",
    limit: int = 3,
    condition_override: str = "",
):
    """Run LoRA inference on a few cases with detailed output."""
    import torch
    from unsloth import FastVisionModel
    from qwen_vl_utils import process_vision_info

    SEP = "=" * 70

    # ── Load model ────────────────────────────────────────────────────────
    adapter_path = f"/checkpoints{adapter_dir}"
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        contents = os.listdir(adapter_path) if os.path.exists(adapter_path) else []
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_path}. Contents: {contents}"
        )

    print(SEP)
    print("  MSCD LoRA Test Inference")
    print(SEP)
    print(f"  Adapter:    {adapter_path}")
    print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print(f"  Limit:      {limit} cases")
    if condition_override:
        print(f"  Condition:  {condition_override} (override)")
    print()

    print("Loading model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        load_in_4bit=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    FastVisionModel.for_inference(model)
    print("Model ready.\n")

    # ── Load cases ────────────────────────────────────────────────────────
    cases = []
    with open("/data/cases_v3_filtered.jsonl") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    cases = cases[:limit]

    # ── Run inference with detailed output ────────────────────────────────
    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"case_{idx}")
        condition = condition_override or case.get("bench", {}).get("condition", "")
        ground_truth = case.get("ground_truth", {})

        # Apply condition mask
        masked_case = _apply_condition_mask(case, condition)
        messages = _build_messages(masked_case)

        # Summarise inputs
        inputs_meta = masked_case.get("inputs", {})
        n_images = len(inputs_meta.get("images", []))
        has_floorplan = "floorplan_patch" in inputs_meta
        chat = inputs_meta.get("chat_history", [])
        ctx_4d = inputs_meta.get("project_context", {}).get("4d_task_status", "")

        print(f"\n{'#' * 70}")
        print(f"  CASE {idx}/{len(cases)}: {case_id}")
        print(f"  Condition: {condition}")
        print(f"{'#' * 70}")

        # ── INPUT ─────────────────────────────────────────────────────
        print(f"\n{'─' * 40} INPUT {'─' * 40}")
        print(f"  Images:     {n_images}")
        print(f"  Floorplan:  {has_floorplan}")
        print(f"  4D Status:  {ctx_4d or '(none)'}")
        print(f"  Chat ({len(chat)} messages):")
        for msg in chat[:5]:  # show first 5
            role = msg.get("role", "?")
            text = msg.get("text", "")
            wrapped = textwrap.shorten(text, width=80, placeholder="...")
            print(f"    [{role}] {wrapped}")
        if len(chat) > 5:
            print(f"    ... ({len(chat) - 5} more messages)")

        # ── USER PROMPT (what model sees) ─────────────────────────────
        user_text = _build_user_text(masked_case)
        print(f"\n{'─' * 40} PROMPT {'─' * 39}")
        for line in user_text.split("\n"):
            print(f"  {line}")

        # ── INFERENCE ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            if image_inputs:
                tok_inputs = tokenizer(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to(model.device)
            else:
                tok_inputs = tokenizer(
                    text=[text], add_special_tokens=False, return_tensors="pt",
                ).to(model.device)

            n_input_tokens = len(tok_inputs.input_ids[0])

            with torch.no_grad():
                output_ids = model.generate(
                    **tok_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                )

            n_output_tokens = len(output_ids[0]) - n_input_tokens
            trimmed = output_ids[0][n_input_tokens:]
            raw_output = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
            latency_ms = (time.perf_counter() - t0) * 1000
            error = None

        except Exception as e:
            raw_output = ""
            latency_ms = (time.perf_counter() - t0) * 1000
            n_input_tokens = 0
            n_output_tokens = 0
            error = str(e)

        # ── RAW OUTPUT ────────────────────────────────────────────────
        print(f"\n{'─' * 40} RAW OUTPUT {'─' * 35}")
        if error:
            print(f"  ERROR: {error}")
        else:
            print(f"  Tokens: {n_input_tokens} in -> {n_output_tokens} out")
            print(f"  Latency: {latency_ms:.0f} ms")
            print()
            for line in raw_output.split("\n"):
                print(f"  {line}")

        # ── PARSED OUTPUT ─────────────────────────────────────────────
        parsed = _parse_json(raw_output) if not error else None
        print(f"\n{'─' * 40} PARSED {'─' * 39}")
        if parsed:
            print(f"  storey_name:   {parsed.get('storey_name')}")
            print(f"  ifc_class:     {parsed.get('ifc_class')}")
            print(f"  near_keywords: {parsed.get('near_keywords', [])}")
            print(f"  relations:     {parsed.get('relations', [])}")
        else:
            print(f"  PARSE FAILED (raw: {raw_output[:100]})")

        # ── GROUND TRUTH ──────────────────────────────────────────────
        gt_id = ground_truth.get("target_guid", "?")
        gt_class = ground_truth.get("target_ifc_class", "?")
        gt_storey = ground_truth.get("target_storey", "?")
        gt_name = ground_truth.get("target_name", "?")

        print(f"\n{'─' * 40} GROUND TRUTH {'─' * 33}")
        print(f"  element_id:  {gt_id}")
        print(f"  ifc_class:   {gt_class}")
        print(f"  storey:      {gt_storey}")
        print(f"  name:        {gt_name}")

        # ── MATCH CHECK ───────────────────────────────────────────────
        if parsed:
            class_match = (parsed.get("ifc_class") or "").lower() == (gt_class or "").lower()
            storey_match = gt_storey and parsed.get("storey_name") and \
                           gt_storey.lower() in parsed["storey_name"].lower()
            print(f"\n  Class match:  {'YES' if class_match else 'NO'}")
            print(f"  Storey match: {'YES' if storey_match else 'NO (approx check)'}")

    print(f"\n{SEP}")
    print(f"  Test complete — {len(cases)} cases processed")
    print(SEP)


# ── CLI entry point ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    adapter_dir: str = "/mscd-lora/final",
    limit: int = 3,
    condition_override: str = "",
):
    """Run test inference with detailed output."""
    print(f"Launching test inference on Modal...")
    print(f"  Adapter: {adapter_dir}")
    print(f"  Limit:   {limit}")
    if condition_override:
        print(f"  Condition override: {condition_override}")

    test_inference.remote(
        adapter_dir=adapter_dir,
        limit=limit,
        condition_override=condition_override,
    )
