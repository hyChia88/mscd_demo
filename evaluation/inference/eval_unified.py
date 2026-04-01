#!/usr/bin/env python3
"""Unified evaluation script: run multiple LoRA adapters on the unified test set.

Supports: LoRA2 (v2_lora_qwen), LoRA5-r16 (v5_complex_lora_qwen),
           LoRA5-r32 (v5_lora_qwen_r32), and zero-shot baseline.

All runs share the same Modal image, cases file, and condition masking.
Output goes to a NEW folder on the checkpoint volume to avoid conflicts.

Usage:
    # LoRA5-r32, FP condition, full test set
    modal run evaluation/inference/eval_unified.py \
        --adapter /mscd-unified-eval/v5_lora_qwen_r32 \
        --modality FP --tag r32_FP

    # LoRA2, MC condition
    modal run evaluation/inference/eval_unified.py \
        --adapter /mscd-unified-eval/v2_lora_qwen \
        --modality MC --tag lora2_MC \
        --system-prompt lora2

    # All runs (use eval_unified.sh for batch)
    bash evaluation/inference/eval_unified.sh
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import modal

# ── Local data + config paths ─────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
DATA_ROOT     = PROJECT_ROOT.parent / "data_curation"
PROFILES_YAML = PROJECT_ROOT / "profiles.yaml"
PROMPTS_YAML  = PROJECT_ROOT / "prompts" / "constraints_extraction.yaml"
COND_MASK_PY  = PROJECT_ROOT / "src" / "v2" / "condition_mask.py"

# Unified test set
UNIFIED_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_unified_test.jsonl"

# ── Adapter dirs (local paths — will be uploaded to Modal volume) ─────────────
ADAPTERS_DIR = PROJECT_ROOT / "models" / "adapters"
ADAPTER_MAP = {
    "v2_lora_qwen":          ADAPTERS_DIR / "v2_lora_qwen",
    "v5_complex_lora_qwen":  ADAPTERS_DIR / "v5_complex_lora_qwen",
    "v5_lora_qwen_r32":      ADAPTERS_DIR / "v5_lora_qwen_r32",
}

# ── v0.4 image dirs ──────────────────────────────────────────────────────────
AP_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.4_ap"  / "cases" / "imgs"
AP_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.4_ap"  / "cases" / "plans"
BH_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.4_bh"  / "cases" / "imgs"
BH_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.4_bh"  / "cases" / "plans"
DXA_IMGS_DIR  = DATA_ROOT / "datasets" / "synth_v0.4_dxa" / "cases" / "imgs"
DXA_PLANS_DIR = DATA_ROOT / "datasets" / "synth_v0.4_dxa" / "cases" / "plans"

# ── v0.5 image dirs ──────────────────────────────────────────────────────────
V05_AP_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.5"     / "imgs"
V05_AP_WIRE_DIR   = DATA_ROOT / "datasets" / "synth_v0.5"     / "renders" / "wireframes"
V05_AP_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.5"     / "floorplans"
V05_BH_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.5_bh"  / "imgs"
V05_BH_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.5_bh"  / "floorplans"
V05_DXA_IMGS_DIR  = DATA_ROOT / "datasets" / "synth_v0.5_dxa" / "imgs"
V05_DXA_PLANS_DIR = DATA_ROOT / "datasets" / "synth_v0.5_dxa" / "floorplans"


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_LORA5 = (
    "You are a construction site assistant that extracts IFC element search constraints "
    "from multimodal inputs (floorplans, site photos, chat messages, 4D metadata). "
    "The floorplan is your PRIMARY source for spatial reasoning — analyze door arcs, "
    "wall outlines, opening breaks, and element positions to extract spatial relationships "
    "(FILLS, ADJACENT_TO, CONTINUOUS, NEXT_TO, CONNECTS_TO). "
    "Site photos provide supplementary context for element condition and state. "
    "Output valid JSON only — no markdown, no explanation."
)

SYSTEM_PROMPT_LORA2 = (
    "You are a construction site assistant that extracts search constraints from "
    "multimodal inputs (photos, floorplans, chat messages, and metadata). "
    "Given the conversation and any attached images, extract structured JSON constraints "
    "to identify the BIM element being discussed.\n\n"
    "Output ONLY valid JSON with these fields:\n"
    '{\n'
    '  "storey_name": "exact floor name or null",\n'
    '  "ifc_class": "IfcWall|IfcWindow|IfcDoor|IfcSlab|IfcStair|IfcRailing|... or null",\n'
    '  "near_keywords": ["spatial", "hints"],\n'
    '  "relations": ["spatial_relationships"],\n'
    '  "space_name": "containing room or space name or null",\n'
    '  "target_name_keyword": "unique equipment ID like AHU-03 or null",\n'
    '  "neighbor_type": "IfcClass of adjacent reference element or null"\n'
    '}\n\n'
    "Rules:\n"
    "- storey_name must match exact IFC storey names (e.g., '1 - First Floor', 'Level 1', '-1 - Garage')\n"
    "- ifc_class must use Ifc prefix (e.g., 'IfcWindow' not 'window')\n"
    "- space_name: extract room/space if user says 'in the kitchen', 'room 601'; null otherwise\n"
    "- target_name_keyword: extract specific equipment IDs like 'AHU-03'; null for generic names\n"
    "- neighbor_type: extract if user says 'next to the column'; must use Ifc prefix; null otherwise\n"
    "- Be conservative: use null if uncertain\n"
    "- Look at the image carefully for element type and defect clues"
)


# ── Modal infrastructure ─────────────────────────────────────────────────────

app = modal.App("mscd-unified-eval")

def _build_eval_image() -> modal.Image:
    """Build unified eval image while tolerating missing legacy dataset dirs."""
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "unsloth",
            "qwen-vl-utils",
            "datasets==4.3.0",
            "hf-transfer",
            "pyyaml",
        )
        .run_commands(
            "pip install --no-deps --force-reinstall "
            "'unsloth @ git+https://github.com/unslothai/unsloth.git'"
        )
        .pip_install("transformers==4.56.2")
        .run_commands("pip install --no-deps trl==0.22.2")
        .env({"HF_HOME": "/model_cache"})
        .add_local_file(str(PROFILES_YAML), remote_path="/app/profiles.yaml")
        .add_local_file(str(PROMPTS_YAML), remote_path="/app/constraints_extraction.yaml")
        .add_local_file(str(COND_MASK_PY), remote_path="/app/condition_mask.py")
    )

    if UNIFIED_CASES.exists():
        image = image.add_local_file(str(UNIFIED_CASES), remote_path="/data/cases_unified_test.jsonl")

    for local_dir, remote_dir in [
        (AP_IMGS_DIR, "/data/images/ap/imgs"),
        (AP_PLANS_DIR, "/data/images/ap/plans"),
        (BH_IMGS_DIR, "/data/images/bh/imgs"),
        (BH_PLANS_DIR, "/data/images/bh/plans"),
        (DXA_IMGS_DIR, "/data/images/dxa/imgs"),
        (DXA_PLANS_DIR, "/data/images/dxa/plans"),
        (V05_AP_IMGS_DIR, "/data/images/v05_ap/imgs"),
        (V05_AP_WIRE_DIR, "/data/images/v05_ap/wireframes"),
        (V05_AP_PLANS_DIR, "/data/images/v05_ap/plans"),
        (V05_BH_IMGS_DIR, "/data/images/v05_bh/imgs"),
        (V05_BH_PLANS_DIR, "/data/images/v05_bh/plans"),
        (V05_DXA_IMGS_DIR, "/data/images/v05_dxa/imgs"),
        (V05_DXA_PLANS_DIR, "/data/images/v05_dxa/plans"),
    ]:
        if local_dir.exists():
            image = image.add_local_dir(str(local_dir), remote_path=remote_dir)

    return image


eval_image = _build_eval_image()

model_cache    = modal.Volume.from_name("mscd-model-cache",  create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("mscd-checkpoints",  create_if_missing=True)


# ── Runtime helpers (run inside Modal container) ──────────────────────────────

def _load_condition_configs() -> Dict[str, Dict]:
    import yaml
    with open("/app/profiles.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("conditions", {})


def _load_condition_mask():
    sys.path.insert(0, "/app")
    from condition_mask import ConditionMask
    return ConditionMask


def _remap_to_modal(path: str, case_id: str = "") -> str:
    """Remap local case image path to Modal container path."""
    path_str = str(path)
    p = Path(path_str)
    filename = p.name

    # v0.5 paths
    if "synth_v0.5" in path_str:
        if "wireframes" in path_str or "renders" in path_str:
            if "_dxa" in path_str or "v0.5_dxa" in path_str:
                return f"/data/images/v05_dxa/wireframes/{filename}"
            return f"/data/images/v05_ap/wireframes/{filename}"
        if "floorplan" in path_str:
            if "_dxa" in path_str or "v0.5_dxa" in path_str:
                return f"/data/images/v05_dxa/plans/{filename}"
            if "_bh" in path_str or "v0.5_bh" in path_str:
                return f"/data/images/v05_bh/plans/{filename}"
            return f"/data/images/v05_ap/plans/{filename}"
        if "_dxa" in path_str or "v0.5_dxa" in path_str:
            return f"/data/images/v05_dxa/imgs/{filename}"
        if "_bh" in path_str or "v0.5_bh" in path_str:
            return f"/data/images/v05_bh/imgs/{filename}"
        return f"/data/images/v05_ap/imgs/{filename}"

    # v0.4 paths
    model_key = "ap"
    if "v0.4_dxa" in path_str or "_DXA_" in case_id:
        model_key = "dxa"
    elif "v0.4_bh" in path_str or "_BH_" in case_id:
        model_key = "bh"

    _MODEL_IMAGE_ROOT = {"ap": "/data/images/ap", "bh": "/data/images/bh", "dxa": "/data/images/dxa"}
    model_root = _MODEL_IMAGE_ROOT[model_key]
    subdir = "plans" if ("plans" in p.parts or filename.startswith("plan_")) else "imgs"
    return f"{model_root}/{subdir}/{filename}"


def _build_user_text_lora5(case: dict) -> str:
    """Build user text matching LoRA5 training format."""
    parts = []
    ctx = case.get("inputs", {}).get("project_context", {})

    task_status = ctx.get("4d_task_status", "")
    parts.append(f"[4D Task Status] {task_status or 'N/A'}")

    project_phase = ctx.get("project_phase", "")
    parts.append(f"[Project Phase] {project_phase or 'N/A'}")

    location = ctx.get("location", "")
    if not location:
        location = (case.get("labels", {}).get("constraints", {})
                    .get("storey_name", ""))
    parts.append(f"[Location] {location or 'N/A'}")

    chat = case.get("inputs", {}).get("chat_history", [])
    if chat:
        chat_lines = [msg.get("text", "") for msg in chat]
        parts.append(f"[Chat Log]\n{' '.join(chat_lines)}")
    else:
        query = case.get("query_text", "")
        if query:
            parts.append(f"[Chat Log]\n{query}")

    return "\n".join(parts)


def _build_user_text_lora2(case: dict) -> str:
    """Build user text matching LoRA2 training format."""
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
        chat_block = "\n".join(f"  {msg['role']}: {msg['text']}" for msg in chat)
        parts.append(f"[Chat Log]\n{chat_block}")

    query = case.get("query_text", "")
    if query:
        parts.append(f"\n[Query] {query}")

    parts.append("\nExtract the search constraints as JSON.")
    return "\n".join(parts)


def _build_messages(case: dict, system_prompt: str,
                    modality_mode: str = "", prompt_style: str = "lora5") -> list:
    """Build ChatML messages with modality masking."""
    user_content = []
    inputs = case.get("inputs", {})
    case_id = case.get("case_id", "")

    # Site photos (skip if FP or MA)
    if modality_mode not in ("FP", "MA"):
        for img in inputs.get("images", []):
            modal_path = _remap_to_modal(img, case_id)
            if os.path.exists(modal_path):
                user_content.append({"type": "image", "image": f"file://{modal_path}"})

    # Floorplan (skip if SITE or MA)
    if modality_mode not in ("SITE", "MA"):
        fp = inputs.get("floorplan_patch")
        if fp:
            modal_path = _remap_to_modal(fp, case_id)
            if os.path.exists(modal_path):
                user_content.append({"type": "image", "image": f"file://{modal_path}"})

    # Text
    if prompt_style == "lora2":
        user_text = _build_user_text_lora2(case)
    else:
        user_text = _build_user_text_lora5(case)
    user_content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _normalize_sr_field(parsed: dict) -> dict:
    """Remap old 'relations' field → 'spatial_relations' and uppercase predicates."""
    sr = parsed.get("spatial_relations") or []
    if not sr:
        rel = parsed.get("relations")
        if isinstance(rel, list) and rel:
            sr = [r for r in rel if isinstance(r, dict) and "predicate" in r]
            if sr:
                parsed["spatial_relations"] = sr
    for triplet in (parsed.get("spatial_relations") or []):
        if "predicate" in triplet:
            triplet["predicate"] = triplet["predicate"].upper()
    return parsed


def _parse_json(text: str) -> Optional[dict]:
    """Parse JSON from model output (with fallbacks)."""
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
    match = re.search(r'(\{.*"(?:spatial_relations|relations)".*\})', text, re.DOTALL)
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
    timeout=3 * 60 * 60,
)
def run_eval(
    adapter_dir: str,
    tag: str,
    modality_mode: str = "FP",
    prompt_style: str = "lora5",
    limit: int = 0,
):
    """Run constraint extraction on unified test set (Modal A100).

    Args:
        adapter_dir: Path on checkpoint volume (e.g., /mscd-unified-eval/v5_lora_qwen_r32)
        tag: Output tag (e.g., "r32_FP") — used for output filename
        modality_mode: "FP" (floorplan only), "MC" (full), "SITE", "MA"
        prompt_style: "lora5" or "lora2" (controls system prompt + text format)
        limit: Max cases (0=all)
    """
    import torch
    from unsloth import FastVisionModel
    from qwen_vl_utils import process_vision_info

    system_prompt = SYSTEM_PROMPT_LORA2 if prompt_style == "lora2" else SYSTEM_PROMPT_LORA5

    # ── 1. Locate adapter ────────────────────────────────────────────────
    zero_shot = adapter_dir.upper() in ("NONE", "ZERO-SHOT", "BASE")
    adapter_path = None
    if not zero_shot:
        adapter_path = f"/checkpoints{adapter_dir}"
        # Auto-descend into "final/" subdir if adapter_config.json is not at top level
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            final_sub = os.path.join(adapter_path, "final")
            if os.path.exists(os.path.join(final_sub, "adapter_config.json")):
                adapter_path = final_sub
                print(f"  [INFO] Found adapter in {adapter_path}")
            else:
                contents = os.listdir(adapter_path) if os.path.exists(adapter_path) else []
                raise FileNotFoundError(
                    f"adapter_config.json not found in {adapter_path}.\n"
                    f"  Contents: {contents}"
                )

    print("=" * 60)
    print(f"MSCD Unified Evaluation (Modal A100)")
    print("=" * 60)
    print(f"  Adapter:     {adapter_dir}")
    print(f"  Tag:         {tag}")
    print(f"  Modality:    {modality_mode}")
    print(f"  Prompt:      {prompt_style}")
    print(f"  GPU:         {torch.cuda.get_device_name(0)}")

    # ── 2. Load model ────────────────────────────────────────────────────
    print("\nLoading base model (4-bit)...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        load_in_4bit=True,
    )
    if not zero_shot:
        print(f"Loading adapter: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    FastVisionModel.for_inference(model)
    print("Model ready.\n")

    # ── 3. Load cases ────────────────────────────────────────────────────
    cases_path = "/data/cases_unified_test.jsonl"
    with open(cases_path) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if limit > 0:
        cases = cases[:limit]
    print(f"Cases: {len(cases)}")

    # ── 4. Run inference ─────────────────────────────────────────────────
    results = []
    n_parsed = 0
    n_sr = 0
    n_hop2 = 0
    total_latency = 0.0

    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"case_{idx}")
        messages = _build_messages(case, system_prompt,
                                   modality_mode=modality_mode,
                                   prompt_style=prompt_style)

        n_images = sum(
            1 for c in messages[1]["content"]
            if isinstance(c, dict) and c.get("type") == "image"
        )

        t0 = time.perf_counter()
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            if image_inputs:
                inputs = tokenizer(
                    text=[text], images=image_inputs, videos=video_inputs,
                    add_special_tokens=False, return_tensors="pt",
                ).to(model.device)
            else:
                inputs = tokenizer(
                    text=[text], add_special_tokens=False, return_tensors="pt",
                ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=512,
                    do_sample=False, use_cache=True,
                )

            trimmed = output_ids[0][len(inputs.input_ids[0]):]
            raw_output = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
            latency_ms = (time.perf_counter() - t0) * 1000

            parsed = _parse_json(raw_output)
            if parsed:
                parsed = _normalize_sr_field(parsed)
                n_parsed += 1
                status = "OK"
                sr = parsed.get("spatial_relations", [])
                if sr:
                    n_sr += 1
                if len(sr) >= 2:
                    n_hop2 += 1
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
            "condition": modality_mode or "MC",
            "constraints": {
                "storey_name": parsed.get("storey_name"),
                "ifc_class": parsed.get("ifc_class"),
                "near_keywords": parsed.get("near_keywords", []),
                "relations": parsed.get("relations", []),
                "space_name": parsed.get("space_name"),
                "target_name_keyword": parsed.get("target_name_keyword"),
                "neighbor_type": parsed.get("neighbor_type"),
                "spatial_relations": parsed.get("spatial_relations", []),
            },
            "raw_output": raw_output[:500],
            "latency_ms": round(latency_ms, 1),
            "status": status,
        }
        results.append(result)

        ifc = parsed.get("ifc_class", "null")
        storey = parsed.get("storey_name", "null")
        sr_count = len(parsed.get("spatial_relations", []))
        sr_tag = f"SR={sr_count}" if sr_count else "no-SR"
        print(f"  [{idx:>3}/{len(cases)}] {case_id}  "
              f"imgs={n_images}  {latency_ms:.0f}ms  {status}  "
              f"class={ifc}  storey={storey}  {sr_tag}")

    # ── 5. Save results to NEW folder on volume ──────────────────────────
    output_dir = "/checkpoints/mscd-unified-eval"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/eval_constraints_{tag}.jsonl"

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    checkpoint_vol.commit()

    # ── 6. Summary ───────────────────────────────────────────────────────
    parse_rate = n_parsed / len(cases) if cases else 0
    avg_latency = total_latency / len(cases) if cases else 0

    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE: {tag}")
    print(f"{'=' * 60}")
    print(f"  Cases:        {len(cases)}")
    print(f"  Parse rate:   {n_parsed}/{len(cases)} ({parse_rate:.1%})")
    print(f"  SR extracted: {n_sr}/{len(cases)} ({100*n_sr/len(cases):.0f}%)")
    print(f"  2-hop:        {n_hop2}/{len(cases)} ({100*n_hop2/len(cases):.0f}%)")
    print(f"  Avg latency:  {avg_latency:.0f} ms/case")
    print(f"  Output:       {output_path}")
    print(f"\nDownload:")
    print(f"  modal volume get mscd-checkpoints "
          f"/mscd-unified-eval/eval_constraints_{tag}.jsonl "
          f"./output/unified/")

    return {
        "total": len(cases),
        "parsed": n_parsed,
        "parse_rate": parse_rate,
        "n_sr": n_sr,
        "n_hop2": n_hop2,
        "avg_latency_ms": avg_latency,
        "output_path": output_path,
        "tag": tag,
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

def _is_transient_poll_error(exc: BaseException) -> bool:
    cls_name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if "connectionerror" in cls_name:
        return True
    transient_markers = (
        "deadline exceeded", "timed out", "connection reset",
        "temporarily unavailable", "transport is closing",
    )
    return any(marker in msg for marker in transient_markers)


@app.local_entrypoint()
def main(
    adapter: str = "/mscd-unified-eval/v5_lora_qwen_r32",
    tag: str = "r32_FP",
    modality: str = "FP",
    prompt: str = "lora5",
    limit: int = 0,
):
    """Launch unified evaluation on Modal GPU.

    Args:
        adapter:  Checkpoint volume path to adapter dir.
        tag:      Output tag (used in filename).
        modality: "FP" (floorplan only) or "MC" (floorplan + site photo).
        prompt:   "lora5" or "lora2" (system prompt + text format).
        limit:    Max cases (0=all).
    """
    print(f"Launching unified eval on Modal...")
    print(f"  Adapter:  {adapter}")
    print(f"  Tag:      {tag}")
    print(f"  Modality: {modality}")
    print(f"  Prompt:   {prompt}")
    if limit > 0:
        print(f"  Limit:    {limit}")

    call = run_eval.spawn(
        adapter_dir=adapter,
        tag=tag,
        modality_mode=modality,
        prompt_style=prompt,
        limit=limit,
    )

    result = None
    transient_errors = 0
    while result is None:
        try:
            result = call.get(timeout=120)
        except TimeoutError:
            print("  [local] still waiting...")
        except Exception as e:
            if not _is_transient_poll_error(e):
                raise
            transient_errors += 1
            backoff = min(30.0, 1.5 ** min(transient_errors, 8))
            print(f"  [local] transient error, retrying in {backoff:.1f}s...")
            time.sleep(backoff)

    print(f"\nDONE: {result['parsed']}/{result['total']} parsed, "
          f"SR={result['n_sr']}, 2-hop={result['n_hop2']}")
    tag = result["tag"]
    print(f"\nDownload:")
    print(f"  modal volume get mscd-checkpoints "
          f"/mscd-unified-eval/eval_constraints_{tag}.jsonl "
          f"./output/unified/")
