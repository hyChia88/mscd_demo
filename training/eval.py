"""
MSCD VLM LoRA Evaluation — Modal GPU Inference Script

Runs LoRA constraint extraction on Modal GPU for all evaluation cases.
Outputs pre-computed constraints JSONL that can be fed back to the local
pipeline via: python script/run.py --profile v2_lora --precomputed <file>

This separates the GPU-heavy LoRA inference (Modal A100) from retrieval +
scoring (local CPU), keeping eval methodologically clean — same hardware
as training.

Usage:
    modal run training/eval.py --adapter-dir /mscd-lora-v6-g1-fullaug/best --cases /data/ap_eval.jsonl
    modal run training/eval.py --adapter-dir /mscd-lora-v6-g0-canonical/best --cases /data/ap_eval.jsonl
    modal run training/eval.py --adapter-dir /mscd-lora-v6-g1-fullaug/best --cases /data/ap_eval.jsonl --limit 5

    # legacy v0.5 topology cases
    modal run training/eval.py --cases /data/v05_test.jsonl --condition-override MB
    modal run training/eval.py --cases /data/v05_test.jsonl --condition-override MC
"""

import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# ── Local data + config paths ─────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).parent.parent
DATA_ROOT     = PROJECT_ROOT.parent / "data_curation"
PROFILES_YAML = PROJECT_ROOT / "profiles.yaml"
PROMPTS_YAML  = PROJECT_ROOT / "prompts" / "constraints_extraction.yaml"
COND_MASK_PY  = PROJECT_ROOT / "src" / "v2" / "condition_mask.py"

# Default cases file — override with --cases flag
DEFAULT_CASES_FILE = DATA_ROOT / "datasets" / "synth_v0.4_merged" / "train" / "test_holdout_with_images.jsonl"
AP_EVAL_CASES_FILE = DATA_ROOT / "datasets" / "synth_v0.5_ap" / "train" / "lora6_v2_ap_eval_canonical_m.jsonl"
AP_EVAL_CASES_FILE_G7 = DATA_ROOT / "datasets" / "synth_v0.5_ap" / "train" / "lora6_v2_ap_eval_canonical_m_g7.jsonl"

# ── v0.4 image dirs (legacy) ────────────────────────────────────────────────
AP_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.4_ap"  / "cases" / "imgs"
AP_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.4_ap"  / "cases" / "plans"
BH_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.4_bh"  / "cases" / "imgs"
BH_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.4_bh"  / "cases" / "plans"
DXA_IMGS_DIR  = DATA_ROOT / "datasets" / "synth_v0.4_dxa" / "cases" / "imgs"
DXA_PLANS_DIR = DATA_ROOT / "datasets" / "synth_v0.4_dxa" / "cases" / "plans"

# ── v0.5 image dirs (topology skeletons) ────────────────────────────────────
V05_AP_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.5"     / "imgs"
V05_AP_WIRE_DIR   = DATA_ROOT / "datasets" / "synth_v0.5"     / "renders" / "wireframes"
V05_AP_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.5"     / "floorplans"
V05_DXA_IMGS_DIR  = DATA_ROOT / "datasets" / "synth_v0.5_dxa" / "imgs"
V05_AP_CURATED_ROOT = DATA_ROOT / "datasets" / "synth_v0.5_ap"

# v0.5 test cases — 69 topology skeleton cases (file uses "v3" case schema format)
V05_CASES_FILE = PROJECT_ROOT / "eval" / "cases_v3_test.jsonl"
V05_CASES_SITE_FILE = PROJECT_ROOT / "eval" / "cases_v3_test_site.jsonl"
V05_BH_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.5_bh"  / "imgs"

# ── Modal infrastructure ─────────────────────────────────────────────────────

app = modal.App("mscd-vlm-lora-eval")

def _build_eval_image() -> modal.Image:
    """Build the Modal image while tolerating legacy dataset directories missing.

    The AP assembled eval path is the primary target now. Older v0.4/v0.5
    assets are baked only when they still exist locally.
    """
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

    for local_file, remote_file in [
        (DEFAULT_CASES_FILE, "/data/test_holdout.jsonl"),
        (AP_EVAL_CASES_FILE, "/data/ap_eval.jsonl"),
        (AP_EVAL_CASES_FILE_G7, "/data/ap_eval_g7.jsonl"),
        (V05_CASES_FILE, "/data/v05_test.jsonl"),
        (V05_CASES_SITE_FILE, "/data/v05_test_site.jsonl"),
    ]:
        if local_file.exists():
            image = image.add_local_file(str(local_file), remote_path=remote_file)

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
        (V05_AP_CURATED_ROOT, "/data/images/synth_v0.5_ap"),
        (V05_DXA_IMGS_DIR, "/data/images/v05_dxa/imgs"),
        (V05_BH_IMGS_DIR, "/data/images/v05_bh/imgs"),
    ]:
        if local_dir.exists():
            image = image.add_local_dir(str(local_dir), remote_path=remote_dir)

    return image


# Same image as training — ensures identical environment.
# Config files (profiles.yaml, prompts yaml, condition_mask.py) are baked in
# so that CONDITION_CONFIGS, SYSTEM_PROMPT, and blur logic are loaded at
# runtime from the same sources as the local pipeline — no more duplication.
eval_image = _build_eval_image()

model_cache    = modal.Volume.from_name("mscd-model-cache",  create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("mscd-checkpoints",  create_if_missing=True)


# ── Runtime config helpers (run inside Modal container) ───────────────────────

def _load_condition_configs() -> Dict[str, Dict]:
    """Load condition configs from /app/profiles.yaml (single source of truth)."""
    import yaml
    with open("/app/profiles.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("conditions", {})


def _load_system_prompt(prompt_key: str = "lora_system") -> str:
    """Load LoRA system prompt from /app/constraints_extraction.yaml."""
    import yaml
    with open("/app/constraints_extraction.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get(prompt_key, data.get("lora_system", data.get("system", "")))


def _load_condition_mask():
    """Import ConditionMask from /app/condition_mask.py (baked into image)."""
    import sys
    sys.path.insert(0, "/app")
    from condition_mask import ConditionMask  # noqa: PLC0415
    return ConditionMask


def apply_condition_mask(case: dict, condition: str,
                         condition_configs: Dict[str, Dict],
                         ConditionMask) -> dict:
    """Apply condition masking using ConditionMask.apply() + loaded configs."""
    overrides = condition_configs.get(condition, {})
    return ConditionMask.apply(case, overrides)


# ── Image path remapping ─────────────────────────────────────────────────────

# Map case_id model prefix → remote /data/images/<model>/ directory
_MODEL_IMAGE_ROOT = {
    "ap":  "/data/images/ap",
    "bh":  "/data/images/bh",
    "dxa": "/data/images/dxa",
}
_LOCAL_ROOT = "file:///root/cmu/master_thesis/data_curation/datasets/"
_REMOTE_ROOT = "file:///data/images/"
_DATASET_MAP = {
    "synth_v0.5_ap/": "synth_v0.5_ap/",
    "synth_v0.5/": "v05_ap/",
    "synth_v0.5_bh/": "v05_bh/",
    "synth_v0.5_dxa/": "v05_dxa/",
    "synth_v0.4_ap/cases/": "ap/",
    "synth_v0.4_bh/cases/": "bh/",
    "synth_v0.4_dxa/cases/": "dxa/",
}


def _remap_to_modal(path: str, case_id: str = "") -> str:
    """Remap local case image path to Modal container path.

    Handles both v0.4 paths (case_id prefix → model subdir) and v0.5 paths
    (relative paths like ``datasets/synth_v0.5/imgs/SK_136_site.png``).
    """
    path_str = str(path)
    p = Path(path_str)
    filename = p.name

    # ── v0.5 paths: detected by "synth_v0.5" in path ────────────────────
    if "synth_v0.5" in path_str:
        # wireframes
        if "wireframes" in path_str or "renders" in path_str:
            if "_dxa" in path_str or "v0.5_dxa" in path_str:
                return f"/data/images/v05_dxa/wireframes/{filename}"
            return f"/data/images/v05_ap/wireframes/{filename}"
        # floorplans
        if "floorplan" in path_str:
            if "_dxa" in path_str or "v0.5_dxa" in path_str:
                return f"/data/images/v05_dxa/plans/{filename}"
            return f"/data/images/v05_ap/plans/{filename}"
        # site photos
        if "_dxa" in path_str or "v0.5_dxa" in path_str:
            return f"/data/images/v05_dxa/imgs/{filename}"
        if "_bh" in path_str or "v0.5_bh" in path_str:
            return f"/data/images/v05_bh/imgs/{filename}"
        return f"/data/images/v05_ap/imgs/{filename}"

    # ── v0.4 paths (legacy): derive model from case_id ───────────────────
    model_key = "ap"
    if "_BH_" in case_id:
        model_key = "bh"
    elif "_DXA_" in case_id:
        model_key = "dxa"
    model_root = _MODEL_IMAGE_ROOT[model_key]

    subdir = "plans" if ("plans" in p.parts or filename.startswith("plan_")) else "imgs"
    return f"{model_root}/{subdir}/{filename}"


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


def _build_messages(case: dict, system_prompt: str) -> list:
    """Build ChatML messages for VLM inference."""
    user_content = []
    inputs = case.get("inputs", {})
    case_id = case.get("case_id", "")

    # Site photos
    for img in inputs.get("images", []):
        modal_path = _remap_to_modal(img, case_id)
        if os.path.exists(modal_path):
            user_content.append({"type": "image", "image": f"file://{modal_path}"})
        else:
            print(f"    [WARN] Image not found: {modal_path}")

    # Floorplan patch
    fp = inputs.get("floorplan_patch")
    if fp:
        modal_path = _remap_to_modal(fp, case_id)
        if os.path.exists(modal_path):
            user_content.append({"type": "image", "image": f"file://{modal_path}"})

    # Text prompt
    user_content.append({"type": "text", "text": _build_user_text(case)})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _remap_training_record_paths(case: dict) -> dict:
    remapped = copy.deepcopy(case)
    for msg in remapped.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    path = part.get("image", "")
                    if path.startswith(_LOCAL_ROOT):
                        rel = path[len(_LOCAL_ROOT):]
                        for ds_prefix, remote_prefix in _DATASET_MAP.items():
                            if rel.startswith(ds_prefix):
                                rel = remote_prefix + rel[len(ds_prefix):]
                                break
                        part["image"] = _REMOTE_ROOT + rel
    return remapped


def _build_messages_for_eval_case(case: dict, system_prompt: str) -> list:
    if "messages" in case:
        remapped = _remap_training_record_paths(case)
        messages = [m for m in remapped["messages"] if m.get("role") != "assistant"]
        # Keep assembled AP cases aligned with the runtime prompt selection.
        # This avoids silently evaluating a G7 adapter against the legacy short
        # prompt baked into older JSONL records.
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = system_prompt
                break
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages
    return _build_messages(case, system_prompt)


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


def _adapter_tag(adapter_dir: str) -> str:
    """Create a stable, non-colliding tag from adapter path.

    Using only basename ('best') causes different adapters to overwrite each
    other at /checkpoints/mscd-lora/eval_constraints_best.jsonl.
    """
    cleaned = adapter_dir.strip("/").replace("/", "__")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", cleaned)
    return cleaned or "adapter"


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
    adapter_dir: str = "/mscd-lora-v6-g1-fullaug/best",
    limit: int = 0,
    condition_override: str = "",
    cases_file: str = "",
    prompt_key: str = "",
    tag_suffix: str = "",
):
    """Run LoRA constraint extraction on all cases (Modal A100).

    Supported case formats:
    - assembled AP ChatML records with ``messages`` (preferred)
    - legacy retrieval cases with ``inputs`` / ``query_text`` / ``bench``
    """
    import torch
    from unsloth import FastVisionModel
    from qwen_vl_utils import process_vision_info

    # ── 0. Load config from baked-in files (single source of truth) ──────
    condition_configs = _load_condition_configs()   # from /app/profiles.yaml
    if not prompt_key:
        prompt_key = "lora_system_g7" if "g7" in adapter_dir.lower() else "lora_system"
    system_prompt     = _load_system_prompt(prompt_key)       # from /app/constraints_extraction.yaml
    ConditionMask     = _load_condition_mask()      # from /app/condition_mask.py
    print(f"Loaded {len(condition_configs)} conditions from profiles.yaml")
    print(f"System prompt: {len(system_prompt)} chars ({prompt_key})")

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
    if cases_file:
        data_path = cases_file
    else:
        data_path = "/data/ap_eval_g7.jsonl" if prompt_key == "lora_system_g7" else "/data/ap_eval.jsonl"
    print(f"Loading cases from: {data_path}")
    cases = []
    with open(data_path) as f:
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

    if condition_override:
        print(f"Condition override: ALL cases will use condition={condition_override}")

    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id") or case.get("id") or case.get("base_case_id") or f"case_{idx}"
        condition = condition_override if condition_override else case.get("bench", {}).get("condition", "")
        if not condition and "messages" in case:
            condition = "AP_EVAL"

        # Apply condition mask (same as local pipeline)
        masked_case = apply_condition_mask(case, condition, condition_configs, ConditionMask) if "inputs" in case else case

        # Build messages
        messages = _build_messages_for_eval_case(masked_case, system_prompt)

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
        print(f"  [{idx:>3}/{len(cases)}] {case_id}  cond={condition}  "
              f"imgs={n_images}  {latency_ms:.0f}ms  {status}  "
              f"class={ifc}  storey={storey}")

    # ── 5. Save results to Modal volume ──────────────────────────────────
    # Use full adapter path as tag to avoid collisions between different
    # adapters that all end in "/best".
    tag = _adapter_tag(adapter_dir)
    if condition_override:
        tag = f"{tag}_{condition_override}"
    if tag_suffix:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag_suffix.strip())
        if cleaned:
            tag = f"{tag}_{cleaned}"
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
          f"./output/")
    print(f"\nRun local pipeline with pre-computed constraints:")
    print(f"  # legacy retrieval cases example")
    print(f"  python script/run.py --profile v2_lora \\")
    print(f"    --cases eval/cases_v3_test.jsonl \\")
    print(f"    --precomputed output/eval_constraints_{tag}.jsonl")

    return {
        "total": len(cases),
        "parsed": n_parsed,
        "parse_rate": parse_rate,
        "avg_latency_ms": avg_latency,
        "output_path": output_path,
        "tag": tag,
    }


# ── CLI entry point ──────────────────────────────────────────────────────────


def _is_transient_poll_error(exc: BaseException) -> bool:
    """Return True for retryable local poll errors while waiting for Modal output."""
    cls_name = exc.__class__.__name__.lower()
    cls_module = exc.__class__.__module__.lower()
    msg = str(exc).lower()

    if "connectionerror" in cls_name:
        return True
    if "grpc" in cls_module and (
        "deadline exceeded" in msg
        or "timed out" in msg
        or "unavailable" in msg
    ):
        return True

    transient_markers = (
        "deadline exceeded",
        "timed out",
        "connection reset",
        "temporarily unavailable",
        "transport is closing",
    )
    return any(marker in msg for marker in transient_markers)


@app.local_entrypoint()
def main(
    adapter_dir: str = "/mscd-lora-v6-g1-fullaug/best",
    limit: int = 0,
    condition_override: str = "",
    cases: str = "",
    tag_suffix: str = "",
    prompt_key: str = "",
):
    """Launch LoRA evaluation on Modal GPU.

    Args:
        cases: Remote path to cases JSONL inside Modal container.
               Use "/data/v05_test.jsonl" for v0.5 topology cases (69 cases).
               Use "/data/ap_eval.jsonl" for legacy LoRA6-v2 AP eval cases.
               Use "/data/ap_eval_g7.jsonl" for G7/G8 AP eval cases.
               Default: G7 adapters use "/data/ap_eval_g7.jsonl"; others use "/data/ap_eval.jsonl".
        prompt_key: System prompt key from constraints_extraction.yaml.
                    Use "lora_system_g7" for G7/G8 adapters trained with G7 profile.
                    Auto-detected from adapter_dir name ("g7" -> lora_system_g7) if not set.
    """
    if cases:
        cases_label = cases
    else:
        cases_label = "/data/ap_eval_g7.jsonl" if "g7" in adapter_dir.lower() else "/data/ap_eval.jsonl"
    print("Launching MSCD LoRA evaluation on Modal...")
    print(f"  Adapter:  {adapter_dir}")
    print(f"  Cases:    {cases_label}")
    if limit > 0:
        print(f"  Limit:    {limit} cases")
    if condition_override:
        print(f"  Condition override: {condition_override}")
    if tag_suffix:
        print(f"  Tag suffix: {tag_suffix}")

    # Spawn + poll loop: each .get() call opens a fresh gRPC connection with a
    # short window. If the connection drops (ConnectionError / Deadline exceeded)
    # we simply reconnect and keep waiting. This is resilient to multi-hour runs
    # where any single gRPC stream would time out.
    call = run_eval.spawn(
        adapter_dir=adapter_dir,
        limit=limit,
        condition_override=condition_override,
        cases_file=cases,
        tag_suffix=tag_suffix,
        prompt_key=prompt_key,
    )
    result = None
    poll_secs = 120  # re-open gRPC connection every 2 minutes
    transient_errors = 0
    while result is None:
        try:
            result = call.get(timeout=poll_secs)
        except TimeoutError:
            print("  [local] still waiting for Modal function...")
        except Exception as e:
            if not _is_transient_poll_error(e):
                raise
            transient_errors += 1
            backoff_secs = min(30.0, 1.5 ** min(transient_errors, 8))
            print(
                "  [local] transient poll failure "
                f"({type(e).__module__}.{type(e).__name__}: {e}); "
                f"retrying in {backoff_secs:.1f}s"
            )
            time.sleep(backoff_secs)

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
          f"./output/")
    if cases:
        cases_hint = cases
    else:
        cases_hint = "/data/ap_eval_g7.jsonl" if "g7" in adapter_dir.lower() else "/data/ap_eval.jsonl"
    print(f"\nRun local pipeline:")
    print(f"  python script/run.py --profile v2_lora \\")
    print(f"    --cases {cases_hint} \\")
    print(f"    --precomputed output/eval_constraints_{tag}.jsonl")
