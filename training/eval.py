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

    # v0.5 topology cases (69 cases with spatial relations)
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

# v0.5 test cases — 69 topology skeleton cases (file uses "v3" case schema format)
V05_CASES_FILE = PROJECT_ROOT / "eval" / "cases_v3_test.jsonl"
V05_CASES_SITE_FILE = PROJECT_ROOT / "eval" / "cases_v3_test_site.jsonl"
V05_BH_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.5_bh"  / "imgs"

# ── Modal infrastructure ─────────────────────────────────────────────────────

app = modal.App("mscd-vlm-lora-eval")

# Same image as training — ensures identical environment.
# Config files (profiles.yaml, prompts yaml, condition_mask.py) are baked in
# so that CONDITION_CONFIGS, SYSTEM_PROMPT, and blur logic are loaded at
# runtime from the same sources as the local pipeline — no more duplication.
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "unsloth",
        "qwen-vl-utils",
        "datasets==4.3.0",
        "hf-transfer",
        "pyyaml",          # required to load profiles.yaml + prompts yaml
    )
    .run_commands(
        "pip install --no-deps --force-reinstall "
        "'unsloth @ git+https://github.com/unslothai/unsloth.git'"
    )
    .pip_install("transformers==4.56.2")
    .run_commands("pip install --no-deps trl==0.22.2")
    .env({"HF_HOME": "/model_cache"})
    # ── Config: loaded at runtime to keep single source of truth ─────────────
    .add_local_file(str(PROFILES_YAML), remote_path="/app/profiles.yaml")
    .add_local_file(str(PROMPTS_YAML),  remote_path="/app/constraints_extraction.yaml")
    .add_local_file(str(COND_MASK_PY),  remote_path="/app/condition_mask.py")
    # ── Evaluation data — v0.4 (legacy) ────────────────────────────────────────
    .add_local_file(str(DEFAULT_CASES_FILE), remote_path="/data/test_holdout.jsonl")
    .add_local_dir(str(AP_IMGS_DIR),   remote_path="/data/images/ap/imgs")
    .add_local_dir(str(AP_PLANS_DIR),  remote_path="/data/images/ap/plans")
    .add_local_dir(str(BH_IMGS_DIR),   remote_path="/data/images/bh/imgs")
    .add_local_dir(str(BH_PLANS_DIR),  remote_path="/data/images/bh/plans")
    .add_local_dir(str(DXA_IMGS_DIR),  remote_path="/data/images/dxa/imgs")
    .add_local_dir(str(DXA_PLANS_DIR), remote_path="/data/images/dxa/plans")
    # ── Evaluation data — v0.5 topology cases (69 cases, "v3" schema) ──────
    .add_local_file(str(V05_CASES_FILE), remote_path="/data/v05_test.jsonl")
    .add_local_file(str(V05_CASES_SITE_FILE), remote_path="/data/v05_test_site.jsonl")
    .add_local_dir(str(V05_AP_IMGS_DIR),  remote_path="/data/images/v05_ap/imgs")
    .add_local_dir(str(V05_AP_WIRE_DIR),  remote_path="/data/images/v05_ap/wireframes")
    .add_local_dir(str(V05_AP_PLANS_DIR), remote_path="/data/images/v05_ap/plans")
    .add_local_dir(str(V05_DXA_IMGS_DIR), remote_path="/data/images/v05_dxa/imgs")
    .add_local_dir(str(V05_BH_IMGS_DIR),  remote_path="/data/images/v05_bh/imgs")
)

model_cache    = modal.Volume.from_name("mscd-model-cache",  create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("mscd-checkpoints",  create_if_missing=True)


# ── Runtime config helpers (run inside Modal container) ───────────────────────

def _load_condition_configs() -> Dict[str, Dict]:
    """Load condition configs from /app/profiles.yaml (single source of truth)."""
    import yaml
    with open("/app/profiles.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("conditions", {})


def _load_system_prompt() -> str:
    """Load LoRA system prompt from /app/constraints_extraction.yaml."""
    import yaml
    with open("/app/constraints_extraction.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("lora_system", data.get("system", ""))


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
    condition_override: str = "",
    cases_file: str = "",
):
    """Run LoRA constraint extraction on all cases (Modal A100)."""
    import torch
    from unsloth import FastVisionModel
    from qwen_vl_utils import process_vision_info

    # ── 0. Load config from baked-in files (single source of truth) ──────
    condition_configs = _load_condition_configs()   # from /app/profiles.yaml
    system_prompt     = _load_system_prompt()       # from /app/constraints_extraction.yaml
    ConditionMask     = _load_condition_mask()      # from /app/condition_mask.py
    print(f"Loaded {len(condition_configs)} conditions from profiles.yaml")
    print(f"System prompt: {len(system_prompt)} chars (lora_system)")

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
    data_path = cases_file if cases_file else "/data/test_holdout.jsonl"
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
        case_id = case.get("case_id", f"case_{idx}")
        condition = condition_override if condition_override else case.get("bench", {}).get("condition", "")

        # Apply condition mask (same as local pipeline)
        masked_case = apply_condition_mask(case, condition, condition_configs, ConditionMask)

        # Build messages
        messages = _build_messages(masked_case, system_prompt)

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
    # Use adapter dir name as tag (e.g., "final" or "checkpoint-180")
    tag = adapter_dir.rstrip("/").split("/")[-1]
    if condition_override:
        tag = f"{tag}_{condition_override}"
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
          f"./logs/evaluation_output/")
    print(f"\nRun local pipeline with pre-computed constraints:")
    print(f"  python script/run.py --profile v2_lora \\")
    print(f"    --cases eval/cases_v3_test.jsonl \\")
    print(f"    --precomputed logs/evaluation_output/eval_constraints_{tag}.jsonl")

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
    adapter_dir: str = "/mscd-lora/final",
    limit: int = 0,
    condition_override: str = "",
    cases: str = "",
):
    """Launch LoRA evaluation on Modal GPU.

    Args:
        cases: Remote path to cases JSONL inside Modal container.
               Use "/data/v05_test.jsonl" for v0.5 topology cases (69 cases).
               Default: "/data/test_holdout.jsonl" (v0.4 holdout, 50 cases).
    """
    cases_label = cases if cases else str(DEFAULT_CASES_FILE)
    print("Launching MSCD LoRA evaluation on Modal...")
    print(f"  Adapter:  {adapter_dir}")
    print(f"  Cases:    {cases_label}")
    if limit > 0:
        print(f"  Limit:    {limit} cases")
    if condition_override:
        print(f"  Condition override: {condition_override}")

    # Spawn + poll loop: each .get() call opens a fresh gRPC connection with a
    # short window. If the connection drops (ConnectionError / Deadline exceeded)
    # we simply reconnect and keep waiting. This is resilient to multi-hour runs
    # where any single gRPC stream would time out.
    call = run_eval.spawn(
        adapter_dir=adapter_dir,
        limit=limit,
        condition_override=condition_override,
        cases_file=cases,
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
          f"./logs/evaluation_output/")
    cases_hint = cases if cases else "evaluation/cases/cases_v3_test.jsonl"
    print(f"\nRun local pipeline:")
    print(f"  python script/run.py --profile v2_lora \\")
    print(f"    --cases {cases_hint} \\")
    print(f"    --precomputed logs/evaluation_output/eval_constraints_{tag}.jsonl")
