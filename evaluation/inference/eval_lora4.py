"""
MSCD VLM LoRA_4 Evaluation — Modal GPU Inference Script

Runs LoRA_4 constraint extraction on Modal GPU for all evaluation cases.
Outputs pre-computed constraints JSONL that can be fed back to the local
pipeline via: python script/run.py --profile v2_lora --precomputed <file>

LoRA_4 changes from eval.py (LoRA_3):
  1. Default adapter: /mscd-lora-v4/final (was /mscd-lora/final)
  2. max_new_tokens: 512 (was 256) — multi-triplet JSON is longer
  3. Output path: /mscd-lora-v4/eval_constraints_... (was /mscd-lora/...)
  4. App name: mscd-vlm-lora4-eval (was mscd-vlm-lora-eval)
  5. v0.5 BH/DXA floorplan dirs baked in (were missing)
  6. SR extraction logging (hop-1/hop-2 stats in summary)

Usage:
    modal run training/eval_lora4.py
    modal run training/eval_lora4.py --adapter-dir /mscd-lora-v4/final
    modal run training/eval_lora4.py --limit 5  # Quick test

    # v0.5 topology cases (69 cases with spatial relations)
    modal run training/eval_lora4.py --cases /data/v05_test.jsonl --condition-override MC

    # v0.4 holdout (50 cases, backward compatibility check)
    modal run training/eval_lora4.py --condition-override MB
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional

import modal

# ── Local data + config paths ─────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).parent.parent.parent
DATA_ROOT     = PROJECT_ROOT.parent / "data_curation"
PROFILES_YAML = PROJECT_ROOT / "profiles.yaml"
PROMPTS_YAML  = PROJECT_ROOT / "prompts" / "constraints_extraction.yaml"
COND_MASK_PY  = PROJECT_ROOT / "src" / "v2" / "condition_mask.py"

# Default cases file — override with --cases flag
DEFAULT_CASES_FILE = DATA_ROOT / "datasets" / "synth_v0.4_merged" / "train" / "test_holdout_with_images.jsonl"

# LoRA_4 test file (training conversation format → converted to eval format)
LORA4_TEST_SRC = DATA_ROOT / "datasets" / "synth_v0.5" / "train" / "lora4_test.jsonl"
LORA4_SKELETON_PATHS = [
    DATA_ROOT / "datasets" / "synth_v0.5"     / "skeletons" / "skeletons_v2_5.jsonl",
    DATA_ROOT / "datasets" / "synth_v0.5_bh"  / "skeletons" / "skeletons_v2_5.jsonl",
    DATA_ROOT / "datasets" / "synth_v0.5_dxa" / "skeletons" / "skeletons_v2_5.jsonl",
]

# ── v0.4 image dirs (legacy) ────────────────────────────────────────────────
AP_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.4_ap"  / "cases" / "imgs"
AP_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.4_ap"  / "cases" / "plans"
BH_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.4_bh"  / "cases" / "imgs"
BH_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.4_bh"  / "cases" / "plans"
DXA_IMGS_DIR  = DATA_ROOT / "datasets" / "synth_v0.4_dxa" / "cases" / "imgs"
DXA_PLANS_DIR = DATA_ROOT / "datasets" / "synth_v0.4_dxa" / "cases" / "plans"

# ── v0.5 image dirs (topology skeletons — all 3 models) ───────────────────
V05_AP_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.5"     / "imgs"
V05_AP_WIRE_DIR   = DATA_ROOT / "datasets" / "synth_v0.5"     / "renders" / "wireframes"
V05_AP_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.5"     / "floorplans"
V05_BH_IMGS_DIR   = DATA_ROOT / "datasets" / "synth_v0.5_bh"  / "imgs"
V05_BH_PLANS_DIR  = DATA_ROOT / "datasets" / "synth_v0.5_bh"  / "floorplans"
V05_DXA_IMGS_DIR  = DATA_ROOT / "datasets" / "synth_v0.5_dxa" / "imgs"
V05_DXA_PLANS_DIR = DATA_ROOT / "datasets" / "synth_v0.5_dxa" / "floorplans"

# v0.5 test cases — 75 cases (auto-converted from lora4_test.jsonl)
V05_CASES_FILE = PROJECT_ROOT / "evaluation" / "cases" / "cases_v4_test.jsonl"

# ── Convert lora4_test.jsonl → eval case format ─────────────────────────────

def _convert_lora4_test():
    """Convert lora4_test.jsonl (training format) → cases_v4_test.jsonl (eval format).

    Runs at module load time so the cases file is available for Modal image bake.
    Image paths are taken directly from the training data (already correct).
    """
    if V05_CASES_FILE.exists() and \
       V05_CASES_FILE.stat().st_mtime >= LORA4_TEST_SRC.stat().st_mtime:
        return  # Already up-to-date

    print(f"Converting {LORA4_TEST_SRC.name} → {V05_CASES_FILE.name} ...")

    # Load skeletons for ground truth
    skeletons = {}
    for spath in LORA4_SKELETON_PATHS:
        if spath.exists():
            with open(spath) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        skeletons[rec["id"]] = rec

    # Load test records
    with open(LORA4_TEST_SRC) as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    cases = []
    for rec in test_records:
        record_id = rec["id"]
        sk_match = re.search(r"SK_(\d+)", record_id)
        if not sk_match:
            continue
        sk_id = f"SK_{sk_match.group(1)}"
        skel = skeletons.get(sk_id)

        msgs = rec.get("messages", [])
        user_msg = next((m for m in msgs if m["role"] == "user"), None)
        asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)
        if not user_msg or not asst_msg:
            continue

        # Extract user text
        content = user_msg["content"]
        if isinstance(content, list):
            user_text = " ".join(
                c["text"] for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        else:
            user_text = str(content)

        # Parse structured fields
        parsed_text = {"task_status": "", "project_phase": "",
                       "chat_lines": [], "query_text": ""}
        m = re.search(r"\[4D Task Status\]\s*(.+?)(?:\n|$)", user_text)
        if m:
            parsed_text["task_status"] = m.group(1).strip()
        m = re.search(r"\[Project Phase\]\s*(.+?)(?:\n|$)", user_text)
        if m:
            parsed_text["project_phase"] = m.group(1).strip()
        chat_m = re.search(
            r"\[Chat Log\]\s*\n(.*?)(?:\n\[Query\]|\nExtract the search)",
            user_text, re.DOTALL)
        if chat_m:
            for line in chat_m.group(1).strip().split("\n"):
                line = line.strip()
                m2 = re.match(r"(\w[\w\s]*?):\s*(.+)", line)
                if m2:
                    parsed_text["chat_lines"].append({
                        "role": m2.group(1).strip(),
                        "text": m2.group(2).strip(),
                    })
        m = re.search(r"\[Query\]\s*(.+?)(?:\nExtract|\n\n|$)", user_text, re.DOTALL)
        if m:
            parsed_text["query_text"] = m.group(1).strip()

        # Parse ground truth constraints
        try:
            gt = json.loads(asst_msg["content"])
        except (json.JSONDecodeError, TypeError):
            continue

        # Extract image paths from training message content
        images = []
        floorplan = None
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                img_path = re.sub(r"^file://", "", item.get("image", ""))
                mp = re.search(r"(datasets/.+)", img_path)
                if mp:
                    img_path = mp.group(1)
                fname = Path(img_path).name
                # Classify: floorplan vs site photo
                # v0.5: "floorplan" in path; v0.4: "plan_" prefix or /plans/ dir
                is_fp = ("floorplan" in img_path
                         or fname.startswith("plan_")
                         or "/plans/" in img_path)
                if is_fp:
                    floorplan = img_path
                else:
                    images.append(img_path)

        # Ground truth from skeleton
        if skel:
            target_props = skel.get("target_props", {})
            ground_truth = {
                "target_guid": skel["target_guid"],
                "target_storey": target_props.get("Storey", ""),
                "target_ifc_class": target_props.get("Type", ""),
                "target_name": target_props.get("Name", ""),
            }
        else:
            ground_truth = {
                "target_guid": "",
                "target_storey": gt.get("storey_name", ""),
                "target_ifc_class": gt.get("ifc_class", ""),
                "target_name": "",
            }

        spatial_rels = gt.get("spatial_relations", [])
        has_images = len(images) > 0
        has_fp = floorplan is not None
        if has_images and has_fp:
            group, condition = "C", "C1"
        elif has_images:
            group, condition = "B", "B1"
        else:
            group, condition = "A", "A1"

        cases.append({
            "case_id": record_id,
            "bench": {"group": group, "condition": condition},
            "difficulty_tags": {
                "tier": skel.get("difficulty", "Tier 3") if skel else "Tier 3",
                "requires_relation": skel.get("requires_relation", False) if skel else False,
                "spatial_predicate": skel.get("spatial_predicate") if skel else None,
                "pattern": skel.get("pattern", "") if skel else "",
            },
            "ground_truth": ground_truth,
            "inputs": {
                "chat_history": parsed_text["chat_lines"],
                "chat_quality": "clear",
                "project_context": {
                    "4d_task_status": parsed_text["task_status"],
                    "project_phase": parsed_text["project_phase"],
                },
                "images": images,
                "floorplan_patch": floorplan,
            },
            "labels": {
                "constraints": {
                    "storey_name": gt.get("storey_name"),
                    "ifc_class": gt.get("ifc_class"),
                    "space_name": gt.get("space_name"),
                    "target_name_keyword": gt.get("target_name_keyword"),
                    **({"spatial_relations": spatial_rels} if spatial_rels else {}),
                },
            },
            "query_text": parsed_text["query_text"],
        })

    with open(V05_CASES_FILE, "w") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    n_fp = sum(1 for c in cases if c["inputs"]["floorplan_patch"])
    n_sr = sum(1 for c in cases if c["labels"]["constraints"].get("spatial_relations"))
    print(f"  {len(cases)} cases, {n_fp} with floorplan, {n_sr} with spatial_relations")


# Run conversion at import time (before Modal image build)
if LORA4_TEST_SRC.exists():
    _convert_lora4_test()


# ── Modal infrastructure ─────────────────────────────────────────────────────

app = modal.App("mscd-vlm-lora4-eval")

eval_image = (
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
    # ── Evaluation data — v0.5 topology cases ──────────────────────────────────
    .add_local_file(str(V05_CASES_FILE), remote_path="/data/v05_test.jsonl")
    # v0.5 AP images + wireframes + floorplans
    .add_local_dir(str(V05_AP_IMGS_DIR),  remote_path="/data/images/v05_ap/imgs")
    .add_local_dir(str(V05_AP_WIRE_DIR),  remote_path="/data/images/v05_ap/wireframes")
    .add_local_dir(str(V05_AP_PLANS_DIR), remote_path="/data/images/v05_ap/plans")
    # v0.5 BH images + floorplans (NEW in LoRA_4)
    .add_local_dir(str(V05_BH_IMGS_DIR),  remote_path="/data/images/v05_bh/imgs")
    .add_local_dir(str(V05_BH_PLANS_DIR), remote_path="/data/images/v05_bh/plans")
    # v0.5 DXA images + floorplans (floorplans NEW in LoRA_4)
    .add_local_dir(str(V05_DXA_IMGS_DIR), remote_path="/data/images/v05_dxa/imgs")
    .add_local_dir(str(V05_DXA_PLANS_DIR), remote_path="/data/images/v05_dxa/plans")
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

_MODEL_IMAGE_ROOT = {
    "ap":  "/data/images/ap",
    "bh":  "/data/images/bh",
    "dxa": "/data/images/dxa",
}


def _remap_to_modal(path: str, case_id: str = "") -> str:
    """Remap local case image path to Modal container path.

    Handles both v0.4 paths (case_id prefix -> model subdir) and v0.5 paths
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
            if "_bh" in path_str or "v0.5_bh" in path_str:
                return f"/data/images/v05_bh/plans/{filename}"
            return f"/data/images/v05_ap/plans/{filename}"
        # site photos
        if "_dxa" in path_str or "v0.5_dxa" in path_str:
            return f"/data/images/v05_dxa/imgs/{filename}"
        if "_bh" in path_str or "v0.5_bh" in path_str:
            return f"/data/images/v05_bh/imgs/{filename}"
        return f"/data/images/v05_ap/imgs/{filename}"

    # ── v0.4 paths (legacy): derive model from path first, then case_id ──
    model_key = "ap"
    if "v0.4_dxa" in path_str or "_DXA_" in case_id:
        model_key = "dxa"
    elif "v0.4_bh" in path_str or "_BH_" in case_id:
        model_key = "bh"
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


def _normalize_sr_field(parsed: dict) -> dict:
    """Remap old 'relations' field → 'spatial_relations' and uppercase predicates.

    LoRA_4 model sometimes outputs the old LoRA_3 field name 'relations' instead
    of 'spatial_relations' due to the system prompt still listing 'relations'.
    Also normalises lowercase predicates (e.g. 'adjacent_to' → 'ADJACENT_TO').
    """
    # If spatial_relations already present and non-empty, just normalise case
    sr = parsed.get("spatial_relations") or []

    # Fall back to 'relations' if spatial_relations is empty
    if not sr:
        rel = parsed.get("relations")
        if isinstance(rel, list) and rel:
            # Filter: only dicts with 'predicate' key are SR-like
            sr = [r for r in rel if isinstance(r, dict) and "predicate" in r]
            if sr:
                parsed["spatial_relations"] = sr

    # Uppercase predicates
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

    # Multi-triplet JSON may have nested arrays — use greedy match
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
    timeout=2 * 60 * 60,
)
def run_eval(
    adapter_dir: str = "/mscd-lora-v4/final",
    limit: int = 0,
    condition_override: str = "",
    cases_file: str = "",
):
    """Run LoRA_4 constraint extraction on all cases (Modal A100)."""
    import torch
    from unsloth import FastVisionModel
    from qwen_vl_utils import process_vision_info

    # ── 0. Load config from baked-in files (single source of truth) ──────
    condition_configs = _load_condition_configs()
    system_prompt     = _load_system_prompt()
    ConditionMask     = _load_condition_mask()
    print(f"Loaded {len(condition_configs)} conditions from profiles.yaml")
    print(f"System prompt: {len(system_prompt)} chars (lora_system)")

    # ── 1. Locate adapter (or run zero-shot baseline) ───────────────────
    zero_shot = adapter_dir.upper() in ("NONE", "ZERO-SHOT", "BASE")

    if zero_shot:
        print("=" * 60)
        print("MSCD ZERO-SHOT BASELINE Evaluation (Modal A100)")
        print("  No LoRA adapter — base Qwen2.5-VL-7B-Instruct only")
        print("=" * 60)
    else:
        adapter_path = f"/checkpoints{adapter_dir}"
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            contents = os.listdir(adapter_path) if os.path.exists(adapter_path) else []
            raise FileNotFoundError(
                f"adapter_config.json not found in {adapter_path}.\n"
                f"  Contents: {contents}"
            )
        print("=" * 60)
        print("MSCD LoRA_4 Evaluation (Modal A100)")
        print("=" * 60)
        print(f"  Adapter:  {adapter_path}")

    print(f"  GPU:      {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ── 2. Load model (+ adapter unless zero-shot) ───────────────────────
    print("\nLoading base model (4-bit)...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        load_in_4bit=True,
    )

    if zero_shot:
        print("Running ZERO-SHOT (no adapter loaded)")
    else:
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
    n_sr = 0         # cases with spatial_relations extracted
    n_hop2 = 0       # cases with 2+ spatial_relations (multi-triplet)
    total_latency = 0.0

    if condition_override:
        print(f"Condition override: ALL cases will use condition={condition_override}")

    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"case_{idx}")
        condition = condition_override if condition_override else case.get("bench", {}).get("condition", "")

        masked_case = apply_condition_mask(case, condition, condition_configs, ConditionMask)
        messages = _build_messages(masked_case, system_prompt)

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

            # LoRA_4: max_new_tokens=512 (was 256 in LoRA_3)
            # Multi-triplet JSON with 2+ spatial_relations is longer
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
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
        sr_count = len(parsed.get("spatial_relations", []))
        sr_tag = f"SR={sr_count}" if sr_count else "no-SR"
        print(f"  [{idx:>3}/{len(cases)}] {case_id}  cond={condition}  "
              f"imgs={n_images}  {latency_ms:.0f}ms  {status}  "
              f"class={ifc}  storey={storey}  {sr_tag}")
        if limit > 0:
            print(f"        RAW: {raw_output[:300]}")

    # ── 5. Save results to Modal volume ──────────────────────────────────
    tag = "zeroshot" if zero_shot else adapter_dir.rstrip("/").split("/")[-1]
    if condition_override:
        tag = f"{tag}_{condition_override}"
    output_path = f"/checkpoints/mscd-lora-v4/eval_constraints_{tag}.jsonl"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    checkpoint_vol.commit()

    # ── 6. Summary ───────────────────────────────────────────────────────
    parse_rate = n_parsed / len(cases) if cases else 0
    avg_latency = total_latency / len(cases) if cases else 0

    print(f"\n{'=' * 60}")
    print(f"LORA_4 EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Adapter:      {adapter_dir}")
    print(f"  Cases:        {len(cases)}")
    print(f"  Parse rate:   {n_parsed}/{len(cases)} ({parse_rate:.1%})")
    print(f"  SR extracted: {n_sr}/{len(cases)} ({100*n_sr/len(cases):.0f}%)")
    print(f"  2-hop (multi-triplet): {n_hop2}/{len(cases)} ({100*n_hop2/len(cases):.0f}%)")
    print(f"  Avg latency:  {avg_latency:.0f} ms/case")
    print(f"  Output:       {output_path}")
    print(f"\nDownload with:")
    print(f"  modal volume get mscd-checkpoints "
          f"/mscd-lora-v4/eval_constraints_{tag}.jsonl "
          f"./output/")
    print(f"\nRun local pipeline with pre-computed constraints:")
    print(f"  python script/run.py --profile v2_lora \\")
    print(f"    --cases eval/cases_v3_test.jsonl \\")
    print(f"    --precomputed output/eval_constraints_{tag}.jsonl")

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
    adapter_dir: str = "/mscd-lora-v4/final",
    limit: int = 0,
    condition_override: str = "",
    cases: str = "",
):
    """Launch LoRA_4 evaluation on Modal GPU.

    Args:
        cases: Remote path to cases JSONL inside Modal container.
               Use "/data/v05_test.jsonl" for v0.5 topology cases (69 cases).
               Default: "/data/test_holdout.jsonl" (v0.4 holdout, 50 cases).
    """
    cases_label = cases if cases else str(DEFAULT_CASES_FILE)
    print("Launching MSCD LoRA_4 evaluation on Modal...")
    print(f"  Adapter:  {adapter_dir}")
    print(f"  Cases:    {cases_label}")
    if limit > 0:
        print(f"  Limit:    {limit} cases")
    if condition_override:
        print(f"  Condition override: {condition_override}")

    call = run_eval.spawn(
        adapter_dir=adapter_dir,
        limit=limit,
        condition_override=condition_override,
        cases_file=cases,
    )
    result = None
    poll_secs = 120
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
    print(f"  SR extracted: {result['n_sr']}/{result['total']}")
    print(f"  2-hop: {result['n_hop2']}/{result['total']}")
    print(f"  Avg latency: {result['avg_latency_ms']:.0f} ms/case")
    tag = result["tag"]
    print(f"\nDownload results:")
    print(f"  modal volume get mscd-checkpoints "
          f"/mscd-lora-v4/eval_constraints_{tag}.jsonl "
          f"./output/")
    cases_hint = cases if cases else "evaluation/cases/cases_v3_test.jsonl"
    print(f"\nRun local pipeline:")
    print(f"  python script/run.py --profile v2_lora \\")
    print(f"    --cases {cases_hint} \\")
    print(f"    --precomputed output/eval_constraints_{tag}.jsonl")
