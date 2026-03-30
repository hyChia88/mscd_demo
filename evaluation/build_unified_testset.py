#!/usr/bin/env python3
"""
Build unified evaluation test set by merging:
  1. cases_v5_test.jsonl  (70 cases — LoRA5/Gemini test set, mostly AP + spatial)
  2. LoRA2 v0.4 test      (50 cases — AP/BH/DXA, attribute-only)

Output: evaluation/cases/cases_unified_test.jsonl (~120 cases, deduplicated)

Usage (from mscd_demo/):
    python evaluation/build_unified_testset.py
    python evaluation/build_unified_testset.py --dry-run
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "data_curation"

# Source 1: existing cases_v5
CASES_V5 = PROJECT_ROOT / "evaluation" / "cases" / "cases_v5_test.jsonl"

# Source 2: LoRA2 test (ChatML training format)
LORA2_TEST = DATA_ROOT / "datasets" / "synth_v0.4_merged" / "train" / "lora_test.jsonl"

# Skeletons for ground truth
SKELETON_PATHS = {
    "AP": DATA_ROOT / "datasets" / "synth_v0.4_ap" / "skeletons" / "skeletons_v3.jsonl",
    "BH": DATA_ROOT / "datasets" / "synth_v0.4_bh" / "skeletons" / "skeletons_v3.jsonl",
    "DXA": DATA_ROOT / "datasets" / "synth_v0.4_dxa" / "skeletons" / "skeletons_v3.jsonl",
}

OUTPUT = PROJECT_ROOT / "evaluation" / "cases" / "cases_unified_test.jsonl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_skeletons() -> dict:
    """Load v0.4 skeletons keyed by merged/export IDs such as AP_SK_084."""
    skeletons = {}
    for model, spath in SKELETON_PATHS.items():
        if not spath.exists():
            print(f"  WARNING: skeleton file not found: {spath}")
            continue
        for rec in load_jsonl(spath):
            skeletons[rec["id"]] = rec
    return skeletons


def detect_ifc_model(case_id: str, ground_truth: dict | None = None) -> str:
    """Extract IFC model tag from case ID or ground truth storey."""
    if "_AP_" in case_id:
        return "AP"
    elif "_BH_" in case_id:
        return "BH"
    elif "_DXA_" in case_id:
        return "DXA"
    # Fallback: infer from storey naming convention
    if ground_truth:
        storey = ground_truth.get("target_storey", "")
        if "First Floor" in storey or "Second Floor" in storey or "Sixth Floor" in storey or "Garage" in storey:
            return "AP"
        if storey.startswith("Floor "):
            return "BH"
        if storey.startswith("Level ") or "- Fifth Floor" in storey:
            return "DXA"
    return "AP"  # default to AP (most TEST_DISCARD cases are AP)


def convert_lora2_record(rec: dict, skeletons: dict) -> dict:
    """Convert one LoRA2 ChatML training record → cases_v5 eval format."""
    record_id = rec["id"]
    msgs = rec.get("messages", [])
    user_msg = next((m for m in msgs if m["role"] == "user"), None)
    asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)

    if not user_msg or not asst_msg:
        return None

    # ── Parse user text ───────────────────────────────────────────────────
    content = user_msg["content"]
    if isinstance(content, list):
        user_text = " ".join(
            c["text"] for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        )
    else:
        user_text = str(content)

    parsed = {"task_status": "", "project_phase": "", "chat_lines": [], "query_text": ""}

    m = re.search(r"\[4D Task Status\]\s*(.+?)(?:\n|$)", user_text)
    if m:
        parsed["task_status"] = m.group(1).strip()

    m = re.search(r"\[Project Phase\]\s*(.+?)(?:\n|$)", user_text)
    if m:
        parsed["project_phase"] = m.group(1).strip()

    chat_m = re.search(
        r"\[Chat Log\]\s*\n(.*?)(?:\n\[Query\]|\nExtract the search)",
        user_text, re.DOTALL)
    if chat_m:
        for line in chat_m.group(1).strip().split("\n"):
            line = line.strip()
            m2 = re.match(r"(\w[\w\s]*?):\s*(.+)", line)
            if m2:
                parsed["chat_lines"].append({
                    "role": m2.group(1).strip(),
                    "text": m2.group(2).strip(),
                })

    m = re.search(r"\[Query\]\s*(.+?)(?:\nExtract|\n\n|$)", user_text, re.DOTALL)
    if m:
        parsed["query_text"] = m.group(1).strip()

    # ── Parse GT constraints (old LoRA2 schema) ──────────────────────────
    try:
        gt = json.loads(asst_msg["content"])
    except (json.JSONDecodeError, TypeError):
        return None

    # ── Extract image paths ───────────────────────────────────────────────
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
            is_fp = ("floorplan" in img_path
                     or fname.startswith("plan_")
                     or "/plans/" in img_path)
            if is_fp:
                floorplan = img_path
            else:
                images.append(img_path)

    # ── Ground truth from skeleton ────────────────────────────────────────
    sk_match = re.search(r"((?:AP|BH|DXA)_SK_\d+)", record_id)
    sk_id = sk_match.group(1) if sk_match else None
    skel = skeletons.get(sk_id) if sk_id else None

    if skel:
        target_props = skel.get("target_props", {})
        ground_truth = {
            "target_guid": skel["target_guid"],
            "target_storey": target_props.get("Storey", ""),
            "target_ifc_class": target_props.get("Type", ""),
            "target_name": target_props.get("Name", ""),
        }
        difficulty = skel.get("difficulty", skel.get("v3_tier", "Tier 2"))
        requires_relation = skel.get("requires_relation", False)
        pattern = skel.get("pattern", "")
    else:
        ground_truth = {
            "target_guid": "",
            "target_storey": gt.get("storey_name", ""),
            "target_ifc_class": gt.get("ifc_class", ""),
            "target_name": "",
        }
        difficulty = "Tier 2"
        requires_relation = False
        pattern = ""

    # ── Bench group ───────────────────────────────────────────────────────
    has_images = len(images) > 0
    has_fp = floorplan is not None
    if has_images and has_fp:
        group, condition = "C", "C1"
    elif has_images:
        group, condition = "B", "B1"
    elif has_fp:
        group, condition = "A", "A1"
    else:
        group, condition = "D", "D1"

    return {
        "case_id": record_id,
        "source": "lora2_v04",
        "ifc_model": detect_ifc_model(record_id, ground_truth),
        "bench": {"group": group, "condition": condition},
        "difficulty_tags": {
            "tier": difficulty,
            "requires_relation": requires_relation,
            "spatial_predicate": None,
            "pattern": pattern,
        },
        "ground_truth": ground_truth,
        "inputs": {
            "chat_history": parsed["chat_lines"],
            "chat_quality": "clear",
            "project_context": {
                "4d_task_status": parsed["task_status"],
                "project_phase": parsed["project_phase"],
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
            },
        },
        "query_text": parsed["query_text"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build unified eval test set")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=== Building unified test set ===\n")

    # 1. Load cases_v5 (already in eval format)
    v5_cases = load_jsonl(CASES_V5)
    print(f"Source 1: cases_v5_test.jsonl → {len(v5_cases)} cases")
    # Tag source
    for c in v5_cases:
        if "source" not in c:
            c["source"] = "lora5_v05"
        if "ifc_model" not in c:
            c["ifc_model"] = detect_ifc_model(c["case_id"], c.get("ground_truth"))

    # 2. Convert LoRA2 cases
    skeletons = load_skeletons()
    print(f"  Loaded {len(skeletons)} skeletons for ground truth")

    lora2_records = load_jsonl(LORA2_TEST)
    print(f"Source 2: lora_test.jsonl → {len(lora2_records)} records")

    lora2_cases = []
    for rec in lora2_records:
        case = convert_lora2_record(rec, skeletons)
        if case:
            lora2_cases.append(case)
    print(f"  Converted: {len(lora2_cases)} cases")

    # 3. Deduplicate by base case ID (strip augmentation suffix)
    def base_id(case_id: str) -> str:
        return re.sub(r"_aug[A-Z]$", "", case_id)

    seen = {}
    unified = []

    # v5 cases first (higher priority)
    for c in v5_cases:
        bid = base_id(c["case_id"])
        if bid not in seen:
            seen[bid] = c["case_id"]
            unified.append(c)

    # Then LoRA2 cases (skip duplicates)
    dupes = 0
    for c in lora2_cases:
        bid = base_id(c["case_id"])
        if bid not in seen:
            seen[bid] = c["case_id"]
            unified.append(c)
        else:
            dupes += 1

    print(f"\n  Duplicates skipped: {dupes}")
    print(f"  Unified total: {len(unified)} cases")

    # 4. Stats
    models = Counter(c.get("ifc_model", detect_ifc_model(c["case_id"], c.get("ground_truth"))) for c in unified)
    sources = Counter(c.get("source", "?") for c in unified)
    has_sr = sum(1 for c in unified if c["difficulty_tags"]["requires_relation"])
    has_gt = sum(1 for c in unified if c["ground_truth"]["target_guid"])
    tiers = Counter(c["difficulty_tags"]["tier"] for c in unified)

    print(f"\n  By IFC model:  {dict(models)}")
    print(f"  By source:     {dict(sources)}")
    print(f"  By tier:       {dict(tiers)}")
    print(f"  Spatial cases: {has_sr}")
    print(f"  Has GT GUID:   {has_gt}/{len(unified)}")

    # 5. Validate image paths exist
    missing_imgs = 0
    for c in unified:
        for img in c["inputs"].get("images", []):
            full = DATA_ROOT / img if not Path(img).is_absolute() else Path(img)
            if not full.exists():
                missing_imgs += 1
                if missing_imgs <= 3:
                    print(f"  WARN: missing image {img}")
        fp = c["inputs"].get("floorplan_patch")
        if fp:
            full = DATA_ROOT / fp if not Path(fp).is_absolute() else Path(fp)
            if not full.exists():
                missing_imgs += 1
                if missing_imgs <= 3:
                    print(f"  WARN: missing floorplan {fp}")

    if missing_imgs:
        print(f"  Total missing images: {missing_imgs}")
    else:
        print(f"  All images verified ✓")

    # 6. Write
    if args.dry_run:
        print(f"\n  [DRY RUN] Would write {len(unified)} cases to {OUTPUT}")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w") as f:
            for case in unified:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"\n  Written: {OUTPUT}")
        print(f"  {len(unified)} cases ready for 3-way evaluation")


if __name__ == "__main__":
    main()
