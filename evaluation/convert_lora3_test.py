#!/usr/bin/env python3
"""
T2.1 — Convert lora3_test.jsonl → cases_v3_test.jsonl

Converts 69 held-out LoRA_3 test records (ChatML format) into the
cases_v3 format expected by mscd_demo/script/run.py for evaluation.

Sources:
  - lora3_test.jsonl       : user text (4D + chat + query), assistant JSON (ground truth constraints)
  - skeletons_v2_5.jsonl   : target_guid, target_props, spatial_predicate, ref_element_*
  - skins.jsonl            : image_site path
  - floorplans/            : {BENCH}_SK_{NNN}_floorplan.png

Output:
  - mscd_demo/eval/cases_v3_test.jsonl  (69 records, cases_v3 format)
"""

import json
import os
import re
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data_curation" / "datasets" / "synth_v0.5"
LORA3_TEST  = DATA_ROOT / "train" / "lora3_test.jsonl"
SKELETONS   = DATA_ROOT / "skeletons" / "skeletons_v2_5.jsonl"
SKINS       = DATA_ROOT / "skins" / "skins.jsonl"
FLOORPLANS  = DATA_ROOT / "floorplans"
IMGS_DIR    = DATA_ROOT / "imgs"

OUTPUT      = Path(__file__).resolve().parent / "cases" / "cases_v3_test.jsonl"


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_user_text(text: str) -> dict:
    """
    Parse the structured user text block into components:
      [4D Task Status] ...
      [Project Phase] ...
      [Chat Log] ...
      [Query] ...
    """
    result = {
        "task_status": "",
        "project_phase": "",
        "chat_lines": [],
        "query_text": "",
    }

    # Extract 4D Task Status
    m = re.search(r"\[4D Task Status\]\s*(.+?)(?:\n|$)", text)
    if m:
        result["task_status"] = m.group(1).strip()

    # Extract Project Phase
    m = re.search(r"\[Project Phase\]\s*(.+?)(?:\n|$)", text)
    if m:
        result["project_phase"] = m.group(1).strip()

    # Extract Chat Log lines
    chat_match = re.search(r"\[Chat Log\]\s*\n(.*?)(?:\n\[Query\]|\nExtract the search)", text, re.DOTALL)
    if chat_match:
        chat_block = chat_match.group(1).strip()
        for line in chat_block.split("\n"):
            line = line.strip()
            # Parse "Role: text" format
            m2 = re.match(r"(\w[\w\s]*?):\s*(.+)", line)
            if m2:
                result["chat_lines"].append({
                    "role": m2.group(1).strip(),
                    "text": m2.group(2).strip(),
                })

    # Extract Query
    m = re.search(r"\[Query\]\s*(.+?)(?:\nExtract|\n\n|$)", text, re.DOTALL)
    if m:
        result["query_text"] = m.group(1).strip()

    return result


def main():
    # ── Load sources ───────────────────────────────────────────────────────────
    test_records = load_jsonl(LORA3_TEST)
    print(f"Loaded {len(test_records)} test records from {LORA3_TEST.name}")

    skeletons = {}
    for rec in load_jsonl(SKELETONS):
        skeletons[rec["id"]] = rec
    print(f"Loaded {len(skeletons)} skeletons")

    skins = {}
    for rec in load_jsonl(SKINS):
        skins[rec["skeleton_id"]] = rec
    print(f"Loaded {len(skins)} skins")

    # ── Convert ────────────────────────────────────────────────────────────────
    cases = []
    errors = []

    for rec in test_records:
        record_id = rec["id"]

        # Extract skeleton ID (SK_NNN) from record ID
        sk_match = re.search(r"SK_(\d+)", record_id)
        if not sk_match:
            errors.append(f"{record_id}: no SK_ in ID")
            continue
        sk_id = f"SK_{sk_match.group(1)}"

        # Extract bench (AP/BH/DXA)
        bench_match = re.search(r"(AP|BH|DXA)", record_id)
        bench = bench_match.group(1) if bench_match else "AP"

        # Lookup skeleton
        skel = skeletons.get(sk_id)
        if not skel:
            errors.append(f"{record_id}: skeleton {sk_id} not found")
            continue

        # Lookup skin (optional — for image path)
        skin = skins.get(sk_id)

        # ── Parse messages ─────────────────────────────────────────────────
        msgs = rec.get("messages", [])
        user_msg = next((m for m in msgs if m["role"] == "user"), None)
        asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)

        if not user_msg or not asst_msg:
            errors.append(f"{record_id}: missing user/assistant message")
            continue

        # Get user text
        content = user_msg["content"]
        if isinstance(content, list):
            user_text = " ".join(
                c["text"] for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        else:
            user_text = str(content)

        # Parse structured fields from user text
        parsed = parse_user_text(user_text)

        # Parse ground truth constraints from assistant
        try:
            gt_constraints = json.loads(asst_msg["content"])
        except (json.JSONDecodeError, TypeError):
            errors.append(f"{record_id}: invalid assistant JSON")
            continue

        # ── Build image paths ──────────────────────────────────────────────
        images = []
        # Site photo from skins
        if skin and skin.get("image_site"):
            site_path = skin["image_site"]
            # Resolve to absolute to check existence, store as relative
            abs_site = DATA_ROOT.parent.parent / site_path
            if abs_site.exists():
                images.append(site_path)

        # Wireframe as fallback
        wireframe_path = f"datasets/synth_v0.5/renders/wireframes/{sk_id}_wireframe.png"
        abs_wf = DATA_ROOT.parent.parent / wireframe_path
        if not images and abs_wf.exists():
            images.append(wireframe_path)

        # Floorplan
        fp_name = f"{bench}_{sk_id}_floorplan.png"
        fp_path = f"datasets/synth_v0.5/floorplans/{fp_name}"
        abs_fp = FLOORPLANS / fp_name
        floorplan = fp_path if abs_fp.exists() else None

        # ── Build ground truth ─────────────────────────────────────────────
        target_props = skel.get("target_props", {})
        ground_truth = {
            "target_guid": skel["target_guid"],
            "target_storey": target_props.get("Storey", ""),
            "target_ifc_class": target_props.get("Type", ""),
            "target_name": target_props.get("Name", ""),
        }

        # ── Build labels.constraints ───────────────────────────────────────
        # Use the assistant output as ground truth constraints
        labels_constraints = {
            "storey_name": gt_constraints.get("storey_name"),
            "ifc_class": gt_constraints.get("ifc_class"),
            "near_keywords": gt_constraints.get("near_keywords", []),
            "relations": gt_constraints.get("relations", []),
            "space_name": gt_constraints.get("space_name"),
            "target_name_keyword": gt_constraints.get("target_name_keyword"),
            "neighbor_type": gt_constraints.get("neighbor_type"),
        }

        # Add spatial_relations if present
        spatial_rels = gt_constraints.get("spatial_relations", [])
        if spatial_rels:
            labels_constraints["spatial_relations"] = spatial_rels

        # ── Determine difficulty / group ───────────────────────────────────
        has_spatial = bool(spatial_rels)
        has_floorplan = floorplan is not None
        has_images = len(images) > 0

        # Bench group: A (text-only), B (site photo), C (site photo + floorplan)
        if has_images and has_floorplan:
            group, condition = "C", "C1"
        elif has_images:
            group, condition = "B", "B1"
        else:
            group, condition = "A", "A1"

        # ── Assemble case ─────────────────────────────────────────────────
        case = {
            "case_id": record_id,
            "bench": {
                "group": group,
                "condition": condition,
            },
            "difficulty_tags": {
                "tier": skel.get("difficulty", "Tier 3"),
                "requires_relation": skel.get("requires_relation", False),
                "spatial_predicate": skel.get("spatial_predicate"),
                "pattern": skel.get("pattern", ""),
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
                "constraints": labels_constraints,
            },
            "query_text": parsed["query_text"],
        }

        cases.append(case)

    # ── Write output ───────────────────────────────────────────────────────────
    with open(OUTPUT, "w") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"\nConverted {len(cases)} cases → {OUTPUT}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    # ── Summary stats ──────────────────────────────────────────────────────────
    groups = {}
    spatial_count = 0
    with_images = 0
    with_floorplan = 0
    for c in cases:
        g = c["bench"]["group"]
        groups[g] = groups.get(g, 0) + 1
        if c["labels"]["constraints"].get("spatial_relations"):
            spatial_count += 1
        if c["inputs"]["images"]:
            with_images += 1
        if c["inputs"]["floorplan_patch"]:
            with_floorplan += 1

    print(f"\nStats:")
    print(f"  Groups: {groups}")
    print(f"  With spatial_relations: {spatial_count}/{len(cases)}")
    print(f"  With images: {with_images}/{len(cases)}")
    print(f"  With floorplan: {with_floorplan}/{len(cases)}")


if __name__ == "__main__":
    main()
