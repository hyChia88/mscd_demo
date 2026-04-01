#!/usr/bin/env python3
"""Build AP held-out end-to-end benchmark cases for Track B-2.

Converts the assembled AP eval split into the legacy case schema consumed by
`script/run.py`, while preserving trustworthy AP main-dataset ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data_curation"
PROJECT_ROOT = REPO_ROOT / "mscd_demo"

DEFAULT_AP_EVAL = (
    DATA_ROOT
    / "datasets"
    / "synth_v0.5_ap"
    / "train"
    / "lora6_v2_ap_eval_canonical_m.jsonl"
)
DEFAULT_SKINS = (
    DATA_ROOT
    / "datasets"
    / "synth_v0.5_ap"
    / "skins"
    / "skins_integrated.jsonl"
)
DEFAULT_SKELETONS = (
    DATA_ROOT
    / "datasets"
    / "synth_v0.5_ap"
    / "skeletons"
    / "skeletons.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_assistant_constraints(messages: List[dict]) -> Dict[str, Any]:
    assistant = next((m for m in messages if m.get("role") == "assistant"), {})
    raw = (assistant.get("content") or "").strip()
    return json.loads(raw) if raw else {}


def _extract_user_payload(messages: List[dict]) -> tuple[str, List[str]]:
    user = next((m for m in messages if m.get("role") == "user"), {})
    content = user.get("content")
    texts: List[str] = []
    images: List[str] = []
    if isinstance(content, str):
        texts.append(content.strip())
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                texts.append((item.get("text") or "").strip())
            elif item.get("type") == "image":
                images.append(item.get("image") or "")
    return " ".join(t for t in texts if t), images


def _to_case_image_path(raw: str) -> Optional[str]:
    if not raw:
        return None
    parsed = urlparse(raw)
    text = parsed.path if parsed.scheme == "file" else raw
    path = Path(text)
    try:
        rel = path.relative_to(DATA_ROOT)
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def _infer_floorplan_path(image_paths: Iterable[str]) -> Optional[str]:
    for path in image_paths:
        lowered = path.lower()
        if "floorplan" in lowered:
            return path
    return None


def _infer_site_images(image_paths: Iterable[str]) -> List[str]:
    results = []
    for path in image_paths:
        lowered = path.lower()
        if "floorplan" not in lowered:
            results.append(path)
    return results


def _make_chat_history(text: str) -> List[dict]:
    if not text:
        return []
    return [{"role": "Inspector", "text": text}]


def build_cases(
    ap_eval_rows: List[dict],
    skins_rows: List[dict],
    skeleton_rows: List[dict],
) -> List[dict]:
    skin_by_id = {row["skeleton_id"]: row for row in skins_rows}
    skeleton_by_id = {row["id"]: row for row in skeleton_rows}

    cases: List[dict] = []
    for row in ap_eval_rows:
        case_id = row["id"]
        skin = skin_by_id.get(case_id)
        skeleton = skeleton_by_id.get(case_id)
        if skin is None or skeleton is None:
            raise KeyError(f"Missing AP main-dataset row for {case_id}")

        constraints = _parse_assistant_constraints(row.get("messages") or [])
        query_text, raw_images = _extract_user_payload(row.get("messages") or [])
        image_paths = [_to_case_image_path(img) for img in raw_images]
        image_paths = [p for p in image_paths if p]
        site_images = _infer_site_images(image_paths)
        floorplan_patch = _infer_floorplan_path(image_paths)

        spatial_relations = constraints.get("spatial_relations") or []
        spatial_predicate = None
        if spatial_relations and isinstance(spatial_relations[0], dict):
            spatial_predicate = spatial_relations[0].get("predicate")

        target_props = skeleton.get("target_props") or {}

        cases.append(
            {
                "case_id": case_id,
                "bench": {
                    "group": "C",
                    "condition": "C1",
                },
                "difficulty_tags": {
                    "tier": skin.get("difficulty") or skin.get("base_entropy_tier") or "Tier 3",
                    "requires_relation": bool(spatial_relations),
                    "spatial_predicate": spatial_predicate,
                    "pattern": skin.get("pattern"),
                },
                "ground_truth": {
                    "target_guid": skin.get("retrieval_gt_guid") or skin.get("target_guid"),
                    "target_storey": skin.get("storey_name"),
                    "target_ifc_class": skin.get("subject_type"),
                    "target_name": target_props.get("Name"),
                },
                "inputs": {
                    "chat_history": _make_chat_history(query_text),
                    "chat_quality": "clear",
                    "project_context": {
                        "4d_task_status": "AP held-out evaluation",
                        "project_phase": "LoRA6-v2 AP benchmark",
                    },
                    "images": site_images,
                    "floorplan_patch": floorplan_patch,
                },
                "labels": {"constraints": constraints},
                "query_text": query_text,
            }
        )

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ap-eval", type=Path, default=DEFAULT_AP_EVAL)
    parser.add_argument("--skins", type=Path, default=DEFAULT_SKINS)
    parser.add_argument("--skeletons", type=Path, default=DEFAULT_SKELETONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ap_eval_rows = _load_jsonl(args.ap_eval)
    skins_rows = _load_jsonl(args.skins)
    skeleton_rows = _load_jsonl(args.skeletons)
    cases = build_cases(ap_eval_rows, skins_rows, skeleton_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Wrote {len(cases)} AP held-out e2e cases to {args.output}")


if __name__ == "__main__":
    main()
