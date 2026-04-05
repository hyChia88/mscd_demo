#!/usr/bin/env python3
"""Shared helpers for LoRA6 Group 4 post-hoc analysis."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PROJECT_ROOT.parent
EXPERIMENT_ROOT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331"
GROUP4_ROOT = EXPERIMENT_ROOT / "group4_post-hoc_analysis"
DEFAULT_DATE_TAG = "20260404"

CASES_PATH = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
GT_EVAL_PATH = (
    REPO_ROOT
    / "data_curation"
    / "datasets"
    / "synth_v0.5_ap"
    / "train"
    / "lora6_v2_ap_eval_canonical_m.jsonl"
)
ELEMENT_INDEX_PATH = REPO_ROOT / "data_curation" / "references" / "element_index.jsonl"
WALL_REGION_INDEX_PATH = (
    REPO_ROOT / "data_curation" / "references" / "wall_region_index_ap_20260331_c.jsonl"
)
AP_IFC_PATH = REPO_ROOT / "data_curation" / "ifc_models" / "AdvancedProject.ifc"
METRICS_DIR = EXPERIMENT_ROOT / "metrics"
ORACLE_PHASE3_DIR = EXPERIMENT_ROOT / "oracle_phase3_fixed"

G3_PRED_PATH = EXPERIMENT_ROOT / "g3_fullaug_r32__ap_eval.jsonl"
G4_PRED_PATH = EXPERIMENT_ROOT / "g4_ultimate__ap_eval.jsonl"
G7_PRED_PATH = EXPERIMENT_ROOT / "g7_position_context__ap_eval.jsonl"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    import csv

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def extract_assistant_label(record: dict) -> dict:
    for message in record.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return json.loads(content)
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return json.loads(first["text"])
    raise ValueError(f"Assistant label missing in record {record.get('id')}")


def load_cases_map(path: Path = CASES_PATH) -> Dict[str, dict]:
    return {row["case_id"]: row for row in load_jsonl(path)}


def load_gt_eval_labels(path: Path = GT_EVAL_PATH) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in load_jsonl(path):
        out[row["id"]] = extract_assistant_label(row)
    return out


def load_prediction_constraints(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in load_jsonl(path):
        out[row["case_id"]] = row.get("constraints", {}) or {}
    return out


def load_element_index(path: Path = ELEMENT_INDEX_PATH) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    for row in load_jsonl(path):
        guid = row.get("global_id")
        if guid:
            rows[guid] = row
    return rows


def normalize_storey(storey_name: Optional[str]) -> str:
    if not storey_name:
        return ""
    s = str(storey_name).strip()
    lowered = s.lower()
    if "garage" in lowered:
        return "-1"
    match = re.match(r"^\s*(-?\d+)\s*-\s*", s)
    if match:
        return match.group(1)
    match = re.match(r"^\s*level\s*(-?\d+)\b", lowered)
    if match:
        return match.group(1)
    match = re.match(r"^\s*(-?\d+)\s*$", s)
    if match:
        return match.group(1)
    return s


def type_matches(actual: Optional[str], expected: Optional[str]) -> bool:
    if not actual or not expected:
        return False
    a = str(actual)
    e = str(expected)
    return a == e or a.startswith(e)


def subtype_keyword_from_row(row: Optional[dict]) -> Optional[str]:
    if not row:
        return None
    value = (
        row.get("target_name_keyword")
        or row.get("object_type")
        or row.get("name")
        or ""
    )
    if not value:
        return None
    value = str(value).strip()
    parts = [part.strip() for part in value.split(":") if part.strip()]
    if len(parts) >= 2 and not row.get("target_name_keyword"):
        value = parts[1]
    return value.lower() or None


def relation_key(rel: dict, level: str) -> Tuple[str, ...]:
    predicate = str(rel.get("predicate") or "")
    object_type = str(rel.get("object_type") or "")
    direction = str(rel.get("direction") or "")
    object_subtype = str(rel.get("object_subtype") or "")
    object_material = str(rel.get("object_material") or "")
    host_name = str(rel.get("host_name") or "")

    if level == "predicate_only":
        return (predicate,)
    if level == "pred_obj":
        return (predicate, object_type)
    if level == "pred_obj_dir":
        return (predicate, object_type, direction)
    if level == "sr_full":
        return (predicate, object_type, direction, object_subtype, object_material, host_name)
    raise ValueError(f"Unknown relation level: {level}")


def relation_signature(rels: Iterable[dict], level: str) -> Tuple[Tuple[str, ...], ...]:
    return tuple(sorted(relation_key(rel, level) for rel in rels))


def label_signature(label: dict, level: str) -> Any:
    if level in {"predicate_only", "pred_obj", "pred_obj_dir", "sr_full"}:
        return relation_signature(label.get("spatial_relations", []) or [], level)
    if level != "label_full":
        raise ValueError(f"Unknown label level: {level}")
    return (
        normalize_storey(label.get("storey_name")),
        str(label.get("ifc_class") or ""),
        str(label.get("space_name") or ""),
        str(label.get("target_name_keyword") or ""),
        str(label.get("position_context") or ""),
        relation_signature(label.get("spatial_relations", []) or [], "sr_full"),
    )


def topology_family(rels: List[dict]) -> str:
    preds = Counter(str(r.get("predicate") or "") for r in rels)
    n = len(rels)
    if n == 0:
        return "empty"
    if n == 1:
        return f"singleton:{rels[0].get('predicate')}"
    if n == 2:
        if preds == Counter({"FILLS": 1, "NEXT_TO": 1}):
            return "paired:FILLS+NEXT_TO"
        return "paired:other"
    if n == 3:
        if preds == Counter({"FILLS": 1, "NEXT_TO": 2}):
            next_types = {
                str(r.get("object_type") or "")
                for r in rels
                if str(r.get("predicate") or "") == "NEXT_TO"
            }
            if len(next_types) > 1:
                return "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)"
            return "triad:FILLS+NEXT_TO+NEXT_TO"
        return "triad:other"
    return f"{n}-rel:other"


def universe_key(rels: List[dict]) -> str:
    family = topology_family(rels)
    if family == "singleton:CONNECTS_TO":
        return "U1"
    if family == "singleton:ADJACENT_TO":
        return "U2"
    if family == "paired:FILLS+NEXT_TO":
        return "U3"
    if family == "triad:FILLS+NEXT_TO+NEXT_TO":
        return "U4"
    if family == "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)":
        return "U5"
    return "U6"


def ordered_unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_next_to_context(ifc_path: Path, element_index: Dict[str, Dict]) -> Dict[str, Dict]:
    try:
        import ifcopenshell
        import ifcopenshell.util.element as ifc_util
        import ifcopenshell.util.placement
        import numpy as np
    except Exception:
        return {}

    if not ifc_path.exists():
        return {}

    ifc = ifcopenshell.open(str(ifc_path))
    opening_to_host: Dict[str, str] = {}
    host_fillers: Dict[str, List[Any]] = defaultdict(list)

    for rel in ifc.by_type("IfcRelVoidsElement"):
        host = rel.RelatingBuildingElement
        opening = rel.RelatedOpeningElement
        if host and opening:
            opening_to_host[opening.GlobalId] = host.GlobalId

    for rel in ifc.by_type("IfcRelFillsElement"):
        filler = rel.RelatedBuildingElement
        opening = rel.RelatingOpeningElement
        if not filler or not opening:
            continue
        wall_guid = opening_to_host.get(opening.GlobalId)
        if wall_guid:
            host_fillers[wall_guid].append(filler)

    out: Dict[str, Dict] = {}
    for wall_guid, fillers in host_fillers.items():
        if len(fillers) < 2:
            continue
        try:
            wall = ifc.by_guid(wall_guid)
            wall_mat = ifcopenshell.util.placement.get_local_placement(wall.ObjectPlacement)
            wall_dir = np.array([wall_mat[0][0], wall_mat[1][0], wall_mat[2][0]])
            wall_origin = np.array([wall_mat[0][3], wall_mat[1][3], wall_mat[2][3]])
        except Exception:
            continue

        storey_groups: Dict[str, List[Any]] = defaultdict(list)
        for filler in fillers:
            storey = element_index.get(filler.GlobalId, {}).get("storey_name")
            if not storey:
                container = ifc_util.get_container(filler)
                storey = container.Name if container else "_unknown"
            storey_groups[storey].append(filler)

        for storey_name, group in storey_groups.items():
            projections = []
            for filler in group:
                try:
                    mat = ifcopenshell.util.placement.get_local_placement(filler.ObjectPlacement)
                    centroid = np.array([mat[0][3], mat[1][3], mat[2][3]])
                    proj = float(np.dot(centroid - wall_origin, wall_dir))
                    projections.append((proj, filler))
                except Exception:
                    continue
            projections.sort(key=lambda x: x[0])
            total = len(projections)
            for idx, (_, filler) in enumerate(projections):
                left = projections[idx - 1][1] if idx > 0 else None
                right = projections[idx + 1][1] if idx + 1 < total else None
                out[filler.GlobalId] = {
                    "host_guid": wall_guid,
                    "storey_name": storey_name,
                    "position_index": idx + 1,
                    "position_total": total,
                    "left_neighbor_guid": left.GlobalId if left else None,
                    "left_neighbor_type": left.is_a() if left else None,
                    "right_neighbor_guid": right.GlobalId if right else None,
                    "right_neighbor_type": right.is_a() if right else None,
                }
    return out


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)
