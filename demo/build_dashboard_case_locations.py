"""Build first-floor dashboard cases from canonical AP training data.

Usage:
    conda activate mscd_demo
    python demo/build_dashboard_case_locations.py
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import ifcopenshell
import ifcopenshell.util.placement as placement


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data_curation"
CASES_PATH = REPO_ROOT / "demo" / "static" / "dashboard_first_floor_cases.json"
PASSPORTS_PATH = REPO_ROOT / "demo" / "static" / "dashboard_traceability_passports.jsonl"
CALIBRATION_PATH = DATA_ROOT / "datasets" / "synth_v0.5_ap" / "floorplans_full" / "calibration.json"
IFC_PATH = DATA_ROOT / "ifc_models" / "AdvancedProject.ifc"
TRAIN_CASES_PATH = DATA_ROOT / "datasets" / "synth_v0.5_ap" / "train" / "lora6_v2_ap_train_canonical_m_g9.jsonl"
SKELETONS_PATH = DATA_ROOT / "datasets" / "synth_v0.5_ap" / "skeletons" / "skeletons.jsonl"
SKINS_PATH = DATA_ROOT / "datasets" / "synth_v0.5_ap" / "skins" / "skins_integrated.jsonl"
FIRST_FLOOR_KEYS = {"1 - first floor", "level 1"}
CANONICAL_DASHBOARD_STOREY = "1 - First Floor"


def _load_calibration_index() -> dict[str, dict]:
    raw = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return {
        str(entry.get("storey_name", "")).strip().lower(): entry
        for entry in raw.get("storeys", [])
        if entry.get("storey_name")
    }


def _load_jsonl_index(path: Path, *, key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_key = str(row.get(key) or "").strip()
            if row_key:
                out[row_key] = row
    return out


def _normalize_asset_path(raw_path: str | None) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    if text.startswith("file://"):
        parsed = urlparse(text)
        local_path = Path(parsed.path)
        try:
            rel = local_path.resolve().relative_to(PROJECT_ROOT)
            return f"/{rel.as_posix()}"
        except ValueError:
            return local_path.as_posix()
    if text.startswith("data_curation/"):
        return f"/{text}"
    if text.startswith("datasets/"):
        return f"/data_curation/{text}"
    return text if text.startswith("/") else f"/{text}"


def _extract_user_payload(messages: list[dict]) -> tuple[str, list[str]]:
    query_text = ""
    image_paths: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            query_text = content.strip() or query_text
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "text" and item.get("text"):
                query_text = str(item["text"]).strip() or query_text
            elif item.get("type") == "image" and item.get("image"):
                image_paths.append(str(item["image"]).strip())
    return query_text, image_paths


def _infer_floorplan_patch(image_paths: list[str]) -> str:
    for path in image_paths:
        lowered = path.lower()
        if "floorplan" in lowered or "/floorplans/" in lowered or "/floorplans_v2/" in lowered:
            return _normalize_asset_path(path)
    return ""


def _infer_site_images(image_paths: list[str]) -> list[str]:
    out: list[str] = []
    for path in image_paths:
        lowered = path.lower()
        if "site" in lowered or "/imgs/" in lowered:
            out.append(_normalize_asset_path(path))
    return out


def _parse_assistant_constraints(messages: list[dict]) -> dict:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_dashboard_storey(storey: str) -> str:
    text = str(storey or "").strip()
    if text.lower() in FIRST_FLOOR_KEYS:
        return CANONICAL_DASHBOARD_STOREY
    return text


def _project_guid_to_pixel(model, guid: str, calibration: dict) -> dict | None:
    element = model.by_guid(guid)
    if not element or not getattr(element, "ObjectPlacement", None):
        return None

    matrix = placement.get_local_placement(element.ObjectPlacement)
    world_x_m = float(matrix[0][3]) / 1000.0
    world_y_m = float(matrix[1][3]) / 1000.0

    world_bbox = calibration["world_bbox"]
    pixel_size = calibration["pixel_size"]
    span_x = float(world_bbox["xmax"]) - float(world_bbox["xmin"])
    span_y = float(world_bbox["ymax"]) - float(world_bbox["ymin"])
    if span_x <= 0 or span_y <= 0:
        return None

    px = (world_x_m - float(world_bbox["xmin"])) / span_x * float(pixel_size["width"])
    py = float(pixel_size["height"]) - (
        (world_y_m - float(world_bbox["ymin"])) / span_y * float(pixel_size["height"])
    )
    x_pct = px / float(pixel_size["width"]) * 100.0
    y_pct = py / float(pixel_size["height"]) * 100.0
    in_bounds = 0.0 <= px <= float(pixel_size["width"]) and 0.0 <= py <= float(pixel_size["height"])

    return {
        "world_xy_m": {"x": round(world_x_m, 4), "y": round(world_y_m, 4)},
        "pixel_xy": {"x": round(px, 1), "y": round(py, 1)},
        "point": {"x": round(x_pct, 3), "y": round(y_pct, 3)},
        "in_bounds": in_bounds,
    }


def _passport_statuses(item: dict) -> list[dict]:
    evidence = item.get("evidence") or {}
    relation = ((item.get("parsed") or {}).get("spatial_relations") or [{}])[0]
    predicate = str(item.get("predicate") or relation.get("predicate") or "LOCATED_AT")
    object_type = str(relation.get("object_type") or "IfcElement")
    material = str(relation.get("object_material") or "project finish")
    position_context = str((item.get("parsed") or {}).get("position_context") or "field-located target")
    site_image = Path((evidence.get("images") or ["site_capture.png"])[0]).name
    floorplan_patch = Path(str(evidence.get("floorplan_patch") or "floorplan_patch.png")).name
    full_floorplan = Path(str(item.get("floorplan_image") or "AP_storey_1_first_floor.png")).name
    target_type = str(item.get("target_ifc_class") or "IfcElement")

    return [
        {
            "date": "Mar 03, 2026 • 08:15 AM",
            "status": "pass",
            "title": "Design Intent Registered",
            "desc": f"{target_type} scheduled on {item['target_storey']} with BIM baseline linked to {item['target_guid']}.",
            "visual": full_floorplan,
            "source": "BIM Register",
        },
        {
            "date": "Mar 07, 2026 • 10:20 AM",
            "status": "pass",
            "title": "Multimodal Evidence Captured",
            "desc": f"Field request recorded as {position_context}; site photo and floorplan patch attached for interpreter review.",
            "visual": site_image,
            "source": "Field Capture",
        },
        {
            "date": "Mar 07, 2026 • 10:24 AM",
            "status": "warning",
            "title": "Interpreter Grounding Candidate",
            "desc": f"Neuro-symbolic interpreter mapped the case to {target_type} with relation {predicate} {object_type} and material cue {material}.",
            "visual": floorplan_patch,
            "source": "Interpreter",
        },
        {
            "date": "Mar 07, 2026 • 10:29 AM",
            "status": "pass" if item.get("projection_status") == "guid_calibrated" else "warning",
            "title": "Allocentric BIM Confirmation",
            "desc": (
                f"Target GUID {item['target_guid']} projected onto the full first-floor dashboard and synchronized with the clickable BIM surface."
                if item.get("projection_status") == "guid_calibrated"
                else f"Target GUID {item['target_guid']} matched symbolically, but floorplan projection required fallback handling."
            ),
            "visual": full_floorplan,
            "source": "Dashboard",
        },
    ]


def _write_traceability_passports(cases: list[dict]) -> None:
    with PASSPORTS_PATH.open("w", encoding="utf-8") as handle:
        for item in cases:
            record = {
                "case_id": item["case_id"],
                "target_guid": item["target_guid"],
                "target_ifc_class": item["target_ifc_class"],
                "target_storey": item["target_storey"],
                "history": _passport_statuses(item),
            }
            handle.write(json.dumps(record) + "\n")


def _build_dashboard_cases() -> list[dict]:
    skeletons = _load_jsonl_index(SKELETONS_PATH, key="id")
    skins = _load_jsonl_index(SKINS_PATH, key="base_case_id")
    cases: list[dict] = []

    with TRAIN_CASES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            train_row = json.loads(line)
            case_id = str(train_row.get("base_case_id") or train_row.get("id") or "").strip()
            if not case_id:
                continue

            skeleton = skeletons.get(case_id)
            skin = skins.get(case_id)
            if not skeleton or not skin:
                continue

            raw_storey = str(skeleton.get("target_props", {}).get("Storey") or skin.get("storey_name") or "").strip()
            if raw_storey.lower() not in FIRST_FLOOR_KEYS:
                continue
            storey = _normalize_dashboard_storey(raw_storey)

            query_text, image_paths = _extract_user_payload(train_row.get("messages") or [])
            parsed = _parse_assistant_constraints(train_row.get("messages") or [])
            site_images = _infer_site_images(image_paths)
            floorplan_patch = _infer_floorplan_patch(image_paths)
            if not site_images and skin.get("image_site"):
                site_images = [_normalize_asset_path(skin["image_site"])]

            cases.append({
                "case_id": case_id,
                "target_guid": skeleton.get("target_guid") or skin.get("target_guid") or "",
                "target_name": skeleton.get("target_props", {}).get("Name") or "",
                "target_storey": storey,
                "target_storey_original": raw_storey,
                "target_ifc_class": skeleton.get("target_props", {}).get("Type") or skin.get("subject_type") or "",
                "predicate": train_row.get("predicate") or skeleton.get("spatial_predicate") or skin.get("predicate") or "",
                "modality": train_row.get("modality") or "site+floorplan+chat",
                "query_text": query_text or skin.get("text_chat") or "",
                "parsed": parsed,
                "projection_status": "missing_calibration",
                "floorplan_image": "",
                "point": None,
                "world_xy_m": None,
                "pixel_xy": None,
                "evidence": {
                    "chat_history": [{"role": "Inspector", "text": query_text or skin.get("text_chat") or ""}],
                    "chat_quality": "clear",
                    "project_context": {
                        "project_phase": "LoRA6-v2 AP canonical training sample",
                        "dataset_split": "train canonical g9",
                    },
                    "images": site_images,
                    "floorplan_patch": floorplan_patch,
                },
            })

    cases.sort(key=lambda item: item["case_id"])
    return cases


def main() -> None:
    payload = {"cases": _build_dashboard_cases()}
    calibration_by_storey = _load_calibration_index()
    model = ifcopenshell.open(str(IFC_PATH))

    enriched = []
    for item in payload["cases"]:
        case = dict(item)
        storey_key = str(case.get("target_storey", "")).strip().lower()
        calibration = calibration_by_storey.get(storey_key)

        if calibration:
            png_name = Path(calibration.get("png_path", "")).name
            case["floorplan_image"] = f"/demo/static/{png_name}"
            projection = _project_guid_to_pixel(model, case.get("target_guid", ""), calibration)
            if projection:
                case["world_xy_m"] = projection["world_xy_m"]
                case["pixel_xy"] = projection["pixel_xy"]
                if projection["in_bounds"]:
                    case["point"] = projection["point"]
                    case["projection_status"] = "guid_calibrated"
                else:
                    case["projection_status"] = "guid_projected_out_of_bounds"

        enriched.append(case)

    payload["cases"] = enriched
    CASES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_traceability_passports(enriched)

    ok = sum(1 for case in enriched if case.get("projection_status") == "guid_calibrated")
    print(f"Updated {len(enriched)} first-floor dashboard cases from canonical training data ({ok} calibrated in bounds).")


if __name__ == "__main__":
    main()
