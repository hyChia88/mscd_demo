#!/usr/bin/env python3
"""Run Gemini on the unified benchmark cases for Track B."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.track_registry import EXPERIMENT_ROOT

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None


DATA_ROOT = PROJECT_ROOT.parent / "data_curation"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "constraints_extraction.yaml"
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_unified_test.jsonl"
DEFAULT_MODEL = "gemini-2.5-flash"
SCHEMA_HINT = """
Return exactly one JSON object with these exact keys:
{
  "storey_name": string | null,
  "ifc_class": string | null,
  "near_keywords": [string, ...],
  "relations": [],
  "space_name": string | null,
  "target_name_keyword": string | null,
  "neighbor_type": string | null,
  "spatial_relations": [
    {
      "predicate": string,
      "object_type": string | null,
      "direction": string | null
    }
  ]
}
Use the exact key names above: storey_name, ifc_class, spatial_relations, predicate, object_type, direction.
If uncertain, use null or [] instead of renaming keys.
""".strip()

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
    "Output ONLY valid JSON."
)


def _resolve_image_path(relative_path: str) -> Optional[str]:
    if not relative_path:
        return None
    p = Path(relative_path)
    if p.is_absolute() and p.exists():
        return str(p)
    for root in (DATA_ROOT, PROJECT_ROOT):
        candidate = root / relative_path
        if candidate.exists():
            return str(candidate)
    return None


def _parse_json(text: str) -> Optional[dict]:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("{"):
        repaired = raw
        bracket_delta = repaired.count("[") - repaired.count("]")
        brace_delta = repaired.count("{") - repaired.count("}")
        if bracket_delta > 0:
            repaired += "]" * bracket_delta
        if brace_delta > 0:
            repaired += "}" * brace_delta
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for pattern in [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"(\{.*\"(?:spatial_relations|relations)\".*\})",
        r"(\{[^{]*\"storey_name\"[^}]*\})",
    ]:
        m = re.search(pattern, raw, re.DOTALL)
        if not m:
            continue
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    return _salvage_partial_json(raw)


def _extract_json_scalar(raw: str, key: str):
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*(null|"(?:[^"\\]|\\.)*")',
        raw,
        flags=re.DOTALL,
    )
    if not match:
        return None
    value = match.group(1)
    if value == "null":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip('"')


def _extract_json_string_list(raw: str, key: str):
    if f'"{key}"' not in raw:
        return []
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]',
        raw,
        flags=re.DOTALL,
    )
    segment = match.group(1) if match else raw.split(f'"{key}"', 1)[-1][:300]
    values = re.findall(r'"([^"\n]+)"', segment)
    return [v for v in values if v != key]


def _salvage_partial_json(raw: str) -> Optional[dict]:
    if not raw.startswith("{"):
        return None

    parsed = {
        "storey_name": (
            _extract_json_scalar(raw, "storey_name")
            or _extract_json_scalar(raw, "storey")
            or _extract_json_scalar(raw, "ifc_storey")
        ),
        "ifc_class": (
            _extract_json_scalar(raw, "ifc_class")
            or _extract_json_scalar(raw, "ifc_type")
            or _extract_json_scalar(raw, "ifc_element_type")
            or _extract_json_scalar(raw, "element_type")
        ),
        "near_keywords": _extract_json_string_list(raw, "near_keywords"),
        "relations": [],
        "space_name": _extract_json_scalar(raw, "space_name"),
        "target_name_keyword": _extract_json_scalar(raw, "target_name_keyword"),
        "neighbor_type": _extract_json_scalar(raw, "neighbor_type"),
        "spatial_relations": [],
    }

    predicates = re.findall(r'"(?:predicate|relation)"\s*:\s*"([^"]+)"', raw, flags=re.DOTALL)
    object_types = re.findall(
        r'"(?:object_type|related_type|neighbor_type)"\s*:\s*"([^"]+)"',
        raw,
        flags=re.DOTALL,
    )
    directions = re.findall(
        r'"(?:direction|relative_position)"\s*:\s*"([^"]+)"',
        raw,
        flags=re.DOTALL,
    )
    n_rel = max(len(predicates), len(object_types), len(directions))
    for idx in range(n_rel):
        rel = {}
        if idx < len(predicates):
            rel["predicate"] = predicates[idx]
        if idx < len(object_types):
            rel["object_type"] = object_types[idx]
        if idx < len(directions):
            rel["direction"] = directions[idx]
        if rel:
            parsed["spatial_relations"].append(rel)

    if parsed["storey_name"] is None and parsed["ifc_class"] is None and not parsed["spatial_relations"]:
        return None
    return parsed


def _normalize(parsed: dict) -> dict:
    alias_map = {
        "storey": "storey_name",
        "ifc_storey": "storey_name",
        "ifc_type": "ifc_class",
        "ifc_element_type": "ifc_class",
        "element_type": "ifc_class",
        "spatial_relationships": "spatial_relations",
        "spatial_rel": "spatial_relations",
    }
    for old_key, new_key in alias_map.items():
        if old_key in parsed and new_key not in parsed:
            parsed[new_key] = parsed.get(old_key)

    sr = parsed.get("spatial_relations") or []
    if not sr:
        rel = parsed.get("relations")
        if isinstance(rel, list) and rel:
            sr = [r for r in rel if isinstance(r, dict) and "predicate" in r]
            if sr:
                parsed["spatial_relations"] = sr
    normalized_sr = []
    for triplet in parsed.get("spatial_relations", []) or []:
        if not isinstance(triplet, dict):
            continue
        if "relation" in triplet and "predicate" not in triplet:
            triplet["predicate"] = triplet.get("relation")
        if "related_type" in triplet and "object_type" not in triplet:
            triplet["object_type"] = triplet.get("related_type")
        if "neighbor_type" in triplet and "object_type" not in triplet:
            triplet["object_type"] = triplet.get("neighbor_type")
        if "relative_position" in triplet and "direction" not in triplet:
            triplet["direction"] = triplet.get("relative_position")
        if "predicate" in triplet and isinstance(triplet["predicate"], str):
            triplet["predicate"] = triplet["predicate"].upper()
        normalized_sr.append(triplet)
    parsed["spatial_relations"] = normalized_sr

    parsed.setdefault("storey_name", None)
    parsed.setdefault("ifc_class", None)
    parsed.setdefault("near_keywords", [])
    parsed.setdefault("relations", [])
    parsed.setdefault("space_name", None)
    parsed.setdefault("target_name_keyword", None)
    parsed.setdefault("neighbor_type", None)
    return parsed


def _encode_image(path: str) -> Optional[dict]:
    try:
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        print(f"    [WARN] Cannot read image {path}: {exc}")
        return None
    ext = Path(path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    return {"inline_data": {"mime_type": mime, "data": data}}


def _build_user_text_lora5(case: dict) -> str:
    parts = []
    ctx = case.get("inputs", {}).get("project_context", {})
    parts.append(f"[4D Task Status] {ctx.get('4d_task_status') or 'N/A'}")
    parts.append(f"[Project Phase] {ctx.get('project_phase') or 'N/A'}")
    location = ctx.get("location", "")
    if not location:
        location = (case.get("labels", {}).get("constraints", {}) or {}).get("storey_name", "")
    parts.append(f"[Location] {location or 'N/A'}")
    chat = case.get("inputs", {}).get("chat_history", [])
    if chat:
        parts.append(f"[Chat Log]\n{' '.join(msg.get('text', '') for msg in chat)}")
    else:
        query = case.get("query_text", "")
        if query:
            parts.append(f"[Chat Log]\n{query}")
    return "\n".join(parts)


def _build_user_text_lora2(case: dict) -> str:
    parts = []
    ctx = case.get("inputs", {}).get("project_context", {})
    if ctx.get("4d_task_status"):
        parts.append(f"[4D Task Status] {ctx.get('4d_task_status')}")
    if ctx.get("project_phase"):
        parts.append(f"[Project Phase] {ctx.get('project_phase')}")
    chat = case.get("inputs", {}).get("chat_history", [])
    if chat:
        chat_block = "\n".join(f"  {msg['role']}: {msg['text']}" for msg in chat)
        parts.append(f"[Chat Log]\n{chat_block}")
    if case.get("query_text"):
        parts.append(f"\n[Query] {case['query_text']}")
    parts.append("\nExtract the search constraints as JSON.")
    return "\n".join(parts)


def _build_parts(case: dict, modality: str, prompt_style: str) -> list:
    parts = []
    inputs = case.get("inputs", {})

    if modality not in ("FP", "MA"):
        for img_rel in inputs.get("images", []):
            abs_path = _resolve_image_path(img_rel)
            if abs_path:
                encoded = _encode_image(abs_path)
                if encoded:
                    parts.append(encoded)

    if modality not in ("SITE", "MA"):
        fp_rel = inputs.get("floorplan_patch")
        if fp_rel:
            abs_path = _resolve_image_path(fp_rel)
            if abs_path:
                encoded = _encode_image(abs_path)
                if encoded:
                    parts.append(encoded)

    user_text = _build_user_text_lora2(case) if prompt_style == "lora2" else _build_user_text_lora5(case)
    parts.append({"text": user_text})
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "gemini_unified__unified.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--modality", default="MC", choices=["MC", "FP", "SITE", "MA"])
    parser.add_argument("--prompt-style", default="lora5", choices=["lora5", "lora2"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("google-generativeai is required for Gemini unified eval.") from exc

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY before running Gemini unified eval.")
    genai.configure(api_key=api_key)

    system_prompt = SYSTEM_PROMPT_LORA2 if args.prompt_style == "lora2" else SYSTEM_PROMPT_LORA5

    with args.cases.open("r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        cases = cases[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    ok = 0
    total_latency = 0.0

    print("=" * 60)
    print("Gemini Unified Evaluation")
    print("=" * 60)
    print(f"  Cases:      {len(cases)}")
    print(f"  Modality:   {args.modality}")
    print(f"  Prompt:     {args.prompt_style}")
    print(f"  Output:     {args.output}")

    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"case_{idx}")
        parts = _build_parts(case, args.modality, args.prompt_style)
        parts = list(parts)
        parts.append(
            {
                "text": (
                    "\nReturn exactly one valid JSON object only. "
                    "Do not include markdown fences or explanatory text.\n\n"
                    f"{SCHEMA_HINT}"
                )
            }
        )
        t0 = time.perf_counter()
        try:
            gm = genai.GenerativeModel(
                model_name=args.model,
                system_instruction=(
                    f"{system_prompt}\n\n"
                    "Return exactly one JSON object and no extra prose. "
                    "If uncertain, use null or an empty list, but still return valid JSON.\n\n"
                    f"{SCHEMA_HINT}"
                ),
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            response = gm.generate_content([{"role": "user", "parts": parts}])
            raw_output = (response.text or "").strip()
            parsed = _parse_json(raw_output)
            if parsed:
                parsed = _normalize(parsed)
                status = "OK"
                ok += 1
            else:
                parsed = {}
                status = "PARSE_FAIL"
        except Exception as exc:  # pragma: no cover
            raw_output = f"ERROR: {exc}"
            parsed = {}
            status = "ERROR"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        total_latency += latency_ms
        results.append(
            {
                "case_id": case_id,
                "condition": args.modality,
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
                "raw_output": raw_output[:1000],
                "latency_ms": latency_ms,
                "status": status,
            }
        )
        print(
            f"  [{idx:>3}/{len(cases)}] {case_id} {status:10s} "
            f"{latency_ms:>7.0f}ms class={parsed.get('ifc_class', 'null')}"
        )
        if status == "PARSE_FAIL":
            print(f"       Raw: {raw_output[:240]}")
        if idx < len(cases) and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    with args.output.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    parse_rate = ok / len(cases) if cases else 0.0
    avg_latency = total_latency / len(cases) if cases else 0.0
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Parse rate:  {ok}/{len(cases)} ({parse_rate:.1%})")
    print(f"  Avg latency: {avg_latency:.0f} ms/case")
    print(f"  Output:      {args.output}")


if __name__ == "__main__":
    main()
