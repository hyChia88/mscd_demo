#!/usr/bin/env python3
"""Run Gemini on the assembled AP held-out eval set.

This is the Track A Gemini baseline for LoRA6-v2. It consumes the same
assembled AP eval records used by LoRA6, strips the assistant GT, and writes
predictions in the same schema as ``training/eval.py``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.track_registry import EXPERIMENT_ROOT

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


DEFAULT_CASES = (
    PROJECT_ROOT.parent
    / "data_curation"
    / "datasets"
    / "synth_v0.5_ap"
    / "train"
    / "lora6_v2_ap_eval_canonical_m.jsonl"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "gemini_ap__ap_eval.jsonl"
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


def _parse_json(text: str) -> Optional[dict]:
    raw = (text or "").strip()
    if not raw:
        return None

    candidates = [raw]
    fence_matches = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates.extend(m.strip() for m in fence_matches if m.strip())

    if raw.startswith("{"):
        repaired = raw
        bracket_delta = repaired.count("[") - repaired.count("]")
        brace_delta = repaired.count("{") - repaired.count("}")
        if bracket_delta > 0:
            repaired += "]" * bracket_delta
        if brace_delta > 0:
            repaired += "}" * brace_delta
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        candidates.append(repaired)

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = raw.find(open_ch)
        end = raw.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidates.append(raw[start:end + 1].strip())

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
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


def _extract_json_string_list(raw: str, key: str) -> List[str]:
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


def _normalize_spatial_relations(parsed: dict) -> dict:
    alias_map = {
        "storey": "storey_name",
        "ifc_storey": "storey_name",
        "storeyName": "storey_name",
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


def _build_parts(case: dict) -> tuple[str, List[dict]]:
    system_prompt = ""
    user_parts: List[dict] = []

    for msg in case.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant":
            continue
        if role == "system" and isinstance(content, str):
            system_prompt = content
            continue
        if role != "user" or not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                user_parts.append({"text": part.get("text", "")})
            elif part_type == "image":
                image_ref = str(part.get("image", ""))
                if image_ref.startswith("file://"):
                    image_ref = image_ref.replace("file://", "", 1)
                encoded = _encode_image(image_ref)
                if encoded:
                    user_parts.append(encoded)
    return system_prompt, user_parts


def run(
    *,
    cases_file: Path,
    output_path: Path,
    model_name: str,
    limit: int,
    sleep_seconds: float,
) -> None:
    if load_dotenv is not None:
        load_dotenv()

    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "google-generativeai is required for Gemini baseline. "
            "Install it in the mscd_demo env before running this script."
        ) from exc

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY before running Gemini AP eval.")

    genai.configure(api_key=api_key)

    cases: List[dict] = []
    with cases_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    if limit > 0:
        cases = cases[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    ok = 0
    total_latency = 0.0

    print("=" * 60)
    print("Gemini AP Held-out Evaluation")
    print("=" * 60)
    print(f"  Cases:   {len(cases)}")
    print(f"  Model:   {model_name}")
    print(f"  Output:  {output_path}")

    for idx, case in enumerate(cases, 1):
        case_id = case.get("id") or case.get("base_case_id") or case.get("case_id") or f"case_{idx}"
        system_prompt, user_parts = _build_parts(case)
        user_parts = list(user_parts)
        user_parts.append(
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
                model_name=model_name,
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
            response = gm.generate_content(
                [{"role": "user", "parts": user_parts}]
            )
            raw_output = (response.text or "").strip()
            parsed = _parse_json(raw_output)
            if parsed:
                parsed = _normalize_spatial_relations(parsed)
                status = "OK"
                ok += 1
            else:
                parsed = {}
                status = "PARSE_FAIL"
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raw_output = f"ERROR: {exc}"
            parsed = {}
            status = "ERROR"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        total_latency += latency_ms

        results.append(
            {
                "case_id": case_id,
                "condition": "AP_EVAL",
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
            f"  [{idx:>3}/{len(cases)}] {case_id} "
            f"{status:10s} {latency_ms:>7.0f}ms "
            f"class={parsed.get('ifc_class', 'null')} "
            f"storey={parsed.get('storey_name', 'null')}"
        )
        if status == "PARSE_FAIL":
            print(f"       Raw: {raw_output[:240]}")
        if idx < len(cases) and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    parse_rate = ok / len(cases) if cases else 0.0
    avg_latency = total_latency / len(cases) if cases else 0.0
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Parse rate:  {ok}/{len(cases)} ({parse_rate:.1%})")
    print(f"  Avg latency: {avg_latency:.0f} ms/case")
    print(f"  Output:      {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()

    run(
        cases_file=args.cases,
        output_path=args.output,
        model_name=args.model,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
