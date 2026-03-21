#!/usr/bin/env python3
"""
Fair Gemini Zero-Shot Baseline for Constraint Extraction.

Uses the EXACT SAME input structure as LoRA_5 eval:
  - Same system prompt (constraints_extraction.yaml)
  - Same message format: system + [images + text] in one call
  - Same JSON output schema
  - Single API call (images + text together, NOT separate calls)

This isolates one variable: finetuning (LoRA_5) vs zero-shot (Gemini).

Usage:
    # Full run (70 cases, MC condition = floorplan + site + text)
    python eval/eval_gemini_baseline.py --cases eval/cases_v5_test.jsonl

    # Quick smoke test (5 cases)
    python eval/eval_gemini_baseline.py --cases eval/cases_v5_test.jsonl --limit 5

    # Modality ablation
    python eval/eval_gemini_baseline.py --cases eval/cases_v5_test.jsonl --modality FP
    python eval/eval_gemini_baseline.py --cases eval/cases_v5_test.jsonl --modality MA

    # Run all conditions for full comparison
    for m in MC FP SITE MA; do
        python eval/eval_gemini_baseline.py --cases eval/cases_v5_test.jsonl --modality $m
    done

After running, feed results into the same local pipeline:
    python script/run.py --profile v2_lora --precomputed logs/evaluation_output/gemini_baseline/eval_constraints_final_MC.jsonl ...
"""

import argparse
import base64
import json
import os
from dotenv import load_dotenv
load_dotenv()
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data_curation"
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_PATH = PROJECT_DIR / "prompts" / "constraints_extraction.yaml"

GEMINI_MODEL = "gemini-2.5-flash"


# ── Prompt + text builders (identical to eval_lora5.py) ──────────────────────

def _load_system_prompt() -> str:
    with open(PROMPT_PATH) as f:
        prompts = yaml.safe_load(f)
    return prompts["system"]


def _resolve_image_path(relative_path: str, case_id: str = "") -> Optional[str]:
    """Resolve relative image path → absolute, checking data_curation root."""
    if not relative_path:
        return None
    for root in [DATA_ROOT, PROJECT_DIR]:
        p = root / relative_path
        if p.exists():
            return str(p)
    return None


def _build_user_text(case: dict) -> str:
    """Mirrors eval_lora5.py _build_user_text() exactly."""
    parts = []
    ctx = case.get("inputs", {}).get("project_context", {})

    parts.append(f"[4D Task Status] {ctx.get('4d_task_status') or 'N/A'}")
    parts.append(f"[Project Phase] {ctx.get('project_phase') or 'N/A'}")

    location = ctx.get("location", "")
    if not location:
        location = (case.get("labels", {}).get("constraints", {})
                    .get("storey_name", ""))
    parts.append(f"[Location] {location or 'N/A'}")

    chat = case.get("inputs", {}).get("chat_history", [])
    if chat:
        chat_lines = [msg.get("text", "") for msg in chat]
        parts.append(f"[Chat Log]\n{' '.join(chat_lines)}")
    else:
        query = case.get("query_text", "")
        if query:
            parts.append(f"[Chat Log]\n{query}")

    return "\n".join(parts)


# ── Gemini content builder ───────────────────────────────────────────────────

def _encode_image_b64(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"    [WARN] Cannot read image {path}: {e}")
        return None


def _mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")


def _build_contents(case: dict, modality: str = "MC") -> list:
    """Build Gemini API parts — mirrors LoRA_5 _build_messages() structure.

    Order: [site_photos] + [floorplan] + [text]  (same as LoRA training)
    """
    parts = []
    inputs = case.get("inputs", {})
    case_id = case.get("case_id", "")

    # 1. Site photos (skip if FP-only or MA)
    if modality not in ("FP", "MA"):
        for img_rel in inputs.get("images", []):
            abs_path = _resolve_image_path(img_rel, case_id)
            if abs_path:
                b64 = _encode_image_b64(abs_path)
                if b64:
                    parts.append({"inline_data": {"mime_type": _mime(abs_path), "data": b64}})

    # 2. Floorplan (skip if SITE-only or MA)
    if modality not in ("SITE", "MA"):
        fp_rel = inputs.get("floorplan_patch")
        if fp_rel:
            abs_path = _resolve_image_path(fp_rel, case_id)
            if abs_path:
                b64 = _encode_image_b64(abs_path)
                if b64:
                    parts.append({"inline_data": {"mime_type": _mime(abs_path), "data": b64}})

    # 3. Text (always last)
    parts.append({"text": _build_user_text(case)})

    return [{"role": "user", "parts": parts}]


# ── JSON parsing (identical to eval_lora5.py) ────────────────────────────────

def _parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [
        r'```(?:json)?\s*(\{.*?\})\s*```',
        r'(\{.*"(?:spatial_relations|relations)".*\})',
        r'(\{[^{]*"storey_name"[^}]*\})',
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None


def _normalize_sr(parsed: dict) -> dict:
    sr = parsed.get("spatial_relations") or []
    if not sr:
        rel = parsed.get("relations")
        if isinstance(rel, list) and rel:
            sr = [r for r in rel if isinstance(r, dict) and "predicate" in r]
            if sr:
                parsed["spatial_relations"] = sr
    for t in (parsed.get("spatial_relations") or []):
        if "predicate" in t:
            t["predicate"] = t["predicate"].upper()
    return parsed


# ── Main ─────────────────────────────────────────────────────────────────────

def run(cases_file: str, output_dir: str, modality: str = "MC",
        limit: int = 0, model: str = GEMINI_MODEL):
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY env var")
        sys.exit(1)
    genai.configure(api_key=api_key)

    system_prompt = _load_system_prompt()

    cases = []
    with open(cases_file) as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    if limit > 0:
        cases = cases[:limit]
    print(f"Loaded {len(cases)} cases | modality={modality} | model={model}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"eval_constraints_final_{modality}.jsonl")

    gm = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=4096),
    )

    results = []
    ok = 0
    for i, case in enumerate(cases):
        cid = case["case_id"]
        contents = _build_contents(case, modality)

        try:
            resp = gm.generate_content(contents)
            raw = resp.text.strip()
        except Exception as e:
            print(f"  [{i+1:3d}/{len(cases)}] {cid}  ERROR: {e}")
            results.append({"case_id": cid, "status": "ERROR",
                            "constraints": {}, "raw_output": str(e)})
            time.sleep(1)
            continue

        parsed = _parse_json(raw)
        if parsed:
            parsed = _normalize_sr(parsed)
            status = "OK"
            ok += 1
        else:
            parsed = {}
            status = "PARSE_FAIL"

        sr_n = len(parsed.get("spatial_relations", []))
        pred = parsed.get("spatial_relations", [{}])[0].get("predicate", "-") if sr_n else "-"
        print(f"  [{i+1:3d}/{len(cases)}] {cid}  {status}  "
              f"storey={parsed.get('storey_name','?')}  "
              f"class={parsed.get('ifc_class','?')}  SR={sr_n}({pred})")

        results.append({"case_id": cid, "status": status,
                        "constraints": parsed, "raw_output": raw})

        # Rate limit (Gemini free tier ≈ 15 RPM for flash)
        if i < len(cases) - 1:
            time.sleep(0.5)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone: {ok}/{len(cases)} parsed OK → {out_path}")


def main():
    p = argparse.ArgumentParser(description="Fair Gemini zero-shot baseline")
    p.add_argument("--cases", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--modality", default="MC")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", default=GEMINI_MODEL)
    args = p.parse_args()

    out = args.output_dir or str(PROJECT_DIR / "logs" / "evaluation_output" / "gemini_baseline")
    run(args.cases, out, args.modality, args.limit, args.model)


if __name__ == "__main__":
    main()
