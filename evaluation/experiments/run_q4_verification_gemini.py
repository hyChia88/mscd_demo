#!/usr/bin/env python3
"""
Run the Q4 verification quick check with Gemini.

This script consumes a prepared Q4 bundle directory and evaluates two
floorplan+text-only baselines on the sampled cases:

1. chat + candidate descriptions
2. chat + floorplan + candidate descriptions

Outputs:
- per-case predictions + raw model outputs
- aggregate summary.json
- aggregate summary.csv
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import google.generativeai as genai


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BUNDLE_DIR = (
    PROJECT_ROOT
    / "output"
    / "q4_verification"
    / "traces_20260324_191220_v2_lora_p0_union_p1_sample10_seed42_top5_gtin"
)
DEFAULT_MODEL = "gemini-2.5-flash"


def load_bundle(case_dir: Path) -> Dict[str, Any]:
    with open(case_dir / "bundle.json", "r", encoding="utf-8") as f:
        return json.load(f)


def encode_image(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    mime = "image/png"
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {"inline_data": {"mime_type": mime, "data": data}}


def parse_prediction(text: str) -> Dict[str, Any]:
    text = text.strip()
    candidates = []
    try:
        candidates.append(json.loads(text))
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            candidates.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass

    match = re.search(r'(\{[^{}]*"best_candidate_id"[^{}]*\})', text, re.DOTALL)
    if match:
        try:
            candidates.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass

    for obj in candidates:
        if isinstance(obj, dict):
            best = obj.get("best_candidate_id")
            if isinstance(best, str):
                return {
                    "best_candidate_id": best.strip(),
                    "reason": obj.get("reason"),
                    "parse_status": "OK",
                }

    fallback = re.search(r"\b(C[1-9]\d*)\b", text)
    if fallback:
        return {
            "best_candidate_id": fallback.group(1),
            "reason": None,
            "parse_status": "FALLBACK_REGEX",
        }

    return {
        "best_candidate_id": None,
        "reason": None,
        "parse_status": "PARSE_FAIL",
    }


def call_gemini(
    model: Any,
    prompt_text: str,
    floorplan_path: Optional[Path] = None,
) -> str:
    parts: List[Dict[str, Any]] = []
    if floorplan_path is not None:
        image_part = encode_image(floorplan_path)
        if image_part is not None:
            parts.append(image_part)
    parts.append({"text": prompt_text})
    response = model.generate_content([{"role": "user", "parts": parts}])
    return (response.text or "").strip()


def run_mode(
    model: Any,
    bundle: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    if mode not in ("chat_desc", "chat_floorplan_desc"):
        raise ValueError(f"Unsupported mode: {mode}")

    prompt_text = bundle["prompts"][mode]
    floorplan_path = None
    if mode == "chat_floorplan_desc":
        raw_path = bundle.get("evidence", {}).get("floorplan_path")
        if raw_path:
            floorplan_path = Path(raw_path)

    raw_output = call_gemini(model, prompt_text, floorplan_path=floorplan_path)
    parsed = parse_prediction(raw_output)
    gt_candidate_id = bundle.get("ground_truth_candidate_id")
    predicted = parsed.get("best_candidate_id")
    is_correct = bool(predicted and gt_candidate_id and predicted == gt_candidate_id)

    return {
        "mode": mode,
        "best_candidate_id": predicted,
        "ground_truth_candidate_id": gt_candidate_id,
        "is_correct": is_correct,
        "parse_status": parsed.get("parse_status"),
        "reason": parsed.get("reason"),
        "raw_output": raw_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GOOGLE_API_KEY / GEMINI_API_KEY")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=args.model,
        system_instruction=(
            "You are a careful BIM verification assistant. "
            "Return strict JSON when asked."
        ),
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )

    case_dirs = sorted(
        p for p in args.bundle_dir.iterdir()
        if p.is_dir() and (p / "bundle.json").exists()
    )
    if not case_dirs:
        raise SystemExit(f"No case bundles found in {args.bundle_dir}")

    results = []
    for idx, case_dir in enumerate(case_dirs, start=1):
        bundle = load_bundle(case_dir)
        case_id = bundle["case_id"]
        gt_candidate_id = bundle.get("ground_truth_candidate_id")

        chat_desc = run_mode(model, bundle, "chat_desc")
        time.sleep(args.delay_seconds)
        chat_floorplan_desc = run_mode(model, bundle, "chat_floorplan_desc")
        if idx < len(case_dirs):
            time.sleep(args.delay_seconds)

        case_result = {
            "case_id": case_id,
            "ground_truth_candidate_id": gt_candidate_id,
            "chat_desc": chat_desc,
            "chat_floorplan_desc": chat_floorplan_desc,
        }
        results.append(case_result)
        print(
            f"[{idx:02d}/{len(case_dirs)}] {case_id}  "
            f"chat+desc={chat_desc['best_candidate_id']} ({chat_desc['is_correct']})  "
            f"chat+floorplan+desc={chat_floorplan_desc['best_candidate_id']} ({chat_floorplan_desc['is_correct']})"
        )

    summary = {
        "bundle_dir": str(args.bundle_dir),
        "model": args.model,
        "n_cases": len(results),
        "chat_desc_successes": sum(r["chat_desc"]["is_correct"] for r in results),
        "chat_floorplan_desc_successes": sum(
            r["chat_floorplan_desc"]["is_correct"] for r in results
        ),
        "chat_desc_parse_failures": sum(
            r["chat_desc"]["parse_status"] == "PARSE_FAIL" for r in results
        ),
        "chat_floorplan_desc_parse_failures": sum(
            r["chat_floorplan_desc"]["parse_status"] == "PARSE_FAIL" for r in results
        ),
        "results": results,
    }
    summary["chat_desc_success_rate"] = round(
        summary["chat_desc_successes"] / summary["n_cases"], 4
    )
    summary["chat_floorplan_desc_success_rate"] = round(
        summary["chat_floorplan_desc_successes"] / summary["n_cases"], 4
    )

    out_dir = args.bundle_dir / f"gemini_q4_{args.model.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["model", args.model])
        writer.writerow(["n_cases", summary["n_cases"]])
        writer.writerow(["chat_desc_successes", summary["chat_desc_successes"]])
        writer.writerow(["chat_desc_success_rate", summary["chat_desc_success_rate"]])
        writer.writerow(
            ["chat_floorplan_desc_successes", summary["chat_floorplan_desc_successes"]]
        )
        writer.writerow(
            [
                "chat_floorplan_desc_success_rate",
                summary["chat_floorplan_desc_success_rate"],
            ]
        )
        writer.writerow(["chat_desc_parse_failures", summary["chat_desc_parse_failures"]])
        writer.writerow(
            [
                "chat_floorplan_desc_parse_failures",
                summary["chat_floorplan_desc_parse_failures"],
            ]
        )

    print(f"Results written to: {out_dir}")
    print(
        f"chat+desc: {summary['chat_desc_successes']}/{summary['n_cases']} "
        f"({summary['chat_desc_success_rate']:.1%})"
    )
    print(
        f"chat+floorplan+desc: {summary['chat_floorplan_desc_successes']}/{summary['n_cases']} "
        f"({summary['chat_floorplan_desc_success_rate']:.1%})"
    )


if __name__ == "__main__":
    main()
