#!/usr/bin/env python3
"""Build modality-sliced AP held-out eval JSONL files for Track A ablations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PROJECT_ROOT.parent
DATA_ROOT = REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap" / "train"
ASSEMBLE_PATH = REPO_ROOT / "data_curation" / "scripts" / "synth" / "6_assemble_lora6.py"

DEFAULT_STANDARD = DATA_ROOT / "lora6_v2_ap_eval_canonical_m.jsonl"
DEFAULT_G7 = DATA_ROOT / "lora6_v2_ap_eval_canonical_m_g7.jsonl"
DEFAULT_OUT_DIR = DATA_ROOT / "modality_slices"

SLICE_ORDER = ("MC", "MC4D", "FP", "SITE", "MA")


def _load_assemble_module():
    assemble_dir = str(ASSEMBLE_PATH.parent)
    if assemble_dir not in sys.path:
        sys.path.insert(0, assemble_dir)
    spec = importlib.util.spec_from_file_location("assemble_lora6", ASSEMBLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load assemble module from {ASSEMBLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _is_floorplan(image_uri: str) -> bool:
    text = image_uri.lower()
    return "floorplan" in text or "/floorplans/" in text or "_floorplan" in text


def _slice_user_content(content: List[dict], slice_key: str) -> List[dict]:
    result: List[dict] = []
    for part in content:
        if not isinstance(part, dict):
            result.append(part)
            continue
        if part.get("type") != "image":
            result.append(part)
            continue
        image_uri = str(part.get("image") or "")
        is_fp = _is_floorplan(image_uri)
        if slice_key in {"MC", "MC4D"}:
            result.append(part)
        elif slice_key == "FP" and is_fp:
            result.append(part)
        elif slice_key == "SITE" and not is_fp:
            result.append(part)
        elif slice_key == "MA":
            continue
    return result


def _extract_assistant_label(row: dict) -> dict:
    for msg in row.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return json.loads(content)
    raise ValueError(f"Assistant label missing in row: {row.get('id')}")


def _extract_user_text(row: dict) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return str(part.get("text") or "")
    raise ValueError(f"User text block missing in row: {row.get('id')}")


def _replace_user_text(row: dict, new_text: str) -> None:
    for msg in row.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = new_text
                    return
    raise ValueError(f"User text block missing in row: {row.get('id')}")


def _with_4d_metadata(row: dict, asm) -> dict:
    out = json.loads(json.dumps(row))
    label = _extract_assistant_label(out)
    base_case_id = out.get("base_case_id") or out.get("id") or ""
    task_n = int(hashlib.md5(base_case_id.encode("utf-8")).hexdigest()[:4], 16) % 10000
    meta = asm.generate_4d_metadata(
        storey_name=str(label.get("storey_name") or ""),
        subject_type=str(label.get("ifc_class") or ""),
        task_n=task_n,
    )
    chat = _extract_user_text(out)
    _replace_user_text(out, f"{meta}\n[Chat Log]\n{chat}")
    return out


def build_slice_rows(rows: List[dict], slice_key: str, asm) -> List[dict]:
    sliced: List[dict] = []
    for row in rows:
        out = _with_4d_metadata(row, asm) if slice_key == "MC4D" else json.loads(json.dumps(row))
        out["modality_slice"] = slice_key
        messages = out.get("messages", [])
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = _slice_user_content(content, slice_key)
        sliced.append(out)
    return sliced


def write_slices(source_path: Path, out_dir: Path, asm) -> Dict[str, Path]:
    rows = _load_jsonl(source_path)
    written: Dict[str, Path] = {}
    stem = source_path.stem
    for slice_key in SLICE_ORDER:
        out_path = out_dir / f"{stem}_{slice_key}.jsonl"
        _dump_jsonl(out_path, build_slice_rows(rows, slice_key, asm))
        written[slice_key] = out_path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, default=DEFAULT_STANDARD)
    parser.add_argument("--g7", type=Path, default=DEFAULT_G7)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    asm = _load_assemble_module()
    outputs = {
        "standard": write_slices(args.standard, args.out_dir, asm),
        "g7": write_slices(args.g7, args.out_dir, asm),
    }
    print("Built AP modality slices:")
    for family, mapping in outputs.items():
        for slice_key, path in mapping.items():
            print(f"  [{family}:{slice_key}] {path}")


if __name__ == "__main__":
    main()
