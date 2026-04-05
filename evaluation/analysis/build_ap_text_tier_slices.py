#!/usr/bin/env python3
"""Build text-tier AP held-out eval variants (T1/T2/T3) for robustness ablation."""

from __future__ import annotations

import argparse
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
DEFAULT_OUT_DIR = DATA_ROOT / "text_tier_slices"
TIER_ORDER = ("T1", "T2", "T3")


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


def _extract_assistant_label(row: dict) -> dict:
    for msg in row.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return json.loads(content)
    raise ValueError(f"Assistant label missing in row: {row.get('id')}")


def _replace_user_text(row: dict, new_text: str) -> dict:
    out = json.loads(json.dumps(row))
    for msg in out.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = new_text
                    return out
    raise ValueError(f"User text block missing in row: {row.get('id')}")


def build_tier_rows(rows: List[dict], tier: str, asm) -> List[dict]:
    out_rows: List[dict] = []
    for row in rows:
        new_row = json.loads(json.dumps(row))
        new_row["text_tier"] = tier
        if tier != "T1":
            label = _extract_assistant_label(row)
            user_text = asm.build_controlled_text_chat(
                predicate=row.get("predicate") or "",
                subject_type=label.get("ifc_class") or "",
                storey_name=str(label.get("storey_name") or ""),
                variant_key=f"{row.get('base_case_id') or row.get('id')}_{tier}",
                text_tier=tier,
            )
            new_row = _replace_user_text(new_row, user_text)
        out_rows.append(new_row)
    return out_rows


def write_slices(source_path: Path, out_dir: Path, asm) -> Dict[str, Path]:
    rows = _load_jsonl(source_path)
    stem = source_path.stem
    written: Dict[str, Path] = {}
    for tier in TIER_ORDER:
        out_path = out_dir / f"{stem}_{tier}.jsonl"
        _dump_jsonl(out_path, build_tier_rows(rows, tier, asm))
        written[tier] = out_path
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
    print("Built AP text-tier slices:")
    for family, mapping in outputs.items():
        for tier, path in mapping.items():
            print(f"  [{family}:{tier}] {path}")


if __name__ == "__main__":
    main()
