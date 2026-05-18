#!/usr/bin/env python3
"""Fast audit of which Qwen2.5-VL modules were targeted by saved LoRA adapters."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "plots" / "final"
DEFAULT_DATA_DIR = DEFAULT_OUT_DIR / "data"
ADAPTER_ROOT = PROJECT_ROOT / "models" / "lora6_v2_ap_20260331"

MODEL_SPECS = [
    ("g7_position_context", "G7"),
    ("g8_posctx_dim", "G8"),
    ("g9_opencv_cluster", "G9"),
]
GROUPS = [
    ("visual_attn", "Visual attention", "#0EA5E9"),
    ("visual_mlp", "Visual MLP", "#06B6D4"),
    ("language_attn", "Language attention", "#7C3AED"),
    ("language_mlp", "Language MLP", "#A855F7"),
    ("visual_merger", "visual.merger", "#EF4444"),
    ("patch_embed", "patch_embed", "#F59E0B"),
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _group_for_key(key: str) -> str | None:
    if ".visual.blocks." in key and ".attn." in key:
        return "visual_attn"
    if ".visual.blocks." in key and ".mlp." in key:
        return "visual_mlp"
    if ".language_model.layers." in key and ".self_attn." in key:
        return "language_attn"
    if ".language_model.layers." in key and ".mlp." in key:
        return "language_mlp"
    if ".visual.merger." in key:
        return "visual_merger"
    if ".visual.patch_embed." in key:
        return "patch_embed"
    return None


def _write_svg(rows: list[dict], out_path: Path) -> None:
    width = 1180
    height = 680
    margin_left = 180
    margin_right = 50
    margin_top = 110
    margin_bottom = 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_count = max(int(row["count"]) for row in rows) or 1
    model_y = {"G7": 0, "G8": 1, "G9": 2}
    row_gap = 86

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="38" font-size="28" font-weight="700" fill="#0f172a">LoRA target coverage audit</text>',
        f'<text x="{margin_left}" y="66" font-size="16" fill="#475569">What this proves: the saved G7/G8/G9 adapters target visual and language transformer blocks, but not visual.merger.</text>',
        f'<text x="{margin_left}" y="88" font-size="16" fill="#475569">So a thesis/Q&amp;A claim about “visual.merger changing” would currently be unsupported by the actual training setup.</text>',
    ]

    for tick in range(0, max_count + 1, 16):
        xx = margin_left + (tick / max_count) * plot_w
        svg.append(f'<line x1="{xx:.1f}" y1="{margin_top}" x2="{xx:.1f}" y2="{margin_top + plot_h}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{xx:.1f}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-size="12" fill="#64748b">{tick}</text>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#0f172a" stroke-width="1.2"/>')

    for model, y_idx in model_y.items():
        block_y = margin_top + y_idx * (2 * row_gap) + 6
        svg.append(f'<text x="{margin_left - 120}" y="{block_y + 28}" font-size="18" font-weight="700" fill="#0f172a">{model}</text>')
        model_rows = [row for row in rows if row["model"] == model]
        for i, (group_key, group_label, color) in enumerate(GROUPS):
            row = next(r for r in model_rows if r["group"] == group_key)
            yy = block_y + i * 12
            ww = (int(row["count"]) / max_count) * plot_w
            svg.append(f'<text x="{margin_left - 10}" y="{yy + 9}" text-anchor="end" font-size="12" fill="#334155">{_svg_escape(group_label)}</text>')
            svg.append(f'<rect x="{margin_left}" y="{yy}" width="{ww:.1f}" height="10" fill="{color}" opacity="0.92"/>')
            svg.append(f'<text x="{margin_left + ww + 6:.1f}" y="{yy + 9}" font-size="11" fill="{color}">{row["count"]}</text>')

    svg.append(f'<text x="{margin_left}" y="{height - 18}" font-size="13" fill="#991b1b" font-weight="700">Measured fact: visual.merger target count = 0 in all three saved adapters.</text>')
    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    rows = []
    for model_key, model_label in MODEL_SPECS:
        adapter_path = ADAPTER_ROOT / model_key / "best" / "adapter_model.safetensors"
        with adapter_path.open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len).decode("utf-8"))
        counts = defaultdict(int)
        for key, value in header.items():
            if not (isinstance(value, dict) and "dtype" in value):
                continue
            if not key.endswith(".lora_A.weight"):
                continue
            group = _group_for_key(key)
            if group:
                counts[group] += 1
        for group_key, group_label, _color in GROUPS:
            rows.append(
                {
                    "model": model_label,
                    "group": group_key,
                    "group_label": group_label,
                    "count": counts[group_key],
                    "adapter_path": str(adapter_path.relative_to(PROJECT_ROOT.parent)),
                }
            )

    csv_path = DEFAULT_DATA_DIR / "backup_lora_target_coverage.csv"
    _write_csv(csv_path, rows, ["model", "group", "group_label", "count", "adapter_path"])
    svg_path = DEFAULT_OUT_DIR / "backup_lora_target_coverage.svg"
    _write_svg(rows, svg_path)
    print(f"[coverage] wrote csv: {csv_path}")
    print(f"[coverage] wrote svg: {svg_path}")


if __name__ == "__main__":
    main()
