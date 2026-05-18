#!/usr/bin/env python3
"""Fast adapter-only proxy audit for saved Qwen2.5-VL LoRA runs.

This script avoids base-model loading and answers two narrow questions:

1. Which architecture groups carry most of the effective LoRA delta mass?
2. Were `visual.merger` or `visual.patch_embed` targeted at all?

It computes Delta W = (alpha / r) * (B @ A) for each LoRA module, aggregates
the Frobenius energy by group, and writes a simple SVG figure plus CSV.

This is intentionally a proxy, not the stronger base-normalized
||Delta W|| / ||W|| audit.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


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
GROUP_ORDER = ["visual_attn", "visual_mlp", "language_attn", "language_mlp"]
GROUP_LABELS = {
    "visual_attn": "Visual attn",
    "visual_mlp": "Visual MLP",
    "language_attn": "Language attn",
    "language_mlp": "Language MLP",
}
GROUP_COLORS = {
    "visual_attn": "#0EA5E9",
    "visual_mlp": "#06B6D4",
    "language_attn": "#7C3AED",
    "language_mlp": "#A855F7",
}
MODEL_COLORS = {"G7": "#7C3AED", "G8": "#2563EB", "G9": "#0F766E"}
_DTYPE_MAP = {"F16": np.float16, "F32": np.float32, "BF16": "BF16"}


class SafeTensorShard:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len).decode("utf-8"))
        self._data_start = 8 + header_len
        self._tensors = {
            key: value
            for key, value in header.items()
            if isinstance(value, dict) and "dtype" in value and "shape" in value
        }

    def keys(self) -> list[str]:
        return list(self._tensors.keys())

    def get_tensor(self, key: str) -> np.ndarray:
        meta = self._tensors[key]
        dtype_name = str(meta["dtype"])
        shape = tuple(int(x) for x in meta["shape"])
        start, end = meta["data_offsets"]
        byte_start = self._data_start + int(start)
        byte_count = int(end) - int(start)
        if dtype_name == "BF16":
            count = byte_count // 2
            raw = np.memmap(self.path, mode="r", dtype=np.uint16, offset=byte_start, shape=(count,))
            widened = (raw.astype(np.uint32) << 16).view(np.float32)
            return np.asarray(widened).reshape(shape)
        dtype = _DTYPE_MAP[dtype_name]
        count = byte_count // np.dtype(dtype).itemsize
        arr = np.memmap(self.path, mode="r", dtype=dtype, offset=byte_start, shape=(count,))
        return np.asarray(arr).reshape(shape)


def _group_for_key(key: str) -> str | None:
    if ".visual.blocks." in key and ".attn." in key:
        return "visual_attn"
    if ".visual.blocks." in key and ".mlp." in key:
        return "visual_mlp"
    if ".language_model.layers." in key and ".self_attn." in key:
        return "language_attn"
    if ".language_model.layers." in key and ".mlp." in key:
        return "language_mlp"
    return None


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_svg(summary_rows: list[dict], out_path: Path) -> None:
    width = 1280
    height = 720
    margin = 56
    mid = width // 2
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin}" y="36" font-size="27" font-weight="700" fill="#0f172a">LoRA adapter proxy audit: delta mass and target coverage</text>',
        f'<text x="{margin}" y="62" font-size="15" fill="#475569">Fast appendix figure. Shows where the saved adapters place effective delta mass and verifies that visual.merger / patch_embed were not targeted.</text>',
    ]

    # Left: stacked bars
    left_x0 = margin
    left_y0 = 110
    left_w = mid - margin - 24
    left_h = 500
    svg.append(f'<text x="{left_x0}" y="{left_y0 - 18}" font-size="18" font-weight="700" fill="#0f172a">A. Effective delta mass share by group</text>')
    for tick in [0, 25, 50, 75, 100]:
        yy = left_y0 + left_h - (tick / 100.0) * left_h
        svg.append(f'<line x1="{left_x0}" y1="{yy:.1f}" x2="{left_x0 + left_w}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{left_x0 - 8}" y="{yy + 5:.1f}" text-anchor="end" font-size="12" fill="#64748b">{tick}</text>')
    svg.append(f'<line x1="{left_x0}" y1="{left_y0}" x2="{left_x0}" y2="{left_y0 + left_h}" stroke="#0f172a" stroke-width="1.2"/>')
    svg.append(f'<line x1="{left_x0}" y1="{left_y0 + left_h}" x2="{left_x0 + left_w}" y2="{left_y0 + left_h}" stroke="#0f172a" stroke-width="1.2"/>')

    bar_w = 86
    gap = 74
    start_x = left_x0 + 80
    for idx, row in enumerate(summary_rows):
        xx = start_x + idx * (bar_w + gap)
        yy_bottom = left_y0 + left_h
        for group in GROUP_ORDER:
            share = float(row[f"{group}_share_pct"])
            hh = (share / 100.0) * left_h
            yy = yy_bottom - hh
            svg.append(f'<rect x="{xx}" y="{yy:.1f}" width="{bar_w}" height="{hh:.1f}" fill="{GROUP_COLORS[group]}" stroke="#ffffff" stroke-width="1"/>')
            if share >= 8:
                svg.append(f'<text x="{xx + bar_w/2:.1f}" y="{yy + hh/2 + 5:.1f}" text-anchor="middle" font-size="11" fill="#ffffff" font-weight="700">{share:.0f}%</text>')
            yy_bottom = yy
        svg.append(f'<text x="{xx + bar_w/2:.1f}" y="{left_y0 + left_h + 26}" text-anchor="middle" font-size="15" font-weight="700" fill="{MODEL_COLORS[row["model"]]}">{row["model"]}</text>')

    # legend
    leg_x = left_x0 + 12
    leg_y = left_y0 + left_h + 54
    for idx, group in enumerate(GROUP_ORDER):
        x0 = leg_x + idx * 132
        svg.append(f'<rect x="{x0}" y="{leg_y}" width="16" height="16" fill="{GROUP_COLORS[group]}"/>')
        svg.append(f'<text x="{x0 + 24}" y="{leg_y + 13}" font-size="12" fill="#0f172a">{GROUP_LABELS[group]}</text>')

    # Right: coverage bars
    right_x0 = mid + 24
    right_y0 = 110
    right_w = width - right_x0 - margin
    right_h = 500
    svg.append(f'<text x="{right_x0}" y="{right_y0 - 18}" font-size="18" font-weight="700" fill="#0f172a">B. LoRA target coverage (module count)</text>')
    cats = [
        ("visual_attn_count", "Visual attn"),
        ("visual_mlp_count", "Visual MLP"),
        ("language_attn_count", "Language attn"),
        ("language_mlp_count", "Language MLP"),
        ("visual_merger_count", "visual.merger"),
        ("visual_patch_embed_count", "patch_embed"),
    ]
    max_count = max(max(int(row[key]) for row in summary_rows) for key, _ in cats)
    for tick in range(0, max_count + 1, 16):
        xx = right_x0 + (tick / max_count) * right_w if max_count else right_x0
        svg.append(f'<line x1="{xx:.1f}" y1="{right_y0}" x2="{xx:.1f}" y2="{right_y0 + right_h}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{xx:.1f}" y="{right_y0 + right_h + 22}" text-anchor="middle" font-size="12" fill="#64748b">{tick}</text>')
    row_gap = 68
    for idx, (key, label) in enumerate(cats):
        yy = right_y0 + 24 + idx * row_gap
        svg.append(f'<text x="{right_x0}" y="{yy + 13}" font-size="13" fill="#0f172a">{_svg_escape(label)}</text>')
        for m_idx, row in enumerate(summary_rows):
            count = int(row[key])
            xx = right_x0 + 136
            bar_y = yy + m_idx * 14
            ww = (count / max_count) * (right_w - 160) if max_count else 0
            color = MODEL_COLORS[row["model"]]
            svg.append(f'<rect x="{xx}" y="{bar_y}" width="{ww:.1f}" height="10" fill="{color}" opacity="0.92"/>')
            svg.append(f'<text x="{xx + ww + 6:.1f}" y="{bar_y + 9}" font-size="11" fill="{color}">{row["model"]} {count}</text>')

    svg.append(f'<text x="{right_x0}" y="{right_y0 + right_h - 22}" font-size="14" fill="#991b1b" font-weight="700">Measured takeaway: saved LoRAs do touch visual transformer blocks, but not visual.merger.</text>')
    svg.append(f'<text x="{right_x0}" y="{right_y0 + right_h}" font-size="13" fill="#475569">So you can say “visual blocks adapted”, but you should not claim a visual.merger intervention unless you actually run one.</text>')
    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    rows = []
    summaries = []
    for model_key, model_label in MODEL_SPECS:
        adapter_dir = ADAPTER_ROOT / model_key / "best"
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        scale = float(cfg["lora_alpha"]) / float(cfg["r"])
        shard = SafeTensorShard(adapter_dir / "adapter_model.safetensors")
        delta_sq = defaultdict(float)
        counts = defaultdict(int)
        merger_count = 0
        patch_count = 0

        keys = shard.keys()
        a_keys = [k for k in keys if k.endswith(".lora_A.weight")]
        for a_key in a_keys:
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            group = _group_for_key(a_key)
            if ".visual.merger." in a_key:
                merger_count += 1
            if ".visual.patch_embed." in a_key:
                patch_count += 1
            if group is None or b_key not in keys:
                continue
            a = shard.get_tensor(a_key).astype(np.float32)
            b = shard.get_tensor(b_key).astype(np.float32)
            delta = np.matmul(b, a) * scale
            dn = float(np.linalg.norm(delta))
            delta_sq[group] += dn * dn
            counts[group] += 1
            rows.append({"section": "module", "model": model_label, "group": group, "delta_norm": dn, "a_key": a_key})

        total = sum(delta_sq.values()) or 1.0
        summary = {
            "section": "summary",
            "model": model_label,
            "visual_merger_count": merger_count,
            "visual_patch_embed_count": patch_count,
        }
        for group in GROUP_ORDER:
            summary[f"{group}_delta_norm"] = math.sqrt(delta_sq[group])
            summary[f"{group}_share_pct"] = round(100.0 * delta_sq[group] / total, 1)
            summary[f"{group}_count"] = counts[group]
        rows.append(summary)
        summaries.append(summary)

    csv_path = DEFAULT_DATA_DIR / "backup_lora_adapter_delta_proxy.csv"
    fieldnames = [
        "section",
        "model",
        "group",
        "delta_norm",
        "a_key",
        "visual_attn_delta_norm",
        "visual_attn_share_pct",
        "visual_attn_count",
        "visual_mlp_delta_norm",
        "visual_mlp_share_pct",
        "visual_mlp_count",
        "language_attn_delta_norm",
        "language_attn_share_pct",
        "language_attn_count",
        "language_mlp_delta_norm",
        "language_mlp_share_pct",
        "language_mlp_count",
        "visual_merger_count",
        "visual_patch_embed_count",
    ]
    _write_csv(csv_path, rows, fieldnames)
    svg_path = DEFAULT_OUT_DIR / "backup_lora_adapter_delta_proxy.svg"
    _write_svg(summaries, svg_path)
    print(f"[adapter-proxy] wrote csv: {csv_path}")
    print(f"[adapter-proxy] wrote svg: {svg_path}")
    for row in summaries:
        print(
            f"[adapter-proxy] {row['model']}: "
            f"vis_attn={row['visual_attn_share_pct']}%, "
            f"vis_mlp={row['visual_mlp_share_pct']}%, "
            f"lang_attn={row['language_attn_share_pct']}%, "
            f"lang_mlp={row['language_mlp_share_pct']}%, "
            f"merger_targets={row['visual_merger_count']}"
        )


if __name__ == "__main__":
    main()
