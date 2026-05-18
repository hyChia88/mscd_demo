#!/usr/bin/env python3
"""Offline LoRA delta-to-base audit for saved Qwen2.5-VL adapters.

This script computes a post-hoc effective adapter ratio

    ||Delta W||_F / ||W_base||_F

for each LoRA-targeted module, where:

    Delta W = (alpha / r) * (B @ A)

It then aggregates these module-level ratios into architecture groups:
    - visual_attn
    - visual_mlp
    - language_attn
    - language_mlp

The goal is not to prove causality. Instead, it provides a defensible,
measured answer when readers ask whether the learned adapters actually moved
the visual backbone and how that compares with language-side movement.

Important limitation:
    The current saved adapters do not contain LoRA weights for `visual.merger`
    or `visual.patch_embed`. If those modules are not trainable, no ratio can
    be measured for them here.

Usage:
  python mscd_demo/evaluation/analysis/analyze_lora_delta_to_base.py
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent.parent
REPO_ROOT = PROJECT_ROOT.parent

ADAPTER_ROOT = PROJECT_ROOT / "models" / "lora6_v2_ap_20260331"
BASE_SNAPSHOT = (
    Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots")
    / "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
BASE_INDEX = BASE_SNAPSHOT / "model.safetensors.index.json"

DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "plots" / "final"
DEFAULT_DATA_DIR = DEFAULT_OUT_DIR / "data"

MODEL_SPECS = [
    ("g7_position_context", "G7"),
    ("g8_posctx_dim", "G8"),
    ("g9_opencv_cluster", "G9"),
]

GROUP_ORDER = [
    "visual_attn",
    "visual_mlp",
    "language_attn",
    "language_mlp",
]
GROUP_LABELS = {
    "visual_attn": "Visual\nattention",
    "visual_mlp": "Visual\nMLP",
    "language_attn": "Language\nattention",
    "language_mlp": "Language\nMLP",
}
MODEL_COLORS = {
    "G7": "#7C3AED",
    "G8": "#2563EB",
    "G9": "#0F766E",
}


_DTYPE_MAP = {
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
}


class SafeTensorShard:
    """Lightweight safetensors reader using only Python + NumPy."""

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

        dtype = _DTYPE_MAP.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported dtype {dtype_name} in {self.path}")
        count = byte_count // np.dtype(dtype).itemsize
        arr = np.memmap(self.path, mode="r", dtype=dtype, offset=byte_start, shape=(count,))
        return np.asarray(arr).reshape(shape)


def _load_base_index(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in payload["weight_map"].items()}


def _collect_base_norms(weight_map: dict[str, str], base_snapshot: Path, required_keys: set[str]) -> dict[str, float]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for weight_name in required_keys:
        shard_name = weight_map[weight_name]
        by_shard[shard_name].append(weight_name)

    norms: dict[str, float] = {}
    for shard_name, weight_names in by_shard.items():
        shard_path = base_snapshot / shard_name
        handle = SafeTensorShard(shard_path)
        for weight_name in weight_names:
            tensor = handle.get_tensor(weight_name).astype(np.float32)
            norms[weight_name] = float(np.linalg.norm(tensor))
    return norms


def _base_weight_name_from_lora_key(lora_a_key: str) -> str:
    base_name = lora_a_key.replace(".lora_A.weight", ".weight")
    if base_name.startswith("base_model.model.model.language_model."):
        return "model." + base_name[len("base_model.model.model.language_model.") :]
    if base_name.startswith("base_model.model.model."):
        return base_name[len("base_model.model.model.") :]
    raise ValueError(f"Unexpected LoRA key prefix: {lora_a_key}")


def _collect_required_base_keys(model_specs: list[tuple[str, str]]) -> set[str]:
    required: set[str] = set()
    for model_key, _model_label in model_specs:
        adapter_dir = ADAPTER_ROOT / model_key / "best"
        if not adapter_dir.exists():
            continue
        handle = SafeTensorShard(adapter_dir / "adapter_model.safetensors")
        for key in handle.keys():
            if key.endswith(".lora_A.weight"):
                required.add(_base_weight_name_from_lora_key(key))
    return required


def _group_for_base_weight(base_weight_name: str) -> str | None:
    if base_weight_name.startswith("visual.blocks.") and ".attn." in base_weight_name:
        return "visual_attn"
    if base_weight_name.startswith("visual.blocks.") and ".mlp." in base_weight_name:
        return "visual_mlp"
    if base_weight_name.startswith("model.layers.") and ".self_attn." in base_weight_name:
        return "language_attn"
    if base_weight_name.startswith("model.layers.") and ".mlp." in base_weight_name:
        return "language_mlp"
    return None


def _adapter_alpha_and_rank(adapter_dir: Path) -> tuple[float, int]:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    return float(cfg["lora_alpha"]), int(cfg["r"])


def _compute_model_rows(
    adapter_dir: Path,
    model_label: str,
    base_norms: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alpha, rank = _adapter_alpha_and_rank(adapter_dir)
    scale = alpha / float(rank)
    adapter_path = adapter_dir / "adapter_model.safetensors"

    per_group_sum_delta_sq = defaultdict(float)
    per_group_sum_base_sq = defaultdict(float)
    per_group_module_ratios = defaultdict(list)
    per_group_module_count = defaultdict(int)
    unmatched_base_names: list[str] = []

    rows: list[dict[str, Any]] = []
    handle = SafeTensorShard(adapter_path)
    keys = list(handle.keys())
    a_keys = [key for key in keys if key.endswith(".lora_A.weight")]
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in keys:
            continue
        base_weight_name = _base_weight_name_from_lora_key(a_key)
        group = _group_for_base_weight(base_weight_name)
        if group is None:
            continue
        base_norm = base_norms.get(base_weight_name)
        if base_norm is None or base_norm <= 0:
            unmatched_base_names.append(base_weight_name)
            continue

        a = handle.get_tensor(a_key).astype(np.float32)
        b = handle.get_tensor(b_key).astype(np.float32)
        delta = np.matmul(b, a) * scale
        delta_norm = float(np.linalg.norm(delta))
        ratio = delta_norm / base_norm

        per_group_sum_delta_sq[group] += delta_norm ** 2
        per_group_sum_base_sq[group] += base_norm ** 2
        per_group_module_ratios[group].append(ratio)
        per_group_module_count[group] += 1

        rows.append(
            {
                "section": "module",
                "model": model_label,
                "group": group,
                "base_weight": base_weight_name,
                "delta_norm": delta_norm,
                "base_norm": base_norm,
                "delta_to_base_ratio": ratio,
            }
        )

    summary = {
        "model": model_label,
        "modules_total": sum(per_group_module_count.values()),
        "missing_base_matches": len(unmatched_base_names),
        "has_visual_merger_lora": False,
        "has_visual_patch_embed_lora": False,
    }
    for group in GROUP_ORDER:
        ratios = per_group_module_ratios[group]
        global_ratio = math.sqrt(per_group_sum_delta_sq[group]) / math.sqrt(per_group_sum_base_sq[group]) if per_group_sum_base_sq[group] > 0 else 0.0
        rows.append(
            {
                "section": "group",
                "model": model_label,
                "group": group,
                "base_weight": "",
                "delta_norm": math.sqrt(per_group_sum_delta_sq[group]),
                "base_norm": math.sqrt(per_group_sum_base_sq[group]),
                "delta_to_base_ratio": global_ratio,
                "module_count": per_group_module_count[group],
                "mean_module_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
                "median_module_ratio": sorted(ratios)[len(ratios) // 2] if ratios else 0.0,
            }
        )
        summary[f"{group}_global_ratio"] = global_ratio
        summary[f"{group}_module_count"] = per_group_module_count[group]
        summary[f"{group}_mean_module_ratio"] = sum(ratios) / len(ratios) if ratios else 0.0
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_svg_plot(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    width = 1320
    height = 760
    margin_left = 92
    margin_right = 36
    margin_top = 108
    margin_bottom = 108
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    n_groups = len(GROUP_ORDER)
    n_models = len(summary_rows)
    cluster_w = plot_w / n_groups
    bar_w = min(24.0, cluster_w / max(n_models + 2, 3))
    gap = bar_w * 0.2

    all_vals = [
        float(row.get(f"{group}_global_ratio", 0.0))
        for row in summary_rows
        for group in GROUP_ORDER
        if float(row.get(f"{group}_global_ratio", 0.0)) > 0
    ]
    if not all_vals:
        raise RuntimeError("No positive delta-to-base ratios were computed.")
    min_exp = math.floor(math.log10(min(all_vals)))
    max_exp = math.ceil(math.log10(max(all_vals)))
    tick_exps = list(range(min_exp, max_exp + 1))

    def y_for(val: float) -> float:
        log_val = math.log10(max(val, 10 ** min_exp))
        t = (log_val - min_exp) / max(max_exp - min_exp, 1e-9)
        return margin_top + plot_h * (1.0 - t)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="36" font-size="28" font-weight="700" fill="#0f172a">'
        'Post-hoc LoRA delta-to-base ratio by architecture group</text>',
        f'<text x="{margin_left}" y="66" font-size="16" fill="#475569">'
        'Computed offline from saved adapters and cached Qwen2.5-VL base weights.</text>',
        f'<text x="{margin_left}" y="88" font-size="16" fill="#475569">'
        'No visual.merger or visual.patch_embed LoRA weights were found in these runs.</text>',
    ]

    # grid + y ticks
    for exp in tick_exps:
        tick_val = 10 ** exp
        yy = y_for(tick_val)
        svg.append(f'<line x1="{margin_left}" y1="{yy:.1f}" x2="{width - margin_right}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{margin_left - 10}" y="{yy + 5:.1f}" text-anchor="end" font-size="13" fill="#475569">1e{exp}</text>')

    # axes
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#0f172a" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#0f172a" stroke-width="1.5"/>')

    # bars
    for g_idx, group in enumerate(GROUP_ORDER):
        cluster_x = margin_left + cluster_w * g_idx + cluster_w / 2.0
        total_bar_w = n_models * bar_w + (n_models - 1) * gap
        start_x = cluster_x - total_bar_w / 2.0
        for m_idx, row in enumerate(summary_rows):
            val = float(row.get(f"{group}_global_ratio", 0.0))
            if val <= 0:
                continue
            xx = start_x + m_idx * (bar_w + gap)
            yy = y_for(val)
            hh = (height - margin_bottom) - yy
            color = MODEL_COLORS.get(str(row["model"]), "#64748B")
            svg.append(
                f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{hh:.1f}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{xx + bar_w / 2.0:.1f}" y="{yy - 6:.1f}" text-anchor="middle" '
                f'font-size="10" fill="#334155" transform="rotate(-90 {xx + bar_w / 2.0:.1f},{yy - 6:.1f})">{val:.1e}</text>'
            )
        label = GROUP_LABELS[group].replace("\n", " ")
        svg.append(
            f'<text x="{cluster_x:.1f}" y="{height - margin_bottom + 26}" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#0f172a">{_svg_escape(label)}</text>'
        )

    # legend
    leg_x = margin_left
    leg_y = height - 46
    for idx, row in enumerate(summary_rows):
        color = MODEL_COLORS.get(str(row["model"]), "#64748B")
        x0 = leg_x + idx * 120
        svg.append(f'<rect x="{x0}" y="{leg_y}" width="18" height="18" fill="{color}" stroke="none"/>')
        svg.append(f'<text x="{x0 + 26}" y="{leg_y + 14}" font-size="14" fill="#0f172a">{_svg_escape(str(row["model"]))}</text>')

    # footer note
    note = (
        "Interpretation: this is a post-hoc movement audit of the saved adapters. "
        "It does not make a causal visual.merger claim."
    )
    svg.append(f'<text x="{width - margin_right}" y="{height - 18}" text-anchor="end" font-size="13" fill="#64748B">{_svg_escape(note)}</text>')
    svg.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    if not BASE_INDEX.exists():
        raise FileNotFoundError(f"Missing cached base-model index: {BASE_INDEX}")

    base_weight_map = _load_base_index(BASE_INDEX)
    required_base_keys = _collect_required_base_keys(MODEL_SPECS)
    print(f"[lora-delta] required base weights: {len(required_base_keys)}", flush=True)
    base_norms = _collect_base_norms(base_weight_map, BASE_SNAPSHOT, required_base_keys)

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_key, model_label in MODEL_SPECS:
        adapter_dir = ADAPTER_ROOT / model_key / "best"
        if not adapter_dir.exists():
            continue
        print(f"[lora-delta] processing {model_label} from {adapter_dir}...", flush=True)
        rows, summary = _compute_model_rows(adapter_dir, model_label, base_norms)
        all_rows.extend(rows)
        summary_rows.append(summary)

    csv_rows = all_rows + [
        {
            "section": "summary",
            **summary,
        }
        for summary in summary_rows
    ]
    fieldnames = [
        "section",
        "model",
        "group",
        "base_weight",
        "delta_norm",
        "base_norm",
        "delta_to_base_ratio",
        "module_count",
        "mean_module_ratio",
        "median_module_ratio",
        "modules_total",
        "missing_base_matches",
        "has_visual_merger_lora",
        "has_visual_patch_embed_lora",
        "visual_attn_global_ratio",
        "visual_attn_module_count",
        "visual_attn_mean_module_ratio",
        "visual_mlp_global_ratio",
        "visual_mlp_module_count",
        "visual_mlp_mean_module_ratio",
        "language_attn_global_ratio",
        "language_attn_module_count",
        "language_attn_mean_module_ratio",
        "language_mlp_global_ratio",
        "language_mlp_module_count",
        "language_mlp_mean_module_ratio",
    ]

    csv_path = DEFAULT_DATA_DIR / "backup_lora_delta_to_base_ratio.csv"
    _write_csv(csv_path, csv_rows, fieldnames)
    plot_path = DEFAULT_OUT_DIR / "backup_lora_delta_to_base_ratio.svg"
    _write_svg_plot(summary_rows, plot_path)

    print(f"[lora-delta] wrote csv: {csv_path}")
    print(f"[lora-delta] wrote svg: {plot_path}")
    print("[lora-delta] key takeaway:")
    for row in summary_rows:
        print(
            f"  {row['model']}: "
            f"vis_attn={row['visual_attn_global_ratio']:.3e}, "
            f"vis_mlp={row['visual_mlp_global_ratio']:.3e}, "
            f"lang_attn={row['language_attn_global_ratio']:.3e}, "
            f"lang_mlp={row['language_mlp_global_ratio']:.3e}"
        )


if __name__ == "__main__":
    main()
