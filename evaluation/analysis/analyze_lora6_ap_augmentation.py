#!/usr/bin/env python3
"""Analyze LoRA6-AP dataset growth and augmentation patterns.

This complements the AP held-out topology analysis with a whole-dataset view:
- how canonical train/eval/augmented assets differ in size
- which augmentation axes expand the dataset
- how predicates are amplified after augmentation
- how many variants each canonical train case spawns
"""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mscd")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_TRAIN_CANONICAL = (
    REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap" / "train" / "lora6_v2_ap_train_canonical_m.jsonl"
)
DEFAULT_EVAL_CANONICAL = (
    REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap" / "train" / "lora6_v2_ap_eval_canonical_m.jsonl"
)
DEFAULT_TRAIN_AUG = (
    REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap" / "train" / "lora6_v2_ap_train_aug.jsonl"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "topology_analysis" / "lora6_ap_all"

PREDICATE_ORDER = ["CONNECTS_TO", "FILLS", "NEXT_TO", "ADJACENT_TO"]
MODALITY_ORDER = ["site+floorplan+chat", "floorplan+chat", "site+chat"]
SCALE_ORDER = ["S", "M", "L"]
TEXT_ORDER = ["T1", "T2", "T3"]
SPLIT_ORDER = ["Train Canonical", "Eval Canonical", "Train Aug"]
AP_FLOW_PALETTE = ["#4f6d7a", "#3b8ea5", "#c8553d", "#7a9e7e", "#9d4edd", "#e09f3e", "#6c757d", "#2a9d8f"]
ANCHOR_PALETTE = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#84a59d", "#8d99ae"]
CONTEXT_SHADES = ["#e2e8f0", "#cbd5e1", "#94a3b8", "#64748b", "#475569", "#334155"]
SPLIT_SHADES = ["#dbeafe", "#93c5fd", "#2563eb"]
SCALE_SHADES = ["#d1fae5", "#6ee7b7", "#047857"]
TEXT_SHADES = ["#fee2e2", "#fca5a5", "#b91c1c"]
SHORT_OBJ_LABELS = {
    "IfcWallStandardCase": "WallStd",
    "IfcWall": "Wall",
    "IfcWindow": "Window",
    "IfcDoor": "Door",
    "IfcBeam": "Beam",
    "IfcSlab": "Slab",
    "IfcRailing": "Railing",
    "IfcStair": "Stair",
    "IfcColumn": "Column",
}
FAMILY_ORDER = [
    "singleton:CONNECTS_TO",
    "singleton:ADJACENT_TO",
    "paired:FILLS+NEXT_TO",
    "triad:FILLS+NEXT_TO+NEXT_TO",
    "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)",
    "singleton:FILLS",
]


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _count(rows: list[dict], key: str) -> Counter:
    return Counter(str(row.get(key, "?")) for row in rows)


def _cross(rows: list[dict], left: str, right: str) -> Counter:
    return Counter((str(row.get(left, "?")), str(row.get(right, "?"))) for row in rows)


def _relation_signature(rels: list[dict]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (
                str(r.get("predicate", "?")),
                str(r.get("object_type", "?")),
                str(r.get("direction", "")),
            )
            for r in rels
        )
    )


def _topology_family(rels: list[dict]) -> str:
    preds = [str(r.get("predicate", "?")) for r in rels]
    pred_hist = Counter(preds)
    n = len(rels)

    if n == 0:
        return "empty"
    if n == 1:
        return f"singleton:{preds[0]}"
    if n == 2:
        if pred_hist == Counter({"FILLS": 1, "NEXT_TO": 1}):
            return "paired:FILLS+NEXT_TO"
        return "paired:other"
    if n == 3:
        if pred_hist == Counter({"FILLS": 1, "NEXT_TO": 2}):
            objs = [str(r.get("object_type", "?")) for r in rels if r.get("predicate") == "NEXT_TO"]
            if len(set(objs)) > 1:
                return "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)"
            return "triad:FILLS+NEXT_TO+NEXT_TO"
        return "triad:other"
    return f"{n}-rel:other"


def _short_obj_label(name: str | None) -> str:
    if not name:
        return "?"
    return SHORT_OBJ_LABELS.get(str(name), str(name).replace("Ifc", ""))


def _compact_label_sequence(labels: list[str]) -> str:
    counts = Counter(labels)
    parts = []
    for label, count in sorted(counts.items()):
        parts.append(f"{label}x{count}" if count > 1 else label)
    return "+".join(parts) if parts else "none"


def _anchor_pattern(rels: list[dict]) -> str:
    next_to = [_short_obj_label(r.get("object_type")) for r in rels if r.get("predicate") == "NEXT_TO"]
    if next_to:
        return _compact_label_sequence(next_to)
    others = [_short_obj_label(r.get("object_type")) for r in rels]
    return _compact_label_sequence(others)


def _display_label(label: str) -> str:
    mapping = {
        "site+floorplan+chat": "site+fp+chat",
        "floorplan+chat": "fp+chat",
        "site+chat": "site+chat",
        "Train Canonical": "Train\nCanonical",
        "Eval Canonical": "Eval\nCanonical",
        "Train Aug": "Train\nAug",
        "singleton:CONNECTS_TO": "1: CONNECTS_TO",
        "singleton:ADJACENT_TO": "1: ADJACENT_TO",
        "paired:FILLS+NEXT_TO": "2: FILLS+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO": "3: FILLS+NEXT_TOx2",
        "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)": "3: FILLS+NEXT_TOx2*",
        "singleton:FILLS": "1: FILLS",
        "singleton:NEXT_TO": "1: NEXT_TO",
        "paired:other": "2: other",
        "triad:other": "3: other",
    }
    return mapping.get(label, label)


def _shade_map(labels: list[str]) -> dict[str, str]:
    return {label: CONTEXT_SHADES[min(i, len(CONTEXT_SHADES) - 1)] for i, label in enumerate(labels)}


def _shade_map_from_palette(labels: list[str], palette: list[str]) -> dict[str, str]:
    return {label: palette[min(i, len(palette) - 1)] for i, label in enumerate(labels)}


def _complexity_score(rels: list[dict]) -> float:
    preds = [str(r.get("predicate", "?")) for r in rels]
    obj_types = [str(r.get("object_type", "?")) for r in rels]
    next_objs = {str(r.get("object_type", "?")) for r in rels if r.get("predicate") == "NEXT_TO"}
    repeated_same = 1 if any(v > 1 for v in Counter(preds).values()) else 0
    mixed_anchor = 1 if len(next_objs) > 1 else 0
    directional = len({str(r.get("direction") or "") for r in rels if r.get("direction")})
    return (
        len(rels)
        + 0.45 * max(len(set(preds)) - 1, 0)
        + 0.55 * max(len(set(obj_types)) - 1, 0)
        + 0.35 * mixed_anchor
        + 0.15 * repeated_same
        + 0.1 * max(directional - 1, 0)
    )


def _assistant_constraints(row: dict) -> dict:
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {}
    return {}


def _user_payload(row: dict) -> tuple[list[str], str]:
    for msg in row.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content") or []
        images = []
        text = ""
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                images.append(str(item.get("image", "")))
            elif item.get("type") == "text":
                text = str(item.get("text", ""))
        return images, text
    return [], ""


def _file_uri_to_path(uri: str | None) -> Path | None:
    if not uri:
        return None
    raw = str(uri)
    if raw.startswith("file://"):
        return Path(raw[7:])
    p = Path(raw)
    return p if p.exists() else None


def _load_image_or_blank(path: Path | None) -> np.ndarray:
    if path is None or not path.exists():
        return np.ones((128, 128, 3), dtype=np.float32)
    try:
        img = mpimg.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img
    except Exception:
        return np.ones((128, 128, 3), dtype=np.float32)


def _ordered(counter: Counter, order: list[str]) -> list[tuple[str, int]]:
    extra = sorted([k for k in counter if k not in order])
    return [(k, counter[k]) for k in order if k in counter] + [(k, counter[k]) for k in extra]


def _stack_boxes(items: list[tuple[str, int]], x: float, width: float, gap: float = 0.03) -> dict[str, tuple[float, float, float]]:
    total = sum(v for _, v in items)
    usable = 1.0 - gap * max(len(items) - 1, 0)
    y = 1.0
    boxes = {}
    for label, value in items:
        h = usable * value / total if total else 0.0
        y0 = y - h
        boxes[label] = (x, y0, h)
        y = y0 - gap
    return boxes


def _draw_flow(ax, x0: float, y0: float, h0: float, x1: float, y1: float, h1: float, color: str, alpha: float = 0.35) -> None:
    c = 0.22 * (x1 - x0)
    verts = [
        (x0, y0),
        (x0 + c, y0),
        (x1 - c, y1),
        (x1, y1),
        (x1, y1 + h1),
        (x1 - c, y1 + h1),
        (x0 + c, y0 + h0),
        (x0, y0 + h0),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def _plot_growth_overview(out_dir: Path, stats: dict) -> None:
    train_canonical = stats["train_canonical_rows"]
    eval_canonical = stats["eval_canonical_rows"]
    train_aug = stats["train_aug_rows"]
    aug_extra = max(train_aug - train_canonical, 0)

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    y = np.arange(2)
    h = 0.56

    colors = {
        "train_canonical": "#3b8ea5",
        "eval_canonical": "#7a9e7e",
        "aug_extra": "#c8553d",
    }

    ax.barh(y[0], train_canonical, height=h, color=colors["train_canonical"], label="Train Canonical")
    ax.barh(y[0], eval_canonical, left=train_canonical, height=h, color=colors["eval_canonical"], label="Eval Canonical")

    ax.barh(y[1], train_canonical, height=h, color=colors["train_canonical"])
    ax.barh(y[1], aug_extra, left=train_canonical, height=h, color=colors["aug_extra"], label="Augmented Extra")

    ax.set_yticks(y)
    ax.set_yticklabels(["Canonical Dataset Split", "Training Supervision Pool"])
    ax.invert_yaxis()
    ax.set_xlabel("Rows")
    ax.set_title("LoRA6-AP Dataset Composition And Augmentation Growth")

    ax.text(train_canonical / 2, y[0], f"train\n{train_canonical}", ha="center", va="center", color="white", fontsize=10, weight="bold")
    ax.text(
        train_canonical + eval_canonical / 2,
        y[0],
        f"eval\n{eval_canonical}",
        ha="center",
        va="center",
        color="#17324d",
        fontsize=10,
        weight="bold",
    )
    ax.text(train_canonical / 2, y[1], f"base\n{train_canonical}", ha="center", va="center", color="white", fontsize=10, weight="bold")
    ax.text(
        train_canonical + aug_extra / 2,
        y[1],
        f"aug extra\n{aug_extra}",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        weight="bold",
    )

    ax.text(train_canonical + eval_canonical + max(train_aug, train_canonical) * 0.015, y[0], f"total={train_canonical + eval_canonical}", va="center", ha="left", fontsize=10, weight="bold")
    ax.text(train_aug + max(train_aug, train_canonical) * 0.015, y[1], f"total={train_aug}", va="center", ha="left", fontsize=10, weight="bold")

    train_factor = train_aug / max(train_canonical, 1)
    canon_factor = train_aug / max(train_canonical + eval_canonical, 1)
    note = (
        f"Train canonical -> train aug: {train_factor:.2f}x\n"
        f"Canonical all -> train aug: {canon_factor:.2f}x\n"
        f"Unique train bases in aug: {stats['train_aug_unique_bases']}"
    )
    ax.text(0.98, 0.06, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, family="monospace")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_growth_overview.png", dpi=180)
    plt.close(fig)


def _plot_aug_flow(
    out_dir: Path,
    split: Counter,
    relation_mult: Counter,
    topology_family: Counter,
    anchor_pattern: Counter,
    scale: Counter,
    text: Counter,
    links01: Counter,
    links12: Counter,
    links23: Counter,
    links34: Counter,
    links45: Counter,
) -> None:
    stage0 = _ordered(split, SPLIT_ORDER)
    stage1 = _ordered(relation_mult, ["1-rel", "2-rel", "3-rel"])
    stage2 = _ordered(topology_family, FAMILY_ORDER)
    stage3 = sorted(anchor_pattern.items(), key=lambda kv: (-kv[1], kv[0]))
    stage4 = _ordered(scale, SCALE_ORDER)
    stage5 = _ordered(text, TEXT_ORDER)

    x_positions = [0.03, 0.19, 0.36, 0.55, 0.75, 0.90]
    width = 0.07
    box0 = _stack_boxes(stage0, x_positions[0], width)
    box1 = _stack_boxes(stage1, x_positions[1], width)
    box2 = _stack_boxes(stage2, x_positions[2], width)
    box3 = _stack_boxes(stage3, x_positions[3], width)
    box4 = _stack_boxes(stage4, x_positions[4], width)
    box5 = _stack_boxes(stage5, x_positions[5], width)

    stage0_labels = [k for k, _ in stage0]
    stage1_labels = [k for k, _ in stage1]
    stage2_labels = [k for k, _ in stage2]
    stage3_labels = [k for k, _ in stage3]
    stage4_labels = [k for k, _ in stage4]
    stage5_labels = [k for k, _ in stage5]

    split_colors = _shade_map_from_palette(stage0_labels, SPLIT_SHADES)
    mult_colors = _shade_map(stage1_labels)
    family_colors = {label: AP_FLOW_PALETTE[i % len(AP_FLOW_PALETTE)] for i, label in enumerate(stage2_labels)}
    anchor_colors = {label: ANCHOR_PALETTE[i % len(ANCHOR_PALETTE)] for i, label in enumerate(stage3_labels)}
    scale_colors = _shade_map_from_palette(stage4_labels, SCALE_SHADES)
    text_colors = _shade_map_from_palette(stage5_labels, TEXT_SHADES)

    fig, ax = plt.subplots(figsize=(21, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top_margin = 0.13
    bottom_margin = 0.04
    usable = 1.0 - top_margin - bottom_margin

    def _remap_boxes(boxes: dict[str, tuple[float, float, float]]) -> dict[str, tuple[float, float, float]]:
        return {label: (x, bottom_margin + y * usable, h * usable) for label, (x, y, h) in boxes.items()}

    box0 = _remap_boxes(box0)
    box1 = _remap_boxes(box1)
    box2 = _remap_boxes(box2)
    box3 = _remap_boxes(box3)
    box4 = _remap_boxes(box4)
    box5 = _remap_boxes(box5)

    from_offsets = defaultdict(float)
    to_offsets = defaultdict(float)
    for src, dst in sorted(links01.keys(), key=lambda pair: (SPLIT_ORDER.index(pair[0]), stage1_labels.index(pair[1]))):
        count = links01[(src, dst)]
        x0, y0, h0 = box0[src]
        x1, y1, h1 = box1[dst]
        seg0 = h0 * count / split[src]
        seg1 = h1 * count / relation_mult[dst]
        y_from = y0 + from_offsets[src]
        y_to = y1 + to_offsets[dst]
        _draw_flow(ax, x0 + width, y_from, seg0, x1, y_to, seg1, mult_colors[dst], alpha=0.28)
        from_offsets[src] += seg0
        to_offsets[dst] += seg1

    from_offsets = defaultdict(float)
    to_offsets = defaultdict(float)
    for src, dst in sorted(
        links12.keys(),
        key=lambda pair: (stage1_labels.index(pair[0]), stage2_labels.index(pair[1]) if pair[1] in stage2_labels else len(stage2_labels)),
    ):
        count = links12[(src, dst)]
        x0, y0, h0 = box1[src]
        x1, y1, h1 = box2[dst]
        seg0 = h0 * count / relation_mult[src]
        seg1 = h1 * count / topology_family[dst]
        y_from = y0 + from_offsets[src]
        y_to = y1 + to_offsets[dst]
        _draw_flow(ax, x0 + width, y_from, seg0, x1, y_to, seg1, family_colors.get(dst, "#999999"), alpha=0.38)
        from_offsets[src] += seg0
        to_offsets[dst] += seg1

    from_offsets = defaultdict(float)
    to_offsets = defaultdict(float)
    for src, dst in sorted(
        links23.keys(),
        key=lambda pair: (stage2_labels.index(pair[0]), stage3_labels.index(pair[1]) if pair[1] in stage3_labels else len(stage3_labels)),
    ):
        count = links23[(src, dst)]
        x0, y0, h0 = box2[src]
        x1, y1, h1 = box3[dst]
        seg0 = h0 * count / topology_family[src]
        seg1 = h1 * count / anchor_pattern[dst]
        y_from = y0 + from_offsets[src]
        y_to = y1 + to_offsets[dst]
        _draw_flow(ax, x0 + width, y_from, seg0, x1, y_to, seg1, anchor_colors.get(dst, "#999999"), alpha=0.38)
        from_offsets[src] += seg0
        to_offsets[dst] += seg1

    from_offsets = defaultdict(float)
    to_offsets = defaultdict(float)
    for src, dst in sorted(links34.keys(), key=lambda pair: (stage3_labels.index(pair[0]), SCALE_ORDER.index(pair[1]))):
        count = links34[(src, dst)]
        x0, y0, h0 = box3[src]
        x1, y1, h1 = box4[dst]
        seg0 = h0 * count / anchor_pattern[src]
        seg1 = h1 * count / scale[dst]
        y_from = y0 + from_offsets[src]
        y_to = y1 + to_offsets[dst]
        _draw_flow(ax, x0 + width, y_from, seg0, x1, y_to, seg1, scale_colors[dst], alpha=0.3)
        from_offsets[src] += seg0
        to_offsets[dst] += seg1

    from_offsets = defaultdict(float)
    to_offsets = defaultdict(float)
    for src, dst in sorted(links45.keys(), key=lambda pair: (SCALE_ORDER.index(pair[0]), TEXT_ORDER.index(pair[1]))):
        count = links45[(src, dst)]
        x0, y0, h0 = box4[src]
        x1, y1, h1 = box5[dst]
        seg0 = h0 * count / scale[src]
        seg1 = h1 * count / text[dst]
        y_from = y0 + from_offsets[src]
        y_to = y1 + to_offsets[dst]
        _draw_flow(ax, x0 + width, y_from, seg0, x1, y_to, seg1, text_colors[dst], alpha=0.3)
        from_offsets[src] += seg0
        to_offsets[dst] += seg1

    for label, (x, y, h) in box0.items():
        ax.add_patch(Rectangle((x, y), width, h, facecolor=split_colors[label], edgecolor="#556", linewidth=1))
        ax.text(x - 0.008, y + h / 2, f"{_display_label(label)}\n{split[label]}", ha="right", va="center", fontsize=9.2, color="#1f2933", weight="bold")
    for label, (x, y, h) in box1.items():
        ax.add_patch(Rectangle((x, y), width, h, facecolor=mult_colors[label], edgecolor="#556", linewidth=1))
        ax.text(x + width / 2, y + h / 2, f"{label}\n{relation_mult[label]}", ha="center", va="center", fontsize=9.0, color="#1f2933", weight="bold")
    for label, (x, y, h) in box2.items():
        ax.add_patch(Rectangle((x, y), width, h, facecolor=family_colors.get(label, "#cccccc"), edgecolor="#445", linewidth=1))
        ax.text(x + width / 2, y + h / 2, f"{_display_label(label)}\n{topology_family[label]}", ha="center", va="center", fontsize=8.1, color="white", weight="bold")
    for label, (x, y, h) in box3.items():
        ax.add_patch(Rectangle((x, y), width, h, facecolor=anchor_colors[label], edgecolor="#556", linewidth=1))
        ax.text(x + width + 0.008, y + h / 2, f"{label}\n{anchor_pattern[label]}", ha="left", va="center", fontsize=8.8, color="#1f2933", weight="bold")
    for label, (x, y, h) in box4.items():
        ax.add_patch(Rectangle((x, y), width, h, facecolor=scale_colors[label], edgecolor="#556", linewidth=1))
        ax.text(x + width / 2, y + h / 2, f"{label}\n{scale[label]}", ha="center", va="center", fontsize=9.2, color="#1f2933", weight="bold")
    for label, (x, y, h) in box5.items():
        ax.add_patch(Rectangle((x, y), width, h, facecolor=text_colors[label], edgecolor="#556", linewidth=1))
        ax.text(x + width + 0.008, y + h / 2, f"{label}\n{text[label]}", ha="left", va="center", fontsize=9.0, color="#1f2933", weight="bold")

    header_y = 1.0 - top_margin * 0.35
    headers = ["Dataset Split", "Relation Multiplicity", "Topology Family", "Anchor Pattern", "Scale", "Text Tier"]
    for xpos, header in zip(x_positions, headers):
        ax.text(xpos + width / 2, header_y, header, ha="center", va="bottom", fontsize=11.5, weight="bold")

    fig.suptitle("LoRA6-AP Dataset Composition, Augmentation, and Topology Flow", fontsize=16, y=0.995)
    fig.subplots_adjust(top=0.88, left=0.03, right=0.97, bottom=0.05)
    fig.savefig(out_dir / "augmentation_flow_alluvial.png", dpi=180)
    plt.close(fig)


def _plot_predicate_dumbbell(out_dir: Path, canonical_pred: Counter, aug_pred: Counter) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(PREDICATE_ORDER))
    for yi, pred in enumerate(PREDICATE_ORDER):
        x0 = canonical_pred.get(pred, 0)
        x1 = aug_pred.get(pred, 0)
        ax.plot([x0, x1], [yi, yi], color="#9aa5b1", linewidth=2.5)
        ax.scatter(x0, yi, s=70, color="#4f6d7a", zorder=3, label="Train canonical" if yi == 0 else None)
        ax.scatter(x1, yi, s=70, color="#c8553d", zorder=3, label="Train aug" if yi == 0 else None)
        factor = x1 / x0 if x0 else math.inf
        factor_txt = f"{factor:.2f}x" if math.isfinite(factor) else "inf"
        ax.text(max(x0, x1) + 4, yi, factor_txt, va="center", ha="left", fontsize=10, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(PREDICATE_ORDER)
    ax.set_xlabel("Rows")
    ax.set_title("Predicate Amplification: Train Canonical vs Train Aug")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "predicate_growth_dumbbell.png", dpi=180)
    plt.close(fig)


def _plot_base_expansion(out_dir: Path, base_counts: Counter) -> None:
    multiplicity = Counter(base_counts.values())
    xs = sorted(multiplicity)
    ys = [multiplicity[x] for x in xs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(xs, ys, width=0.6, color="#3b8ea5")
    ax.set_xticks(xs)
    ax.set_xlabel("Augmented rows per canonical train base")
    ax.set_ylabel("# base cases")
    ax.set_title("Per-base Augmentation Multiplicity")
    for x, y in zip(xs, ys):
        ax.text(x, y + max(ys) * 0.02, str(y), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "per_base_expansion_histogram.png", dpi=180)
    plt.close(fig)


def _plot_mix_heatmaps(out_dir: Path, aug_rows: list[dict]) -> None:
    mod_scale = _cross(aug_rows, "modality", "scale")
    scale_text = _cross(aug_rows, "scale", "text_tier")

    mod_labels = [k for k, _ in _ordered(_count(aug_rows, "modality"), MODALITY_ORDER)]
    scale_labels = [k for k, _ in _ordered(_count(aug_rows, "scale"), SCALE_ORDER)]
    text_labels = [k for k, _ in _ordered(_count(aug_rows, "text_tier"), TEXT_ORDER)]

    mod_scale_arr = np.zeros((len(mod_labels), len(scale_labels)), dtype=int)
    scale_text_arr = np.zeros((len(scale_labels), len(text_labels)), dtype=int)

    for i, m in enumerate(mod_labels):
        for j, s in enumerate(scale_labels):
            mod_scale_arr[i, j] = mod_scale.get((m, s), 0)
    for i, s in enumerate(scale_labels):
        for j, t in enumerate(text_labels):
            scale_text_arr[i, j] = scale_text.get((s, t), 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, arr, xlab, ylab, title in [
        (axes[0], mod_scale_arr, scale_labels, mod_labels, "Modality × Scale"),
        (axes[1], scale_text_arr, text_labels, scale_labels, "Scale × Text Tier"),
    ]:
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks(np.arange(len(xlab)))
        ax.set_xticklabels(xlab)
        ax.set_yticks(np.arange(len(ylab)))
        ax.set_yticklabels(ylab)
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="#111", fontsize=10, weight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("LoRA6-AP Augmentation Mix", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "augmentation_mix_heatmaps.png", dpi=180)
    plt.close(fig)


def _plot_train_topology_dumbbell(out_dir: Path, canonical_family: Counter, aug_family: Counter) -> None:
    families = [f for f in FAMILY_ORDER if canonical_family.get(f, 0) or aug_family.get(f, 0)]
    extras = sorted([f for f in set(canonical_family) | set(aug_family) if f not in families])
    families.extend(extras)

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.7 * len(families) + 1)))
    y = np.arange(len(families))
    for yi, family in enumerate(families):
        x0 = canonical_family.get(family, 0)
        x1 = aug_family.get(family, 0)
        ax.plot([x0, x1], [yi, yi], color="#b6c2cf", linewidth=2.5)
        ax.scatter(x0, yi, s=70, color="#4f6d7a", zorder=3, label="Train canonical" if yi == 0 else None)
        ax.scatter(x1, yi, s=70, color="#c8553d", zorder=3, label="Train aug" if yi == 0 else None)
        factor = x1 / x0 if x0 else math.inf
        factor_txt = f"{factor:.2f}x" if math.isfinite(factor) else "inf"
        ax.text(max(x0, x1) + 4, yi, factor_txt, va="center", ha="left", fontsize=10, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(families)
    ax.set_xlabel("Rows")
    ax.set_title("Train Topology Families: Canonical vs Augmented")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "train_topology_family_dumbbell.png", dpi=180)
    plt.close(fig)


def _plot_train_topology_mix_heatmap(out_dir: Path, aug_rows: list[dict], family_counter: Counter) -> None:
    families = [f for f in FAMILY_ORDER if family_counter.get(f, 0)]
    extras = sorted([f for f in family_counter if f not in families])
    families.extend(extras)
    modalities = [m for m in MODALITY_ORDER if any(r.get("modality") == m for r in aug_rows)]
    scales = [s for s in SCALE_ORDER if any(r.get("scale") == s for r in aug_rows)]

    fam_mod = Counter((r["topology_family"], r.get("modality", "?")) for r in aug_rows)
    fam_scale = Counter((r["topology_family"], r.get("scale", "?")) for r in aug_rows)

    arr1 = np.zeros((len(families), len(modalities)), dtype=int)
    arr2 = np.zeros((len(families), len(scales)), dtype=int)
    for i, fam in enumerate(families):
        for j, mod in enumerate(modalities):
            arr1[i, j] = fam_mod.get((fam, mod), 0)
        for j, sc in enumerate(scales):
            arr2[i, j] = fam_scale.get((fam, sc), 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.75 * len(families))))
    for ax, arr, xlab, title in [
        (axes[0], arr1, modalities, "Topology Family × Modality"),
        (axes[1], arr2, scales, "Topology Family × Scale"),
    ]:
        im = ax.imshow(arr, cmap="Purples")
        ax.set_xticks(np.arange(len(xlab)))
        ax.set_xticklabels(xlab, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(families)))
        ax.set_yticklabels(families)
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] > 0:
                    ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="#111", fontsize=9, weight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Train Aug Topology Distribution", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "train_topology_mix_heatmaps.png", dpi=180)
    plt.close(fig)


def _plot_train_topology_scatter(out_dir: Path, aug_rows: list[dict], family_counter: Counter) -> None:
    families = [f for f in FAMILY_ORDER if family_counter.get(f, 0)]
    extras = sorted([f for f in family_counter if f not in families])
    families.extend(extras)
    modalities = [m for m in MODALITY_ORDER if any(r.get("modality") == m for r in aug_rows)]
    scale_colors = {"S": AP_FLOW_PALETTE[0], "M": AP_FLOW_PALETTE[1], "L": AP_FLOW_PALETTE[2]}
    scale_offsets = {"S": -0.22, "M": 0.0, "L": 0.22}

    grouped_counts = Counter((r["modality"], r["topology_family"], r.get("scale", "?")) for r in aug_rows)
    grouped_complexity = defaultdict(list)
    for row in aug_rows:
        grouped_complexity[(row["modality"], row["topology_family"], row.get("scale", "?"))].append(row["complexity"])

    x_values = [row["complexity"] for row in aug_rows]
    x_min = min(x_values) - 0.15
    x_max = max(x_values) + 0.2

    fig, axes = plt.subplots(1, len(modalities), figsize=(17, max(5.5, 0.8 * len(families) + 1.5)), sharey=True)
    if len(modalities) == 1:
        axes = [axes]

    for ax, modality in zip(axes, modalities):
        for yi, family in enumerate(families):
            for scale in SCALE_ORDER:
                count = grouped_counts.get((modality, family, scale), 0)
                if count <= 0:
                    continue
                x = float(np.median(grouped_complexity[(modality, family, scale)]))
                y = yi + scale_offsets.get(scale, 0.0)
                size = 55 + 9 * count
                ax.scatter(
                    x,
                    y,
                    s=size,
                    c=scale_colors.get(scale, "#999999"),
                    alpha=0.82,
                    edgecolors="white",
                    linewidths=0.9,
                    zorder=3,
                )
                if count >= 12:
                    ax.text(x, y, str(count), ha="center", va="center", fontsize=8, color="white", weight="bold", zorder=4)

        ax.set_title(modality, fontsize=11, weight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Median topology complexity")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_yticks(range(len(families)))
    axes[0].set_yticklabels(families)
    axes[0].set_ylabel("Topology family")
    axes[0].invert_yaxis()

    handles = [axes[-1].scatter([], [], s=110, c=scale_colors[s], label=f"scale={s}") for s in SCALE_ORDER]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=8, title="Bubble Color", title_fontsize=9)
    fig.suptitle("Train Aug Topology Overview By Modality (bubble size = row count)", fontsize=15, y=0.98)
    fig.text(
        0.99,
        0.02,
        "Aggregated by family + modality + scale; text-tier distribution remains in augmentation_flow_alluvial.png",
        ha="right",
        va="bottom",
        fontsize=8.8,
        family="monospace",
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.98, 0.95))
    fig.savefig(out_dir / "train_topology_scatter.png", dpi=180)
    plt.close(fig)


def _select_train_gallery_rows(aug_rows: list[dict]) -> list[dict]:
    desired = [
        ("site+floorplan+chat", "M", "singleton:CONNECTS_TO"),
        ("site+floorplan+chat", "M", "triad:FILLS+NEXT_TO+NEXT_TO"),
        ("site+floorplan+chat", "L", "paired:FILLS+NEXT_TO"),
        ("site+floorplan+chat", "S", "paired:FILLS+NEXT_TO"),
        ("floorplan+chat", "M", "paired:FILLS+NEXT_TO"),
        ("site+chat", "M", "singleton:ADJACENT_TO"),
    ]
    picked = []
    used = set()
    for modality, scale, family in desired:
        for row in aug_rows:
            key = row["id"]
            if key in used:
                continue
            if row.get("modality") == modality and row.get("scale") == scale and row["topology_family"] == family:
                picked.append(row)
                used.add(key)
                break
    if len(picked) < 6:
        for row in aug_rows:
            if row["id"] in used:
                continue
            picked.append(row)
            used.add(row["id"])
            if len(picked) >= 6:
                break
    return picked[:6]


def _trim_json_text(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _wrap_block(text: str, width: int = 44, max_lines: int | None = None) -> str:
    if not text:
        return ""
    wrapped_lines = []
    for raw_line in text.splitlines() or [""]:
        line = raw_line.rstrip()
        if not line:
            wrapped_lines.append("")
            continue
        indent = len(line) - len(line.lstrip(" "))
        prefix = " " * min(indent, 8)
        chunks = textwrap.wrap(
            line.strip(),
            width=max(12, width - len(prefix)),
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not chunks:
            wrapped_lines.append(prefix)
        else:
            wrapped_lines.extend(prefix + chunk for chunk in chunks)
    if max_lines is not None and len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[: max_lines - 1] + ["..."]
    return "\n".join(wrapped_lines)


def _format_json_for_panel(text: str, width: int = 44, max_lines: int = 14) -> str:
    raw = _trim_json_text(text, limit=900)
    try:
        parsed = json.loads(raw)
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        pretty = raw
    return _wrap_block(pretty, width=width, max_lines=max_lines)


def _plot_train_io_gallery(out_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(18, max(4.2 * n, 7)),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.45]},
    )
    if n == 1:
        axes = np.array([axes])

    for i, row in enumerate(rows):
        images, prompt = _user_payload(row)
        site_img = _load_image_or_blank(_file_uri_to_path(images[0] if len(images) > 0 else None))
        floor_img = _load_image_or_blank(_file_uri_to_path(images[1] if len(images) > 1 else None))

        axes[i, 0].imshow(site_img)
        axes[i, 1].imshow(floor_img)
        axes[i, 0].set_title(f"{row['id']} | {row['modality']}\nSITE", fontsize=10)
        axes[i, 1].set_title(f"Scale={row['scale']} | {row['topology_family']}\nFLOORPLAN", fontsize=10)
        for ax in axes[i, :2]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_anchor("N")

        axes[i, 2].axis("off")
        axes[i, 2].add_patch(
            FancyBboxPatch(
                (0.0, 0.0),
                1.0,
                1.0,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                facecolor="#f8f9fa",
                edgecolor="#ced4da",
                linewidth=1.0,
                transform=axes[i, 2].transAxes,
            )
        )
        wrapped_prompt = _wrap_block(prompt, width=46, max_lines=7)
        wrapped_json = _format_json_for_panel(row["assistant_json"], width=46, max_lines=15)
        io_text = (
            f"Prompt:\n{wrapped_prompt}\n\n"
            f"Target JSON:\n{wrapped_json}\n\n"
            f"Predicate={row['predicate']} | Text={row['text_tier']}"
        )
        axes[i, 2].text(
            0.03,
            0.97,
            io_text,
            ha="left",
            va="top",
            fontsize=8.2,
            family="monospace",
            transform=axes[i, 2].transAxes,
            wrap=True,
        )

    fig.suptitle("Representative Training Records (Train Aug)", fontsize=15)
    fig.subplots_adjust(top=0.97, bottom=0.03, left=0.03, right=0.98, wspace=0.16, hspace=0.3)
    fig.savefig(out_dir / "train_io_gallery.png", dpi=180)
    plt.close(fig)


def analyze(train_canonical: list[dict], eval_canonical: list[dict], train_aug: list[dict]) -> dict:
    train_canonical_pred = _count(train_canonical, "predicate")
    eval_canonical_pred = _count(eval_canonical, "predicate")
    train_aug_pred = _count(train_aug, "predicate")
    train_aug_mod = _count(train_aug, "modality")
    train_aug_scale = _count(train_aug, "scale")
    train_aug_text = _count(train_aug, "text_tier")
    train_aug_base_counts = _count(train_aug, "base_case_id")
    links12 = _cross(train_aug, "modality", "scale")
    links23 = _cross(train_aug, "scale", "text_tier")

    train_canonical_enriched = []
    for row in train_canonical:
        cons = _assistant_constraints(row)
        rels = list(cons.get("spatial_relations", []) or [])
        train_canonical_enriched.append(
            {
                **row,
                "constraints": cons,
                "relations": rels,
                "topology_family": _topology_family(rels),
                "relation_count": len(rels),
                "relation_multiplicity": f"{len(rels)}-rel",
                "anchor_pattern": _anchor_pattern(rels),
                "complexity": round(_complexity_score(rels), 3),
                "assistant_json": next((m.get("content", "") for m in row.get("messages", []) if m.get("role") == "assistant"), ""),
            }
        )

    eval_canonical_enriched = []
    for row in eval_canonical:
        cons = _assistant_constraints(row)
        rels = list(cons.get("spatial_relations", []) or [])
        eval_canonical_enriched.append(
            {
                **row,
                "constraints": cons,
                "relations": rels,
                "topology_family": _topology_family(rels),
                "relation_count": len(rels),
                "relation_multiplicity": f"{len(rels)}-rel",
                "anchor_pattern": _anchor_pattern(rels),
                "complexity": round(_complexity_score(rels), 3),
                "assistant_json": next((m.get("content", "") for m in row.get("messages", []) if m.get("role") == "assistant"), ""),
            }
        )

    train_aug_enriched = []
    for row in train_aug:
        cons = _assistant_constraints(row)
        rels = list(cons.get("spatial_relations", []) or [])
        train_aug_enriched.append(
            {
                **row,
                "constraints": cons,
                "relations": rels,
                "topology_family": _topology_family(rels),
                "relation_count": len(rels),
                "relation_multiplicity": f"{len(rels)}-rel",
                "anchor_pattern": _anchor_pattern(rels),
                "complexity": round(_complexity_score(rels), 3),
                "assistant_json": next((m.get("content", "") for m in row.get("messages", []) if m.get("role") == "assistant"), ""),
            }
        )

    all_split_rows = (
        [{"split": "Train Canonical", **row} for row in train_canonical_enriched]
        + [{"split": "Eval Canonical", **row} for row in eval_canonical_enriched]
        + [{"split": "Train Aug", **row} for row in train_aug_enriched]
    )
    split_counter = _count(all_split_rows, "split")
    scale_counter_all = _count(all_split_rows, "scale")
    text_counter_all = _count(all_split_rows, "text_tier")
    relation_mult_all = _count(all_split_rows, "relation_multiplicity")
    topology_family_all = _count(all_split_rows, "topology_family")
    anchor_pattern_all = _count(all_split_rows, "anchor_pattern")
    split_rel_all = _cross(all_split_rows, "split", "relation_multiplicity")
    rel_family_all = _cross(all_split_rows, "relation_multiplicity", "topology_family")
    family_anchor_all = _cross(all_split_rows, "topology_family", "anchor_pattern")
    anchor_scale_all = _cross(all_split_rows, "anchor_pattern", "scale")
    scale_text_all = _cross(all_split_rows, "scale", "text_tier")

    canonical_family = Counter(r["topology_family"] for r in train_canonical_enriched)
    aug_family = Counter(r["topology_family"] for r in train_aug_enriched)
    canonical_rel_mult = Counter(r["relation_count"] for r in train_canonical_enriched)
    aug_rel_mult = Counter(r["relation_count"] for r in train_aug_enriched)
    representative_rows = _select_train_gallery_rows(train_aug_enriched)

    return {
        "canonical_all_rows": len(train_canonical) + len(eval_canonical),
        "train_canonical_rows": len(train_canonical),
        "eval_canonical_rows": len(eval_canonical),
        "train_aug_rows": len(train_aug),
        "train_aug_unique_bases": len(train_aug_base_counts),
        "train_aug_base_multiplicity": dict(sorted(Counter(train_aug_base_counts.values()).items())),
        "train_canonical_predicate": dict(train_canonical_pred),
        "eval_canonical_predicate": dict(eval_canonical_pred),
        "train_aug_predicate": dict(train_aug_pred),
        "train_aug_modality": dict(train_aug_mod),
        "train_aug_scale": dict(train_aug_scale),
        "train_aug_text_tier": dict(train_aug_text),
        "train_canonical_topology_family": dict(canonical_family),
        "train_aug_topology_family": dict(aug_family),
        "train_canonical_relation_multiplicity": dict(canonical_rel_mult),
        "train_aug_relation_multiplicity": dict(aug_rel_mult),
        "all_split_counts": dict(split_counter),
        "all_scale_counts": dict(scale_counter_all),
        "all_text_tier_counts": dict(text_counter_all),
        "all_relation_multiplicity_counts": dict(relation_mult_all),
        "all_topology_family_counts": dict(topology_family_all),
        "all_anchor_pattern_counts": dict(anchor_pattern_all),
        "all_split_relation_multiplicity": [{"from": k[0], "to": k[1], "count": v} for k, v in sorted(split_rel_all.items())],
        "all_relation_multiplicity_family": [{"from": k[0], "to": k[1], "count": v} for k, v in sorted(rel_family_all.items())],
        "all_family_anchor_pattern": [{"from": k[0], "to": k[1], "count": v} for k, v in sorted(family_anchor_all.items())],
        "all_anchor_scale": [{"from": k[0], "to": k[1], "count": v} for k, v in sorted(anchor_scale_all.items())],
        "all_scale_text": [{"from": k[0], "to": k[1], "count": v} for k, v in sorted(scale_text_all.items())],
        "train_aug_modality_scale": [{"from": k[0], "to": k[1], "count": v} for k, v in sorted(links12.items())],
        "train_aug_scale_text": [{"from": k[0], "to": k[1], "count": v} for k, v in sorted(links23.items())],
        "representative_train_rows": [
            {
                "id": row["id"],
                "base_case_id": row["base_case_id"],
                "modality": row["modality"],
                "scale": row["scale"],
                "text_tier": row["text_tier"],
                "predicate": row["predicate"],
                "topology_family": row["topology_family"],
                "signature": str(_relation_signature(row["relations"])),
            }
            for row in representative_rows
        ],
        "_train_canonical_enriched": train_canonical_enriched,
        "_train_aug_enriched": train_aug_enriched,
        "_representative_rows": representative_rows,
    }


def write_outputs(
    out_dir: Path,
    summary: dict,
    train_canonical: list[dict],
    eval_canonical: list[dict],
    train_aug: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_growth_overview(out_dir, summary)
    _plot_aug_flow(
        out_dir,
        Counter(summary["all_split_counts"]),
        Counter(summary["all_relation_multiplicity_counts"]),
        Counter(summary["all_topology_family_counts"]),
        Counter(summary["all_anchor_pattern_counts"]),
        Counter(summary["all_scale_counts"]),
        Counter(summary["all_text_tier_counts"]),
        Counter({(r["from"], r["to"]): r["count"] for r in summary["all_split_relation_multiplicity"]}),
        Counter({(r["from"], r["to"]): r["count"] for r in summary["all_relation_multiplicity_family"]}),
        Counter({(r["from"], r["to"]): r["count"] for r in summary["all_family_anchor_pattern"]}),
        Counter({(r["from"], r["to"]): r["count"] for r in summary["all_anchor_scale"]}),
        Counter({(r["from"], r["to"]): r["count"] for r in summary["all_scale_text"]}),
    )
    _plot_predicate_dumbbell(
        out_dir,
        Counter(summary["train_canonical_predicate"]),
        Counter(summary["train_aug_predicate"]),
    )
    _plot_base_expansion(out_dir, Counter(_count(train_aug, "base_case_id")))
    _plot_mix_heatmaps(out_dir, train_aug)
    _plot_train_topology_dumbbell(
        out_dir,
        Counter(summary["train_canonical_topology_family"]),
        Counter(summary["train_aug_topology_family"]),
    )
    _plot_train_topology_mix_heatmap(out_dir, summary["_train_aug_enriched"], Counter(summary["train_aug_topology_family"]))
    _plot_train_topology_scatter(out_dir, summary["_train_aug_enriched"], Counter(summary["train_aug_topology_family"]))
    _plot_train_io_gallery(out_dir, summary["_representative_rows"])

    lines = [
        "# LoRA6-AP Dataset Growth & Augmentation Analysis",
        "",
        "## Key Counts",
        "",
        f"- Canonical all: {summary['canonical_all_rows']}",
        f"- Train canonical: {summary['train_canonical_rows']}",
        f"- Eval canonical: {summary['eval_canonical_rows']}",
        f"- Train aug: {summary['train_aug_rows']}",
        f"- Unique train bases in aug: {summary['train_aug_unique_bases']}",
        "",
        "## Augmentation Readout",
        "",
        f"- Modality: {summary['train_aug_modality']}",
        f"- Scale: {summary['train_aug_scale']}",
        f"- Text tier: {summary['train_aug_text_tier']}",
        f"- Base multiplicity: {summary['train_aug_base_multiplicity']}",
        "",
        "## Predicate Growth",
        "",
        f"- Train canonical predicates: {summary['train_canonical_predicate']}",
        f"- Train aug predicates: {summary['train_aug_predicate']}",
        "",
        "## Train Topology Overview",
        "",
        f"- Train canonical topology families: {summary['train_canonical_topology_family']}",
        f"- Train aug topology families: {summary['train_aug_topology_family']}",
        f"- Train canonical relation multiplicity: {summary['train_canonical_relation_multiplicity']}",
        f"- Train aug relation multiplicity: {summary['train_aug_relation_multiplicity']}",
        "",
        "## Visualization Outputs",
        "",
        "- `dataset_growth_overview.png`",
        "- `augmentation_flow_alluvial.png`",
        "- `predicate_growth_dumbbell.png`",
        "- `per_base_expansion_histogram.png`",
        "- `augmentation_mix_heatmaps.png`",
        "- `train_topology_family_dumbbell.png`",
        "- `train_topology_mix_heatmaps.png`",
        "- `train_topology_scatter.png`",
        "- `train_io_gallery.png`",
    ]
    (out_dir / "lora6_ap_growth_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    clean_summary = {k: v for k, v in summary.items() if not k.startswith("_")}
    (out_dir / "lora6_ap_growth_summary.json").write_text(
        json.dumps(clean_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-canonical", type=Path, default=DEFAULT_TRAIN_CANONICAL)
    parser.add_argument("--eval-canonical", type=Path, default=DEFAULT_EVAL_CANONICAL)
    parser.add_argument("--train-aug", type=Path, default=DEFAULT_TRAIN_AUG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    train_canonical = _load_jsonl(args.train_canonical)
    eval_canonical = _load_jsonl(args.eval_canonical)
    train_aug = _load_jsonl(args.train_aug)
    summary = analyze(train_canonical, eval_canonical, train_aug)
    write_outputs(args.out_dir, summary, train_canonical, eval_canonical, train_aug)
    print(f"Wrote dataset growth analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
