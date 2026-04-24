#!/usr/bin/env python3
"""Shared Phase 4 thesis plot palette.

The JSON file is the source of truth for model, metric, strategy, and
annotation colors used by the Phase 4 LoRA6 plot scripts.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PALETTE_PATH = Path(__file__).with_name("phase4_plot_colors.json")


@lru_cache(maxsize=1)
def load_phase4_palette(path: str | Path | None = None) -> dict[str, Any]:
    palette_path = Path(path) if path is not None else PALETTE_PATH
    return json.loads(palette_path.read_text(encoding="utf-8"))


def palette_section(name: str) -> dict[str, Any]:
    section = load_phase4_palette().get(name)
    if not isinstance(section, dict):
        raise KeyError(f"Unknown Phase 4 palette section: {name}")
    return section


def color(section: str, key: str, fallback: str = "#777777") -> str:
    value = palette_section(section).get(key, fallback)
    if isinstance(value, dict):
        return str(value.get("color", fallback))
    return str(value)


MODELS = palette_section("models")
STRATEGIES = palette_section("strategies")
COLORS = {**MODELS, **STRATEGIES}
METRIC_COLORS = palette_section("metrics")
HIGHLIGHT_COLORS = palette_section("highlights")
UNIVERSE_META = palette_section("universes")
FAMILY_TO_UNIVERSE = palette_section("family_to_universe")
RELATION_FAMILY_COLORS = palette_section("relation_families")
FINGERPRINT_WATERFALL_COLORS = palette_section("fingerprint_waterfall")

STRATEGY_META = {
    "p0_only": {"label": "p0_only", "color": STRATEGIES["p0_only"]},
    "p1_only_strategy": {"label": "p1_only", "color": STRATEGIES["p1_only_strategy"]},
    "p0_intersect_p1": {"label": "p0∩p1", "color": STRATEGIES["p0_intersect_p1"]},
    "p0_union_p1": {"label": "p0∪p1", "color": STRATEGIES["p0_union_p1"]},
}

GRAPH_RAG_COLORS = {
    "g7": MODELS["g7_position_context"],
    "g8": MODELS["g8_posctx_dim"],
    "p1": STRATEGIES["p1"],
    "oracle": STRATEGIES["oracle"],
    "fallback": STRATEGIES["fallback"],
}
