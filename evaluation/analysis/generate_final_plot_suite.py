#!/usr/bin/env python3
"""Generate the final thesis plot suite for Chapter 7."""

from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mscd_demo_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from phase4_plot_style import (
    FINGERPRINT_WATERFALL_COLORS,
    GRAPH_RAG_COLORS,
    HIGHLIGHT_COLORS,
    METRIC_COLORS,
    MODELS,
)


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent.parent
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "plots" / "final"
DEFAULT_DATA_DIR = DEFAULT_OUTPUT_DIR / "data"
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
AP_TRACE_ROOT = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "ap_e2e_phase6_1_g9_resnet_band_v2"
    / "traces"
)

ORACLE_FINGERPRINT_SOURCE = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "group4_post-hoc_analysis"
    / "oracle_ceiling"
    / "20260404"
    / "fingerprint_loss_by_level.csv"
)
LORA_VS_GEMINI_SOURCE = (
    PROJECT_ROOT
    / "docs"
    / "plots"
    / "phase4_lora6_main"
    / "fig03_trackB2_strict_downstream_summary.csv"
)
MODALITY_SOURCE = (
    PROJECT_ROOT
    / "docs"
    / "plots"
    / "phase4_lora6_main"
    / "fig09_trackA_modality_ablation_summary.csv"
)
GRAPH_RAG_SOURCE = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "graph_rag_rerank"
    / "phase6_1_g9_resnet_f4_fused"
    / "graph_rag_rerank_summary.json"
)
GRAPH_RAG_RESULTS_SOURCE = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "graph_rag_rerank"
    / "phase6_1_g9_resnet_f4_fused"
    / "graph_rag_rerank_results.jsonl"
)
PHASE6_T2_SUMMARY_SOURCE = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "graph_rag_rerank"
    / "phase6_f4_t2_comparison"
    / "graph_rag_rerank_summary.csv"
)
G7_RETRIEVAL_SUMMARY = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "ap_e2e_phase5_g8"
    / "g7_position_context"
    / "summary_20260407_195114_v2_lora_p0_union_p1.csv"
)
G8_RETRIEVAL_SUMMARY = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "ap_e2e_phase5_g8"
    / "g8_posctx_dim"
    / "summary_20260407_195021_v2_lora_p0_union_p1.csv"
)
GEMINI_V2_RETRIEVAL_SUMMARY = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "ap_e2e_phase5_g8"
    / "gemini_ap_v2"
    / "summary_20260407_235044_v2_lora_p0_union_p1.csv"
)
G9_RESNET_F4_RETRIEVAL_SUMMARY = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "ap_e2e_phase6_1_g9_resnet_f4"
    / "summary_20260429_094931_v2_lora_p0_union_p1.csv"
)
G9_LORA_ONLY_RETRIEVAL_SUMMARY = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "ap_e2e_phase5_g9"
    / "summary_20260422_213416_v2_lora_p0_union_p1.csv"
)
G9_RESNET_BAND_RETRIEVAL_SUMMARY = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "ap_e2e_phase6_1_g9_resnet_band_v2"
    / "summary_20260429_034720_v2_lora_p0_union_p1.csv"
)
LORA2_ATTR_SUMMARY = (
    PROJECT_ROOT
    / "output"
    / "ap_lora2_vs_lora5_floorplan_only"
    / "fig13_lora2_vs_lora5_fp_ap_summary.csv"
)
ORACLE_TOPOLOGY_METRICS = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "legacy"
    / "oracle_ap_heldout"
    / "oracle_topology_metrics.json"
)
G7_EVAL_SOURCE = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "g7_position_context__ap_eval.jsonl"
GEMINI_V2_EVAL_SOURCE = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "gemini_ap_v2__ap_eval.jsonl"
G9_LORA_ONLY_EVAL_SOURCE = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "g9_opencv_cluster__ap_eval.jsonl"
G9_RESNET_BAND_EVAL_SOURCE = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "g9_resnet_band__ap_eval.jsonl"
G9_F4_RESNET_EVAL_SOURCE = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "g9_resnet_band_f4__ap_eval.jsonl"
G3_EVAL_SOURCE = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "g3_fullaug_r32__ap_eval.jsonl"
GT_G7_LABEL_SOURCE = REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap" / "train" / "lora6_v2_ap_eval_canonical_m_g7.jsonl"
GT_G9_LABEL_SOURCE = REPO_ROOT / "data_curation" / "datasets" / "synth_v0.5_ap" / "train" / "lora6_v2_ap_eval_canonical_m_g9.jsonl"
SIZE_CLUSTER_TAXONOMY_SOURCE = PROJECT_ROOT / "prompts" / "size_cluster_taxonomy.json"
CLUSTER_CLASSIFIER_TEST_METRICS = PROJECT_ROOT / "models" / "cluster_classifier_ap" / "test_metrics.json"

FIGSIZE_4X3 = (12.0, 9.0)
FIGSIZE_4X3_TIGHT = (10.8, 8.1)
FIGSIZE_2X1 = (14.0, 7.0)

MAIN_FIGURE_IDS = (
    "fig00_symbolic_reasoning_trace",
    "fig00c_p0_topk_flow",
    "fig01_oracle_symbolic_ceiling",
    "fig03_lora_vs_gemini",
    "fig04_multimodal_alignment_gain",
    "fig05_graph_rag_evidence_dependent",
    "fig06_summary_findings_table",
    "fig07_retrieval_pipeline_comparison",
)
FIGURE_IDS = MAIN_FIGURE_IDS + (
    "backup_model_capability_proof",
)

ORACLE_LEVEL_ORDER = [
    "L0_p1_only",
    "L1_pred_obj",
    "L2_pred_obj_dir",
    "L3_pred_obj_dir_sub",
    "L4_full_fingerprint",
]
ORACLE_LEVEL_META = {
    "L0_p1_only": {"display_level": "L0", "level_name": "Storey + class", "fields_active": "storey, IFC class"},
    "L1_pred_obj": {"display_level": "L1", "level_name": "+ relation + target", "fields_active": "relation, target type"},
    "L2_pred_obj_dir": {
        "display_level": "L2",
        "level_name": "+ direction",
        "fields_active": "relation, target type, direction",
    },
    "L3_pred_obj_dir_sub": {
        "display_level": "L3",
        "level_name": "+ subtype",
        "fields_active": "relation, target type, direction, subtype",
    },
    "L4_full_fingerprint": {
        "display_level": "L4",
        "level_name": "+ position slot",
        "fields_active": "relation, target type, direction, subtype, position slot",
    },
}
ORACLE_SHORT_LABELS = {
    "L0": "Storey +\nclass",
    "L1": "+ relation\n+ target",
    "L2": "+ direction",
    "L3": "+ subtype",
    "L4": "+ position slot",
}

P0_INLINE = "P0 (spatial relation)"
P1_INLINE = "P1 (storey + IFC class)"
P1_ONLY_INLINE = "P1-only (storey + IFC class)"
P0_SHORT_LABEL = "P0\nspatial\nrelation"
P1_SHORT_LABEL = "P1\nstorey +\nIFC class"
P1_RERANK_SHORT_LABEL = "P1\nstorey + IFC class\n+ rerank"
G3_SHORT_LABEL = "G3"
G4_SHORT_LABEL = "G4"
G7_SHORT_LABEL = "G7\nposition\ncontext"
G8_SHORT_LABEL = "G8\nposition\ncontext"
G9_BASE_SHORT_LABEL = "G9\nOpenCV/\nResNet cues"
G9_RERANK_SHORT_LABEL = "G9\nOpenCV/ResNet\n+ rerank"
G9_LORA_SHORT_LABEL = "G9\nLoRA-only"

G9_BASE_COLOR = MODELS.get("g9_resnet_f4", "#0F766E")
G9_LORA_ONLY_COLOR = MODELS.get("g9_lora_only", "#F97316")
G9_RERANK_COLOR = MODELS.get("g9_opencv/resnet_rerank", "#310b66")


def _pretty_strategy_name(name: str | None) -> str:
    if not name:
        return "spatial relation chain"
    return name.replace("_", " ")

FIG03_BASE_ORDER = [
    "LoRA5-r32 FP",
    "G3 FullAug r32 (MM)",
    "G4 Ultimate (MM)",
]
FIG03_PLOT_ORDER = [
    "LoRA5-r32 FP",
    "Gemini AP (MM)",
    "G3 FullAug r32 (MM)",
    "G4 Ultimate (MM)",
    "G7 Position Context (MM)",
    "G9 + OpenCV F4 + ResNet",
    "Oracle ceiling",
]
FIG03_LABELS = {
    "LoRA5-r32 FP": "LoRA5\nquery\nfields",
    "Gemini AP (MM)": "Gemini",
    "G3 FullAug r32 (MM)": G3_SHORT_LABEL,
    "G4 Ultimate (MM)": G4_SHORT_LABEL,
    "G7 Position Context (MM)": G7_SHORT_LABEL,
    "G9 + OpenCV F4 + ResNet": G9_BASE_SHORT_LABEL,
    "Oracle ceiling": "Oracle",
}
FIG03_COLORS = {
    "LoRA5-r32 FP": MODELS.get("lora5_fp", "#FB8C00"),
    "Gemini AP (MM)": MODELS.get("gemini_ap_v2", "#1565C0"),
    "G3 FullAug r32 (MM)": MODELS.get("g3_fullaug_r32", "#D32F2F"),
    "G4 Ultimate (MM)": MODELS.get("g4_ultimate", "#B71C1C"),
    "G7 Position Context (MM)": MODELS.get("g7_position_context", "#6A1B9A"),
    "G9 + OpenCV F4 + ResNet": G9_BASE_COLOR,
    "Oracle ceiling": GRAPH_RAG_COLORS.get("oracle", "#4A148C"),
}
FIG03_TIGHT_ORDER = [
    "Gemini AP (MM)",
    "LoRA2 attr",
    "G3 key",
    "G9 + OpenCV F4 + ResNet + Graph-RAG",
    "Oracle ceiling",
]
FIG03_TIGHT_LABELS = {
    "Gemini AP (MM)": "Gemini",
    "LoRA2 attr": "LoRA2\nattribute-only",
    "G3 key": "G3\nspatial\nquery fields",
    "G9 + OpenCV F4 + ResNet + Graph-RAG": G9_RERANK_SHORT_LABEL,
    "Oracle ceiling": "Oracle",
}
FIG03_TIGHT_COLORS = {
    "Gemini AP (MM)": MODELS.get("gemini_ap_v2", "#1565C0"),
    "LoRA2 attr": MODELS.get("lora2_fp", "#78909C"),
    "G3 key": MODELS.get("g3_fullaug_r32", "#D32F2F"),
    "G9 + OpenCV F4 + ResNet + Graph-RAG": G9_RERANK_COLOR,
    "Oracle ceiling": HIGHLIGHT_COLORS.get("edge_dark", "#252525"),
}

FIG04_SLICE_ORDER = ["MC", "MC4D", "FP", "SITE", "FPSITE", "MA"]
FIG04_SLICE_LABELS = {
    "MC": "Site + FP + Chat",
    "MC4D": "Site + FP + Chat + 4D",
    "FP": "FP + Chat",
    "SITE": "Site + Chat",
    "FPSITE": "Visual only",
    "MA": "Chat only",
}
FIG04_MODALITY_MODELS = ["g7_position_context", "g8_posctx_dim", "gemini_ap_v2"]
FIG04_MODALITY_LABELS = {
    "g7_position_context": G7_SHORT_LABEL,
    "g8_posctx_dim": G8_SHORT_LABEL,
    "gemini_ap_v2": "Gemini",
}
FIG04_MODALITY_COLORS = {
    "g7_position_context": MODELS.get("g7_position_context", "#6A1B9A"),
    "g8_posctx_dim": MODELS.get("g8_posctx_dim", "#3E1080"),
    "gemini_ap_v2": MODELS.get("gemini_ap_v2", "#1565C0"),
}

FIELD_MODEL_ORDER = ["Gemini", "G7", "G9 + OpenCV F4 + ResNet"]
FIELD_METRIC_ORDER = [
    "class_acc",
    "storey_acc",
    "hop1_acc",
    "spatial_set_acc",
    "predicate_recall",
    "direction_acc",
    "position_emission",
    "position_exact",
    "size_cluster_exact",
]
FIELD_METRIC_LABELS = {
    "class_acc": "Class",
    "storey_acc": "Storey",
    "hop1_acc": "1-hop",
    "spatial_set_acc": "All SR",
    "predicate_recall": "Predicate",
    "direction_acc": "Direction",
    "position_emission": "Pos emit",
    "position_exact": "Pos exact",
    "size_cluster_exact": "Size exact",
}
FIELD_MODEL_COLORS = {
    "Gemini": MODELS.get("gemini_ap_v2", "#1565C0"),
    "G7": MODELS.get("g7_position_context", "#6A1B9A"),
    "G9 + OpenCV F4 + ResNet": G9_BASE_COLOR,
}
BACKUP_CAPABILITY_MODEL_ORDER = [
    "Gemini",
    "G7",
    "G9 LoRA-only",
    "G9 + ResNet",
]
BACKUP_CAPABILITY_MODEL_COLORS = {
    "Gemini": MODELS.get("gemini_ap_v2", "#94A3B8"),
    "G7": MODELS.get("g7_position_context", "#7C3AED"),
    "G9 LoRA-only": G9_LORA_ONLY_COLOR,
    "G9 + ResNet": G9_BASE_COLOR,
}

FIG05_ORDER = [
    "P1-only control",
    "P1-only + Graph-RAG",
    "G9 + OpenCV F4 + ResNet",
    "G9 + OpenCV F4 + ResNet + Graph-RAG",
]
FIG05_LABELS = {
    "P1-only control": P1_SHORT_LABEL,
    "P1-only + Graph-RAG": P1_RERANK_SHORT_LABEL,
    "G9 + OpenCV F4 + ResNet": G9_BASE_SHORT_LABEL,
    "G9 + OpenCV F4 + ResNet + Graph-RAG": G9_RERANK_SHORT_LABEL,
}
FIG05_COLORS = {
    "P1-only control": GRAPH_RAG_COLORS.get("p1", "#F5A623"),
    "P1-only + Graph-RAG": G9_LORA_ONLY_COLOR,
    "G9 + OpenCV F4 + ResNet": G9_BASE_COLOR,
    "G9 + OpenCV F4 + ResNet + Graph-RAG": G9_RERANK_COLOR,
}

FIG07_ORDER = [
    "G8 + OpenCV F4 baseline",
    "P1-only control",
    "G7 full topology",
    "G9 + OpenCV F4 + ResNet",
    "G9 + OpenCV F4 + ResNet + Graph-RAG",
]
FIG07_LABELS = {
    "G8 + OpenCV F4 baseline": G8_SHORT_LABEL,
    "P1-only control": P1_SHORT_LABEL,
    "G7 full topology": G7_SHORT_LABEL,
    "G9 + OpenCV F4 + ResNet": G9_BASE_SHORT_LABEL,
    "G9 + OpenCV F4 + ResNet + Graph-RAG": G9_RERANK_SHORT_LABEL,
}
FIG07_COLORS = {
    "G8 + OpenCV F4 baseline": MODELS.get("g8_posctx_dim", "#3E1080"),
    "P1-only control": GRAPH_RAG_COLORS.get("p1", "#F5A623"),
    "G7 full topology": MODELS.get("g7_position_context", "#6A1B9A"),
    "G9 + OpenCV F4 + ResNet": G9_BASE_COLOR,
    "G9 + OpenCV F4 + ResNet + Graph-RAG": G9_RERANK_COLOR,
}

SUMMARY_ROWS = [
    {
        "section": "Oracle ceiling",
        "finding": "Correct symbolic constraints preserve the right IFC element and compress the pool from median 76 to 1.",
        "interpretation": "The graph backend is viable when the extracted fields are trustworthy.",
        "rq_link": "RQ2",
    },
    {
        "section": "Learned extraction",
        "finding": "Recovered Gemini improves over the older zero-shot row, but G-series adapters still dominate strict retrieval quality.",
        "interpretation": "AEC spatial language remains supervision-sensitive rather than purely prompt-limited.",
        "rq_link": "RQ1",
    },
    {
        "section": "Multimodal grounding",
        "finding": "One-hop topology relies on text plus floorplan/site cues; the G9 OpenCV/ResNet pipeline adds position-context emission and size-aware evidence.",
        "interpretation": "The system behaves as text-grounded multimodal topology extraction, not generic scene understanding.",
        "rq_link": "RQ1",
    },
    {
        "section": "Deterministic helpers",
        "finding": "OpenCV ordinal cues and ResNet size bands help most as auxiliary signals rather than brittle hard filters.",
        "interpretation": "Noisy visual fields should support ranking and verification, not destructive retrieval cuts.",
        "rq_link": "RQ1|RQ2",
    },
    {
        "section": "Graph-RAG reranking",
        "finding": "Reranking helps both the P1 storey/class control and the G9 OpenCV/ResNet pipeline; the G9 trace lifts Top-1 from 6.7% to 8.3% and rescues 3/12 near-misses.",
        "interpretation": "Graph-RAG helps most when the correct candidate is already in the pool but misordered.",
        "rq_link": "RQ2",
    },
    {
        "section": "Design principle",
        "finding": "Stable fields should filter, uncertain fields should rerank, and low-confidence cues should remain auxiliary.",
        "interpretation": "The contribution is a field-routed neuro-symbolic interpreter layer.",
        "rq_link": "RQ1|RQ2",
    },
]


@dataclass(frozen=True)
class FigureArtifact:
    figure_id: str
    title: str
    claim: str
    rq: tuple[str, ...]
    png_path: Path
    pdf_path: Path
    data_path: Path
    sources: tuple[Path, ...]


@dataclass(frozen=True)
class SymbolicTraceSelection:
    case_id: str
    trace_path: Path
    query_text: str
    gt_guid: str
    gt_name: str
    gt_storey: str
    constraints: dict[str, Any]
    query_plan_used: dict[str, Any]
    strategy_used: str
    initial_pool_size: int
    p0_pool_size: int
    p1_pool_size: int
    union_pool_size: int
    topk_pool_size: int
    base_rank: int | None
    reranked_rank: int | None
    candidate_guids: list[str]
    reranked_candidate_guids: list[str]
    top5_guids: list[str]
    planner_expected_p0: int | None
    planner_expected_p1: int | None
    topology_relations: list[dict[str, Any]]
    target_name_keyword: str
    position_context: str
    size_cluster: str
    trace_run_dir: Path
    case_record: dict[str, Any] | None


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _case_id_from_row(row: dict[str, Any]) -> str | None:
    case_id = row.get("case_id") or row.get("id") or row.get("scenario_id")
    return str(case_id) if case_id else None


def _load_case_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        case_id = _case_id_from_row(row)
        if case_id:
            index[case_id] = row
    return index


def _latest_trace_run_dir(trace_root: Path) -> Path:
    run_dirs = [path for path in trace_root.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No trace run directories found under {trace_root}")
    return max(run_dirs, key=lambda path: path.name)


def _rank_of_guid(guid_list: Iterable[str], target_guid: str) -> int | None:
    for idx, guid in enumerate(guid_list, start=1):
        if guid == target_guid:
            return idx
    return None


def _truncate_guid(guid: str) -> str:
    if len(guid) <= 16:
        return guid
    return f"{guid[:8]}...{guid[-4:]}"


def _pretty_ifc_class(name: str) -> str:
    return name.replace("Ifc", "") if name.startswith("Ifc") else name


def _relation_display_line(rel: dict[str, Any]) -> str:
    predicate = str(rel.get("predicate") or "?")
    object_type = _pretty_ifc_class(str(rel.get("object_type") or ""))
    object_subtype = str(rel.get("object_subtype") or "")
    direction = str(rel.get("direction") or "")
    detail = object_subtype or object_type or "context"
    line = f"{predicate} {detail}"
    if direction:
        line += f" ({direction})"
    return line


def _summarize_relations(relations: list[dict[str, Any]], limit: int = 3) -> str:
    lines = [_relation_display_line(rel) for rel in relations[:limit]]
    if len(relations) > limit:
        lines.append(f"+{len(relations) - limit} more")
    return "\n".join(f"- {line}" for line in lines) if lines else "- none"


def _planner_expected_pool(query_plans: list[dict[str, Any]], strategy_name: str) -> int | None:
    for plan in query_plans:
        if str(plan.get("strategy") or "") == strategy_name:
            return _safe_int(plan.get("expected_pool_size"), default=0)
    return None


def _select_symbolic_reasoning_case() -> SymbolicTraceSelection:
    case_index = _load_case_index(DEFAULT_CASES)
    rerank_rows = {row["case_id"]: row for row in _load_jsonl(GRAPH_RAG_RESULTS_SOURCE) if row.get("case_id")}
    trace_run_dir = _latest_trace_run_dir(AP_TRACE_ROOT)
    trace_paths = sorted(trace_run_dir.glob("*.trace.json"))
    if not trace_paths:
        raise FileNotFoundError(f"No trace files found in latest trace run: {trace_run_dir}")

    best_score = float("-inf")
    best_payload: SymbolicTraceSelection | None = None
    for trace_path in trace_paths:
        trace_obj = _load_json(trace_path)
        internals = trace_obj.get("internals") or {}
        query_plans = internals.get("query_plans") or []
        retrieval_results = internals.get("retrieval_results") or []
        constraints = internals.get("constraints") or {}
        scenario = trace_obj.get("scenario") or {}
        ground_truth = scenario.get("ground_truth") or {}
        case_id = str(trace_obj.get("scenario_id") or scenario.get("id") or trace_path.stem)
        if not query_plans or not retrieval_results or not ground_truth.get("target_guid"):
            continue

        retrieval = retrieval_results[0] if isinstance(retrieval_results[0], dict) else {}
        candidates = retrieval.get("candidates") or []
        candidate_guids = [str(candidate.get("guid") or "") for candidate in candidates if candidate.get("guid")]
        if not candidate_guids:
            continue

        gt_guid = str(ground_truth.get("target_guid") or "")
        base_rank = _rank_of_guid(candidate_guids, gt_guid)
        if base_rank is None:
            continue

        relations = _normalize_spatial_relations(constraints.get("spatial_relations"))
        direction_count = sum(1 for rel in relations if rel.get("direction"))
        rerank_row = rerank_rows.get(case_id, {})
        reranked_guids = [str(guid) for guid in rerank_row.get("reranked_topk_guids") or [] if guid]
        reranked_rank = _safe_int(rerank_row.get("reranked_rank"), default=0) or None
        initial_pool_size = _safe_int(trace_obj.get("initial_pool_size"))
        p0_pool_size = _safe_int(retrieval.get("raw_pool_size"), default=len(candidate_guids))
        union_pool_size = _safe_int(retrieval.get("pool_size"), default=_safe_int(trace_obj.get("final_pool_size"), default=len(candidate_guids)))
        query_plan_used = retrieval.get("query_plan_used") or query_plans[0]
        strategy_used = str(retrieval.get("strategy_actually_used") or query_plan_used.get("strategy") or "")
        p1_pool_size = union_pool_size if "storey+type" in strategy_used else max(union_pool_size, p0_pool_size)

        score = 0.0
        score += min(initial_pool_size / max(union_pool_size, 1), 300.0) * 0.08
        score += min(max(union_pool_size - p0_pool_size, 0), 25)
        score += len(relations) * 6
        score += direction_count * 4
        if constraints.get("position_context"):
            score += 12
        if constraints.get("size_cluster") or query_plan_used.get("params", {}).get("target_size_cluster"):
            score += 7
        if reranked_rank is not None and reranked_rank <= 5:
            score += 55
        elif base_rank <= 5:
            score += 35
        if reranked_rank is not None and reranked_rank < base_rank:
            score += 20
        score -= union_pool_size * 0.04

        top5_source = reranked_guids or candidate_guids
        top5_guids = top5_source[:5]
        payload = SymbolicTraceSelection(
            case_id=case_id,
            trace_path=trace_path,
            query_text=str(scenario.get("query_text") or (case_index.get(case_id, {}).get("query_text")) or ""),
            gt_guid=gt_guid,
            gt_name=str(ground_truth.get("target_name") or (case_index.get(case_id, {}).get("ground_truth", {}) or {}).get("target_name") or ""),
            gt_storey=str(ground_truth.get("target_storey") or (case_index.get(case_id, {}).get("ground_truth", {}) or {}).get("target_storey") or ""),
            constraints=constraints,
            query_plan_used=query_plan_used,
            strategy_used=strategy_used,
            initial_pool_size=initial_pool_size,
            p0_pool_size=p0_pool_size,
            p1_pool_size=p1_pool_size,
            union_pool_size=union_pool_size,
            topk_pool_size=min(10, len(top5_source) if top5_source else len(candidate_guids)),
            base_rank=base_rank,
            reranked_rank=reranked_rank,
            candidate_guids=candidate_guids,
            reranked_candidate_guids=reranked_guids,
            top5_guids=top5_guids,
            planner_expected_p0=_planner_expected_pool(query_plans, "spatial_triplet"),
            planner_expected_p1=_planner_expected_pool(query_plans, "storey+type"),
            topology_relations=relations,
            target_name_keyword=str(constraints.get("target_name_keyword") or ""),
            position_context=str(constraints.get("position_context") or ""),
            size_cluster=str(constraints.get("size_cluster") or query_plan_used.get("params", {}).get("target_size_cluster") or ""),
            trace_run_dir=trace_run_dir,
            case_record=case_index.get(case_id),
        )
        if score > best_score:
            best_score = score
            best_payload = payload

    if best_payload is None:
        raise RuntimeError("Could not find a valid held-out case for fig00_symbolic_reasoning_trace")
    return best_payload


def _require_columns(rows: list[dict[str, Any]], columns: Iterable[str], context: str) -> None:
    if not rows:
        raise ValueError(f"{context}: expected at least one row")
    missing = [col for col in columns if col not in rows[0]]
    if missing:
        raise KeyError(f"{context}: missing required columns {missing}")


def _save_figure(fig: plt.Figure, base_path: Path) -> tuple[Path, Path]:
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {}
    if base_path.name != "fig07_retrieval_pipeline_comparison":
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(png_path, **save_kwargs)
    fig.savefig(pdf_path, **save_kwargs)
    plt.close(fig)
    return png_path, pdf_path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _annotate_bars(ax: plt.Axes, bars: Iterable[Any], values: Iterable[float], dy: float, fmt: str = ".1f") -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(bar.get_height()) + dy,
            format(float(value), fmt),
            ha="center",
            va="bottom",
            fontsize=8.8,
        )


def _safe_ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _parse_gt_in_pool(text: str) -> float:
    pct = text.split("(")[-1].rstrip(")")
    return float(pct.rstrip("%"))


def _load_summary_metrics(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        in_metrics = False
        for row in reader:
            if not row:
                continue
            if row[0].startswith("=== OVERALL METRICS"):
                in_metrics = True
                continue
            if row[0].startswith("=== V2 DIAGNOSTIC"):
                break
            if not in_metrics or row[0] == "Metric":
                continue
            if len(row) >= 2:
                metrics[row[0]] = row[1]
    return metrics


def _extract_assistant_label(case: dict[str, Any]) -> dict[str, Any]:
    for message in case.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return json.loads(content)
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return json.loads(first["text"])
    raise ValueError(f"Assistant label missing for eval case: {case.get('id')}")


def _load_gt_labels(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    case_order: list[str] = []
    gt_by_case: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        case_id = row.get("id") or row.get("base_case_id") or row.get("case_id")
        if not case_id:
            raise ValueError(f"Missing case id in GT row from {path}")
        gt_by_case[case_id] = _extract_assistant_label(row)
        case_order.append(case_id)
    return case_order, gt_by_case


def _normalize_spatial_relations(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _relation_predicate(rel: Any) -> str:
    if isinstance(rel, dict):
        pred = rel.get("predicate", "")
        return pred if isinstance(pred, str) else ""
    return ""


def _relation_signature(rel: Any) -> tuple[str, str, str]:
    if not isinstance(rel, dict):
        return ("", "", "")
    return (
        str(rel.get("predicate") or ""),
        str(rel.get("object_type") or ""),
        str(rel.get("direction") or ""),
    )


def _prepare_fig00_rows() -> tuple[list[dict[str, Any]], dict[str, Any], SymbolicTraceSelection]:
    selection = _select_symbolic_reasoning_case()
    count_rows = [
        {
            "stage": "all_ifc",
            "stage_label": "All IFC elements",
            "count": selection.initial_pool_size,
            "detail": "full IFC graph",
        },
        {
            "stage": "p1_backstop",
            "stage_label": "P1 storey + IFC class backstop",
            "count": selection.p1_pool_size,
            "detail": "storey + IFC class valid pool",
        },
        {
            "stage": "p0_topology",
            "stage_label": "P0 spatial relation",
            "count": selection.p0_pool_size,
            "detail": "spatial relation chain match",
        },
        {
            "stage": "union_pool",
            "stage_label": "P0 spatial relation ∪ P1 backstop",
            "count": selection.union_pool_size,
            "detail": "recall-preserving valid pool",
        },
        {
            "stage": "top5",
            "stage_label": "Top-5 shortlist",
            "count": len(selection.top5_guids),
            "detail": "final ranked GUIDs shown",
        },
    ]
    case_payload = {
        "case_id": selection.case_id,
        "trace_path": _repo_rel(selection.trace_path),
        "trace_run_dir": _repo_rel(selection.trace_run_dir),
        "query_text": selection.query_text,
        "gt_guid": selection.gt_guid,
        "gt_name": selection.gt_name,
        "gt_storey": selection.gt_storey,
        "constraints": selection.constraints,
        "query_plan_used": selection.query_plan_used,
        "strategy_used": selection.strategy_used,
        "initial_pool_size": selection.initial_pool_size,
        "p0_pool_size": selection.p0_pool_size,
        "p1_pool_size": selection.p1_pool_size,
        "union_pool_size": selection.union_pool_size,
        "base_rank": selection.base_rank,
        "reranked_rank": selection.reranked_rank,
        "candidate_guids": selection.candidate_guids,
        "reranked_candidate_guids": selection.reranked_candidate_guids,
        "top5_guids": selection.top5_guids,
        "planner_expected_p0": selection.planner_expected_p0,
        "planner_expected_p1": selection.planner_expected_p1,
        "position_context": selection.position_context,
        "size_cluster": selection.size_cluster,
        "topology_relations": selection.topology_relations,
    }
    return count_rows, case_payload, selection


def _prepare_fig00c_rows(selection: SymbolicTraceSelection) -> list[dict[str, Any]]:
    p0 = selection.p0_pool_size
    p1_total = selection.p1_pool_size
    union = selection.union_pool_size
    p1_only = max(union - p0, 0)
    top_k = min(10, len(selection.reranked_candidate_guids) or len(selection.candidate_guids) or 10)
    p0_in_topk = min(p0, top_k)
    p1_only_in_topk = max(top_k - p0_in_topk, 0)
    p1_only_dropped = max(p1_only - p1_only_in_topk, 0)
    top5 = min(5, len(selection.top5_guids))
    return [
        {"stage": "all_ifc", "total": selection.initial_pool_size, "p0": 0, "p1_only": 0, "dropped": selection.initial_pool_size - union},
        {"stage": "p1_valid", "total": union, "p0": p0, "p1_only": p1_only, "dropped": 0},
        {"stage": "topk_input", "total": top_k, "p0": p0_in_topk, "p1_only": p1_only_in_topk, "dropped": p1_only_dropped},
        {"stage": "top5", "total": top5, "p0": 0, "p1_only": 0, "dropped": max(top_k - top5, 0)},
    ]


def _prepare_oracle_rows(slice_name: str) -> list[dict[str, Any]]:
    rows = _load_csv(ORACLE_FINGERPRINT_SOURCE)
    _require_columns(
        rows,
        [
            "slice",
            "level",
            "coverage",
            "n_total_cases",
            "n_applicable_cases",
            "avg_pool",
            "median_pool",
            "top10_rate",
            "top1_rate",
        ],
        "oracle fingerprint source",
    )
    by_level = {row["level"]: row for row in rows if row["slice"] == slice_name}
    prepared: list[dict[str, Any]] = []
    for level_key in ORACLE_LEVEL_ORDER:
        src = by_level[level_key]
        meta = ORACLE_LEVEL_META[level_key]
        prepared.append(
            {
                "level": meta["display_level"],
                "level_code": level_key,
                "level_name": meta["level_name"],
                "fields_active": meta["fields_active"],
                "coverage": round(float(src["coverage"]) * 100.0, 1),
                "n_cases": int(src["n_total_cases"]),
                "n_cases_covered": int(src["n_applicable_cases"]),
                "top10": round(float(src["top10_rate"]) * 100.0, 1),
                "top1": round(float(src["top1_rate"]) * 100.0, 1),
                "median_pool": round(float(src["median_pool"]), 3),
                "avg_pool": round(float(src["avg_pool"]), 3),
            }
        )
    return prepared


def _summary_row_from_metrics(system: str, display_name: str, path: Path, source_trace: str) -> dict[str, Any]:
    metrics = _load_summary_metrics(path)
    return {
        "system": system,
        "display_name": display_name,
        "top10": round(float(metrics["Top-10 Accuracy"]) * 100.0, 1),
        "top1": round(float(metrics["Top-1 Accuracy"]) * 100.0, 1),
        "mrr10": round(float(metrics["MRR@10"]), 4),
        "gt_in_pool": round(_parse_gt_in_pool(metrics["GT-in-Pool"]), 1),
        "source_trace": source_trace,
    }


def _prepare_fig03_rows() -> list[dict[str, Any]]:
    base_rows = _load_csv(LORA_VS_GEMINI_SOURCE)
    _require_columns(base_rows, ["system", "top10", "top1", "mrr10", "gt_in_pool"], "fig03 source")
    selected = {row["system"]: row for row in base_rows if row["system"] in FIG03_BASE_ORDER}

    prepared = [
        {
            "system": system,
            "display_name": FIG03_LABELS[system],
            "top10": round(float(selected[system]["top10"]), 1),
            "top1": round(float(selected[system]["top1"]), 1),
            "mrr10": round(float(selected[system]["mrr10"]), 4),
            "gt_in_pool": round(float(selected[system]["gt_in_pool"]), 1),
            "source_trace": _repo_rel(LORA_VS_GEMINI_SOURCE),
        }
        for system in FIG03_BASE_ORDER
    ]
    prepared.insert(
        1,
        _summary_row_from_metrics(
            "Gemini AP (MM)",
            FIG03_LABELS["Gemini AP (MM)"],
            GEMINI_V2_RETRIEVAL_SUMMARY,
            _repo_rel(GEMINI_V2_RETRIEVAL_SUMMARY),
        ),
    )
    prepared.append(
        _summary_row_from_metrics(
            "G7 Position Context (MM)",
            FIG03_LABELS["G7 Position Context (MM)"],
            G7_RETRIEVAL_SUMMARY,
            _repo_rel(G7_RETRIEVAL_SUMMARY),
        )
    )
    prepared.append(
        _summary_row_from_metrics(
            "G9 + OpenCV F4 + ResNet",
            FIG03_LABELS["G9 + OpenCV F4 + ResNet"],
            G9_RESNET_F4_RETRIEVAL_SUMMARY,
            _repo_rel(G9_RESNET_F4_RETRIEVAL_SUMMARY),
        )
    )

    oracle = _load_json(ORACLE_TOPOLOGY_METRICS)["overall"]["full_topology_union"]["overall"]
    prepared.append(
        {
            "system": "Oracle ceiling",
            "display_name": FIG03_LABELS["Oracle ceiling"],
            "top10": round(float(oracle["top10_pct"]), 1),
            "top1": round(float(oracle["top1_pct"]), 1),
            "mrr10": round(float(oracle["mrr"]), 4),
            "gt_in_pool": round(float(oracle["gt_in_pct"]), 1),
            "source_trace": _repo_rel(ORACLE_TOPOLOGY_METRICS),
        }
    )

    by_system = {row["system"]: row for row in prepared}
    ordered = [by_system[system] for system in FIG03_PLOT_ORDER]
    learned_rows = [row for row in ordered if row["system"] != "Oracle ceiling"]
    best_system = max(learned_rows, key=lambda row: (float(row["mrr10"]), float(row["top1"]), float(row["top10"])))["system"]
    for row in ordered:
        row["is_best"] = row["system"] == best_system
    return ordered


def _prepare_fig03_tight_rows() -> list[dict[str, Any]]:
    base_rows = {row["system"]: dict(row) for row in _prepare_fig03_rows()}
    graph_rows = {row["system"]: row for row in _prepare_fig05_rows()}
    lora2_rows = _load_csv(LORA2_ATTR_SUMMARY)
    _require_columns(lora2_rows, ["system", "top10", "top1", "mrr10", "gt_in_pool"], "lora2 attr summary")
    lora2_row = next(row for row in lora2_rows if row["system"] == "LoRA2 FP")

    base_rows["LoRA2 attr"] = {
        "system": "LoRA2 attr",
        "display_name": FIG03_TIGHT_LABELS["LoRA2 attr"],
        "top10": round(float(lora2_row["top10"]), 1),
        "top1": round(float(lora2_row["top1"]), 1),
        "mrr10": round(float(lora2_row["mrr10"]), 4),
        "gt_in_pool": round(float(lora2_row["gt_in_pool"]), 1),
        "source_trace": _repo_rel(LORA2_ATTR_SUMMARY),
    }
    g3_base = base_rows.pop("G3 FullAug r32 (MM)")
    base_rows["G3 key"] = {
        **g3_base,
        "system": "G3 key",
        "display_name": FIG03_TIGHT_LABELS["G3 key"],
    }
    g9_rerank = graph_rows["G9 + OpenCV F4 + ResNet + Graph-RAG"]
    base_rows["G9 + OpenCV F4 + ResNet + Graph-RAG"] = {
        "system": "G9 + OpenCV F4 + ResNet + Graph-RAG",
        "display_name": FIG03_TIGHT_LABELS["G9 + OpenCV F4 + ResNet + Graph-RAG"],
        "top10": float(g9_rerank["top10"]),
        "top1": float(g9_rerank["top1"]),
        "mrr10": float(g9_rerank["mrr10"]),
        "gt_in_pool": float(base_rows["G9 + OpenCV F4 + ResNet"]["gt_in_pool"]),
        "source_trace": g9_rerank["source_trace"],
        "rerank_delta_top1_pp": round(float(g9_rerank["top1"]) - float(base_rows["G9 + OpenCV F4 + ResNet"]["top1"]), 1),
    }

    ordered: list[dict[str, Any]] = []
    for system in FIG03_TIGHT_ORDER:
        row = dict(base_rows[system])
        row["display_name"] = FIG03_TIGHT_LABELS[system]
        ordered.append(row)

    learned_rows = [row for row in ordered if row["system"] != "Oracle ceiling"]
    best_system = max(learned_rows, key=lambda row: (float(row["mrr10"]), float(row["top1"]), float(row["top10"])))["system"]
    for row in ordered:
        row["is_best"] = row["system"] == best_system
    return ordered


def _score_field_metrics(pred_path: Path, gt_path: Path) -> dict[str, float]:
    case_order, gt_by_case = _load_gt_labels(gt_path)
    preds = {
        row.get("case_id"): (row.get("constraints") or {})
        for row in _load_jsonl(pred_path)
        if row.get("case_id")
    }

    n_total = len(case_order)
    n_class_match = 0
    n_storey_match = 0
    n_spatial_total = 0
    n_spatial_match = 0
    n_spatial_set_match = 0
    n_pred_match = 0
    n_pred_total = 0
    n_direction_match = 0
    n_direction_total = 0
    n_size_cluster_gt = 0
    n_size_cluster_pred = 0
    n_size_cluster_match = 0
    n_position_context_gt = 0
    n_position_context_pred = 0
    n_position_context_match = 0

    for case_id in case_order:
        gt = gt_by_case[case_id]
        pred = preds.get(case_id, {})

        if str(gt.get("ifc_class") or "") == str(pred.get("ifc_class") or ""):
            n_class_match += 1
        if str(gt.get("storey_name") or "") == str(pred.get("storey_name") or ""):
            n_storey_match += 1

        gt_rels = _normalize_spatial_relations(gt.get("spatial_relations"))
        pred_rels = _normalize_spatial_relations(pred.get("spatial_relations"))
        if gt_rels:
            n_spatial_total += 1
            gt_sig1 = _relation_signature(gt_rels[0])
            pred_sig1 = _relation_signature(pred_rels[0]) if pred_rels else ("", "", "")
            if gt_sig1 == pred_sig1:
                n_spatial_match += 1

            if Counter(_relation_signature(rel) for rel in gt_rels) == Counter(_relation_signature(rel) for rel in pred_rels):
                n_spatial_set_match += 1

            gt_pred_counts = Counter(_relation_predicate(rel) for rel in gt_rels)
            pred_pred_counts = Counter(_relation_predicate(rel) for rel in pred_rels)
            n_pred_match += sum(min(gt_pred_counts[p], pred_pred_counts[p]) for p in gt_pred_counts)
            n_pred_total += sum(gt_pred_counts.values())

            for gt_rel in gt_rels:
                gt_dir = gt_rel.get("direction")
                if not gt_dir:
                    continue
                n_direction_total += 1
                pred_dirs = [
                    rel.get("direction")
                    for rel in pred_rels
                    if rel.get("predicate") == gt_rel.get("predicate")
                    and rel.get("object_type") == gt_rel.get("object_type")
                ]
                if gt_dir in pred_dirs:
                    n_direction_match += 1

        gt_sc = gt.get("size_cluster")
        pred_sc = pred.get("size_cluster")
        if gt_sc:
            n_size_cluster_gt += 1
            if gt_sc == pred_sc:
                n_size_cluster_match += 1
        if pred_sc:
            n_size_cluster_pred += 1

        gt_pc = gt.get("position_context")
        pred_pc = pred.get("position_context")
        if gt_pc:
            n_position_context_gt += 1
            if gt_pc == pred_pc:
                n_position_context_match += 1
        if pred_pc:
            n_position_context_pred += 1

    return {
        "class_acc": round(_safe_ratio(n_class_match, n_total) * 100.0, 1),
        "storey_acc": round(_safe_ratio(n_storey_match, n_total) * 100.0, 1),
        "hop1_acc": round(_safe_ratio(n_spatial_match, n_spatial_total) * 100.0, 1),
        "spatial_set_acc": round(_safe_ratio(n_spatial_set_match, n_spatial_total) * 100.0, 1),
        "predicate_recall": round(_safe_ratio(n_pred_match, n_pred_total) * 100.0, 1),
        "direction_acc": round(_safe_ratio(n_direction_match, n_direction_total) * 100.0, 1),
        "position_emission": round(_safe_ratio(n_position_context_pred, n_total) * 100.0, 1),
        "position_exact": round(_safe_ratio(n_position_context_match, n_position_context_gt) * 100.0, 1),
        "size_cluster_exact": round(_safe_ratio(n_size_cluster_match, n_size_cluster_gt) * 100.0, 1),
    }


def _prepare_fig04_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_csv(MODALITY_SOURCE)
    _require_columns(
        rows,
        [
            "slice",
            "model",
            "class_acc",
            "storey_acc",
            "one_hop_spatial_accuracy",
            "predicate_recall",
            "direction_accuracy",
        ],
        "fig04 source",
    )
    modality_rows = [
        {
            "slice": row["slice"],
            "slice_label": FIG04_SLICE_LABELS[row["slice"]],
            "model": row["model"],
            "model_label": FIG04_MODALITY_LABELS[row["model"]],
            "class_acc": round(float(row["class_acc"]), 1),
            "storey_acc": round(float(row["storey_acc"]), 1),
            "one_hop_spatial_accuracy": round(float(row["one_hop_spatial_accuracy"]), 1),
            "predicate_recall": round(float(row["predicate_recall"]), 1),
            "direction_accuracy": round(float(row["direction_accuracy"]), 1),
            "source_trace": _repo_rel(MODALITY_SOURCE),
        }
        for row in rows
        if row["slice"] in FIG04_SLICE_ORDER and row["model"] in FIG04_MODALITY_MODELS
    ]
    modality_rows.sort(
        key=lambda row: (
            FIG04_SLICE_ORDER.index(row["slice"]),
            FIG04_MODALITY_MODELS.index(row["model"]),
        )
    )

    field_specs = [
        ("Gemini", GEMINI_V2_EVAL_SOURCE, GT_G7_LABEL_SOURCE),
        ("G7", G7_EVAL_SOURCE, GT_G7_LABEL_SOURCE),
        ("G9 + OpenCV F4 + ResNet", G9_F4_RESNET_EVAL_SOURCE, GT_G9_LABEL_SOURCE),
    ]
    field_rows: list[dict[str, Any]] = []
    for model_label, pred_path, gt_path in field_specs:
        metrics = _score_field_metrics(pred_path, gt_path)
        for metric_key in FIELD_METRIC_ORDER:
            field_rows.append(
                {
                    "model_label": model_label,
                    "metric": metric_key,
                    "metric_label": FIELD_METRIC_LABELS[metric_key],
                    "value": metrics[metric_key],
                    "source_trace": _repo_rel(pred_path),
                }
            )
    return modality_rows, field_rows


def _prepare_fig03_tight_field_rows() -> list[dict[str, Any]]:
    field_specs = [
        ("Gemini", GEMINI_V2_EVAL_SOURCE, GT_G7_LABEL_SOURCE),
        ("G3", G3_EVAL_SOURCE, GT_G7_LABEL_SOURCE),
        ("G9 + OpenCV F4 + ResNet", G9_F4_RESNET_EVAL_SOURCE, GT_G9_LABEL_SOURCE),
    ]
    metric_keys = ["predicate_recall", "direction_acc", "position_exact"]
    rows: list[dict[str, Any]] = []
    for model_label, pred_path, gt_path in field_specs:
        metrics = _score_field_metrics(pred_path, gt_path)
        for metric_key in metric_keys:
            rows.append(
                {
                    "model_label": model_label,
                    "metric": metric_key,
                    "metric_label": FIELD_METRIC_LABELS[metric_key],
                    "value": metrics[metric_key],
                    "source_trace": _repo_rel(pred_path),
                }
            )
    return rows


def _prepare_fig05_rows() -> list[dict[str, Any]]:
    t2_rows = _load_csv(PHASE6_T2_SUMMARY_SOURCE)
    _require_columns(t2_rows, ["system", "top10", "top1", "mrr10"], "phase6 t2 summary")
    t2_by_system = {row["system"]: row for row in t2_rows}

    fused = _load_json(GRAPH_RAG_SOURCE)
    full_topology = fused["modes"]["full_topology"]
    subset = fused["subsets"]["full_topology_topk_not_top1"]

    rows = [
        {
            "system": "P1-only control",
            "pipeline": "P1-only control",
            "source_trace": _repo_rel(PHASE6_T2_SUMMARY_SOURCE),
            "top10": round(float(t2_by_system["P1-only control"]["top10"]), 1),
            "top1": round(float(t2_by_system["P1-only control"]["top1"]), 1),
            "mrr10": round(float(t2_by_system["P1-only control"]["mrr10"]), 4),
        },
        {
            "system": "P1-only + Graph-RAG",
            "pipeline": "P1-only + single-shot rerank",
            "source_trace": _repo_rel(PHASE6_T2_SUMMARY_SOURCE),
            "top10": round(float(t2_by_system["P1-only + single-shot rerank"]["top10"]), 1),
            "top1": round(float(t2_by_system["P1-only + single-shot rerank"]["top1"]), 1),
            "mrr10": round(float(t2_by_system["P1-only + single-shot rerank"]["mrr10"]), 4),
        },
        {
            "system": "G9 + OpenCV F4 + ResNet",
            "pipeline": "Full-topology fused baseline",
            "source_trace": _repo_rel(GRAPH_RAG_SOURCE),
            "top10": round(float(full_topology["baseline"]["top10_pct"]), 1),
            "top1": round(float(full_topology["baseline"]["top1_pct"]), 1),
            "mrr10": round(float(full_topology["baseline"]["mrr10"]), 4),
        },
        {
            "system": "G9 + OpenCV F4 + ResNet + Graph-RAG",
            "pipeline": "Full-topology fused reranked",
            "source_trace": _repo_rel(GRAPH_RAG_SOURCE),
            "top10": round(float(full_topology["reranked"]["top10_pct"]), 1),
            "top1": round(float(full_topology["reranked"]["top1_pct"]), 1),
            "mrr10": round(float(full_topology["reranked"]["mrr10"]), 4),
            "subset_n": int(subset["n"]),
            "subset_before_top1": round(float(subset["baseline"]["top1_pct"]), 1),
            "subset_after_top1": round(float(subset["reranked"]["top1_pct"]), 1),
        },
    ]
    by_system = {row["system"]: row for row in rows}
    return [by_system[system] for system in FIG05_ORDER]


def _prepare_fig07_rows() -> list[dict[str, Any]]:
    graph_rows = _prepare_fig05_rows()
    graph_by_system = {row["system"]: row for row in graph_rows}

    rows = [
        _summary_row_from_metrics(
            "G8 + OpenCV F4 baseline",
            FIG07_LABELS["G8 + OpenCV F4 baseline"],
            G8_RETRIEVAL_SUMMARY,
            _repo_rel(G8_RETRIEVAL_SUMMARY),
        ),
        {
            "system": "P1-only control",
            "display_name": FIG07_LABELS["P1-only control"],
            "top10": graph_by_system["P1-only control"]["top10"],
            "top1": graph_by_system["P1-only control"]["top1"],
            "mrr10": graph_by_system["P1-only control"]["mrr10"],
            "source_trace": graph_by_system["P1-only control"]["source_trace"],
        },
        _summary_row_from_metrics(
            "G7 full topology",
            FIG07_LABELS["G7 full topology"],
            G7_RETRIEVAL_SUMMARY,
            _repo_rel(G7_RETRIEVAL_SUMMARY),
        ),
        _summary_row_from_metrics(
            "G9 + OpenCV F4 + ResNet",
            FIG07_LABELS["G9 + OpenCV F4 + ResNet"],
            G9_RESNET_F4_RETRIEVAL_SUMMARY,
            _repo_rel(G9_RESNET_F4_RETRIEVAL_SUMMARY),
        ),
        {
            "system": "G9 + OpenCV F4 + ResNet + Graph-RAG",
            "display_name": FIG07_LABELS["G9 + OpenCV F4 + ResNet + Graph-RAG"],
            "top10": graph_by_system["G9 + OpenCV F4 + ResNet + Graph-RAG"]["top10"],
            "top1": graph_by_system["G9 + OpenCV F4 + ResNet + Graph-RAG"]["top1"],
            "mrr10": graph_by_system["G9 + OpenCV F4 + ResNet + Graph-RAG"]["mrr10"],
            "source_trace": graph_by_system["G9 + OpenCV F4 + ResNet + Graph-RAG"]["source_trace"],
        },
    ]
    by_system = {row["system"]: row for row in rows}
    ordered = [by_system[system] for system in FIG07_ORDER]
    best_system = max(ordered, key=lambda row: (float(row["mrr10"]), float(row["top1"]), float(row["top10"])))["system"]
    for row in ordered:
        row["is_best"] = row["system"] == best_system
    return ordered


def _band_for_cluster(cluster: str | None, taxonomy: dict[str, Any]) -> str | None:
    if not cluster:
        return None
    for family, clusters in (taxonomy.get("clusters") or {}).items():
        props = (clusters or {}).get(cluster)
        if props is None:
            continue
        band = props.get("band")
        if family == "IfcWindow" and band:
            return f"window_{band}"
        if family == "IfcDoor" and band:
            return f"door_{band}"
    return None


def _score_g9_size_band(pred_path: Path, gt_path: Path, taxonomy_path: Path) -> dict[str, float]:
    taxonomy = _load_json(taxonomy_path)
    case_order, gt_by_case = _load_gt_labels(gt_path)
    preds = {
        row.get("case_id"): (row.get("constraints") or {})
        for row in _load_jsonl(pred_path)
        if row.get("case_id")
    }

    n_gt = 0
    n_emit = 0
    n_match = 0
    for case_id in case_order:
        gt_band = _band_for_cluster(str(gt_by_case[case_id].get("size_cluster") or ""), taxonomy)
        if not gt_band:
            continue
        n_gt += 1
        pred_band = _band_for_cluster(str(preds.get(case_id, {}).get("size_cluster") or ""), taxonomy)
        if pred_band:
            n_emit += 1
        if pred_band == gt_band:
            n_match += 1

    return {
        "band_exact": round(_safe_ratio(n_match, n_gt) * 100.0, 1),
        "band_emit": round(_safe_ratio(n_emit, n_gt) * 100.0, 1),
        "n_gt": n_gt,
        "n_emit": n_emit,
        "n_match": n_match,
    }


def _prepare_backup_model_capability_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    topology_specs = [
        ("Gemini", GEMINI_V2_EVAL_SOURCE, GT_G7_LABEL_SOURCE),
        ("G7", G7_EVAL_SOURCE, GT_G7_LABEL_SOURCE),
        ("G9 LoRA-only", G9_LORA_ONLY_EVAL_SOURCE, GT_G9_LABEL_SOURCE),
    ]
    topology_rows: list[dict[str, Any]] = []
    for model_label, pred_path, gt_path in topology_specs:
        metrics = _score_field_metrics(pred_path, gt_path)
        for metric_key in ("predicate_recall", "direction_acc", "position_exact"):
            topology_rows.append(
                {
                    "section": "topology_field",
                    "model_label": model_label,
                    "metric": metric_key,
                    "metric_label": FIELD_METRIC_LABELS[metric_key],
                    "value": metrics[metric_key],
                    "source_trace": _repo_rel(pred_path),
                }
            )

    size_rows = [
        {
            "section": "size_band",
            "system": "G9 LoRA-only",
            "metric": "size_band_exact",
            "metric_label": "Size band exact",
            "value": _score_g9_size_band(
                G9_LORA_ONLY_EVAL_SOURCE,
                GT_G9_LABEL_SOURCE,
                SIZE_CLUSTER_TAXONOMY_SOURCE,
            )["band_exact"],
            "support_n": 38,
            "source_trace": _repo_rel(G9_LORA_ONLY_EVAL_SOURCE),
        },
        {
            "section": "size_band",
            "system": "ResNet-18 helper",
            "metric": "size_band_exact",
            "metric_label": "Size band exact",
            "value": round(float(_load_json(CLUSTER_CLASSIFIER_TEST_METRICS).get("accuracy", 0.0)) * 100.0, 1),
            "support_n": int(_load_json(CLUSTER_CLASSIFIER_TEST_METRICS).get("n", 0)),
            "source_trace": _repo_rel(CLUSTER_CLASSIFIER_TEST_METRICS),
        },
    ]

    case_order, gt_by_case = _load_gt_labels(GT_G9_LABEL_SOURCE)
    g9_preds = {
        row.get("case_id"): (row.get("constraints") or {})
        for row in _load_jsonl(G9_LORA_ONLY_EVAL_SOURCE)
        if row.get("case_id")
    }
    hard_filter_counts = {
        "no_size_cluster_emitted": 0,
        "correct_size_cluster": 0,
        "wrong_size_cluster": 0,
    }
    for case_id in case_order:
        gt_sc = str(gt_by_case[case_id].get("size_cluster") or "")
        pred_sc = str(g9_preds.get(case_id, {}).get("size_cluster") or "")
        if not pred_sc:
            hard_filter_counts["no_size_cluster_emitted"] += 1
        elif gt_sc and pred_sc == gt_sc:
            hard_filter_counts["correct_size_cluster"] += 1
        else:
            hard_filter_counts["wrong_size_cluster"] += 1

    hard_filter_rows = [
        {
            "section": "hard_filter",
            "case_group": "no_size_cluster_emitted",
            "case_group_label": "No emit",
            "count": hard_filter_counts["no_size_cluster_emitted"],
            "effect": "neutral",
            "source_trace": _repo_rel(G9_LORA_ONLY_EVAL_SOURCE),
        },
        {
            "section": "hard_filter",
            "case_group": "correct_size_cluster",
            "case_group_label": "Correct",
            "count": hard_filter_counts["correct_size_cluster"],
            "effect": "gain",
            "source_trace": _repo_rel(G9_LORA_ONLY_EVAL_SOURCE),
        },
        {
            "section": "hard_filter",
            "case_group": "wrong_size_cluster",
            "case_group_label": "Wrong",
            "count": hard_filter_counts["wrong_size_cluster"],
            "effect": "loss_gt_excluded",
            "source_trace": _repo_rel(G9_LORA_ONLY_EVAL_SOURCE),
        },
    ]

    retrieval_rows = [
        _summary_row_from_metrics("G8 baseline", "G8 baseline", G8_RETRIEVAL_SUMMARY, _repo_rel(G8_RETRIEVAL_SUMMARY)),
        _summary_row_from_metrics("G9 LoRA-only", "G9 LoRA-only", G9_LORA_ONLY_RETRIEVAL_SUMMARY, _repo_rel(G9_LORA_ONLY_RETRIEVAL_SUMMARY)),
        _summary_row_from_metrics("G9 + ResNet", "G9 + ResNet", G9_RESNET_BAND_RETRIEVAL_SUMMARY, _repo_rel(G9_RESNET_BAND_RETRIEVAL_SUMMARY)),
    ]
    return topology_rows, size_rows, hard_filter_rows, retrieval_rows


def _prepare_fig06_rows() -> list[dict[str, Any]]:
    return [dict(row) for row in SUMMARY_ROWS]


def _plot_fig00(selection: SymbolicTraceSelection, count_rows: list[dict[str, Any]]) -> plt.Figure:
    fig = plt.figure(figsize=FIGSIZE_4X3, constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.15, 1.2])
    ax_top = fig.add_subplot(grid[0])
    ax_bottom = fig.add_subplot(grid[1])
    ax_top.axis("off")

    fig.suptitle("How the Graph Turns Language into IFC-Valid Candidates", fontsize=16, fontweight="bold")

    rank_note = f"GT rank {selection.base_rank}" if selection.base_rank is not None else "GT rank hidden"
    if selection.reranked_rank is not None and selection.base_rank is not None:
        rank_note = f"GT rank {selection.base_rank} -> {selection.reranked_rank} after optional rerank"
    ax_top.text(
        0.01,
        0.98,
        f"Selected automatically from the latest AP trace run: {selection.case_id} | {rank_note}",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        color="#334155",
    )

    box_specs = [
        {"x": 0.01, "w": 0.22, "face": "#EFF6FF", "edge": "#2563EB", "title": "1. Typed Constraints"},
        {"x": 0.26, "w": 0.20, "face": "#F5F3FF", "edge": "#7C3AED", "title": "2. Query Planner"},
        {"x": 0.50, "w": 0.21, "face": "#ECFDF5", "edge": "#059669", "title": "3. Graph Traversal"},
        {"x": 0.75, "w": 0.24, "face": "#FFF7ED", "edge": "#EA580C", "title": "4. Ranked IFC Candidates"},
    ]
    y = 0.13
    h = 0.74

    for spec in box_specs:
        patch = FancyBboxPatch(
            (spec["x"], y),
            spec["w"],
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax_top.transAxes,
            linewidth=1.6,
            facecolor=spec["face"],
            edgecolor=spec["edge"],
        )
        ax_top.add_patch(patch)
        ax_top.text(
            spec["x"] + 0.012,
            y + h - 0.03,
            spec["title"],
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            fontsize=11.0,
            fontweight="bold",
            color="#0F172A",
        )

    for left, right in zip(box_specs, box_specs[1:]):
        arrow = FancyArrowPatch(
            (left["x"] + left["w"] + 0.012, y + h / 2.0),
            (right["x"] - 0.012, y + h / 2.0),
            transform=ax_top.transAxes,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color="#94A3B8",
        )
        ax_top.add_patch(arrow)

    relations_block = _summarize_relations(selection.topology_relations, limit=3)
    position_short = selection.position_context
    if " openings on the same wall" in position_short:
        position_short = position_short.replace(" openings on the same wall", " on wall")
    constraint_lines = [
        f'Query: "{selection.query_text}"',
        f"Storey: {selection.constraints.get('storey_name') or selection.gt_storey}",
        f"Class: {selection.constraints.get('ifc_class') or ''}",
    ]
    if selection.target_name_keyword:
        constraint_lines.append(f"Target cue: {selection.target_name_keyword}")
    if position_short:
        constraint_lines.append(f"Pos slot: {position_short}")
    constraint_lines.append("Relations:")
    constraint_lines.append(relations_block)
    ax_top.text(
        box_specs[0]["x"] + 0.012,
        y + h - 0.09,
        textwrap.fill(constraint_lines[0], width=26, break_long_words=False),
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
        color="#0F172A",
    )
    ax_top.text(
        box_specs[0]["x"] + 0.012,
        y + h - 0.20,
        "\n".join(constraint_lines[1:5]),
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color="#0F172A",
    )
    ax_top.text(
        box_specs[0]["x"] + 0.012,
        y + 0.18,
        "\n".join(constraint_lines[5:]),
        transform=ax_top.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#334155",
    )

    planner_lines = [
        "P1: storey + IFC class backstop",
        f"P0: {_pretty_strategy_name(selection.query_plan_used.get('strategy'))}",
        "Route: P0 spatial relation ∪ P1 storey/class",
        "Used: topology union storey+class",
    ]
    if selection.planner_expected_p0 is not None or selection.planner_expected_p1 is not None:
        planner_lines.append(
            f"Planner estimate: P0≈{selection.planner_expected_p0 or '?'} | P1≈{selection.planner_expected_p1 or '?'}"
        )
    planner_lines.append("Union preserves recall when topology is too tight.")
    ax_top.text(
        box_specs[1]["x"] + 0.012,
        y + h - 0.09,
        "\n\n".join(
            [
                "\n".join(planner_lines[:3]),
                "\n".join(planner_lines[3:5]),
                planner_lines[5],
            ]
        ),
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color="#0F172A",
    )

    traversal_lines = [
        f"Containment: {selection.gt_storey or selection.constraints.get('storey_name')}",
        f"Type filter: {selection.constraints.get('ifc_class') or ''}",
        f"P0 spatial-relation pool: {selection.p0_pool_size}",
        f"P1 valid pool: {selection.p1_pool_size}",
        f"Union pool: {selection.union_pool_size}",
    ]
    fingerprint_bits = []
    if any(rel.get("direction") for rel in selection.topology_relations):
        fingerprint_bits.append("direction")
    if any(rel.get("object_subtype") for rel in selection.topology_relations):
        fingerprint_bits.append("subtype")
    if selection.position_context:
        fingerprint_bits.append("position slot")
    if selection.size_cluster:
        fingerprint_bits.append("size band")
    if fingerprint_bits:
        if len(fingerprint_bits) > 2:
            traversal_lines.append("Cues: " + ", ".join(fingerprint_bits[:2]))
            traversal_lines.append("       " + ", ".join(fingerprint_bits[2:]))
        else:
            traversal_lines.append("Cues: " + ", ".join(fingerprint_bits))
    ax_top.text(
        box_specs[2]["x"] + 0.012,
        y + h - 0.09,
        "\n".join(traversal_lines),
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=8.9,
        color="#0F172A",
    )
    ax_top.text(
        box_specs[2]["x"] + 0.012,
        y + 0.14,
        "Every surviving candidate is an\nexisting IFC GUID.",
        transform=ax_top.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#065F46",
    )

    candidate_title = "Final Top-5 (after optional rerank)" if selection.reranked_candidate_guids else "Final Top-5 symbolic shortlist"
    ax_top.text(
        box_specs[3]["x"] + 0.012,
        y + h - 0.09,
        candidate_title,
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        fontweight="bold",
        color="#9A3412",
    )
    for idx, guid in enumerate(selection.top5_guids, start=1):
        is_gt = guid == selection.gt_guid
        suffix = "  <- GT" if is_gt else ""
        ax_top.text(
            box_specs[3]["x"] + 0.016,
            y + h - 0.16 - (idx - 1) * 0.095,
            f"{idx}. {_truncate_guid(guid)}{suffix}",
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            fontfamily="monospace",
            color="#166534" if is_gt else "#0F172A",
            fontweight="bold" if is_gt else "normal",
        )
    ax_top.text(
        box_specs[3]["x"] + 0.012,
        y + 0.10,
        textwrap.fill(
            f"GT GUID: {_truncate_guid(selection.gt_guid)} | {_pretty_ifc_class(selection.constraints.get('ifc_class') or '')} on {selection.gt_storey}",
            width=28,
            break_long_words=False,
        ),
        transform=ax_top.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#7C2D12",
    )

    stages = ["All IFC", P1_SHORT_LABEL, P0_SHORT_LABEL, "P0 ∪ P1\nshortlist", "Top-5\nshortlist"]
    counts = [max(int(row["count"]), 1) for row in count_rows]
    colors = ["#2563EB", "#7C3AED", "#059669", "#16A34A", "#EA580C"]
    bars = ax_bottom.bar(range(len(count_rows)), counts, color=colors, width=0.62, zorder=3)
    ax_bottom.set_yscale("log")
    ax_bottom.set_ylabel("Candidates (log scale)")
    ax_bottom.set_xticks(range(len(count_rows)), stages)
    ax_bottom.grid(axis="y", which="major", alpha=0.25, zorder=0)
    ax_bottom.set_title("Candidate Compression Funnel")
    for bar, row in zip(bars, count_rows):
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(row["count"]) * 1.18,
            str(int(row["count"])),
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color="#0F172A",
        )
    ax_bottom.text(
        0.01,
        0.95,
        "Symbolic retrieval constrains output to IFC-valid GUIDs and makes each filtering step inspectable.",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#F8FAFC", "edgecolor": "none"},
    )
    ax_bottom.text(
        0.99,
        0.95,
        "P0 = spatial-relation match | P1 = storey + IFC class backstop",
        transform=ax_bottom.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#475569",
    )
    return fig


def _plot_fig00_tight(selection: SymbolicTraceSelection, count_rows: list[dict[str, Any]]) -> plt.Figure:
    fig = plt.figure(figsize=FIGSIZE_4X3_TIGHT, constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.45, 1.05])
    ax_top = fig.add_subplot(grid[0])
    ax_bottom = fig.add_subplot(grid[1])
    ax_top.axis("off")

    fig.suptitle("Symbolic Graph Reasoning Is Inspectable", fontsize=16, fontweight="bold")
    ax_top.text(
        0.01,
        0.98,
        f"{selection.case_id} | GT rank {selection.base_rank} -> {selection.reranked_rank or selection.base_rank}",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        color="#475569",
    )

    box_specs = [
        {"x": 0.01, "w": 0.22, "face": "#EFF6FF", "edge": "#2563EB", "title": "Constraints"},
        {"x": 0.26, "w": 0.19, "face": "#F5F3FF", "edge": "#7C3AED", "title": "Planner"},
        {"x": 0.49, "w": 0.22, "face": "#ECFDF5", "edge": "#059669", "title": "Traversal"},
        {"x": 0.75, "w": 0.24, "face": "#FFF7ED", "edge": "#EA580C", "title": "GUID Shortlist"},
    ]
    y = 0.12
    h = 0.73

    for spec in box_specs:
        patch = FancyBboxPatch(
            (spec["x"], y),
            spec["w"],
            h,
            boxstyle="round,pad=0.012,rounding_size=0.022",
            transform=ax_top.transAxes,
            linewidth=1.6,
            facecolor=spec["face"],
            edgecolor=spec["edge"],
        )
        ax_top.add_patch(patch)
        ax_top.text(
            spec["x"] + 0.012,
            y + h - 0.035,
            spec["title"],
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            fontsize=11.2,
            fontweight="bold",
            color="#0F172A",
        )

    for left, right in zip(box_specs, box_specs[1:]):
        arrow = FancyArrowPatch(
            (left["x"] + left["w"] + 0.012, y + h / 2.0),
            (right["x"] - 0.012, y + h / 2.0),
            transform=ax_top.transAxes,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color="#94A3B8",
        )
        ax_top.add_patch(arrow)

    first_rel = selection.topology_relations[0] if selection.topology_relations else {}
    second_rel = selection.topology_relations[1] if len(selection.topology_relations) > 1 else {}
    rel_1 = _relation_display_line(first_rel) if first_rel else "NEXT_TO cue"
    rel_2 = _relation_display_line(second_rel) if second_rel else "FILLS host"
    ax_top.text(
        box_specs[0]["x"] + 0.012,
        y + h - 0.11,
        "\n".join(
            [
                textwrap.fill(f'"{selection.query_text}"', width=23, break_long_words=False),
                f"Storey: {selection.constraints.get('storey_name') or selection.gt_storey}",
                f"Class: {selection.constraints.get('ifc_class') or ''}",
                f"Pos: {selection.position_context.replace(' openings on the same wall', ' on wall')}" if selection.position_context else "",
                f"Rel: {rel_1}",
                f"     {rel_2}",
            ]
        ),
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=8.9,
        color="#0F172A",
    )

    ax_top.text(
        box_specs[1]["x"] + 0.012,
        y + h - 0.11,
        "\n".join(
            [
                "P1 = storey + IFC class",
                "P0 = spatial relation chain",
                "Route = P0 spatial relation ∪ P1 storey/class",
                f"Expect P0≈{selection.planner_expected_p0 or '?'}",
                f"Expect P1≈{selection.planner_expected_p1 or '?'}",
                "Union keeps recall alive",
            ]
        ),
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color="#0F172A",
    )

    cue_parts = []
    if any(rel.get("direction") for rel in selection.topology_relations):
        cue_parts.append("direction")
    if any(rel.get("object_subtype") for rel in selection.topology_relations):
        cue_parts.append("subtype")
    if selection.position_context:
        cue_parts.append("slot")
    if selection.size_cluster:
        cue_parts.append("size")
    ax_top.text(
        box_specs[2]["x"] + 0.012,
        y + h - 0.11,
        "\n".join(
            [
                f"All IFC: {selection.initial_pool_size}",
                f"P0 spatial relation: {selection.p0_pool_size}",
                f"P1 storey/class: {selection.p1_pool_size}",
                f"Union pool: {selection.union_pool_size}",
                "Cues: " + ", ".join(cue_parts) if cue_parts else "Cues: typed filters",
                "All survivors are real IFC GUIDs",
            ]
        ),
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color="#0F172A",
    )

    shortlist = selection.top5_guids[:4]
    lines = []
    for idx, guid in enumerate(shortlist, start=1):
        mark = " <- GT" if guid == selection.gt_guid else ""
        lines.append(f"{idx}. {_truncate_guid(guid)}{mark}")
    if selection.gt_guid not in shortlist:
        lines.append(f"GT: {_truncate_guid(selection.gt_guid)}")
    ax_top.text(
        box_specs[3]["x"] + 0.012,
        y + h - 0.11,
        "\n".join(lines),
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        color="#166534",
        fontfamily="monospace",
        fontweight="bold",
    )
    ax_top.text(
        box_specs[3]["x"] + 0.012,
        y + 0.13,
        "GT becomes rank 1 after optional rerank",
        transform=ax_top.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#9A3412",
    )

    stages = ["All IFC", "P1\nstorey/class", "P0\nspatial relation", "P0 ∪ P1", "Top-5"]
    counts = [max(int(row["count"]), 1) for row in count_rows]
    colors = ["#2563EB", "#7C3AED", "#059669", "#16A34A", "#EA580C"]
    bars = ax_bottom.bar(range(len(count_rows)), counts, color=colors, width=0.58, zorder=3)
    ax_bottom.set_yscale("log")
    ax_bottom.set_ylabel("Candidates")
    ax_bottom.set_xticks(range(len(count_rows)), stages)
    ax_bottom.grid(axis="y", which="major", alpha=0.25, zorder=0)
    ax_bottom.set_title("1666 -> 3 -> 33 -> Top-5", fontsize=13.5)
    for bar, count in zip(bars, counts):
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(count) * 1.15,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color="#0F172A",
        )
    ax_bottom.text(
        0.01,
        0.95,
        "Topology narrows. Union preserves recall. Output stays IFC-valid.",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F8FAFC", "edgecolor": "none"},
    )
    return fig


def _plot_fig00c(selection: SymbolicTraceSelection, rows: list[dict[str, Any]]) -> plt.Figure:
    p0 = selection.p0_pool_size
    union = selection.union_pool_size
    p1_only = max(union - p0, 0)
    top_k = min(10, len(selection.reranked_candidate_guids) or len(selection.candidate_guids) or 10)
    p0_in_topk = min(p0, top_k)
    p1_only_in_topk = max(top_k - p0_in_topk, 0)
    p1_only_dropped = max(p1_only - p1_only_in_topk, 0)
    top5 = min(5, len(selection.top5_guids))

    col_p0 = "#059669"
    col_p1 = "#7C3AED"
    col_drop = "#CBD5E1"
    col_gt = "#16A34A"

    fig = plt.figure(figsize=FIGSIZE_4X3, constrained_layout=False)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.7, 1.0], hspace=0.34, left=0.06, right=0.97, top=0.90, bottom=0.08)
    ax_top = fig.add_subplot(grid[0])
    ax_bot = fig.add_subplot(grid[1])
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    fig.suptitle("Where P0 Spatial-Relation Hits Help: Top-K Keeps Them in the Room", fontsize=16, fontweight="bold", y=0.965)
    fig.text(
        0.06,
        0.928,
        f"{selection.case_id} | GT rank {selection.base_rank} -> {selection.reranked_rank or selection.base_rank} | P0 spatial-relation={p0}, P1-only={p1_only}, Top-K={top_k}",
        ha="left",
        va="top",
        fontsize=10.0,
        color="#475569",
    )

    unit = 0.34 / max(union, 1)
    pool_top = 0.64
    pool_bot = pool_top - union * unit
    x_positions = {"p1": 0.14, "split": 0.38, "topk": 0.63, "top5": 0.86}

    def ribbon(x0: float, x1: float, y0_t: float, y0_b: float, y1_t: float, y1_b: float, color: str, alpha: float) -> None:
        midx = (x0 + x1) / 2.0
        verts = [
            (x0, y0_t), (midx, y0_t), (midx, y1_t), (x1, y1_t),
            (x1, y1_b), (midx, y1_b), (midx, y0_b), (x0, y0_b), (x0, y0_t),
        ]
        codes = [1, 4, 4, 4, 2, 4, 4, 4, 79]
        path = matplotlib.path.Path(verts, codes)
        ax_top.add_patch(matplotlib.patches.PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha, zorder=2))

    def block(x: float, y_top: float, height: float, color: str, width: float = 0.045) -> None:
        ax_top.add_patch(
            FancyBboxPatch(
                (x - width / 2, y_top - height),
                width,
                height,
                boxstyle="round,pad=0.003,rounding_size=0.008",
                linewidth=0,
                facecolor=color,
                edgecolor="none",
                alpha=0.95,
                zorder=4,
            )
        )

    for x, label in [(x_positions["p1"], "P1 storey+class pool"), (x_positions["split"], "P0 spatial-relation vs P1-only"), (x_positions["topk"], "Top-K cut"), (x_positions["top5"], "Final Top-5")]:
        ax_top.text(x, 0.72, label, ha="center", va="bottom", fontsize=11.0, fontweight="bold", color="#0F172A")

    h_pool = union * unit
    block(x_positions["p1"], pool_top, h_pool, col_p1)
    ax_top.text(x_positions["p1"], pool_bot - 0.03, f"{union}\nstorey + IFC class", ha="center", va="top", fontsize=9.3, color="#334155")

    h_p0 = p0 * unit
    h_p1_only = p1_only * unit
    p0_top = pool_top
    p0_bot = p0_top - h_p0
    p1_only_top = p0_bot
    p1_only_bot = p1_only_top - h_p1_only
    block(x_positions["split"], p0_top, h_p0, col_p0)
    block(x_positions["split"], p1_only_top, h_p1_only, col_p1)
    ax_top.text(x_positions["split"] + 0.035, p0_bot + h_p0 / 2, f"P0 spatial-relation branch = {p0}", ha="left", va="center", fontsize=9.6, color=col_p0, fontweight="bold")
    ax_top.text(x_positions["split"] + 0.035, p1_only_bot + h_p1_only / 2, f"{P1_ONLY_INLINE} = {p1_only}", ha="left", va="center", fontsize=9.2, color=col_p1)

    h_p0_topk = p0_in_topk * unit
    h_p1_topk = p1_only_in_topk * unit
    topk_p0_top = pool_top
    topk_p0_bot = topk_p0_top - h_p0_topk
    topk_p1_top = topk_p0_bot
    topk_p1_bot = topk_p1_top - h_p1_topk
    block(x_positions["topk"], topk_p0_top, h_p0_topk, col_p0)
    block(x_positions["topk"], topk_p1_top, h_p1_topk, col_p1)
    ax_top.text(x_positions["topk"] + 0.035, topk_p1_bot + (h_p0_topk + h_p1_topk) / 2, f"{p0_in_topk} P0 spatial-relation + {p1_only_in_topk} P1-only", ha="left", va="center", fontsize=9.2, color="#0F172A")

    drop_top = 0.20
    drop_h = p1_only_dropped * unit
    block(x_positions["topk"], drop_top, drop_h, col_drop)
    ax_top.text(x_positions["topk"] + 0.035, drop_top - drop_h / 2, f"{p1_only_dropped} P1-only\n(storey+class) dropped", ha="left", va="center", fontsize=8.9, color="#64748B")

    top5_h = top5 * unit
    block(x_positions["top5"], pool_top, top5_h, col_gt)
    ax_top.text(x_positions["top5"], pool_top - top5_h - 0.03, f"Top-{top5}\nGT promoted", ha="center", va="top", fontsize=9.2, color="#065F46", fontweight="bold")

    ribbon(x_positions["p1"] + 0.022, x_positions["split"] - 0.022, pool_top, p0_bot, p0_top, p0_bot, col_p0, 0.5)
    ribbon(x_positions["p1"] + 0.022, x_positions["split"] - 0.022, p0_bot, pool_bot, p1_only_top, p1_only_bot, col_p1, 0.42)
    ribbon(x_positions["split"] + 0.022, x_positions["topk"] - 0.022, p0_top, p0_bot, topk_p0_top, topk_p0_bot, col_p0, 0.55)
    ribbon(x_positions["split"] + 0.022, x_positions["topk"] - 0.022, p1_only_top, p1_only_top - h_p1_topk, topk_p1_top, topk_p1_bot, col_p1, 0.55)
    ribbon(x_positions["split"] + 0.022, x_positions["topk"] - 0.022, p1_only_top - h_p1_topk, p1_only_bot, drop_top, drop_top - drop_h, col_drop, 0.65)
    ribbon(x_positions["topk"] + 0.022, x_positions["top5"] - 0.022, pool_top, topk_p1_bot, pool_top, pool_top - top5_h, "#34D399", 0.48)

    ax_top.text(
        0.06,
        0.96,
        "P0 spatial-relation advantage survives at Top-K.",
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color="#065F46",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ECFDF5", "edgecolor": "#059669", "linewidth": 0.8},
    )
    ax_top.text(
        0.56,
        0.96,
        "Inside rerank, source tags disappear.",
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color="#9A3412",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF7ED", "edgecolor": "#EA580C", "linewidth": 0.8},
    )

    x_idx = np.arange(len(rows))
    labels = ["All IFC", "P1 storey/class", "Top-K in", "Top-5 out"]
    p0_counts = [int(row["p0"]) for row in rows]
    p1_counts = [int(row["p1_only"]) for row in rows]
    drop_counts = [int(row["dropped"]) for row in rows]
    totals = [int(row["total"]) for row in rows]
    width = 0.56
    ax_bot.bar(x_idx, p0_counts, width=width, color=col_p0, label="P0 spatial-relation hit", zorder=3)
    ax_bot.bar(x_idx, p1_counts, width=width, bottom=p0_counts, color=col_p1, label="P1-only storey/class filler", zorder=3)
    ax_bot.bar(x_idx, drop_counts, width=width, bottom=np.array(p0_counts) + np.array(p1_counts), color=col_drop, alpha=0.75, label="Dropped", zorder=3)
    for idx, total in enumerate(totals):
        ax_bot.text(idx, total * 1.06 if total > 0 else 1, str(total), ha="center", va="bottom", fontsize=10.0, fontweight="bold", color="#0F172A")
    ax_bot.set_yscale("symlog", linthresh=1)
    ax_bot.set_ylabel("Candidates")
    ax_bot.set_xticks(x_idx, labels)
    ax_bot.set_ylim(0, max(selection.initial_pool_size * 2.0, 10))
    ax_bot.grid(axis="y", which="major", alpha=0.25, zorder=0)
    ax_bot.set_title("Candidate compression, split by source", fontsize=13)
    ax_bot.legend(loc="upper right", framealpha=0.95, edgecolor="#CBD5E1", fontsize=9.0)
    ax_bot.text(0.01, 0.95, "Top-K is where P0 spatial-relation filtering removes weak P1-only storey/class fillers before rerank.", transform=ax_bot.transAxes, ha="left", va="top", fontsize=9.4)
    return fig


def _plot_fig01(rows: list[dict[str, Any]]) -> plt.Figure:
    x = list(range(len(rows)))
    labels = [f"{row['level']}\n{ORACLE_SHORT_LABELS[row['level']]}" for row in rows]
    pools = [max(float(row["median_pool"]), 0.5) for row in rows]
    top10 = [float(row["top10"]) for row in rows]
    top1 = [float(row["top1"]) for row in rows]
    coverage = [float(row["coverage"]) for row in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE_4X3, constrained_layout=True)
    bars = ax.bar(
        x,
        pools,
        width=0.58,
        color=FINGERPRINT_WATERFALL_COLORS.get("bar_blue", "#4472C4"),
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Median candidate pool (log scale)")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", which="major", alpha=0.25, zorder=0)

    for bar, pool, cov in zip(bars, pools, coverage):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            pool * 1.15,
            f"{pool:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.4,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            0.54,
            f"cov {cov:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color="#475569",
        )

    ax2 = ax.twinx()
    top10_line = ax2.plot(
        x,
        top10,
        color=METRIC_COLORS.get("ideal_top1", "#F97316"),
        marker="o",
        linewidth=3.4,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="Top-10",
        zorder=4,
    )[0]
    top1_line = ax2.plot(
        x,
        top1,
        color=HIGHLIGHT_COLORS.get("edge_dark", "#252525"),
        marker="D",
        linestyle="--",
        linewidth=3.0,
        markersize=7.4,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="Top-1",
        zorder=4,
    )[0]
    ax2.set_ylabel("Retrieval accuracy (%)")
    ax2.set_ylim(0, 105)
    for xx, val in zip(x, top10):
        ax2.text(
            xx,
            val + 2.0,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color=top10_line.get_color(),
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    for xx, val in zip(x, top1):
        if val > 0:
            ax2.text(
                xx,
                val + 2.0,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.9,
                fontweight="bold",
                color=top1_line.get_color(),
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )

    ax.legend([bars, top10_line, top1_line], ["Median pool", "Top-10", "Top-1"], loc="upper left", frameon=False)
    ax.set_title("Oracle Ceiling and Candidate-Pool Compression")
    ax.text(
        0.02,
        0.95,
        "Pool 76 -> 1, Top-10 +96.7pp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": HIGHLIGHT_COLORS["oracle_fill"], "edgecolor": "none"},
    )
    return fig


def _plot_fig03(rows: list[dict[str, Any]]) -> plt.Figure:
    x = list(range(len(rows)))
    labels = [FIG03_LABELS[row["system"]] for row in rows]
    gt = [float(row["gt_in_pool"]) for row in rows]
    top10 = [float(row["top10"]) for row in rows]
    top1 = [float(row["top1"]) for row in rows]
    mrr10 = [float(row["mrr10"]) for row in rows]
    colors = [FIG03_COLORS[row["system"]] for row in rows]
    best_idx = next(idx for idx, row in enumerate(rows) if row.get("is_best"))

    fig, ax = plt.subplots(figsize=FIGSIZE_2X1, constrained_layout=True)
    width = 0.20

    ax.axvspan(best_idx - 0.55, best_idx + 0.55, color="#DCFCE7", alpha=0.9, zorder=0)

    gt_bars = ax.bar([xx - width for xx in x], gt, width=width, color=colors, label="GT-in-Pool", zorder=3)
    top10_bars = ax.bar([xx for xx in x], top10, width=width, color=colors, alpha=0.72, label="Top-10", zorder=3)
    top1_bars = ax.bar([xx + width for xx in x], top1, width=width, color=colors, alpha=0.42, label="Top-1", zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.25)

    _annotate_bars(ax, gt_bars, gt, 0.8)
    _annotate_bars(ax, top10_bars, top10, 0.8)
    _annotate_bars(ax, top1_bars, top1, 0.8)

    ax2 = ax.twinx()
    mrr_line = ax2.plot(
        x,
        mrr10,
        color=METRIC_COLORS.get("mrr_track", "#7B1FA2"),
        marker="s",
        linestyle="--",
        linewidth=3.0,
        markersize=8,
        label="MRR@10",
        zorder=4,
    )[0]
    ax2.set_ylabel("MRR@10")
    ax2.set_ylim(0.0, max(mrr10) * 1.35)
    for xx, val in zip(x, mrr10):
        ax2.text(xx, val + 0.0025, f"{val:.4f}", ha="center", va="bottom", fontsize=8.8, color=mrr_line.get_color())

    ax.legend([gt_bars, top10_bars, top1_bars, mrr_line], ["GT-in-Pool", "Top-10", "Top-1", "MRR@10"], loc="upper right", frameon=False)
    ax.set_title("Strict Retrieval Comparison Across Milestones")
    ax.text(
        0.02,
        0.97,
        "Best learned row highlighted. Gemini uses recovered `gemini_v2`.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "none"},
    )
    ax.text(
        best_idx,
        111.5,
        f"Best learned: {rows[best_idx]['display_name']}",
        ha="center",
        va="top",
        fontsize=9.1,
        color="#166534",
        fontweight="bold",
    )
    return fig


def _plot_fig03_tight(rows: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> plt.Figure:
    x = np.arange(len(rows))
    labels = [row["display_name"] for row in rows]
    gt = [float(row["gt_in_pool"]) for row in rows]
    top10 = [float(row["top10"]) for row in rows]
    top1 = [float(row["top1"]) for row in rows]
    mrr10 = [float(row["mrr10"]) for row in rows]
    colors = [FIG03_TIGHT_COLORS[row["system"]] for row in rows]
    best_idx = next(idx for idx, row in enumerate(rows) if row.get("is_best"))
    oracle_idx = next(idx for idx, row in enumerate(rows) if row["system"] == "Oracle ceiling")
    g9_idx = next(idx for idx, row in enumerate(rows) if row["system"] == "G9 + OpenCV F4 + ResNet + Graph-RAG")
    rerank_delta = float(rows[g9_idx].get("rerank_delta_top1_pp", 0.0))
    baseline_top1 = max(float(row["top1"]) for idx, row in enumerate(rows) if idx != g9_idx and row["system"] != "Oracle ceiling")
    best_delta_vs_prev = float(rows[g9_idx]["top1"]) - baseline_top1

    fig, ax = plt.subplots(figsize=FIGSIZE_4X3_TIGHT, constrained_layout=True)
    width = 0.20

    ax.axvspan(best_idx - 0.52, best_idx + 0.52, color="#DCFCE7", alpha=0.95, zorder=0)
    ax.axvspan(oracle_idx - 0.52, oracle_idx + 0.52, color=HIGHLIGHT_COLORS.get("oracle_fill", "#F3E8FF"), alpha=0.55, zorder=0)

    gt_bars = ax.bar([xx - width for xx in x], gt, width=width, color=colors, label="GT-in-Pool", zorder=3)
    top10_bars = ax.bar([xx for xx in x], top10, width=width, color=colors, alpha=0.74, label="Top-10", zorder=3)
    top1_bars = ax.bar([xx + width for xx in x], top1, width=width, color=colors, alpha=0.40, label="Top-1", zorder=3)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.22)

    _annotate_bars(ax, gt_bars, gt, 0.8)
    _annotate_bars(ax, top10_bars, top10, 0.7)
    _annotate_bars(ax, top1_bars, top1, 0.8)

    ax2 = ax.twinx()
    mrr_line = ax2.plot(
        x,
        mrr10,
        color=METRIC_COLORS.get("mrr_track", "#1565C0"),
        marker="s",
        linestyle="--",
        linewidth=3.0,
        markersize=8.0,
        label="MRR@10",
        zorder=4,
    )[0]
    ax2.set_ylabel("MRR@10")
    ax2.set_ylim(0.0, max(mrr10) * 1.35)
    for xx, val in zip(x, mrr10):
        ax2.text(
            xx,
            val + 0.0028,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.4,
            color=mrr_line.get_color(),
            fontweight="bold" if xx == best_idx else None,
        )

    ax.legend([gt_bars, top10_bars, top1_bars, mrr_line], ["GT-in-Pool", "Top-10", "Top-1", "MRR@10"], loc="upper right", frameon=False, fontsize=10.0)
    ax.set_title("Strict Retrieval: Tight Comparison")
    ax.text(
        0.02,
        0.97,
        f"Best: G9 OpenCV/ResNet +{best_delta_vs_prev:.1f}pp Top-1",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.6,
        fontweight="bold",
        color=HIGHLIGHT_COLORS.get("safe_green_text", "#166534"),
        bbox={"boxstyle": "round,pad=0.28", "facecolor": HIGHLIGHT_COLORS.get("safe_green_fill", "#E8F5E9"), "edgecolor": "none"},
    )
    ax.text(
        best_idx,
        109.0,
        f"+{rerank_delta:.1f}pp rerank",
        ha="center",
        va="top",
        fontsize=12.0,
        color=HIGHLIGHT_COLORS.get("safe_green_text", "#166534"),
        fontweight="bold",
    )
    ax.text(
        best_idx,
        102.5,
        "best learned",
        ha="center",
        va="top",
        fontsize=10.2,
        color=HIGHLIGHT_COLORS.get("safe_green_text", "#166534"),
        fontweight="bold",
    )
    ax.text(
        oracle_idx,
        109.0,
        "Ceiling",
        ha="center",
        va="top",
        fontsize=10.5,
        color=HIGHLIGHT_COLORS.get("edge_dark", "#252525"),
        fontweight="bold",
    )

    field_metric_order = ["predicate_recall", "direction_acc", "position_exact"]
    field_metric_labels = ["Predicate", "Direction", "Pos exact"]
    field_metric_markers = {
        "predicate_recall": "o",
        "direction_acc": "s",
        "position_exact": "D",
    }
    field_metric_offsets = {
        "predicate_recall": -0.11,
        "direction_acc": 0.0,
        "position_exact": 0.11,
    }
    field_system_to_model = {
        "Gemini AP (MM)": "Gemini",
        "G3 key": "G3",
        "G9 + OpenCV F4 + ResNet + Graph-RAG": "G9 + OpenCV F4 + ResNet",
    }
    field_model_colors = {
        "Gemini": MODELS.get("gemini_ap_v2", "#1565C0"),
        "G3": MODELS.get("g3_fullaug_r32", "#D32F2F"),
        "G9 + OpenCV F4 + ResNet": G9_RERANK_COLOR,
    }
    field_map = {(row["model_label"], row["metric"]): float(row["value"]) for row in field_rows}

    ax_field = ax.twinx()
    ax_field.set_ylim(0, 110)
    ax_field.set_yticks(np.arange(0, 101, 20))
    ax_field.set_ylabel("Field accuracy (%)")
    ax_field.spines["right"].set_position(("outward", 48))
    ax_field.spines["top"].set_visible(False)
    ax_field.patch.set_visible(False)
    ax_field.tick_params(axis="y", colors="#475569")
    ax_field.spines["right"].set_color("#94A3B8")

    for idx, row in enumerate(rows):
        model_label = field_system_to_model.get(row["system"])
        if not model_label:
            continue
        dot_color = field_model_colors[model_label]
        for metric_key in field_metric_order:
            value = field_map.get((model_label, metric_key), 0.0)
            dot_x = idx + field_metric_offsets[metric_key]
            ax_field.scatter(
                dot_x,
                value,
                s=92,
                marker=field_metric_markers[metric_key],
                color=dot_color,
                edgecolor="white",
                linewidth=0.9,
                zorder=5,
            )
            ax_field.text(
                dot_x + 0.015,
                value + 2.0,
                f"{value:.1f}",
                ha="left",
                va="bottom",
                fontsize=8.4,
                color=dot_color,
                fontweight="bold",
                zorder=6,
            )

    field_handles = [
        Line2D(
            [0],
            [0],
            marker=field_metric_markers[metric_key],
            color="#475569",
            markerfacecolor="white",
            markeredgecolor="#475569",
            markersize=6.5,
            linewidth=0,
            label=metric_label,
        )
        for metric_key, metric_label in zip(field_metric_order, field_metric_labels)
    ]
    field_legend = ax.legend(
        field_handles,
        field_metric_labels,
        title="Field dots",
        loc="lower left",
        bbox_to_anchor=(0.012, 0.02),
        frameon=True,
        fontsize=8.8,
        title_fontsize=9.1,
        facecolor="white",
        edgecolor="#E2E8F0",
    )
    ax.add_artist(field_legend)

    return fig


def _plot_backup_model_capability_proof(
    topology_rows: list[dict[str, Any]],
    size_rows: list[dict[str, Any]],
    hard_filter_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> plt.Figure:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIGSIZE_2X1, gridspec_kw={"wspace": 0.24})
    fig.suptitle(
        "Technical Evidence: learned topology is useful, but noisy size fields are unsafe as hard filters",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    topology_map = {(row["model_label"], row["metric"]): float(row["value"]) for row in topology_rows}
    metric_keys = ["predicate_recall", "direction_acc", "position_exact"]
    metric_labels = ["Predicate", "Direction", "Pos exact"]
    model_order = ["Gemini", "G7", "G9 LoRA-only"]
    width = 0.22
    x = np.arange(len(metric_keys))
    offsets = (-width, 0.0, width)
    for offset, model_label in zip(offsets, model_order):
        values = [topology_map.get((model_label, metric_key), 0.0) for metric_key in metric_keys]
        bars = ax_l.bar(
            x + offset,
            values,
            width=width,
            color=BACKUP_CAPABILITY_MODEL_COLORS[model_label],
            alpha=0.92 if model_label != "Gemini" else 0.78,
            edgecolor="white",
            linewidth=0.8,
            label=model_label,
            zorder=3,
        )
        _annotate_bars(ax_l, bars, values, 1.0)

    ax_l.set_xticks(x)
    ax_l.set_xticklabels(metric_labels)
    ax_l.set_ylim(0, 102)
    ax_l.set_ylabel("Accuracy (%)")
    ax_l.set_title("A. What the VLM actually learns", loc="left", fontsize=13, fontweight="bold", pad=8)
    ax_l.grid(axis="y", alpha=0.28)
    ax_l.legend(loc="upper left", frameon=False, ncol=3)
    ax_l.text(
        0.0,
        1.02,
        "Domain fine-tuning buys direction and predicate control; exact position remains partial.",
        transform=ax_l.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#475569",
    )
    ax_l.annotate(
        "Direction is the clearest learned\nmultimodal spatial gain.",
        xy=(1.0 + width, topology_map.get(("G9 LoRA-only", "direction_acc"), 0.0)),
        xytext=(1.35, 88),
        textcoords="data",
        ha="left",
        va="top",
        fontsize=10.0,
        color="#7C2D12",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#FFF7ED", "edgecolor": "#FB923C"},
        arrowprops={"arrowstyle": "->", "color": "#FB923C", "lw": 1.4},
    )

    size_ax = ax_l.inset_axes([0.60, 0.52, 0.37, 0.34])
    size_labels = [row["system"] for row in size_rows]
    size_values = [float(row["value"]) for row in size_rows]
    size_colors = ["#F97316", "#0F766E"]
    sx = np.arange(len(size_labels))
    size_bars = size_ax.bar(sx, size_values, color=size_colors, alpha=0.94, edgecolor="white", linewidth=0.8, zorder=3)
    for xx, val, row in zip(sx, size_values, size_rows):
        size_ax.text(xx, val + 1.5, f"{val:.1f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
        size_ax.text(xx, 4, f"n={row['support_n']}", ha="center", va="bottom", fontsize=7.8, color="#475569")
    size_ax.set_ylim(0, 92)
    size_ax.set_xticks(sx)
    size_ax.set_xticklabels(["G9\nband", "ResNet-18"], fontsize=8.2)
    size_ax.set_yticks([0, 40, 80])
    size_ax.tick_params(axis="y", labelsize=8.0)
    size_ax.set_title("Size band", fontsize=9.5, fontweight="bold", pad=4)
    size_ax.grid(axis="y", alpha=0.20)
    size_ax.text(
        0.5,
        1.06,
        "Dedicated crop classifier beats the VLM on local scale cues.",
        transform=size_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.9,
        color="#065F46",
    )

    count_map = {row["case_group"]: int(row["count"]) for row in hard_filter_rows}
    case_groups = [
        ("no_size_cluster_emitted", "No emit", "#CBD5E1"),
        ("correct_size_cluster", "Correct", "#34D399"),
        ("wrong_size_cluster", "Wrong", "#F87171"),
    ]
    cx = np.arange(len(case_groups))
    counts = [count_map[key] for key, _label, _color in case_groups]
    colors = [color for _key, _label, color in case_groups]
    bars = ax_r.bar(cx, counts, color=colors, alpha=0.95, edgecolor="white", linewidth=0.8, zorder=3)
    _annotate_bars(ax_r, bars, counts, 0.8, fmt=".0f")
    ax_r.set_xticks(cx)
    ax_r.set_xticklabels([label for _key, label, _color in case_groups])
    ax_r.set_ylim(0, 32)
    ax_r.set_ylabel("Cases (n=60)")
    ax_r.set_title("B. Why `size_cluster` should not be a hard equality filter", loc="left", fontsize=13, fontweight="bold", pad=8)
    ax_r.grid(axis="y", alpha=0.28)
    ax_r.text(
        0.0,
        1.02,
        "G9 LoRA-only emits the field on 38 cases, but wrong emissions outnumber correct ones.",
        transform=ax_r.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#475569",
    )
    ax_r.annotate(
        "26 wrong > 12 correct.\nA hard filter excludes GT\nmore often than it helps.",
        xy=(2.0, count_map["wrong_size_cluster"]),
        xytext=(1.55, 28.5),
        textcoords="data",
        ha="left",
        va="top",
        fontsize=10.0,
        color="#991B1B",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FEF2F2", "edgecolor": "#EF4444"},
        arrowprops={"arrowstyle": "->", "color": "#EF4444", "lw": 1.4},
    )

    retr_ax = ax_r.inset_axes([0.57, 0.08, 0.38, 0.34])
    retr_labels = ["G8", "G9", "G9+\nResNet"]
    retr_top10 = [float(row["top10"]) for row in retrieval_rows]
    retr_colors = ["#3E1080", "#F97316", "#0F766E"]
    rx = np.arange(len(retr_labels))
    retr_bars = retr_ax.bar(rx, retr_top10, color=retr_colors, alpha=0.94, edgecolor="white", linewidth=0.8, zorder=3)
    for xx, val in zip(rx, retr_top10):
        retr_ax.text(xx, val + 0.6, f"{val:.1f}", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    retr_ax.set_ylim(0, 34)
    retr_ax.set_xticks(rx)
    retr_ax.set_xticklabels(retr_labels, fontsize=8.2)
    retr_ax.set_yticks([0, 10, 20, 30])
    retr_ax.tick_params(axis="y", labelsize=8.0)
    retr_ax.set_title("Top-10 retrieval (%)", fontsize=9.5, fontweight="bold", pad=4)
    retr_ax.grid(axis="y", alpha=0.20)
    retr_ax.text(
        0.5,
        -0.28,
        "Helper band cues recover some losses, but the main lesson is routing: noisy size should rank, not filter.",
        transform=retr_ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.7,
        color="#475569",
    )

    return fig


def _plot_fig04(modality_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]], *, slide_tight: bool = False) -> plt.Figure:
    """Panel A (16:9) — multimodal grounding story.

    Direction accuracy across modality slices for G7/G8/Gemini, plus a small
    per-field strip showing how richer models expose partial position/size cues.
    """
    return _plot_fig04a_multimodal(modality_rows, field_rows)


def _plot_fig04a_multimodal(modality_rows: list[dict[str, Any]],
                             field_rows: list[dict[str, Any]]) -> plt.Figure:
    # richest → poorest modality ordering
    slice_order = ["MC4D", "MC", "FP", "SITE", "FPSITE", "MA"]
    slice_labels = {
        "MC4D": "Site + FP\n+ Chat + 4D",
        "MC":   "Site + FP\n+ Chat",
        "FP":   "FP + Chat",
        "SITE": "Site + Chat",
        "FPSITE": "Visual\nonly",
        "MA":   "Chat\nonly",
    }
    models = ["g7_position_context", "g8_posctx_dim", "gemini_ap_v2"]
    model_labels = {"g7_position_context": G7_SHORT_LABEL,
                    "g8_posctx_dim": G8_SHORT_LABEL,
                    "gemini_ap_v2": "Gemini"}
    model_colors = {"g7_position_context": "#7C3AED",
                    "g8_posctx_dim": "#0EA5E9",
                    "gemini_ap_v2": "#94A3B8"}

    fig = plt.figure(figsize=(14.4, 8.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0],
                          wspace=0.20, top=0.83, bottom=0.10,
                          left=0.06, right=0.97)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    fig.suptitle(
        "Multimodal alignment lifts per-field accuracy → better retrieval",
        fontsize=17, fontweight="bold", y=0.96)
    fig.text(0.5, 0.905,
             "Site + floorplan evidence improves directional grounding; "
             "richer models expose partial position / size cues.",
             ha="center", fontsize=12, color="#475569")

    # ── LEFT: Direction accuracy across modality slices ────────────────────
    xs = np.arange(len(slice_order))
    for model_key in models:
        rows_by_slice = {r["slice"]: r for r in modality_rows
                         if r["model"] == model_key}
        ys = [float(rows_by_slice[s]["direction_accuracy"])
              for s in slice_order]
        ax_l.plot(xs, ys, "o-",
                  color=model_colors[model_key],
                  lw=3.0, markersize=10,
                  label=model_labels[model_key],
                  zorder=4 if "gemini" not in model_key else 3,
                  alpha=0.95 if "gemini" not in model_key else 0.85)
        for x, y in zip(xs, ys):
            if y >= 1:
                ax_l.text(x, y + 2.2, f"{y:.0f}",
                          ha="center", va="bottom", fontsize=10,
                          color=model_colors[model_key], fontweight="bold")

    # Shaded regions: rich vs poor grounding
    ax_l.axvspan(-0.5, 1.5, color="#ECFDF5", alpha=0.55, zorder=0)
    ax_l.axvspan(3.5, 5.5, color="#FEF3C7", alpha=0.55, zorder=0)
    ax_l.text(0.5, 102, "Rich grounding\n(Site + FP)",
              ha="center", va="top", fontsize=11,
              fontweight="bold", color="#065F46")
    ax_l.text(4.5, 102, "Sparse grounding",
              ha="center", va="top", fontsize=11,
              fontweight="bold", color="#92400E")

    ax_l.set_xticks(xs)
    ax_l.set_xticklabels([slice_labels[s] for s in slice_order],
                          fontsize=10.5)
    ax_l.set_ylim(0, 110)
    ax_l.set_ylabel("Direction accuracy (%)", fontsize=12)
    ax_l.set_title("A1. Visual grounding helps direction most",
                   fontsize=13, loc="left", pad=6, fontweight="bold")
    ax_l.grid(axis="y", alpha=0.30)
    ax_l.legend(loc="lower left", fontsize=11, frameon=False)

    # Annotations
    ax_l.annotate(
        "G7/G8 position-context models\nhold 70–82% with Site + FP",
        xy=(1, 82.1), xytext=(2.0, 95),
        ha="left", fontsize=10.5, color="#4C1D95", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F5F3FF",
              "edgecolor": "#6D28D9", "linewidth": 1.0},
        arrowprops={"arrowstyle": "->", "color": "#6D28D9", "lw": 1.5})
    ax_l.annotate(
        "Chat-only drops to\n~29% (G8 position context) / 54% (G7 position context)",
        xy=(5, 28.6), xytext=(3.4, 18),
        ha="center", fontsize=10.5, color="#92400E", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FFFBEB",
              "edgecolor": "#D97706", "linewidth": 1.0},
        arrowprops={"arrowstyle": "->", "color": "#D97706", "lw": 1.5})
    ax_l.annotate(
        "Gemini = 0% on every slice\n(no spatial grounding from text)",
        xy=(2.5, 0), xytext=(2.5, 38),
        ha="center", fontsize=10.5, color="#475569", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F1F5F9",
              "edgecolor": "#64748B", "linewidth": 1.0},
        arrowprops={"arrowstyle": "->", "color": "#64748B", "lw": 1.4})

    # ── RIGHT: Per-field breakdown for richest modality (MC) ──────────────
    field_keys = ["predicate_recall", "direction_acc",
                  "position_exact", "size_cluster_exact"]
    field_labels = {"predicate_recall": "Predicate",
                    "direction_acc": "Direction",
                    "position_exact": "Pos exact",
                    "size_cluster_exact": "Size exact"}
    field_models = ["Gemini", "G7", "G9 + OpenCV F4 + ResNet"]
    field_colors = {"Gemini": "#94A3B8", "G7": "#7C3AED",
                    "G9 + OpenCV F4 + ResNet": "#0F766E"}
    field_short = {"Gemini": "Gemini", "G7": G7_SHORT_LABEL,
                   "G9 + OpenCV F4 + ResNet": G9_BASE_SHORT_LABEL}

    field_map = {(row["model_label"], row["metric"]): float(row["value"])
                 for row in field_rows}
    fxs = np.arange(len(field_keys))
    fbw = 0.27
    for offset, m in zip((-fbw, 0.0, fbw), field_models):
        vals = [field_map.get((m, k), 0.0) for k in field_keys]
        ax_r.bar(fxs + offset, vals, width=fbw,
                 color=field_colors[m],
                 alpha=0.95 if m != "Gemini" else 0.78,
                 label=field_short[m], zorder=3,
                 edgecolor="white", linewidth=0.8)
        for xx, v in zip(fxs + offset, vals):
            ax_r.text(xx, v + 1.5,
                      f"{v:.0f}" if v >= 1 else "—",
                      ha="center", va="bottom", fontsize=9.5,
                      color="#0F172A" if v > 0 else "#9CA3AF",
                      fontweight="bold")

    # Highlight position/size zone
    ax_r.axvspan(1.5, 3.5, color="#ECFDF5", alpha=0.55, zorder=0)
    ax_r.text(2.5, 108, "Partial cues (helpers)",
              ha="center", fontsize=10.5,
              fontweight="bold", color="#065F46",
              bbox={"boxstyle": "round,pad=0.20", "facecolor": "white",
                    "edgecolor": "#059669", "linewidth": 1.0})

    ax_r.set_xticks(fxs)
    ax_r.set_xticklabels([field_labels[k] for k in field_keys],
                          fontsize=10.5)
    ax_r.set_ylim(0, 118)
    ax_r.set_yticks([0, 25, 50, 75, 100])
    ax_r.set_ylabel("Field accuracy (%)", fontsize=12)
    ax_r.set_title("A2. Position / size partial — useful as rerank evidence",
                   fontsize=13, loc="left", pad=6, fontweight="bold")
    ax_r.grid(axis="y", alpha=0.30)
    ax_r.legend(loc="upper left", frameon=False, fontsize=10.5, ncol=3)

    return fig


def _plot_fig04b_helper_motivation(field_rows: list[dict[str, Any]]) -> plt.Figure:
    """Panel B (4:3) — why deterministic helpers? G8 position-context / G9 LoRA / Gemini only."""
    field_map = {(row["model_label"], row["metric"]): float(row["value"])
                 for row in field_rows}
    opencv_pos_exact = field_map.get(
        ("G9 + OpenCV F4 + ResNet", "position_exact"), 27.1)
    resnet_size_exact = field_map.get(
        ("G9 + OpenCV F4 + ResNet", "size_cluster_exact"), 31.6)

    # legacy field study (n=59 / n=38) — only the three reference points
    pos_models = ["Gemini", G8_SHORT_LABEL, G9_LORA_SHORT_LABEL]
    pos_match = [20.3, 52.5, 57.6]
    pos_exact = [0.0, 8.5, 15.3]
    g8_w, g8_h = 5.3, 7.9   # G8 dimension attempts (mm), n=38

    fig = plt.figure(figsize=FIGSIZE_4X3)
    ax = fig.add_axes([0.10, 0.08, 0.86, 0.74])
    fig.suptitle("Why deterministic helpers? — VLM LoRAs hit a low ceiling",
                 fontsize=17, fontweight="bold", y=0.965)
    fig.text(0.5, 0.905,
             "OpenCV (position) and ResNet (size) outperform every LoRA we trained.",
             ha="center", fontsize=11.5, color="#475569")

    xs = np.arange(len(pos_models))
    bw = 0.34
    g8_idx = pos_models.index(G8_SHORT_LABEL)
    gem_idx = pos_models.index("Gemini")
    g9_idx = pos_models.index(G9_LORA_SHORT_LABEL)

    # Highlight strip behind G8
    ax.axvspan(g8_idx - 0.5, g8_idx + 0.5, color="#FEF3C7",
               alpha=0.55, zorder=0)

    match_colors = ["#CBD5E1", "#FB923C", "#67E8F9"]
    exact_colors = ["#64748B", "#C2410C", "#0E7490"]

    ax.bar(xs - bw / 2, pos_match, width=bw,
           color=match_colors, alpha=0.65, zorder=3,
           edgecolor="white", linewidth=1.0)
    ax.bar(xs + bw / 2, pos_exact, width=bw,
           color=exact_colors, alpha=0.95, zorder=3,
           edgecolor="white", linewidth=1.0)
    for x_, v in zip(xs, pos_match):
        ax.text(x_ - bw / 2, v + 1.5, f"{v:.0f}",
                ha="center", va="bottom", fontsize=11,
                color="#1E293B")
    for x_, v, ec in zip(xs, pos_exact, exact_colors):
        ax.text(x_ + bw / 2, v + 1.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=12,
                color=ec, fontweight="bold")

    # G8 width / height markers
    ax.scatter([g8_idx - 0.10], [g8_w], marker="v", s=180,
               color="#0EA5E9", zorder=4,
               edgecolor="white", linewidth=1.4)
    ax.scatter([g8_idx + 0.10], [g8_h], marker="^", s=180,
               color="#0284C7", zorder=4,
               edgecolor="white", linewidth=1.4)
    ax.text(g8_idx, max(g8_w, g8_h) + 4.0,
            "G8 position-context\nwidth / height (mm)",
            ha="center", va="bottom", fontsize=10.5,
            fontstyle="italic", color="#0369A1", fontweight="bold")

    # Reference lines
    ax.axhline(opencv_pos_exact, ls="--", color="#059669", lw=2.6, zorder=2)
    ax.text(len(pos_models) - 0.45, opencv_pos_exact + 1.6,
            f"OpenCV F4 → {opencv_pos_exact:.1f}% pos exact",
            ha="right", va="bottom", fontsize=12,
            color="#059669", fontweight="bold",
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "#ECFDF5",
                  "edgecolor": "#059669", "linewidth": 1.4})
    ax.axhline(resnet_size_exact, ls=":", color="#0F766E", lw=2.6, zorder=2)
    ax.text(0.05, resnet_size_exact + 1.6,
            f"ResNet size cluster → {resnet_size_exact:.1f}%",
            ha="left", va="bottom", fontsize=12,
            color="#0F766E", fontweight="bold",
            bbox={"boxstyle": "round,pad=0.30", "facecolor": "#F0FDFA",
                  "edgecolor": "#0F766E", "linewidth": 1.4})

    # Takeaway block placed lower-left under the ResNet reference line
    takeaway = (
        "Three reference points:\n"
        "•  Gemini: no schema → 0% pos exact\n"
        "•  G8 position context: trained for it → 8.5% exact,\n"
        "       5–8% on width / height (mm)\n"
        "•  G9 LoRA-only: best LoRA at 15.3%\n"
        "       — still below OpenCV (27%)"
    )
    ax.text(0.02, 0.62, takeaway,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, color="#0F172A",
            bbox={"boxstyle": "round,pad=0.40",
                  "facecolor": "#FFFBEB", "edgecolor": "#D97706",
                  "linewidth": 1.4})

    # Legend (proxy patches)
    from matplotlib.patches import Patch as _Patch
    handles = [
        _Patch(facecolor="#A78BFA", alpha=0.65, label="pos type-match (n=59)"),
        _Patch(facecolor="#7C3AED", alpha=0.95, label="pos exact (n=59)"),
    ]
    ax.legend(handles=handles, loc="lower right",
              frameon=False, fontsize=11, ncol=2)

    ax.set_xticks(xs); ax.set_xticklabels(pos_models, fontsize=14)
    ax.set_ylabel("Field accuracy (%)", fontsize=13)
    ax.set_ylim(0, 88)
    ax.grid(axis="y", alpha=0.30, zorder=0)
    ax.tick_params(axis="y", labelsize=11)

    return fig


def _plot_fig04_capability_ladder(field_rows: list[dict[str, Any]], *, slide_tight: bool = False) -> plt.Figure:
    tier_layout = [
        ("Anchor", ["class_acc", "storey_acc"], "#EFF6FF", "#1E40AF",
         "Trivial for every system"),
        ("Topology (LoRA)", ["predicate_recall", "direction_acc"], "#F5F3FF", "#6D28D9",
         "LoRA training unlocks these"),
        ("Explicit (OpenCV + ResNet)", ["position_emission", "position_exact", "size_cluster_exact"],
         "#ECFDF5", "#059669", "Deterministic helpers add"),
    ]
    metric_keys = [k for _, ks, *_ in tier_layout for k in ks]
    model_order = ["Gemini", "G7", "G9 + OpenCV F4 + ResNet"]
    model_short = {"Gemini": "Gemini", "G7": G7_SHORT_LABEL,
                   "G9 + OpenCV F4 + ResNet": G9_BASE_SHORT_LABEL}
    base_fs = 14 if slide_tight else 11

    fig = plt.figure(figsize=FIGSIZE_4X3, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.7, 0.85], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    x = np.arange(len(metric_keys))
    width = 0.27

    # tier shading; labels rendered after ylim is set so we know the y-band
    cursor = 0
    tier_spans = []
    for tier_name, keys, face, edge, _ in tier_layout:
        x0 = cursor - 0.5
        x1 = cursor + len(keys) - 0.5
        ax.axvspan(x0, x1, color=face, alpha=0.85, zorder=0)
        tier_spans.append((tier_name, x0, x1, edge))
        cursor += len(keys)

    field_map = {(row["model_label"], row["metric"]): float(row["value"])
                 for row in field_rows}
    # (model, metric) cells that are zero *by design* — model wasn't trained
    # for this field. Rendered as outlined ghost bars to distinguish them
    # from "tried and failed" zeros (which stay as solid 0-height bars + em-dash).
    not_trained = {
        ("G7", "position_emission"),
        ("G7", "position_exact"),
        ("G7", "size_cluster_exact"),
    }
    GHOST_H = 9.0  # nominal height for ghost bars (visible but small)

    for offset, model_label in zip((-width, 0.0, width), model_order):
        values = [field_map.get((model_label, k), 0.0) for k in metric_keys]
        color = FIELD_MODEL_COLORS[model_label]
        alpha = 0.95 if model_label != "Gemini" else 0.78
        for i, (k, v) in enumerate(zip(metric_keys, values)):
            bar_x = x[i] + offset
            bar_label = model_short[model_label] if i == 0 else None
            if (model_label, k) in not_trained:
                ax.bar(bar_x, GHOST_H, width=width,
                       facecolor="none", edgecolor=color,
                       linewidth=1.8, linestyle="--",
                       alpha=0.85, zorder=3, label=bar_label)
                ax.text(bar_x, GHOST_H + 1.0, "n/a",
                        ha="center", va="bottom",
                        fontsize=base_fs - 2,
                        color=color, fontstyle="italic", fontweight="bold")
            else:
                ax.bar(bar_x, v, width=width,
                       color=color, alpha=alpha,
                       zorder=3, edgecolor="white", linewidth=0.8,
                       label=bar_label)
                ax.text(bar_x, v + 1.5,
                        f"{v:.0f}" if v >= 1 else "—",
                        ha="center", va="bottom",
                        fontsize=base_fs - 2,
                        color="#0F172A" if v > 0 else "#9CA3AF",
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([FIELD_METRIC_LABELS[k] for k in metric_keys],
                       fontsize=base_fs)
    ax.set_ylabel("Field accuracy / emission (%)", fontsize=base_fs)
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.30, linewidth=1.2, zorder=0)
    ax.tick_params(axis="y", labelsize=base_fs - 1)
    ax.set_title("A. Capability ladder — what each layer of the stack adds",
                 fontsize=base_fs + 2, pad=12, loc="left")
    # Legend anchored above the axes (between title and tier labels)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005),
              frameon=False, fontsize=base_fs - 1, ncol=3)
    # Tier labels positioned in the headroom strip (y=118-126)
    for tier_name, x0, x1, edge in tier_spans:
        ax.text((x0 + x1) / 2.0, 122, tier_name,
                ha="center", va="center", fontsize=base_fs,
                fontweight="bold", color=edge,
                bbox={"boxstyle": "round,pad=0.24", "facecolor": "white",
                      "edgecolor": edge, "linewidth": 1.2})
    # Footnote moved to figure level after both panels render

    # Take-home callouts (2 arrows)
    g7_dir = field_map.get(("G7", "direction_acc"), 0)
    gem_dir = field_map.get(("Gemini", "direction_acc"), 0)
    ax.annotate(
        f"+{g7_dir - gem_dir:.0f}pp LoRA learns direction",
        xy=(metric_keys.index("direction_acc"), g7_dir),
        xytext=(metric_keys.index("direction_acc") - 0.7, 55),
        ha="center", va="center",
        fontsize=base_fs - 1, fontweight="bold", color="#6D28D9",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
              "edgecolor": "#6D28D9", "linewidth": 1.2},
        arrowprops={"arrowstyle": "->", "color": "#6D28D9", "lw": 1.5},
    )
    g9_pos = field_map.get(("G9 + OpenCV F4 + ResNet", "position_emission"), 0)
    g9_size = field_map.get(("G9 + OpenCV F4 + ResNet", "size_cluster_exact"), 0)
    ax.annotate(
        f"+{g9_pos:.0f}pp pos emit · +{g9_size:.0f}pp size",
        xy=(metric_keys.index("position_emission"), g9_pos),
        xytext=(metric_keys.index("position_exact") + 0.5, 55),
        ha="center", va="center",
        fontsize=base_fs - 1, fontweight="bold", color="#065F46",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
              "edgecolor": "#059669", "linewidth": 1.2},
        arrowprops={"arrowstyle": "->", "color": "#059669", "lw": 1.5},
    )

    # ── Bottom panel: VLM LoRA struggle on pos/size (n=59 / n=38 prior eval) ─
    # Source: legacy LoRA-only field analysis (G3..G9 LoRA, no aux helpers).
    # Answers: "Why bolt on OpenCV + ResNet instead of just training the LoRA?"
    pos_models = [G3_SHORT_LABEL, G4_SHORT_LABEL, G7_SHORT_LABEL, G8_SHORT_LABEL, "Gemini", G9_LORA_SHORT_LABEL]
    pos_exact = [0.0, 1.7, 0.0, 8.5, 0.0, 15.3]       # n=59
    pos_match = [35.6, 52.5, 35.6, 52.5, 20.3, 57.6]  # n=59
    size_w = 5.3   # G8 only (n=38)
    size_h = 7.9
    opencv_pos_exact = field_map.get(
        ("G9 + OpenCV F4 + ResNet", "position_exact"), 27.1)
    resnet_size_exact = field_map.get(
        ("G9 + OpenCV F4 + ResNet", "size_cluster_exact"), 31.6)

    xs = np.arange(len(pos_models))
    g8_idx = pos_models.index(G8_SHORT_LABEL)
    g9_idx = pos_models.index(G9_LORA_SHORT_LABEL)
    gem_idx = pos_models.index("Gemini")
    bw = 0.38

    # Highlight strip behind G8 — the LoRA designed for this task
    ax_b.axvspan(g8_idx - 0.5, g8_idx + 0.5, color="#FEF3C7",
                 alpha=0.65, zorder=0)

    # Per-bar colours so G8 / G9 / Gemini stand out
    match_colors = ["#C4B5FD"] * len(pos_models)
    exact_colors = ["#7C3AED"] * len(pos_models)
    match_colors[g8_idx] = "#FB923C"; exact_colors[g8_idx] = "#C2410C"
    match_colors[g9_idx] = "#67E8F9"; exact_colors[g9_idx] = "#0E7490"
    match_colors[gem_idx] = "#CBD5E1"; exact_colors[gem_idx] = "#64748B"

    ax_b.bar(xs - bw / 2, pos_match, width=bw,
             color=match_colors, alpha=0.65, label="pos type-match",
             zorder=3, edgecolor="white", linewidth=0.6)
    ax_b.bar(xs + bw / 2, pos_exact, width=bw,
             color=exact_colors, alpha=0.95, label="pos exact",
             zorder=3, edgecolor="white", linewidth=0.6)
    for x_, v in zip(xs, pos_match):
        ax_b.text(x_ - bw / 2, v + 1.5, f"{v:.0f}",
                  ha="center", va="bottom",
                  fontsize=base_fs - 3, color="#1E293B")
    for x_, v, ec in zip(xs, pos_exact, exact_colors):
        ax_b.text(x_ + bw / 2, v + 1.5, f"{v:.1f}",
                  ha="center", va="bottom",
                  fontsize=base_fs - 2, color=ec,
                  fontweight="bold")

    # G8 dimension attempts (small markers above its bars)
    ax_b.scatter([g8_idx - 0.10], [size_w], marker="v", s=110,
                 color="#0EA5E9", zorder=4, edgecolor="white", linewidth=0.8)
    ax_b.scatter([g8_idx + 0.10], [size_h], marker="^", s=110,
                 color="#0284C7", zorder=4, edgecolor="white", linewidth=0.8)
    ax_b.text(g8_idx, max(size_w, size_h) + 3.0,
              "G8 position-context\nwidth / height (mm)",
              ha="center", va="bottom", fontsize=base_fs - 3,
              color="#0369A1", fontstyle="italic")

    # Reference lines for deterministic helpers
    ax_b.axhline(opencv_pos_exact, ls="--", color="#059669", lw=2.2, zorder=2)
    ax_b.text(len(pos_models) - 0.45, opencv_pos_exact + 1.5,
              f"OpenCV F4 → {opencv_pos_exact:.1f}% pos exact",
              ha="right", va="bottom", fontsize=base_fs - 1,
              color="#059669", fontweight="bold",
              bbox={"boxstyle": "round,pad=0.22", "facecolor": "#ECFDF5",
                    "edgecolor": "#059669", "linewidth": 1.0})
    ax_b.axhline(resnet_size_exact, ls=":", color="#0F766E", lw=2.2, zorder=2)
    ax_b.text(0.05, resnet_size_exact + 1.5,
              f"ResNet size cluster → {resnet_size_exact:.1f}%",
              ha="left", va="bottom", fontsize=base_fs - 1,
              color="#0F766E", fontweight="bold",
              bbox={"boxstyle": "round,pad=0.22", "facecolor": "#F0FDFA",
                    "edgecolor": "#0F766E", "linewidth": 1.0})

    # Single consolidated takeaway block (top-right) — calls out G8, G9, Gemini
    takeaway = (
        "Three reference points:\n"
        "•  Gemini: no schema — 0% pos exact\n"
        "•  G8 position context: only 8.5% exact, 5–8% on width/height (mm)\n"
        "•  G9 LoRA-only: best LoRA at 15.3% — still below OpenCV F4 (27%)"
    )
    ax_b.text(0.99, 0.97, takeaway,
              transform=ax_b.transAxes, ha="right", va="top",
              fontsize=base_fs - 2, color="#0F172A",
              bbox={"boxstyle": "round,pad=0.36",
                    "facecolor": "#FFFBEB", "edgecolor": "#D97706",
                    "linewidth": 1.4})

    ax_b.set_xticks(xs); ax_b.set_xticklabels(pos_models, fontsize=base_fs)
    ax_b.set_ylabel("Field accuracy (%)", fontsize=base_fs)
    ax_b.set_ylim(0, 88)
    ax_b.grid(axis="y", alpha=0.30, zorder=0)
    # Proxy artists for legend (per-bar colors don't auto-populate cleanly)
    from matplotlib.patches import Patch as _Patch
    legend_handles = [
        _Patch(facecolor="#C4B5FD", alpha=0.65, label="pos type-match"),
        _Patch(facecolor="#7C3AED", alpha=0.95, label="pos exact"),
    ]
    ax_b.legend(handles=legend_handles, loc="upper left",
                frameon=False, fontsize=base_fs - 1, ncol=2)
    ax_b.set_title(
        "B. VLM LoRAs peak at 8–15% pos exact; helpers reach 27–32%",
        fontsize=base_fs + 1, pad=8, loc="left",
    )
    return fig


def _plot_fig04_legacy_unused(modality_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]], *, slide_tight: bool = False) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE_4X3,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
        constrained_layout=True,
    )

    x1 = list(range(len(FIG04_SLICE_ORDER)))
    width1 = 0.24
    for offset, model_key in zip((-width1, 0.0, width1), FIG04_MODALITY_MODELS):
        model_rows = {row["slice"]: row for row in modality_rows if row["model"] == model_key}
        values = [float(model_rows[slice_key]["direction_accuracy"]) for slice_key in FIG04_SLICE_ORDER]
        bars = ax1.bar(
            [xx + offset for xx in x1],
            values,
            width=width1,
            color=FIG04_MODALITY_COLORS[model_key],
            label=FIG04_MODALITY_LABELS[model_key],
            zorder=3,
            alpha=0.96 if model_key != "gemini_ap_v2" else 0.75,
        )
        if model_key == "g8_posctx_dim":
            for bar, slice_key in zip(bars, FIG04_SLICE_ORDER):
                row = model_rows[slice_key]
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.0,
                    f"C{row['class_acc']:.0f}/S{row['storey_acc']:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    rotation=90,
                    color="#334155",
                    clip_on=False,
                )

    ax1.set_xticks(x1, [FIG04_SLICE_LABELS[slice_key] for slice_key in FIG04_SLICE_ORDER], rotation=18, ha="right")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("A. Direction survives under multimodal grounding")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, loc="lower left")
    if slide_tight:
        ax1.annotate(
            "Direction survives in learned MM models.",
            xy=(0.0, 82.1),
            xytext=(1.0, 92.0),
            textcoords="data",
            ha="left",
            va="top",
            fontsize=8.4,
            color="#4C1D95",
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "#F5F3FF", "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": "#6D28D9", "lw": 1.4},
        )
        ax1.annotate(
            "Gemini direction = 0%.",
            xy=(0.24, 1.5),
            xytext=(3.35, 18.0),
            textcoords="data",
            ha="left",
            va="bottom",
            fontsize=8.4,
            color="#1E40AF",
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "#EFF6FF", "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": "#2563EB", "lw": 1.3},
        )
    else:
        ax1.text(
            0.02,
            0.96,
            "Bars now show direction accuracy, not 1-hop SR.\n`C/S` annotations still show class/storey accuracy on the G8 position-context bar.",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=8.9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#EFF6FF", "edgecolor": "none"},
        )
        ax1.annotate(
            "Learned multimodal rows keep direction alive:\nG7 position context stays around 71-82% on MC/MC4D/FP/SITE.",
            xy=(0.0, 82.1),
            xytext=(0.9, 92.0),
            textcoords="data",
            ha="left",
            va="top",
            fontsize=8.5,
            color="#4C1D95",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F5F3FF", "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": "#6D28D9", "lw": 1.4},
        )
        ax1.annotate(
            "Gemini collapses to 0% direction on every slice.\nThat makes direction a weak but useful proof of multimodal alignment.",
            xy=(0.24, 1.5),
            xytext=(3.15, 18.0),
            textcoords="data",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#1E40AF",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#EFF6FF", "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": "#2563EB", "lw": 1.3},
        )
        ax1.annotate(
            "Ablations weaken direction most.\nVisual-only and chat-only lose grounding cues.",
            xy=(4.0, 62.5),
            xytext=(4.2, 66.0),
            textcoords="data",
            ha="left",
            va="bottom",
            fontsize=8.2,
            color="#475569",
        )

    field_plot_order = [metric_key for metric_key in FIELD_METRIC_ORDER if metric_key != "hop1_acc"]
    x2 = list(range(len(field_plot_order)))
    width2 = 0.24
    for offset, model_label in zip((-width2, 0.0, width2), FIELD_MODEL_ORDER):
        model_map = {
            row["metric"]: row["value"]
            for row in field_rows
            if row["model_label"] == model_label
        }
        values = [float(model_map[metric_key]) for metric_key in field_plot_order]
        bars = ax2.bar(
            [xx + offset for xx in x2],
            values,
            width=width2,
            color=FIELD_MODEL_COLORS[model_label],
            label=model_label,
            zorder=3,
            alpha=0.96 if model_label != "Gemini" else 0.78,
        )
        for bar, value in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                rotation=90,
                color="#334155",
                clip_on=False,
            )

    ax2.set_xticks(x2, [FIELD_METRIC_LABELS[key] for key in field_plot_order], rotation=18, ha="right")
    ax2.set_ylim(0, 108)
    ax2.set_title("B. Added G9 OpenCV/ResNet field scores: direction, position, size, full SR set")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, loc="lower left")
    direction_idx = field_plot_order.index("direction_acc")
    pos_emit_idx = field_plot_order.index("position_emission")
    pos_exact_idx = field_plot_order.index("position_exact")
    size_exact_idx = field_plot_order.index("size_cluster_exact")
    ax2.axvspan(direction_idx - 0.45, direction_idx + 0.45, color="#F5F3FF", alpha=0.95, zorder=0)
    ax2.axvspan(pos_emit_idx - 0.55, size_exact_idx + 0.55, color="#ECFDF5", alpha=0.9, zorder=0)
    if slide_tight:
        ax2.annotate(
            "Direction stays alive.\nG7 position context / G9 OpenCV-ResNet > Gemini.",
            xy=(direction_idx + width2, 76.8),
            xytext=(3.1, 101.5),
            textcoords="data",
            ha="left",
            va="top",
            fontsize=8.4,
            color="#4C1D95",
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "#F5F3FF", "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": "#6D28D9", "lw": 1.4},
        )
        ax2.annotate(
            "G9 OpenCV/ResNet: 100% pos emit.\n27.1% slot exact.\n31.6% size exact.",
            xy=(pos_emit_idx + width2, 100.0),
            xytext=(5.55, 101.5),
            textcoords="data",
            ha="left",
            va="top",
            fontsize=8.4,
            color="#065F46",
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "#ECFDF5", "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": "#059669", "lw": 1.4},
        )
    else:
        ax2.text(
            0.02,
            0.96,
            "Panel B is scored from the AP eval JSONL traces.\nIt surfaces the extra G9 OpenCV/ResNet signals that used to live only in the write-up.",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=8.9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "none"},
        )
        ax2.annotate(
            "Direction does not collapse in learned multimodal models:\nG7 position context 82.1%, G9 OpenCV/ResNet 76.8%, Gemini 0.0%.",
            xy=(direction_idx + width2, 76.8),
            xytext=(2.4, 101.5),
            textcoords="data",
            ha="left",
            va="top",
            fontsize=8.5,
            color="#4C1D95",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F5F3FF", "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": "#6D28D9", "lw": 1.4},
        )
        ax2.annotate(
            "G9 OpenCV/ResNet emits position context on 100% of cases.\nExact slot 27.1% and exact size 31.6% are still hard,\nbut they are non-zero only when the multimodal cues are working.",
            xy=(pos_emit_idx + width2, 100.0),
            xytext=(4.8, 101.5),
            textcoords="data",
            ha="left",
            va="top",
            fontsize=8.5,
            color="#065F46",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#ECFDF5", "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": "#059669", "lw": 1.4},
        )

    fig.suptitle("Multimodal Alignment and Richer Constraint Fields", fontsize=15, fontweight="bold")
    return fig


def _plot_fig04_tight_single(modality_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> plt.Figure:
    metric_order = ["predicate_recall", "spatial_set_acc", "direction_acc", "position_emission", "position_exact", "size_cluster_exact"]
    metric_labels = {
        "predicate_recall": "Predicate",
        "spatial_set_acc": "All SR",
        "direction_acc": "Direction",
        "position_emission": "Pos emit",
        "position_exact": "Pos exact",
        "size_cluster_exact": "Size exact",
    }
    model_order = ["Gemini", "G7", "G9 + OpenCV F4 + ResNet"]
    x = list(range(len(metric_order)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_4X3, gridspec_kw={"width_ratios": [1.05, 1.1]}, constrained_layout=True)

    slice_x = list(range(len(FIG04_SLICE_ORDER)))
    for model_key in FIG04_MODALITY_MODELS:
        model_rows = {row["slice"]: row for row in modality_rows if row["model"] == model_key}
        values = [float(model_rows[slice_key]["direction_accuracy"]) for slice_key in FIG04_SLICE_ORDER]
        ax1.plot(
            slice_x,
            values,
            marker="o",
            linewidth=2.6,
            markersize=7.0,
            color=FIG04_MODALITY_COLORS[model_key],
            label=FIG04_MODALITY_LABELS[model_key],
            zorder=3,
        )
        for xx, value in zip(slice_x, values):
            ax1.text(xx, value + 2.0, f"{value:.0f}", ha="center", va="bottom", fontsize=8.0, color="#334155")
    ax1.set_xticks(slice_x, [FIG04_SLICE_LABELS[s].replace(" + ", "\n") for s in FIG04_SLICE_ORDER], rotation=0)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Direction accuracy (%)")
    ax1.set_title("A. Direction across modality slices")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, loc="lower left")
    ax1.text(
        0.03,
        0.95,
        "Learned MM keeps direction alive.\nGemini stays at 0%.",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9.3,
        color="#4C1D95",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#F5F3FF", "edgecolor": "none"},
    )

    ax2.axvspan(-0.45, 2.45, color="#F5F3FF", alpha=0.95, zorder=0)
    ax2.axvspan(2.45, 5.45, color="#ECFDF5", alpha=0.9, zorder=0)
    for model_label in model_order:
        model_map = {
            row["metric"]: float(row["value"])
            for row in field_rows
            if row["model_label"] == model_label
        }
        values = [model_map[metric_key] for metric_key in metric_order]
        ax2.plot(
            x,
            values,
            marker="o",
            linewidth=2.6 if model_label != "Gemini" else 2.2,
            markersize=7.0,
            color=FIELD_MODEL_COLORS[model_label],
            alpha=0.98 if model_label != "Gemini" else 0.82,
            label=G9_BASE_SHORT_LABEL if model_label == "G9 + OpenCV F4 + ResNet" else model_label,
            zorder=3,
        )
        for xx, value in zip(x, values):
            ax2.text(xx, value + 2.0, f"{value:.1f}", ha="center", va="bottom", fontsize=7.8, color="#334155", rotation=90 if value >= 20 else 0, clip_on=False)
    ax2.set_xticks(x, [metric_labels[key] for key in metric_order])
    ax2.set_ylim(0, 108)
    ax2.set_title("B. G9 OpenCV/ResNet adds explicit cues")
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(
        0.05,
        0.95,
        "100% emit\n+27.1pp slot exact\n+31.6pp size exact",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#065F46",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#ECFDF5", "edgecolor": "none"},
    )

    fig.suptitle("Direction Survives; G9 OpenCV/ResNet Adds Position/Size Cues", fontsize=15)
    return fig


def _plot_fig05(rows: list[dict[str, Any]]) -> plt.Figure:
    paired = [
        (
            P1_INLINE,
            next(row for row in rows if row["system"] == "P1-only control"),
            next(row for row in rows if row["system"] == "P1-only + Graph-RAG"),
            FIG05_COLORS["P1-only + Graph-RAG"],
        ),
        (
            "G9 OpenCV/ResNet aug.",
            next(row for row in rows if row["system"] == "G9 + OpenCV F4 + ResNet"),
            next(row for row in rows if row["system"] == "G9 + OpenCV F4 + ResNet + Graph-RAG"),
            FIG05_COLORS["G9 + OpenCV F4 + ResNet + Graph-RAG"],
        ),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_4X3, constrained_layout=True)
    rescue_row = next(row for row in rows if row["system"] == "G9 + OpenCV F4 + ResNet + Graph-RAG")

    y_base = np.arange(len(paired)) * 2.4
    for idx, (label, before, after, color) in enumerate(paired):
        top1_y = y_base[idx] + 0.42
        mrr_y = y_base[idx] - 0.42

        before_top1 = float(before["top1"])
        after_top1 = float(after["top1"])
        ax.plot([before_top1, after_top1], [top1_y, top1_y], color="#F97316", linewidth=5.5, zorder=2, solid_capstyle="round")
        ax.scatter(before_top1, top1_y, s=320, color="white", edgecolors="#F97316", linewidths=3.5, zorder=3)
        ax.scatter(after_top1, top1_y, s=360, color="#F97316", edgecolors="white", linewidths=2.0, zorder=4)
        ax.text(after_top1 + 0.6, top1_y, f"+{after_top1 - before_top1:.1f}pp", ha="left", va="center", fontsize=14.5, color="#F97316", fontweight="bold")

        before_mrr = float(before["mrr10"]) * 100.0
        after_mrr = float(after["mrr10"]) * 100.0
        ax.plot([before_mrr, after_mrr], [mrr_y, mrr_y], color="#7B1FA2", linewidth=5.5, zorder=2, solid_capstyle="round")
        ax.scatter(before_mrr, mrr_y, s=320, color="white", edgecolors="#7B1FA2", linewidths=3.5, zorder=3)
        ax.scatter(after_mrr, mrr_y, s=360, color="#7B1FA2", edgecolors="white", linewidths=2.0, zorder=4)
        ax.text(after_mrr + 0.6, mrr_y, f"+{after_mrr - before_mrr:.1f}pp", ha="left", va="center", fontsize=14.5, color="#7B1FA2", fontweight="bold")

        ax.text(-1.8, y_base[idx], label, ha="left", va="center", fontsize=15.5, fontweight="bold", color=color)
        ax.text(before_top1 - 0.25, top1_y - 0.30, f"{before_top1:.1f}", ha="right", va="bottom", fontsize=12.5, color="#475569")
        ax.text(after_top1, top1_y - 0.30, f"{after_top1:.1f}", ha="center", va="bottom", fontsize=13.0, color="#F97316", fontweight="bold")
        ax.text(before_mrr - 0.25, mrr_y - 0.30, f"{before_mrr/100.0:.3f}", ha="right", va="bottom", fontsize=12.5, color="#475569")
        ax.text(after_mrr, mrr_y - 0.30, f"{after_mrr/100.0:.3f}", ha="center", va="bottom", fontsize=13.0, color="#7B1FA2", fontweight="bold")

    ax.set_xlim(-2.0, 16.5)
    ax.set_ylim(-1.5, y_base[-1] + 1.6)
    ax.set_yticks([])
    ax.set_xlabel("Shared slide scale: Top-1 (%) and MRR@10 ×100", fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", alpha=0.30, linewidth=1.5)
    ax.set_title("Rerank reorders, doesn't expand recall", fontsize=18, fontweight="bold", pad=12)
    ax.text(
        0.02, 0.96,
        "Orange = Top-1.   Purple = MRR@10 ×100.   Top-10 stays flat.",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=12.5, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.40", "facecolor": "#F1F5F9", "edgecolor": "#94A3B8", "linewidth": 1.0},
    )
    ax.text(
        0.02, 0.06,
        f"G9 OpenCV/ResNet + rerank rescue: 3/{rescue_row.get('subset_n', 0)} Top-10 near-misses → Top-1",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12.5, fontweight="bold", color="#6B21A8",
        bbox={"boxstyle": "round,pad=0.40", "facecolor": HIGHLIGHT_COLORS["oracle_fill"], "edgecolor": "#7C3AED", "linewidth": 1.0},
    )
    return fig


def _plot_fig06(rows: list[dict[str, Any]]) -> plt.Figure:
    headers = ["Section", "Finding", "Interpretation", "RQ"]
    wrapped_rows = [
        [
            textwrap.fill(row["section"], width=16),
            textwrap.fill(row["finding"], width=46),
            textwrap.fill(row["interpretation"], width=40),
            row["rq_link"],
        ]
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_4X3)
    ax.axis("off")
    table = ax.table(
        cellText=wrapped_rows,
        colLabels=headers,
        colWidths=[0.16, 0.40, 0.34, 0.10],
        cellLoc="left",
        bbox=[0.0, 0.10, 1.0, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.1)
    table.scale(1.0, 2.0)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(weight="bold", color="#0F172A")
        else:
            cell.set_facecolor("#F8FAFC" if row_idx % 2 == 1 else "#FFFFFF")
        cell.set_edgecolor("#CBD5E1")
        cell.PAD = 0.03
        if col_idx < 3:
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_ha("center")

    fig.suptitle("Final Findings Summary", fontsize=15, fontweight="bold", y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _plot_fig07(rows: list[dict[str, Any]]) -> plt.Figure:
    x = list(range(len(rows)))
    labels = [FIG07_LABELS[row["system"]] for row in rows]
    top10 = [float(row["top10"]) for row in rows]
    top1 = [float(row["top1"]) for row in rows]
    mrr10 = [float(row["mrr10"]) for row in rows]
    colors = [FIG07_COLORS[row["system"]] for row in rows]
    best_idx = next(idx for idx, row in enumerate(rows) if row.get("is_best"))
    best_display = str(rows[best_idx]["display_name"]).replace("\n", " ")

    fig, ax = plt.subplots(figsize=FIGSIZE_2X1, constrained_layout=True)
    width = 0.28
    ax.axvspan(best_idx - 0.55, best_idx + 0.55, color="#FEF3C7", alpha=0.9, zorder=0)
    top10_bars = ax.bar([xx - width / 2 for xx in x], top10, width=width, color=colors, alpha=0.88, label="Top-10", zorder=3)
    top1_bars = ax.bar([xx + width / 2 for xx in x], top1, width=width, color=colors, alpha=0.48, label="Top-1", zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(top10) + 12)
    ax.grid(axis="y", alpha=0.25)
    _annotate_bars(ax, top10_bars, top10, 0.7)
    _annotate_bars(ax, top1_bars, top1, 0.7)

    ax2 = ax.twinx()
    mrr_line = ax2.plot(
        x,
        mrr10,
        color=METRIC_COLORS.get("mrr_track", "#7B1FA2"),
        marker="s",
        linestyle="--",
        linewidth=2.8,
        markersize=8,
        label="MRR@10",
        zorder=4,
    )[0]
    ax2.set_ylabel("MRR@10")
    ax2.set_ylim(0, max(mrr10) * 1.35)
    for xx, val in zip(x, mrr10):
        ax2.text(xx, val + 0.0025, f"{val:.4f}", ha="center", va="bottom", fontsize=8.8, color=mrr_line.get_color())

    ax.legend([top10_bars, top1_bars, mrr_line], ["Top-10", "Top-1", "MRR@10"], loc="upper left", frameon=False)
    ax.set_title("Retrieval Comparison Across Pipeline Variants")
    ax.text(
        0.02,
        0.97,
        "Best overall highlighted. G9 OpenCV/ResNet + rerank adds +1.6pp Top-1 over the non-reranked G9 pipeline.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "none"},
    )
    ax.text(
        best_idx,
        max(top10) + 9.0,
        f"Best overall: {best_display}",
        ha="center",
        va="top",
        fontsize=9.1,
        color="#92400E",
        fontweight="bold",
    )
    return fig


def _manifest_entry(artifact: FigureArtifact) -> dict[str, Any]:
    return {
        "id": artifact.figure_id,
        "title": artifact.title,
        "png": _repo_rel(artifact.png_path),
        "pdf": _repo_rel(artifact.pdf_path),
        "data": _repo_rel(artifact.data_path),
        "slide_use": True,
        "thesis_use": True,
        "rq": list(artifact.rq),
        "claim": artifact.claim,
        "sources": [_repo_rel(path) for path in artifact.sources],
    }


def _generate_fig00(output_dir: Path, data_dir: Path, dry_run: bool) -> FigureArtifact:
    figure_id = "fig00_symbolic_reasoning_trace"
    base_path = output_dir / figure_id
    tight_base_path = output_dir / "fig00_symbolic_reasoning_trace_tight"
    data_path = data_dir / "symbolic_reasoning_trace_counts.csv"
    case_json_path = data_dir / "symbolic_reasoning_trace_case.json"
    count_rows, case_payload, selection = _prepare_fig00_rows()
    if not dry_run:
        _write_csv(data_path, count_rows, ["stage", "stage_label", "count", "detail"])
        case_json_path.write_text(json.dumps(case_payload, indent=2), encoding="utf-8")
        _save_figure(_plot_fig00(selection, count_rows), base_path)
        _save_figure(_plot_fig00_tight(selection, count_rows), tight_base_path)
    return FigureArtifact(
        figure_id=figure_id,
        title="Symbolic Reasoning Trace",
        claim="The symbolic backend is inspectable: it turns typed constraints into IFC-valid candidates through topology-aware filtering and recall-preserving union.",
        rq=("RQ2",),
        png_path=base_path.with_suffix(".png"),
        pdf_path=base_path.with_suffix(".pdf"),
        data_path=data_path,
        sources=(DEFAULT_CASES, selection.trace_path, GRAPH_RAG_RESULTS_SOURCE),
    )


def _generate_fig00c(output_dir: Path, data_dir: Path, dry_run: bool) -> FigureArtifact:
    figure_id = "fig00c_p0_topk_flow"
    base_path = output_dir / figure_id
    data_path = data_dir / "p0_topk_flow_breakdown.csv"
    _, _, selection = _prepare_fig00_rows()
    rows = _prepare_fig00c_rows(selection)
    if not dry_run:
        _write_csv(data_path, rows, ["stage", "total", "p0", "p1_only", "dropped"])
        _save_figure(_plot_fig00c(selection, rows), base_path)
    return FigureArtifact(
        figure_id=figure_id,
        title="P0 Spatial-Relation to Top-K Flow",
        claim="P0 spatial-relation filtering helps most at the Top-K cut, where weak P1-only storey/class fillers are removed before rerank.",
        rq=("RQ2",),
        png_path=base_path.with_suffix(".png"),
        pdf_path=base_path.with_suffix(".pdf"),
        data_path=data_path,
        sources=(DEFAULT_CASES, selection.trace_path, GRAPH_RAG_RESULTS_SOURCE),
    )


def _generate_fig01(output_dir: Path, data_dir: Path, dry_run: bool) -> FigureArtifact:
    figure_id = "fig01_oracle_symbolic_ceiling"
    base_path = output_dir / figure_id
    data_path = data_dir / "oracle_symbolic_ceiling.csv"
    rows = _prepare_oracle_rows("all_cases")
    if not dry_run:
        _write_csv(
            data_path,
            rows,
            ["level", "level_code", "level_name", "fields_active", "coverage", "n_cases", "n_cases_covered", "top10", "top1", "median_pool", "avg_pool"],
        )
        _save_figure(_plot_fig01(rows), base_path)
    return FigureArtifact(
        figure_id=figure_id,
        title="Oracle Ceiling and Pool Compression",
        claim="Adding more symbolic query fields both preserves the correct element and collapses the symbolic candidate pool.",
        rq=("RQ2",),
        png_path=base_path.with_suffix(".png"),
        pdf_path=base_path.with_suffix(".pdf"),
        data_path=data_path,
        sources=(ORACLE_FINGERPRINT_SOURCE,),
    )


def _generate_fig03(output_dir: Path, data_dir: Path, dry_run: bool) -> FigureArtifact:
    figure_id = "fig03_lora_vs_gemini"
    base_path = output_dir / figure_id
    tight_base_path = output_dir / f"{figure_id}_tight"
    data_path = data_dir / "lora_vs_gemini.csv"
    tight_data_path = data_dir / "lora_vs_gemini_tight.csv"
    tight_field_data_path = data_dir / "lora_vs_gemini_tight_field_scores.csv"
    rows = _prepare_fig03_rows()
    tight_rows = _prepare_fig03_tight_rows()
    tight_field_rows = _prepare_fig03_tight_field_rows()
    if not dry_run:
        _write_csv(data_path, rows, ["system", "display_name", "top10", "top1", "mrr10", "gt_in_pool", "source_trace", "is_best"])
        _write_csv(
            tight_data_path,
            tight_rows,
            ["system", "display_name", "top10", "top1", "mrr10", "gt_in_pool", "source_trace", "is_best", "rerank_delta_top1_pp"],
        )
        _write_csv(
            tight_field_data_path,
            tight_field_rows,
            ["model_label", "metric", "metric_label", "value", "source_trace"],
        )
        _save_figure(_plot_fig03(rows), base_path)
        _save_figure(_plot_fig03_tight(tight_rows, tight_field_rows), tight_base_path)
    return FigureArtifact(
        figure_id=figure_id,
        title="Strict Retrieval Comparison",
        claim="Recovered Gemini narrows the gap, but the stronger G-series models still produce the best strict retrieval quality.",
        rq=("RQ1", "RQ2"),
        png_path=base_path.with_suffix(".png"),
        pdf_path=base_path.with_suffix(".pdf"),
        data_path=data_path,
        sources=(
            LORA_VS_GEMINI_SOURCE,
            GEMINI_V2_RETRIEVAL_SUMMARY,
            G7_RETRIEVAL_SUMMARY,
            G9_RESNET_F4_RETRIEVAL_SUMMARY,
            ORACLE_TOPOLOGY_METRICS,
            GEMINI_V2_EVAL_SOURCE,
            G3_EVAL_SOURCE,
            G9_F4_RESNET_EVAL_SOURCE,
        ),
    )


def _generate_fig04(output_dir: Path, data_dir: Path, dry_run: bool) -> FigureArtifact:
    figure_id = "fig04_multimodal_alignment_gain"
    base_path = output_dir / figure_id
    tight_base_path = output_dir / "fig04_multimodal_alignment_gain_tight"
    data_path = data_dir / "multimodal_alignment_gain.csv"
    modality_rows, field_rows = _prepare_fig04_rows()
    csv_rows: list[dict[str, Any]] = []
    for row in modality_rows:
        csv_rows.append(
            {
                "section": "modality",
                "slice": row["slice"],
                "slice_label": row["slice_label"],
                "model": row["model"],
                "model_label": row["model_label"],
                "metric": "one_hop_spatial_accuracy",
                "metric_label": "One-hop spatial accuracy",
                "value": row["one_hop_spatial_accuracy"],
                "class_acc": row["class_acc"],
                "storey_acc": row["storey_acc"],
                "predicate_recall": row["predicate_recall"],
                "direction_accuracy": row["direction_accuracy"],
                "source_trace": row["source_trace"],
            }
        )
    for row in field_rows:
        csv_rows.append(
            {
                "section": "field",
                "slice": "",
                "slice_label": "",
                "model": row["model_label"],
                "model_label": row["model_label"],
                "metric": row["metric"],
                "metric_label": row["metric_label"],
                "value": row["value"],
                "class_acc": "",
                "storey_acc": "",
                "predicate_recall": "",
                "direction_accuracy": "",
                "source_trace": row["source_trace"],
            }
        )
    if not dry_run:
        _write_csv(
            data_path,
            csv_rows,
            [
                "section",
                "slice",
                "slice_label",
                "model",
                "model_label",
                "metric",
                "metric_label",
                "value",
                "class_acc",
                "storey_acc",
                "predicate_recall",
                "direction_accuracy",
                "source_trace",
            ],
        )
        # Panel A (16:9) — multimodal grounding story
        _save_figure(_plot_fig04(modality_rows, field_rows), base_path)
        _save_figure(_plot_fig04(modality_rows, field_rows), tight_base_path)
        # Panel B (4:3) — helper motivation, G8 position-context / G9 LoRA / Gemini only
        panel_b_path = output_dir / "fig04b_helper_motivation"
        _save_figure(_plot_fig04b_helper_motivation(field_rows), panel_b_path)
    return FigureArtifact(
        figure_id=figure_id,
        title="Multimodal Alignment and Richer Fields",
        claim="Site and floorplan evidence drive one-hop grounding, while the G9 OpenCV/ResNet pipeline adds explicit position and size cues that were previously hidden.",
        rq=("RQ1",),
        png_path=base_path.with_suffix(".png"),
        pdf_path=base_path.with_suffix(".pdf"),
        data_path=data_path,
        sources=(MODALITY_SOURCE, G7_EVAL_SOURCE, GEMINI_V2_EVAL_SOURCE, G9_F4_RESNET_EVAL_SOURCE, GT_G7_LABEL_SOURCE, GT_G9_LABEL_SOURCE),
    )


def _generate_fig05(output_dir: Path, data_dir: Path, dry_run: bool) -> FigureArtifact:
    figure_id = "fig05_graph_rag_evidence_dependent"
    base_path = output_dir / figure_id
    data_path = data_dir / "graph_rag_evidence_dependent.csv"
    rows = _prepare_fig05_rows()
    if not dry_run:
        _write_csv(
            data_path,
            rows,
            ["system", "pipeline", "source_trace", "top10", "top1", "mrr10", "subset_n", "subset_before_top1", "subset_after_top1"],
        )
        _save_figure(_plot_fig05(rows), base_path)
    return FigureArtifact(
        figure_id=figure_id,
        title="Rerank on P1 and G9 Pipelines",
        claim="Reranking helps both the coarse P1 storey/class control and the richer G9 OpenCV/ResNet pipeline, with the strongest gains appearing when the right candidate is already in the pool.",
        rq=("RQ2",),
        png_path=base_path.with_suffix(".png"),
        pdf_path=base_path.with_suffix(".pdf"),
        data_path=data_path,
        sources=(GRAPH_RAG_SOURCE, PHASE6_T2_SUMMARY_SOURCE),
    )


def _generate_fig06(output_dir: Path, data_dir: Path, dry_run: bool) -> FigureArtifact:
    figure_id = "fig06_summary_findings_table"
    base_path = output_dir / figure_id
    data_path = data_dir / "summary_findings_table.csv"
    rows = _prepare_fig06_rows()
    if not dry_run:
        _write_csv(data_path, rows, ["section", "finding", "interpretation", "rq_link"])
        _save_figure(_plot_fig06(rows), base_path)
    return FigureArtifact(
        figure_id=figure_id,
        title="Summary Findings Table",
        claim="The strongest overall behavior comes from assigning extraction, filtering, and reranking to different field roles rather than forcing one mechanism to do everything.",
        rq=("RQ1", "RQ2"),
        png_path=base_path.with_suffix(".png"),
        pdf_path=base_path.with_suffix(".pdf"),
        data_path=data_path,
        sources=tuple(),
    )


def _generate_fig07(output_dir: Path, data_dir: Path, dry_run: bool) -> FigureArtifact:
    figure_id = "fig07_retrieval_pipeline_comparison"
    base_path = output_dir / figure_id
    data_path = data_dir / "retrieval_pipeline_comparison.csv"
    rows = _prepare_fig07_rows()
    if not dry_run:
        _write_csv(data_path, rows, ["system", "display_name", "top10", "top1", "mrr10", "gt_in_pool", "source_trace", "is_best"])
        _save_figure(_plot_fig07(rows), base_path)
    return FigureArtifact(
        figure_id=figure_id,
        title="Retrieval Pipeline Comparison",
        claim="The G9 OpenCV/ResNet pipeline with rerank now provides the strongest overall early-rank retrieval among the compared pipeline variants.",
        rq=("RQ1", "RQ2"),
        png_path=base_path.with_suffix(".png"),
        pdf_path=base_path.with_suffix(".pdf"),
        data_path=data_path,
        sources=(G8_RETRIEVAL_SUMMARY, G7_RETRIEVAL_SUMMARY, G9_RESNET_F4_RETRIEVAL_SUMMARY, GRAPH_RAG_SOURCE, PHASE6_T2_SUMMARY_SOURCE),
    )


def _generate_backup_model_capability_proof(output_dir: Path, data_dir: Path, dry_run: bool) -> FigureArtifact:
    figure_id = "backup_model_capability_proof"
    base_path = output_dir / figure_id
    data_path = data_dir / "backup_model_capability_proof.csv"
    topology_rows, size_rows, hard_filter_rows, retrieval_rows = _prepare_backup_model_capability_rows()
    csv_rows: list[dict[str, Any]] = []
    csv_rows.extend(topology_rows)
    csv_rows.extend(size_rows)
    csv_rows.extend(hard_filter_rows)
    for row in retrieval_rows:
        csv_rows.append(
            {
                "section": "retrieval",
                "system": row["system"],
                "metric": "top10",
                "metric_label": "Top-10 retrieval",
                "value": row["top10"],
                "source_trace": row["source_trace"],
                "top1": row["top1"],
                "mrr10": row["mrr10"],
                "gt_in_pool": row["gt_in_pool"],
            }
        )
    if not dry_run:
        _write_csv(
            data_path,
            csv_rows,
            [
                "section",
                "model_label",
                "system",
                "case_group",
                "case_group_label",
                "metric",
                "metric_label",
                "value",
                "count",
                "effect",
                "support_n",
                "top1",
                "mrr10",
                "gt_in_pool",
                "source_trace",
            ],
        )
        _save_figure(
            _plot_backup_model_capability_proof(topology_rows, size_rows, hard_filter_rows, retrieval_rows),
            base_path,
        )
    return FigureArtifact(
        figure_id=figure_id,
        title="Technical Capability Proof",
        claim="The learned model reliably acquires coarse topology fields, but size remains a fragile perceptual cue whose wrong emissions are more harmful than helpful when used as a hard filter.",
        rq=("RQ1", "RQ2"),
        png_path=base_path.with_suffix(".png"),
        pdf_path=base_path.with_suffix(".pdf"),
        data_path=data_path,
        sources=(
            GEMINI_V2_EVAL_SOURCE,
            G7_EVAL_SOURCE,
            G9_LORA_ONLY_EVAL_SOURCE,
            GT_G7_LABEL_SOURCE,
            GT_G9_LABEL_SOURCE,
            CLUSTER_CLASSIFIER_TEST_METRICS,
            G8_RETRIEVAL_SUMMARY,
            G9_LORA_ONLY_RETRIEVAL_SUMMARY,
            G9_RESNET_BAND_RETRIEVAL_SUMMARY,
        ),
    )


GENERATORS = {
    "fig00_symbolic_reasoning_trace": _generate_fig00,
    "fig00c_p0_topk_flow": _generate_fig00c,
    "fig01_oracle_symbolic_ceiling": _generate_fig01,
    "fig03_lora_vs_gemini": _generate_fig03,
    "fig04_multimodal_alignment_gain": _generate_fig04,
    "fig05_graph_rag_evidence_dependent": _generate_fig05,
    "fig06_summary_findings_table": _generate_fig06,
    "fig07_retrieval_pipeline_comparison": _generate_fig07,
    "backup_model_capability_proof": _generate_backup_model_capability_proof,
}


def _remove_retired_artifacts(output_dir: Path, data_dir: Path) -> None:
    retired = [
        output_dir / "fig02_fingerprint_ladder.png",
        output_dir / "fig02_fingerprint_ladder.pdf",
        data_dir / "fingerprint_ladder.csv",
    ]
    for path in retired:
        if path.exists():
            path.unlink()


def _write_manifest(output_dir: Path, data_dir: Path, artifacts: list[FigureArtifact], dry_run: bool) -> Path:
    manifest_path = output_dir / "plot_manifest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": _repo_rel(output_dir),
        "data_dir": _repo_rel(data_dir),
        "source_cases": _repo_rel(DEFAULT_CASES),
        "dry_run": dry_run,
        "figures": [_manifest_entry(artifact) for artifact in artifacts],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs without writing files.")
    parser.add_argument("--figures", nargs="*", choices=FIGURE_IDS, default=None, help="Optional subset of figures to generate.")
    args = parser.parse_args()

    _configure_matplotlib()
    selected = list(args.figures) if args.figures else list(MAIN_FIGURE_IDS)
    artifacts: list[FigureArtifact] = []

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.data_dir.mkdir(parents=True, exist_ok=True)
        _remove_retired_artifacts(args.output_dir, args.data_dir)

    for figure_id in selected:
        artifact = GENERATORS[figure_id](args.output_dir, args.data_dir, args.dry_run)
        artifacts.append(artifact)
        print(f"[{figure_id}] {'validated' if args.dry_run else 'generated'}")
        print(f"  data: {_repo_rel(artifact.data_path)}")
        print(f"  png : {_repo_rel(artifact.png_path)}")
        print(f"  pdf : {_repo_rel(artifact.pdf_path)}")

    if args.dry_run:
        print("\nDry run complete. No files were written.")
        return

    manifest_path = _write_manifest(args.output_dir, args.data_dir, artifacts, args.dry_run)
    print(f"\nWrote manifest: {_repo_rel(manifest_path)}")


if __name__ == "__main__":
    main()
