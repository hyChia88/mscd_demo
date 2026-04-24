#!/usr/bin/env python3
"""Shared group registry for LoRA6-v2 dual-track evaluation."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENT_ROOT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331"
METRICS_DIR = EXPERIMENT_ROOT / "metrics"
LOGS_DIR = EXPERIMENT_ROOT / "logs"

ADAPTER_ROOT = PROJECT_ROOT / "models" / "lora6_v2_ap_20260331"


GROUP_DISPLAY = {
    "g0_canonical": "G0 Canonical",
    "g1_fullaug": "G1 FullAug",
    "g2_fullaug_lowlr": "G2 FullAug LowLR",
    "g3_fullaug_r32": "G3 FullAug r32",
    "g4_ultimate": "G4 Ultimate",
    "g7_position_context": "G7 Position Context",
    "g8_posctx_dim": "G8 PosCtx+Dim",
    "g9_opencv_cluster": "G9 OpenCV+SizeCluster",
    "g6_baseline": "G6 Baseline",
    "gemini_ap": "Gemini AP",
    "gemini_ap_v2": "Gemini AP v2",
    "gemini_unified": "Gemini Unified",
    "lora5_r16_unified": "LoRA5-r16",
    "lora5_r32_unified": "LoRA5-r32",
    "lora2_unified": "LoRA2",
}

TRACK_A_ORDER = [
    "g0_canonical",
    "g1_fullaug",
    "g2_fullaug_lowlr",
    "g3_fullaug_r32",
    "g4_ultimate",
    "g7_position_context",
    "g9_opencv_cluster",
    "g6_baseline",
    "gemini_ap",
]

G_SERIES_ORDER = [g for g in TRACK_A_ORDER if g != "gemini_ap"]

TRACK_B_ORDER = [
    "gemini_unified",
    "lora5_r16_unified",
    "lora5_r32_unified",
    "lora2_unified",
]

TRACK_B2_ORDER = [
    "g0_canonical",
    "g1_fullaug",
    "g2_fullaug_lowlr",
    "g3_fullaug_r32",
    "g4_ultimate",
    "g7_position_context",
    "g8_posctx_dim",
    "g9_opencv_cluster",
    "g6_baseline",
    "gemini_ap",
    "gemini_ap_v2",
]

AP_HELDOUT_E2E_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"


TRACK_A_PREDICTION_FILES = {
    key: EXPERIMENT_ROOT / f"{key}__ap_eval.jsonl"
    for key in TRACK_A_ORDER
}

TRACK_A_ADAPTERS = {
    "g0_canonical": ADAPTER_ROOT / "g0_canonical" / "best",
    "g1_fullaug": ADAPTER_ROOT / "g1_fullaug" / "best",
    "g2_fullaug_lowlr": ADAPTER_ROOT / "g2_fullaug_lowlr" / "best",
    "g3_fullaug_r32": ADAPTER_ROOT / "g3_fullaug_r32" / "best",
    "g4_ultimate": ADAPTER_ROOT / "g4_ultimate" / "best",
    "g7_position_context": ADAPTER_ROOT / "g7_position_context" / "best",
    "g9_opencv_cluster": ADAPTER_ROOT / "g9_opencv_cluster" / "best",
    "g6_baseline": ADAPTER_ROOT / "g6_baseline" / "checkpoint-20",
}


# Remote adapter dirs already used by Modal evaluation scripts.
UNIFIED_REMOTE_ADAPTERS = {
    "g0_canonical": "/mscd-lora-v6-g0-canonical/best",
    "g1_fullaug": "/mscd-lora-v6-g1-fullaug/best",
    "g2_fullaug_lowlr": "/mscd-lora-v6-g2-fullaug-lowlr/best",
    "g3_fullaug_r32": "/mscd-lora-v6-g3-fullaug-r32/best",
    "g4_ultimate": "/mscd-lora-v6-g4-ultimate/best",
    "g7_position_context": "/mscd-lora-v6-g7-position-context/best",
    "g9_opencv_cluster": "/mscd-lora-v6-g9-opencv-cluster/best",
    "g6_baseline": "/mscd-lora-v6-g6-baseline/checkpoint-20",
    "lora5_r16_unified": "/mscd-unified-eval/v5_complex_lora_qwen",
    "lora5_r32_unified": "/mscd-unified-eval/v5_lora_qwen_r32",
    "lora2_unified": "/mscd-unified-eval/v2_lora_qwen",
}


def display_name(group_key: str) -> str:
    return GROUP_DISPLAY.get(group_key, group_key)
