from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.neurosym.constraints_extractor_lora import LoRAConstraintsExtractor
from src.neurosym.constraints_to_query import QueryPlanner
from src.neurosym.floorplan_counter import (
    FloorplanCountResult,
    FloorplanCounter,
)
from src.neurosym.types import Constraints, SpatialTriplet


def _make_synthetic_wall_scene() -> np.ndarray:
    image = np.full((480, 480, 3), 255, dtype=np.uint8)
    wall_p1 = np.array([60.0, 240.0], dtype=np.float32)
    wall_p2 = np.array([420.0, 240.0], dtype=np.float32)
    wall_dir = wall_p2 - wall_p1
    wall_dir /= np.linalg.norm(wall_dir)

    cv2.line(
        image,
        tuple(wall_p1.astype(int)),
        tuple(wall_p2.astype(int)),
        (165, 165, 165),
        3,
        lineType=cv2.LINE_AA,
    )

    colors = [
        (255, 0, 0),
        (0, 180, 0),
        (0, 0, 255),
        (255, 0, 0),
    ]
    target_center = None
    for idx, t in enumerate((0.16, 0.38, 0.60, 0.82)):
        center = wall_p1 + (wall_p2 - wall_p1) * t
        if idx == 2:
            target_center = center
        half = wall_dir * 20.0
        start = tuple(np.round(center - half).astype(int))
        end = tuple(np.round(center + half).astype(int))
        cv2.line(image, start, end, colors[idx], 8)

    assert target_center is not None
    center_xy = tuple(np.round(target_center).astype(int))
    cv2.circle(image, center_xy, 9, (0, 0, 255), -1)
    return image


def test_floorplan_counter_counts_target_wall(tmp_path):
    full = _make_synthetic_wall_scene()
    patch = full[170:310, 215:355].copy()

    full_path = tmp_path / "full.png"
    patch_path = tmp_path / "patch.png"
    assert cv2.imwrite(str(full_path), full)
    assert cv2.imwrite(str(patch_path), patch)

    counter = FloorplanCounter()
    result = counter.count_from_paths(patch_path, full_path)

    assert result is not None
    assert result.mode == "full_floorplan"
    assert result.position == 3
    assert result.total == 4
    assert result.confidence >= 0.8
    assert result.position_context == "3rd of 4 openings on the same wall"


def test_floorplan_counter_patch_only_fallback_is_low_confidence(tmp_path):
    full = _make_synthetic_wall_scene()
    patch = full[170:310, 215:355].copy()
    patch_path = tmp_path / "patch_only.png"
    assert cv2.imwrite(str(patch_path), patch)

    counter = FloorplanCounter()
    result = counter.count_from_paths(patch_path, None)

    assert result is not None
    assert result.mode == "patch_only"
    assert result.total >= 1
    assert result.confidence <= 0.45


def test_query_planner_uses_position_confidence_for_hard_slot_filter():
    planner = QueryPlanner()
    constraints = Constraints(
        storey_name="3",
        ifc_class="IfcWindow",
        position_context="3rd of 4 openings on the same wall",
        position_context_confidence=0.45,
        position_context_source="opencv",
        spatial_relations=[
            SpatialTriplet(
                subject_type="IfcWindow",
                predicate="FILLS",
                object_type="IfcWallStandardCase",
                confidence=1.0,
            )
        ],
    )

    plan = planner.plan(constraints)[0]
    assert "position_context" in plan.params
    assert "position_index" not in plan.params

    strong_constraints = constraints.model_copy(
        update={"position_context_confidence": 0.92}
    )
    strong_plan = planner.plan(strong_constraints)[0]
    assert strong_plan.params["position_index"] == 3
    assert strong_plan.params["position_total"] == 4


def test_lora_user_text_includes_opencv_counting_block():
    extractor = LoRAConstraintsExtractor(adapter_path=None)
    opencv_result = FloorplanCountResult(
        position=2,
        total=6,
        confidence=0.91,
        position_context="2nd of 6 openings on the same wall",
    )
    case = {
        "inputs": {
            "project_context": {"4d_task_status": "TASK_001", "project_phase": "Fit-out"},
            "chat_history": [{"role": "Inspector", "text": "Check this opening."}],
        },
        "query_text": "Inspect this.",
    }

    user_text = extractor._build_user_text(case, opencv_result)

    assert "[OpenCV Counting]" in user_text
    assert "2nd of 6 openings on the same wall" in user_text
    assert "confidence: 0.91" in user_text
