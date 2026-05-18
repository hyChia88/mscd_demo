"""
OpenCV-based floorplan slot counting for Phase 6.

The counter is intentionally conservative:
- If a full-floorplan reference can be resolved, localize the patch inside it.
- Detect the dominant wall line near the patch center.
- Count elongated coloured opening symbols aligned with that wall.
- Fall back to patch-only counting at low confidence when the wider context is
  unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# PROJECT_ROOT is only consulted by `_resolve_input_path` /
# `_resolve_case_full_floorplan` (the case-driven entry point). Modal-baked
# copies of this module live at /perception/floorplan_counter.py with fewer
# than 4 ancestors — fall back to None there so module import does not crash.
def _maybe_project_root() -> Optional[Path]:
    try:
        return Path(__file__).resolve().parents[3]
    except IndexError:
        return None


PROJECT_ROOT = _maybe_project_root()
_POSITION_CONTEXT_RE = re.compile(
    r"(?P<index>\d+)(?:st|nd|rd|th)\s+of\s+(?P<total>\d+)\s+openings",
    flags=re.IGNORECASE,
)
_SKELETON_ID_RE = re.compile(r"(?P<model>AP|BH|DXA)_SK_\d+", flags=re.IGNORECASE)
_DATASET_BY_MODEL = {
    "AP": "synth_v0.5_ap",
    "BH": "synth_v0.5_bh",
    "DXA": "synth_v0.5_dxa",
}


@dataclass
class FloorplanCountResult:
    position: int
    total: int
    confidence: float
    position_context: str
    mode: str = "full_floorplan"
    match_score: float = 0.0
    matched_bbox: Optional[Dict[str, int]] = None
    patch_opening_count: int = 0
    debug: Dict[str, Any] = field(default_factory=dict)


def parse_position_context_tuple(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    match = _POSITION_CONTEXT_RE.search(str(value).strip())
    if not match:
        return None
    return int(match.group("index")), int(match.group("total"))


def ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def format_position_context(position: int, total: int) -> str:
    return f"{ordinal(position)} of {total} openings on the same wall"


def merge_position_context(
    model_position_context: Optional[str],
    opencv_result: Optional[FloorplanCountResult],
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    model_value = (model_position_context or "").strip() or None
    if opencv_result is None:
        return model_value, None, ("model" if model_value else None)

    opencv_value = opencv_result.position_context
    if not model_value:
        return opencv_value, opencv_result.confidence, "opencv"

    model_slot = parse_position_context_tuple(model_value)
    opencv_slot = (opencv_result.position, opencv_result.total)
    if model_slot == opencv_slot:
        return opencv_value, max(opencv_result.confidence, 0.8), "opencv+model"

    if opencv_result.confidence >= 0.9:
        return opencv_value, max(0.8, opencv_result.confidence - 0.1), "opencv_override"

    return model_value, None, "model"


class FloorplanCounter:
    def __init__(
        self,
        image_dir: str = "",
        *,
        min_template_score: float = 0.30,
        high_confidence_threshold: float = 0.80,
    ):
        self.image_dir = Path(image_dir) if image_dir else None
        self.min_template_score = float(min_template_score)
        self.high_confidence_threshold = float(high_confidence_threshold)

    def count_from_case(self, case: Dict[str, Any]) -> Optional[FloorplanCountResult]:
        patch_path = self._resolve_case_patch(case)
        if patch_path is None:
            return None

        full_floorplan_path = self._resolve_case_full_floorplan(case, patch_path)
        return self.count_from_paths(patch_path, full_floorplan_path)

    def count_from_paths(
        self,
        patch_path: Path | str,
        full_floorplan_path: Path | str | None = None,
    ) -> Optional[FloorplanCountResult]:
        patch = self._load_image(patch_path)
        if patch is None:
            return None

        full = self._load_image(full_floorplan_path) if full_floorplan_path else None
        mode = "patch_only"
        match_score = 1.0
        matched_bbox = {
            "x": 0,
            "y": 0,
            "w": int(patch.shape[1]),
            "h": int(patch.shape[0]),
        }

        if full is not None:
            localized = self._localize_patch(patch, full)
            if localized is None:
                return None
            matched_bbox, match_score = localized
            if match_score < self.min_template_score:
                return None
            mode = "full_floorplan"
        else:
            full = patch

        patch_center = (
            matched_bbox["x"] + (matched_bbox["w"] / 2.0),
            matched_bbox["y"] + (matched_bbox["h"] / 2.0),
        )
        wall_line = self._detect_wall_line(
            full,
            patch_center,
            search_span=max(matched_bbox["w"], matched_bbox["h"]) * 1.8,
        )
        if wall_line is None:
            return None

        openings = self._detect_openings_on_wall(full, wall_line)
        if not openings:
            return None

        target_idx = self._nearest_opening_index(openings, patch_center)
        position = target_idx + 1
        total = len(openings)
        patch_opening_count = 0
        consistency_score = 0.5

        if mode == "full_floorplan":
            patch_wall_line = self._detect_wall_line(
                patch,
                (patch.shape[1] / 2.0, patch.shape[0] / 2.0),
                search_span=max(patch.shape[0], patch.shape[1]) * 0.9,
            )
            patch_openings = (
                self._detect_openings_on_wall(patch, patch_wall_line)
                if patch_wall_line is not None
                else []
            )
            patch_opening_count = len(patch_openings)
            consistency_score = self._patch_consistency_score(
                wall_line=wall_line,
                matched_bbox=matched_bbox,
                full_openings=openings,
                patch_openings=patch_openings,
            )

        confidence = self._estimate_confidence(
            mode=mode,
            match_score=match_score,
            wall_line=wall_line,
            openings=openings,
            target_idx=target_idx,
            patch_opening_count=patch_opening_count,
            consistency_score=consistency_score,
        )

        return FloorplanCountResult(
            position=position,
            total=total,
            confidence=confidence,
            position_context=format_position_context(position, total),
            mode=mode,
            match_score=match_score,
            matched_bbox=matched_bbox,
            patch_opening_count=patch_opening_count,
            debug={
                "wall_line": {
                    "x1": round(float(wall_line[0][0]), 2),
                    "y1": round(float(wall_line[0][1]), 2),
                    "x2": round(float(wall_line[1][0]), 2),
                    "y2": round(float(wall_line[1][1]), 2),
                },
                "consistency_score": round(float(consistency_score), 3),
            },
        )

    def _resolve_case_patch(self, case: Dict[str, Any]) -> Optional[Path]:
        patch = (case.get("inputs") or {}).get("floorplan_patch")
        return self._resolve_input_path(patch)

    def _resolve_case_full_floorplan(
        self,
        case: Dict[str, Any],
        patch_path: Path,
    ) -> Optional[Path]:
        inputs = case.get("inputs") or {}
        for key in ("full_floorplan", "project_floorplan", "floorplan_full"):
            resolved = self._resolve_input_path(inputs.get(key))
            if resolved is not None:
                return resolved
        return self._infer_full_floorplan_path(patch_path)

    def _resolve_input_path(self, raw_path: Any) -> Optional[Path]:
        if not raw_path:
            return None
        raw = Path(str(raw_path))

        candidates: List[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(raw)
            if PROJECT_ROOT is not None:
                candidates.append(PROJECT_ROOT / raw)
                candidates.append(PROJECT_ROOT / "data_curation" / raw)
            if self.image_dir is not None:
                candidates.append(self.image_dir / raw)
                parts = list(raw.parts)
                if parts and parts[0] == "datasets":
                    candidates.append(self.image_dir / Path(*parts[1:]))
                candidates.append(self.image_dir / raw.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _infer_full_floorplan_path(self, patch_path: Path) -> Optional[Path]:
        name = patch_path.name
        base_id = re.sub(r"_scale_[MSL](?=\.[^.]+$)", "", patch_path.stem)
        sibling = patch_path.parent.parent / "floorplans" / f"{base_id}_floorplan.png"
        if sibling.exists():
            return sibling.resolve()

        match = _SKELETON_ID_RE.search(name)
        if not match:
            return None

        skeleton_id = match.group(0).upper()
        dataset_name = _DATASET_BY_MODEL.get(match.group("model").upper())
        if not dataset_name:
            return None

        search_roots: List[Path] = []
        if PROJECT_ROOT is not None:
            search_roots.append(PROJECT_ROOT / "data_curation" / "datasets")
        if self.image_dir is not None:
            search_roots.extend(
                [
                    self.image_dir,
                    self.image_dir / "datasets",
                ]
            )

        seen = set()
        for root in search_roots:
            root_key = str(root)
            if root_key in seen:
                continue
            seen.add(root_key)
            candidate = root / dataset_name / "floorplans" / f"{skeleton_id}_floorplan.png"
            if candidate.exists():
                return candidate.resolve()
        return None

    def _load_image(self, path_like: Path | str | None) -> Optional[np.ndarray]:
        if path_like is None:
            return None
        path = Path(path_like)
        if not path.exists():
            return None
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return image

    def _localize_patch(
        self,
        patch: np.ndarray,
        full: np.ndarray,
    ) -> Optional[Tuple[Dict[str, int], float]]:
        if full.shape[0] < patch.shape[0] or full.shape[1] < patch.shape[1]:
            return None

        patch_edges = self._preprocess_for_match(patch)
        full_edges = self._preprocess_for_match(full)
        response = cv2.matchTemplate(full_edges, patch_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        x, y = int(max_loc[0]), int(max_loc[1])
        return (
            {
                "x": x,
                "y": y,
                "w": int(patch.shape[1]),
                "h": int(patch.shape[0]),
            },
            float(max_val),
        )

    def _preprocess_for_match(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(gray, 40, 120)

    def _detect_wall_line(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
        *,
        search_span: float,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        half = int(max(32, search_span / 2.0))
        cx, cy = int(center[0]), int(center[1])
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(image.shape[1], cx + half)
        y1 = min(image.shape[0], cy + half)
        crop = edges[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        lines = cv2.HoughLinesP(
            crop,
            rho=1,
            theta=np.pi / 180.0,
            threshold=20,
            minLineLength=max(14, min(crop.shape[:2]) // 6),
            maxLineGap=18,
        )
        if lines is None:
            return None

        best_line = None
        best_score = -1e9
        for line in lines[:, 0, :]:
            p1 = (float(line[0] + x0), float(line[1] + y0))
            p2 = (float(line[2] + x0), float(line[3] + y0))
            length = self._segment_length(p1, p2)
            if length < 16:
                continue
            dist = self._distance_point_to_line(center, (p1, p2))
            score = length - (dist * 2.5)
            if score > best_score:
                best_score = score
                best_line = (p1, p2)
        return best_line

    def _detect_openings_on_wall(
        self,
        image: np.ndarray,
        wall_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> List[Dict[str, float]]:
        if wall_line is None:
            return []

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] >= 45).astype(np.uint8) * 255
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        wall_angle = self._wall_angle_deg(wall_line)
        wall_origin = np.array(wall_line[0], dtype=np.float32)
        wall_dir = self._unit_vector(wall_line)
        line_tol = max(18.0, min(image.shape[:2]) * 0.04)

        components: List[Dict[str, float]] = []
        dropped = {"area": 0, "too_few_pts": 0, "major_or_aspect": 0, "line_dist": 0, "orientation": 0}
        for idx in range(1, n_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            # T1.2 filter-loosening: full-storey renders have thinner lines than the
            # 12m patches the original thresholds were tuned on. Drop area to 10 and
            # aspect to 1.3 so small but genuine openings aren't filtered out.
            if area < 10:
                dropped["area"] += 1
                continue

            ys, xs = np.where(labels == idx)
            pts = np.column_stack((xs, ys)).astype(np.float32)
            if len(pts) < 5:
                dropped["too_few_pts"] += 1
                continue

            rect = cv2.minAreaRect(pts)
            (_, _), (w, h), _ = rect
            major = max(float(w), float(h))
            minor = max(1.0, min(float(w), float(h)))
            aspect = major / minor
            if major < 8.0 or aspect < 1.3:
                dropped["major_or_aspect"] += 1
                continue

            cx, cy = float(centroids[idx][0]), float(centroids[idx][1])
            dist = self._distance_point_to_line((cx, cy), wall_line)
            if dist > line_tol:
                dropped["line_dist"] += 1
                continue

            orientation = self._rect_angle_deg(rect)
            if self._angle_delta_deg(orientation, wall_angle) > 25.0:
                dropped["orientation"] += 1
                continue

            projection = float(np.dot(np.array([cx, cy], dtype=np.float32) - wall_origin, wall_dir))
            components.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "projection": projection,
                    "major": major,
                }
            )

        if not components:
            return []

        components.sort(key=lambda item: item["projection"])
        merge_gap = max(22.0, np.median([c["major"] for c in components]) * 0.9)
        clusters: List[List[Dict[str, float]]] = []
        for comp in components:
            if not clusters:
                clusters.append([comp])
                continue
            prev_proj = np.mean([item["projection"] for item in clusters[-1]])
            if abs(comp["projection"] - prev_proj) <= merge_gap:
                clusters[-1].append(comp)
            else:
                clusters.append([comp])

        merged: List[Dict[str, float]] = []
        for cluster in clusters:
            merged.append(
                {
                    "cx": float(np.mean([item["cx"] for item in cluster])),
                    "cy": float(np.mean([item["cy"] for item in cluster])),
                    "projection": float(np.mean([item["projection"] for item in cluster])),
                    "major": float(np.max([item["major"] for item in cluster])),
                }
            )
        return merged

    def _nearest_opening_index(
        self,
        openings: Sequence[Dict[str, float]],
        point: Tuple[float, float],
    ) -> int:
        px, py = point
        return min(
            range(len(openings)),
            key=lambda idx: (openings[idx]["cx"] - px) ** 2 + (openings[idx]["cy"] - py) ** 2,
        )

    def _patch_consistency_score(
        self,
        *,
        wall_line: Tuple[Tuple[float, float], Tuple[float, float]],
        matched_bbox: Dict[str, int],
        full_openings: Sequence[Dict[str, float]],
        patch_openings: Sequence[Dict[str, float]],
    ) -> float:
        patch_count = len(patch_openings)
        if patch_count == 0:
            return 0.4

        wall_origin = np.array(wall_line[0], dtype=np.float32)
        wall_dir = self._unit_vector(wall_line)
        corners = [
            np.array([matched_bbox["x"], matched_bbox["y"]], dtype=np.float32),
            np.array([matched_bbox["x"] + matched_bbox["w"], matched_bbox["y"]], dtype=np.float32),
            np.array([matched_bbox["x"], matched_bbox["y"] + matched_bbox["h"]], dtype=np.float32),
            np.array(
                [matched_bbox["x"] + matched_bbox["w"], matched_bbox["y"] + matched_bbox["h"]],
                dtype=np.float32,
            ),
        ]
        min_proj = min(float(np.dot(corner - wall_origin, wall_dir)) for corner in corners)
        max_proj = max(float(np.dot(corner - wall_origin, wall_dir)) for corner in corners)
        visible = [
            item
            for item in full_openings
            if min_proj - 24.0 <= item["projection"] <= max_proj + 24.0
        ]
        visible_count = len(visible)
        if visible_count == 0:
            return 0.2
        if patch_count <= visible_count:
            return 1.0
        return max(0.3, visible_count / float(patch_count))

    def _estimate_confidence(
        self,
        *,
        mode: str,
        match_score: float,
        wall_line: Tuple[Tuple[float, float], Tuple[float, float]],
        openings: Sequence[Dict[str, float]],
        target_idx: int,
        patch_opening_count: int,
        consistency_score: float,
    ) -> float:
        template_score = 1.0 if mode == "patch_only" else np.clip((match_score - 0.25) / 0.5, 0.0, 1.0)
        wall_score = 1.0 if wall_line is not None else 0.0
        density_score = min(1.0, len(openings) / 4.0)

        target_opening = openings[target_idx]
        other_distances = [
            math.hypot(target_opening["cx"] - op["cx"], target_opening["cy"] - op["cy"])
            for idx, op in enumerate(openings)
            if idx != target_idx
        ]
        target_score = 1.0
        if other_distances:
            nearest_other = min(other_distances)
            target_score = np.clip(nearest_other / 40.0, 0.35, 1.0)

        confidence = (
            0.2
            + (0.35 * float(template_score))
            + (0.15 * float(wall_score))
            + (0.15 * float(density_score))
            + (0.15 * float(target_score))
            + (0.20 * float(consistency_score))
        )

        if mode == "patch_only":
            confidence = min(confidence, 0.45)
            if patch_opening_count <= 1:
                confidence = min(confidence, 0.35)

        return float(np.clip(confidence, 0.0, 0.99))

    @staticmethod
    def _segment_length(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def _distance_point_to_line(
        point: Tuple[float, float],
        line: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> float:
        (x0, y0), (x1, y1) = line
        px, py = point
        dx, dy = x1 - x0, y1 - y0
        denom = math.hypot(dx, dy)
        if denom < 1e-6:
            return float("inf")
        return abs((dy * px) - (dx * py) + (x1 * y0) - (y1 * x0)) / denom

    @staticmethod
    def _unit_vector(
        line: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> np.ndarray:
        (x0, y0), (x1, y1) = line
        vec = np.array([x1 - x0, y1 - y0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        return vec / norm

    @staticmethod
    def _wall_angle_deg(
        line: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> float:
        (x0, y0), (x1, y1) = line
        return math.degrees(math.atan2(y1 - y0, x1 - x0))

    @staticmethod
    def _rect_angle_deg(rect: Tuple[Tuple[float, float], Tuple[float, float], float]) -> float:
        (_, _), (w, h), angle = rect
        if h > w:
            angle += 90.0
        return float(angle)

    @staticmethod
    def _angle_delta_deg(a: float, b: float) -> float:
        delta = abs(a - b) % 180.0
        return min(delta, 180.0 - delta)

    # ── Phase 6 T1.1: full-storey entry point ─────────────────────────────────
    def count_from_full_storey(
        self,
        storey_png_path: Path | str,
        calibration: Dict[str, Any],
        target_world_xy: Tuple[float, float],
    ) -> Optional[FloorplanCountResult]:
        """
        Count openings on the wall nearest `target_world_xy` inside a
        full-storey floorplan image.

        `calibration` is the sidecar JSON emitted by
        data_curation/scripts/synth/3c_render_full_storeys.py and contains
            world_bbox: {xmin, ymin, xmax, ymax}
            pixel_size: {width, height}

        Returns a FloorplanCountResult whose `debug` dict records the
        per-stage resolution / wall-detection / opening-detection status so
        failures are attributable without re-running.
        """
        debug: Dict[str, Any] = {
            "resolution_status": "pending",
            "wall_detection_status": "pending",
            "opening_detection_status": "pending",
            "fallback_reason": None,
        }

        image = self._load_image(storey_png_path)
        if image is None:
            debug["resolution_status"] = "png_not_found"
            return FloorplanCountResult(
                position=0, total=0, confidence=0.0,
                position_context="", mode="full_storey",
                debug=debug,
            )
        debug["resolution_status"] = "resolved"
        debug["image_size"] = [int(image.shape[1]), int(image.shape[0])]

        try:
            px, py = self._world_to_pixel(target_world_xy, calibration)
        except Exception as exc:
            debug["resolution_status"] = "calibration_invalid"
            debug["fallback_reason"] = f"world_to_pixel: {exc!r}"
            return FloorplanCountResult(
                position=0, total=0, confidence=0.0,
                position_context="", mode="full_storey",
                debug=debug,
            )
        debug["target_pixel"] = [int(px), int(py)]

        search_span = max(image.shape[0], image.shape[1]) * 0.25
        wall_line = self._detect_wall_line(image, (float(px), float(py)), search_span=search_span)
        if wall_line is None:
            debug["wall_detection_status"] = "no_candidate"
            debug["fallback_reason"] = "detect_wall_line_returned_None"
            return FloorplanCountResult(
                position=0, total=0, confidence=0.0,
                position_context="", mode="full_storey",
                debug=debug,
            )
        # T1.2 ordering fix: canonicalise so wall_origin = leftmost pixel (tiebreak topmost),
        # making projection ordering deterministic across runs and matching image-space
        # left-to-right convention (which aligns with the teacher's IFC-axis ordering in
        # matplotlib-rendered floorplans where +x world → +x pixel).
        p1, p2 = wall_line
        if (p2[0], p2[1]) < (p1[0], p1[1]):
            wall_line = (p2, p1)
        debug["wall_detection_status"] = "detected"
        debug["wall_line"] = {
            "x1": round(float(wall_line[0][0]), 1),
            "y1": round(float(wall_line[0][1]), 1),
            "x2": round(float(wall_line[1][0]), 1),
            "y2": round(float(wall_line[1][1]), 1),
        }

        openings = self._detect_openings_on_wall(image, wall_line)
        if not openings:
            debug["opening_detection_status"] = "none_on_wall"
            debug["fallback_reason"] = "no_openings_detected_along_wall"
            return FloorplanCountResult(
                position=0, total=0, confidence=0.0,
                position_context="", mode="full_storey",
                debug=debug,
            )
        debug["opening_detection_status"] = "detected"
        debug["opening_count_raw"] = len(openings)

        target_idx = self._nearest_opening_index(openings, (float(px), float(py)))
        position = target_idx + 1
        total = len(openings)

        # confidence heuristic (single-image mode — no template-match component)
        wall_score = 1.0
        density_score = min(1.0, total / 4.0)
        nearest_dist = math.hypot(
            openings[target_idx]["cx"] - px,
            openings[target_idx]["cy"] - py,
        )
        nearest_score = np.clip(1.0 - (nearest_dist / 60.0), 0.2, 1.0)
        confidence = float(np.clip(
            0.20 + 0.30 * wall_score + 0.25 * density_score + 0.25 * nearest_score,
            0.0, 0.95,
        ))

        debug["nearest_distance_px"] = round(float(nearest_dist), 1)

        return FloorplanCountResult(
            position=position,
            total=total,
            confidence=confidence,
            position_context=format_position_context(position, total),
            mode="full_storey",
            match_score=1.0,
            matched_bbox=None,
            patch_opening_count=0,
            debug=debug,
        )

    # ── Phase 6 T1.3 F4: oracle mode with IFC-supplied wall bounds ─────────────
    def count_from_full_storey_with_wall_bounds(
        self,
        storey_png_path: Path | str,
        calibration: Dict[str, Any],
        target_world_xy: Tuple[float, float],
        wall_endpoints_world: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> Optional[FloorplanCountResult]:
        """
        F4 oracle entry point: count openings on a single IFC wall whose
        world-space endpoints are supplied externally (from skeletons +
        IFC). This bypasses Hough wall detection entirely, so the resulting
        (position, total) is per-IFC-wall — the same frame as the teacher
        labels — avoiding the frame-mismatch failures in F3 mode.
        """
        debug: Dict[str, Any] = {
            "mode": "full_storey_with_bounds",
            "resolution_status": "pending",
            "opening_detection_status": "pending",
            "fallback_reason": None,
        }

        image = self._load_image(storey_png_path)
        if image is None:
            debug["resolution_status"] = "png_not_found"
            return FloorplanCountResult(
                position=0, total=0, confidence=0.0,
                position_context="", mode="full_storey_with_bounds",
                debug=debug,
            )
        debug["resolution_status"] = "resolved"

        try:
            tgt_px, tgt_py = self._world_to_pixel(target_world_xy, calibration)
            a_px, a_py = self._world_to_pixel(wall_endpoints_world[0], calibration)
            b_px, b_py = self._world_to_pixel(wall_endpoints_world[1], calibration)
        except Exception as exc:
            debug["resolution_status"] = "calibration_invalid"
            debug["fallback_reason"] = f"world_to_pixel: {exc!r}"
            return FloorplanCountResult(
                position=0, total=0, confidence=0.0,
                position_context="", mode="full_storey_with_bounds",
                debug=debug,
            )

        # F4: trust the validator's endpoint ordering (aligned to IFC local-X in
        # _wall_endpoints_world). Do NOT canonicalise to image-space lex-order,
        # because teacher's position_index ordering comes from IFC local-X which
        # can point either +world-x or -world-x depending on how the wall was
        # modeled.
        p1 = (float(a_px), float(a_py))
        p2 = (float(b_px), float(b_py))
        wall_line = (p1, p2)
        seg_length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        debug["wall_line"] = {
            "x1": round(p1[0], 1), "y1": round(p1[1], 1),
            "x2": round(p2[0], 1), "y2": round(p2[1], 1),
            "segment_length_px": round(seg_length, 1),
        }
        debug["target_pixel"] = [int(tgt_px), int(tgt_py)]

        raw_openings = self._detect_openings_on_wall(image, wall_line)
        # F4 key step: clip openings to the IFC wall segment only.
        # Anything with projection outside [0, seg_length] (plus small tolerance)
        # belongs to an adjacent colinear IFC wall, not this one.
        tol = 12.0  # px — account for end-cap symbol width
        openings = [
            op for op in raw_openings
            if -tol <= op["projection"] <= seg_length + tol
        ]
        debug["opening_count_pre_clip"] = len(raw_openings)
        debug["opening_count_post_clip"] = len(openings)

        if not openings:
            debug["opening_detection_status"] = "none_on_wall"
            debug["fallback_reason"] = "no_openings_within_wall_segment"
            return FloorplanCountResult(
                position=0, total=0, confidence=0.0,
                position_context="", mode="full_storey_with_bounds",
                debug=debug,
            )
        debug["opening_detection_status"] = "detected"

        target_idx = self._nearest_opening_index(openings, (float(tgt_px), float(tgt_py)))
        position = target_idx + 1
        total = len(openings)

        nearest_dist = math.hypot(
            openings[target_idx]["cx"] - tgt_px,
            openings[target_idx]["cy"] - tgt_py,
        )
        # F4 confidence is high-base because wall identity is oracle-provided;
        # remaining uncertainty is only in opening detection quality.
        density_score = min(1.0, total / 4.0)
        nearest_score = float(np.clip(1.0 - (nearest_dist / 60.0), 0.25, 1.0))
        confidence = float(np.clip(
            0.45 + 0.25 * density_score + 0.25 * nearest_score,
            0.0, 0.98,
        ))
        debug["nearest_distance_px"] = round(float(nearest_dist), 1)

        return FloorplanCountResult(
            position=position,
            total=total,
            confidence=confidence,
            position_context=format_position_context(position, total),
            mode="full_storey_with_bounds",
            match_score=1.0,
            matched_bbox=None,
            patch_opening_count=0,
            debug=debug,
        )

    @staticmethod
    def _world_to_pixel(
        world_xy: Tuple[float, float],
        calibration: Dict[str, Any],
    ) -> Tuple[int, int]:
        wb = calibration["world_bbox"]
        ps = calibration["pixel_size"]
        wx, wy = float(world_xy[0]), float(world_xy[1])
        W = int(ps["width"])
        H = int(ps["height"])
        span_x = float(wb["xmax"]) - float(wb["xmin"])
        span_y = float(wb["ymax"]) - float(wb["ymin"])
        if span_x <= 0 or span_y <= 0:
            raise ValueError(f"invalid world_bbox: {wb}")
        px = (wx - float(wb["xmin"])) / span_x * W
        py = H - (wy - float(wb["ymin"])) / span_y * H   # image y is inverted
        return int(round(px)), int(round(py))


def load_storey_calibration_index(index_path: Path | str) -> Dict[str, Dict[str, Any]]:
    """
    Load the aggregated calibration.json emitted by 3c_render_full_storeys.py
    and return a dict keyed by (lowercase) storey name → calibration entry.
    """
    with open(index_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for entry in raw.get("storeys", []):
        name = str(entry.get("storey_name", "")).strip().lower()
        if name:
            out[name] = entry
    return out
