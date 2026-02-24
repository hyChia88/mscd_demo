"""
Trace loader — lists runs/cases and loads EvalTrace from disk.
"""
import json
import re
from pathlib import Path
from typing import Optional

# Resolve paths relative to the repo root (mscd_demo/)
REPO_ROOT = Path(__file__).parent.parent          # mscd_demo/
IFC_PATH  = REPO_ROOT / "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"

# Building code → IFC path relative to REPO_ROOT (all served by the static server)
_BUILDING_IFC: dict[str, str] = {
    "AP":  "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc",
    "BH":  "data/ifc/BasicHouse.ifc",
    "DXA": "data/ifc/Duplex_A_20110505.ifc",   # symlinked from data_curation/
}


def _building_code(case_id: str) -> str:
    """Extract building code from case_id, e.g. 'SYNTH_V3_005_BH_SK_005' → 'BH'."""
    m = re.search(r'_([A-Z]+)_SK_', case_id)
    return m.group(1) if m else ""


# All directories that may contain per-run trace subdirs
TRACE_ROOTS = [
    REPO_ROOT / "outputs" / "traces",
    REPO_ROOT / "logs" / "evaluations" / "synth_v03" / "traces",
    REPO_ROOT / "logs" / "evaluations" / "synth_v04" / "traces",
]


def _run_to_root(run_id: str) -> Optional[Path]:
    """Return the trace root that contains run_id, or None."""
    for root in TRACE_ROOTS:
        candidate = root / run_id
        if candidate.is_dir():
            return root
    return None


def list_runs() -> list[str]:
    seen: set[str] = set()
    for root in TRACE_ROOTS:
        if not root.exists():
            continue
        for d in root.iterdir():
            if d.is_dir() and d.name not in seen and list(d.glob("*.trace.json")):
                seen.add(d.name)
    return sorted(seen)


def list_cases(run_id: str) -> list[str]:
    root = _run_to_root(run_id)
    if root is None:
        return []
    return sorted(
        p.stem.replace(".trace", "") for p in (root / run_id).glob("*.trace.json")
    )


def load_trace(run_id: str, case_id: str) -> Optional[dict]:
    root = _run_to_root(run_id)
    if root is None:
        return None
    path = root / run_id / f"{case_id}.trace.json"
    if not path.exists():
        return None
    trace = json.loads(path.read_text(encoding="utf-8"))

    # Normalize legacy V1 format (Feb-7 runs) to current schema.
    # Old traces lack pipeline_type / scenario / top-level eval booleans.
    if "pipeline_type" not in trace:
        ev     = trace.get("evaluation") or {}
        gt     = trace.get("ground_truth") or {}
        inputs = trace.get("inputs") or {}

        trace["pipeline_type"]    = "v1"
        trace["guid_match"]       = ev.get("guid_match", False)
        trace["name_match"]       = ev.get("name_match", False)
        trace["storey_match"]     = False
        trace["total_latency_ms"] = (ev.get("elapsed_time") or 0) * 1000
        trace["scenario"] = {
            "id":           trace.get("case_id", ""),
            "chat_history": [],
            "image_paths":  inputs.get("images") or [],
            "context_meta": {},
            "query_text":   inputs.get("user_input", ""),
            "ground_truth": {
                "target_guid":      gt.get("target_guid", ""),
                "target_name":      gt.get("target_name", ""),
                "target_storey":    gt.get("target_storey", ""),
                "target_ifc_class": gt.get("target_ifc_class", ""),
            },
        }

    return trace


def get_ifc_path(case_id: str = "") -> Path:
    """Return the IFC file path for the given case_id, falling back to AdvancedProject."""
    code = _building_code(case_id)
    rel = _BUILDING_IFC.get(code)
    if rel:
        p = REPO_ROOT / rel
        if p.exists():
            return p
    return IFC_PATH


def get_ifc_url(case_id: str, static_base_url: str) -> str:
    """Build the HTTP URL for the IFC file served by the static server."""
    code = _building_code(case_id)
    rel = _BUILDING_IFC.get(code, _BUILDING_IFC["AP"])
    return static_base_url + "/" + rel
