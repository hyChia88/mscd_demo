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


def _discover_trace_roots() -> list[Path]:
    """Dynamically discover all directories that directly contain run subdirs with *.trace.json.

    Scans output/, outputs/, evaluation/, and logs/ up to 6 levels deep for any
    directory named 'traces' whose immediate children contain *.trace.json files.
    This covers both shallow paths (output/synth_v05_lora5/traces/) and deeply
    nested lora6 paths (output/lora6_v2_ap_20260331/ap_e2e_phase5_g8/g7/traces/).
    """
    search_roots = [
        REPO_ROOT / "output",
        REPO_ROOT / "outputs",
        REPO_ROOT / "evaluation",
        REPO_ROOT / "logs",
    ]
    found: list[Path] = []
    for base in search_roots:
        if not base.exists():
            continue
        # glob up to 6 levels: base/**/traces  (** expands to 0-N dirs)
        for traces_dir in base.glob("**/traces"):
            if not traces_dir.is_dir():
                continue
            # A valid trace root has at least one child subdir with *.trace.json
            for child in traces_dir.iterdir():
                if child.is_dir() and any(child.glob("*.trace.json")):
                    found.append(traces_dir)
                    break
    return found


# Cached after first call (module-level singleton)
_TRACE_ROOTS_CACHE: list[Path] | None = None


def _get_trace_roots() -> list[Path]:
    global _TRACE_ROOTS_CACHE
    if _TRACE_ROOTS_CACHE is None:
        _TRACE_ROOTS_CACHE = _discover_trace_roots()
    return _TRACE_ROOTS_CACHE


# Maps label (as shown in sidebar) → (trace_root, bare_run_id)
_RUN_LABEL_TO_PATH: dict[str, tuple[Path, str]] = {}


def list_runs() -> list[str]:
    """Return sorted list of run labels in the form 'model_context / run_id'.

    For flat roots (e.g. synth_v05_lora5/traces/), the parent folder name is used
    as the model context.  For nested lora6 roots like
    lora6_v2_ap_20260331/ap_e2e_phase5_g8/g7_position_context/traces/, the last two
    path segments above 'traces' are combined: 'ap_e2e_phase5_g8/g7_position_context'.

    The label is also stored in `_RUN_LABEL_TO_PATH` so that `_run_to_root` can
    resolve it back to (root, run_id).
    """
    _RUN_LABEL_TO_PATH.clear()
    labels: list[str] = []
    for root in _get_trace_roots():
        # Build a human-readable context from the path above 'traces/'
        parts = root.parts
        try:
            traces_idx = [p.lower() for p in parts].index("traces")
            ctx_parts = parts[max(traces_idx - 2, 0): traces_idx]
            ctx = "/".join(ctx_parts) if ctx_parts else root.name
        except ValueError:
            ctx = root.name

        for d in root.iterdir():
            if not d.is_dir() or not any(d.glob("*.trace.json")):
                continue
            label = f"{ctx}  /  {d.name}"
            if label not in _RUN_LABEL_TO_PATH:
                _RUN_LABEL_TO_PATH[label] = (root, d.name)
                labels.append(label)

    return sorted(labels)


def _run_to_root(run_id: str) -> Optional[Path]:
    """Return the trace root for a run label or bare run_id, or None."""
    # Fast path: label form "ctx / run_id" already in cache
    if run_id in _RUN_LABEL_TO_PATH:
        return _RUN_LABEL_TO_PATH[run_id][0]
    # Fallback: bare run_id — search all roots
    for root in _get_trace_roots():
        candidate = root / run_id
        if candidate.is_dir():
            return root
    return None


def _bare_run_id(run_label: str) -> str:
    """Extract the bare run_id from a label ('ctx / run_id' → 'run_id')."""
    if run_label in _RUN_LABEL_TO_PATH:
        return _RUN_LABEL_TO_PATH[run_label][1]
    # Legacy: bare id passed directly
    return run_label


def list_cases(run_id: str) -> list[str]:
    bare = _bare_run_id(run_id)
    root = _run_to_root(run_id)
    if root is None:
        return []
    return sorted(
        p.stem.replace(".trace", "") for p in (root / bare).glob("*.trace.json")
    )


def load_trace(run_id: str, case_id: str) -> Optional[dict]:
    bare = _bare_run_id(run_id)
    root = _run_to_root(run_id)
    if root is None:
        return None
    path = root / bare / f"{case_id}.trace.json"
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

    # Enrich AP_SK_* traces that have empty image_paths (lora6 e2e eval didn't store them).
    _enrich_missing_images(trace)

    return trace


# ── Image enrichment for lora6 AP eval traces ────────────────────────────────

# Lazy-loaded cache: case_id → {"images": [...], "floorplan": str|None, "condition": str}
_AP_CASES_CACHE: dict[str, dict] | None = None

# Modality profile from profiles.yaml (C1 = FP only, C3 = full, etc.)
_CONDITION_MODALITIES: dict[str, dict] = {
    "C1":  {"use_images": False, "use_floorplan": True,  "has_4d": False, "label": "Floorplan only"},
    "C2":  {"use_images": True,  "use_floorplan": True,  "has_4d": False, "label": "Site + Floorplan"},
    "C3":  {"use_images": True,  "use_floorplan": True,  "has_4d": True,  "label": "Full (Site+FP+4D)"},
    "MA":  {"use_images": True,  "use_floorplan": False, "has_4d": False, "label": "Site only"},
    "MB":  {"use_images": False, "use_floorplan": False, "has_4d": False, "label": "Chat only"},
    "MC":  {"use_images": True,  "use_floorplan": True,  "has_4d": False, "label": "Site + Floorplan"},
    "MC4D":{"use_images": True,  "use_floorplan": True,  "has_4d": True,  "label": "Site+FP+4D"},
    "FP":  {"use_images": False, "use_floorplan": True,  "has_4d": False, "label": "Floorplan only"},
    "SITE":{"use_images": True,  "use_floorplan": False, "has_4d": False, "label": "Site only"},
    "FPSITE": {"use_images": True, "use_floorplan": True, "has_4d": False, "label": "Site + Floorplan"},
}

DATA_ROOT = REPO_ROOT.parent / "data_curation"


def _load_ap_cases_index() -> dict[str, dict]:
    global _AP_CASES_CACHE
    if _AP_CASES_CACHE is not None:
        return _AP_CASES_CACHE
    _AP_CASES_CACHE = {}
    cases_file = REPO_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
    if not cases_file.exists():
        return _AP_CASES_CACHE
    with cases_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get("case_id", "")
            if not cid:
                continue
            inp = row.get("inputs") or {}
            _AP_CASES_CACHE[cid] = {
                "images": inp.get("images") or [],
                "floorplan": inp.get("floorplan_patch"),
                "condition": (row.get("bench") or {}).get("condition", ""),
            }
    return _AP_CASES_CACHE


def _resolve_data_path(rel_path: str) -> str:
    """Convert a data_curation-relative path to an absolute path string."""
    if not rel_path:
        return ""
    # paths like "datasets/synth_v0.5_ap/imgs/AP_SK_325_site.png"
    abs_p = DATA_ROOT / rel_path
    if abs_p.exists():
        return str(abs_p)
    # try under REPO_ROOT as well (older traces use mscd_demo-relative paths)
    abs_p2 = REPO_ROOT / rel_path
    if abs_p2.exists():
        return str(abs_p2)
    return str(abs_p)  # return best guess even if not found


def _enrich_missing_images(trace: dict) -> None:
    """Inject image/floorplan paths into lora6 AP eval traces that stored empty lists.

    Mutates trace in place — adds paths to scenario.image_paths and
    internals.image_parse_result so that tab_context.py can render them.
    Also tags trace with bench_condition_label for display.
    """
    scenario = trace.get("scenario") or {}
    if scenario.get("image_paths") or scenario.get("image_files"):
        return  # already has image data

    case_id = scenario.get("id", "")
    if not case_id:
        return

    ap_cases = _load_ap_cases_index()
    entry = ap_cases.get(case_id)
    if not entry:
        return

    condition = entry.get("condition") or (trace.get("bench") or {}).get("condition", "")
    cond_profile = _CONDITION_MODALITIES.get(condition, {})

    # Inject site photo paths
    if cond_profile.get("use_images", True):
        resolved = [_resolve_data_path(p) for p in entry["images"] if p]
        if resolved:
            scenario["image_paths"] = resolved

    # Inject floorplan into internals.image_parse_result
    if cond_profile.get("use_floorplan", True) and entry.get("floorplan"):
        fp_path = _resolve_data_path(entry["floorplan"])
        if fp_path:
            internals = trace.setdefault("internals", {})
            if not isinstance(internals, dict):
                internals = {}
                trace["internals"] = internals
            ipr = internals.get("image_parse_result") or {}
            internals["image_parse_result"] = ipr
            if not ipr.get("floorplan"):
                ipr["floorplan"] = {"image_path": fp_path}

    # Tag the condition label for display in UI
    if condition:
        scenario["_bench_condition"] = condition
        scenario["_bench_condition_label"] = cond_profile.get("label", condition)


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
