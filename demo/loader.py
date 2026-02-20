"""
Trace loader — lists runs/cases and loads EvalTrace from disk.
"""
import json
from pathlib import Path
from typing import Optional

# Resolve paths relative to the repo root (mscd_demo/)
REPO_ROOT   = Path(__file__).parent.parent          # mscd_demo/
TRACES_DIR  = REPO_ROOT / "outputs" / "traces"
IFC_PATH    = REPO_ROOT / "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"


def list_runs() -> list[str]:
    if not TRACES_DIR.exists():
        return []
    return sorted(
        d.name for d in TRACES_DIR.iterdir()
        if d.is_dir() and list(d.glob("*.trace.json"))
    )


def list_cases(run_id: str) -> list[str]:
    run_dir = TRACES_DIR / run_id
    if not run_dir.exists():
        return []
    return sorted(p.stem.replace(".trace", "") for p in run_dir.glob("*.trace.json"))


def load_trace(run_id: str, case_id: str) -> Optional[dict]:
    path = TRACES_DIR / run_id / f"{case_id}.trace.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_ifc_path() -> Path:
    return IFC_PATH
