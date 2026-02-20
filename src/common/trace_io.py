"""
Centralized trace I/O service.

Single source of truth for persisting and loading EvalTrace objects.
Works for both V1 and V2 pipelines — V2 internals are embedded in
EvalTrace.v2_internals and written transparently.

Usage:
    from common.trace_io import write_trace, read_trace

    path = write_trace(trace, out_dir="outputs/traces")
    trace = read_trace("outputs/traces/run_id/SYNTH_001.trace.json")
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.eval.contracts import EvalTrace


def write_trace(trace: "EvalTrace", out_dir: str = "outputs/traces") -> str:
    """
    Write EvalTrace to {out_dir}/{run_id}/{scenario_id}.trace.json

    Returns the path written.
    """
    trace_dir = Path(out_dir) / trace.run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{trace.scenario_id}.trace.json"
    path.write_text(
        json.dumps(trace.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return str(path)


def read_trace(path: str) -> "EvalTrace":
    """
    Load EvalTrace from a .trace.json file.
    """
    from src.eval.contracts import EvalTrace

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalTrace.model_validate(data)
