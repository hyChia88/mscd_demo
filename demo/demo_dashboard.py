"""Shared helpers for the minimal interpretability dashboard."""

from pathlib import Path
from urllib.parse import urlencode

from demo import loader

_DASHBOARD_HTML = Path(__file__).parent / "static" / "interpreter_dashboard.html"


def build_dashboard_url(
    static_base_url: str,
    case_id: str = "",
    guid: str = "",
    mode: str = "grounding",
) -> str:
    """Build the standalone dashboard URL with an mtime cache-buster."""
    try:
        version = str(int(_DASHBOARD_HTML.stat().st_mtime))
    except OSError:
        version = ""
    params = {
        "ifc": loader.get_ifc_url(case_id, static_base_url) if static_base_url else "",
        "guid": guid or "",
        "case_id": case_id or "",
        "mode": mode or "grounding",
        "_v": version,
    }
    return f"{static_base_url}/demo/static/interpreter_dashboard.html?{urlencode(params)}"
