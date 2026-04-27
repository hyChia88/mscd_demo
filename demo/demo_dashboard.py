"""Shared helpers for the minimal interpretability dashboard."""

from urllib.parse import urlencode

from demo import loader


def build_dashboard_url(
    static_base_url: str,
    case_id: str = "",
    guid: str = "",
    mode: str = "grounding",
) -> str:
    """Build the standalone dashboard URL."""
    params = {
        "ifc": loader.get_ifc_url(case_id, static_base_url) if static_base_url else "",
        "guid": guid or "",
        "case_id": case_id or "",
        "mode": mode or "grounding",
    }
    return f"{static_base_url}/demo/static/interpreter_dashboard.html?{urlencode(params)}"
