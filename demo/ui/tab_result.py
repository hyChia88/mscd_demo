"""
Tab 3 — Result Visualisation
  Left  → IFC STEP text (predicted + GT)
  Right → 3D IFC viewer (iframe with @thatopen/components)
"""
import json
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from demo.loader import get_ifc_path


VIEWER_HTML = Path(__file__).parent.parent / "templates" / "viewer.html"


def render(trace: dict, static_base_url: str) -> None:
    predicted_guid = _get_predicted_guid(trace)
    gt_guid        = _get_gt_guid(trace)
    guid_match     = trace.get("guid_match", False)

    left, right = st.columns([1, 1], gap="medium")

    # ── LEFT: IFC STEP text ───────────────────────────────────────────────
    with left:
        _render_ifc_text(predicted_guid, gt_guid, guid_match)

    # ── RIGHT: 3D viewer ─────────────────────────────────────────────────
    with right:
        _render_3d_viewer(predicted_guid, gt_guid, guid_match, static_base_url)


def _render_ifc_text(predicted_guid: str, gt_guid: str, guid_match: bool) -> None:
    st.markdown("**IFC STEP Text**")
    ifc_path = get_ifc_path()

    if predicted_guid:
        label = "Predicted element" + (" ✓" if guid_match else " ✗")
        st.caption(label)
        step = _get_step(ifc_path, predicted_guid)
        st.code(step, language=None)

    if gt_guid and gt_guid != predicted_guid:
        st.caption("Ground truth element")
        step = _get_step(ifc_path, gt_guid)
        st.code(step, language=None)

    if not predicted_guid:
        st.info("No predicted GUID in this trace.")


def _render_3d_viewer(
    target_guid: str,
    gt_guid: str,
    guid_match: bool,
    static_base_url: str,
) -> None:
    st.markdown("**3D BIM Viewer**")

    ifc_url = static_base_url + "/data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"
    bundle_url = static_base_url + "/demo/static/viewer.bundle.js"

    config = {
        "ifc_url":     ifc_url,
        "target_guid": target_guid or "",
        "gt_guid":     gt_guid if (gt_guid and gt_guid != target_guid) else "",
        "guid_match":  guid_match,
        "static_base": static_base_url + "/demo/static",
    }

    # build legend HTML
    if guid_match:
        legend_html = (
            '<span class="dot" style="background:#22c55e;"></span> Predicted (correct)<br>'
        )
    else:
        legend_html = (
            '<span class="dot" style="background:#ef4444;"></span> Predicted (wrong)<br>'
            '<span class="dot" style="background:#3b82f6;"></span> Ground truth'
        )

    template = VIEWER_HTML.read_text(encoding="utf-8")
    html = (
        template
        .replace("__CONFIG_JSON__", json.dumps(config))
        .replace("__BUNDLE_URL__", bundle_url)
        .replace("__LEGEND__", legend_html)
    )

    components.html(html, height=520, scrolling=False)


@st.cache_data(show_spinner=False)
def _get_step(ifc_path: Path, guid: str) -> str:
    """Cached STEP text lookup — ifcopenshell is slow to open."""
    import ifcopenshell
    model   = ifcopenshell.open(str(ifc_path))
    element = model.by_guid(guid)
    if element is None:
        return f"# GUID {guid!r} not found"
    return str(element)


def _get_predicted_guid(trace: dict) -> str:
    # V2 pipeline — first candidate
    internals = trace.get("internals") or {}
    results   = internals.get("retrieval_results") or []
    for r in results:
        cands = r.get("candidates") or []
        if cands:
            return cands[0].get("guid", "")

    # V1 / old format
    interp = trace.get("interpreter_output") or {}
    guids  = interp.get("mentioned_guids") or []
    if guids:
        return guids[0]

    pred = trace.get("prediction") or {}
    return pred.get("element_guid", "")


def _get_gt_guid(trace: dict) -> str:
    scenario = trace.get("scenario") or {}
    gt       = scenario.get("ground_truth") or {}
    if gt.get("target_guid"):
        return gt["target_guid"]
    ev = trace.get("evaluation") or {}
    return ev.get("target_guid", "")
