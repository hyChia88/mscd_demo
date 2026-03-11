"""
Tab 3 — Result Visualisation
  Left  → IFC STEP text (predicted + GT)
  Right → 3D IFC viewer link (opens standalone page in new tab)
"""
import urllib.parse
import streamlit as st
from pathlib import Path
from demo.loader import get_ifc_path, get_ifc_url


def render(trace: dict, static_base_url: str) -> None:
    predicted_guid = _get_predicted_guid(trace)
    gt_guid        = _get_gt_guid(trace)
    guid_match     = trace.get("guid_match", False)
    case_id        = trace.get("scenario_id") or (trace.get("scenario") or {}).get("id", "")

    left, right = st.columns([1, 1], gap="medium")

    # ── LEFT: IFC STEP text ───────────────────────────────────────────────
    with left:
        _render_ifc_text(predicted_guid, gt_guid, guid_match, case_id)

    # ── RIGHT: 3D viewer ─────────────────────────────────────────────────
    with right:
        _render_3d_viewer(predicted_guid, gt_guid, guid_match, static_base_url, case_id)


def _render_ifc_text(predicted_guid: str, gt_guid: str, guid_match: bool, case_id: str = "") -> None:
    st.markdown("**IFC STEP Text**")
    ifc_path = get_ifc_path(case_id)

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
    case_id: str = "",
) -> None:
    st.markdown("**3D BIM Viewer**")

    ifc_url      = get_ifc_url(case_id, static_base_url)
    static_base  = static_base_url + "/demo/static"
    # Always pass GT GUID so viewer receives the true case target, even when GT == prediction.
    gt_param     = gt_guid or ""

    params = urllib.parse.urlencode({
        "ifc":    ifc_url,
        "target": target_guid or "",
        "gt":     gt_param,
        "match":  "1" if guid_match else "0",
        "base":   static_base,
    })
    viewer_url = f"{static_base}/test_viewer.html?{params}"

    # GUID summary chips
    if target_guid:
        color = "#22c55e" if guid_match else "#ef4444"
        icon  = "✓" if guid_match else "✗"
        st.markdown(
            f'<span style="font-family:monospace;font-size:0.82em;">'
            f'<span style="color:{color}">{icon}</span> '
            f'Predicted &nbsp;<code>{target_guid}</code></span>',
            unsafe_allow_html=True,
        )
    if gt_param:
        st.markdown(
            f'<span style="font-family:monospace;font-size:0.82em;">'
            f'<span style="color:#3b82f6">●</span> '
            f'Ground truth &nbsp;<code>{gt_param}</code></span>',
            unsafe_allow_html=True,
        )

    # Open-in-new-tab button
    st.markdown(
        f'<a href="{viewer_url}" target="_blank" style="text-decoration:none;">'
        f'<button style="'
        f'margin-top:12px;padding:8px 18px;background:#1e293b;color:#e2e8f0;'
        f'border:1px solid #334155;border-radius:6px;font-family:monospace;'
        f'font-size:13px;cursor:pointer;">'
        f'&#9881; Open 3D Viewer &#8599;</button></a>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _open_ifc(ifc_path_str: str):
    """Cache the open ifcopenshell model — shared across all GUID lookups."""
    import ifcopenshell
    return ifcopenshell.open(ifc_path_str)


@st.cache_data(show_spinner=False)
def _get_step(ifc_path: Path, guid: str) -> str:
    """Cached STEP text lookup — reuses the cached open model."""
    model = _open_ifc(str(ifc_path))
    try:
        element = model.by_guid(guid)
    except RuntimeError:
        return f"# GUID {guid!r} not found in this IFC"
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
