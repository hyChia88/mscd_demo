"""
Tab 3 — Result Visualisation
  Left  → IFC STEP text (predicted + GT)
  Right → 3D IFC viewer link (opens standalone page in new tab)
"""
import urllib.parse
import streamlit as st
from pathlib import Path
from demo.demo_dashboard import build_dashboard_url
from demo.loader import get_ifc_path, get_ifc_url


def render(trace: dict, static_base_url: str, approved_guid: str | None = None) -> None:
    predicted_guid = _get_effective_guid(trace, approved_guid=approved_guid)
    gt_guid        = _get_gt_guid(trace)
    guid_match     = (predicted_guid == gt_guid) if gt_guid else True
    case_id        = trace.get("scenario_id") or (trace.get("scenario") or {}).get("id", "")
    target_label   = "Approved element" if approved_guid else "Predicted element"

    left, right = st.columns([1, 1], gap="medium")

    # ── LEFT: IFC STEP text ───────────────────────────────────────────────
    with left:
        _render_ifc_text(predicted_guid, gt_guid, guid_match, case_id, target_label=target_label)

    # ── RIGHT: 3D viewer ─────────────────────────────────────────────────
    with right:
        _render_3d_viewer(
            predicted_guid,
            gt_guid,
            guid_match,
            static_base_url,
            case_id,
            target_label=target_label,
        )


def _render_ifc_text(
    predicted_guid: str,
    gt_guid: str,
    guid_match: bool,
    case_id: str = "",
    *,
    target_label: str = "Predicted element",
) -> None:
    st.markdown("**IFC STEP Text**")
    ifc_path = get_ifc_path(case_id)

    if predicted_guid:
        label = target_label + (" ✓" if guid_match else " ✗")
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
    *,
    target_label: str = "Predicted",
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
            f'{target_label} &nbsp;<code>{target_guid}</code></span>',
            unsafe_allow_html=True,
        )
    if gt_param:
        st.markdown(
            f'<span style="font-family:monospace;font-size:0.82em;">'
            f'<span style="color:#3b82f6">●</span> '
            f'Ground truth &nbsp;<code>{gt_param}</code></span>',
            unsafe_allow_html=True,
        )

    btn_col1, btn_col2 = st.columns(2)
    btn_col1.link_button("Open 3D Viewer", viewer_url, use_container_width=True)
    btn_col2.link_button(
        "Open Dashboard",
        build_dashboard_url(static_base_url, case_id=case_id, guid=target_guid, mode="grounding"),
        use_container_width=True,
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


def _get_effective_guid(trace: dict, approved_guid: str | None = None) -> str:
    """Use approved GUID from live demo when present, otherwise fall back to prediction."""
    explicit_guid = (approved_guid or "").strip()
    if explicit_guid:
        return explicit_guid
    return _get_predicted_guid(trace)


def _get_gt_guid(trace: dict) -> str:
    scenario = trace.get("scenario") or {}
    gt       = scenario.get("ground_truth") or {}
    if gt.get("target_guid"):
        return gt["target_guid"]
    ev = trace.get("evaluation") or {}
    return ev.get("target_guid", "")
