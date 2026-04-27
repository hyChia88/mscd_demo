"""Shared query review tabs used by the review page and live-demo analysis."""

import streamlit as st

from demo.ui import tab_context, tab_pipeline, tab_result


def _render_tabs(
    trace: dict,
    *,
    static_base_url: str = "",
    case_id: str = "",
    approved_guid: str | None = None,
) -> None:
    """Render the shared Context / Trace / Result review tabs."""
    t1, t2, t3 = st.tabs(["📋 Context", "🔍 Trace", "🏗️ Result"])
    with t1:
        tab_context.render(trace)
    with t2:
        tab_pipeline.render(trace, static_base_url=static_base_url, case_id=case_id)
    with t3:
        tab_result.render(trace, static_base_url=static_base_url, approved_guid=approved_guid)


def render(
    trace: dict,
    *,
    static_base_url: str = "",
    case_id: str = "",
    approved_guid: str | None = None,
    border: bool = False,
) -> None:
    """Render the shared review tabs, optionally inside a bordered panel."""
    if border:
        with st.container(border=True):
            _render_tabs(
                trace,
                static_base_url=static_base_url,
                case_id=case_id,
                approved_guid=approved_guid,
            )
        return

    _render_tabs(
        trace,
        static_base_url=static_base_url,
        case_id=case_id,
        approved_guid=approved_guid,
    )
