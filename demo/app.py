"""
MSCD Demo UI — Streamlit entry point.

Run:
    cd mscd_demo
    streamlit run demo/app.py
"""
import sys
from pathlib import Path

# Ensure repo root is on path so `from src...` and `from demo...` work
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from demo import server, loader
from demo.ui import sidebar, tab_context, tab_pipeline, tab_result

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSCD Demo",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── start static file server (once per session) ───────────────────────────
@st.cache_resource
def get_static_base() -> str:
    """Start background HTTP server serving the repo root on port 8502."""
    return server.start(root=str(REPO_ROOT), port=8502)

static_base = get_static_base()

# ── sidebar: run / case selector ─────────────────────────────────────────
run_id, case_id = sidebar.render()

if run_id is None or case_id is None:
    st.info("Select a run and case in the sidebar to begin.")
    st.stop()

# ── load trace ────────────────────────────────────────────────────────────
trace = loader.load_trace(run_id, case_id)

if trace is None:
    st.error(f"Trace not found: {run_id} / {case_id}")
    st.stop()

sidebar.render_metrics(trace)

# ── header ────────────────────────────────────────────────────────────────
guid_match = trace.get("guid_match", False)
status_icon = "✅" if guid_match else "❌"
pipeline_type = trace.get("pipeline_type", "?").upper()

st.markdown(
    f"### {status_icon} `{case_id}`"
    f"<span style='color:#94a3b8;font-size:0.85em;margin-left:12px;'>"
    f"  {pipeline_type} pipeline  ·  {run_id}"
    f"</span>",
    unsafe_allow_html=True,
)

# ── main tabs ─────────────────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["📋 Query Context", "🔍 Pipeline Trace", "🏗️ Result"])

with t1:
    tab_context.render(trace)

with t2:
    tab_pipeline.render(trace)

with t3:
    tab_result.render(trace, static_base_url=static_base)
