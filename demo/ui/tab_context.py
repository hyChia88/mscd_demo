"""Tab 1 — Query Context: chat history + input images."""
import streamlit as st
from pathlib import Path


def render(trace: dict) -> None:
    scenario = trace.get("scenario") or {}
    chat     = scenario.get("chat_history") or []
    images   = scenario.get("image_paths") or []

    st.subheader("Chat History")
    if not chat:
        st.info("No chat history in this trace.")
    else:
        lines = []
        for msg in chat:
            role = msg.get("role", "user")
            text = msg.get("text", "").strip()
            lines.append(f"{role}: {text}")
        st.text("\n".join(lines))

    if images:
        st.divider()
        st.subheader("Input Images")
        cols = st.columns(min(len(images), 3))
        for i, path in enumerate(images):
            p = Path(path)
            if p.exists():
                cols[i % 3].image(str(p), caption=p.name, width=260)
            else:
                cols[i % 3].warning(f"Image not found:\n{p.name}")

    # 4D context metadata
    ctx = scenario.get("context_meta") or {}
    if ctx:
        st.divider()
        st.subheader("4D Context")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Role**  \n{ctx.get('sender_role','—')}")
        col2.markdown(f"**Phase**  \n{ctx.get('project_phase','—')}")
        col3.markdown(f"**Task**  \n{ctx.get('task_status','—')}")
