"""Tab 1 — Query Context: modality badges, chat history, input images + floorplan."""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ── Modality badge colours ────────────────────────────────────────────────────
_ACTIVE = {
    "chat":    ("#22c55e", "white"),   # green
    "photos":  ("#3b82f6", "white"),   # blue
    "plan":    ("#f59e0b", "white"),   # amber
    "ctx_4d":  ("#8b5cf6", "white"),   # purple
}
_INACTIVE_BG  = "#e2e8f0"
_INACTIVE_FG  = "#94a3b8"


def _badge(label: str, active: bool, key: str) -> str:
    """Return an HTML badge string."""
    if active:
        bg, fg = _ACTIVE[key]
    else:
        bg, fg = _INACTIVE_BG, _INACTIVE_FG
    strike = "text-decoration:line-through;" if not active else ""
    return (
        f"<span style='background:{bg};color:{fg};padding:3px 11px;"
        f"border-radius:12px;font-size:0.82em;font-weight:600;"
        f"margin-right:6px;{strike}'>{label}</span>"
    )


def _modality_signals(trace: dict) -> dict:
    """
    Derive modality presence directly from trace data — no condition look-up.

    Returns a dict with keys:
      has_chat    bool
      has_photos  bool   + photo_paths list[str]
      has_plan    bool   + plan_path str | None
      has_4d      bool
    """
    scenario = trace.get("scenario") or {}
    ctx      = scenario.get("context_meta") or {}
    ipr      = (trace.get("internals") or {}).get("image_parse_result") or {}

    # 4D context: task_status present and not masked
    task_status = ctx.get("task_status", "") or ""
    has_4d = task_status not in ("", "N/A")

    # Chat
    chat = scenario.get("chat_history") or []
    has_chat = bool(chat) or bool(scenario.get("query_text", "").strip())

    # Site photos — prefer image_parse_result paths (authoritative), fall back to scenario
    site_photos = ipr.get("site_photos") or []
    photo_paths = [p["image_path"] for p in site_photos if p.get("image_path")]
    if not photo_paths:
        photo_paths = scenario.get("image_paths") or []
    has_photos = bool(photo_paths)

    # Floorplan — from image_parse_result only; no guessing from file system
    fp_entry  = ipr.get("floorplan") or {}
    plan_path = fp_entry.get("image_path") if fp_entry else None
    has_plan  = bool(plan_path)

    return dict(
        has_chat=has_chat,   chat=chat,
        has_photos=has_photos, photo_paths=photo_paths,
        has_plan=has_plan,   plan_path=plan_path,
        has_4d=has_4d,       task_status=task_status,
    )


def _annotate_image(image_path: Path, rels: list[dict], label: str) -> Image.Image:
    """Overlay spatial relation banner on top of image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Build annotation text from first relation
    rel = rels[0]
    pred = rel.get("predicate", "?")
    obj_type = rel.get("object_type", "?")
    obj_mat = rel.get("object_material")
    mat_tag = f" ({obj_mat})" if obj_mat else ""
    text = f"{pred} → {obj_type}{mat_tag}"

    pred_colors = {
        "FILLS": (59, 130, 246),       # blue
        "ADJACENT_TO": (245, 158, 11),  # amber
        "CONTINUOUS": (139, 92, 246),   # purple
    }
    color = pred_colors.get(pred, (107, 114, 128))

    # Draw banner at top of image
    w, h = img.size
    banner_h = max(24, h // 12)
    draw.rectangle([(0, 0), (w, banner_h)], fill=(*color, 200))

    # Draw text centered in banner
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                                  max(12, banner_h - 8))
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) / 2, (banner_h - th) / 2 - 1), text, fill=(255, 255, 255), font=font)

    # Draw thin border in predicate color
    draw.rectangle([(0, 0), (w - 1, h - 1)], outline=(*color, 255), width=3)

    return img


def _infer_condition_from_signals(m: dict) -> tuple[str, str]:
    """Derive condition code + label from actual modality signals in the trace.

    Reflects what was ACTUALLY fed to the model (not the case's base condition),
    so modality-ablation traces (MA, SITE, MC4D, …) display correctly.
    """
    has_photos = m.get("has_photos", False)
    has_plan   = m.get("has_plan",   False)
    has_4d     = m.get("has_4d",     False)
    if has_photos and has_plan and has_4d:
        return "MC4D", "Site+FP+4D"
    if has_photos and has_plan:
        return "FPSITE", "Site + Floorplan"
    if has_photos:
        return "SITE", "Site only"
    if has_plan:
        return "FP", "Floorplan only"
    return "MB", "Chat only"


def render(trace: dict) -> None:
    m = _modality_signals(trace)

    # ── Modality badge bar ────────────────────────────────────────────────────
    badges = (
        _badge("💬 Chat",        m["has_chat"],   "chat")
      + _badge("📸 Site Photos", m["has_photos"], "photos")
      + _badge("🗺️ Floorplan",   m["has_plan"],   "plan")
      + _badge("📊 4D Context",  m["has_4d"],     "ctx_4d")
    )
    # Derive condition from what was actually present in the trace (not bench metadata),
    # so modality-ablation runs (MA, SITE, MC4D, …) display the right label.
    cond_code, cond_label = _infer_condition_from_signals(m)
    cond_tag = (
        f'&nbsp;&nbsp;<span style="font-size:0.78em;padding:2px 7px;border-radius:4px;'
        f'background:#1e293b;color:#94a3b8;border:1px solid #334155;">'
        f'Condition: {cond_code} — {cond_label}</span>'
    )
    st.markdown(
        f"<div style='margin-bottom:8px;'><strong>Input Modalities</strong>&nbsp;&nbsp;{badges}{cond_tag}</div>",
        unsafe_allow_html=True,
    )

    # ── Chat history ──────────────────────────────────────────────────────────
    st.subheader("Chat History")
    if not m["chat"]:
        st.info("No chat history in this trace.")
    else:
        lines = [f"{msg.get('role','user')}: {msg.get('text','').strip()}"
                 for msg in m["chat"]]
        st.text("\n".join(lines))

    # ── Images & Floorplan ───────────────────────────────────────────────────
    if m["has_photos"] or m["has_plan"]:
        st.divider()
        st.subheader("Input Images & Floorplan")

        # Get spatial relations for annotation overlay
        rels = ((trace.get("internals") or {}).get("constraints") or {}).get("spatial_relations") or []

        items: list[tuple[Path, str]] = []
        for path in m["photo_paths"]:
            items.append((Path(path), "Site Photo"))
        if m["has_plan"]:
            items.append((Path(m["plan_path"]), "Floorplan"))

        cols = st.columns(len(items))
        for col, (p, label) in zip(cols, items):
            if p.exists():
                if rels:
                    img = _annotate_image(p, rels, label)
                    col.image(img, caption=f"{label}\n{p.name}", width=260)
                else:
                    col.image(str(p), caption=f"{label}\n{p.name}", width=260)
            else:
                col.warning(f"{label} not found:\n{p.name}")

    # ── 4D context metadata ───────────────────────────────────────────────────
    scenario = trace.get("scenario") or {}
    ctx = scenario.get("context_meta") or {}
    if ctx:
        st.divider()
        st.subheader("4D Context")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Role**  \n{ctx.get('sender_role','—')}")
        col2.markdown(f"**Phase**  \n{ctx.get('project_phase','—')}")
        col3.markdown(
            f"**Task**  \n"
            + (m["task_status"] if m["has_4d"] else f"~~{ctx.get('task_status','N/A')}~~ *(masked)*")
        )
