"""
ui_components.py
---------------------------------------------------------
Pure rendering helpers. No RAG / business logic lives here —
only Streamlit markup. Keeping these separate from app.py
keeps the main file readable.
---------------------------------------------------------
"""

import streamlit as st


def render_logo():
    st.markdown(
        """
        <div class="rg-logo-wrap">
            <div class="rg-logo-icon">🧠</div>
            <div>
                <div class="rg-logo-text">AskDelve</div>
                <div class="rg-logo-sub">Document Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_doc_card(filename: str, num_pages: int, num_chunks: int) -> None:
    st.markdown(
        f"""
        <div class="rg-doc-card">
            <div class="rg-doc-name" title="{filename}">📄 {filename}</div>
            <div class="rg-doc-meta">{num_pages} pages · {num_chunks} chunks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_message(role: str, content: str):

    """
    Renders one full chat row — avatar + bubble — aligned ChatGPT/Claude-style:
    the user's messages float right, the assistant's float left.

    We deliberately build the whole row (avatar included) ourselves instead
    of relying on st.chat_message()'s built-in avatar layout, since Streamlit
    always lays that out left-aligned regardless of role and there's no
    public, version-stable hook to flip it from CSS alone.

    IMPORTANT: st.markdown() calls do NOT nest — each call is an
    independent block in the DOM. Opening a <div> in one call and
    closing it in another leaves the text OUTSIDE the styled div,
    which is why text was unstyled / unreadable. We render markdown
    to HTML ourselves and emit exactly one st.markdown call.
    """
    import markdown as _md

    is_user = role == "user"
    row_class = "rg-row-user" if is_user else "rg-row-ai"
    bubble_class = "rg-bubble-user" if is_user else "rg-bubble-ai"
    avatar = "🧑‍💻" if is_user else "🧠"

    html_body = _md.markdown(
        content, extensions=["fenced_code", "tables", "sane_lists"]
    )
    st.markdown(
        f"""
        <div class="rg-row {row_class}">
            <div class="rg-avatar">{avatar}</div>
            <div class="rg-bubble {bubble_class}">{html_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_citation(source_name: str, page: int, score: float, preview: str, key: str):
    score_pct = f"{score * 100:.1f}%" if score is not None else "N/A"
    st.markdown(
        f"""
        <div class="rg-citation-card">
            <div class="rg-citation-header">
                <span class="rg-citation-name">📄 {source_name}</span>
                <span class="rg-citation-score">{score_pct} match</span>
            </div>
            <div class="rg-citation-page">Page {page}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Preview source text", expanded=False):
        st.write(preview)


def render_summary_block(title: str, content: str):
    st.markdown(
        f"""
        <div class="rg-summary-section">
            <div class="rg-summary-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(content)