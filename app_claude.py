"""
app.py — AskDelve
---------------------------------------------------------
Production-quality Streamlit UI wrapped around an existing,
unmodified RAG chain (rag_chain.py: load_llm, create_rag_chain).

Run:
    streamlit run app.py
---------------------------------------------------------
"""

import time
import uuid
from pathlib import Path

import streamlit as st

from ingestion import (
    process_pdf,
    build_vectorstore,
    get_retriever,
    get_rag_chain,
    retrieve_with_scores,
    summarize_documents,
)
from ui_components import (
    render_logo,
    render_doc_card,
    render_chat_message,
    render_citation,
    render_summary_block,
)

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="AskDelve",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = Path(__file__).parent / "assets" / "style.css"


def load_css():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =========================================================
# Session state initialization
# =========================================================
def init_session_state():
    defaults = {
        "dark_mode": True,
        "documents": {},          # doc_id -> DocumentRecord
        "all_chunks": [],         # flat list across all docs, used to rebuild vectorstore
        "vectorstore": None,
        "rag_chain": None,
        "messages": [],           # [{role, content, citations}]
        "show_summary": False,
        "summary_data": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()
load_css()

# Theme attribute on root html element (read by CSS [data-theme="dark"])
theme_flag = "dark" if st.session_state.dark_mode else "light"

# Reliable theme switch: Streamlit's root element is [data-testid="stAppViewContainer"]'s
# parent <body>, but injecting via <script> on documentElement is unreliable inside
# Streamlit's render cycle (it gets wiped on rerun before paint). Instead we inject a
# <style> block that is regenerated every run with the CORRECT variable values baked in
# directly — no runtime DOM mutation required, so it can never fail to apply.
THEME_VARS = {
    "dark": {
        "bg": "#0e0f13", "bg2": "#161821", "sidebar": "#11121a",
        "text": "#f1f2f6", "muted": "#a7adba", "border": "#2a2d3a",
        "accent": "#8b7bff", "accent_soft": "#2a2452",
        "user_bubble": "#5b4cf0", "user_text": "#ffffff",
        "ai_bubble": "#1e2030", "ai_text": "#f1f2f6",
    },
    "light": {
        "bg": "#ffffff", "bg2": "#f7f8fa", "sidebar": "#f3f4f7",
        "text": "#1a1d23", "muted": "#5b6270", "border": "#e2e4ea",
        "accent": "#5b4cf0", "accent_soft": "#ece9ff",
        "user_bubble": "#5b4cf0", "user_text": "#ffffff",
        "ai_bubble": "#eef0f4", "ai_text": "#1a1d23",
    },
}
v = THEME_VARS[theme_flag]
st.markdown(
    f"""
    <style>
    :root {{
        --rg-bg: {v['bg']};
        --rg-bg-secondary: {v['bg2']};
        --rg-sidebar-bg: {v['sidebar']};
        --rg-text: {v['text']};
        --rg-text-muted: {v['muted']};
        --rg-border: {v['border']};
        --rg-accent: {v['accent']};
        --rg-accent-soft: {v['accent_soft']};
        --rg-user-bubble: {v['user_bubble']};
        --rg-user-text: {v['user_text']};
        --rg-ai-bubble: {v['ai_bubble']};
        --rg-ai-text: {v['ai_text']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Core actions
# =========================================================
def rebuild_index():
    """Rebuilds vectorstore + rag_chain from current all_chunks."""
    if not st.session_state.all_chunks:
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        return
    vs = build_vectorstore(st.session_state.all_chunks)
    st.session_state.vectorstore = vs
    retriever = get_retriever(vs)
    st.session_state.rag_chain = get_rag_chain(retriever)  # uses your unmodified load_llm/create_rag_chain


def handle_uploads(uploaded_files):
    progress = st.sidebar.progress(0, text="Starting upload...")
    total = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        existing_names = {d.filename for d in st.session_state.documents.values()}
        if file.name in existing_names:
            progress.progress((i + 1) / total, text=f"Skipped duplicate: {file.name}")
            continue

        progress.progress((i + 1) / total, text=f"Processing {file.name}...")
        doc_id = str(uuid.uuid4())[:8]
        record, chunks = process_pdf(file.read(), file.name, doc_id)

        st.session_state.documents[doc_id] = record
        st.session_state.all_chunks.extend(chunks)

    progress.progress(1.0, text="Indexing documents...")
    rebuild_index()
    progress.empty()
    st.toast("Documents indexed successfully ✅")


def delete_document(doc_id: str):
    st.session_state.documents.pop(doc_id, None)
    st.session_state.all_chunks = [
        c for c in st.session_state.all_chunks if c.metadata.get("doc_id") != doc_id
    ]
    rebuild_index()
    st.toast("Document removed.")


def new_chat():
    st.session_state.messages = []
    st.session_state.show_summary = False
    st.session_state.summary_data = None


def run_summarization():
    if not st.session_state.all_chunks:
        st.warning("Upload at least one document before summarizing.")
        return
    with st.spinner("Generating summary..."):
        st.session_state.summary_data = summarize_documents(st.session_state.all_chunks)
    st.session_state.show_summary = True


def stream_text(text: str, placeholder):
    """Word-by-word streaming effect for the AI response, rendered safely."""
    import html as _html

    words = text.split(" ")
    rendered = ""
    for word in words:
        rendered += word + " "
        safe = _html.escape(rendered)
        placeholder.markdown(
            f"""
            <div class="rg-row rg-row-ai">
                <div class="rg-avatar">🧠</div>
                <div class="rg-bubble rg-bubble-ai">{safe}▌</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.012)
    # Final pass: render real markdown (code blocks, bold, etc.) via the shared helper
    placeholder.empty()
    with placeholder.container():
        render_chat_message("assistant", text)


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    render_logo()

    st.markdown('<div class="rg-section-label">Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in {d.filename for d in st.session_state.documents.values()}
        ]
        if new_files:
            handle_uploads(new_files)

    if st.session_state.documents:
        for doc_id, record in list(st.session_state.documents.items()):
            col1, col2 = st.columns([5, 1])
            with col1:
                render_doc_card(record.filename, record.num_pages, record.num_chunks)
            with col2:
                if st.button("🗑️", key=f"del_{doc_id}", help=f"Delete {record.filename}"):
                    delete_document(doc_id)
                    st.rerun()
    else:
        st.caption("No documents uploaded yet.")

    st.markdown('<div class="rg-section-label">Actions</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ New Chat", use_container_width=True):
            new_chat()
            st.rerun()
    with col_b:
        if st.button("📋 Summarize", use_container_width=True):
            run_summarization()

    st.markdown('<div class="rg-section-label">Preferences</div>', unsafe_allow_html=True)
    st.toggle("🌙 Dark mode", key="dark_mode")

    st.markdown("---")
    st.caption("AskDelve · powered by your RAG chain")


# =========================================================
# Main area
# =========================================================
st.markdown("## 🧠 AskDelve")
st.caption("Ask questions across your uploaded research documents.")

# ---- Summary panel ----
if st.session_state.show_summary and st.session_state.summary_data:
    with st.container(border=True):
        top_col1, top_col2 = st.columns([8, 1])
        with top_col1:
            st.markdown("### 📋 Document Summary")
        with top_col2:
            if st.button("✖", key="close_summary"):
                st.session_state.show_summary = False
                st.rerun()

        tabs = st.tabs(["Short Summary", "Detailed Summary", "Key Insights", "Research Notes"])
        data = st.session_state.summary_data
        with tabs[0]:
            render_summary_block("Short Summary", data.get("short_summary", "—"))
        with tabs[1]:
            render_summary_block("Detailed Summary", data.get("detailed_summary", "—"))
        with tabs[2]:
            render_summary_block("Key Insights", data.get("key_insights", "—"))
        with tabs[3]:
            render_summary_block("Research Notes", data.get("research_notes", "—"))

# ---- Chat history ----
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        render_chat_message(msg["role"], msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"📚 Sources ({len(msg['citations'])})", expanded=False):
                for i, c in enumerate(msg["citations"]):
                    render_citation(
                        c["source"], c["page"], c["score"], c["preview"],
                        key=f"cite_{msg.get('id', i)}_{i}",
                    )

# ---- Chat input ----
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    if st.session_state.rag_chain is None:
        st.warning("Please upload at least one PDF before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with chat_container:
            render_chat_message("user", user_query)

            placeholder = st.empty()
            with st.spinner("Retrieving relevant context..."):
                # ===========================================
                # INTEGRATION POINT: your unmodified RAG chain
                # ===========================================
                answer = st.session_state.rag_chain.invoke(user_query)

                scored_docs = retrieve_with_scores(
                    st.session_state.vectorstore, user_query, k=4
                )

            stream_text(answer, placeholder)

            citations = []
            for doc, score in scored_docs:
                citations.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 0) + 1
                    if isinstance(doc.metadata.get("page"), int) else "-",
                    "score": score,
                    "preview": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                })

            if citations:
                with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
                    for i, c in enumerate(citations):
                        render_citation(c["source"], c["page"], c["score"], c["preview"], key=f"live_{i}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "id": str(uuid.uuid4())[:6],
        })