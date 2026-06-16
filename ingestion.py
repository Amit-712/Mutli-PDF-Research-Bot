"""
ingestion.py
---------------------------------------------------------
Document ingestion, vectorstore management, retriever
construction, and summarization helpers.

This is the ONLY layer that should change if your chunking /
embedding / vectorstore choice changes. The UI (app.py) never
talks to PDFs or embeddings directly — it only calls functions
in this file and in rag_chain.py.

Swap the internals of `_build_vectorstore`, `_load_pdf`, or
`summarize_documents` freely; their signatures are the
integration contract the UI relies on.
---------------------------------------------------------
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.rag_chain import load_llm, create_rag_chain


# ---------------------------------------------------------
# Data model for one ingested PDF
# ---------------------------------------------------------
@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    raw_chunks: list = field(default_factory=list)  # kept for summarization


_EMBEDDINGS = None


def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDINGS


def _load_pdf(file_bytes: bytes, filename: str):
    """Writes the uploaded bytes to a temp file and loads pages via PyPDFLoader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    for p in pages:
        p.metadata["source"] = filename

    os.unlink(tmp_path)
    return pages


def process_pdf(file_bytes: bytes, filename: str, doc_id: str) -> tuple[DocumentRecord, list]:
    """Loads + splits a single PDF. Returns (DocumentRecord, chunked_docs)."""
    pages = _load_pdf(file_bytes, filename)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    for c in chunks:
        c.metadata["source"] = filename
        c.metadata["doc_id"] = doc_id

    record = DocumentRecord(
        doc_id=doc_id,
        filename=filename,
        num_pages=len(pages),
        num_chunks=len(chunks),
        raw_chunks=chunks,
    )
    return record, chunks


def build_vectorstore(all_chunks: list):
    """Builds (or rebuilds) a FAISS vectorstore from a flat list of chunk Documents."""
    if not all_chunks:
        return None
    embeddings = _get_embeddings()
    return FAISS.from_documents(all_chunks, embeddings)


def get_retriever(vectorstore, k: int = 4):
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_rag_chain(retriever):
    """Wraps your unmodified rag_chain.create_rag_chain + load_llm."""
    llm = load_llm()
    return create_rag_chain(retriever, llm)


def retrieve_with_scores(vectorstore, query: str, k: int = 4):
    """Returns list of (Document, similarity_score) for the citation panel."""
    if vectorstore is None:
        return []
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return results


# ---------------------------------------------------------
# Summarization (separate light-weight chain, reuses load_llm)
# ---------------------------------------------------------
def summarize_documents(all_chunks: list) -> dict:
    """
    Produces a 4-part summary dict:
    short_summary, detailed_summary, key_insights, research_notes
    Uses your existing load_llm() — no change to your RAG logic.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = load_llm()
    text = "\n\n".join(c.page_content for c in all_chunks[:60])  # cap context

    summary_prompt = ChatPromptTemplate.from_template(
        """You are a research assistant summarizing uploaded documents.

Context:
{context}

Produce your response in exactly this structure with these four headers:

### Short Summary
(2-3 sentences)

### Detailed Summary
(detailed multi-paragraph summary)

### Key Insights
(bullet list of the most important insights)

### Research Notes
(bullet list of open questions, caveats, or follow-up directions)
"""
    )

    chain = summary_prompt | llm | StrOutputParser()
    raw = chain.invoke({"context": text})

    sections = {"short_summary": "", "detailed_summary": "", "key_insights": "", "research_notes": ""}
    current = None
    header_map = {
        "short summary": "short_summary",
        "detailed summary": "detailed_summary",
        "key insights": "key_insights",
        "research notes": "research_notes",
    }
    for line in raw.splitlines():
        stripped = line.strip().lstrip("#").strip().lower()
        if stripped in header_map:
            current = header_map[stripped]
            continue
        if current:
            sections[current] += line + "\n"

    return {k: v.strip() for k, v in sections.items()}