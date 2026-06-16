import os

os.makedirs(
    "uploads",
    exist_ok=True
)


import streamlit as st

from src.pdf_loader import load_pdf
from src.chunker import chunk_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain,load_llm
from src.summarizer import summarize_documents

st.title(
    "ResearchGPT"
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

process = st.sidebar.button(
    "Process Documents"
)   

if process:

    if not uploaded_files:
        st.warning("Please upload PDFs first.")
        st.stop()


    documents = []

    for file in uploaded_files:

        temp_path = f"uploads/{file.name}"

        with open(
            temp_path,
            "wb"
        ) as f:

            f.write(
                file.getbuffer()
            )

        documents.extend(
            load_pdf(temp_path)
        )

    chunks = chunk_documents(documents)

    embeddings = get_embedding_model()

    vector_store = create_vector_store(chunks,embeddings)

    retriever = get_retriever(vector_store)

    llm = load_llm()

    rag_chain =  create_rag_chain(retriever,llm)



    st.session_state.rag_chain = rag_chain
    st.session_state.retriever = retriever
    st.session_state.documents = documents
    st.session_state.llm = llm

    st.success("Documents processed")


question = st.chat_input("Ask anything...")

if (question and "rag_chain" in st.session_state):

    answer = st.session_state.rag_chain.invoke(question)   

    st.write(answer)

    docs = st.session_state.retriever.invoke(question)

    st.subheader("Sources")

    for doc in docs:

       st.write(
            f"{doc.metadata.get('source', 'Unknown')} "
            f"- Page {doc.metadata.get('page', 'Unknown')}"
        )

elif question:

    st.warning(
        "Please process the PDFs first."
    )
    

if (
    st.sidebar.button("Summarize PDFs")
    and "documents" in st.session_state
):
    
    summary = summarize_documents(st.session_state.documents,st.session_state.llm)

    st.write(summary.content)
