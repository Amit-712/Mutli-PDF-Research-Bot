def summarize_documents(docs,llm):
    
    combined_text = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
    Summarize the uploaded
    research papers.

    Include:

    1. Main topic
    2. Key findings
    3. Important conclusions

    Documents: {combined_text}"""

    return llm.invoke(prompt)