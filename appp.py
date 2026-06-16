from src.pdf_loader import load_pdf
from src.chunker import chunk_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.rag_chain import load_llm,create_rag_chain
from src.memory import add_interaction, get_chat_history
from langchain.messages import HumanMessage,AIMessage
from src.summarizer import summarize_documents

docs = load_pdf("dl-curriculum.pdf")

chunks = chunk_documents(docs)

embeddings = get_embedding_model()

vector_store = create_vector_store(chunks,embeddings)

retriever = get_retriever(vector_store)

llm = load_llm()

rag_chain = create_rag_chain(retriever,llm)



while True:

    question = input("\nAsk your Query : ")
    if question == 'exit':
        break

    # Adding Source Citations
    doc = retriever.invoke(question)

    sources = []
    
    seen = set()

    for d in doc:

        source = d.metadata.get("source")

        page = d.metadata.get("page")


        source_page = (
            f"{source} - Page {page}"
        )

    # preventing the same pages from being repeated
        if source_page not in seen:
            seen.add(source_page)
            sources.append(source_page)

    # Invoking the chain and print the response
    response = rag_chain.invoke(question)

    # saving the chat
    add_interaction(question, response)

    print("\n",response)

    # Printing the sources 
    print("\nSources:")

    for source in sources:
        print(source)


# for chat in get_chat_history():
    
#     if isinstance(chat,HumanMessage):
#         print("Human: ", chat.content)
#     elif isinstance(chat,AIMessage):
#         print("AI: ",chat.content)

#     print()


# summarizing the documents
summarize = summarize_documents(docs,llm)   
print(type(summarize))
print(summarize.content)