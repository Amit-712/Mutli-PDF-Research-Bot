from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

def load_llm():

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )

    return llm

def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template(
    """You are a research assistant.

        Answer only from the provided context.

        Context:{context}

        Question:{question}

        Provide concise answers."""
)


def create_rag_chain(retriever, llm):

    rag_chain_parallel = ({
        "context" : retriever | RunnableLambda(format_docs),
        "question" : RunnablePassthrough()
    })

    main_chain = rag_chain_parallel | prompt | llm | StrOutputParser()

    return main_chain
