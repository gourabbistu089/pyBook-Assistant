from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import RetrievalQA

from app.core.config import (
    GEMINI_API_KEY,
    PINECONE_INDEX_NAME
)

# -----------------------------
# GLOBAL CACHE (IMPORTANT)
# -----------------------------
_qa_chain = None


def build_qa_chain():
    """
    Build the RAG pipeline ONLY once.
    This runs on first request, not on import.
    """
    global _qa_chain

    # If already built, reuse it
    if _qa_chain is not None:
        return _qa_chain

    print("🔧 Building RAG pipeline...")

    # ---------- Embeddings ----------
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # ---------- Vector Store (EXISTING INDEX) ----------
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # ---------- LLM ----------
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY
    )

    # ---------- RAG Chain ----------
    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"
    )

    print("✅ RAG pipeline ready")
    return _qa_chain


def ask_question(query: str) -> str:
    """
    Called by FastAPI route.
    """
    qa_chain = build_qa_chain()
    response = qa_chain.invoke({"query": query})
    return response["result"]
