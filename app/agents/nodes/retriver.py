import logfire
from app.agents.state import agentState
from app.services.retrival.ranking_service import rerank_documents
from app.services.retrival.qdrant_service import search_enterprise_knowledge


# D:\rag\advanced-rag\app\services\retrival
def retrieve_node(state: agentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]

    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        doc_contents = [doc["content"] for doc in raw_results]

        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")

        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]

    return {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"],
    }
