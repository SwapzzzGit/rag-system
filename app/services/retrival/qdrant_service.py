import logfire
from app.config import Settings
from app.services.retrival.embeddings import embed_query
from qdrant_client import QdrantClient

# Initializing Qdrant client

client = QdrantClient(url=Settings.QDRANT_URL, api_key=Settings.QDRANT_API_KEY)


def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        query_provider = embed_query(query)

        # Using query_points - the modern standard for Qdrant

        response = client.query_points(
            collection_name=Settings.QDRANT_COLLECTION,
            query=query_provider,
            limit=limit,
            with_payload=True,
        )
        results = []
        for res in response.points:
            results.append(
                {
                    "content": res.payload.get("text", ""),
                    "source": res.payload.get("score", ""),
                    "score": res.score,
                }
            )
        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []
