from app.domain.models import ChatQuery, Citation, QueryMode
from app.services.bm25_index import bm25_index
from app.services.graph_store import graph_store
from app.services.model_tasks import embed_texts
from app.services.vector_store import vector_store


def classify_mode(question: str, requested: QueryMode) -> QueryMode:
    if requested != "auto":
        return requested

    lowered = question.lower()
    if any(token in lowered for token in ["таблиц", "table", "скан", "scan", "ocr", "1987", "appendix", "восстанов"]):
        return "table"
    if any(
        token in lowered
        for token in ["бажен", "bazhenov", "керн", "core", "multi-hop", "граф", "исследован"]
    ):
        return "graph"
    if any(token in lowered for token in ["bs10", "бс10", "нефтенасыщ", "oil saturation", "порист"]):
        return "rag"
    return "rag"


def _merge_citations(*groups: list[Citation], limit: int = 5) -> list[Citation]:
    merged: dict[str, Citation] = {}
    for group in groups:
        for citation in group:
            current = merged.get(citation.id)
            if current is None or citation.score > current.score:
                merged[citation.id] = citation
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]


async def retrieve(query: ChatQuery) -> tuple[QueryMode, list[Citation]]:
    mode = classify_mode(query.question, query.mode)

    if mode == "graph":
        return mode, graph_store.search_citations(query.question, k=5)

    try:
        embedding_response = await embed_texts([query.question])
        query_embedding = embedding_response.embeddings[0]
    except Exception:
        query_embedding = [0.0] * 32

    vector_hits = vector_store.search(query.question, query_embedding, k=5)
    bm25_hits = bm25_index.search(query.question, k=5)
    citations = _merge_citations(vector_hits, bm25_hits, limit=5)

    if mode == "table" and not citations:
        citations = graph_store.search_citations(query.question, k=3)

    return mode, citations
