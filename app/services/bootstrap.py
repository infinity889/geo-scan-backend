import asyncio

from app.services.bm25_index import bm25_index
from app.services.indexer import index_chunks
from app.services.seed import CITATIONS
from app.services.vector_store import vector_store


SEED_CORPUS: list[dict[str, str]] = [
    {
        "document_id": citation.document_id,
        "document_name": citation.document_name,
        "page": citation.page,
        "chunk": (
            f"{citation.quote} "
            f"Документ: {citation.document_name}, страница {citation.page}. "
            f"Идентификатор фрагмента: {citation.id}."
        ),
    }
    for citation in CITATIONS.values()
]


def _seed_already_present() -> bool:
    seed_names = {item["document_name"] for item in SEED_CORPUS}
    return any(doc["document_name"] in seed_names for doc in vector_store.documents)


async def bootstrap_indexes() -> None:
    if _seed_already_present():
        return

    for item in SEED_CORPUS:
        await index_chunks(
            item["document_id"],
            item["document_name"],
            [item["chunk"]],
            [item["page"]],
        )

    # Ensure BM25 has the same seed even if vector store was partially populated earlier.
    if not bm25_index.documents:
        for item in SEED_CORPUS:
            bm25_index.add_chunks(
                item["document_id"],
                item["document_name"],
                [item["chunk"]],
                [item["page"]],
            )


def bootstrap_indexes_sync() -> None:
    asyncio.run(bootstrap_indexes())
