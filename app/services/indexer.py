from collections.abc import Callable

from app.services.bm25_index import bm25_index
from app.services.model_tasks import embed_texts
from app.services.vector_store import vector_store


async def index_chunks(
    document_id: str,
    document_name: str,
    chunks: list[str],
    pages: list[str],
    log_callback: Callable[[str], None] | None = None,
) -> None:
    if not chunks:
        return

    def log(message: str) -> None:
        if log_callback:
            log_callback(message)

    log(f"Indexing {len(chunks)} chunks (vector + BM25)...")
    all_embeddings: list[list[float]] = []
    batch_size = 10
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        response = await embed_texts(batch)
        all_embeddings.extend(response.embeddings)
        log(f"Embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

    vector_store.add_chunks(document_id, document_name, chunks, all_embeddings, pages)
    bm25_index.add_chunks(document_id, document_name, chunks, pages)
    log("Hybrid indexes updated (vector + BM25).")
