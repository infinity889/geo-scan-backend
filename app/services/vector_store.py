import json
import math
import uuid
from pathlib import Path
from typing import List

from app.core.config import settings
from app.db.repository import chunk_repo
from app.db.session import get_session, is_db_enabled
from app.domain.models import Citation


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class JsonVectorStore:
    """Fallback vector store for local development without PostgreSQL."""

    def __init__(self) -> None:
        self.db_path = settings.upload_dir / "simple_vector_store.json"
        self.documents: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.db_path.exists():
            try:
                with open(self.db_path, encoding="utf-8") as handle:
                    self.documents = json.load(handle)
            except Exception:
                self.documents = []

    def _save(self) -> None:
        with open(self.db_path, "w", encoding="utf-8") as handle:
            json.dump(self.documents, handle, ensure_ascii=False)

    def add_chunks(
        self,
        document_id: str,
        document_name: str,
        chunks: List[str],
        embeddings: List[List[float]],
        pages: List[str],
    ) -> None:
        if not chunks:
            return

        for index in range(len(chunks)):
            self.documents.append(
                {
                    "id": f"{document_id}-{index}",
                    "document_id": document_id,
                    "document_name": document_name,
                    "page": pages[index],
                    "chunk": chunks[index],
                    "embedding": embeddings[index],
                }
            )
        self._save()

    def search(self, query_text: str, embedding: List[float], k: int = 5) -> List[Citation]:
        if not self.documents:
            return []

        q_words = {w.strip(',.?!"\'') for w in query_text.lower().split()}
        results: list[tuple[float, dict]] = []

        for doc in self.documents:
            score = cosine_similarity(embedding, doc["embedding"])
            if score < 0.1 and q_words:
                t_words = {w.strip(',.?!"\'') for w in doc["chunk"].lower().split()}
                overlap = len(q_words.intersection(t_words))
                score = min(1.0, overlap * 0.15)
            results.append((score, doc))

        results.sort(key=lambda item: item[0], reverse=True)
        citations: list[Citation] = []
        for score, doc in results[:k]:
            if score < 0.1:
                continue
            citations.append(
                Citation(
                    id=f"{doc['document_name']}:p.{doc['page']}",
                    label=f"{doc['document_name']}:p.{doc['page']}",
                    document_id=doc["document_id"],
                    document_name=doc["document_name"],
                    page=str(doc["page"]),
                    chunk_id=str(uuid.uuid4().hex)[:8],
                    quote=doc["chunk"],
                    score=score,
                )
            )
        return citations


class VectorStore:
    def __init__(self) -> None:
        self._fallback = JsonVectorStore()

    def add_chunks(
        self,
        document_id: str,
        document_name: str,
        chunks: List[str],
        embeddings: List[List[float]],
        pages: List[str],
    ) -> None:
        if is_db_enabled():
            with get_session() as session:
                chunk_repo.add_chunks(
                    session,
                    document_id=document_id,
                    document_name=document_name,
                    chunks=chunks,
                    embeddings=embeddings,
                    pages=pages,
                )
            return
        self._fallback.add_chunks(
            document_id, document_name, chunks, embeddings, pages
        )

    def search(self, query_text: str, embedding: List[float], k: int = 5) -> List[Citation]:
        if is_db_enabled():
            with get_session() as session:
                return chunk_repo.search(session, query_text, embedding, k=k)
        return self._fallback.search(query_text, embedding, k=k)


vector_store = VectorStore()
