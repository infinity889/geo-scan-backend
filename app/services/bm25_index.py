import json
import math
import re
from collections import Counter
from pathlib import Path

from app.core.config import settings
from app.domain.models import Citation


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())


class BM25Index:
    def __init__(self) -> None:
        self.db_path = settings.upload_dir / "bm25_index.json"
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
        chunks: list[str],
        pages: list[str],
    ) -> None:
        if not chunks:
            return
        for index, chunk in enumerate(chunks):
            self.documents.append(
                {
                    "id": f"{document_id}-bm25-{index}",
                    "document_id": document_id,
                    "document_name": document_name,
                    "page": pages[index],
                    "chunk": chunk,
                    "tokens": _tokenize(chunk),
                }
            )
        self._save()

    def search(self, query_text: str, k: int = 5) -> list[Citation]:
        if not self.documents:
            return []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        avg_len = sum(len(doc["tokens"]) for doc in self.documents) / len(self.documents)
        doc_freq: Counter[str] = Counter()
        for doc in self.documents:
            doc_freq.update(set(doc["tokens"]))

        scored: list[tuple[float, dict]] = []
        k1, b = 1.5, 0.75
        for doc in self.documents:
            tf = Counter(doc["tokens"])
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq == 0:
                    continue
                idf = math.log(1 + (len(self.documents) - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
                denom = freq + k1 * (1 - b + b * len(doc["tokens"]) / avg_len)
                score += idf * (freq * (k1 + 1)) / denom
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        citations: list[Citation] = []
        for score, doc in scored[:k]:
            citations.append(
                Citation(
                    id=f"{doc['document_name']}:p.{doc['page']}",
                    label=f"{doc['document_name']}:p.{doc['page']}",
                    document_id=doc["document_id"],
                    document_name=doc["document_name"],
                    page=str(doc["page"]),
                    chunk_id=doc["id"],
                    quote=doc["chunk"][:500],
                    score=min(1.0, score / 10),
                )
            )
        return citations


bm25_index = BM25Index()
