import uuid
import json
from pathlib import Path
from typing import List, Dict, Any
import math

from app.core.config import settings
from app.domain.models import Citation

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class VectorStore:
    def __init__(self) -> None:
        self.db_path = settings.upload_dir / "simple_vector_store.json"
        self.documents = []
        self._load()

    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception:
                self.documents = []

    def _save(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def add_chunks(
        self,
        document_id: str,
        document_name: str,
        chunks: List[str],
        embeddings: List[List[float]],
        pages: List[str]
    ) -> None:
        if not chunks:
            return
            
        for i in range(len(chunks)):
            self.documents.append({
                "id": f"{document_id}-{i}",
                "document_id": document_id,
                "document_name": document_name,
                "page": pages[i],
                "chunk": chunks[i],
                "embedding": embeddings[i]
            })
            
        self._save()

    def search(self, query_text: str, embedding: List[float], k: int = 5) -> List[Citation]:
        if not self.documents:
            return []
            
        results = []
        q_words = set([w.strip(',.?!"\'') for w in query_text.lower().split()])
        
        import re
        target_pages = set(re.findall(r'\b\d+\b', query_text))
        
        for doc in self.documents:
            score = cosine_similarity(embedding, doc["embedding"])
            
            # Fallback to simple keyword overlap if embeddings are broken (e.g. out of credits)
            if score < 0.1 and q_words:
                t_words = set([w.strip(',.?!"\'') for w in doc["chunk"].lower().split()])
                overlap = len(q_words.intersection(t_words))
                # Boost score based on overlap so it passes the threshold
                score = max(score, min(1.0, overlap * 0.15))
                
            # Boost score if the user explicitly asked for this page number
            if target_pages and str(doc["page"]) in target_pages:
                # Add a strong boost so exact page matches always float to the top
                score = min(1.0, score + 0.8)
                
            results.append((score, doc))
            
        # Sort by highest score first
        results.sort(key=lambda x: x[0], reverse=True)
        top_k = results[:k]
        
        citations = []
        for score, doc in top_k:
            if score < 0.1: # Relaxed threshold
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
                    score=score
                )
            )
        return citations

vector_store = VectorStore()
