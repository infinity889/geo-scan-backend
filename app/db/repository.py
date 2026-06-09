from datetime import datetime, timezone
from typing import Any, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
    delete,
    desc,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.orm import Session, joinedload

from app.db.models import (
    Document,
    DocumentChunk,
    ExtractedEntityRecord,
    ExtractedRelationRecord,
    FigureRecord,
    GraphEdgeRecord,
    GraphNodeRecord,
    PipelineStepRecord,
)
from app.domain.models import (
    Citation,
    DocumentSummary,
    FigureSummary,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    PipelineStep,
)
from app.services.seed import FIGURES, GRAPH_EDGES, GRAPH_NODES, STEP_LOGS, initial_steps


def _normalize_embedding(embedding: list[float], dimensions: int = 1024) -> list[float]:
    if len(embedding) == dimensions:
        return embedding
    if len(embedding) > dimensions:
        return embedding[:dimensions]
    return embedding + [0.0] * (dimensions - len(embedding))


class DocumentRepository:
    def list_documents(self, session: Session) -> list[DocumentSummary]:
        documents = (
            session.execute(
                select(Document)
                .options(joinedload(Document.steps))
                .order_by(Document.created_at.desc())
            )
            .unique()
            .scalars()
            .all()
        )
        return [self._to_summary(doc) for doc in documents]

    def delete_document(self, session: Session, document_id: str) -> None:
        session.execute(delete(Document).where(Document.id == document_id))
        session.commit()

    def get_document(self, session: Session, document_id: str) -> DocumentSummary | None:
        doc = session.execute(
            select(Document)
            .options(joinedload(Document.steps))
            .where(Document.id == document_id)
        ).unique().scalar_one_or_none()
        if doc is None:
            return None
        return self._to_summary(doc)

    def create_document(
        self,
        session: Session,
        *,
        document_id: str,
        name: str,
        size_bytes: int,
        file_format: str,
        file_path: str,
        status: str = "pending",
        steps: list[PipelineStep] | None = None,
    ) -> DocumentSummary:
        step_models = steps or initial_steps()
        doc = Document(
            id=document_id,
            name=name,
            size_bytes=size_bytes,
            format=file_format,
            status=status,
            file_path=file_path,
            created_at=datetime.now(timezone.utc),
        )
        doc.steps = [
            PipelineStepRecord(
                document_id=document_id,
                step_id=step.id,
                title=step.title,
                detail=step.detail,
                status=step.status,
                log=step.log,
            )
            for step in step_models
        ]
        session.add(doc)
        session.flush()
        return self._to_summary(doc)

    def update_document_status(self, session: Session, document_id: str, status: str) -> None:
        doc = session.get(Document, document_id)
        if doc is None:
            return
        doc.status = status

    def update_step(
        self,
        session: Session,
        document_id: str,
        step_id: int,
        status: str,
        log: list[str] | None = None,
    ) -> None:
        step = session.execute(
            select(PipelineStepRecord).where(
                PipelineStepRecord.document_id == document_id,
                PipelineStepRecord.step_id == step_id,
            )
        ).scalar_one_or_none()
        if step is None:
            return
        step.status = status
        if log is not None:
            step.log = log

    def append_step_log(
        self, session: Session, document_id: str, step_id: int, line: str
    ) -> None:
        step = session.execute(
            select(PipelineStepRecord).where(
                PipelineStepRecord.document_id == document_id,
                PipelineStepRecord.step_id == step_id,
            )
        ).scalar_one_or_none()
        if step is None:
            return
        step.log = [*step.log, line]

    def seed_demo_documents(self, session: Session) -> None:
        existing = session.execute(select(Document.id)).scalars().first()
        if existing is not None:
            return

        for doc_id, name, size_bytes, file_format in [
            ("doc-1", "Priobskoye_Report_2015.pdf", 8_808_038, "PDF"),
            ("doc-2", "Well_247_Core_Analysis.docx", 2_202_010, "DOCX"),
        ]:
            steps = [
                step.model_copy(
                    update={"status": "completed", "log": STEP_LOGS.get(step.id, [])}
                )
                for step in initial_steps()
            ]
            self.create_document(
                session,
                document_id=doc_id,
                name=name,
                size_bytes=size_bytes,
                file_format=file_format,
                file_path=f"seed/{name}",
                status="completed",
                steps=steps,
            )

    @staticmethod
    def _to_summary(doc: Document) -> DocumentSummary:
        from app.services.utils import human_size

        steps = sorted(doc.steps, key=lambda item: item.step_id)
        return DocumentSummary(
            id=doc.id,
            name=doc.name,
            size=human_size(doc.size_bytes),
            size_bytes=doc.size_bytes,
            format=doc.format,
            status=doc.status,  # type: ignore[arg-type]
            created_at=doc.created_at,
            steps=[
                PipelineStep(
                    id=step.step_id,
                    title=step.title,
                    detail=step.detail,
                    status=step.status,  # type: ignore[arg-type]
                    log=step.log or [],
                )
                for step in steps
            ],
        )


class ChunkRepository:
    def add_chunks(
        self,
        session: Session,
        *,
        document_id: str,
        document_name: str,
        chunks: list[str],
        embeddings: list[list[float]],
        pages: list[str],
        chunk_type: str = "text",
    ) -> None:
        if not chunks:
            return

        session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )

        for index, (content, embedding, page) in enumerate(zip(chunks, embeddings, pages)):
            session.add(
                DocumentChunk(
                    id=f"{document_id}-{index}",
                    document_id=document_id,
                    document_name=document_name,
                    chunk_index=index,
                    page=str(page),
                    content=content,
                    chunk_type=chunk_type,
                    chunk_metadata={},
                    embedding=_normalize_embedding(embedding),
                )
            )
        session.flush()
        # Update tsvector for BM25
        session.execute(
            text(
                "UPDATE document_chunks SET content_tsvector = to_tsvector('russian', content) "
                "WHERE document_id = :doc_id"
            ),
            {"doc_id": document_id},
        )

    def search(
        self,
        session: Session,
        query_text: str,
        embedding: list[float],
        k: int = 5,
    ) -> list[Citation]:
        query_vector = _normalize_embedding(embedding)
        
        # 1. Vector Search
        distance = DocumentChunk.embedding.cosine_distance(query_vector)
        vector_results = session.execute(
            select(DocumentChunk, (1.0 - distance).label("score"))
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(distance)
            .limit(k * 2)
        ).all()

        # 2. BM25 (Full Text Search)
        # Using plainto_tsquery for better natural language handling
        fts_results = session.execute(
            text(f"""
                SELECT id, ts_rank(content_tsvector, plainto_tsquery('russian', :q)) as fts_score
                FROM document_chunks
                WHERE content_tsvector @@ plainto_tsquery('russian', :q)
                ORDER BY fts_score DESC
                LIMIT :limit
            """),
            {"q": query_text, "limit": k * 2}
        ).all()

        # 3. Hybrid Ranking (Simplified)
        scores: dict[str, float] = {}
        chunks_map: dict[str, DocumentChunk] = {}

        for chunk, score in vector_results:
            scores[chunk.id] = float(score) * 0.7  # Vector weight
            chunks_map[chunk.id] = chunk
        
        for chunk_id, fts_score in fts_results:
            if chunk_id in scores:
                scores[chunk_id] += float(fts_score) * 0.3 # FTS weight
            else:
                # If not in vector top, we might need to fetch the chunk record
                scores[chunk_id] = float(fts_score) * 0.3
                if chunk_id not in chunks_map:
                    chunks_map[chunk_id] = session.get(DocumentChunk, chunk_id)

        # Sort and return top K
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        
        citations: list[Citation] = []
        for cid in sorted_ids:
            chunk = chunks_map[cid]
            if not chunk: continue
            
            citations.append(
                Citation(
                    id=f"{chunk.document_name}:p.{chunk.page}",
                    label=f"{chunk.document_name}:p.{chunk.page}",
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page=chunk.page,
                    chunk_id=chunk.id,
                    quote=chunk.content,
                    score=min(1.0, scores[cid]),
                )
            )

        return citations


class GraphRepository:
    def seed_demo_graph(self, session: Session) -> None:
        if session.execute(select(GraphNodeRecord.id)).first() is not None:
            return

        for node in GRAPH_NODES:
            session.add(
                GraphNodeRecord(
                    id=node.id,
                    label=node.label,
                    node_type=node.type,
                    properties=node.properties,
                )
            )
        session.flush()
        for edge in GRAPH_EDGES:
            session.add(
                GraphEdgeRecord(
                    id=edge.id,
                    source_node_id=edge.source,
                    target_node_id=edge.target,
                    label=edge.label,
                    evidence=edge.evidence,
                )
            )

    def get_graph(self, session: Session) -> KnowledgeGraph:
        nodes = session.execute(select(GraphNodeRecord)).scalars().all()
        edges = session.execute(select(GraphEdgeRecord)).scalars().all()
        return KnowledgeGraph(
            nodes=[
                GraphNode(
                    id=node.id,
                    label=node.label,
                    type=node.node_type,  # type: ignore[arg-type]
                    properties=node.properties or {},
                )
                for node in nodes
            ],
            edges=[
                GraphEdge(
                    id=edge.id,
                    source=edge.source_node_id,
                    target=edge.target_node_id,
                    label=edge.label,
                    evidence=edge.evidence or [],
                )
                for edge in edges
            ],
        )


class FigureRepository:
    def seed_demo_figures(self, session: Session) -> None:
        if session.execute(select(FigureRecord.id)).first() is not None:
            return

        for figure in FIGURES:
            session.add(
                FigureRecord(
                    id=figure.id,
                    title=figure.title,
                    figure_type=figure.type,
                    document_id=figure.document_id,
                    page=figure.page,
                    description=figure.description,
                    citations=figure.citations,
                )
            )

    def list_figures(self, session: Session) -> list[FigureSummary]:
        rows = session.execute(select(FigureRecord)).scalars().all()
        return [
            FigureSummary(
                id=row.id,
                title=row.title,
                type=row.figure_type,  # type: ignore[arg-type]
                document_id=row.document_id,
                page=row.page,
                description=row.description,
                citations=row.citations or [],
            )
            for row in rows
        ]


class EntityRepository:
    def save_extraction(
        self,
        session: Session,
        *,
        document_id: str | None,
        chunk_id: str | None,
        entities: list[dict],
        relations: list[dict],
    ) -> None:
        for entity in entities:
            session.add(
                ExtractedEntityRecord(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=entity["text"],
                    entity_type=entity["type"],
                    normalized=entity.get("normalized"),
                    confidence=float(entity.get("confidence", 0.0)),
                )
            )
        for relation in relations:
            session.add(
                ExtractedRelationRecord(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    source_text=relation["source"],
                    target_text=relation["target"],
                    relation_type=relation["type"],
                    confidence=float(relation.get("confidence", 0.0)),
                )
            )


document_repo = DocumentRepository()
chunk_repo = ChunkRepository()
graph_repo = GraphRepository()
figure_repo = FigureRepository()
entity_repo = EntityRepository()
