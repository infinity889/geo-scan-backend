from datetime import datetime, timezone
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    format: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    file_path: Mapped[Optional[str]] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    steps: Mapped[list["PipelineStepRecord"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class PipelineStepRecord(Base):
    __tablename__ = "pipeline_steps"
    __table_args__ = (UniqueConstraint("document_id", "step_id", name="uq_doc_step"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    step_id: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    detail: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    log: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)

    document: Mapped["Document"] = relationship(back_populates="steps")


class DocumentChunk(Base):
    """Text chunk with pgvector embedding for hybrid RAG retrieval."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_name: Mapped[str] = mapped_column(String(512), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(64), nullable=False, default="text")
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    # Note: content_tsvector is managed by triggers/manual updates in PG for BM25

    document: Mapped["Document"] = relationship(back_populates="chunks")


class GraphNodeRecord(Base):
    __tablename__ = "graph_nodes"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(512), nullable=False)
    node_type: Mapped[str] = mapped_column(String(64), nullable=False)
    properties: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    source_document_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="SET NULL")
    )
    source_chunk_id: Mapped[Optional[str]] = mapped_column(String(128))


class GraphEdgeRecord(Base):
    __tablename__ = "graph_edges"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_node_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False
    )
    target_node_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    source_document_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="SET NULL")
    )


class ExtractedEntityRecord(Base):
    __tablename__ = "extracted_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_id: Mapped[Optional[str]] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(String(512), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    normalized: Mapped[Optional[str]] = mapped_column(String(512))
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class ExtractedRelationRecord(Base):
    __tablename__ = "extracted_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_id: Mapped[Optional[str]] = mapped_column(String(128))
    source_text: Mapped[str] = mapped_column(String(512), nullable=False)
    target_text: Mapped[str] = mapped_column(String(512), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class FigureRecord(Base):
    __tablename__ = "figures"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    figure_type: Mapped[str] = mapped_column(String(64), nullable=False)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
