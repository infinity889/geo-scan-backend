from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


StepStatus = Literal["pending", "processing", "completed", "failed"]
QueryMode = Literal["auto", "rag", "graph", "table"]


class PipelineStep(BaseModel):
    id: int
    title: str
    detail: str
    status: StepStatus = "pending"
    log: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    id: str
    name: str
    size: str
    size_bytes: int
    format: str
    status: StepStatus
    created_at: datetime
    steps: list[PipelineStep]


class Scenario(BaseModel):
    key: str
    label: str
    question: str
    mode: QueryMode = "auto"


class Citation(BaseModel):
    id: str
    label: str
    document_id: str
    document_name: str
    page: str
    chunk_id: str
    quote: str
    score: float = Field(ge=0, le=1)


class ChatQuery(BaseModel):
    question: str = Field(min_length=1)
    mode: QueryMode = "auto"
    conversation_id: str | None = None


class ChatAnswer(BaseModel):
    id: str
    role: Literal["assistant"] = "assistant"
    content: str
    mode: QueryMode
    citations: list[Citation] = Field(default_factory=list)
    suggested_tab: Literal["preview", "graph", "figures"] | None = None
    confidence: float = Field(ge=0, le=1)
    provider: str = "seed"
    model: str | None = None


class GraphNode(BaseModel):
    id: str
    label: str
    type: Literal["Field", "Well", "Formation", "Layer", "Horizon", "Measurement", "Study"]
    properties: dict[str, str | int | float] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str
    evidence: list[str] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class FigureSummary(BaseModel):
    id: str
    title: str
    type: Literal["structural_map", "well_log", "cross_section", "core_photo", "other"]
    document_id: str
    page: str
    description: str
    citations: list[str] = Field(default_factory=list)


class SourcePreview(BaseModel):
    citation_id: str
    document_id: str
    document_name: str
    page: str
    title: str
    text: str
    highlighted_text: str
    tables: list[dict[str, object]] = Field(default_factory=list)
    formulas: list[dict[str, str]] = Field(default_factory=list)


class MetricValue(BaseModel):
    name: str
    value: float
    unit: str
    description: str


class MetricsReport(BaseModel):
    parser: list[MetricValue]
    retrieval: list[MetricValue]
    qa: list[MetricValue]
    latency: list[MetricValue]


class ModelStatus(BaseModel):
    provider: str
    configured: bool
    base_url: str
    llm_model: str
    ocr_model: str
    embedding_model: str


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class EmbeddingResponse(BaseModel):
    provider: str
    model: str
    dimensions: int
    embeddings: list[list[float]]


class EntityExtractionRequest(BaseModel):
    text: str = Field(min_length=1)
    source_id: str | None = None


class ExtractedEntity(BaseModel):
    text: str
    type: str
    normalized: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)


class ExtractedRelation(BaseModel):
    source: str
    target: str
    type: str
    confidence: float = Field(default=0.0, ge=0, le=1)


class EntityExtractionResponse(BaseModel):
    provider: str
    model: str | None
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    raw: str | None = None
