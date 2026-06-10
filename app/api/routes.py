from fastapi import APIRouter, File, HTTPException, UploadFile

from app.domain.models import (
    ChatAnswer,
    ChatQuery,
    DocumentSummary,
    EmbeddingRequest,
    EmbeddingResponse,
    EntityExtractionRequest,
    EntityExtractionResponse,
    FigureSummary,
    KnowledgeGraph,
    MetricsReport,
    ModelStatus,
    PipelineStep,
    Scenario,
    SourcePreview,
)
from app.core.config import settings
from app.services.answerer import build_answer
from app.services.model_tasks import embed_texts, extract_geo_knowledge
from app.services.seed import (
    FIGURES,
    GRAPH_EDGES,
    GRAPH_NODES,
    METRICS,
    SCENARIOS,
    SOURCE_PREVIEWS,
)
from app.services.store import store
from app.db.repository import figure_repo, graph_repo
from app.db.session import get_session, is_db_enabled

router = APIRouter()


@router.get("/documents", response_model=list[DocumentSummary], tags=["documents"])
def list_documents() -> list[DocumentSummary]:
    return store.list_documents()


@router.post("/documents", response_model=DocumentSummary, status_code=201, tags=["documents"])
async def upload_document(file: UploadFile = File(...)) -> DocumentSummary:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    try:
        return store.save_upload(file.filename, file.file)
    finally:
        await file.close()


@router.get("/documents/{document_id}", response_model=DocumentSummary, tags=["documents"])
def get_document(document_id: str) -> DocumentSummary:
    document = store.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@router.get(
    "/documents/{document_id}/pipeline",
    response_model=list[PipelineStep],
    tags=["documents"],
)
def get_document_pipeline(document_id: str) -> list[PipelineStep]:
    document = store.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document.steps


@router.get("/scenarios", response_model=list[Scenario], tags=["demo"])
def list_scenarios() -> list[Scenario]:
    return SCENARIOS


@router.post("/chat", response_model=ChatAnswer, tags=["qa"])
async def ask_question(query: ChatQuery) -> ChatAnswer:
    return await build_answer(query)


@router.get("/models/status", response_model=ModelStatus, tags=["models"])
def get_model_status() -> ModelStatus:
    return ModelStatus(
        provider="groq",
        configured=settings.groq_enabled,
        base_url=settings.groq_base_url,
        llm_model=settings.groq_llm_model,
        ocr_model=settings.groq_vision_model,
        embedding_model="local-fastembed/multilingual-e5-large",
    )


@router.post("/models/embeddings", response_model=EmbeddingResponse, tags=["models"])
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    return await embed_texts(request.texts)


@router.post("/models/ner", response_model=EntityExtractionResponse, tags=["models"])
async def extract_entities(request: EntityExtractionRequest) -> EntityExtractionResponse:
    return await extract_geo_knowledge(request.text, request.source_id)


@router.get("/citations/{citation_id}", response_model=SourcePreview, tags=["sources"])
def get_citation(citation_id: str) -> SourcePreview:
    preview = SOURCE_PREVIEWS.get(citation_id)
    if preview is not None:
        return preview

    if citation_id.startswith("Report_2015:p."):
        base = SOURCE_PREVIEWS["Report_2015:p.24"]
        return base.model_copy(
            update={
                "citation_id": citation_id,
                "page": citation_id.split(":p.")[-1],
            }
        )

    raise HTTPException(status_code=404, detail="Citation not found.")


@router.get("/graph", response_model=KnowledgeGraph, tags=["graph"])
def get_graph() -> KnowledgeGraph:
    if is_db_enabled():
        with get_session() as session:
            return graph_repo.get_graph(session)
    return KnowledgeGraph(nodes=GRAPH_NODES, edges=GRAPH_EDGES)


@router.get("/figures", response_model=list[FigureSummary], tags=["figures"])
def list_figures() -> list[FigureSummary]:
    if is_db_enabled():
        with get_session() as session:
            return figure_repo.list_figures(session)
    return FIGURES


@router.get("/metrics", response_model=MetricsReport, tags=["metrics"])
def get_metrics() -> MetricsReport:
    return METRICS


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"])
def delete_document(document_id: str) -> None:
    store.delete_document(document_id)
