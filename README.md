# Geo Scan Backend

FastAPI backend prototype for the GeoRAG hackathon demo. The current version
implements stable REST contracts and in-memory demo services for document
ingestion, pipeline status, grounded QA, citations, graph fragments, figures,
and metrics.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/health

## OpenRouter Models

Create `.env` from `.env.example` and set your key:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_LLM_MODEL=deepseek/deepseek-v4-flash
OPENROUTER_OCR_MODEL=deepseek/deepseek-ocr-2
OPENROUTER_EMBEDDING_MODEL=baai/bge-m3
```

The model slugs are environment-driven because OpenRouter catalog names can
change. The defaults keep the project on open-weight/open models:

- LLM for NER, RE, and generation: `deepseek/deepseek-v4-flash`
- OCR/Vision priority: `deepseek/deepseek-ocr-2`
- Russian/multilingual embeddings: `baai/bge-m3`

If `OPENROUTER_API_KEY` is not set, the API stays usable through deterministic
local fallbacks and seeded demo answers.

## Docker

```bash
docker compose up --build
```

The API will be available at http://localhost:8000.

## Main Endpoints

- `GET /api/v1/documents` - list indexed documents
- `POST /api/v1/documents` - upload a document
- `GET /api/v1/documents/{document_id}` - get document details
- `GET /api/v1/documents/{document_id}/pipeline` - get pipeline steps
- `GET /api/v1/scenarios` - get demo scenario prompts
- `POST /api/v1/chat` - ask a grounded question
- `GET /api/v1/citations/{citation_id}` - get source preview by citation id
- `GET /api/v1/graph` - get knowledge graph fragment
- `GET /api/v1/figures` - get extracted figure summaries
- `GET /api/v1/metrics` - get prototype quality metrics
- `GET /api/v1/models/status` - show configured model provider and slugs
- `POST /api/v1/models/embeddings` - create embeddings
- `POST /api/v1/models/ner` - extract geological entities and relations

## Example

```bash
curl -X POST http://localhost:8000/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Какая нефтенасыщенность пласта БС10 в скважине №247?\"}"
```

```bash
curl -X POST http://localhost:8000/api/v1/models/ner ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"В скважине 247 пласт БС10 содержит исследования керна.\"}"
```

## Architecture

This prototype keeps the API shape close to the target system:

- Parser: format detection and structured document placeholder
- Vision: figure descriptions as indexable chunks
- NER/RE: seeded geological entities and relations
- Indexer: in-memory hybrid-search facade
- Retriever: heuristic router for factual, graph, and table queries
- Answerer: grounded answers with citation objects

Production replacements can be added behind the existing service interfaces:

- PostgreSQL or pgvector for metadata and vectors
- Qdrant, Weaviate, Milvus, or pgvector for vector retrieval
- Neo4j, Memgraph, or NetworkX for GraphRAG
- DeepSeek-OCR-2, Qwen2.5-VL, InternVL for OCR/Vision
- BGE-M3, multilingual-e5, or jina-embeddings-v3 for embeddings
- DeepSeek-V4-Flash, Qwen, or Llama for NER/RE and answer generation
