from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_seed_documents() -> None:
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert len(response.json()) >= 2


def test_fact_question_returns_citation() -> None:
    response = client.post(
        "/api/v1/chat",
        json={"question": "Какая нефтенасыщенность пласта БС10 в скважине №247?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "rag"
    assert payload["citations"][0]["id"] == "Report_2015:p.24"
    assert payload["provider"].startswith("seed")


def test_model_status() -> None:
    response = client.get("/api/v1/models/status")
    assert response.status_code == 200
    assert response.json()["embedding_model"] == "baai/bge-m3"


def test_embeddings_fallback_without_api_key() -> None:
    response = client.post("/api/v1/models/embeddings", json={"texts": ["пласт БС10"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "local-fallback"
    assert payload["dimensions"] > 0


def test_ner_fallback_without_api_key() -> None:
    response = client.post(
        "/api/v1/models/ner",
        json={"text": "В скважине 247 пласт БС10 содержит исследования керна."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "local-fallback"
    assert any(entity["type"] == "Well" for entity in payload["entities"])
