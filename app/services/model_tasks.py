from app.core.config import settings
from app.domain.models import (
    EmbeddingResponse,
    EntityExtractionResponse,
    ExtractedEntity,
    ExtractedRelation,
)
from app.services.openrouter import (
    OpenRouterError,
    deterministic_embedding,
    openrouter_client,
    parse_json_object,
)


async def embed_texts(texts: list[str]) -> EmbeddingResponse:
    if not openrouter_client.configured:
        embeddings = [deterministic_embedding(text) for text in texts]
        return EmbeddingResponse(
            provider="local-fallback",
            model="deterministic-dev-embedding",
            dimensions=len(embeddings[0]),
            embeddings=embeddings,
        )

    vectors = await openrouter_client.embeddings(texts)
    dimensions = len(vectors[0]) if vectors else 0
    return EmbeddingResponse(
        provider="openrouter",
        model=settings.openrouter_embedding_model,
        dimensions=dimensions,
        embeddings=vectors,
    )


async def extract_geo_knowledge(text: str, source_id: str | None = None) -> EntityExtractionResponse:
    if not openrouter_client.configured:
        return _fallback_geo_extraction(text)

    prompt = (
        "You extract geological NER and relation triples from Russian/Kazakh/oilfield "
        "documents. Return only JSON with keys `entities` and `relations`. "
        "Entity fields: text, type, normalized, confidence. Relation fields: "
        "source, target, type, confidence. Use types such as Field, Well, Layer, "
        "Formation, Horizon, Measurement, Date, Unit, Study. Every extracted item "
        "must be grounded in the provided text."
    )
    if source_id:
        prompt += f" Source id: {source_id}."

    try:
        result = await openrouter_client.chat_completion(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            model=settings.openrouter_llm_model,
            temperature=0,
            max_tokens=1600,
            response_format={"type": "json_object"},
        )
        payload = parse_json_object(result.content)
        return EntityExtractionResponse(
            provider="openrouter",
            model=result.model,
            entities=[ExtractedEntity(**item) for item in payload.get("entities", [])],
            relations=[ExtractedRelation(**item) for item in payload.get("relations", [])],
            raw=result.content,
        )
    except (OpenRouterError, ValueError, TypeError):
        fallback = _fallback_geo_extraction(text)
        return fallback.model_copy(update={"provider": "local-fallback-after-openrouter-error"})


def _fallback_geo_extraction(text: str) -> EntityExtractionResponse:
    lowered = text.lower()
    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []

    if "247" in text:
        entities.append(
            ExtractedEntity(text="247", type="Well", normalized="Well 247", confidence=0.7)
        )
    if "бс10" in lowered or "bs10" in lowered:
        entities.append(
            ExtractedEntity(text="БС10", type="Layer", normalized="Layer BS10", confidence=0.7)
        )
    if "бажен" in lowered or "bazhenov" in lowered:
        entities.append(
            ExtractedEntity(
                text="баженовская свита",
                type="Formation",
                normalized="Bazhenov Formation",
                confidence=0.65,
            )
        )
    if "керн" in lowered or "core" in lowered:
        entities.append(
            ExtractedEntity(text="керн", type="Study", normalized="Core Studies", confidence=0.65)
        )

    if any(entity.type == "Well" for entity in entities) and any(
        entity.type == "Layer" for entity in entities
    ):
        relations.append(
            ExtractedRelation(
                source="Well 247",
                target="Layer BS10",
                type="intersects",
                confidence=0.55,
            )
        )

    return EntityExtractionResponse(
        provider="local-fallback",
        model=None,
        entities=entities,
        relations=relations,
        raw=None,
    )
