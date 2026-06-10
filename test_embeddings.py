import asyncio
from app.services.model_tasks import embed_texts

async def test():
    texts = ["подскажи что в 6 модуле", "6.7. Модуль метрик и оценки качества"]
    res = await embed_texts(texts)
    print(f"Provider: {res.provider}, Model: {res.model}")
    print(f"Dimensions: {res.dimensions}")
    print(f"Embeddings count: {len(res.embeddings)}")
    print(f"First embedding snippet: {res.embeddings[0][:5]}")
    
    from app.services.vector_store import cosine_similarity
    sim = cosine_similarity(res.embeddings[0], res.embeddings[1])
    print(f"Similarity: {sim}")

if __name__ == "__main__":
    asyncio.run(test())
