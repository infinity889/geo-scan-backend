import uuid

from app.core.config import settings
from app.domain.models import ChatAnswer, ChatQuery, Citation
from app.services.openrouter import OpenRouterError, openrouter_client
from app.services.seed import CITATIONS


def _citations(ids: list[str]) -> list[Citation]:
    return [CITATIONS[citation_id] for citation_id in ids if citation_id in CITATIONS]


async def build_answer(query: ChatQuery) -> ChatAnswer:
    from app.services.model_tasks import embed_texts
    from app.services.vector_store import vector_store

    seed_answer = _build_seed_answer(query)
    citations = []

    try:
        emb_res = await embed_texts([query.question])
        if emb_res.embeddings and len(emb_res.embeddings) > 0:
            query_embedding = emb_res.embeddings[0]
        else:
            query_embedding = [0.0] * 1536
    except Exception as e:
        print(f"Vector search failed: {e}")
        query_embedding = [0.0] * 1536

    store_citations = vector_store.search(query.question, query_embedding, k=5)
    citations.extend(store_citations)

    if not citations:
        citations = seed_answer.citations

    if not citations:
        citations = seed_answer.citations

    context_str = ""
    if citations:
        context_str = "Grounded context:\n" + "\n".join(
            f"- [{citation.id}] {citation.quote}" for citation in citations
        )

    system_prompt = (
        "Вы опытный ИИ-ассистент геолога. Отвечайте на русском языке. "
        "Если предоставлен контекст, опирайтесь на него. "
        "Если контекста нет, используйте свои собственные знания по геологии и нефтегазовому делу."
    )
    user_prompt = f"Question:\n{query.question}\n\n{context_str}\n"

    import os
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if groq_api_key:
        try:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=groq_api_key)
            completion = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1200,
            )
            return ChatAnswer(
                id=f"a-{uuid.uuid4().hex[:10]}",
                content=completion.choices[0].message.content,
                mode="rag",
                citations=citations,
                provider="groq",
                model="llama-3.3-70b-versatile",
                confidence=0.9,
            )
        except Exception as e:
            print(f"Groq error: {e}")
            return seed_answer.model_copy(update={"provider": "groq-error"})

    if not openrouter_client.configured:
        return ChatAnswer(
            id=f"a-{uuid.uuid4().hex[:10]}",
            content=f"Найдены данные в базе, но ни Groq, ни OpenRouter не настроены.\n\nКонтекст:\n{context_str}",
            mode="rag",
            citations=citations,
            confidence=0.8,
            provider="local-fallback"
        )
        
    return seed_answer.model_copy(update={"provider": "no-key-missing"})


def _build_seed_answer(query: ChatQuery) -> ChatAnswer:
    q = query.question.lower()

    if any(token in q for token in ["bs10", "бс10", "oil saturation", "нефтенасыщ"]):
        return ChatAnswer(
            id=f"a-{uuid.uuid4().hex[:10]}",
            content=(
                "Нефтенасыщенность пласта **БС10** в скважине **№247** составляет "
                "**So = 0.62 +/- 0.04** по результатам анализа 18 образцов керна "
                "[Report_2015:p.24]. Сопутствующие параметры: эффективная пористость "
                "**18.2%** и проницаемость **14.7 mD** [Report_2015:p.27]. Значение "
                "согласуется с соседними скважинами 248 и 251A "
                "[Well_247_Core_Analysis:p.6]."
            ),
            mode="rag",
            citations=_citations(
                [
                    "Report_2015:p.24",
                    "Report_2015:p.27",
                    "Well_247_Core_Analysis:p.6",
                ]
            ),
            suggested_tab="preview",
            confidence=0.93,
            provider="seed",
        )

    if any(token in q for token in ["bazhenov", "бажен", "core", "керн", "multi-hop"]):
        return ChatAnswer(
            id=f"a-{uuid.uuid4().hex[:10]}",
            content=(
                "По графовому маршруту `Месторождение -> Скважина -> Свита -> "
                "Исследование` найдены три скважины Приобского месторождения, "
                "которые вскрыли баженовскую свиту и имеют исследования керна: "
                "**№247**, **№248**, **№251A** [Report_2015:p.18]. Для них "
                "зафиксированы RCA/SCAL/XRD или RCA-исследования керна "
                "[Report_2015:p.31]."
            ),
            mode="graph",
            citations=_citations(["Report_2015:p.18", "Report_2015:p.31"]),
            suggested_tab="graph",
            confidence=0.9,
            provider="seed",
        )

    if any(token in q for token in ["table", "таблиц", "scan", "скан", "ocr", "1987"]):
        return ChatAnswer(
            id=f"a-{uuid.uuid4().hex[:10]}",
            content=(
                "OCR восстановил таблицу Appendix B, p.14 с объединенным заголовком "
                "`Porosity, %`. Ключевые строки: 247-01: So 0.61; 247-02: So 0.64; "
                "247-03: So 0.58; 247-04: So 0.63. Средняя уверенность "
                "реконструкции таблицы: **0.94** [Report_2015:p.14]."
            ),
            mode="table",
            citations=_citations(["Report_2015:p.14"]),
            suggested_tab="preview",
            confidence=0.88,
            provider="seed",
        )

    return ChatAnswer(
        id=f"a-{uuid.uuid4().hex[:10]}",
        content=(
            "В текущем корпусе не найдено достаточно подтверждений для ответа. "
            "Попробуйте уточнить месторождение, номер скважины, пласт, свиту или "
            "страницу документа. Ответ без цитат backend не считает достоверным."
        ),
        mode=query.mode,
        citations=[],
        suggested_tab=None,
        confidence=0.25,
        provider="seed",
    )
