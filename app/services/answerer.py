import uuid

from app.core.config import settings
from app.domain.models import ChatAnswer, ChatQuery, Citation
from app.services.llm import LLMError, llm_client
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
            query_embedding = [0.0] * settings.embedding_dimensions
    except Exception as e:
        print(f"Vector search failed: {e}")
        query_embedding = [0.0] * settings.embedding_dimensions

    store_citations = vector_store.search(query.question, query_embedding, k=5)
    citations.extend(store_citations)

    if not citations:
        citations = seed_answer.citations

    context_str = ""
    if citations:
        context_str = "Grounded context:\n" + "\n".join(
            f"- [{citation.id}] {citation.quote}" for citation in citations
        )

    system_prompt = (
        "Вы опытный ИИ-ассистент геолога. Отвечайте на русском языке. "
        "Если предоставлен контекст (Grounded context или Graph context), опирайтесь на него. "
        "Если контекста нет, используйте свои собственные знания по геологии и нефтегазовому делу."
    )

    graph_context = ""
    from app.db.session import is_db_enabled, get_session
    from app.db.repository import graph_repo
    
    if is_db_enabled():
        try:
            # 1. Extract entities for graph lookup
            extraction_prompt = f"Извлеки из геологического вопроса только ключевые сущности (месторождения, пласты, скважины, свиты). Верни JSON список строк. Вопрос: {query.question}"
            extract_res = await llm_client.chat_completion(
                [{"role": "user", "content": extraction_prompt}],
                response_format={"type": "json_object"},
                max_tokens=200
            )
            from app.services.llm import parse_json_object
            entities_data = parse_json_object(extract_res.content)
            # Assume json like {"entities": [...]} or just list
            entities = entities_data.get("entities", []) if isinstance(entities_data, dict) else entities_data
            
            if entities and isinstance(entities, list):
                with get_session() as session:
                    # Search for these entities in graph
                    found_nodes = []
                    for ent in entities:
                        # Simple case-insensitive search by label
                        from sqlalchemy import select
                        from app.db.models import GraphNodeRecord
                        nodes = session.execute(
                            select(GraphNodeRecord).where(GraphNodeRecord.label.ilike(f"%{ent}%"))
                        ).scalars().all()
                        found_nodes.extend(nodes)
                    
                    if found_nodes:
                        graph_lines = ["Graph context (relationships):"]
                        for node in found_nodes[:5]:
                            # Get related edges
                            from app.db.models import GraphEdgeRecord
                            from sqlalchemy import or_
                            edges = session.execute(
                                select(GraphEdgeRecord).where(
                                    or_(
                                        GraphEdgeRecord.source_node_id == node.id,
                                        GraphEdgeRecord.target_node_id == node.id
                                    )
                                )
                            ).scalars().all()
                            
                            for edge in edges:
                                graph_lines.append(f"- {edge.source_node_id} --({edge.label})--> {edge.target_node_id}")
                        
                        graph_context = "\n".join(graph_lines)
        except Exception as graph_exc:
            print(f"GraphRAG search failed: {graph_exc}")

    user_prompt = f"Question:\n{query.question}\n\n{graph_context}\n\n{context_str}\n"

    if llm_client.configured:
        try:
            completion = await llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )
            return ChatAnswer(
                id=f"a-{uuid.uuid4().hex[:10]}",
                content=completion.content,
                mode="rag",
                citations=citations,
                provider="groq",
                model=completion.model,
                confidence=0.9,
            )
        except Exception as e:
            print(f"Groq error: {e}")
            return seed_answer.model_copy(update={"provider": "groq-error"})

    return ChatAnswer(
        id=f"a-{uuid.uuid4().hex[:10]}",
        content=f"Найдены данные в базе, но Groq не настроен.\n\nКонтекст:\n{context_str}",
        mode="rag",
        citations=citations,
        confidence=0.8,
        provider="local-fallback"
    )


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
