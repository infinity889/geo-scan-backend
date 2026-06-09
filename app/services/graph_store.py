import json
from pathlib import Path

from app.core.config import settings
from app.domain.models import Citation, GraphEdge, GraphNode, KnowledgeGraph
from app.services.seed import CITATIONS, GRAPH_EDGES, GRAPH_NODES


class GraphStore:
    def __init__(self) -> None:
        self.db_path = settings.upload_dir / "knowledge_graph.json"
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []
        self._load()

    def _load(self) -> None:
        if self.db_path.exists():
            try:
                with open(self.db_path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.nodes = [GraphNode(**item) for item in payload.get("nodes", [])]
                self.edges = [GraphEdge(**item) for item in payload.get("edges", [])]
                return
            except Exception:
                pass
        self.nodes = list(GRAPH_NODES)
        self.edges = list(GRAPH_EDGES)
        self._save()

    def _save(self) -> None:
        payload = {
            "nodes": [node.model_dump() for node in self.nodes],
            "edges": [edge.model_dump() for edge in self.edges],
        }
        with open(self.db_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    def get_graph(self) -> KnowledgeGraph:
        return KnowledgeGraph(nodes=self.nodes, edges=self.edges)

    def add_from_text(self, document_id: str, text: str) -> None:
        lowered = text.lower()
        node_ids = {node.id for node in self.nodes}

        def ensure_node(node_id: str, label: str, node_type: str) -> None:
            if node_id not in node_ids:
                self.nodes.append(GraphNode(id=node_id, label=label, type=node_type))
                node_ids.add(node_id)

        if "247" in text:
            ensure_node(f"well-247-{document_id[:6]}", "Well 247", "Well")
        if "248" in text:
            ensure_node(f"well-248-{document_id[:6]}", "Well 248", "Well")
        if "бс10" in lowered or "bs10" in lowered:
            ensure_node(f"layer-bs10-{document_id[:6]}", "Layer BS10", "Layer")
        if "бажен" in lowered or "bazhenov" in lowered:
            ensure_node(f"formation-bazhenov-{document_id[:6]}", "Bazhenov Formation", "Formation")
        if "керн" in lowered or "core" in lowered:
            ensure_node(f"study-core-{document_id[:6]}", "Core Studies", "Study")
        self._save()

    def search_citations(self, query_text: str, k: int = 5) -> list[Citation]:
        lowered = query_text.lower()
        citation_ids: list[str] = []

        graph_terms = ["бажен", "bazhenov", "керн", "core", "скважин", "well", "multi-hop", "граф"]
        if any(term in lowered for term in graph_terms):
            citation_ids.extend(["Report_2015:p.18", "Report_2015:p.31"])

        if any(term in lowered for term in ["bs10", "бс10", "нефтенасыщ", "oil saturation", "247"]):
            citation_ids.extend(["Report_2015:p.24", "Report_2015:p.27", "Well_247_Core_Analysis:p.6"])

        if any(term in lowered for term in ["таблиц", "table", "скан", "scan", "ocr", "1987", "appendix"]):
            citation_ids.append("Report_2015:p.14")

        seen: set[str] = set()
        citations: list[Citation] = []
        for citation_id in citation_ids:
            if citation_id in seen:
                continue
            seen.add(citation_id)
            citation = CITATIONS.get(citation_id)
            if citation is not None:
                citations.append(citation.model_copy(update={"score": min(1.0, citation.score + 0.02)}))
            if len(citations) >= k:
                break

        if not citations:
            for citation in CITATIONS.values():
                if any(token in citation.quote.lower() for token in _tokenize(lowered)[:6]):
                    citations.append(citation)
                if len(citations) >= k:
                    break
        return citations[:k]


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())


graph_store = GraphStore()
