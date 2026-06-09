from app.domain.models import MetricValue, MetricsReport
from app.services.bm25_index import bm25_index
from app.services.graph_store import graph_store
from app.services.seed import METRICS
from app.services.store import store
from app.services.vector_store import vector_store


def build_metrics_report() -> MetricsReport:
    chunk_count = len(vector_store.documents)
    graph = graph_store.get_graph()
    completed_docs = sum(1 for doc in store.list_documents() if doc.status == "completed")

    return MetricsReport(
        parser=[
            MetricValue(
                name="indexed_chunks",
                value=float(chunk_count),
                unit="count",
                description="Chunks in vector + BM25 indexes.",
            ),
            *METRICS.parser,
        ],
        retrieval=[
            MetricValue(
                name="bm25_documents",
                value=float(len(bm25_index.documents)),
                unit="count",
                description="Documents in BM25 lexical index.",
            ),
            *METRICS.retrieval,
        ],
        qa=[
            MetricValue(
                name="completed_documents",
                value=float(completed_docs),
                unit="count",
                description="Documents with completed pipeline status.",
            ),
            *METRICS.qa,
        ],
        latency=[
            MetricValue(
                name="graph_nodes",
                value=float(len(graph.nodes)),
                unit="count",
                description="Nodes in knowledge graph index.",
            ),
            MetricValue(
                name="graph_edges",
                value=float(len(graph.edges)),
                unit="count",
                description="Edges in knowledge graph index.",
            ),
            *METRICS.latency,
        ],
    )
