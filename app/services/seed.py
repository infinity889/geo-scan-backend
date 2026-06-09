from app.domain.models import (
    Citation,
    FigureSummary,
    GraphEdge,
    GraphNode,
    MetricValue,
    MetricsReport,
    PipelineStep,
    Scenario,
    SourcePreview,
)


def initial_steps() -> list[PipelineStep]:
    return [
        PipelineStep(
            id=1,
            title="File Format Detection",
            detail="PDF / Scan / DOCX / DJVU / Image",
        ),
        PipelineStep(
            id=2,
            title="Layout-aware OCR",
            detail="Multi-column, tables, formulas, captions",
        ),
        PipelineStep(
            id=3,
            title="Domain NER Extraction",
            detail="Fields, wells, layers, horizons, measurements",
        ),
        PipelineStep(
            id=4,
            title="KG & Hybrid Indexing",
            detail="Metadata, vectors, BM25, knowledge graph",
        ),
    ]


STEP_LOGS: dict[int, list[str]] = {
    1: [
        "$ detect --magic-bytes",
        "container: application/pdf; pages=42",
        "encoding: latin-1; scanned=true",
    ],
    2: [
        "$ ocr.layout --engine deepseek-ocr-2 --dpi 300",
        "regions: 318 (text=271, table=29, figure=18)",
        "formulas: 11 captured; latex restored",
        "confidence: 0.967",
    ],
    3: [
        "$ ner.geo --ontology kmg-v1",
        "wells: 247, 248, 251A; field: Priobskoye",
        "layers: BS10, BS11, AS4; formation: Bazhenov",
        "measurements: phi=18.2%; Kp=14.7 mD; So=0.62",
    ],
    4: [
        "$ index --hybrid vector+bm25+graph",
        "chunks: 271; vector dimensions: 1024",
        "graph: 84 nodes, 142 edges",
        "commit ok; doc_id=Report_2015",
    ],
}


SCENARIOS = [
    Scenario(
        key="s1",
        label="Scenario 1 - Fact Retrieval",
        question="Какая нефтенасыщенность пласта БС10 в скважине №247?",
        mode="rag",
    ),
    Scenario(
        key="s2",
        label="Scenario 2 - GraphRAG",
        question=(
            "В каких скважинах Приобского месторождения, вскрывших "
            "баженовскую свиту, проводились исследования керна?"
        ),
        mode="graph",
    ),
    Scenario(
        key="s3",
        label="Scenario 3 - Soviet Scan Table OCR",
        question="Восстанови петрофизическую таблицу из скана отчета 1987 года.",
        mode="table",
    ),
]


CITATIONS: dict[str, Citation] = {
    "Report_2015:p.24": Citation(
        id="Report_2015:p.24",
        label="Report_2015:p.24",
        document_id="doc-1",
        document_name="Priobskoye_Report_2015.pdf",
        page="24",
        chunk_id="chunk-2015-024-04",
        quote="Oil saturation of layer BS10 in well No. 247 averages 0.62 +/- 0.04.",
        score=0.96,
    ),
    "Report_2015:p.27": Citation(
        id="Report_2015:p.27",
        label="Report_2015:p.27",
        document_id="doc-1",
        document_name="Priobskoye_Report_2015.pdf",
        page="27",
        chunk_id="chunk-2015-027-02",
        quote="Effective porosity is 18.2 percent; permeability Kp is 14.7 mD.",
        score=0.92,
    ),
    "Well_247_Core_Analysis:p.6": Citation(
        id="Well_247_Core_Analysis:p.6",
        label="Well_247_Core_Analysis:p.6",
        document_id="doc-2",
        document_name="Well_247_Core_Analysis.docx",
        page="6",
        chunk_id="chunk-core-006-01",
        quote="Adjacent wells 248 and 251A show comparable saturation values.",
        score=0.88,
    ),
    "Report_2015:p.18": Citation(
        id="Report_2015:p.18",
        label="Report_2015:p.18",
        document_id="doc-1",
        document_name="Priobskoye_Report_2015.pdf",
        page="18",
        chunk_id="chunk-2015-018-03",
        quote="Wells 247, 248, and 251A penetrate the Bazhenov formation.",
        score=0.9,
    ),
    "Report_2015:p.31": Citation(
        id="Report_2015:p.31",
        label="Report_2015:p.31",
        document_id="doc-1",
        document_name="Priobskoye_Report_2015.pdf",
        page="31",
        chunk_id="chunk-2015-031-02",
        quote="Core research was performed for wells 247, 248, and 251A.",
        score=0.91,
    ),
    "Report_2015:p.14": Citation(
        id="Report_2015:p.14",
        label="Report_2015:p.14",
        document_id="doc-1",
        document_name="Priobskoye_Report_2015.pdf",
        page="14",
        chunk_id="chunk-2015-014-table-01",
        quote="Appendix B contains a reconstructed petrophysical table for well 247 samples.",
        score=0.94,
    ),
}


SOURCE_PREVIEWS: dict[str, SourcePreview] = {
    "Report_2015:p.24": SourcePreview(
        citation_id="Report_2015:p.24",
        document_id="doc-1",
        document_name="Priobskoye_Report_2015.pdf",
        page="24",
        title="Section 4.2 Petrophysical characterization - Layer BS10",
        text=(
            "The Bazhenov-equivalent interval encountered in Well No. 247 of the "
            "Priobskoye field exhibits stable reservoir properties across the BS10 unit."
        ),
        highlighted_text=(
            "Oil saturation (So) of layer BS10 in well No. 247 averages "
            "0.62 +/- 0.04 (n=18), with effective porosity phi=18.2% and "
            "permeability Kp=14.7 mD."
        ),
        tables=[
            {
                "title": "Extracted petrophysical samples",
                "columns": ["Sample", "Porosity total", "Porosity eff.", "So"],
                "rows": [
                    ["247-01", "19.4", "18.1", "0.61"],
                    ["247-02", "20.1", "18.7", "0.64"],
                    ["247-03", "18.8", "17.6", "0.58"],
                    ["247-04", "19.7", "18.4", "0.63"],
                ],
            }
        ],
        formulas=[
            {
                "id": "eq-4.2",
                "title": "Oil saturation",
                "latex": "S_o = 1 - (R_w / R_t)^{1/n} * (1 / phi^m)",
            }
        ],
    )
}


GRAPH_NODES = [
    GraphNode(id="field-priobskoye", label="Field Priobskoye", type="Field"),
    GraphNode(id="formation-bazhenov", label="Bazhenov Formation", type="Formation"),
    GraphNode(id="well-247", label="Well 247", type="Well"),
    GraphNode(id="well-248", label="Well 248", type="Well"),
    GraphNode(id="well-251a", label="Well 251A", type="Well"),
    GraphNode(id="layer-bs10", label="Layer BS10", type="Layer"),
    GraphNode(id="layer-bs11", label="Layer BS11", type="Layer"),
    GraphNode(id="study-core", label="Core Studies", type="Study"),
    GraphNode(id="measurement-so", label="So=0.62", type="Measurement"),
    GraphNode(id="measurement-phi", label="phi=18.2%", type="Measurement"),
]


GRAPH_EDGES = [
    GraphEdge(id="e1", source="field-priobskoye", target="well-247", label="contains"),
    GraphEdge(id="e2", source="field-priobskoye", target="well-248", label="contains"),
    GraphEdge(id="e3", source="field-priobskoye", target="well-251a", label="contains"),
    GraphEdge(id="e4", source="well-247", target="formation-bazhenov", label="penetrates"),
    GraphEdge(id="e5", source="well-248", target="formation-bazhenov", label="penetrates"),
    GraphEdge(id="e6", source="well-251a", target="formation-bazhenov", label="penetrates"),
    GraphEdge(id="e7", source="well-247", target="layer-bs10", label="intersects"),
    GraphEdge(id="e8", source="well-248", target="layer-bs11", label="intersects"),
    GraphEdge(id="e9", source="well-247", target="study-core", label="has_study"),
    GraphEdge(id="e10", source="well-248", target="study-core", label="has_study"),
    GraphEdge(id="e11", source="well-251a", target="study-core", label="has_study"),
    GraphEdge(id="e12", source="layer-bs10", target="measurement-so", label="measured"),
    GraphEdge(id="e13", source="layer-bs10", target="measurement-phi", label="porosity"),
]


FIGURES = [
    FigureSummary(
        id="fig-1",
        title="Fig. 4.1 - Structural map, top of BS10",
        type="structural_map",
        document_id="doc-1",
        page="22",
        description=(
            "Structural contour map of the top of layer BS10 across the Priobskoye "
            "field with wells 247, 248, and 251A projected."
        ),
        citations=["Report_2015:p.22"],
    ),
    FigureSummary(
        id="fig-2",
        title="Fig. 4.7 - Composite log, Well 247",
        type="well_log",
        document_id="doc-1",
        page="25",
        description=(
            "Composite well log for Well 247 over interval 2612-2684 m with BS10 "
            "highlighted at 2634-2651 m."
        ),
        citations=["Report_2015:p.25"],
    ),
    FigureSummary(
        id="fig-3",
        title="Fig. 5.2 - W-E cross-section through wells 247-248-251A",
        type="cross_section",
        document_id="doc-1",
        page="36",
        description=(
            "Stratigraphic cross-section showing lateral continuity of BS10 and "
            "pinch-out of BS11 east of well 251A."
        ),
        citations=["Report_2015:p.36"],
    ),
]


METRICS = MetricsReport(
    parser=[
        MetricValue(
            name="table_reconstruction_accuracy",
            value=0.91,
            unit="ratio",
            description="Share of table cells reconstructed correctly on the demo set.",
        ),
        MetricValue(
            name="ocr_confidence",
            value=0.967,
            unit="ratio",
            description="Average OCR confidence on seeded scanned pages.",
        ),
    ],
    retrieval=[
        MetricValue(
            name="hybrid_recall_at_5",
            value=0.86,
            unit="ratio",
            description="Share of expected supporting chunks retrieved in top 5.",
        )
    ],
    qa=[
        MetricValue(
            name="citation_accuracy",
            value=0.94,
            unit="ratio",
            description="Share of answer citations pointing to supporting fragments.",
        ),
        MetricValue(
            name="faithfulness",
            value=0.9,
            unit="ratio",
            description="Open-weight judge estimate for grounded factual consistency.",
        ),
    ],
    latency=[
        MetricValue(
            name="answer_latency",
            value=1.8,
            unit="seconds",
            description="Average answer time for seeded demo questions.",
        )
    ],
)
