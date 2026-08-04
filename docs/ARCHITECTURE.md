# System Architecture

The formal DocuMind AI 2.0 Phase 2 architecture package is maintained at:

- [Phase 2 - System Architecture](phases/PHASE_02_SYSTEM_ARCHITECTURE.md)

## Logical Architecture

```text
React Analyst Console
        |
Nginx / API Gateway
        |
FastAPI Backend
        |
Document Intelligence Pipeline
        |
+-------------------+-------------------+------------------+
| Ingestion/OCR     | AI Reasoning      | Enterprise Layer |
| PDF/DOCX/Image    | RAG               | Workflow         |
| Layout Analysis   | Multi-Agent       | Reports          |
| Classification    | Compliance        | Audit Logs       |
| Extraction        | Fraud Detection   | RBAC             |
+-------------------+-------------------+------------------+
        |
+-------------------+-------------------+------------------+
| PostgreSQL        | Chroma/pgvector   | Neo4j            |
| metadata/audit    | retrieval index   | knowledge graph  |
+-------------------+-------------------+------------------+
```

## Runtime Flow

1. Upload documents.
2. Parse and OCR content.
3. Classify each document.
4. Extract structured entities and clauses.
5. Index chunks for retrieval.
6. Run compliance, fraud, and cross-document checks.
7. Run domain agents.
8. Build explainability package and knowledge graph.
9. Generate reports and workflow status.
10. Persist metadata, evidence, vectors, graph relationships, and audit logs.

## Deployment Architecture

- Frontend: React static build served by Nginx.
- API: FastAPI with Uvicorn workers.
- Workers: future Celery/RQ/Kafka consumers for OCR, embeddings, agents, reports.
- PostgreSQL: primary metadata store.
- ChromaDB or pgvector: vector retrieval layer.
- Neo4j: relationship and case graph.
- Object storage: original files, OCR artifacts, signed report exports.
- Observability: OpenTelemetry traces, Prometheus metrics, structured logs.

## Extension Points

- `OCRLayoutService`: PaddleOCR, EasyOCR, Tesseract, LayoutLMv3, DocFormer.
- `VectorIntelligenceLayer`: ChromaDB, FAISS, pgvector, rerankers.
- `RAGEngine`: Llama 3, Gemini, OpenAI-compatible models.
- `MultiAgentReasoningSystem`: LangGraph, CrewAI, AutoGen.
- `KnowledgeGraphEngine`: Neo4j persistence and entity resolution.
