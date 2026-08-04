from app.core.config import settings


class PhaseReadinessService:
    def mlops(self) -> dict:
        return {
            "status": "ready_for_local_evaluation",
            "llm_provider_enabled": settings.llm_enabled,
            "tracked_stages": [
                "model_registry",
                "selective_model_routing",
                "ocr_layout",
                "classification",
                "extraction",
                "retrieval",
                "rag",
                "agents",
                "compliance",
                "fraud",
                "cross_document",
                "xai",
                "prediction",
            ],
            "evaluation_gates": [
                "classification_f1",
                "field_extraction_f1",
                "retrieval_ndcg",
                "citation_precision",
                "fraud_precision",
                "explanation_completeness",
            ],
            "registry_plan": "MLflow-compatible model, retriever, prompt, and rule-pack registry.",
        }

    def security(self) -> dict:
        return {
            "status": "local_controls_enabled",
            "controls": [
                "document_sha256_hashing",
                "constant_time_secret_compare",
                "pii_redaction_helper",
                "cors_allowlist",
                "kubernetes_secret_example",
                "rbac_abac_schema_design",
                "audit_log_schema_design",
            ],
            "production_gate": "Connect OIDC/SAML, KMS, TLS ingress, RLS, SIEM, and external secrets.",
        }

    def deployment(self) -> dict:
        return {
            "status": "container_and_kubernetes_ready",
            "docker": [
                "backend",
                "worker",
                "frontend",
                "postgres",
                "mongodb",
                "redis",
                "qdrant",
                "minio",
                "chroma",
                "neo4j",
            ],
            "kubernetes": [
                "namespace",
                "backend",
                "frontend",
                "postgres",
                "mongodb_target",
                "redis_target",
                "qdrant_target",
                "minio_target",
                "chroma",
                "neo4j",
                "ingress",
                "hpa",
                "configmap",
                "secret_example",
            ],
            "worker_target": "Docker Compose includes a Celery worker using the VisionProcessingWorker contract; local FastAPI background tasks remain the default outside ENABLE_REDIS=true.",
            "vision_infrastructure_endpoint": "/api/v1/vision/infrastructure/readiness",
            "cloud_targets": ["aws_eks", "azure_aks", "gcp_gke"],
            "local_runtime_note": "Vision AI defaults to local in-memory repositories and FastAPI background tasks until external services are enabled.",
        }

    def testing(self) -> dict:
        return {
            "status": "regression_path_defined",
            "commands": [
                "cd backend && python -m pytest",
                "cd backend && ruff check app",
                "cd frontend && npm run lint",
                "cd frontend && npm run build",
            ],
            "coverage_areas": [
                "pipeline",
                "cross_document_verification",
                "api_contracts",
                "frontend_build",
                "ci_container_build",
                "kubernetes_dry_run",
            ],
        }

    def research(self) -> dict:
        return {
            "status": "paper_scaffold_ready",
            "claims": [
                "multi_document_document_intelligence",
                "citation_grounded_rag",
                "multi_agent_case_review",
                "graph_enhanced_fraud_and_compliance_reasoning",
                "explainable_evidence_packaging",
            ],
            "metrics": [
                "macro_f1",
                "extraction_f1",
                "retrieval_ndcg",
                "citation_precision",
                "reviewer_acceptance_rate",
                "calibration_error",
            ],
        }

    def portfolio(self) -> dict:
        return {
            "status": "portfolio_ready",
            "artifacts": [
                "README.md",
                "docs/RESUME_DESCRIPTION.md",
                "docs/GITHUB_PORTFOLIO_SETUP.md",
                "docs/RESEARCH_PAPER_STRUCTURE.md",
                "docs/phases/PHASE_04_TO_29_COMPLETION.md",
            ],
            "phase_range": "1-29 complete",
            "positioning": "Enterprise AI document intelligence scaffold with runnable local intelligence.",
        }
