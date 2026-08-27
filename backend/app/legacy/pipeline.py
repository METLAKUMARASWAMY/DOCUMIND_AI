from typing import Any
from uuid import uuid4

from app.core.security import hash_document
from app.legacy.agents import MultiAgentReasoningSystem
from app.legacy.audit import AuditService
from app.legacy.chunking import EvidenceChunkingService
from app.legacy.compliance import ComplianceEngine
from app.legacy.cross_document import CrossDocumentVerificationService
from app.legacy.dataset_case_service import DatasetCaseService
from app.legacy.extraction import KnowledgeExtractionService
from app.legacy.fraud import FraudDetectionEngine
from app.legacy.knowledge_graph import KnowledgeGraphEngine
from app.legacy.ocr import OCRLayoutService
from app.legacy.predictive import PredictiveIntelligenceService
from app.legacy.report import ReportGenerationService
from app.legacy.retrieval import VectorIntelligenceLayer
from app.legacy.scoring import ScoringService
from app.legacy.workflow import WorkflowEngine
from app.legacy.xai import ExplainabilityLayer
from app.models.schemas import (
    AnalysisResponse,
    DocumentRecord,
    Evidence,
    TextAnalysisRequest,
    WorkflowAction,
)
from app.repositories.case_repository import CaseRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.evidence_repository import EvidenceRepository
from app.repositories.finding_repository import FindingRepository
from app.repositories.graph_repository import GraphRepository
from app.services.classification import DocumentClassificationService
from app.services.ingestion import DocumentIngestionService
from app.services.model_registry import ModelRegistryService


class DocumentIntelligencePipeline:
    def __init__(self) -> None:
        self.ingestion = DocumentIngestionService()
        self.ocr = OCRLayoutService()
        self.model_registry = ModelRegistryService()
        self.classifier = DocumentClassificationService()
        self.extractor = KnowledgeExtractionService()
        self.chunking = EvidenceChunkingService()
        self.retrieval = VectorIntelligenceLayer()
        self.compliance = ComplianceEngine()
        self.fraud = FraudDetectionEngine()
        self.cross_document = CrossDocumentVerificationService()
        self.agents = MultiAgentReasoningSystem()
        self.predictive = PredictiveIntelligenceService()
        self.workflow = WorkflowEngine()
        self.knowledge_graph = KnowledgeGraphEngine()
        self.explainability = ExplainabilityLayer()
        self.reports = ReportGenerationService()
        self.scoring = ScoringService()
        self.dataset_cases = DatasetCaseService(reports=self.reports)
        self.audit = AuditService()
        self.case_repository = CaseRepository()
        self.document_repository = DocumentRepository()
        self.evidence_repository = EvidenceRepository()
        self.finding_repository = FindingRepository()
        self.graph_repository = GraphRepository()
        self.cases: dict[str, AnalysisResponse] = self.case_repository._cases

    def process_text(self, request: TextAnalysisRequest) -> AnalysisResponse:
        return self.process_files(
            [(request.filename, request.text.encode("utf-8"), request.metadata)]
        )

    def process_files(self, files: list[tuple[str, bytes, dict]]) -> AnalysisResponse:
        case_id = f"case-{uuid4().hex[:10]}"
        tenant_id = "tenant-local"
        documents: list[DocumentRecord] = []
        self.audit.record(case_id, "case_created", "Case created for uploaded document bundle.")

        for filename, content, metadata in files:
            text, pages, parse_evidence, stored_path = self.ingestion.parse(filename, content)
            doc_type, confidence = self.classifier.classify(text, filename)
            document_id = f"doc-{uuid4().hex[:10]}"
            for evidence in parse_evidence:
                evidence.document_id = document_id
            parse_confidence = parse_evidence[0].confidence if parse_evidence else 0.5
            layout = self.ocr.analyze_layout(document_id, filename, text)
            model_routing = self.model_registry.route_document(
                filename=filename,
                text=text,
                parse_confidence=parse_confidence,
                layout_result=layout,
            )
            document = DocumentRecord(
                document_id=document_id,
                filename=filename,
                document_type=doc_type,
                checksum=hash_document(content),
                text=text,
                pages=pages,
                stored_path=stored_path,
                tenant_id=tenant_id,
                metadata={
                    **metadata,
                    "classification_confidence": confidence,
                    "rvl_cdip_available": self.classifier.rvl_cdip_available,
                    "parse_confidence": parse_confidence,
                    "layout": layout,
                    "model_routing": model_routing,
                },
            )
            document.extracted_fields = self.extractor.extract(document)
            documents.append(document)
            self.audit.record(
                case_id,
                "document_ingested",
                f"Ingested {filename} as {doc_type.value}.",
                metadata={
                    "document_id": document_id,
                    "checksum": document.checksum,
                    "stored_path": stored_path,
                },
            )

        evidence_chunks = self.chunking.chunk_documents(case_id, documents)
        self.document_repository.save_many(documents)
        self.evidence_repository.save_chunks(case_id, evidence_chunks)
        self.retrieval.clear_case(case_id)
        self.retrieval.index_chunks(evidence_chunks)
        self.audit.record(
            case_id, "evidence_indexed", f"Indexed {len(evidence_chunks)} document evidence chunks."
        )

        compliance_score, compliance_findings = self.compliance.evaluate(documents)
        cross_document_checks = self.cross_document.verify(documents)
        fraud_probability, fraud_signals, fraud_confidence = self.fraud.evaluate(
            documents,
            cross_document_checks,
        )

        finding_evidence = self._dedupe_evidence(
            [
                *[field.evidence for document in documents for field in document.extracted_fields],
                *[evidence for finding in compliance_findings for evidence in finding.evidence],
                *[evidence for signal in fraud_signals for evidence in signal.evidence],
                *[evidence for check in cross_document_checks for evidence in check.evidence],
            ]
        )
        self.finding_repository.save_compliance(case_id, compliance_findings)
        self.finding_repository.save_fraud(case_id, fraud_signals)
        self.evidence_repository.save_evidence(case_id, finding_evidence)
        self.retrieval.index_evidence(case_id, finding_evidence)

        fraud_probability_score = self.scoring.fraud_probability_score(fraud_signals)
        fraud_probability = fraud_probability_score / 100
        preliminary_action = self._preliminary_workflow_action(
            compliance_score,
            fraud_probability_score,
            compliance_findings,
            fraud_signals,
        )
        risk_score = self.scoring.risk_score(
            compliance_score, fraud_probability_score, preliminary_action
        )
        agent_decisions = self.agents.decide(
            documents,
            compliance_findings,
            fraud_signals,
            evidence_chunks,
        )
        workflow = self.workflow.assign(
            case_id,
            risk_score,
            fraud_probability,
            compliance_score,
            compliance_findings=compliance_findings,
            fraud_signals=fraud_signals,
            agent_decisions=agent_decisions,
            evidence_count=len(finding_evidence) + len(evidence_chunks),
        )
        if workflow.recommended_action != preliminary_action:
            risk_score = self.scoring.risk_score(
                compliance_score, fraud_probability_score, workflow.recommended_action
            )
            workflow = self.workflow.assign(
                case_id,
                risk_score,
                fraud_probability,
                compliance_score,
                compliance_findings=compliance_findings,
                fraud_signals=fraud_signals,
                agent_decisions=agent_decisions,
                evidence_count=len(finding_evidence) + len(evidence_chunks),
            )
        executive_summary = self.reports.executive_summary(
            len(documents),
            compliance_score,
            risk_score,
            fraud_probability,
            workflow.recommended_action.value,
        )
        graph = self.knowledge_graph.build(
            case_id,
            documents,
            compliance_findings=compliance_findings,
            fraud_signals=fraud_signals,
            cross_document_checks=cross_document_checks,
            agent_decisions=agent_decisions,
            workflow=workflow,
            risk_score=risk_score,
            tenant_id=tenant_id,
        )
        graph["explainability"] = self.explainability.package(
            agent_decisions,
            compliance_findings,
            fraud_signals,
        )
        self.graph_repository.save(case_id, graph)

        self.audit.record(
            case_id,
            "analysis_completed",
            "Completed classification, extraction, retrieval, compliance, fraud, graph, and workflow analysis.",
            metadata={
                "compliance_score": compliance_score,
                "risk_score": risk_score,
                "fraud_probability": fraud_probability,
                "recommended_action": workflow.recommended_action.value,
            },
        )
        self.audit.record(case_id, "report_generated", "Generated case report package.")

        analysis = AnalysisResponse(
            case_id=case_id,
            tenant_id=tenant_id,
            documents=documents,
            evidence_chunks=evidence_chunks,
            executive_summary=executive_summary,
            compliance_score=compliance_score,
            risk_score=risk_score,
            fraud_probability=fraud_probability,
            fraud_confidence=fraud_confidence,
            compliance_findings=compliance_findings,
            fraud_signals=fraud_signals,
            cross_document_checks=cross_document_checks,
            agent_decisions=agent_decisions,
            workflow=workflow,
            report_url=f"/api/v1/reports/{case_id}",
            knowledge_graph=graph,
            graph_nodes=graph.get("nodes", []),
            graph_edges=graph.get("edges", []),
            audit_events=self.audit.list_for_case(case_id),
        )
        analysis.reports = {
            "markdown": self.reports.render_markdown(analysis),
            "json": self.reports.render_payload(analysis),
        }
        analysis.scores = {
            "compliance_score": compliance_score,
            "fraud_probability": fraud_probability,
            "fraud_probability_score": fraud_probability_score,
            "risk_score": risk_score,
            "risk_severity": self.scoring.risk_severity(risk_score),
            "approval_blocked": workflow.approval_blocked,
        }
        self.case_repository.save(analysis)
        return analysis

    def process_demo_case(self, case_id: str | None = None) -> AnalysisResponse:
        analysis = self.dataset_cases.build_case(case_id)
        self.retrieval.clear_case(analysis.case_id)
        self.retrieval.index_chunks(analysis.evidence_chunks)
        self.retrieval.index_evidence(analysis.case_id, self.collect_evidence(analysis))
        self.case_repository.save(analysis)
        return analysis

    def dataset_case_summaries(self) -> list[dict[str, Any]]:
        return self.dataset_cases.list_cases()

    def dashboard_metrics(self) -> dict:
        cases = list(self.cases.values())
        if not cases:
            return {
                "documents_processed": 0,
                "open_cases": 0,
                "average_compliance_score": 0,
                "high_risk_cases": 0,
                "fraud_alerts": 0,
                "sla_breach_probability": 0,
                "evidence_count": 0,
                "open_findings": 0,
                "approval_blocked_cases": 0,
                "average_risk_score": 0,
            }
        return {
            "documents_processed": sum(len(case.documents) for case in cases),
            "open_cases": len(cases),
            "average_compliance_score": round(
                sum(case.compliance_score for case in cases) / len(cases)
            ),
            "high_risk_cases": sum(1 for case in cases if case.risk_score >= 60),
            "fraud_alerts": sum(len(case.fraud_signals) for case in cases),
            "sla_breach_probability": self.predictive.predict_approval_delay(
                max(case.risk_score for case in cases),
                sum(len(case.compliance_findings) for case in cases),
            ),
            "evidence_count": sum(
                len(case.evidence_chunks) + len(self.collect_evidence(case)) for case in cases
            ),
            "open_findings": sum(
                len(case.compliance_findings) + len(case.fraud_signals) for case in cases
            ),
            "approval_blocked_cases": sum(1 for case in cases if case.workflow.approval_blocked),
            "average_risk_score": round(sum(case.risk_score for case in cases) / len(cases)),
        }

    def case_summary(self, analysis: AnalysisResponse) -> dict[str, Any]:
        return {
            "case_id": analysis.case_id,
            "document_count": len(analysis.documents),
            "compliance_score": analysis.compliance_score,
            "risk_score": analysis.risk_score,
            "fraud_probability": analysis.fraud_probability,
            "workflow_status": analysis.workflow.status,
            "assigned_team": analysis.workflow.assigned_team,
            "recommended_action": analysis.workflow.recommended_action,
            "approval_blocked": analysis.workflow.approval_blocked,
            "report_url": analysis.report_url,
        }

    def collect_evidence(self, analysis: AnalysisResponse) -> list[Evidence]:
        evidence: list[Evidence] = []
        for document in analysis.documents:
            layout_evidence = document.metadata.get("layout", {}).get("evidence")
            if layout_evidence:
                evidence.append(Evidence(**layout_evidence))
            evidence.extend(field.evidence for field in document.extracted_fields)
        for finding in analysis.compliance_findings:
            evidence.extend(finding.evidence)
        for signal in analysis.fraud_signals:
            evidence.extend(signal.evidence)
        for check in analysis.cross_document_checks:
            evidence.extend(check.evidence)
        for decision in analysis.agent_decisions:
            evidence.extend(decision.evidence)
        return self._dedupe_evidence(evidence)

    def _dedupe_evidence(self, evidence: list[Evidence]) -> list[Evidence]:
        seen = set()
        result = []
        by_key: dict[tuple, Evidence] = {}
        for item in evidence:
            key = (item.document_id, item.source_document, item.page_number, item.text)
            existing = by_key.get(key)
            if existing:
                merged_findings = sorted(set(existing.linked_findings + item.linked_findings))
                if merged_findings:
                    existing.linked_findings = merged_findings
                if item.reasoning and (
                    not existing.reasoning
                    or existing.reasoning == "Dataset evidence chunk."
                    or len(item.reasoning) > len(existing.reasoning)
                ):
                    existing.reasoning = item.reasoning
                existing.confidence = max(existing.confidence, item.confidence)
                if item.extraction_method != "chunking":
                    existing.extraction_method = item.extraction_method
                continue
            seen.add(key)
            copied = item.model_copy(deep=True)
            by_key[key] = copied
            result.append(copied)
        return result

    def _preliminary_workflow_action(
        self,
        compliance_score: int,
        fraud_probability_score: int,
        compliance_findings: list,
        fraud_signals: list,
    ) -> WorkflowAction:
        if fraud_probability_score >= 60 or any(
            signal.severity.value in {"high", "critical"} for signal in fraud_signals
        ):
            return WorkflowAction.hold_payment
        if any(finding.severity.value == "critical" for finding in compliance_findings):
            return WorkflowAction.escalate_to_compliance
        if (
            any(finding.approval_blocking for finding in compliance_findings)
            or compliance_score < 85
        ):
            return WorkflowAction.request_changes
        return WorkflowAction.approve


pipeline = DocumentIntelligencePipeline()
