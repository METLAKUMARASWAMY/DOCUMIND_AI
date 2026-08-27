from pathlib import Path
from typing import Any

from app.core.security import hash_document, utc_now_iso
from app.legacy.dataset_loader import DatasetLoader, dataset_loader
from app.legacy.report import ReportGenerationService
from app.legacy.scoring import ScoringService
from app.models.schemas import (
    AgentDecision,
    AnalysisResponse,
    AuditEvent,
    ComplianceFinding,
    CrossDocumentCheck,
    DocumentRecord,
    DocumentType,
    Evidence,
    EvidenceChunk,
    ExtractedField,
    FraudSignal,
    RiskLevel,
    WorkflowAction,
    WorkflowState,
)


class DatasetCaseService:
    def __init__(
        self,
        loader: DatasetLoader | None = None,
        scoring: ScoringService | None = None,
        reports: ReportGenerationService | None = None,
    ) -> None:
        self.loader = loader or dataset_loader
        self.scoring = scoring or ScoringService()
        self.reports = reports or ReportGenerationService()

    def list_cases(self) -> list[dict[str, Any]]:
        return [
            {
                **row,
                "approval_blocked": self._bool(row.get("approval_blocked")),
                "documents_count": self._int(row.get("documents_count")),
            }
            for row in self.loader.load_all_cases()
        ]

    def build_case(self, case_id: str | None = None) -> AnalysisResponse:
        selected_case_id = case_id or self._first_case_id()
        case_row = self.loader.load_case(selected_case_id)
        if not case_row:
            raise ValueError(f"case_id not found in dataset: {selected_case_id}")

        evidence_chunks = self._evidence_chunks(selected_case_id)
        evidence_by_id = {
            chunk.chunk_id: self._evidence_from_chunk(chunk) for chunk in evidence_chunks
        }
        documents = self._documents(selected_case_id, evidence_by_id)
        entities = self._entities(selected_case_id)
        self._attach_entities_to_documents(documents, entities, evidence_by_id)
        compliance_findings = self._compliance_findings(selected_case_id, evidence_by_id)
        fraud_signals = self._fraud_signals(selected_case_id, evidence_by_id)
        cross_checks = self._cross_checks(selected_case_id, evidence_by_id)
        cross_checks.extend(self._derived_bank_change_checks(selected_case_id, fraud_signals))

        workflow_row = self.loader.load_case_workflow(selected_case_id) or {}
        workflow_action = self._workflow_action(
            workflow_row.get("recommended_action") or case_row.get("recommended_workflow_action")
        )
        compliance_score = self.scoring.compliance_score(compliance_findings)
        fraud_probability_score = self.scoring.fraud_probability_score(fraud_signals)
        risk_score = self.scoring.risk_score(
            compliance_score, fraud_probability_score, workflow_action
        )
        approval_blocked, blocked_reasons = self.scoring.approval_blocked(
            compliance_findings,
            fraud_probability_score,
            risk_score,
            workflow_action,
        )
        workflow = self._workflow(
            selected_case_id,
            workflow_row,
            workflow_action,
            approval_blocked,
            blocked_reasons,
            risk_score,
        )
        agents = self._agents(
            selected_case_id, compliance_findings, fraud_signals, workflow, evidence_by_id
        )
        graph = self._graph(
            selected_case_id,
            documents,
            entities,
            evidence_chunks,
            compliance_findings,
            fraud_signals,
            workflow,
            risk_score,
        )
        executive_summary = self.reports.executive_summary(
            len(documents),
            compliance_score,
            risk_score,
            fraud_probability_score / 100,
            workflow.recommended_action.value,
        )
        audit_events = self._audit_events(
            selected_case_id, documents, compliance_findings, fraud_signals, graph
        )
        analysis = AnalysisResponse(
            case_id=selected_case_id,
            tenant_id="tenant-local",
            industry=case_row.get("industry"),
            company=case_row.get("company"),
            vendor=case_row.get("vendor"),
            documents=documents,
            entities=entities,
            evidence_chunks=evidence_chunks,
            executive_summary=executive_summary,
            compliance_score=compliance_score,
            risk_score=risk_score,
            fraud_probability=fraud_probability_score / 100,
            fraud_confidence="high" if fraud_signals else "low",
            compliance_findings=compliance_findings,
            fraud_signals=fraud_signals,
            cross_document_checks=cross_checks,
            agent_decisions=agents,
            workflow=workflow,
            report_url=f"/api/v1/reports/{selected_case_id}",
            knowledge_graph=graph,
            graph_nodes=graph["nodes"],
            graph_edges=graph["edges"],
            audit_events=audit_events,
        )
        analysis.reports = {
            "markdown": self.reports.render_markdown(analysis),
            "json": self.reports.render_payload(analysis),
        }
        analysis.scores = {
            "compliance_score": compliance_score,
            "fraud_probability": fraud_probability_score / 100,
            "fraud_probability_score": fraud_probability_score,
            "risk_score": risk_score,
            "risk_severity": self.scoring.risk_severity(risk_score),
            "approval_blocked": approval_blocked,
        }
        return analysis

    def _first_case_id(self) -> str:
        cases = self.loader.load_all_cases()
        if not cases:
            raise ValueError("Dataset contains no cases")
        return cases[0]["case_id"]

    def _documents(self, case_id: str, evidence_by_id: dict[str, Evidence]) -> list[DocumentRecord]:
        documents = []
        for row in self.loader.load_case_documents(case_id):
            text = row.get("text") or ""
            document_id = row["document_id"]
            documents.append(
                DocumentRecord(
                    document_id=document_id,
                    filename=Path(row["filename"]).name,
                    document_type=self._document_type(row.get("document_type")),
                    checksum=hash_document(text.encode("utf-8")),
                    text=text,
                    pages=self._int(row.get("pages"), 1),
                    stored_path=row.get("path"),
                    tenant_id="tenant-local",
                    metadata={
                        "dataset_filename": row["filename"],
                        "parser": row.get("parser"),
                        "ocr_required": self._bool(row.get("ocr_required")),
                        "classification_confidence": self._float(row.get("confidence"), 0.9),
                    },
                )
            )
        return documents

    def _attach_entities_to_documents(
        self,
        documents: list[DocumentRecord],
        entities: list[dict[str, Any]],
        evidence_by_id: dict[str, Evidence],
    ) -> None:
        doc_map = {document.document_id: document for document in documents}
        evidence_for_doc = self._first_evidence_by_doc(evidence_by_id.values())
        for entity in entities:
            if self._is_invalid_entity(entity):
                continue
            document = doc_map.get(entity["source_document_id"])
            if not document:
                continue
            evidence = evidence_for_doc.get(document.document_id) or Evidence(
                evidence_id=f"ev-{document.document_id}-entity",
                document_id=document.document_id,
                source_document=document.filename,
                page_number=1,
                text=document.text[:420],
                confidence=self._float(entity.get("confidence"), 0.8),
                extraction_method="dataset_entity",
                reasoning=f"Dataset label for {entity['entity_type']}.",
            )
            document.extracted_fields.append(
                ExtractedField(
                    name=entity["entity_type"],
                    value=entity["entity_value"],
                    normalized_value=self._normalize(entity["entity_value"]),
                    confidence=self._float(entity.get("confidence"), 0.8),
                    extraction_method="dataset_label",
                    evidence=evidence,
                )
            )

    def _entities(self, case_id: str) -> list[dict[str, Any]]:
        result = []
        missing_indemnity = any(
            "missing indemnity" in row.get("finding", "").lower()
            for row in self.loader.load_case_compliance_findings(case_id)
        )
        for index, row in enumerate(self.loader.load_case_entities(case_id), start=1):
            if missing_indemnity and row.get("entity_type") == "indemnity_clause":
                continue
            if self._is_invalid_entity(row):
                continue
            result.append(
                {
                    "entity_id": f"ent-{case_id}-{index:03d}",
                    "case_id": row["case_id"],
                    "entity_type": row["entity_type"],
                    "entity_value": row["entity_value"],
                    "source_document_id": row["source_document_id"],
                    "confidence": self._float(row.get("confidence"), 0.8),
                }
            )
        return result

    def _evidence_chunks(self, case_id: str) -> list[EvidenceChunk]:
        chunks = []
        for row in self.loader.load_case_evidence(case_id):
            chunks.append(
                EvidenceChunk(
                    chunk_id=row["evidence_id"],
                    case_id=case_id,
                    document_id=row["document_id"],
                    source_document=self._filename_for_document(case_id, row["document_id"]),
                    page_number=self._int(row.get("page_number"), 1),
                    text=row["chunk_text"],
                    extraction_method=f"dataset_{row.get('document_type', 'evidence')}",
                    confidence=self._float(row.get("confidence"), 0.75),
                )
            )
        return chunks

    def _compliance_findings(
        self,
        case_id: str,
        evidence_by_id: dict[str, Evidence],
    ) -> list[ComplianceFinding]:
        findings = []
        for row in self.loader.load_case_compliance_findings(case_id):
            evidence = evidence_by_id.get(row["evidence_id"])
            if evidence:
                evidence = evidence.model_copy(
                    update={
                        "reasoning": self._finding_reason(row["finding"]),
                        "linked_findings": [row["finding_id"]],
                    }
                )
            findings.append(
                ComplianceFinding(
                    finding_id=row["finding_id"],
                    title=row["finding"],
                    severity=self._risk_level(row.get("severity")),
                    framework=row.get("framework") or "custom",
                    clause_or_policy=self._clause_or_policy(row["finding"]),
                    evidence=[evidence] if evidence else [],
                    recommendation=row.get("recommendation") or "Review finding.",
                    approval_blocking=self._bool(row.get("approval_blocking")),
                    score_impact=0,
                )
            )
        return findings

    def _fraud_signals(
        self, case_id: str, evidence_by_id: dict[str, Evidence]
    ) -> list[FraudSignal]:
        signals = []
        for row in self.loader.load_case_fraud_findings(case_id):
            title = row["finding"]
            evidence = evidence_by_id.get(row["evidence_id"])
            finding_id = row["finding_id"]
            if evidence:
                evidence = evidence.model_copy(
                    update={
                        "reasoning": self._fraud_reason(title),
                        "linked_findings": [finding_id],
                    }
                )
            score_impact = self._fraud_signal_score(title)
            signals.append(
                FraudSignal(
                    signal_id=finding_id,
                    title=title,
                    severity=self._risk_level(row.get("severity")),
                    probability=min(0.95, score_impact / 100),
                    evidence=[evidence] if evidence else [],
                    recommendation=row.get("recommendation") or "Review fraud signal.",
                    score_impact=score_impact,
                )
            )
        return signals

    def _cross_checks(
        self,
        case_id: str,
        evidence_by_id: dict[str, Evidence],
    ) -> list[CrossDocumentCheck]:
        evidence_by_doc = self._first_evidence_by_doc(evidence_by_id.values())
        checks = []
        for index, row in enumerate(self.loader.load_case_cross_checks(case_id), start=1):
            status = "Passed" if row["result"] == "pass" else "Failed"
            left_evidence = evidence_by_doc.get(row["left_document_id"])
            right_evidence = evidence_by_doc.get(row["right_document_id"])
            checks.append(
                CrossDocumentCheck(
                    check_id=f"xdoc-{case_id}-{index:03d}",
                    title=self._humanize_check(row["check_name"]),
                    status=status,
                    confidence=0.92 if status == "Passed" else 0.78,
                    evidence=[item for item in [left_evidence, right_evidence] if item],
                    left_value=row.get("left_value") or None,
                    right_value=row.get("right_value") or None,
                    recommendation=None if status == "Passed" else "Review Required",
                )
            )
        return checks

    def _derived_bank_change_checks(
        self,
        case_id: str,
        fraud_signals: list[FraudSignal],
    ) -> list[CrossDocumentCheck]:
        signals = [
            signal for signal in fraud_signals if "bank account change" in signal.title.lower()
        ]
        if not signals:
            return []
        evidence = [item for signal in signals for item in signal.evidence]
        return [
            CrossDocumentCheck(
                check_id=f"xdoc-{case_id}-bank-change",
                title="Bank account change request detected",
                status="Review Required",
                confidence=0.86,
                evidence=evidence,
                recommendation="Verify vendor banking change through an independent channel.",
            )
        ]

    def _workflow(
        self,
        case_id: str,
        row: dict[str, Any],
        action: WorkflowAction,
        approval_blocked: bool,
        blocked_reasons: list[str],
        risk_score: int,
    ) -> WorkflowState:
        allowed = [
            WorkflowAction.request_changes,
            WorkflowAction.escalate_to_legal,
            WorkflowAction.escalate_to_finance,
            WorkflowAction.escalate_to_compliance,
            WorkflowAction.hold_payment,
            WorkflowAction.reject,
        ]
        if not approval_blocked:
            allowed = [WorkflowAction.approve, *allowed]
        return WorkflowState(
            case_id=case_id,
            status=self.scoring.risk_severity(risk_score),
            assigned_team=row.get("assigned_team") or "Risk Operations",
            next_action=row.get("reason") or f"Recommended action: {action.value}",
            recommended_action=action,
            allowed_actions=allowed,
            approval_required=True,
            approval_blocked=approval_blocked,
            blocked_reasons=blocked_reasons,
        )

    def _agents(
        self,
        case_id: str,
        compliance_findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
        workflow: WorkflowState,
        evidence_by_id: dict[str, Evidence],
    ) -> list[AgentDecision]:
        decisions = []
        for index, row in enumerate(self.loader.load_case_agents(case_id), start=1):
            agent = row["agent_name"]
            status = "completed"
            if workflow.approval_blocked and agent in {
                "Legal Agent",
                "Financial Agent",
                "Compliance Agent",
                "Risk Agent",
            }:
                status = (
                    "blocked" if agent in {"Financial Agent", "Risk Agent"} else "review_required"
                )
            confidence = max(
                0.35, min(0.97, self._float(row.get("confidence"), 0.75) - index * 0.007)
            )
            evidence = self._agent_evidence(
                agent, compliance_findings, fraud_signals, evidence_by_id
            )
            evidence_ids = [item.evidence_id for item in evidence]
            reasoning = self._agent_reasoning(agent, compliance_findings, fraud_signals, workflow)
            next_action = (
                row.get("next_action")
                if row.get("next_action") != "none"
                else workflow.recommended_action.value
            )
            decisions.append(
                AgentDecision(
                    agent_name=agent,
                    decision=self._agent_decision(
                        agent, row.get("decision", ""), compliance_findings, fraud_signals, workflow
                    ),
                    confidence=round(confidence, 3),
                    evidence_ids=evidence_ids,
                    evidence=evidence,
                    reasoning_summary=reasoning,
                    reasoning_chain=[reasoning, f"Next action: {next_action}."],
                    next_action=next_action,
                    status=status,
                )
            )
        return decisions

    def _agent_evidence(
        self,
        agent: str,
        compliance_findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
        evidence_by_id: dict[str, Evidence],
    ) -> list[Evidence]:
        all_evidence = list(evidence_by_id.values())
        compliance_evidence = [item for finding in compliance_findings for item in finding.evidence]
        fraud_evidence = [item for signal in fraud_signals for item in signal.evidence]
        if agent == "Document Understanding Agent":
            selected = all_evidence[:3]
        elif agent == "Retrieval Agent":
            selected = all_evidence[:5]
        elif agent == "Legal Agent":
            selected = compliance_evidence or [
                item for item in all_evidence if "contract" in item.source_document.lower()
            ]
        elif agent == "Financial Agent":
            selected = [
                item
                for item in fraud_evidence
                if any(
                    term in f"{item.source_document} {item.text}".lower()
                    for term in ("invoice", "email", "bank", "wire", "override")
                )
            ] or fraud_evidence
        elif agent == "Compliance Agent":
            selected = compliance_evidence
        elif agent == "Risk Agent":
            selected = [*compliance_evidence, *fraud_evidence]
        elif agent == "Summary Agent":
            selected = [*compliance_evidence[:1], *fraud_evidence[:2]]
        else:
            selected = all_evidence[:3]
        return self._dedupe_agent_evidence(selected)[:5]

    def _dedupe_agent_evidence(self, evidence: list[Evidence]) -> list[Evidence]:
        seen = set()
        result = []
        for item in evidence:
            key = (
                item.evidence_id or item.chunk_id or (item.document_id, item.page_number, item.text)
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _graph(
        self,
        case_id: str,
        documents: list[DocumentRecord],
        entities: list[dict[str, Any]],
        evidence_chunks: list[EvidenceChunk],
        compliance_findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
        workflow: WorkflowState,
        risk_score: int,
    ) -> dict[str, Any]:
        nodes = {}
        edges = {}

        def add_node(node_id: str, label: str, properties: dict[str, Any] | None = None) -> None:
            nodes[node_id] = {"id": node_id, "label": label, "properties": properties or {}}

        def add_edge(
            source: str, relationship: str, target: str, properties: dict[str, Any] | None = None
        ) -> None:
            key = (source, relationship, target)
            edges[key] = {
                "source": source,
                "type": relationship,
                "target": target,
                "properties": properties or {},
            }

        add_node(case_id, "Case", {"case_id": case_id})
        for document in documents:
            add_node(
                document.document_id,
                "Document",
                {"filename": document.filename, "type": document.document_type.value},
            )
            add_edge(case_id, "CASE_CONTAINS_DOCUMENT", document.document_id)
        for entity in entities:
            node_id = f"{entity['entity_type']}:{self._normalize(entity['entity_value'])}"
            add_node(
                node_id,
                self._entity_label(entity["entity_type"]),
                {"value": entity["entity_value"]},
            )
            add_edge(entity["source_document_id"], "DOCUMENT_MENTIONS_ENTITY", node_id)
            if entity["entity_type"] == "vendor":
                invoice_docs = [
                    doc for doc in documents if doc.document_type == DocumentType.invoice
                ]
                for invoice in invoice_docs:
                    add_edge(invoice.document_id, "INVOICE_ISSUED_BY_VENDOR", node_id)
            if entity["entity_type"] == "purchase_order_number":
                invoice_docs = [
                    doc for doc in documents if doc.document_type == DocumentType.invoice
                ]
                for invoice in invoice_docs:
                    add_edge(invoice.document_id, "INVOICE_REFERENCES_PO", node_id)
        for finding in compliance_findings:
            finding_node = f"finding:{finding.finding_id}"
            add_node(
                finding_node,
                "Finding",
                {"title": finding.title, "severity": finding.severity.value},
            )
            if "missing" in finding.title.lower():
                add_node(
                    f"clause:{finding.clause_or_policy}",
                    "Clause",
                    {"name": finding.clause_or_policy, "status": "missing"},
                )
                add_edge(
                    finding_node, "CONTRACT_MISSING_CLAUSE", f"clause:{finding.clause_or_policy}"
                )
            for evidence in finding.evidence:
                evidence_node = f"evidence:{evidence.evidence_id}"
                add_node(
                    evidence_node,
                    "Evidence",
                    {"document": evidence.source_document, "page": evidence.page_number},
                )
                add_edge(finding_node, "FINDING_SUPPORTED_BY_EVIDENCE", evidence_node)
        for signal in fraud_signals:
            signal_node = f"fraud:{signal.signal_id}"
            add_node(
                signal_node,
                "FraudSignal",
                {"title": signal.title, "severity": signal.severity.value},
            )
            for evidence in signal.evidence:
                evidence_node = f"evidence:{evidence.evidence_id}"
                add_node(
                    evidence_node,
                    "Evidence",
                    {"document": evidence.source_document, "page": evidence.page_number},
                )
                add_edge(signal_node, "FINDING_SUPPORTED_BY_EVIDENCE", evidence_node)
        risk_node = f"risk:{case_id}"
        workflow_node = f"workflow:{case_id}"
        add_node(risk_node, "Risk", {"risk_score": risk_score})
        add_node(
            workflow_node, "Workflow", {"recommended_action": workflow.recommended_action.value}
        )
        add_edge(case_id, "CASE_HAS_RISK", risk_node)
        add_edge(case_id, "CASE_HAS_WORKFLOW_ACTION", workflow_node)
        return {"nodes": list(nodes.values()), "edges": list(edges.values())}

    def _audit_events(
        self,
        case_id: str,
        documents: list[DocumentRecord],
        compliance_findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
        graph: dict[str, Any],
    ) -> list[AuditEvent]:
        events = [
            ("case_created", "Dataset case loaded."),
            ("document_ingested", f"Loaded {len(documents)} dataset documents."),
            ("evidence_indexed", "Indexed dataset evidence chunks."),
            ("compliance_analyzed", f"Loaded {len(compliance_findings)} compliance finding(s)."),
            ("fraud_analyzed", f"Loaded {len(fraud_signals)} fraud finding(s)."),
            (
                "graph_created",
                f"Created {len(graph['nodes'])} graph nodes and {len(graph['edges'])} edges.",
            ),
            ("report_generated", "Generated markdown, JSON, and PDF report outputs."),
        ]
        return [
            AuditEvent(
                case_id=case_id, event_type=event_type, message=message, created_at=utc_now_iso()
            )
            for event_type, message in events
        ]

    def _evidence_from_chunk(self, chunk: EvidenceChunk) -> Evidence:
        return Evidence(
            evidence_id=chunk.chunk_id,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_document=chunk.source_document,
            page_number=chunk.page_number,
            text=chunk.text,
            extraction_method=chunk.extraction_method,
            confidence=chunk.confidence,
            reasoning="Dataset evidence chunk.",
        )

    def _first_evidence_by_doc(self, evidence: Any) -> dict[str, Evidence]:
        result = {}
        for item in evidence:
            result.setdefault(item.document_id, item)
        return result

    def _filename_for_document(self, case_id: str, document_id: str) -> str:
        for document in self.loader.load_case_documents(case_id):
            if document["document_id"] == document_id:
                return Path(document["filename"]).name
        return document_id

    def _document_type(self, value: str | None) -> DocumentType:
        aliases = {"policy": "compliance_policy"}
        normalized = aliases.get(value or "", value or "unknown")
        try:
            return DocumentType(normalized)
        except ValueError:
            return DocumentType.unknown

    def _risk_level(self, value: str | None) -> RiskLevel:
        try:
            return RiskLevel((value or "medium").lower())
        except ValueError:
            return RiskLevel.medium

    def _workflow_action(self, value: str | None) -> WorkflowAction:
        try:
            return WorkflowAction(value or "request_changes")
        except ValueError:
            return WorkflowAction.request_changes

    def _bool(self, value: Any) -> bool:
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    def _int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize(self, value: str) -> str:
        return " ".join(str(value).lower().strip().split())

    def _is_invalid_entity(self, row: dict[str, Any]) -> bool:
        entity_type = row.get("entity_type", "")
        value = row.get("entity_value", "")
        if entity_type == "vendor" and len(value.split()) > 6:
            return True
        if entity_type == "indemnity_clause" and any(
            marker in value.lower()
            for marker in ("missing indemnity", "does not include", "without", "not found", "lacks")
        ):
            return True
        return False

    def _clause_or_policy(self, finding: str) -> str:
        lower = finding.lower()
        for key in (
            "indemnity",
            "confidentiality",
            "termination",
            "governing law",
            "audit rights",
            "hipaa baa",
        ):
            if key in lower:
                return key
        return "policy"

    def _finding_reason(self, title: str) -> str:
        if "indemnity" in title.lower():
            return "Required indemnity language was not found in extracted text."
        return title

    def _fraud_reason(self, title: str) -> str:
        lower = title.lower()
        if "bank account change" in lower:
            return "Payment instructions changed by email and require independent verification."
        if "manual override" in lower:
            return "Manual override was applied without approval."
        if "urgent" in lower:
            return "Urgent payment wording increases payment risk."
        return title

    def _fraud_signal_score(self, title: str) -> int:
        lower = title.lower()
        if "duplicate invoice" in lower:
            return 35
        if "bank account change" in lower:
            return 30
        if "manual override" in lower:
            return 25
        if "amount does not match" in lower or "amount mismatch" in lower:
            return 25
        if "urgent" in lower:
            return 15
        if "off-cycle" in lower:
            return 10
        if "vendor anomaly" in lower:
            return 20
        return 10

    def _humanize_check(self, value: str) -> str:
        return value.replace("_", " ").capitalize()

    def _entity_label(self, entity_type: str) -> str:
        labels = {
            "vendor": "Vendor",
            "company": "Company",
            "invoice_number": "Invoice",
            "purchase_order_number": "PurchaseOrder",
            "contract_number": "Contract",
            "payment_terms": "Clause",
            "governing_law": "Clause",
            "invoice_amount": "Amount",
            "po_amount": "Amount",
        }
        return labels.get(entity_type, "Entity")

    def _agent_decision(
        self,
        agent: str,
        dataset_decision: str,
        compliance_findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
        workflow: WorkflowState,
    ) -> str:
        if agent == "Compliance Agent" and compliance_findings:
            return f"{len(compliance_findings)} compliance finding(s) require review"
        if agent == "Financial Agent" and fraud_signals:
            return "Payment hold recommended"
        if agent == "Risk Agent" and workflow.approval_blocked:
            return "Case blocked due to critical risk controls"
        return dataset_decision

    def _agent_reasoning(
        self,
        agent: str,
        compliance_findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
        workflow: WorkflowState,
    ) -> str:
        if agent == "Legal Agent":
            return "Reviewed contract clauses and approval-blocking compliance findings."
        if agent == "Financial Agent":
            return "Reviewed invoice, PO, bank-change, and urgent payment evidence."
        if agent == "Compliance Agent":
            return f"Reviewed {len(compliance_findings)} dataset compliance finding(s)."
        if agent == "Risk Agent":
            return f"Risk action is {workflow.recommended_action.value}; approval blocked={workflow.approval_blocked}."
        if agent == "Retrieval Agent":
            return "Indexed dataset evidence chunks for citation-grounded answers."
        return f"Reviewed {len(compliance_findings)} compliance and {len(fraud_signals)} fraud signals."
