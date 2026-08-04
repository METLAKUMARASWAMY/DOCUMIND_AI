from collections import defaultdict
from typing import Any

from app.models.schemas import (
    AgentDecision,
    AnalysisResponse,
    ComplianceFinding,
    CrossDocumentCheck,
    DocumentRecord,
    DocumentType,
    FraudSignal,
    RiskLevel,
    WorkflowState,
)

GRAPH_FIELD_LABELS = {
    "vendor": "Vendor",
    "amount": "Amount",
    "invoice_number": "Invoice",
    "purchase_order_number": "PurchaseOrder",
    "bank_identifier": "Company",
    "bank_account_holder": "Company",
    "termination_clause": "Clause",
    "confidentiality_clause": "Clause",
    "indemnity_clause": "Clause",
    "governing_law_clause": "Clause",
    "payment_terms": "Clause",
    "audit_rights_clause": "Clause",
}


class KnowledgeGraphEngine:
    def build(
        self,
        case_id: str,
        documents: list[DocumentRecord],
        compliance_findings: list[ComplianceFinding] | None = None,
        fraud_signals: list[FraudSignal] | None = None,
        cross_document_checks: list[CrossDocumentCheck] | None = None,
        agent_decisions: list[AgentDecision] | None = None,
        workflow: WorkflowState | None = None,
        risk_score: int = 0,
        tenant_id: str = "tenant-local",
    ) -> dict:
        compliance_findings = compliance_findings or []
        fraud_signals = fraud_signals or []
        cross_document_checks = cross_document_checks or []
        agent_decisions = agent_decisions or []
        nodes = []
        edges = []
        seen_nodes = set()
        seen_edges = set()

        def add_node(node_id: str, label: str, properties: dict) -> None:
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                nodes.append({"id": node_id, "label": label, "properties": properties})

        def add_edge(source: str, target: str, relation_type: str, properties: dict | None = None) -> None:
            edge_key = (source, target, relation_type, tuple(sorted((properties or {}).items())))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "type": relation_type,
                        "properties": properties or {},
                    }
                )

        add_node(case_id, "Case", {"case_id": case_id, "tenant_id": tenant_id})
        risk_id = f"risk:{case_id}"
        add_node(risk_id, "Risk", {"risk_score": risk_score})
        add_edge(case_id, risk_id, "CASE_HAS_RISK")

        if workflow:
            workflow_id = f"workflow:{case_id}"
            add_node(workflow_id, "Workflow", workflow.model_dump(mode="json"))
            add_edge(case_id, workflow_id, "CASE_HAS_WORKFLOW_ACTION", {"workflow_status": workflow.status})

        entity_index: dict[str, list[str]] = defaultdict(list)
        document_by_type: dict[DocumentType, list[DocumentRecord]] = defaultdict(list)

        for document in documents:
            document_by_type[document.document_type].append(document)
            add_node(
                document.document_id,
                "Document",
                {
                    "filename": document.filename,
                    "type": document.document_type,
                    "checksum": document.checksum,
                    "pages": document.pages,
                },
            )
            add_edge(case_id, document.document_id, "CASE_CONTAINS_DOCUMENT")
            for field in document.extracted_fields:
                if field.name not in GRAPH_FIELD_LABELS:
                    continue
                label = GRAPH_FIELD_LABELS[field.name]
                entity_id = f"{field.name}:{self._normalize_key(field.value)}"
                add_node(
                    entity_id,
                    label,
                    {
                        "value": field.value,
                        "field": field.name,
                        "confidence": field.confidence,
                    },
                )
                add_edge(document.document_id, entity_id, self._relationship_for_field(document, field.name))
                entity_index[entity_id].append(document.document_id)

        self._add_check_relationships(add_edge, cross_document_checks)

        for finding in compliance_findings:
            finding_id = f"finding:{finding.finding_id}"
            framework_id = f"framework:{finding.framework}"
            add_node(
                finding_id,
                "Finding",
                {
                    "category": "compliance",
                    "title": finding.title,
                    "severity": finding.severity,
                    "status": finding.status,
                    "approval_blocking": finding.approval_blocking,
                    "recommendation": finding.recommendation,
                },
            )
            add_node(framework_id, "ComplianceFramework", {"framework": finding.framework})
            add_edge(risk_id, finding_id, "CASE_HAS_RISK")
            add_edge(finding_id, framework_id, "DOCUMENT_MENTIONS_ENTITY")
            for evidence in finding.evidence[:3]:
                evidence_id = f"evidence:{evidence.evidence_id}"
                add_node(evidence_id, "Evidence", {"text": evidence.text[:240], "page": evidence.page_number})
                add_edge(finding_id, evidence_id, "FINDING_SUPPORTED_BY_EVIDENCE")
                add_edge(evidence_id, evidence.document_id, "DOCUMENT_MENTIONS_ENTITY")

        for signal in fraud_signals:
            signal_id = f"finding:{signal.signal_id}"
            add_node(
                signal_id,
                "Finding",
                {
                    "category": "fraud",
                    "title": signal.title,
                    "severity": signal.severity,
                    "probability": signal.probability,
                    "recommendation": signal.recommendation,
                },
            )
            add_edge(risk_id, signal_id, "CASE_HAS_RISK")
            for evidence in signal.evidence[:3]:
                evidence_id = f"evidence:{evidence.evidence_id}"
                add_node(evidence_id, "Evidence", {"text": evidence.text[:240], "page": evidence.page_number})
                add_edge(signal_id, evidence_id, "FINDING_SUPPORTED_BY_EVIDENCE")
                add_edge(evidence_id, evidence.document_id, "DOCUMENT_MENTIONS_ENTITY")

        for decision in agent_decisions:
            agent_id = f"agent:{self._normalize_key(decision.agent_name)}"
            add_node(agent_id, "Agent", {"agent_name": decision.agent_name, "status": decision.status})
            for finding_id in decision.evidence_ids[:4]:
                add_edge(agent_id, f"evidence:{finding_id}", "AGENT_CREATED_FINDING")

        for entity_id, doc_ids in entity_index.items():
            if len(doc_ids) > 1:
                for doc_id in doc_ids:
                    add_edge(doc_id, entity_id, "DOCUMENT_MENTIONS_ENTITY", {"cross_document": True})

        return {
            "nodes": nodes,
            "edges": edges,
            "neo4j_projection": {
                "node_labels": [
                    "Case",
                    "Document",
                    "Vendor",
                    "Company",
                    "Invoice",
                    "PurchaseOrder",
                    "Contract",
                    "Clause",
                    "Amount",
                    "Finding",
                    "Risk",
                    "ComplianceFramework",
                    "WorkflowDecision",
                    "Evidence",
                    "Agent",
                ],
                "relationship_types": sorted({edge["type"] for edge in edges}),
                "merge_keys": ["case_id", "document_id", "field:value", "finding_id", "evidence_id"],
                "fallback_repository": "local_graph_repository",
            },
        }

    def entity_neighborhood(self, graph: dict[str, Any], entity_id: str) -> dict[str, Any]:
        edges = [
            edge
            for edge in graph.get("edges", [])
            if edge.get("source") == entity_id or edge.get("target") == entity_id
        ]
        connected_ids = {entity_id}
        for edge in edges:
            connected_ids.add(edge["source"])
            connected_ids.add(edge["target"])
        nodes = [node for node in graph.get("nodes", []) if node.get("id") in connected_ids]
        return {
            "entity_id": entity_id,
            "nodes": nodes,
            "edges": edges,
            "relationship_count": len(edges),
        }

    def risk_paths(self, cases: list[AnalysisResponse], min_risk_score: int = 50) -> list[dict]:
        paths = []
        for case in cases:
            if case.risk_score < min_risk_score and case.fraud_probability < 0.5:
                continue
            case_findings = [
                finding
                for finding in case.compliance_findings
                if finding.severity in {RiskLevel.high, RiskLevel.critical}
            ]
            case_signals = [
                signal
                for signal in case.fraud_signals
                if signal.severity in {RiskLevel.high, RiskLevel.critical} or signal.probability >= 0.6
            ]
            paths.append(
                {
                    "case_id": case.case_id,
                    "risk_score": case.risk_score,
                    "fraud_probability": case.fraud_probability,
                    "workflow_status": case.workflow.status,
                    "paths": [
                        {
                            "start": case.case_id,
                            "relationship": "CASE_HAS_RISK",
                            "end": item.finding_id if hasattr(item, "finding_id") else item.signal_id,
                            "title": item.title,
                            "severity": item.severity,
                        }
                        for item in [*case_findings, *case_signals]
                    ],
                }
            )
        return paths

    def _add_check_relationships(self, add_edge, checks: list[CrossDocumentCheck]) -> None:
        for check in checks:
            documents_with_evidence = [evidence.document_id for evidence in check.evidence if evidence.document_id]
            if len(documents_with_evidence) < 2:
                continue
            relation_type = "DOCUMENT_MENTIONS_ENTITY"
            if "PO reference" in check.title:
                relation_type = "INVOICE_REFERENCES_PO"
            elif check.status.lower() == "passed":
                relation_type = "DOCUMENT_MENTIONS_ENTITY"
            elif "bank" in check.title.lower():
                relation_type = "PAYMENT_HAS_ANOMALY"
            else:
                relation_type = "CASE_HAS_RISK"
            add_edge(
                documents_with_evidence[0],
                documents_with_evidence[1],
                relation_type,
                {
                    "check_id": check.check_id,
                    "title": check.title,
                    "status": check.status,
                    "confidence": check.confidence,
                },
            )

    def _relationship_for_field(self, document: DocumentRecord, field_name: str) -> str:
        if field_name == "vendor" and document.document_type == DocumentType.invoice:
            return "VENDOR_ISSUED_INVOICE"
        if field_name == "purchase_order_number" and document.document_type == DocumentType.invoice:
            return "INVOICE_REFERENCES_PO"
        if field_name.endswith("_clause") and document.document_type == DocumentType.contract:
            return "CONTRACT_HAS_CLAUSE"
        return "DOCUMENT_MENTIONS_ENTITY"

    def _normalize_key(self, value: str) -> str:
        return " ".join(value.lower().split()).strip(" .,:;")
