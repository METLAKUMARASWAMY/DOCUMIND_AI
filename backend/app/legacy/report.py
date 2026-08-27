import json
import re
from textwrap import wrap

from app.models.schemas import AnalysisResponse


class ReportGenerationService:
    def executive_summary(
        self,
        document_count: int,
        compliance_score: int,
        risk_score: int,
        fraud_probability: float,
        workflow_action: str | None = None,
    ) -> str:
        risk_label = "low"
        if risk_score >= 75:
            risk_label = "critical"
        elif risk_score >= 50:
            risk_label = "high"
        elif risk_score >= 25:
            risk_label = "medium"
        action = self._humanize(workflow_action or "review")
        return (
            f"Processed {document_count} enterprise document(s). Compliance score is {compliance_score}/100, "
            f"risk is {risk_label} at {risk_score}/100, and fraud probability is {fraud_probability:.0%}. "
            f"Recommended workflow action is {action}."
        )

    def render_payload(self, analysis: AnalysisResponse) -> dict:
        executive_summary = self.summary_for_analysis(analysis)
        return {
            "case_id": analysis.case_id,
            "tenant_id": analysis.tenant_id,
            "executive_summary": executive_summary,
            "uploaded_documents": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "document_type": document.document_type.value,
                    "checksum": document.checksum,
                    "pages": document.pages,
                }
                for document in analysis.documents
            ],
            "extracted_entities": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "name": field.name,
                    "value": field.value,
                    "normalized_value": field.normalized_value,
                    "confidence": field.confidence,
                    "evidence_id": field.evidence.evidence_id,
                }
                for document in analysis.documents
                for field in document.extracted_fields
            ],
            "scores": {
                "compliance_score": analysis.compliance_score,
                "risk_score": analysis.risk_score,
                "fraud_probability": analysis.fraud_probability,
                "fraud_confidence": analysis.fraud_confidence,
                "risk_severity": analysis.scores.get("risk_severity")
                if analysis.scores
                else analysis.workflow.status,
                "approval_blocked": analysis.workflow.approval_blocked,
            },
            "compliance_report": [
                finding.model_dump(mode="json") for finding in analysis.compliance_findings
            ],
            "fraud_report": [signal.model_dump(mode="json") for signal in analysis.fraud_signals],
            "risk_report": {
                "cross_document_checks": [
                    check.model_dump(mode="json") for check in analysis.cross_document_checks
                ],
                "workflow": analysis.workflow.model_dump(mode="json"),
            },
            "agent_decisions": [
                decision.model_dump(mode="json") for decision in analysis.agent_decisions
            ],
            "audit_trail": [event.model_dump(mode="json") for event in analysis.audit_events],
            "evidence_citations": [
                chunk.model_dump(mode="json") for chunk in analysis.evidence_chunks
            ],
        }

    def render_markdown(self, analysis: AnalysisResponse) -> str:
        payload = self.render_payload(analysis)
        findings = (
            "\n".join(
                f"| {finding.title} | {self._humanize(str(finding.severity.value))} | {finding.framework} | "
                f"{self._humanize(finding.clause_or_policy)} | {'Yes' if finding.approval_blocking else 'No'} | "
                f"{self._evidence_ids(finding.evidence)} |"
                for finding in analysis.compliance_findings
            )
            or "| None | - | - | - | - | - |"
        )
        fraud = (
            "\n".join(
                f"| {signal.title} | {self._humanize(str(signal.severity.value))} | {signal.probability:.0%} | "
                f"{self._evidence_ids(signal.evidence)} | {signal.recommendation} |"
                for signal in analysis.fraud_signals
            )
            or "| None | - | - | - | - |"
        )
        cross_checks = (
            "\n".join(
                f"| {check.title} | {self._humanize(check.status)} | {check.confidence:.0%} | "
                f"{check.left_value or '-'} | {check.right_value or '-'} |"
                for check in analysis.cross_document_checks
            )
            or "| None | - | - | - | - |"
        )
        documents = "\n".join(
            f"| {doc.filename} | {self._humanize(doc.document_type.value)} | {doc.pages} | {doc.checksum[:12]} |"
            for doc in analysis.documents
        )
        entities = self._entity_table(payload["extracted_entities"])
        agents = "\n".join(
            f"| {decision.agent_name} | {self._humanize(decision.status)} | {decision.confidence:.0%} | "
            f"{self._humanize(decision.next_action)} | {self._humanize(decision.decision)} |"
            for decision in analysis.agent_decisions
        )
        evidence_refs = "\n".join(
            f"| {chunk.chunk_id} | {chunk.source_document} | {chunk.page_number} | {chunk.confidence:.0%} |"
            for chunk in analysis.evidence_chunks[:30]
        )
        audit = (
            "\n".join(
                f"| {event.created_at} | {self._humanize(event.event_type)} | {event.actor} | {event.message} |"
                for event in analysis.audit_events
            )
            or "| None | - | - | - |"
        )
        final_recommendation = (
            "Do not release payment until bank details are independently verified and approval-blocking findings are resolved."
            if analysis.workflow.approval_blocked
            else "Proceed through the configured approval workflow."
        )
        return f"""# DocuMind AI Report

## Case Metadata
| Field | Value |
| --- | --- |
| Case ID | {analysis.case_id} |
| Tenant | {analysis.tenant_id} |
| Industry | {analysis.industry or "-"} |
| Company | {analysis.company or "-"} |
| Vendor | {analysis.vendor or "-"} |

## Executive Summary
{analysis.executive_summary}

## Scores
| Score | Value |
| --- | --- |
| Compliance | {analysis.compliance_score}/100 |
| Risk | {analysis.risk_score}/100 |
| Fraud Probability | {analysis.fraud_probability:.0%} |
| Fraud Confidence | {self._humanize(analysis.fraud_confidence)} |
| Approval Gate | {"Blocked" if analysis.workflow.approval_blocked else "Open"} |

## Uploaded Documents
| Document | Type | Pages | Checksum |
| --- | --- | ---: | --- |
{documents}

## Extracted Entities
| Document | Entity | Value | Confidence |
| --- | --- | --- | ---: |
{entities}

## Compliance Findings
| Finding | Severity | Framework | Policy | Approval Blocking | Evidence |
| --- | --- | --- | --- | --- | --- |
{findings}

## Fraud Findings
| Signal | Severity | Probability | Evidence | Recommendation |
| --- | --- | ---: | --- | --- |
{fraud}

## Cross-Document Checks
| Check | Status | Confidence | Left Value | Right Value |
| --- | --- | ---: | --- | --- |
{cross_checks}

## Workflow Recommendation
| Field | Value |
| --- | --- |
| Status | {self._humanize(analysis.workflow.status)} |
| Assigned Team | {analysis.workflow.assigned_team} |
| Recommended Action | {self._humanize(analysis.workflow.recommended_action.value)} |
| Next Action | {analysis.workflow.next_action} |
| Blocked Reasons | {", ".join(self._humanize(reason) for reason in analysis.workflow.blocked_reasons) or "None"} |

## Agent Decisions
| Agent | Status | Confidence | Next Action | Decision |
| --- | --- | ---: | --- | --- |
{agents}

## Evidence References
| Chunk ID | Document | Page | Confidence |
| --- | --- | ---: | ---: |
{evidence_refs}

## Audit Trail
| Timestamp | Event | Actor | Message |
| --- | --- | --- | --- |
{audit}

## Final Recommendation
{final_recommendation}
"""

    def render_json(self, analysis: AnalysisResponse) -> str:
        return json.dumps(self.render_payload(analysis), indent=2, default=str)

    def summary_for_analysis(self, analysis: AnalysisResponse) -> str:
        return self.executive_summary(
            len(analysis.documents),
            analysis.compliance_score,
            analysis.risk_score,
            analysis.fraud_probability,
            analysis.workflow.recommended_action.value,
        )

    def render_pdf_bytes(self, analysis: AnalysisResponse) -> bytes:
        lines = []
        for raw_line in self.render_markdown(analysis).splitlines():
            if raw_line.startswith("|") and "---" in raw_line:
                continue
            cleaned = raw_line.replace("|", " ").strip()
            lines.extend(wrap(cleaned, width=92) or [""])
        stream_lines = []
        y = 780
        for index, line in enumerate(lines[:58]):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if index == 0:
                stream_lines.append(f"BT /F1 16 Tf 40 {y} Td ({escaped}) Tj ET")
            else:
                stream_lines.append(f"BT /F1 9 Tf 40 {y - index * 12} Td ({escaped}) Tj ET")
        stream = "\n".join(stream_lines)
        objects = [
            "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
            "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
            "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
            f"5 0 obj << /Length {len(stream)} >> stream\n{stream}\nendstream endobj",
        ]
        body = "%PDF-1.4\n" + "\n".join(objects) + "\n"
        offsets = []
        cursor = len("%PDF-1.4\n")
        for obj in objects:
            offsets.append(cursor)
            cursor += len(obj) + 1
        xref_start = len(body)
        xref = "xref\n0 6\n0000000000 65535 f \n" + "".join(
            f"{offset:010d} 00000 n \n" for offset in offsets
        )
        trailer = f"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF"
        return (body + xref + trailer).encode("utf-8")

    def _entity_table(self, entities: list[dict]) -> str:
        if not entities:
            return "| None | - | - | - |"
        rows = []
        for item in entities[:40]:
            rows.append(
                f"| {item['filename']} | {self._humanize(item['name'])} | {item['value']} | "
                f"{item['confidence']:.0%} |"
            )
        return "\n".join(rows)

    def _evidence_ids(self, evidence) -> str:
        ids = [item.evidence_id for item in evidence[:3]]
        return ", ".join(ids) if ids else "-"

    def _humanize(self, value: str) -> str:
        overrides = {
            "hold_payment": "Hold Payment",
            "review_required": "Review Required",
            "approval_blocking": "Approval Blocking",
            "case_created": "Case Created",
            "document_ingested": "Document Ingested",
            "evidence_indexed": "Evidence Indexed",
            "compliance_analyzed": "Compliance Analyzed",
            "fraud_analyzed": "Fraud Analyzed",
            "graph_created": "Graph Created",
            "workflow_decision_recorded": "Workflow Decision Recorded",
            "purchase_order": "Purchase Order",
            "bank_statement": "Bank Statement",
            "compliance_policy": "Compliance Policy",
        }
        raw = str(value).strip()
        if not raw:
            return "-"
        key = raw.lower()
        if key in overrides:
            return overrides[key]
        protected = raw
        protected = re.sub(r"(?i)sha[-_ ]?256", " SHA256TOKEN ", protected)
        protected = re.sub(r"(?i)ci[/_ -]?cd", " CICDTOKEN ", protected)
        protected = re.sub(r"[_-]+", " ", protected)
        protected = re.sub(r"\s+", " ", protected).strip()
        label = protected.title()
        replacements = {
            "Api": "API",
            "Json": "JSON",
            "Pdf": "PDF",
            "Pii": "PII",
            "Rbac": "RBAC",
            "Sla": "SLA",
            "Soc2": "SOC2",
            "Hipaa": "HIPAA",
            "Gdpr": "GDPR",
            "Xai": "XAI",
            "Rag": "RAG",
            "Ocr": "OCR",
            "Mlops": "MLOps",
            "Sha256Token": "SHA-256",
            "Cicdtoken": "CI/CD",
        }
        for source, target in replacements.items():
            label = re.sub(rf"\b{source}\b", target, label)
        return label
