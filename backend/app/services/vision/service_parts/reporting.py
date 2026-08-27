"""Finance narratives and reproducible report generation.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

from typing import Any

from app.models.vision import (
    ReviewerStatus,
    VisionFinanceNarrative,
    VisionFinanceNarrativeRequest,
    VisionReportRecord,
    VisionReportRequest,
)
from app.services.vision.finance_narratives import verified_finance_narrative_service
from app.services.vision.reporting import vision_report_service


class ReportingMixin:
    def generate_finance_narrative(
        self,
        request: VisionFinanceNarrativeRequest,
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionFinanceNarrative:
        with self._artifact_lock:
            narrative = verified_finance_narrative_service.generate(
                request,
                self.list_documents(tenant_id),
                tenant_id=tenant_id,
                actor=actor,
            )
            contributing = [
                self.get_document(document_id, tenant_id) for document_id in narrative.document_ids
            ]
            source_ids = narrative.source_document_ids or narrative.document_ids
            selected = [self.get_document(document_id, tenant_id) for document_id in source_ids]
            source_data_version = self._artifact_source_data_version(selected)
            request_scope = request.model_dump(
                mode="json",
                exclude={"generate_new_version"},
            )
            request_scope["document_ids"] = sorted(request.document_ids or source_ids)
            request_scope["requested_document_ids"] = sorted(request.document_ids)
            request_scope["source_document_ids"] = sorted(source_ids)
            request_scope["excluded_document_ids"] = sorted(narrative.excluded_document_ids)
            duplicate_exclusion_signature = sorted(
                {
                    trace.calculation_function
                    for trace in narrative.calculations
                    if trace.label == "Duplicate invoice documents excluded"
                }
            )
            request_scope["duplicate_invoice_exclusions"] = duplicate_exclusion_signature
            identity_hash = self._artifact_identity_hash(
                {
                    "organisation_id": tenant_id,
                    "narrative_type": request.narrative_type.value,
                    "vendor_filter": self._normalize(request.vendor_name or ""),
                    "from_date": request.date_from,
                    "to_date": request.date_to,
                    "currency": sorted(
                        {
                            self._field_map(document).get("currency", "UNSPECIFIED")
                            for document in selected
                        }
                    ),
                    "document_ids": sorted(set(narrative.document_ids)),
                    "source_document_ids": sorted(set(source_ids)),
                    "duplicate_invoice_exclusions": duplicate_exclusion_signature,
                    "verified_data_version": source_data_version,
                }
            )
            matches = [
                item
                for item in self.list_finance_narratives(tenant_id)
                if item.identity_hash == identity_hash
            ]
            latest = max(matches, key=lambda item: item.version, default=None)
            if latest and not request.generate_new_version:
                return latest
            narrative.identity_hash = identity_hash
            narrative.source_data_version = source_data_version
            narrative.request_scope = request_scope
            if latest:
                latest.status = "superseded"
                self._persist_finance_narrative(latest)
                narrative.version = latest.version + 1
                narrative.supersedes = latest.narrative_id
            self._persist_finance_narrative(narrative)
            for document in selected:
                self._audit(
                    document,
                    "finance_narrative_generated",
                    f"Generated {narrative.narrative_type.value} version {narrative.version}.",
                    actor=actor,
                    metadata={
                        "narrative_id": narrative.narrative_id,
                        "identity_hash": narrative.identity_hash,
                        "version": narrative.version,
                        "contributing": document.document_id
                        in {item.document_id for item in contributing},
                        "excluded_duplicate": document.document_id
                        in set(narrative.excluded_document_ids),
                    },
                )
                self._persist_document(document)
            return narrative

    def get_finance_narrative(
        self,
        narrative_id: str,
        tenant_id: str,
    ) -> VisionFinanceNarrative:
        narrative = self.finance_narratives.get(narrative_id)
        if not narrative:
            payload = self._read_json(self._state_file("narratives", narrative_id))
            if payload:
                try:
                    narrative = VisionFinanceNarrative.model_validate(payload)
                except ValueError:
                    narrative = None
        if not narrative:
            narrative = self.persistence.get_narrative(narrative_id, tenant_id)
        if not narrative or narrative.tenant_id != tenant_id:
            raise ValueError("narrative_id not found")
        self.finance_narratives[narrative.narrative_id] = narrative
        return narrative

    def list_finance_narratives(self, tenant_id: str) -> list[VisionFinanceNarrative]:
        for path in (self._state_root / "narratives").glob("*.json"):
            payload = self._read_json(path)
            if not payload:
                continue
            try:
                narrative = VisionFinanceNarrative.model_validate(payload)
            except ValueError:
                continue
            self.finance_narratives[narrative.narrative_id] = narrative
        for narrative in self.persistence.list_narratives(tenant_id):
            self.finance_narratives[narrative.narrative_id] = narrative
        scoped = [item for item in self.finance_narratives.values() if item.tenant_id == tenant_id]
        canonical: dict[str, VisionFinanceNarrative] = {}
        for item in scoped:
            identity = item.identity_hash or self._artifact_identity_hash(
                {
                    "organisation_id": item.tenant_id,
                    "narrative_type": item.narrative_type.value,
                    "request_scope": item.request_scope,
                    "document_ids": sorted(set(item.document_ids)),
                    "source_document_ids": sorted(
                        set(item.source_document_ids or item.document_ids)
                    ),
                    "verified_data_version": item.source_data_version,
                }
            )
            current = canonical.get(identity)
            if current is None or (item.version, item.created_at) > (
                current.version,
                current.created_at,
            ):
                canonical[identity] = item
        active = [
            item
            for item in canonical.values()
            if item.status not in {"archived", "superseded", "stale"}
        ]
        return sorted(active, key=lambda item: item.created_at, reverse=True)

    def generate_report(
        self,
        request: VisionReportRequest,
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionReportRecord:
        with self._artifact_lock:
            narratives = self.list_finance_narratives(tenant_id)
            if request.narrative_id and not any(
                item.narrative_id == request.narrative_id for item in narratives
            ):
                try:
                    narratives.append(self.get_finance_narrative(request.narrative_id, tenant_id))
                except ValueError:
                    pass
            report = vision_report_service.generate(
                request,
                self.list_documents(tenant_id),
                narratives,
                tenant_id=tenant_id,
                actor=actor,
            )
            selected = [
                self.get_document(document_id, tenant_id) for document_id in report.document_ids
            ]
            source_data_version = self._artifact_source_data_version(selected)
            request_scope = request.model_dump(
                mode="json",
                exclude={"generate_new_version"},
            )
            request_scope["document_ids"] = sorted(report.document_ids)
            identity_hash = self._artifact_identity_hash(
                {
                    "tenant_id": tenant_id,
                    "report_type": request.report_type.value,
                    "document_ids": sorted(report.document_ids),
                    "document_data_versions": source_data_version,
                    "review_state": sorted(
                        (item.document_id, item.status.value) for item in selected
                    ),
                    "template_version": report.template_version,
                    "narrative_id": request.narrative_id,
                    "vendor_filter": self._normalize(request.vendor_name or ""),
                    "evaluation": request.evaluation,
                }
            )
            matches = [
                item for item in self.list_reports(tenant_id) if item.identity_hash == identity_hash
            ]
            latest = max(matches, key=lambda item: item.version, default=None)
            if latest and not request.generate_new_version:
                return latest
            report.identity_hash = identity_hash
            report.source_data_version = source_data_version
            report.request_scope = request_scope
            if latest:
                latest.status = "superseded"
                self._persist_report_record(latest)
                report.version = latest.version + 1
                report.supersedes = latest.report_id
            pdf = vision_report_service.pdf(report)
            report.object_uri = self.persistence.save_report(
                report.model_dump(mode="json"),
                pdf,
            )
            self._persist_report_record(report)
            for document in selected:
                self._audit(
                    document,
                    "report_generated",
                    f"Generated {report.report_type.value} version {report.version}.",
                    actor=actor,
                    metadata={
                        "report_id": report.report_id,
                        "identity_hash": report.identity_hash,
                        "version": report.version,
                    },
                )
                self._persist_document(document)
            return report

    def archive_report(
        self,
        report_id: str,
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionReportRecord:
        report = self.get_report(report_id, tenant_id)
        report.status = "archived"
        self._persist_report_record(report)
        for document_id in report.document_ids:
            document = self.get_document(document_id, tenant_id)
            self._audit(
                document,
                "report_archived",
                "Archived a generated report without deleting its audit record.",
                actor=actor,
                metadata={"report_id": report_id, "version": report.version},
            )
            self._persist_document(document)
        return report

    def audit_report_download(
        self,
        resource_id: str,
        *,
        tenant_id: str,
        actor: str,
    ) -> None:
        try:
            report = self.get_report(resource_id, tenant_id)
            document_ids = report.document_ids
            report_id = report.report_id
        except ValueError:
            document = self.get_document(resource_id, tenant_id)
            document_ids = [document.document_id]
            report_id = None
        for document_id in document_ids:
            document = self.get_document(document_id, tenant_id)
            self._audit(
                document,
                "report_downloaded",
                "Downloaded an evidence report as PDF.",
                actor=actor,
                metadata={"resource_id": resource_id, "report_id": report_id},
            )
            self._persist_document(document)
        self._system_audit(
            "report_downloaded",
            "Downloaded an evidence report as PDF.",
            actor=actor,
            tenant_id=tenant_id,
            metadata={"resource_id": resource_id, "report_id": report_id},
        )

    def get_report(self, report_id: str, tenant_id: str) -> VisionReportRecord:
        report = self.reports.get(report_id)
        if not report:
            payload = self._read_json(self._state_file("reports", report_id))
            if payload:
                try:
                    report = VisionReportRecord.model_validate(payload)
                except ValueError:
                    report = None
        if not report:
            report = self.persistence.get_report(report_id, tenant_id)
        if not report or report.tenant_id != tenant_id:
            raise ValueError("report_id not found")
        self.reports[report.report_id] = report
        return report

    def list_reports(self, tenant_id: str) -> list[VisionReportRecord]:
        for path in (self._state_root / "reports").glob("*.json"):
            payload = self._read_json(path)
            if not payload:
                continue
            try:
                report = VisionReportRecord.model_validate(payload)
            except ValueError:
                continue
            self.reports[report.report_id] = report
        for report in self.persistence.list_reports(tenant_id):
            self.reports[report.report_id] = report
        return sorted(
            [item for item in self.reports.values() if item.tenant_id == tenant_id],
            key=lambda item: item.created_at,
            reverse=True,
        )

    def render_stored_report_markdown(self, report_id: str, tenant_id: str) -> str:
        return vision_report_service.markdown(self.get_report(report_id, tenant_id))

    def render_stored_report_pdf(self, report_id: str, tenant_id: str) -> bytes:
        return vision_report_service.pdf(self.get_report(report_id, tenant_id))

    def render_report_payload(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        document = self.get_document(document_id, tenant_id)
        return {
            "report_type": "document_intelligence_report",
            "document": document.model_dump(mode="json"),
            "summary": self._document_summary(document),
            "risk_policy": "Suspicious indicators require human review; this report does not confirm fraud.",
            "active_learning_available": bool(
                any(
                    field.reviewer_status == ReviewerStatus.edited
                    for field in document.extracted_fields
                )
            ),
        }

    def render_report_markdown(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> str:
        payload = self.render_report_payload(document_id, tenant_id)
        document = self.get_document(document_id, tenant_id)
        fields = (
            "\n".join(
                f"| {field.name} | {self._mask_sensitive(field.name, field.reviewed_value or field.value)} | "
                f"{field.confidence:.0%} | {field.evidence.evidence_id} | {field.reviewer_status.value} |"
                for field in document.extracted_fields
                if field.confidence is not None
            )
            or "| None | - | - | - | - |"
        )
        risks = (
            "\n".join(
                f"| {risk.title} | {risk.severity.value} | {risk.category} | {risk.recommended_action} |"
                for risk in document.risk_indicators
            )
            or "| None | - | - | - |"
        )
        tables = (
            "\n\n".join(table.markdown for table in document.tables)
            or "No structural tables detected."
        )
        return f"""# DOCUMIND VISION AI Report

## Summary
{payload["summary"]}

## Document
| Field | Value |
| --- | --- |
| Document ID | {document.document_id} |
| Filename | {document.filename} |
| Type | {document.document_type.value} |
| Status | {document.status.value} |
| Pages | {document.page_count} |

## Extracted Fields
| Field | Value | Confidence | Evidence | Review |
| --- | --- | ---: | --- | --- |
{fields}

## Risk Indicators
| Indicator | Severity | Category | Recommendation |
| --- | --- | --- | --- |
{risks}

## Tables
{tables}

## Policy
Suspicious indicators require human review. Only an authorised reviewer may set CONFIRMED_FRAUD.
"""

    def render_pdf_bytes(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> bytes:
        text = self.render_report_markdown(document_id, tenant_id)
        lines = [line.replace("|", " ").strip() for line in text.splitlines()]
        stream_lines = []
        y = 780
        for index, line in enumerate(lines[:58]):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            size = 16 if index == 0 else 9
            stream_lines.append(f"BT /F1 {size} Tf 40 {y - index * 12} Td ({escaped}) Tj ET")
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
        for item in objects:
            offsets.append(cursor)
            cursor += len(item) + 1
        xref_start = len(body)
        xref = "xref\n0 6\n0000000000 65535 f \n" + "".join(
            f"{offset:010d} 00000 n \n" for offset in offsets
        )
        trailer = f"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF"
        return (body + xref + trailer).encode("utf-8")
