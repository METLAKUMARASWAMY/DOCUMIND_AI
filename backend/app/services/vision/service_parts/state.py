"""On-disk state: documents, jobs, reviews, indexes and derived artifacts.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.security import hash_document
from app.models.vision import (
    ReviewerStatus,
    VisionDocumentRecord,
    VisionFinanceNarrative,
    VisionReportRecord,
    VisionReviewTask,
)


class StateMixin:
    def _ensure_state_dirs(self) -> None:
        for child in (
            "documents",
            "jobs",
            "reviews",
            "pending",
            "questions",
            "conversations",
            "narratives",
            "reports",
        ):
            (self._state_root / child).mkdir(parents=True, exist_ok=True)

    def _state_file(self, collection: str, item_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", item_id)
        return self._state_root / collection / f"{safe_id}.json"

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _persist_document(self, document: VisionDocumentRecord) -> None:
        self._document_tenant_index[document.document_id] = document.tenant_id
        self.persistence.save_document(document)
        self._write_json(
            self._state_file("documents", document.document_id),
            document.model_dump(mode="json"),
        )

    def _load_document(
        self,
        document_id: str,
        *,
        tenant_id: str | None = None,
    ) -> VisionDocumentRecord | None:
        payload = self._read_json(self._state_file("documents", document_id))
        if not payload:
            return None
        payload_tenant = str(payload.get("tenant_id") or "tenant-local")
        self._document_tenant_index[document_id] = payload_tenant
        if tenant_id and payload_tenant != tenant_id:
            return None
        cached = self.documents.get(document_id)
        if (
            cached is not None
            and cached.tenant_id == payload_tenant
            and cached.updated_at == payload.get("updated_at")
            and cached.data_version == int(payload.get("data_version", 1))
        ):
            return cached
        try:
            document = VisionDocumentRecord.model_validate(payload)
        except ValueError:
            return None
        self.documents[document.document_id] = document
        if document.review_task:
            self.review_tasks[document.review_task.review_id] = document.review_task
        self._index_document(document)
        return document

    def _load_all_documents(self, tenant_id: str | None = None) -> None:
        index_changed = False
        for path in (self._state_root / "documents").glob("*.json"):
            document_id = path.stem
            cached = self.documents.get(document_id)
            if cached is not None:
                self._document_tenant_index[document_id] = cached.tenant_id
                if tenant_id and cached.tenant_id != tenant_id:
                    continue
                self._load_document(document_id, tenant_id=tenant_id)
                continue
            known_tenant = self._document_tenant_index.get(document_id)
            if tenant_id and known_tenant and known_tenant != tenant_id:
                continue
            previous_tenant = known_tenant
            self._load_document(document_id, tenant_id=tenant_id)
            index_changed = index_changed or (
                previous_tenant != self._document_tenant_index.get(document_id)
            )
        if index_changed:
            self._persist_indexes()

    def _remove_document(self, document_id: str) -> None:
        self.retrieval_index.remove_document(document_id)
        self._state_file("documents", document_id).unlink(missing_ok=True)

    def _persist_job(self, job: dict[str, Any]) -> None:
        self.jobs[job["job_id"]] = job
        self._write_json(self._state_file("jobs", job["job_id"]), job)
        self.persistence.save_job(job)

    def _load_job(self, job_id: str) -> dict[str, Any] | None:
        payload = self._read_json(self._state_file("jobs", job_id))
        if not payload:
            return None
        self.jobs[job_id] = payload
        return payload

    def _persist_review(self, review: VisionReviewTask) -> None:
        self.review_tasks[review.review_id] = review
        self._write_json(
            self._state_file("reviews", review.review_id), review.model_dump(mode="json")
        )
        document = self.documents.get(review.document_id)
        if document:
            self.persistence.save_review(review, document.tenant_id)

    def _load_review(self, review_id: str) -> VisionReviewTask | None:
        payload = self._read_json(self._state_file("reviews", review_id))
        if not payload:
            return None
        try:
            review = VisionReviewTask.model_validate(payload)
        except ValueError:
            return None
        self.review_tasks[review.review_id] = review
        return review

    def _load_all_reviews(self) -> None:
        for path in (self._state_root / "reviews").glob("*.json"):
            review_id = path.stem
            if review_id not in self.review_tasks:
                self._load_review(review_id)

    def _remove_review(self, review_id: str) -> None:
        self._state_file("reviews", review_id).unlink(missing_ok=True)

    def _persist_pending_content(
        self,
        document_id: str,
        filename: str,
        content: bytes,
        metadata: dict[str, Any],
    ) -> None:
        pending_dir = self._state_root / "pending" / document_id
        pending_dir.mkdir(parents=True, exist_ok=True)
        (pending_dir / "content.bin").write_bytes(content)
        self._write_json(
            pending_dir / "metadata.json", {"filename": filename, "metadata": metadata}
        )

    def _load_pending_content(self, document_id: str) -> tuple[str, bytes, dict[str, Any]] | None:
        pending_dir = self._state_root / "pending" / document_id
        metadata = self._read_json(pending_dir / "metadata.json")
        content_path = pending_dir / "content.bin"
        if not metadata or not content_path.exists():
            return None
        try:
            pending = (
                str(metadata.get("filename") or "uploaded-document"),
                content_path.read_bytes(),
                dict(metadata.get("metadata") or {}),
            )
        except OSError:
            return None
        self._pending_content[document_id] = pending
        return pending

    def _remove_pending_content(self, document_id: str) -> None:
        pending_dir = self._state_root / "pending" / document_id
        for filename in ("content.bin", "metadata.json"):
            (pending_dir / filename).unlink(missing_ok=True)
        try:
            pending_dir.rmdir()
        except OSError:
            pass

    def _persist_question(self, question_id: str, question: dict[str, Any]) -> None:
        self.questions[question_id] = question
        self._write_json(self._state_file("questions", question_id), question)
        self.persistence.save_question(question)

    def _persist_finance_narrative(
        self,
        narrative: VisionFinanceNarrative,
    ) -> None:
        self.finance_narratives[narrative.narrative_id] = narrative
        self._write_json(
            self._state_file("narratives", narrative.narrative_id),
            narrative.model_dump(mode="json"),
        )
        self.persistence.save_narrative(narrative)

    def _mark_derived_artifacts_stale(
        self,
        document: VisionDocumentRecord,
        *,
        actor: str,
    ) -> dict[str, list[str]]:
        """Invalidate outputs whose evidence snapshot predates a reprocess.

        The records remain durable for audit/version history, but normal listings
        no longer present them as current evidence-backed outputs.
        """
        stale_narratives: list[str] = []
        stale_reports: list[str] = []
        for narrative in self.list_finance_narratives(document.tenant_id):
            source_ids = narrative.source_document_ids or narrative.document_ids
            if document.document_id not in source_ids:
                continue
            narrative.status = "stale"
            self._persist_finance_narrative(narrative)
            stale_narratives.append(narrative.narrative_id)
        for report in self.list_reports(document.tenant_id):
            if document.document_id not in report.document_ids:
                continue
            report.status = "stale"
            self._persist_report_record(report)
            stale_reports.append(report.report_id)
        if stale_narratives or stale_reports:
            self._audit(
                document,
                "derived_artifacts_invalidated",
                "Marked evidence-derived narratives and reports stale before reprocessing.",
                actor=actor,
                metadata={
                    "narrative_ids": stale_narratives,
                    "report_ids": stale_reports,
                    "source_data_version": document.data_version,
                },
            )
        return {
            "narrative_ids": stale_narratives,
            "report_ids": stale_reports,
        }

    def _persist_report_record(self, report: VisionReportRecord) -> None:
        self.reports[report.report_id] = report
        self._write_json(
            self._state_file("reports", report.report_id),
            report.model_dump(mode="json"),
        )
        self.persistence.save_report(report.model_dump(mode="json"))

    def _artifact_source_data_version(
        self,
        documents: list[VisionDocumentRecord],
    ) -> str:
        snapshot = [
            {
                "document_id": document.document_id,
                "data_version": document.data_version,
                "status": document.status.value,
                "review_status": (
                    document.review_task.status.value
                    if document.review_task
                    else document.status.value
                ),
                "verified_fields": sorted(
                    (
                        field.name,
                        field.reviewed_value or field.value,
                        field.reviewer_status.value,
                    )
                    for field in document.extracted_fields
                    if field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
                ),
            }
            for document in sorted(documents, key=lambda item: item.document_id)
        ]
        return self._artifact_identity_hash(snapshot)

    def _artifact_identity_hash(self, payload: Any) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hash_document(encoded)

    def _load_question(self, question_id: str) -> dict[str, Any] | None:
        payload = self._read_json(self._state_file("questions", question_id))
        if payload:
            self.questions[question_id] = payload
            self.question_feedback[question_id] = list(payload.get("feedback") or [])
        return payload

    def _persist_indexes(self) -> None:
        self._write_json(
            self._state_root / "indexes.json",
            {
                "idempotency": self._idempotency_index,
                "hash": self._hash_index,
                "semantic": self._semantic_index,
                "document_tenants": self._document_tenant_index,
            },
        )

    def _load_indexes(self) -> None:
        payload = self._read_json(self._state_root / "indexes.json") or {}
        self._idempotency_index.update(dict(payload.get("idempotency") or {}))
        self._hash_index.update(dict(payload.get("hash") or {}))
        self._semantic_index.update(dict(payload.get("semantic") or {}))
        self._document_tenant_index.update(dict(payload.get("document_tenants") or {}))
