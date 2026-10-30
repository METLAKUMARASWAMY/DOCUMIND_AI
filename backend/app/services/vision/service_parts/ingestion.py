"""Upload, queueing, job lifecycle and deletion.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.security import hash_document, utc_now_iso
from app.models.schemas import TextAnalysisRequest
from app.models.vision import (
    VisionDocumentRecord,
    VisionProcessingStatus,
    VisionRiskSeverity,
    VisionUploadResponse,
)
from app.services.vision.ingestion_errors import PendingContentUnavailable


class IngestionMixin:
    def queue_documents(
        self,
        files: list[tuple[str, bytes, dict[str, Any]]],
        *,
        actor: str = "system",
        process_inline: bool = False,
        idempotency_key: str | None = None,
        tenant_id: str = "tenant-local",
    ) -> VisionUploadResponse:
        if not files:
            raise ValueError("At least one document is required")
        self._load_indexes()
        scoped_idempotency_key = f"{tenant_id}:{idempotency_key}" if idempotency_key else None
        if scoped_idempotency_key:
            existing_job_id = self._idempotency_index.get(scoped_idempotency_key)
            if existing_job_id:
                self.get_job(existing_job_id)
            if existing_job_id and existing_job_id in self.jobs:
                return self._upload_response_from_job(existing_job_id)
        job_id = f"vision-job-{uuid4().hex[:10]}"
        case_id = f"vision-case-{uuid4().hex[:10]}"
        document_ids: list[str] = []
        for filename, content, metadata in files:
            self.ingestion.validate_file(
                filename,
                content,
                declared_mime=metadata.get("declared_mime"),
            )
            now = utc_now_iso()
            document = VisionDocumentRecord(
                case_id=case_id,
                filename=Path(filename).name or "uploaded-document",
                tenant_id=tenant_id,
                status=VisionProcessingStatus.queued,
                file_hash=hash_document(content),
                created_at=now,
                updated_at=now,
                metadata=metadata,
            )
            self._audit(
                document,
                "document_queued",
                "Document queued for Vision AI processing.",
                actor=actor,
            )
            self.documents[document.document_id] = document
            self._pending_content[document.document_id] = (filename, content, metadata)
            source_uri = self.persistence.store_original(
                document,
                filename,
                content,
                metadata.get("declared_mime"),
            )
            if source_uri:
                document.metadata["source_object_uri"] = source_uri
            self._persist_document(document)
            self._persist_pending_content(document.document_id, filename, content, metadata)
            document_ids.append(document.document_id)
        self.jobs[job_id] = {
            "job_id": job_id,
            "case_id": case_id,
            "document_ids": document_ids,
            "status": VisionProcessingStatus.queued.value,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "errors": [],
            "idempotency_key": idempotency_key,
            "tenant_id": tenant_id,
        }
        if scoped_idempotency_key:
            self._idempotency_index[scoped_idempotency_key] = job_id
            self._persist_indexes()
        self._persist_job(self.jobs[job_id])
        if process_inline:
            self.process_job(job_id)
        return self._upload_response_from_job(job_id)

    def process_text(
        self,
        request: TextAnalysisRequest,
        *,
        actor: str = "system",
        idempotency_key: str | None = None,
        tenant_id: str = "tenant-local",
    ) -> VisionDocumentRecord:
        response = self.queue_documents(
            [(request.filename, request.text.encode("utf-8"), request.metadata)],
            actor=actor,
            process_inline=True,
            idempotency_key=idempotency_key,
            tenant_id=tenant_id,
        )
        return self.get_document(response.document_ids[0])

    def queue_reprocess(
        self,
        document_id: str,
        content: bytes,
        *,
        actor: str,
        tenant_id: str,
    ) -> VisionUploadResponse:
        document = self.get_document(document_id, tenant_id)
        retained_metadata = {
            key: value
            for key, value in document.metadata.items()
            if key
            not in {
                "classification",
                "schema_selection",
                "extraction",
                "extraction_diagnostics",
                "pipeline_status",
                "decision_status",
                "status_summary",
                "legacy_flags",
            }
        }
        reprocessing_history = list(retained_metadata.get("reprocessing_history") or [])
        reprocessing_history.append(
            {
                "status": document.status.value,
                "updated_at": document.updated_at,
                "review": (
                    {
                        "review_id": document.review_task.review_id,
                        "status": document.review_task.status.value,
                        "reviewer_comments": document.review_task.reviewer_comments,
                        "history_event_ids": [
                            event.event_id for event in document.review_task.history
                        ],
                    }
                    if document.review_task
                    else None
                ),
                "data_version": document.data_version,
                "pipeline_version": document.metadata.get("pipeline_version", "legacy"),
            }
        )
        metadata = {
            **retained_metadata,
            "reprocessed_from": document.document_id,
            "reprocess_count": int(document.metadata.get("reprocess_count", 0)) + 1,
            "reprocessing_history": reprocessing_history,
        }
        if document.review_task:
            metadata["_reprocess_review_id"] = document.review_task.review_id
        stale_artifacts = self._mark_derived_artifacts_stale(document, actor=actor)
        metadata["stale_derived_artifacts"] = stale_artifacts
        self._reset_for_reprocess(document, metadata, actor)

        job_id = f"vision-job-{uuid4().hex[:10]}"
        now = utc_now_iso()
        self.jobs[job_id] = {
            "job_id": job_id,
            "case_id": document.case_id,
            "document_ids": [document.document_id],
            "status": VisionProcessingStatus.queued.value,
            "created_at": now,
            "updated_at": now,
            "errors": [],
            "idempotency_key": None,
            "tenant_id": tenant_id,
            "reprocess": True,
        }
        self._pending_content[document.document_id] = (
            document.filename,
            content,
            metadata,
        )
        self._persist_pending_content(
            document.document_id,
            document.filename,
            content,
            metadata,
        )
        self._persist_document(document)
        self._persist_job(self.jobs[job_id])
        return self._upload_response_from_job(job_id)

    def process_job(self, job_id: str, tenant_id: str | None = None) -> None:
        job = self.get_job(job_id, tenant_id)
        if not job:
            return
        for document_id in job["document_ids"]:
            if document_id not in self._pending_content:
                self._load_pending_content(document_id)
            if document_id not in self.documents:
                self._load_document(document_id)
        if not any(document_id in self._pending_content for document_id in job["document_ids"]):
            # Pending content is consumed once a document is processed, so a
            # re-run of a finished job legitimately finds none: that is
            # idempotent, not a failure.
            unfinished = [
                document_id
                for document_id in job["document_ids"]
                if (document := self.documents.get(document_id)) is None
                or document.status
                in {
                    VisionProcessingStatus.queued,
                    VisionProcessingStatus.preprocessing,
                }
            ]
            if not unfinished:
                return
            # Still-queued documents with no readable upload is the real fault.
            # Returning here left them in QUEUED for ever: no error, no status
            # change, and the Celery task reported success, so nothing retried
            # and the console showed the upload processing indefinitely. Raise
            # so the task's retry policy re-runs it once the upload has landed,
            # and so a genuine failure becomes visible.
            raise PendingContentUnavailable(job_id, unfinished)
        job["status"] = VisionProcessingStatus.preprocessing.value
        job["updated_at"] = utc_now_iso()
        self._persist_job(job)
        for document_id in job["document_ids"]:
            pending = self._pending_content.get(document_id)
            document = self.documents.get(document_id)
            if not pending or not document:
                continue
            filename, content, metadata = pending
            success = False
            try:
                self._process_document(document, filename, content, metadata)
                success = True
            except Exception as exc:  # pragma: no cover - defensive safety net
                document.status = VisionProcessingStatus.review_required
                document.updated_at = utc_now_iso()
                timing = dict(document.metadata.get("processing_timing") or {})
                processing_started_at = timing.get("processing_started_at")
                processing_completed_at = utc_now_iso()
                processing_duration_ms: float | None = None
                if processing_started_at:
                    try:
                        processing_duration_ms = max(
                            0.0,
                            (
                                datetime.fromisoformat(processing_completed_at)
                                - datetime.fromisoformat(str(processing_started_at))
                            ).total_seconds()
                            * 1000,
                        )
                    except (TypeError, ValueError):
                        processing_duration_ms = None
                document.metadata["processing_timing"] = {
                    **timing,
                    "processing_completed_at": processing_completed_at,
                    "processing_duration_ms": (
                        round(processing_duration_ms, 3)
                        if processing_duration_ms is not None
                        else None
                    ),
                    "reliable": False,
                    "exclusion_reason": "processing_failed",
                }
                warning = self._risk(
                    document,
                    title="Processing Failure",
                    severity=VisionRiskSeverity.high,
                    category="processing",
                    description="The processing pipeline failed before a complete result was available.",
                    recommended_action="Retry processing or assign a reviewer.",
                    source_text=str(exc),
                )
                document.processing_warnings.append(warning)
                self._audit(
                    document,
                    "processing_failed",
                    "Vision AI processing failed and was routed to review.",
                    metadata={"error": str(exc)},
                )
                if document.review_task is None:
                    document.review_task = self._create_review_task(document)
                    self.review_tasks[document.review_task.review_id] = document.review_task
                job["errors"].append({"document_id": document_id, "error": str(exc)})
            finally:
                if success:
                    self._pending_content.pop(document_id, None)
                    self._remove_pending_content(document_id)
                if document:
                    self._persist_document(document)
                    if document.review_task:
                        self._persist_review(document.review_task)
        job["status"] = (
            VisionProcessingStatus.completed.value
            if not job["errors"]
            else VisionProcessingStatus.review_required.value
        )
        processed_documents = [
            self.documents[document_id]
            for document_id in job["document_ids"]
            if document_id in self.documents
        ]
        decision_statuses = sorted({document.status.value for document in processed_documents})
        job["pipeline_status"] = (
            VisionProcessingStatus.completed.value
            if not job["errors"]
            else VisionProcessingStatus.review_required.value
        )
        job["decision_status"] = (
            VisionProcessingStatus.review_required.value
            if any(
                document.status == VisionProcessingStatus.review_required
                for document in processed_documents
            )
            else decision_statuses[0]
            if len(decision_statuses) == 1
            else "MIXED"
        )
        job["status_summary"] = (
            "Processing completed - awaiting human review"
            if job["decision_status"] == VisionProcessingStatus.review_required.value
            and not job["errors"]
            else "Processing completed"
            if not job["errors"]
            else "Processing requires technical review"
        )
        job["extraction_outcomes"] = [
            {
                "document_id": document.document_id,
                **dict(document.metadata.get("extraction") or {}),
            }
            for document in processed_documents
        ]
        job["updated_at"] = utc_now_iso()
        self._persist_job(job)
        self._persist_indexes()

    def _upload_response_from_job(self, job_id: str) -> VisionUploadResponse:
        job = self.get_job(job_id) or self.jobs[job_id]
        return VisionUploadResponse(
            case_id=job["case_id"],
            job_id=job_id,
            document_ids=list(job["document_ids"]),
            status=VisionProcessingStatus(job["status"]),
            status_url=f"/api/v1/vision/jobs/{job_id}",
        )

    def get_job(self, job_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
        # API and Celery worker are separate processes.  Prefer the durable job
        # snapshot so a queued in-memory object cannot mask worker progress.
        job = self._load_job(job_id) or self.jobs.get(job_id)
        if not job and tenant_id:
            job = self.persistence.get_job(job_id, tenant_id)
            if job:
                self.jobs[job_id] = job
        if job and tenant_id and job.get("tenant_id", "tenant-local") != tenant_id:
            return None
        return job

    def retry_job(
        self,
        job_id: str,
        tenant_id: str | None = None,
        *,
        actor: str = "system",
    ) -> dict[str, Any]:
        job = self.get_job(job_id, tenant_id)
        if not job:
            raise ValueError("job_id not found")
        job["errors"] = []
        job["status"] = VisionProcessingStatus.queued.value
        job["updated_at"] = utc_now_iso()
        self._persist_job(job)
        self._system_audit(
            "job_retry_queued",
            "Queued a failed or incomplete processing job for retry.",
            actor=actor,
            tenant_id=str(job.get("tenant_id") or tenant_id or "tenant-local"),
            metadata={"job_id": job_id, "document_ids": list(job.get("document_ids") or [])},
        )
        return job

    def record_job_dispatch(
        self,
        job_id: str,
        dispatch: dict[str, Any],
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        job = self.get_job(job_id, tenant_id)
        if not job:
            return None
        job["dispatch"] = dispatch
        job["updated_at"] = utc_now_iso()
        self._persist_job(job)
        return job

    def delete_document(
        self,
        document_id: str,
        tenant_id: str | None = None,
        *,
        actor: str = "system",
    ) -> VisionDocumentRecord:
        document = self.get_document(document_id, tenant_id)
        self.documents.pop(document_id, None)
        self._document_tenant_index.pop(document_id, None)
        self._remove_document(document_id)
        self._pending_content.pop(document_id, None)
        self._remove_pending_content(document_id)
        if document.review_task:
            self.review_tasks.pop(document.review_task.review_id, None)
            self._remove_review(document.review_task.review_id)
        self._hash_index = {
            key: value for key, value in self._hash_index.items() if value != document_id
        }
        self._semantic_index = {
            key: value for key, value in self._semantic_index.items() if value != document_id
        }
        self._persist_indexes()
        self.persistence.delete_document(document_id, document.tenant_id)
        self._system_audit(
            "document_deleted",
            "Deleted a document and its local derived state.",
            actor=actor,
            tenant_id=document.tenant_id,
            metadata={
                "document_id": document.document_id,
                "case_id": document.case_id,
                "filename": document.filename,
            },
        )
        return document
