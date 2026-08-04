from __future__ import annotations

from dataclasses import dataclass

from app.core.security import utc_now_iso


@dataclass(frozen=True)
class VisionWorkerResult:
    job_id: str
    status: str
    processed_at: str
    errors: list[dict]


class VisionProcessingWorker:
    """Small local worker facade matching the future Celery task contract."""

    queue_name = "vision-processing"
    retryable_stages = (
        "PREPROCESSING",
        "OCR_COMPLETED",
        "EXTRACTION_COMPLETED",
        "VALIDATION_COMPLETED",
        "INDEXED",
        "REVIEW_REQUIRED",
    )

    def __init__(self, service) -> None:
        self.service = service

    def process(self, job_id: str, tenant_id: str | None = None) -> VisionWorkerResult:
        self.service.process_job(job_id, tenant_id)
        job = self.service.get_job(job_id, tenant_id) or {"status": "missing", "errors": []}
        return VisionWorkerResult(
            job_id=job_id,
            status=job.get("status", "missing"),
            processed_at=utc_now_iso(),
            errors=job.get("errors", []),
        )

    def retry(self, job_id: str, tenant_id: str | None = None) -> VisionWorkerResult:
        job = self.service.get_job(job_id, tenant_id)
        if job:
            job["errors"] = []
            job["status"] = "QUEUED"
        return self.process(job_id, tenant_id)
