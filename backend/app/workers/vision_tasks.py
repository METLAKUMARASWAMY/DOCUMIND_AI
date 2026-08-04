from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.services.vision import vision_ai_service
from app.services.vision.worker import VisionProcessingWorker
from app.workers.celery_app import celery_app, celery_readiness


def _process_vision_job(job_id: str, tenant_id: str | None = None) -> dict[str, Any]:
    worker = VisionProcessingWorker(vision_ai_service)
    return worker.process(job_id, tenant_id).__dict__


if celery_app is not None:

    @celery_app.task(
        name="vision.process_job",
        autoretry_for=(Exception,),
        retry_backoff=True,
        retry_jitter=True,
        max_retries=3,
    )
    def process_vision_job(job_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        return _process_vision_job(job_id, tenant_id)

else:

    def process_vision_job(job_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        return _process_vision_job(job_id, tenant_id)


def dispatch_vision_job(job_id: str, tenant_id: str | None = None) -> dict[str, Any]:
    readiness = celery_readiness()
    if settings.enable_redis and celery_app is not None and hasattr(process_vision_job, "apply_async"):
        try:
            result = process_vision_job.apply_async(
                args=[job_id, tenant_id],
                queue=VisionProcessingWorker.queue_name,
            )
            return {
                "mode": "celery",
                "queue": VisionProcessingWorker.queue_name,
                "task_id": result.id,
                "status": "queued",
            }
        except Exception as exc:  # pragma: no cover - broker failure fallback
            return {
                "mode": "fastapi_background_task",
                "queue": VisionProcessingWorker.queue_name,
                "task_id": None,
                "status": "fallback",
                "fallback_reason": f"Celery dispatch failed: {type(exc).__name__}",
            }
    return {
        "mode": "fastapi_background_task",
        "queue": VisionProcessingWorker.queue_name,
        "task_id": None,
        "status": readiness["status"],
        "fallback_reason": readiness["reason"],
    }
