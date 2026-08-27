from __future__ import annotations

import re
from importlib.util import find_spec
from typing import Any

from app.core.config import settings
from app.services.vision.worker import VisionProcessingWorker

try:
    from celery import Celery
except ImportError:  # pragma: no cover - exercised through readiness when celery is absent
    Celery = None


class WorkerUnavailableError(RuntimeError):
    pass


def celery_available() -> bool:
    return Celery is not None and find_spec("celery") is not None


def create_celery_app() -> Any:
    if not celery_available():
        raise WorkerUnavailableError(
            "Celery is not installed. Install backend runtime requirements first."
        )
    app = Celery(
        "documind_vision_worker",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["app.workers.vision_tasks"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        broker_connection_retry_on_startup=True,
        task_default_queue=VisionProcessingWorker.queue_name,
        task_routes={"vision.process_job": {"queue": VisionProcessingWorker.queue_name}},
    )
    return app


def celery_readiness() -> dict[str, Any]:
    installed = celery_available()
    if not settings.enable_redis:
        status = "disabled"
        reason = "Set ENABLE_REDIS=true to dispatch Vision jobs through Celery/Redis."
    elif not installed:
        status = "unavailable"
        reason = "Celery package is not installed in the current Python environment."
    else:
        status = "ready_for_connection"
        reason = (
            "Celery app is configured; broker connectivity is checked by deployment smoke tests."
        )
    return {
        "backend_id": "celery_worker",
        "queue": VisionProcessingWorker.queue_name,
        "enabled": settings.enable_redis,
        "installed": installed,
        "status": status,
        "reason": reason,
        "broker": _redact_url(settings.redis_url),
        "task_name": "vision.process_job",
        "shared_state": "var/vision_state disk fallback; MongoDB repository is the production target.",
    }


try:
    celery_app = create_celery_app()
except WorkerUnavailableError:
    celery_app = None


def _redact_url(value: str) -> str:
    return re.sub(r"(?i)(password|pwd|secret|token|key)=([^&]+)", r"\1=****", value or "")
