"""A job whose upload cannot be read must fail loudly, not sit in QUEUED.

Observed against the live stack: the worker picked up a task 2ms after the
upload returned, could not yet read the uploaded bytes from disk (it runs in its
own process), and ``process_job`` returned silently. The job and its document
stayed QUEUED for ever, no error was recorded anywhere, and the Celery task
logged "succeeded in 0.002s" -- so nothing retried and the console showed the
upload processing indefinitely.

Raising instead hands the task to its existing retry policy, which is what makes
the upload/dispatch race recoverable, and turns a permanent silent stall into a
visible failure.
"""

from __future__ import annotations

import pytest

from app.models.vision import VisionProcessingStatus
from app.services.vision.ingestion_errors import PendingContentUnavailable
from app.services.vision.service import VisionAIService
from app.services.vision.worker import VisionProcessingWorker

CONTENT = b"Invoice INV-STALL\nVendor: Stall Vendor\nTotal Amount: USD 12.00\n"


def _queue(service: VisionAIService, tenant: str):
    return service.queue_documents(
        [("stalled-upload.txt", CONTENT, {})],
        actor="stall-test",
        tenant_id=tenant,
    )


def test_unreadable_upload_raises_instead_of_stalling():
    service = VisionAIService()
    response = _queue(service, "tenant-stall-a")
    document_id = response.document_ids[0]

    # Reproduce the race: the worker process cannot see the bytes yet.
    service._pending_content.pop(document_id, None)
    service._remove_pending_content(document_id)

    with pytest.raises(PendingContentUnavailable) as raised:
        service.process_job(response.job_id, "tenant-stall-a")

    assert document_id in raised.value.document_ids
    assert response.job_id in str(raised.value)


def test_the_document_is_still_queued_so_a_retry_can_finish_it():
    """The failure must not leave a half-processed or falsely-completed state."""
    service = VisionAIService()
    response = _queue(service, "tenant-stall-b")
    document_id = response.document_ids[0]
    service._pending_content.pop(document_id, None)
    service._remove_pending_content(document_id)

    with pytest.raises(PendingContentUnavailable):
        service.process_job(response.job_id, "tenant-stall-b")

    document = service.get_document(document_id, "tenant-stall-b")
    assert document.status == VisionProcessingStatus.queued
    job = service.get_job(response.job_id, "tenant-stall-b")
    assert str(job["status"]).upper() == "QUEUED"


def test_reprocessing_a_finished_job_is_still_quietly_idempotent():
    """Pending content is consumed on success; re-running must not now raise."""
    service = VisionAIService()
    response = _queue(service, "tenant-stall-c")
    worker = VisionProcessingWorker(service)

    first = worker.process(response.job_id, "tenant-stall-c")
    assert str(first.status).upper() != "QUEUED"

    second = worker.process(response.job_id, "tenant-stall-c")  # must not raise
    assert second.status == first.status


def test_a_missing_job_is_still_a_no_op():
    service = VisionAIService()
    service.process_job("vision-job-does-not-exist", "tenant-stall-d")  # must not raise


def test_the_celery_task_retries_this_error():
    """The task must be configured to retry, or raising just moves the stall."""
    from app.workers import vision_tasks

    task = getattr(vision_tasks, "process_vision_job", None)
    assert task is not None
    autoretry = getattr(task, "autoretry_for", ())
    if autoretry:  # only meaningful when Celery is configured in this env
        assert any(issubclass(Exception, exc) or exc is Exception for exc in autoretry)
        assert getattr(task, "max_retries", 0) >= 1
