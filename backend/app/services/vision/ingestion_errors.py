"""Errors the ingestion path raises so a stalled job cannot pass as a success."""

from __future__ import annotations


class PendingContentUnavailable(RuntimeError):
    """The uploaded bytes for a queued job could not be read.

    The worker runs in its own process, so it reads the upload from disk rather
    than from the API process's memory. Dispatch can win that race: a task was
    observed being picked up 2ms after the upload returned, before the content
    was readable, and ``process_job`` then returned silently -- leaving the job
    and the document in QUEUED for ever, with no error recorded, while Celery
    logged the task as succeeded. Raising instead lets the task's existing retry
    policy re-run it once the write has landed, and makes a genuine failure
    visible rather than silent.
    """

    def __init__(self, job_id: str, document_ids: list[str]) -> None:
        self.job_id = job_id
        self.document_ids = document_ids
        super().__init__(
            f"No uploaded content is readable for job {job_id} "
            f"(documents: {', '.join(document_ids) or 'none'})"
        )
