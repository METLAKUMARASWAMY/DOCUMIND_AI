from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.models.vision import VisionDocumentRecord, VisionReviewTask


class VisionRepository(ABC):
    @abstractmethod
    def save_document(self, document: VisionDocumentRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_document(self, document_id: str) -> VisionDocumentRecord | None:
        raise NotImplementedError

    @abstractmethod
    def list_documents(self) -> list[VisionDocumentRecord]:
        raise NotImplementedError

    @abstractmethod
    def save_job(self, job: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_job(self, job_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def save_review(self, review: VisionReviewTask) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_review(self, review_id: str) -> VisionReviewTask | None:
        raise NotImplementedError

    @abstractmethod
    def resolve_idempotency_key(self, key: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def bind_idempotency_key(self, key: str, job_id: str) -> None:
        raise NotImplementedError


class LocalVisionRepository(VisionRepository):
    def __init__(self) -> None:
        self.documents: dict[str, VisionDocumentRecord] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.reviews: dict[str, VisionReviewTask] = {}
        self.idempotency: dict[str, str] = {}

    def save_document(self, document: VisionDocumentRecord) -> None:
        self.documents[document.document_id] = document

    def get_document(self, document_id: str) -> VisionDocumentRecord | None:
        return self.documents.get(document_id)

    def list_documents(self) -> list[VisionDocumentRecord]:
        return sorted(self.documents.values(), key=lambda item: item.created_at, reverse=True)

    def save_job(self, job: dict[str, Any]) -> None:
        self.jobs[job["job_id"]] = job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    def save_review(self, review: VisionReviewTask) -> None:
        self.reviews[review.review_id] = review

    def get_review(self, review_id: str) -> VisionReviewTask | None:
        return self.reviews.get(review_id)

    def resolve_idempotency_key(self, key: str) -> str | None:
        return self.idempotency.get(key)

    def bind_idempotency_key(self, key: str, job_id: str) -> None:
        self.idempotency[key] = job_id
