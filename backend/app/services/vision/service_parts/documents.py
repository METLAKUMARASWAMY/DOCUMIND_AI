"""Document lookup, listing, scoping and index membership.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.config import settings
from app.models.vision import (
    VisionDocumentRecord,
)
from app.services.vision.analytics import vision_analytics_service


class DocumentMixin:
    def get_document(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> VisionDocumentRecord:
        # Reload the shared durable snapshot first.  Without this, the API keeps
        # returning its pre-queue object after a Celery worker has completed the
        # same document in another process.
        document = self._load_document(document_id, tenant_id=tenant_id) or self.documents.get(
            document_id
        )
        if not document and tenant_id:
            document = self.persistence.get_document(document_id, tenant_id)
            if document:
                self.documents[document.document_id] = document
                self._index_document(document)
        if not document or (tenant_id and document.tenant_id != tenant_id):
            raise ValueError("document_id not found")
        vision_analytics_service.refresh_document(document)
        self._ensure_review_task(document)
        vision_analytics_service.refresh_document(document)
        return document

    def list_documents(self, tenant_id: str | None = None) -> list[VisionDocumentRecord]:
        self._load_all_documents(tenant_id)
        if tenant_id:
            for document in self.persistence.list_documents(tenant_id):
                current = self.documents.get(document.document_id)
                if current is None or (
                    current.updated_at,
                    current.data_version,
                ) != (document.updated_at, document.data_version):
                    self.documents[document.document_id] = document
                    self._document_tenant_index[document.document_id] = document.tenant_id
                    self._index_document(document)
        documents = self.documents.values()
        if tenant_id:
            documents = [item for item in documents if item.tenant_id == tenant_id]
        for document in list(documents):
            vision_analytics_service.refresh_document(document)
            self._ensure_review_task(document)
            vision_analytics_service.refresh_document(document)
        return sorted(documents, key=lambda item: item.created_at, reverse=True)

    def _documents_for_question(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> list[VisionDocumentRecord]:
        normalized_question = self._normalize(question)
        identity_fields = {
            "invoice_number",
            "purchase_order_number",
            "contract_number",
            "vendor_name",
        }
        matches = [
            document
            for document in documents
            if any(
                field.name in identity_fields
                and len(self._normalize(field.reviewed_value or field.value)) >= 3
                and self._normalize(field.reviewed_value or field.value) in normalized_question
                for field in document.extracted_fields
            )
        ]
        return matches or documents

    def _index_document(self, document: VisionDocumentRecord) -> None:
        self._hash_index.setdefault(
            f"{document.tenant_id}:{document.file_hash}",
            document.document_id,
        )
        semantic_key = self._semantic_key(document)
        if semantic_key:
            self._semantic_index.setdefault(
                f"{document.tenant_id}:{semantic_key}",
                document.document_id,
            )
        self.retrieval_index.upsert_document(document.document_id, document.evidence_chunks)

    def _semantic_key(self, document: VisionDocumentRecord) -> str:
        fields = self._field_map(document)
        parts = [
            fields.get("vendor_name", ""),
            fields.get("invoice_number", ""),
            fields.get("purchase_order_number", ""),
            fields.get("invoice_date", ""),
            fields.get("total_amount", ""),
        ]
        normalized = [self._normalize(part) for part in parts if part]
        return "|".join(normalized)

    def _structured_duplicate_similarity(
        self,
        document: VisionDocumentRecord,
        candidate: VisionDocumentRecord | None,
    ) -> tuple[float, list[str], int]:
        if candidate is None:
            return 0.0, [], 0
        left = self._field_map(document)
        right = self._field_map(candidate)
        weights = {
            "vendor_name": 1.5,
            "invoice_number": 2.0,
            "total_amount": 1.5,
            "invoice_date": 1.0,
            "purchase_order_number": 1.5,
        }
        compared = [name for name in weights if left.get(name) and right.get(name)]
        if not compared:
            return 0.0, [], 0
        matches = [
            name for name in compared if self._normalize(left[name]) == self._normalize(right[name])
        ]
        denominator = sum(weights[name] for name in compared)
        score = sum(weights[name] for name in matches) / denominator
        return score, matches, len(compared)

    def _related_document(
        self,
        document: VisionDocumentRecord,
    ) -> tuple[VisionDocumentRecord, float] | None:
        self._load_all_documents(document.tenant_id)
        tokens = set(re.findall(r"[a-z0-9]+", document.parsed_text.lower()))
        if not tokens:
            return None
        best: tuple[VisionDocumentRecord, float] | None = None
        for candidate in self.documents.values():
            if (
                candidate.document_id == document.document_id
                or candidate.tenant_id != document.tenant_id
                or not candidate.parsed_text
            ):
                continue
            other_tokens = set(re.findall(r"[a-z0-9]+", candidate.parsed_text.lower()))
            if not other_tokens:
                continue
            similarity = len(tokens & other_tokens) / max(1, len(tokens | other_tokens))
            if similarity >= settings.related_document_threshold and (
                best is None or similarity > best[1]
            ):
                best = (candidate, similarity)
        return best

    def _document_summary(self, document: VisionDocumentRecord) -> str:
        return (
            f"Processed {document.filename} as {document.document_type.value} with "
            f"{len(document.ocr_words)} OCR words, {len(document.extracted_fields)} fields, "
            f"{len(document.tables)} table(s), {len(document.visual_detections)} visual cue(s), "
            f"and {len(document.risk_indicators)} review indicator(s)."
        )

    def _backend_status(self, registry: dict[str, Any], model_id: str) -> str:
        for backend in registry.get("backends", []):
            if backend["model_id"] == model_id or model_id.lower() in backend["model_id"].lower():
                return backend["status"]
        return "unregistered"
