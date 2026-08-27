"""Construction of fields, evidence, audit events and model runs.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

from typing import Any

from app.core.security import hash_document, utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    ModelRunRecord,
    OCRWord,
    VisionAuditEvent,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionEvidence,
    VisionProcessingStatus,
)
from app.services.vision.schema_registry import vision_schema_registry


class ProvenanceMixin:
    def _field(
        self, document: VisionDocumentRecord, name: str, value: str, source_text: str
    ) -> ExtractedVisionField:
        box = self._box_for_text(value, document.ocr_words) or self._box_for_text(
            source_text, document.ocr_words
        )
        confidence, components = self._field_confidence(value, document.ocr_words, box)
        evidence = self._evidence(
            document,
            source_text,
            "local-schema-extraction-adapter",
            confidence,
            bounding_box=box,
            metadata={
                "confidence_basis": "weighted_observed_components_not_calibrated_model_accuracy",
                "confidence_components": components,
                "confidence_policy": self.confidence_policy.get("calibration", {}),
            },
        )
        return ExtractedVisionField(
            name=name,
            schema_field=name,
            display_name=(
                schema_field.display_name
                if (
                    schema_field := (
                        vision_schema_registry.get(document.document_type).field(name)
                        if vision_schema_registry.get(document.document_type)
                        else None
                    )
                )
                else name.replace("_", " ").title()
            ),
            value=value,
            normalized_value=self._normalize(value),
            confidence=confidence,
            combined_confidence=confidence,
            ocr_confidence=components.get("ocr_confidence"),
            extraction_confidence=components.get("extraction_confidence"),
            validation_confidence=components.get("field_format_validity"),
            evidence=evidence,
            confidence_components=components,
            calibrated=bool(self.confidence_policy.get("calibration", {}).get("calibrated")),
            extractor="deterministic-fallback",
            required=bool(schema_field and schema_field.required),
        )

    def _evidence(
        self,
        document: VisionDocumentRecord,
        source_text: str,
        model_name: str,
        confidence: float,
        *,
        bounding_box: VisionBoundingBox | None = None,
        page_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VisionEvidence:
        matched_words = self._words_for_text(source_text, document.ocr_words)
        resolved_page = matched_words[0].page if matched_words else (page_number or 1)
        resolved_source = (
            source_text or document.parsed_text[:400] or "No source text available."
        )[:700]
        resolved_box = bounding_box or self._box_for_text(
            source_text,
            document.ocr_words,
        )
        box_identity = (
            ",".join(
                f"{value:.3f}"
                for value in (
                    resolved_box.x1,
                    resolved_box.y1,
                    resolved_box.x2,
                    resolved_box.y2,
                )
            )
            if resolved_box
            else "no-box"
        )
        evidence_identity = hash_document(
            (
                f"{document.document_id}|{resolved_page}|{model_name}|"
                f"{self._normalize(resolved_source)}|{box_identity}"
            ).encode()
        )
        return VisionEvidence(
            evidence_id=f"vis-ev-{evidence_identity[:12]}",
            document_id=document.document_id,
            page_number=resolved_page,
            source_text=resolved_source,
            bounding_box=resolved_box,
            visual_region=f"page_{resolved_page}",
            model_confidence=max(0.0, min(1.0, confidence)),
            model_name=model_name,
            model_version="v1",
            processing_timestamp=utc_now_iso(),
            metadata=metadata or {},
        )

    def _word_evidence(self, document: VisionDocumentRecord, word: OCRWord) -> VisionEvidence:
        box_identity = ",".join(
            f"{value:.3f}"
            for value in (
                word.bounding_box.x1,
                word.bounding_box.y1,
                word.bounding_box.x2,
                word.bounding_box.y2,
            )
        )
        evidence_identity = hash_document(
            (
                f"{document.document_id}|{word.page}|{word.reading_order}|"
                f"{word.model_name}|{self._normalize(word.text)}|{box_identity}"
            ).encode()
        )
        return VisionEvidence(
            evidence_id=f"vis-ev-{evidence_identity[:12]}",
            document_id=document.document_id,
            page_number=word.page,
            source_text=word.text,
            bounding_box=word.bounding_box,
            visual_region=word.line_id,
            model_confidence=word.confidence,
            model_name=word.model_name,
            model_version=word.model_version,
            processing_timestamp=utc_now_iso(),
            metadata={"reading_order": word.reading_order, "low_confidence": word.review_required},
        )

    def _audit(
        self,
        document: VisionDocumentRecord,
        event_type: str,
        message: str,
        *,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> VisionAuditEvent:
        event = VisionAuditEvent(
            document_id=document.document_id,
            actor=actor,
            event_type=event_type,
            message=message,
            created_at=utc_now_iso(),
            metadata=metadata or {},
        )
        document.audit_events.append(event)
        document.updated_at = event.created_at
        return event

    def _transition(
        self, document: VisionDocumentRecord, status: VisionProcessingStatus, message: str
    ) -> None:
        document.status = status
        self._audit(document, status.value.lower(), message)
        self._persist_document(document)

    def _model_run(
        self,
        stage: str,
        model_name: str,
        model_version: str,
        status: str,
        confidence: float | None,
        *,
        routing_reason: str | None = None,
        notes: list[str] | None = None,
        duration_ms: float | None = None,
        document: VisionDocumentRecord | None = None,
        input_schema: str | None = None,
        output_schema: str | None = None,
        selected_schema: str | None = None,
        metrics: dict[str, Any] | None = None,
        result_status: str | None = None,
        reason: str | None = None,
    ) -> ModelRunRecord:
        now = utc_now_iso()
        runtime_ms = round(max(0.0, duration_ms or 0.0), 3)
        return ModelRunRecord(
            stage=stage,
            model_name=model_name,
            model_version=model_version,
            status=status,
            started_at=now,
            completed_at=now,
            confidence=confidence,
            processing_duration_ms=runtime_ms,
            runtime_ms=runtime_ms,
            source_document=document.document_id if document else None,
            source_pages=list(range(1, document.page_count + 1)) if document else [],
            routing_reason=routing_reason,
            notes=notes or [],
            input_schema=input_schema,
            output_schema=output_schema,
            document_type=document.document_type if document else None,
            selected_schema=selected_schema,
            metrics=metrics or {},
            result_status=result_status,
            reason=reason,
        )
