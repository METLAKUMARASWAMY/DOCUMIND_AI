"""The processing pipeline: OCR, layout, classification and field discovery.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from app.core.security import utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    TypedVisionEntity,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidence,
    VisionExtractionOutcome,
    VisionGraphProjection,
    VisionProcessingStatus,
    VisionRiskIndicator,
    VisionRiskSeverity,
    VisionSemanticType,
)
from app.models.vision import (
    GenericDiscoveredField as VisionGenericDiscoveredField,
)
from app.services.vision.analytics import vision_analytics_service
from app.services.vision.deterministic_extraction import deterministic_schema_extractor
from app.services.vision.document_classification import hybrid_document_classifier
from app.services.vision.duplicate_detection import (
    document_dhash,
)
from app.services.vision.entity_resolution import conservative_entity_resolver
from app.services.vision.generic_field_discovery import generic_field_discovery_engine
from app.services.vision.language_analysis import vision_language_analysis_service
from app.services.vision.legacy_audit import CURRENT_PIPELINE_VERSION
from app.services.vision.ocr_lines import ocr_line_reconstructor
from app.services.vision.schema_registry import vision_schema_registry
from app.services.vision.service_parts.constants import (
    CONTRACT_CLAUSE_FIELDS,
)


class ExtractionMixin:
    def _process_document(
        self,
        document: VisionDocumentRecord,
        filename: str,
        content: bytes,
        metadata: dict[str, Any],
    ) -> None:
        processing_started_perf = time.perf_counter()
        processing_started_at = utc_now_iso()
        stage_durations_ms: dict[str, float] = {}
        queued_event = next(
            (
                event
                for event in reversed(document.audit_events)
                if event.event_type in {"document_queued", "document_reprocess_queued"}
            ),
            None,
        )
        queued_at = queued_event.created_at if queued_event else None
        queued_duration_ms: float | None = None
        if queued_at:
            try:
                queued_duration_ms = max(
                    0.0,
                    (
                        datetime.fromisoformat(processing_started_at)
                        - datetime.fromisoformat(queued_at)
                    ).total_seconds()
                    * 1000,
                )
            except (TypeError, ValueError):
                queued_duration_ms = None
        document.metadata["pipeline_version"] = CURRENT_PIPELINE_VERSION
        document.metadata["processing_timing"] = {
            "duration_unit": "milliseconds",
            "queued_at": queued_at,
            "processing_started_at": processing_started_at,
            "processing_completed_at": None,
            "queued_duration_ms": (
                round(queued_duration_ms, 3) if queued_duration_ms is not None else None
            ),
            "processing_duration_ms": None,
            "stage_durations_ms": stage_durations_ms,
            "pipeline_version": CURRENT_PIPELINE_VERSION,
            "reliable": False,
            "exclusion_reason": "processing_incomplete",
        }
        self._transition(
            document,
            VisionProcessingStatus.preprocessing,
            "Started quality analysis and preprocessing.",
        )

        ingestion_started = time.perf_counter()
        text, pages, parse_evidence, stored_path = self.ingestion.parse(filename, content)
        stage_durations_ms["ingestion"] = round(
            (time.perf_counter() - ingestion_started) * 1000,
            3,
        )
        parse_confidence = parse_evidence[0].confidence if parse_evidence else 0.5
        document.parsed_text = text
        document.original_storage_path = stored_path

        preprocessing_started = time.perf_counter()
        preprocessing = self.preprocessor.process(
            document.document_id,
            filename,
            content,
            parsed_text=text,
        )
        document.page_artifacts = preprocessing.artifacts
        document.quality = preprocessing.quality
        document.page_count = len(preprocessing.artifacts) or pages
        if not document.parsed_text.strip():
            document.parsed_text = "\n\n".join(
                preprocessing.page_text.get(page_number, "")
                for page_number in sorted(preprocessing.page_text)
            ).strip()
        quality_confidence = (
            sum(item.image_quality_score for item in document.quality) / len(document.quality)
            if document.quality
            else 0.0
        )
        document.model_runs.append(
            self._model_run(
                "preprocessing",
                "OpenCV-Pillow-PyMuPDF",
                "v1",
                "completed",
                round(quality_confidence, 3),
            )
        )
        stage_durations_ms["preprocessing"] = round(
            (time.perf_counter() - preprocessing_started) * 1000,
            3,
        )

        ocr_started = time.perf_counter()
        ocr_adapter = (
            self.paddle_ocr if self.paddle_ocr.status().status == "active" else self.local_ocr
        )
        try:
            document.ocr_words = ocr_adapter.extract_words(document, preprocessing.native_words)
        except Exception as exc:
            if ocr_adapter is self.local_ocr:
                raise
            document.model_runs.append(
                self._model_run(
                    "ocr",
                    ocr_adapter.adapter_id,
                    ocr_adapter.model_version,
                    "failed_fallback_used",
                    None,
                    notes=[str(exc)],
                )
            )
            document.ocr_words = self.local_ocr.extract_words(document, preprocessing.native_words)
            ocr_adapter = self.local_ocr
        if not document.parsed_text.strip() and document.ocr_words:
            document.parsed_text = self._text_from_ocr(document.ocr_words)
        document.ocr_lines = ocr_line_reconstructor.reconstruct(document.ocr_words)
        if not document.parsed_text.strip() and document.ocr_lines:
            document.parsed_text = self._text_from_ocr(document.ocr_words)
        text = document.parsed_text
        parse_confidence = self._mean_word_confidence(document) or parse_confidence
        document.model_runs.append(
            self._model_run(
                "ocr",
                ocr_adapter.adapter_id,
                ocr_adapter.model_version,
                "completed",
                self._mean_word_confidence(document),
                notes=[
                    "Native PDF/source text geometry was preferred where available."
                    if preprocessing.native_words
                    else "OCR executed on processed page images."
                ],
            )
        )
        self._transition(
            document, VisionProcessingStatus.ocr_completed, "Completed OCR with page evidence."
        )
        stage_durations_ms["ocr"] = round(
            (time.perf_counter() - ocr_started) * 1000,
            3,
        )

        layout_started = time.perf_counter()
        layout_adapter = (
            self.paddle_structure
            if self.paddle_structure.status().status == "active"
            else self.local_layout
        )
        try:
            document.layout_regions, document.tables = layout_adapter.analyze_layout(document)
        except Exception as exc:
            if layout_adapter is self.local_layout:
                raise
            document.model_runs.append(
                self._model_run(
                    "layout",
                    layout_adapter.adapter_id,
                    layout_adapter.model_version,
                    "failed_fallback_used",
                    None,
                    notes=[str(exc)],
                )
            )
            layout_adapter = self.local_layout
            document.layout_regions, document.tables = layout_adapter.analyze_layout(document)
        document.model_runs.append(
            self._model_run(
                "layout",
                layout_adapter.adapter_id,
                layout_adapter.model_version,
                "completed",
                self._layout_confidence(document),
            )
        )
        stage_durations_ms["layout"] = round(
            (time.perf_counter() - layout_started) * 1000,
            3,
        )

        classification_started = time.perf_counter()
        document_type_hint = self._document_type_hint(metadata)
        if document_type_hint is not None:
            document.document_type = document_type_hint
            document.metadata["classification"] = {
                "predicted_type": document_type_hint.value,
                "parent_category": None,
                "confidence": None,
                "classifier": "user-selected-document-type",
                "signals": [
                    {
                        "type": "user_selection",
                        "value": document_type_hint.value,
                        "weight": None,
                        "candidate_type": document_type_hint.value,
                    }
                ],
                "alternatives": [],
                "user_override": document_type_hint.value,
            }
            document.model_runs.append(
                self._model_run(
                    "document_classification",
                    "reviewer-selected-document-type",
                    "v1",
                    "completed",
                    None,
                    duration_ms=(time.perf_counter() - classification_started) * 1000,
                    document=document,
                    notes=[
                        "The explicit upload selection was used instead of model classification."
                    ],
                )
            )
            self._audit(
                document,
                "document_type_selected",
                "Applied the document type selected during upload.",
                metadata={"document_type": document_type_hint.value},
            )
        elif self.layoutlmv3_classifier.status().status == "active":
            try:
                statistical_type, classification_confidence = self.layoutlmv3_classifier.classify(
                    document
                )
                classification = hybrid_document_classifier.classify(
                    text,
                    filename,
                    lines=document.ocr_lines,
                    layout_regions=document.layout_regions,
                    statistical_type=statistical_type,
                    statistical_confidence=classification_confidence,
                )
                document.document_type = classification.predicted_type
                document.metadata["classification"] = classification.as_dict()
                document.model_runs.append(
                    self._model_run(
                        "document_classification",
                        self.layoutlmv3_classifier.adapter_id,
                        self.layoutlmv3_classifier.model_version,
                        "completed",
                        classification.confidence,
                        duration_ms=(time.perf_counter() - classification_started) * 1000,
                        document=document,
                        metrics={"signals": len(classification.signals)},
                    )
                )
            except Exception as exc:
                classification = hybrid_document_classifier.classify(
                    text,
                    filename,
                    lines=document.ocr_lines,
                    layout_regions=document.layout_regions,
                )
                document.document_type = classification.predicted_type
                document.metadata["classification"] = classification.as_dict()
                document.model_runs.append(
                    self._model_run(
                        "document_classification",
                        self.layoutlmv3_classifier.adapter_id,
                        self.layoutlmv3_classifier.model_version,
                        "failed_fallback_used",
                        classification.confidence,
                        duration_ms=(time.perf_counter() - classification_started) * 1000,
                        document=document,
                        notes=[str(exc)],
                    )
                )
        else:
            classification = hybrid_document_classifier.classify(
                text,
                filename,
                lines=document.ocr_lines,
                layout_regions=document.layout_regions,
            )
            document.document_type = classification.predicted_type
            document.metadata["classification"] = classification.as_dict()
            document.model_runs.append(
                self._model_run(
                    "document_classification",
                    hybrid_document_classifier.adapter_id,
                    hybrid_document_classifier.model_version,
                    "completed",
                    classification.confidence,
                    duration_ms=(time.perf_counter() - classification_started) * 1000,
                    document=document,
                    metrics={"signals": len(classification.signals)},
                    notes=[
                        "Confidence is derived from configured rule weights, not learned accuracy."
                    ],
                )
            )
        stage_durations_ms["classification"] = round(
            (time.perf_counter() - classification_started) * 1000,
            3,
        )

        extraction_started = time.perf_counter()
        extraction_errors: list[str] = []
        extraction_adapter_name = deterministic_schema_extractor.adapter_id
        extraction_adapter_version = deterministic_schema_extractor.model_version
        try:
            if self.layoutlmv3_extractor.status().status == "active":
                learned_fields = self.layoutlmv3_extractor.extract_fields(document)
                learned_names = {field.name for field in learned_fields}
                supplemental_fields = [
                    field
                    for field in self._extract_fields(document)
                    if field.name not in learned_names
                ]
                document.extracted_fields = [*learned_fields, *supplemental_fields]
                extraction_adapter_name = self.layoutlmv3_extractor.adapter_id
                extraction_adapter_version = self.layoutlmv3_extractor.model_version
            else:
                document.extracted_fields = self._extract_fields(document)
        except Exception as exc:
            extraction_errors.append(f"{type(exc).__name__}: {exc}")
            document.extracted_fields = []
        try:
            self._discover_generic_intelligence(document)
            self._resolve_semantic_entities(document)
        except Exception as exc:
            extraction_errors.append(f"{type(exc).__name__}: {exc}")
        extraction_runtime_ms = (time.perf_counter() - extraction_started) * 1000
        vision_analytics_service.refresh_document(document)
        extraction_outcome = (
            VisionExtractionOutcome.extraction_failed
            if extraction_errors
            else self._extraction_outcome(document)
        )
        extraction_diagnostics = dict(document.metadata.get("extraction_diagnostics") or {})
        schema_selection = dict(document.metadata.get("schema_selection") or {})
        document.metadata["extraction"] = {
            "stage": "extraction",
            "status": extraction_outcome.value,
            "document_type": document.document_type.value,
            "schema": schema_selection.get("selected_schema"),
            "fields_extracted": len(document.extracted_fields),
            "required_fields_missing": len(document.field_completeness.missing_required_fields),
            "extractor": f"{deterministic_schema_extractor.adapter_id}-{deterministic_schema_extractor.model_version}",
            "runtime_ms": round(extraction_runtime_ms, 3),
            "pipeline_version": document.metadata["pipeline_version"],
            "errors": extraction_errors,
            "ocr_retrieval_preserved": bool(document.ocr_lines),
        }
        if document.document_type == VisionDocumentType.contract and not extraction_errors:
            try:
                self._run_contract_extraction(document)
            except Exception as exc:
                extraction_errors.append(f"{type(exc).__name__}: {exc}")
                extraction_outcome = VisionExtractionOutcome.extraction_failed
                document.metadata["extraction"]["status"] = extraction_outcome.value
                document.metadata["extraction"]["errors"] = extraction_errors
        stage_durations_ms["extraction"] = round(extraction_runtime_ms, 3)

        visual_analysis_started = time.perf_counter()
        visual_adapter = (
            self.yolo_detector
            if self.yolo_detector.status().status == "active"
            else self.local_visual_detector
        )
        try:
            document.visual_detections = visual_adapter.detect(document)
        except Exception as exc:
            if visual_adapter is self.local_visual_detector:
                raise
            document.model_runs.append(
                self._model_run(
                    "visual_detection",
                    visual_adapter.adapter_id,
                    visual_adapter.model_version,
                    "failed_fallback_used",
                    None,
                    notes=[str(exc)],
                )
            )
            visual_adapter = self.local_visual_detector
            document.visual_detections = visual_adapter.detect(document)
        document.model_runs.append(
            self._model_run(
                "extraction",
                extraction_adapter_name,
                extraction_adapter_version,
                "failed_ocr_retrieval_preserved" if extraction_errors else "completed",
                self._mean_field_confidence(document),
                duration_ms=extraction_runtime_ms,
                document=document,
                input_schema="ocr-lines-v1",
                output_schema="extracted-fields-v1",
                selected_schema=str(schema_selection.get("selected_schema") or "generic_document"),
                metrics={
                    key: extraction_diagnostics.get(key)
                    for key in (
                        "ocr_words_inspected",
                        "ocr_lines_inspected",
                        "labels_detected",
                        "fields_extracted",
                    )
                },
                result_status=extraction_outcome.value,
                notes=[
                    *extraction_errors,
                    *(
                        ["OCR evidence chunking, indexing, and grounded retrieval continued."]
                        if extraction_errors
                        else []
                    ),
                    "Confidence is uncalibrated and therefore cannot imply model accuracy."
                    if not bool(self.confidence_policy.get("calibration", {}).get("calibrated"))
                    else "Confidence calibration artefact is active.",
                ],
            )
        )
        document.model_runs.append(
            self._model_run(
                "visual_detection",
                visual_adapter.adapter_id,
                visual_adapter.model_version,
                "completed",
                round(
                    sum(item.confidence for item in document.visual_detections)
                    / len(document.visual_detections),
                    3,
                )
                if document.visual_detections
                else None,
                notes=["Candidate detection does not establish authenticity."],
            )
        )
        document.language_analysis = vision_language_analysis_service.analyze(document.ocr_words)
        document.model_routing = self.model_registry.route_document(
            filename=filename,
            text=text,
            parse_confidence=parse_confidence,
            layout_result={
                "engine": "local-layout-adapter",
                "regions": sorted({region.region_type for region in document.layout_regions}),
                "evidence": {"confidence": self._layout_confidence(document) or 0.0},
            },
            mixed_languages=document.language_analysis.mixed_language_routing,
            language_analysis=document.language_analysis.model_dump(mode="json"),
        )
        if document.model_routing.get("fallback_required"):
            self._run_selective_fallbacks(document)
        stage_durations_ms["visual_and_language_analysis"] = round(
            (time.perf_counter() - visual_analysis_started) * 1000,
            3,
        )
        self._audit(
            document,
            "extraction_outcome",
            (
                "Structured extraction failed; OCR evidence indexing and retrieval continued."
                if extraction_errors
                else f"Extraction completed: {len(document.extracted_fields)} fields extracted "
                f"using {schema_selection.get('selected_schema', 'generic_document')} schema."
                if document.extracted_fields
                else "Extraction completed with no source-backed fields. Generic schema fallback was attempted."
            ),
            metadata=document.metadata["extraction"],
        )
        self._transition(
            document,
            VisionProcessingStatus.extraction_completed,
            (
                f"Extraction completed - {len(document.extracted_fields)} fields extracted."
                if document.extracted_fields
                else "Extraction completed - no fields found."
            ),
        )

        forensic_screening_started = time.perf_counter()
        forensics_adapter = (
            self.segformer_forensics
            if self.segformer_forensics.status().status == "active"
            else self.local_forensics
        )
        try:
            document.tamper_analysis = forensics_adapter.analyze(document)
        except Exception as exc:
            if forensics_adapter is self.local_forensics:
                raise
            document.model_runs.append(
                self._model_run(
                    "tamper_screening",
                    forensics_adapter.adapter_id,
                    forensics_adapter.model_version,
                    "failed_fallback_used",
                    None,
                    notes=[str(exc)],
                )
            )
            forensics_adapter = self.local_forensics
            document.tamper_analysis = forensics_adapter.analyze(document)
        document.model_runs.append(
            self._model_run(
                "tamper_screening",
                forensics_adapter.adapter_id,
                forensics_adapter.model_version,
                "completed",
                max((item.screening_score for item in document.tamper_analysis), default=0.0),
                notes=[
                    "Local screening scores are measured signals, not calibrated tampering probabilities."
                    if forensics_adapter is self.local_forensics
                    else "Segmentation softmax is uncalibrated and requires human verification."
                ],
            )
        )
        stage_durations_ms["forensic_screening"] = round(
            (time.perf_counter() - forensic_screening_started) * 1000,
            3,
        )

        validation_started = time.perf_counter()
        validation_risks = self._validation_risks(document, metadata)
        if extraction_errors:
            validation_risks.append(
                self._risk(
                    document,
                    title="Structured Extraction Failure",
                    severity=VisionRiskSeverity.high,
                    category="processing",
                    description=(
                        "Structured extraction failed, but OCR evidence retrieval and indexing "
                        "continued so source-backed review remains available."
                    ),
                    recommended_action="Review OCR evidence and retry structured extraction.",
                    source_text="; ".join(extraction_errors),
                )
            )
        validation_risks.extend(self._generic_semantic_risks(document, metadata))
        validation_risks.extend(self._contract_risks(document))
        self._enrich_field_confidence(document, validation_risks)
        vision_analytics_service.refresh_document(document)
        stage_durations_ms["validation"] = round(
            (time.perf_counter() - validation_started) * 1000,
            3,
        )

        risk_screening_started = time.perf_counter()
        tamper_risks = self._tamper_risks(document)
        document.perceptual_hash = document_dhash(document)
        dino_status = self.dinov2_duplicate.status()
        if dino_status.status == "active":
            try:
                document.visual_embedding = self.dinov2_duplicate.embed(document)
                document.visual_embedding_model = self.dinov2_duplicate.model_version
                document.model_runs.append(
                    self._model_run(
                        self.dinov2_duplicate.stage,
                        self.dinov2_duplicate.adapter_id,
                        self.dinov2_duplicate.model_version,
                        "completed",
                        None,
                        notes=[f"Embedding dimensions: {len(document.visual_embedding)}."],
                    )
                )
            except Exception as exc:
                document.model_runs.append(
                    self._model_run(
                        self.dinov2_duplicate.stage,
                        self.dinov2_duplicate.adapter_id,
                        self.dinov2_duplicate.model_version,
                        "failed_fallback_used",
                        None,
                        notes=[str(exc), "Perceptual-hash duplicate screening remained active."],
                    )
                )
        document.duplicate_candidates = self._duplicate_candidates(document)
        duplicate_risks = self._duplicate_risks(document)
        document.processing_warnings = self._processing_warnings(document)
        document.risk_indicators = [
            *validation_risks,
            *tamper_risks,
            *duplicate_risks,
        ]
        self._run_anomaly_models(document)
        self._transition(
            document,
            VisionProcessingStatus.validation_completed,
            "Completed validation and risk screening.",
        )
        stage_durations_ms["risk_screening"] = round(
            (time.perf_counter() - risk_screening_started) * 1000,
            3,
        )

        chunking_started = time.perf_counter()
        document.evidence_chunks = self._chunks(document)
        self._enrich_field_confidence(document, document.risk_indicators)
        vision_analytics_service.refresh_document(document)
        stage_durations_ms["chunking"] = round(
            (time.perf_counter() - chunking_started) * 1000,
            3,
        )

        graph_started = time.perf_counter()
        document.graph = self._graph(document)
        self.persistence.upsert_graph(
            document.tenant_id,
            document.document_id,
            document.graph,
        )
        stage_durations_ms["graph_projection"] = round(
            (time.perf_counter() - graph_started) * 1000,
            3,
        )

        indexing_started = time.perf_counter()
        self._index_document(document)
        stage_durations_ms["embedding_and_indexing"] = round(
            (time.perf_counter() - indexing_started) * 1000,
            3,
        )
        self._transition(
            document,
            VisionProcessingStatus.indexed,
            "Indexed structure-aware chunks and graph projection.",
        )

        review_routing_started = time.perf_counter()
        previous_review_id = document.metadata.pop("_reprocess_review_id", None)
        if self._requires_review(document):
            document.review_task = self._create_review_task(
                document,
                review_id=str(previous_review_id) if previous_review_id else None,
            )
            self.review_tasks[document.review_task.review_id] = document.review_task
            self._transition(
                document, VisionProcessingStatus.review_required, "Document routed to human review."
            )
        else:
            self._transition(
                document,
                VisionProcessingStatus.completed,
                "Document completed without mandatory review.",
            )
        document.metadata["pipeline_status"] = VisionProcessingStatus.completed.value
        document.metadata["decision_status"] = document.status.value
        document.metadata["status_summary"] = (
            "Processing completed - awaiting human review"
            if document.status == VisionProcessingStatus.review_required
            else "Processing completed"
        )
        stage_durations_ms["review_routing"] = round(
            (time.perf_counter() - review_routing_started) * 1000,
            3,
        )
        processing_completed_at = utc_now_iso()
        document.metadata["processing_timing"] = {
            "duration_unit": "milliseconds",
            "queued_at": queued_at,
            "processing_started_at": processing_started_at,
            "processing_completed_at": processing_completed_at,
            "queued_duration_ms": (
                round(queued_duration_ms, 3) if queued_duration_ms is not None else None
            ),
            "processing_duration_ms": round(
                (time.perf_counter() - processing_started_perf) * 1000,
                3,
            ),
            "stage_durations_ms": stage_durations_ms,
            "pipeline_version": CURRENT_PIPELINE_VERSION,
            "reliable": True,
            "exclusion_reason": None,
        }
        vision_analytics_service.refresh_document(document)
        self._persist_document(document)

    def _reset_for_reprocess(
        self,
        document: VisionDocumentRecord,
        metadata: dict[str, Any],
        actor: str,
    ) -> None:
        previous_review_id = document.review_task.review_id if document.review_task else None
        if previous_review_id:
            self.review_tasks.pop(previous_review_id, None)
            self._remove_review(previous_review_id)
        self.retrieval_index.remove_document(document.document_id)
        document.status = VisionProcessingStatus.queued
        document.parsed_text = ""
        document.page_count = 1
        document.page_artifacts = []
        document.quality = []
        document.ocr_words = []
        document.ocr_lines = []
        document.layout_regions = []
        document.tables = []
        document.visual_detections = []
        document.tamper_analysis = []
        document.generic_fields = []
        document.typed_entities = []
        document.extracted_fields = []
        document.verified_facts = []
        document.schema_profile = type(document.schema_profile)()
        document.confidence_summary = type(document.confidence_summary)()
        document.field_completeness = type(document.field_completeness)()
        document.language_analysis = type(document.language_analysis)()
        document.processing_warnings = []
        document.risk_indicators = []
        document.duplicate_candidates = []
        document.anomaly_analysis = []
        document.evidence_chunks = []
        document.graph = VisionGraphProjection()
        document.model_runs = []
        document.model_routing = {}
        document.review_task = None
        document.perceptual_hash = None
        document.visual_embedding = None
        document.visual_embedding_model = None
        document.metadata = metadata
        document.data_version += 1
        document.analytics_cache_version += 1
        document.updated_at = utc_now_iso()
        self._audit(
            document,
            "document_reprocess_queued",
            "Document queued for in-place reprocessing.",
            actor=actor,
            metadata={
                "document_id_reused": True,
                "review_id_reused": previous_review_id,
                "reprocess_count": metadata["reprocess_count"],
            },
        )

    def _run_selective_fallbacks(self, document: VisionDocumentRecord) -> None:
        requested = {
            str(item.get("model_id"))
            for item in document.model_routing.get("fallback_models", [])
            if isinstance(item, dict)
        }
        reasons = ", ".join(document.model_routing.get("reasons", [])) or "selective routing"

        if "PaddleOCR-VL-1.6" in requested:
            status = self.paddleocr_vl.status()
            if status.status == "active":
                try:
                    regions, tables, blocks = self.paddleocr_vl.parse(document)
                    existing_regions = {
                        (
                            region.page_number,
                            region.region_type,
                            region.text,
                            region.bounding_box.x1,
                            region.bounding_box.y1,
                            region.bounding_box.x2,
                            region.bounding_box.y2,
                        )
                        for region in document.layout_regions
                    }
                    document.layout_regions.extend(
                        region
                        for region in regions
                        if (
                            region.page_number,
                            region.region_type,
                            region.text,
                            region.bounding_box.x1,
                            region.bounding_box.y1,
                            region.bounding_box.x2,
                            region.bounding_box.y2,
                        )
                        not in existing_regions
                    )
                    document.tables.extend(tables)
                    document.model_routing["advanced_parsing"] = {
                        "model": self.paddleocr_vl.model_version,
                        "blocks": blocks,
                        "confidence_available": False,
                    }
                    document.model_runs.append(
                        self._model_run(
                            self.paddleocr_vl.stage,
                            self.paddleocr_vl.adapter_id,
                            self.paddleocr_vl.model_version,
                            "completed",
                            None,
                            routing_reason=reasons,
                            notes=["PaddleOCR-VL does not expose calibrated block confidence."],
                        )
                    )
                except Exception as exc:
                    document.model_runs.append(
                        self._model_run(
                            self.paddleocr_vl.stage,
                            self.paddleocr_vl.adapter_id,
                            self.paddleocr_vl.model_version,
                            "failed",
                            None,
                            routing_reason=reasons,
                            notes=[str(exc)],
                        )
                    )
            else:
                document.model_runs.append(
                    self._model_run(
                        self.paddleocr_vl.stage,
                        self.paddleocr_vl.adapter_id,
                        self.paddleocr_vl.model_version,
                        f"skipped_{status.status}",
                        None,
                        routing_reason=reasons,
                        notes=[status.reason],
                    )
                )

        if "Donut" in requested:
            status = self.donut_extractor.status()
            if status.status == "active":
                try:
                    output = self.donut_extractor.extract_json(document)
                    disagreements = self._donut_disagreements(document, output)
                    document.model_routing["donut_comparison"] = {
                        "model": self.donut_extractor.model_version,
                        "output": output,
                        "disagreements": disagreements,
                        "primary_fields_overwritten": False,
                    }
                    document.model_runs.append(
                        self._model_run(
                            self.donut_extractor.stage,
                            self.donut_extractor.adapter_id,
                            self.donut_extractor.model_version,
                            "completed",
                            None,
                            routing_reason=reasons,
                            notes=[f"Observed {len(disagreements)} field disagreement(s)."],
                        )
                    )
                except Exception as exc:
                    document.model_runs.append(
                        self._model_run(
                            self.donut_extractor.stage,
                            self.donut_extractor.adapter_id,
                            self.donut_extractor.model_version,
                            "failed",
                            None,
                            routing_reason=reasons,
                            notes=[str(exc)],
                        )
                    )
            else:
                document.model_runs.append(
                    self._model_run(
                        self.donut_extractor.stage,
                        self.donut_extractor.adapter_id,
                        self.donut_extractor.model_version,
                        f"skipped_{status.status}",
                        None,
                        routing_reason=reasons,
                        notes=[status.reason],
                    )
                )

    def _donut_disagreements(
        self,
        document: VisionDocumentRecord,
        output: dict[str, Any],
    ) -> list[dict[str, str]]:
        flattened: dict[str, str] = {}

        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(f"{prefix}.{key}" if prefix else str(key), child)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(f"{prefix}.{index}", child)
            elif value is not None:
                flattened[prefix.lower().replace(" ", "_")] = str(value)

        visit("", output)
        disagreements: list[dict[str, str]] = []
        for field in document.extracted_fields:
            match = next(
                (
                    value
                    for key, value in flattened.items()
                    if key.rsplit(".", 1)[-1] == field.name.lower()
                ),
                None,
            )
            if match is not None and self._normalize(match) != self._normalize(field.value):
                disagreements.append(
                    {
                        "field_name": field.name,
                        "primary_value": field.value,
                        "donut_value": match,
                    }
                )
        return disagreements

    def _run_contract_extraction(self, document: VisionDocumentRecord) -> None:
        status = self.contract_extractor.status()
        if status.status != "active":
            document.model_runs.append(
                self._model_run(
                    self.contract_extractor.stage,
                    self.contract_extractor.adapter_id,
                    self.contract_extractor.model_version,
                    f"skipped_{status.status}",
                    None,
                    notes=[
                        status.reason,
                        "Deterministic evidence-localized clause detection remains active.",
                    ],
                )
            )
            return
        try:
            learned = self.contract_extractor.extract_fields(document)
        except Exception as exc:
            document.model_runs.append(
                self._model_run(
                    self.contract_extractor.stage,
                    self.contract_extractor.adapter_id,
                    self.contract_extractor.model_version,
                    "failed_fallback_used",
                    None,
                    notes=[str(exc)],
                )
            )
            return
        learned_names = {field.name for field in learned}
        retained = [
            field
            for field in document.extracted_fields
            if field.name not in CONTRACT_CLAUSE_FIELDS or field.name not in learned_names
        ]
        document.extracted_fields = self._dedupe_fields([*retained, *learned])
        document.model_runs.append(
            self._model_run(
                self.contract_extractor.stage,
                self.contract_extractor.adapter_id,
                self.contract_extractor.model_version,
                "completed",
                self._mean_available_field_confidence(learned),
                notes=["Clause scores are uncalibrated and are not legal judgements."],
            )
        )

    def _document_type_hint(
        self,
        metadata: dict[str, Any],
    ) -> VisionDocumentType | None:
        raw_value = metadata.get("document_type_hint")
        if raw_value in {None, "", "auto", "unknown"}:
            return None
        normalized = str(raw_value).strip().lower().replace(" ", "_")
        aliases = {
            "purchaseorder": "purchase_order",
            "po": "purchase_order",
            "bank": "bank_document",
            "bank_statement": "bank_document",
            "fax": "fax_transmission",
            "facsimile": "fax_transmission",
            "businessform": "business_form",
            "general_form": "business_form",
            "financialreport": "financial_report",
            "generalreport": "general_report",
        }
        try:
            return VisionDocumentType(aliases.get(normalized, normalized))
        except ValueError:
            return None

    def _classify_document(self, text: str, filename: str) -> VisionDocumentType:
        return hybrid_document_classifier.classify(text, filename).predicted_type

    def _extraction_outcome(
        self,
        document: VisionDocumentRecord,
    ) -> VisionExtractionOutcome:
        selection = document.metadata.get("schema_selection") or {}
        if document.extracted_fields:
            if selection.get("fallback_used"):
                return VisionExtractionOutcome.schema_fallback_used
            observed = [
                field.confidence
                for field in document.extracted_fields
                if field.confidence is not None
            ]
            if observed and min(observed) < self._mandatory_review_threshold(document):
                return VisionExtractionOutcome.low_confidence_fields
            return VisionExtractionOutcome.success_with_fields
        if not selection.get("selected_schema"):
            return VisionExtractionOutcome.schema_unavailable
        if document.model_routing.get("fallback_required"):
            return VisionExtractionOutcome.fallback_required
        return VisionExtractionOutcome.completed_no_fields

    def _extract_fields(self, document: VisionDocumentRecord) -> list[ExtractedVisionField]:
        fields = deterministic_schema_extractor.extract(document)
        if document.document_type == VisionDocumentType.contract:
            # Clause detection answers "is this clause present?". It must never
            # displace or duplicate a field the deterministic extractor already
            # located precisely -- doing so previously produced a second
            # agreement_date whose value was 160 characters of raw document text.
            already_extracted = {field.name for field in fields}
            fields.extend(
                clause
                for clause in self._clause_fields(document)
                if clause.name not in already_extracted
            )
        return self._dedupe_fields(fields)

    def _discover_generic_intelligence(self, document: VisionDocumentRecord) -> None:
        """Retain schema-independent OCR facts alongside deterministic extraction.

        The generic layer is deliberately additive: registered-schema fields remain
        authoritative, while unseen labels and reusable typed values stay available
        to review, retrieval, graph, and open-set document workflows.
        """
        result = generic_field_discovery_engine.discover(
            document.ocr_lines,
            document.ocr_words,
            document_type=document.document_type.value,
        )
        registered_schema = vision_schema_registry.get(document.document_type)
        reviewed_aliases = self._feedback_aliases(document.tenant_id)
        embedding_label_matches, embedding_mapping_diagnostics = self._embedding_schema_labels(
            [observed.raw_label for observed in result.fields],
            registered_schema,
        )
        schema_names = (
            {item.name for item in registered_schema.fields}
            if registered_schema is not None
            else set()
        )
        evidence_by_field_id: dict[str, VisionEvidence] = {}
        generic_fields: list[VisionGenericDiscoveredField] = []
        rejected_candidates: list[dict[str, Any]] = []
        rejected_field_ids: set[str] = set()
        for observed in result.fields:
            if self._generic_observation_in_table(document, observed):
                rejected_candidates.append(
                    self._generic_rejection_record(observed, "inside_structured_table")
                )
                rejected_field_ids.add(observed.field_id)
                continue
            semantic_type = self._vision_semantic_type(observed.semantic_type)
            registered_label = self._registered_schema_label(
                observed.raw_label,
                registered_schema,
            )
            reviewed_label = reviewed_aliases.get(self._normalize_token(observed.raw_label))
            embedding_label = embedding_label_matches.get(observed.raw_label)
            normalized_label = (
                registered_label
                or reviewed_label
                or (embedding_label[0] if embedding_label else None)
                or observed.normalized_label
            )
            semantic_mapping_method = (
                "registered_schema_alias"
                if registered_label
                else "reviewer_feedback_alias"
                if reviewed_label
                else "bge_m3_embedding_similarity"
                if embedding_label
                else observed.label_mapping_method
            )
            semantic_mapping_confidence = (
                1.0
                if registered_label or reviewed_label
                else embedding_label[1]
                if embedding_label
                else observed.label_mapping_confidence
            )
            rejection_reason = self._generic_observation_rejection_reason(
                document,
                observed,
                normalized_label=normalized_label,
                registered_schema=registered_schema,
                semantic_type=semantic_type,
            )
            if rejection_reason is not None:
                rejected_candidates.append(
                    self._generic_rejection_record(observed, rejection_reason)
                )
                rejected_field_ids.add(observed.field_id)
                continue
            evidence = VisionEvidence(
                evidence_id=f"vis-ev-{observed.field_id}",
                document_id=document.document_id,
                page_number=observed.page,
                source_text=f"{observed.raw_label}: {observed.raw_value}",
                bounding_box=VisionBoundingBox(**observed.evidence_bbox.as_dict()),
                model_confidence=observed.confidence,
                model_name=generic_field_discovery_engine.adapter_id,
                model_version=generic_field_discovery_engine.model_version,
                processing_timestamp=utc_now_iso(),
                metadata={
                    "raw_label": observed.raw_label,
                    "normalized_label": normalized_label,
                    "generic_engine_normalized_label": observed.normalized_label,
                    "label_mapping_method": observed.label_mapping_method,
                    "label_mapping_confidence": observed.label_mapping_confidence,
                    "semantic_mapping_method": semantic_mapping_method,
                    "semantic_mapping_confidence": semantic_mapping_confidence,
                    "extraction_strategy": observed.extraction_strategy,
                    "source_line_ids": list(observed.source_line_ids),
                    "label_word_ids": list(observed.label_word_ids),
                    "value_word_ids": list(observed.value_word_ids),
                },
            )
            evidence_by_field_id[observed.field_id] = evidence
            generic_fields.append(
                VisionGenericDiscoveredField(
                    field_id=observed.field_id,
                    raw_label=observed.raw_label,
                    normalized_label=normalized_label,
                    raw_value=observed.raw_value,
                    normalized_value=observed.normalized_value,
                    page=observed.page,
                    label_bbox=VisionBoundingBox(**observed.label_bbox.as_dict()),
                    value_bbox=VisionBoundingBox(**observed.value_bbox.as_dict()),
                    confidence=observed.confidence,
                    source_engine=observed.source_engine,
                    semantic_type_if_known=semantic_type,
                    semantic_mapping_confidence=(
                        observed.semantic_confidence if semantic_type is not None else None
                    ),
                    evidence=evidence,
                    review_required=observed.review_required,
                    metadata={
                        "canonical_label": observed.canonical_label,
                        "label_mapping_method": observed.label_mapping_method,
                        "label_mapping_confidence": observed.label_mapping_confidence,
                        "semantic_mapping_method": semantic_mapping_method,
                        "semantic_mapping_confidence": semantic_mapping_confidence,
                        "extraction_strategy": observed.extraction_strategy,
                    },
                )
            )
        document.generic_fields = generic_fields

        entities: list[TypedVisionEntity] = []
        for observed in result.entities:
            if observed.linked_field_id in rejected_field_ids:
                continue
            semantic_type = self._vision_semantic_type(observed.semantic_type)
            if semantic_type is None or semantic_type == VisionSemanticType.text:
                continue
            linked_evidence = evidence_by_field_id.get(observed.linked_field_id or "")
            entities.append(
                TypedVisionEntity(
                    entity_id=observed.entity_id,
                    raw_value=observed.raw_value,
                    normalized_value=observed.normalized_value,
                    semantic_type=semantic_type,
                    confidence=observed.confidence,
                    context={
                        "page": observed.page,
                        "bounding_box": observed.bounding_box.as_dict(),
                        "contextual_label": observed.contextual_label,
                        "source_engine": observed.source_engine,
                        "source_line_ids": list(observed.source_line_ids),
                        "word_ids": list(observed.word_ids),
                    },
                    evidence_ids=(
                        [linked_evidence.evidence_id] if linked_evidence is not None else []
                    ),
                    review_required=observed.confidence < 0.8,
                )
            )
        document.typed_entities = entities
        document.metadata["generic_discovery"] = result.as_dict()["diagnostics"]
        document.metadata["generic_discovery"]["generic_fields_retained"] = len(
            document.generic_fields
        )
        document.metadata["generic_discovery"]["candidates_rejected"] = len(rejected_candidates)
        document.metadata["generic_discovery"]["rejected_candidates"] = rejected_candidates
        document.metadata["generic_discovery"]["embedding_label_mapping"] = (
            embedding_mapping_diagnostics
        )
        supplemental = [
            field for field in document.generic_fields if field.normalized_label not in schema_names
        ]
        if schema_names and supplemental:
            document.schema_profile.mode = "B"
            document.schema_profile.selection_source = "registered_plus_generic"
        elif not schema_names:
            document.schema_profile.mode = "C"
            document.schema_profile.schema_status = "generic"
            document.schema_profile.selection_source = "open_set_generic"

        # Mirror only out-of-schema discoveries into the established field contract
        # so legacy consumers gain Mode B/C structure without duplicating Mode A fields.
        existing_names = {field.name for field in document.extracted_fields}
        for field in supplemental:
            if field.normalized_label in existing_names:
                continue
            document.extracted_fields.append(
                ExtractedVisionField(
                    field_id=f"field-{field.field_id}",
                    name=field.normalized_label,
                    display_name=field.raw_label,
                    value=field.raw_value,
                    raw_value=field.raw_value,
                    normalized_value=field.normalized_value,
                    validation_status=(
                        "valid" if field.semantic_type_if_known is not None else "not_evaluated"
                    ),
                    confidence=field.confidence,
                    evidence=field.evidence,
                    model_name=generic_field_discovery_engine.adapter_id,
                    model_version=generic_field_discovery_engine.model_version,
                    extractor="generic-open-set",
                    required=False,
                )
            )
            existing_names.add(field.normalized_label)
        document.extracted_fields = self._dedupe_fields(document.extracted_fields)

    def _vision_semantic_type(self, value: object) -> VisionSemanticType | None:
        name = str(getattr(value, "name", "")).lower()
        return VisionSemanticType.__members__.get(name)

    def _resolve_semantic_entities(self, document: VisionDocumentRecord) -> None:
        """Deduplicate observations and record only conservative tenant matches."""
        self._load_all_documents(document.tenant_id)
        corpus = [item for item in self.documents.values() if item.tenant_id == document.tenant_id]
        report = conservative_entity_resolver.resolve(
            document,
            corpus,
            reviewed_value_aliases=self.feedback_dataset.reviewed_value_aliases(document.tenant_id),
        )
        document.typed_entities = list(report.entities)
        document.metadata["entity_resolution"] = report.as_dict()

    def _generic_observation_in_table(
        self,
        document: VisionDocumentRecord,
        observed: object,
    ) -> bool:
        box = observed.evidence_bbox
        area = max((box.x2 - box.x1) * (box.y2 - box.y1), 1.0)
        for table in document.tables:
            if table.page_number != observed.page:
                continue
            table_box = table.bounding_box
            overlap_width = max(0.0, min(box.x2, table_box.x2) - max(box.x1, table_box.x1))
            overlap_height = max(0.0, min(box.y2, table_box.y2) - max(box.y1, table_box.y1))
            if overlap_width * overlap_height / area >= 0.8:
                return True
        return False

    def _generic_observation_rejection_reason(
        self,
        document: VisionDocumentRecord,
        observed: object,
        *,
        normalized_label: str,
        registered_schema: object,
        semantic_type: VisionSemanticType | None,
    ) -> str | None:
        """Reject only demonstrable OCR pair noise, preserving uncertain real labels.

        Weak geometry strategies can accidentally split initials or the value-bearing
        tail of a known schema line into a second field.  Explicit delimiters and
        semantically typed values remain eligible even when their labels are short.
        """
        if getattr(observed, "label_mapping_method", "") != "open_set":
            return None
        if self._registered_schema_label(observed.raw_label, registered_schema):
            return None

        strategy = str(getattr(observed, "extraction_strategy", ""))
        weak_geometry = strategy in {
            "same_line_geometry_gap",
            "same_line_whitespace",
            "separate_lines_horizontal",
            "label_above_value",
        }
        raw_label = str(getattr(observed, "raw_label", ""))
        raw_value = str(getattr(observed, "raw_value", ""))
        label_tokens = re.findall(r"[a-z0-9]+", raw_label.casefold())
        value_alnum = re.sub(r"[^a-z0-9]+", "", raw_value.casefold())
        untyped_text = semantic_type in {None, VisionSemanticType.text}

        if weak_geometry and untyped_text:
            if label_tokens and all(len(token) <= 2 for token in label_tokens):
                return "ocr_label_fragment"
            if len(value_alnum) <= 1:
                return "single_character_geometry_value"

        if (
            weak_geometry
            and normalized_label == getattr(observed, "normalized_label", "")
            and self._generic_label_contains_schema_prefix(raw_label, registered_schema)
            and self._generic_observation_overlaps_extracted_field(document, observed)
        ):
            return "known_schema_line_value_leakage"
        return None

    def _generic_label_contains_schema_prefix(
        self,
        raw_label: str,
        registered_schema: object,
    ) -> bool:
        if registered_schema is None:
            return False
        observed_words = re.findall(r"[a-z0-9]+", raw_label.casefold())
        for field in getattr(registered_schema, "fields", ()):
            candidates = (
                field.name,
                getattr(field, "display_name", ""),
                *getattr(field, "aliases", ()),
            )
            for candidate in candidates:
                candidate_words = re.findall(r"[a-z0-9]+", str(candidate).casefold())
                if (
                    len(candidate_words) >= 2
                    and len(observed_words) > len(candidate_words)
                    and observed_words[: len(candidate_words)] == candidate_words
                ):
                    return True
        return False

    def _generic_observation_overlaps_extracted_field(
        self,
        document: VisionDocumentRecord,
        observed: object,
    ) -> bool:
        observed_box = observed.evidence_bbox
        observed_area = max(
            (observed_box.x2 - observed_box.x1) * (observed_box.y2 - observed_box.y1),
            1.0,
        )
        for field in document.extracted_fields:
            evidence_box = field.evidence.bounding_box
            if evidence_box is None or field.evidence.page_number != observed.page:
                continue
            overlap_width = max(
                0.0,
                min(observed_box.x2, evidence_box.x2) - max(observed_box.x1, evidence_box.x1),
            )
            overlap_height = max(
                0.0,
                min(observed_box.y2, evidence_box.y2) - max(observed_box.y1, evidence_box.y1),
            )
            if overlap_width * overlap_height / observed_area >= 0.55:
                return True
        return False

    def _generic_rejection_record(self, observed: object, reason: str) -> dict[str, Any]:
        return {
            "field_id": str(getattr(observed, "field_id", "")),
            "raw_label": str(getattr(observed, "raw_label", "")),
            "normalized_label": str(getattr(observed, "normalized_label", "")),
            "raw_value": str(getattr(observed, "raw_value", "")),
            "confidence": float(getattr(observed, "confidence", 0.0)),
            "extraction_strategy": str(getattr(observed, "extraction_strategy", "unknown")),
            "reason": reason,
        }

    def _registered_schema_label(self, raw_label: str, schema: object) -> str | None:
        if schema is None:
            return None
        observed = self._normalize_token(raw_label)
        for field in getattr(schema, "fields", ()):
            candidates = (
                field.name,
                getattr(field, "display_name", ""),
                *getattr(field, "aliases", ()),
            )
            if observed and observed in {
                self._normalize_token(candidate) for candidate in candidates if candidate
            }:
                return field.name
        return None

    def _embedding_schema_labels(
        self,
        raw_labels: list[str],
        schema: object,
    ) -> tuple[dict[str, tuple[str, float]], dict[str, Any]]:
        """Conservatively map unseen labels with BGE when that runtime is active."""
        embedding = self.retrieval_index.embedding
        if (
            schema is None
            or getattr(embedding, "adapter_id", "") != "bge-m3-embedding"
            or not hasattr(embedding, "embed_many")
        ):
            return {}, {"status": "not_used", "reason": "bge_m3_not_active"}

        unique_labels = list(dict.fromkeys(label.strip() for label in raw_labels if label.strip()))
        fields = list(getattr(schema, "fields", ()))
        if not unique_labels or not fields:
            return {}, {"status": "not_used", "reason": "no_labels_or_schema_fields"}

        schema_id = str(getattr(schema, "name", "anonymous-schema"))
        try:
            cached = self._semantic_label_embedding_cache.get(schema_id)
            if cached is None:
                candidate_texts = [
                    (
                        "canonical document field: "
                        f"{field.display_name}. aliases: "
                        + ", ".join((field.name, *getattr(field, "aliases", ())))
                    )
                    for field in fields
                ]
                candidate_vectors = embedding.embed_many(candidate_texts)
                cached = {
                    field.name: vector.dense
                    for field, vector in zip(fields, candidate_vectors, strict=True)
                }
                self._semantic_label_embedding_cache[schema_id] = cached
            query_vectors = embedding.embed_many(
                [f"document field label: {label}" for label in unique_labels]
            )
        except Exception as exc:
            return {}, {
                "status": "failed_fallback_used",
                "reason": f"{type(exc).__name__}: embedding label mapping unavailable",
            }

        matches: dict[str, tuple[str, float]] = {}
        for label, vector in zip(unique_labels, query_vectors, strict=True):
            ranked = sorted(
                (
                    (self._vector_cosine(vector.dense, candidate), field_name)
                    for field_name, candidate in cached.items()
                ),
                reverse=True,
            )
            if not ranked:
                continue
            best_score, best_name = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            if best_score >= 0.84 and best_score - second_score >= 0.04:
                matches[label] = (best_name, round(best_score, 4))
        return matches, {
            "status": "completed",
            "engine": "bge-m3-embedding",
            "threshold": 0.84,
            "ambiguity_margin": 0.04,
            "labels_considered": len(unique_labels),
            "labels_mapped": len(matches),
        }

    def _vector_cosine(self, left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def _source_label(self, field: ExtractedVisionField) -> str:
        source = field.evidence.source_text.strip()
        matched = re.match(r"^\s*([^:#=]{1,100}?)\s*[:#=]", source)
        return matched.group(1).strip() if matched else field.display_name

    def _clause_fields(self, document: VisionDocumentRecord) -> list[ExtractedVisionField]:
        clauses = {
            "parties": ("between the parties", "by and between", "the parties agree"),
            "agreement_date": ("agreement date", "dated as of", "made on"),
            "effective_date": ("effective date", "effective as of"),
            "expiry_date": ("expiry date", "expiration date", "expires on"),
            "governing_law": ("governing law", "governed by the laws"),
            "renewal_term": ("renewal term", "automatic renewal", "auto-renew"),
            "termination_for_convenience": (
                "termination for convenience",
                "terminate for convenience",
            ),
            "notice_period": ("notice period", "days' notice", "days notice"),
            "anti_assignment": ("anti-assignment", "may not assign", "assignment consent"),
            "exclusivity": ("exclusivity", "exclusive supplier", "exclusive right"),
            "non_compete": ("non-compete", "non compete", "covenant not to compete"),
            "audit_rights": ("audit rights", "right to audit", "inspect the records"),
            "insurance": ("insurance coverage", "maintain insurance", "insurance policy"),
            "cap_on_liability": ("cap on liability", "liability shall not exceed"),
            "uncapped_liability": ("uncapped liability", "unlimited liability"),
            "perpetual_licence": ("perpetual licence", "perpetual license"),
            "revenue_sharing": ("revenue sharing", "share of revenue"),
            "change_of_control": ("change of control", "change in control"),
            "confidentiality": ("confidentiality", "non-disclosure", "confidential information"),
            "indemnification": ("indemnification", "indemnify", "hold harmless"),
            "payment_terms": ("payment terms", "payment shall be due", "net 30", "net 60"),
        }
        lower = document.parsed_text.lower()
        fields: list[ExtractedVisionField] = []
        for name, keywords in clauses.items():
            for keyword in keywords:
                index = lower.find(keyword)
                if index >= 0:
                    # The surrounding text is *evidence*; the value is the clause
                    # phrase that was matched. Storing the snippet as the value
                    # put raw multi-line document text into field values, which
                    # then flowed into chunks, graph nodes and citations.
                    snippet = document.parsed_text[max(0, index - 90) : index + 180].strip()
                    observed = document.parsed_text[index : index + len(keyword)].strip()
                    fields.append(
                        self._field(document, name, observed or keyword, snippet or keyword)
                    )
                    break
        return fields

    def _processing_warnings(
        self,
        document: VisionDocumentRecord,
    ) -> list[VisionRiskIndicator]:
        warnings = []
        if not document.ocr_words:
            warnings.append(
                self._risk(
                    document,
                    title="No OCR Text Available",
                    severity=VisionRiskSeverity.high,
                    category="processing_warning",
                    description="No reliable OCR text was available for structured processing.",
                    recommended_action="Run a production OCR adapter or request a clearer scan.",
                    source_text="No OCR words were extracted by the local adapter.",
                )
            )
        low_words = [word for word in document.ocr_words if word.review_required]
        if low_words:
            warnings.append(
                self._risk(
                    document,
                    title="Low OCR Confidence Words",
                    severity=VisionRiskSeverity.medium,
                    category="processing_warning",
                    description="Some OCR tokens fell below the configured reliability threshold.",
                    recommended_action="Review low-confidence words and consider advanced parsing.",
                    source_text=" ".join(word.text for word in low_words[:16]),
                    evidence=[self._word_evidence(document, word) for word in low_words[:8]],
                )
            )
        if document.model_routing.get("fallback_required"):
            warnings.append(
                self._risk(
                    document,
                    title="Advanced Parser Fallback Requested",
                    severity=VisionRiskSeverity.medium,
                    category="processing_warning",
                    description="One or more measured processing conditions requested an advanced parser.",
                    recommended_action="Enable or run the configured fallback parser for this document.",
                    source_text=", ".join(document.model_routing.get("reasons", [])),
                )
            )
        if document.language_analysis.mixed_language_routing:
            warnings.append(
                self._risk(
                    document,
                    title="Multiple-language uncertainty",
                    severity=VisionRiskSeverity.medium,
                    category="processing_warning",
                    description=(
                        "Cleaned OCR text contains a second script above the configured "
                        "character-ratio and confidence thresholds."
                    ),
                    recommended_action=(
                        "Review the detected-language evidence and run multilingual OCR "
                        "only if the source document confirms both language groups."
                    ),
                    source_text=document.language_analysis.sample_text,
                    rule_id="processing_warning.mixed_languages_detected",
                )
            )
        schema_selection = document.metadata.get("schema_selection") or {}
        if document.field_completeness.schema_available and schema_selection.get("fallback_used"):
            selected_schema = str(schema_selection.get("selected_schema") or "generic_document")
            warnings.append(
                self._risk(
                    document,
                    title="Generic Schema Fallback Used",
                    severity=VisionRiskSeverity.info,
                    category="processing_warning",
                    description=(
                        "Using generic key-value schema."
                        if selected_schema == "generic_key_value_form"
                        else f"Using {selected_schema.replace('_', ' ')} schema."
                    ),
                    recommended_action="Review extracted source evidence for the generic fields.",
                    source_text=selected_schema,
                    rule_id="processing_warning.schema_fallback_used",
                )
            )
        elif not document.field_completeness.schema_available:
            warnings.append(
                self._risk(
                    document,
                    title="Expected Schema Unavailable",
                    severity=VisionRiskSeverity.low,
                    category="processing_warning",
                    description="No expected-field schema is registered for this document type.",
                    recommended_action="Select a supported document type or review the document manually.",
                    source_text=document.document_type.value,
                )
            )
        return warnings
