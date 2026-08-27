"""Evidence retrieval, field matching, chunking and coverage verdicts.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

import re
import time
from typing import Any, ClassVar

from app.core.security import hash_document
from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionAskCitation,
    VisionAskRequest,
    VisionAskResponse,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionEvidenceChunk,
    VisionProcessingStatus,
    VisionRetrievalTrace,
)
from app.services.vision.deterministic_extraction import deterministic_schema_extractor
from app.services.vision.generic_field_discovery import generic_field_discovery_engine
from app.services.vision.ocr_lines import ocr_line_reconstructor
from app.services.vision.semantic_values import (
    SemanticContext,
    SemanticValueType,
    semantic_value_registry,
)
from app.services.vision.service_parts.constants import (
    RETRIEVAL_SOURCE_SEQUENCE,
)


class RetrievalMixin:
    def _attach_retrieval_trace(
        self,
        response: VisionAskResponse,
        question: str,
        documents: list[VisionDocumentRecord],
        tenant_id: str,
        started_at: float,
    ) -> None:
        if response.retriever == "not_used":
            sources_checked: list[str] = []
            candidate_counts: dict[str, int] = {}
            versions: dict[str, str] = {}
            reranker = "not_used"
        else:
            candidate_counts, versions, reranker = self._retrieval_inventory(
                question,
                documents,
                tenant_id,
            )
            sources_checked = list(RETRIEVAL_SOURCE_SEQUENCE)
        selected = list(
            dict.fromkeys(
                str(
                    citation.evidence_id
                    or citation.chunk_id
                    or f"{citation.document_id}:page:{citation.page_number}"
                )
                for citation in response.citations
            )
        )
        trace = VisionRetrievalTrace(
            sources_checked=sources_checked,
            candidate_counts=candidate_counts,
            selected_evidence=selected,
            retrieval_duration_ms=round(
                max(0.0, (time.perf_counter() - started_at) * 1000),
                3,
            ),
            retriever_versions=versions,
        )
        response.retrieval_trace = trace
        response.retrieval_trace_id = trace.trace_id
        response.reranker = reranker

    def _retrieval_inventory(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        tenant_id: str,
    ) -> tuple[dict[str, int], dict[str, str], str]:
        requested_names = self._requested_field_names(question)
        matched_fields = self._field_matches(question.lower(), documents)
        if not matched_fields and any(
            term in question.lower() for term in ("quality", "safe", "approval")
        ):
            matched_fields = [
                field for document in documents for field in document.extracted_fields
            ]
        verified = [
            field for field in matched_fields if field.reviewer_status == ReviewerStatus.accepted
        ]
        reviewed = [
            field for field in matched_fields if field.reviewer_status == ReviewerStatus.edited
        ]
        unverified = [
            field
            for field in matched_fields
            if field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
        ]
        ocr_matches = self._ocr_lookup(question, documents, 50)
        relevant_chunks = [
            chunk
            for document in documents
            for chunk in document.evidence_chunks
            if self._text_relevant_to_question(question, chunk.text)
        ]
        vector_hits: list[Any] = []
        if documents:
            try:
                vector_hits = self.retrieval_index.search(
                    question,
                    tenant_id=tenant_id,
                    top_k=12,
                    document_ids={document.document_id for document in documents},
                )
            except Exception:
                vector_hits = []
        graph_edges = self._matching_graph_edges(question, documents)
        if requested_names:
            graph_edges = [
                item
                for item in graph_edges
                if str(item[1].get("source_field") or "") in requested_names
            ]
        visual = self._matching_visual_detections(question, documents)
        numeric = [
            field
            for document in documents
            for field in document.extracted_fields
            if self._decimal_value(field.reviewed_value or field.value) is not None
        ]
        counts = {
            "verified_structured_fields": len(verified),
            "reviewed_structured_fields": len(reviewed),
            "unverified_extracted_fields": len(unverified),
            "field_aliases": len(requested_names),
            "ocr_key_value_pairs": sum(
                1 for chunk in relevant_chunks if chunk.source_type == "key_value"
            ),
            "reconstructed_ocr_lines": len(ocr_matches)
            or sum(
                1
                for document in documents
                for line in document.ocr_lines
                if self._text_relevant_to_question(question, line.text)
            ),
            "lexical_bm25_chunks": sum(1 for hit in vector_hits if hit.sparse_score > 0),
            "dense_vector_retrieval": sum(1 for hit in vector_hits if hit.dense_score > 0),
            "reranked_evidence": sum(1 for hit in vector_hits if hit.rerank_score is not None),
            "graph_facts_and_paths": len(graph_edges),
            "visual_detections": len(visual),
            "deterministic_calculations": len(numeric),
        }
        reranker_status = self.retrieval_index.reranker.status()
        reranker = (
            f"{self.retrieval_index.reranker.adapter_id}:{self.retrieval_index.reranker.model_version}"
            if reranker_status.status == "active"
            else "not_used"
        )
        versions = {
            "verified_structured_fields": "persisted-structured-field-store-v1",
            "reviewed_structured_fields": "persisted-structured-field-store-v1",
            "unverified_extracted_fields": "persisted-structured-field-store-v1",
            "field_aliases": "vision-field-aliases-v3",
            "ocr_key_value_pairs": "structure-aware-chunker-v1",
            "reconstructed_ocr_lines": "ocr-line-reconstructor-v1",
            "lexical_bm25_chunks": "local-hybrid-sparse-v1",
            "dense_vector_retrieval": self.retrieval_index.embedding.model_version,
            "reranked_evidence": (
                self.retrieval_index.reranker.model_version
                if reranker_status.status == "active"
                else "not_enabled"
            ),
            "graph_facts_and_paths": "documind-evidence-graph-v4",
            "visual_detections": "stored-visual-detections-v1",
            "deterministic_calculations": "deterministic-decimal-v1",
        }
        return counts, versions, reranker

    def _text_relevant_to_question(self, question: str, text: str) -> bool:
        stopwords = {
            "about",
            "and",
            "are",
            "document",
            "does",
            "for",
            "from",
            "have",
            "how",
            "is",
            "of",
            "the",
            "this",
            "to",
            "what",
            "which",
            "who",
        }
        query_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", question.lower())
            if len(token) >= 3 and token not in stopwords
        }
        text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        return bool(query_tokens & text_tokens)

    def _ask_scope_documents(
        self,
        request: VisionAskRequest,
        tenant_id: str | None,
    ) -> tuple[list[VisionDocumentRecord], str]:
        if request.document_id:
            return [self.get_document(request.document_id, tenant_id)], "current_document"
        if request.document_ids:
            seen: set[str] = set()
            selected = []
            for document_id in request.document_ids:
                if document_id in seen:
                    continue
                selected.append(self.get_document(document_id, tenant_id))
                seen.add(document_id)
            return selected, "selected_documents"
        documents = self.list_documents(tenant_id)
        if request.scope == "approved_documents":
            return (
                [
                    document
                    for document in documents
                    if document.status == VisionProcessingStatus.approved
                ],
                "approved_documents",
            )
        return documents, request.scope

    def _ask_scope_payload(
        self,
        scope: str,
        documents: list[VisionDocumentRecord],
    ) -> dict[str, Any]:
        return {
            "scope": scope,
            "document_count": len(documents),
            "documents": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "document_type": document.document_type.value,
                    "thumbnail_url": f"/documents/{document.document_id}/pages/1/image?variant=original",
                    "page_count": document.page_count,
                    "field_count": len(document.extracted_fields),
                    "processing_version": document.metadata.get("pipeline_version", "legacy"),
                }
                for document in documents
            ],
        }

    def _question_has_document_evidence(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> bool:
        if self._field_matches(question.lower(), documents):
            return True
        return bool(self._ocr_lookup(question, documents, 1))

    def _matching_visual_detections(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> list[tuple[VisionDocumentRecord, Any]]:
        lower = question.lower()
        class_terms = {
            "signature": ("signature", "signed", "safety", "safe"),
            "stamp": ("stamp", "safety", "safe"),
            "seal": ("seal", "safety", "safe"),
            "checkbox_checked": ("checkbox", "checked"),
            "checkbox_unchecked": ("checkbox", "unchecked"),
            "company_logo": ("logo",),
            "handwritten_note": ("handwritten",),
        }
        requested_classes = {
            name for name, terms in class_terms.items() if any(term in lower for term in terms)
        }
        return [
            (document, detection)
            for document in documents
            for detection in document.visual_detections
            if not requested_classes or detection.detection_class in requested_classes
        ]

    def _chunks(self, document: VisionDocumentRecord) -> list[VisionEvidenceChunk]:
        chunks: list[VisionEvidenceChunk] = []
        common_metadata = {
            "filename": document.filename,
            "pipeline_version": document.metadata.get("pipeline_version", "legacy"),
            "extractor_version": deterministic_schema_extractor.model_version,
            "verification_status": ReviewerStatus.unverified.value,
        }
        for line in document.ocr_lines:
            line_chunk = VisionEvidenceChunk(
                chunk_id=self._chunk_id(
                    document,
                    line.page,
                    "ocr_line",
                    line.text,
                ),
                document_id=document.document_id,
                tenant_id=document.tenant_id,
                page_number=line.page,
                section=line.line_id,
                document_type=document.document_type,
                vendor_id=self._field_map(document).get("vendor_name"),
                bounding_boxes=[line.bounding_box],
                source_type="ocr_line",
                model_name="ocr-line-retriever",
                model_version="v1",
                text=line.text,
                confidence=line.mean_confidence or 0.0,
                metadata={
                    **common_metadata,
                    "word_ids": line.word_ids,
                    "block_ids": line.block_ids,
                },
            )
            chunks.append(line_chunk)
            if line.metadata.get("key_value_candidate"):
                chunks.append(
                    line_chunk.model_copy(
                        update={
                            "chunk_id": self._chunk_id(
                                document,
                                line.page,
                                "key_value",
                                line.text,
                            ),
                            "section": "key_value",
                            "source_type": "key_value",
                        }
                    )
                )
        lines_by_page: dict[int, list[Any]] = {}
        for line in document.ocr_lines:
            lines_by_page.setdefault(line.page, []).append(line)
        for page_number, page_lines in lines_by_page.items():
            page_text = "\n".join(
                line.text for line in sorted(page_lines, key=lambda item: item.reading_order)
            ).strip()
            if not page_text:
                continue
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(document, page_number, "page_text", page_text),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=page_number,
                    section=f"page_{page_number}",
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[self._union_box([line.bounding_box for line in page_lines])],
                    source_type="page_text",
                    model_name="ocr-page-chunker",
                    model_version="v1",
                    text=page_text[:6000],
                    confidence=round(
                        sum((line.mean_confidence or 0.0) for line in page_lines) / len(page_lines),
                        4,
                    ),
                    metadata=common_metadata,
                )
            )
        for region in document.layout_regions:
            if not region.text.strip():
                continue
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        region.page_number,
                        "layout",
                        region.text,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=region.page_number,
                    section=region.region_type,
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[region.bounding_box],
                    source_type=(
                        "form_section" if region.region_type == "form region" else "paragraph"
                    ),
                    model_name=region.model_name,
                    model_version=region.model_version,
                    text=region.text,
                    confidence=region.confidence,
                    metadata=common_metadata,
                )
            )
        for field in document.extracted_fields:
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        field.evidence.page_number,
                        f"field:{field.name}",
                        field.reviewed_value or field.value,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=field.evidence.page_number,
                    section=field.name,
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[field.evidence.bounding_box]
                    if field.evidence.bounding_box
                    else [],
                    source_type="field",
                    review_status=field.reviewer_status,
                    model_name=field.model_name,
                    model_version=field.model_version,
                    text=f"{field.name}: {field.reviewed_value or field.value}",
                    confidence=field.confidence or 0.0,
                    metadata={
                        **common_metadata,
                        "evidence_id": field.evidence.evidence_id,
                        "verification_status": field.reviewer_status.value,
                        "schema_field": field.schema_field,
                    },
                )
            )
        for field in document.generic_fields:
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        field.page,
                        f"generic_field:{field.normalized_label}",
                        field.normalized_value,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=field.page,
                    section=field.normalized_label,
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[field.evidence.bounding_box]
                    if field.evidence.bounding_box
                    else [],
                    source_type="generic_field",
                    model_name=generic_field_discovery_engine.adapter_id,
                    model_version=generic_field_discovery_engine.model_version,
                    text=f"{field.raw_label}: {field.raw_value}",
                    confidence=field.confidence,
                    metadata={
                        **common_metadata,
                        "evidence_id": field.evidence.evidence_id,
                        "raw_label": field.raw_label,
                        "normalized_label": field.normalized_label,
                        "semantic_type": (
                            field.semantic_type_if_known.value
                            if field.semantic_type_if_known is not None
                            else None
                        ),
                        "review_required": field.review_required,
                    },
                )
            )
        for entity in document.typed_entities:
            page_number = max(1, int(entity.context.get("page", 1)))
            raw_box = entity.context.get("bounding_box")
            entity_box = VisionBoundingBox(**raw_box) if isinstance(raw_box, dict) else None
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        page_number,
                        f"typed_entity:{entity.semantic_type.value}",
                        entity.normalized_value,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=page_number,
                    section=str(entity.context.get("contextual_label") or "typed_entity"),
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[entity_box] if entity_box is not None else [],
                    source_type="typed_entity",
                    model_name=str(
                        entity.context.get("source_engine")
                        or generic_field_discovery_engine.adapter_id
                    ),
                    model_version=generic_field_discovery_engine.model_version,
                    text=(
                        f"{entity.context.get('contextual_label')}: {entity.raw_value}"
                        if entity.context.get("contextual_label")
                        else entity.raw_value
                    ),
                    confidence=entity.confidence,
                    metadata={
                        **common_metadata,
                        "entity_id": entity.entity_id,
                        "semantic_type": entity.semantic_type.value,
                        "evidence_ids": entity.evidence_ids,
                        "review_required": entity.review_required,
                    },
                )
            )
        evidence_by_id = {
            field.evidence.evidence_id: field.evidence for field in document.extracted_fields
        }
        for fact in document.verified_facts:
            evidence = evidence_by_id.get(fact.evidence_id)
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        evidence.page_number if evidence else 1,
                        f"verified_fact:{fact.name}",
                        fact.value,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=evidence.page_number if evidence else 1,
                    section=fact.name,
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=(
                        [evidence.bounding_box] if evidence and evidence.bounding_box else []
                    ),
                    source_type="verified_fact",
                    review_status=fact.reviewer_status,
                    model_name="human-review",
                    model_version="v1",
                    text=f"{fact.name}: {fact.value}",
                    confidence=1.0,
                    metadata={
                        **common_metadata,
                        "evidence_id": fact.evidence_id,
                        "verification_status": fact.reviewer_status.value,
                        "verified_at": fact.verified_at,
                        "reviewer": fact.reviewer,
                    },
                )
            )
        for table in document.tables:
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        table.page_number,
                        "table",
                        table.markdown,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=table.page_number,
                    section="table",
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[table.bounding_box],
                    source_type="table",
                    model_name=table.model_name,
                    model_version=table.model_version,
                    text=table.markdown,
                    confidence=table.confidence,
                    metadata=common_metadata,
                )
            )
        retained: dict[tuple[int, str, str], VisionEvidenceChunk] = {}
        for chunk in chunks:
            key = (chunk.page_number, chunk.source_type, self._normalize(chunk.text))
            current = retained.get(key)
            if current is None or chunk.confidence > current.confidence:
                retained[key] = chunk
        return list(retained.values())

    def _chunk_id(
        self,
        document: VisionDocumentRecord,
        page_number: int,
        source_type: str,
        text: str,
    ) -> str:
        identity = hash_document(
            (f"{document.document_id}|{page_number}|{source_type}|{self._normalize(text)}").encode()
        )
        return f"vis-chk-{identity[:16]}"

    def _field_matches(
        self, question: str, documents: list[VisionDocumentRecord]
    ) -> list[ExtractedVisionField]:
        question_tokens = set(re.findall(r"[a-z0-9_]+", question))
        requested_names = self._requested_field_names(question)
        matches = []
        for document in documents:
            for field in document.extracted_fields:
                name_tokens = set(field.name.lower().split("_"))
                if (
                    field.name in requested_names
                    or name_tokens & question_tokens
                    or field.name.lower() in question
                ):
                    matches.append(field)
        return sorted(matches, key=lambda item: item.confidence or 0.0, reverse=True)

    def _field_citation(self, field: ExtractedVisionField) -> VisionAskCitation:
        source_document = self.documents.get(field.evidence.document_id)
        return VisionAskCitation(
            document_id=field.evidence.document_id,
            filename=source_document.filename if source_document else None,
            page_number=field.evidence.page_number,
            source_text=field.evidence.source_text,
            bounding_boxes=[field.evidence.bounding_box] if field.evidence.bounding_box else [],
            evidence_id=field.evidence.evidence_id,
            model_confidence=field.evidence.model_confidence,
            model_name=field.evidence.model_name,
            model_version=field.evidence.model_version,
            processing_timestamp=field.evidence.processing_timestamp,
            reviewer_status=field.reviewer_status,
            extraction_status=field.reviewer_status.value,
            evidence_type="structured_field",
        )

    def _filter_to_requested(
        self,
        matches: list[ExtractedVisionField],
        requested_names: set[str],
    ) -> list[ExtractedVisionField]:
        """Keep fields the question asked for, tolerating naming variants.

        The requested name and the extracted name do not always agree on
        qualifiers: "what is the bank account number" resolves to
        ``bank_account_number`` while the extractor produced ``account_number``.
        An exact-name filter dropped the field and the question fell through to a
        not-found, even though the value was extracted correctly.

        Exact names win. Only when none match does this fall back to comparing
        the token sets, so a variant is accepted but an unrelated field is not.
        """

        exact = [field for field in matches if field.name in requested_names]
        if exact:
            return exact
        requested_tokens = [set(name.split("_")) for name in requested_names]
        variants = []
        for field in matches:
            tokens = set(field.name.split("_"))
            if any(
                tokens and (tokens <= wanted or wanted <= tokens) for wanted in requested_tokens
            ):
                variants.append(field)
        return variants

    def _requested_field_names(self, question: str) -> set[str]:
        lower = question.lower()
        aliases = {
            "organisation_name": ("organisation", "organization", "company name"),
            "address": ("address",),
            "contact_name": ("contact name", "contact person", "contact"),
            "phone_number": ("phone", "telephone", "contact number"),
            "recipient_phone": ("phone", "telephone", "contact number"),
            "sender_phone": ("sender phone", "from phone"),
            "fax_number": ("fax", "fax number", "facsimile", "telefax"),
            "recipient_fax": ("fax", "fax number", "facsimile", "telefax"),
            "sender_fax": ("sender fax", "from fax"),
            "visible_fax_numbers": ("fax", "fax number", "facsimile", "telefax"),
            "visible_phone_numbers": ("phone", "telephone", "contact number"),
            "email": ("email", "e-mail"),
            "form_title": ("form title", "form name"),
            "vendor_name": ("vendor", "supplier", "seller"),
            "invoice_number": ("invoice number", "invoice id"),
            "invoice_date": ("invoice date",),
            "due_date": ("due date",),
            "purchase_order_number": ("purchase order", "po number", "po reference"),
            "contract_number": ("contract number",),
            "bank_account_number": ("bank account", "account number"),
            "bank_code": ("ifsc", "swift", "routing number", "bank code"),
            "subtotal": ("subtotal",),
            "tax": ("tax",),
            "discount": ("discount",),
            "total_amount": ("total amount", "grand total", "amount due", " total"),
            "currency": ("currency",),
            "payment_terms": ("payment terms",),
            "effective_date": ("effective date",),
            "date": ("date", "transmission date"),
            "sample_number": ("sample number", "sample no", "sample id"),
            "product_type": ("product type", "type of product", "type of cigarette", "sample type"),
            "batch_size": ("batch size", "lot size", "quantity"),
            "requested_by": ("original request", "requested by", "requester", "requestor"),
            "specifications_written_by": (
                "sample specifications",
                "specifications written by",
                "specifications by",
                "who wrote",
            ),
            "visible_dates": ("date", "visible date"),
            "recipient_name": ("recipient", "message to", "sent to"),
            "sender_name": ("sender", "sent by", "from"),
            "recipient_company": ("company", "organisation", "organization"),
            "company_name": ("company", "organisation", "organization"),
            "client_number": ("client number", "client no", "reference number"),
            "reference_number": ("reference number", "reference no", "ref no"),
            "document_title": ("document title", "title"),
            "expiry_date": ("expiry date", "expiration date"),
            "governing_law": ("governing law",),
            "termination_clause": ("termination", "termination clause"),
            "notice_period": ("notice period",),
            "liability_clause": ("liability clause",),
            "confidentiality_clause": ("confidentiality", "non-disclosure"),
            "indemnification_clause": ("indemnification", "indemnity"),
        }
        return {name for name, terms in aliases.items() if any(term in lower for term in terms)}

    def _logical_requested_field(self, question: str) -> str | None:
        lower = question.lower()
        terms = (
            ("sample_number", ("sample number", "sample no", "sample id")),
            (
                "product_type",
                ("product type", "type of product", "type of cigarette", "sample type"),
            ),
            ("batch_size", ("batch size", "lot size", "quantity")),
            ("requested_by", ("original request", "requested by", "requester", "requestor")),
            (
                "specifications_written_by",
                (
                    "sample specifications",
                    "specifications written by",
                    "specifications by",
                    "who wrote",
                ),
            ),
            ("fax_number", ("fax", "facsimile", "telefax")),
            ("phone_number", ("phone", "telephone", "contact number")),
            ("client_number", ("client number", "client no")),
            ("reference_number", ("reference number", "reference no", "ref no")),
            ("recipient_name", ("recipient", "message to", "sent to")),
            ("sender_name", ("sender", "sent by")),
            ("company", ("company", "organisation", "organization")),
            ("date", ("date", "dated")),
            ("email", ("email", "e-mail")),
        )
        return next(
            (name for name, aliases in terms if any(alias in lower for alias in aliases)),
            None,
        )

    def _ocr_lookup(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
    ) -> list[tuple[VisionDocumentRecord, Any, str]]:
        logical = self._logical_requested_field(question)
        if logical is None:
            return []
        value_patterns: dict[str, str] = {
            "fax_number": r"(?i)\b(?:fax|facsimile|telefax)(?:\s+(?:no\.?|number))?\s*[:#=-]?\s*((?:\+?\d[\d()./ -]{6,}\d))",
            "phone_number": r"(?i)\b(?:phone|telephone|tel|contact\s+number)(?:\s+(?:no\.?|number))?\s*[:#=-]?\s*((?:\+?\d[\d()./ -]{6,}\d))",
            "client_number": r"(?i)\b(?:client)(?:\s+(?:no\.?|number))?\s*[:#=-]?\s*([A-Z0-9][A-Z0-9._/-]{2,39})",
            "reference_number": r"(?i)\b(?:reference|ref)(?:\s+(?:no\.?|number))?\s*[:#=-]?\s*([A-Z0-9][A-Z0-9._/-]{2,39})",
            "recipient_name": r"(?i)^(?:message\s+to|recipient|attention|attn|to)\s*[:#=-]?\s*(\S.+)$",
            "sender_name": r"(?i)^(?:sender|sent\s+by|from)\s*[:#=-]?\s*(\S.+)$",
            "company": r"(?i)^(?:company|organisation|organization|firm)\s*[:#=-]?\s*(\S.+)$",
            "date": r"(?i)\b(?:date|dated|transmission\s+date)\s*[:#=-]?\s*((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            "email": r"(?i)\b([^\s@]+@[^\s@]+\.[^\s@]+)\b",
            "sample_number": r"(?i)\b(?:sample\s+(?:no\.?|number|id))\s*[:#=-]?\s*([A-Z0-9][A-Z0-9._/-]{1,39})",
            "product_type": r"(?i)\b(?:type\s+of\s+(?:cigarette|product)|product\s+type|sample\s+type|item\s+type)\s*[:#=-]?\s*(\S.+)$",
            "batch_size": r"(?i)\b(?:batch\s+size|lot\s+size|quantity)\s*[:#=-]?\s*(\d+(?:\.\d+)?\s*[A-Za-z.]+)",
            "requested_by": r"(?i)\b(?:original\s+request\s+made\s+by|requested\s+by|requester|requestor)\s*[:#=-]?\s*(\S.+)$",
            "specifications_written_by": r"(?i)\b(?:sample\s+specifications\s+written\s+by|specifications\s+(?:written\s+)?by|prepared\s+by)\s*[:#=-]?\s*(\S.+)$",
        }
        pattern = re.compile(value_patterns[logical])
        matches: list[tuple[VisionDocumentRecord, Any, str]] = []
        seen: set[tuple[str, str]] = set()
        for document in documents:
            if not document.ocr_lines:
                document.ocr_lines = ocr_line_reconstructor.reconstruct(document.ocr_words)
            lines = sorted(document.ocr_lines, key=lambda item: item.reading_order)
            for index, line in enumerate(lines):
                match = pattern.search(line.text)
                value = match.group(1).strip(" :#=-") if match else ""
                evidence_line = line
                if not value and self._line_has_requested_label(logical, line.text):
                    candidate = lines[index + 1] if index + 1 < len(lines) else None
                    if candidate and candidate.page == line.page:
                        raw_match = re.search(
                            r"(?:\+?\d[\d()./ -]{6,}\d|[A-Z0-9][A-Z0-9._/-]{2,39})",
                            candidate.text,
                            re.I,
                        )
                        if raw_match:
                            value = raw_match.group(0).strip()
                            evidence_line = candidate.model_copy(
                                update={
                                    "text": f"{line.text} {candidate.text}",
                                    "bounding_box": self._union_box(
                                        [line.bounding_box, candidate.bounding_box]
                                    ),
                                    "mean_confidence": round(
                                        (
                                            (line.mean_confidence or 0.0)
                                            + (candidate.mean_confidence or 0.0)
                                        )
                                        / 2,
                                        4,
                                    ),
                                }
                            )
                if not value:
                    continue
                if logical in {"phone_number", "fax_number"}:
                    validation = semantic_value_registry.validate(
                        SemanticValueType.PHONE,
                        value,
                        SemanticContext(label=line.text),
                    )
                    if not validation.valid:
                        continue
                key = (document.document_id, self._normalize(value))
                if key in seen:
                    continue
                seen.add(key)
                matches.append((document, evidence_line, value))
        matches.sort(
            key=lambda item: item[1].mean_confidence or 0.0,
            reverse=True,
        )
        return matches[:top_k]

    def _line_has_requested_label(self, logical: str, text: str) -> bool:
        labels = {
            "fax_number": ("fax", "facsimile", "telefax"),
            "phone_number": ("phone", "telephone", "tel"),
            "client_number": ("client",),
            "reference_number": ("reference", "ref"),
            "recipient_name": ("message to", "recipient", "attention", "attn", "to"),
            "sender_name": ("sender", "sent by", "from"),
            "company": ("company", "organisation", "organization", "firm"),
            "date": ("date", "dated"),
            "email": ("email", "e-mail"),
            "sample_number": ("sample no", "sample number", "sample id"),
            "product_type": ("type of cigarette", "product type", "sample type", "item type"),
            "batch_size": ("batch size", "lot size", "quantity"),
            "requested_by": ("original request made by", "requested by", "requester", "requestor"),
            "specifications_written_by": (
                "sample specifications written by",
                "specifications by",
                "prepared by",
            ),
        }
        normalized = text.lower()
        return any(label in normalized for label in labels.get(logical, ()))

    #: Words that carry no retrieval signal. Kept small on purpose: an
    #: over-eager stopword list silently removes the terms that make a question
    #: specific.
    question_stopwords: ClassVar[frozenset[str]] = frozenset(
        {
            "about",
            "and",
            "are",
            "does",
            "document",
            "for",
            "from",
            "have",
            "how",
            "is",
            "many",
            "much",
            "of",
            "the",
            "this",
            "to",
            "was",
            "were",
            "what",
            "which",
            "who",
        }
    )

    def _question_content_tokens(self, question: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", question.lower())
            if len(token) >= 3 and token not in self.question_stopwords
        }

    def _relevant_semantic_hits(self, question: str, hits: list[Any]) -> list[Any]:
        query_tokens = self._question_content_tokens(question)
        advanced_embedding = self.retrieval_index.embedding.adapter_id != "local-hashing-embedding"
        retained = []
        matched_tokens: set[str] = set()
        for hit in hits:
            chunk_tokens = set(re.findall(r"[a-z0-9]+", hit.chunk.text.lower()))
            overlap = query_tokens & chunk_tokens
            if hit.score < 0.12:
                continue
            if overlap or (advanced_embedding and hit.score >= 0.25):
                retained.append(hit)
                matched_tokens |= overlap
        if self._coverage_verdict(query_tokens, matched_tokens) == "insufficient":
            return []
        return retained

    #: A chunk sharing one incidental word with the question is related context,
    #: not an answer. Below this share of the question's content words the
    #: evidence is treated as insufficient and the system abstains.
    semantic_coverage_threshold: float = 0.5

    def _cascade_addresses_question(
        self,
        question: str,
        response: VisionAskResponse,
    ) -> bool:
        """Does the cascade result actually answer what was asked?

        "How many employees does the contractor have" has no employee evidence
        anywhere in the corpus. The cascade nonetheless matched a field literally
        named ``contractor`` and returned "contractor: Kestrel Instrumentation"
        as a *supported* answer. Sharing one word with the question is not
        answering it.
        """

        query_tokens = self._question_content_tokens(question)
        if not query_tokens:
            return True
        answer_tokens = set(re.findall(r"[a-z0-9]+", response.answer.lower()))
        for citation in response.citations:
            answer_tokens |= set(re.findall(r"[a-z0-9]+", (citation.source_text or "").lower()))
        matched = query_tokens & answer_tokens
        if not matched:
            return False
        if self._coverage_verdict(query_tokens, matched) == "insufficient":
            return False
        self._apply_coverage_verdict(question, response, matched)
        return True

    def _qualify_against_corpus(
        self,
        question: str,
        response: VisionAskResponse,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        """Downgrade when the question is partly about things the corpus lacks.

        A field lookup can match on one part of a question while the rest asks
        about a subject the documents never mention: "what was the share price on
        the invoice date" matched ``invoice_date`` and was returned as *fully
        supported*, even though nothing in the corpus concerns share prices.

        The check is corpus-grounded rather than heuristic: a content word that
        appears in no field, no field name and no evidence chunk of the documents
        in scope is a subject this corpus cannot speak to. Presence is measured,
        not guessed.
        """

        query_tokens = self._question_content_tokens(question)
        if not query_tokens:
            return response
        vocabulary: set[str] = set()
        for document in documents:
            for field in document.extracted_fields:
                vocabulary |= set(re.findall(r"[a-z0-9]+", field.name.lower()))
                vocabulary |= set(re.findall(r"[a-z0-9]+", str(field.value).lower()))
            for chunk in document.evidence_chunks:
                vocabulary |= set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
        absent = {token for token in query_tokens if token not in vocabulary}
        if len(absent) < 2 or len(absent) == len(query_tokens):
            # One unmatched word is usually phrasing ("which vendor *issued*
            # this"); two or more suggests the question is partly about a
            # subject the corpus does not cover. Nothing matched at all is a
            # different case, handled by the coverage verdict.
            return response
        response.answer_status = "partially_supported"
        response.reasoning_summary = (
            f"{response.reasoning_summary} The documents in scope contain no "
            f"evidence about {', '.join(sorted(absent))}."
        ).strip()
        return response

    def _coverage_verdict(self, query_tokens: set[str], matched_tokens: set[str]) -> str:
        """How well does the evidence account for what was asked?

        Returns ``"full"``, ``"partial"`` or ``"insufficient"``.

        This exists because of a real defect: "how many employees does the
        contractor have" retained a contract clause purely because the clause
        contains the word "contractor", and that clause was then returned as a
        *supported* answer. Sharing one word with a question is not answering it.

        Lexical coverage cannot decide this perfectly -- a legitimate paraphrase
        ("describe the arrangement") also leaves words unmatched -- so the middle
        case is reported as ``partially_supported`` rather than discarded.
        Overclaiming is the failure mode worth preventing.
        """

        if not query_tokens:
            # Nothing distinctive was asked; the score filter alone governs.
            return "full"
        if not matched_tokens:
            # Retained purely on dense similarity by an advanced embedding model.
            return "full"
        coverage = len(matched_tokens) / len(query_tokens)
        if coverage < self.semantic_coverage_threshold:
            return "insufficient"
        # One matched word out of three or more is a coincidence, not a topic.
        if len(query_tokens) >= 3 and len(matched_tokens) < 2:
            return "insufficient"
        return "full" if coverage >= 1.0 else "partial"

    def _apply_coverage_verdict(
        self,
        question: str,
        response: VisionAskResponse,
        matched_tokens: set[str],
    ) -> VisionAskResponse:
        """Downgrade an answer that does not account for the whole question."""

        query_tokens = self._question_content_tokens(question)
        verdict = self._coverage_verdict(query_tokens, matched_tokens)
        if verdict == "insufficient":
            return self._not_found_answer(requires_review=False)
        if verdict == "partial":
            response.answer_status = "partially_supported"
            unmatched = ", ".join(sorted(query_tokens - matched_tokens))
            response.reasoning_summary = (
                f"{response.reasoning_summary} The retrieved evidence does not "
                f"cover every part of the question ({unmatched})."
            ).strip()
        return response

    def _matched_question_tokens(self, question: str, hits: list[Any]) -> set[str]:
        query_tokens = self._question_content_tokens(question)
        matched: set[str] = set()
        for hit in hits:
            matched |= query_tokens & set(re.findall(r"[a-z0-9]+", hit.chunk.text.lower()))
        return matched
