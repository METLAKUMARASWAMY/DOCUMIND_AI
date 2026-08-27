"""The ask() surface: conversation state and non-numeric answer strategies.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from app.core.answer_policy import UNSUPPORTED_ANSWER
from app.core.config import settings
from app.core.security import hash_document, utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionAnswerDegradation,
    VisionAskCitation,
    VisionAskRequest,
    VisionAskResponse,
    VisionClarificationOption,
    VisionConversationState,
    VisionDocumentRecord,
    VisionProcessingStatus,
)
from app.services.vision.service_parts.constants import (
    CLARIFICATION_OPTIONS,
    LOGGER,
)


class AnsweringMixin:
    def ask(
        self,
        request: VisionAskRequest,
        tenant_id: str | None = None,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> VisionAskResponse:
        started_at = time.perf_counter()
        resolved_tenant = tenant_id or "tenant-local"
        resolved_user = user_id or "anonymous"
        resolved_conversation_id = (
            request.conversation_id or conversation_id or f"vision-conversation-{uuid4().hex[:12]}"
        )
        conversation = self._conversation_state(
            resolved_conversation_id,
            resolved_tenant,
            resolved_user,
        )
        question = request.question.strip()
        query_type = self.query_router.route(question)

        pending = dict(conversation.pending_clarification)
        if query_type == "clarification_response":
            request = self._request_with_conversation_scope(request, pending)
            selected_option = self._selected_clarification_option(
                request.clarification_option,
                question,
                pending,
            )
            if not pending:
                documents, effective_scope = self._ask_scope_documents(request, tenant_id)
                response = VisionAskResponse(
                    answer=(
                        "I do not have a pending clarification for this conversation. "
                        "Please ask the document question again."
                    ),
                    answer_status="clarification_needed",
                    query_type="clarification_response",
                    reasoning_summary="An acknowledgement cannot be resolved without a pending option set.",
                    requires_review=False,
                    retriever="not_used",
                )
                return self._finish_ask(
                    request,
                    response,
                    documents,
                    effective_scope,
                    conversation,
                    started_at,
                    question,
                    tenant_id,
                )
            if question.lower().rstrip(".! ") in {"no", "n"}:
                conversation.pending_clarification = {}
                documents, effective_scope = self._ask_scope_documents(request, tenant_id)
                response = VisionAskResponse(
                    answer="Okay. Ask a more specific question whenever you are ready.",
                    answer_status="supported",
                    query_type="clarification_response",
                    reasoning_summary="The pending clarification was cancelled without document retrieval.",
                    requires_review=False,
                    retriever="not_used",
                )
                return self._finish_ask(
                    request,
                    response,
                    documents,
                    effective_scope,
                    conversation,
                    started_at,
                    question,
                    tenant_id,
                )
            if selected_option is None:
                documents, effective_scope = self._ask_scope_documents(request, tenant_id)
                response = VisionAskResponse(
                    answer="Please select which type of safety check you want me to perform.",
                    answer_status="clarification_needed",
                    query_type="clarification_response",
                    clarification_options=list(CLARIFICATION_OPTIONS),
                    reasoning_summary=(
                        "The acknowledgement preserved the pending clarification; it did not "
                        "become an independent semantic query."
                    ),
                    requires_review=False,
                    retriever="not_used",
                )
                return self._finish_ask(
                    request,
                    response,
                    documents,
                    effective_scope,
                    conversation,
                    started_at,
                    question,
                    tenant_id,
                )
            question = selected_option.resolved_question
            query_type = selected_option.query_type
            conversation.pending_clarification = {}

        if query_type == "conversation_follow_up":
            request = self._request_with_selected_conversation_scope(request, conversation)
            documents, effective_scope = self._ask_scope_documents(request, tenant_id)
            response = self._conversation_follow_up_response(question, conversation)
            return self._finish_ask(
                request,
                response,
                documents,
                effective_scope,
                conversation,
                started_at,
                conversation.last_resolved_question or question,
                tenant_id,
            )

        documents, effective_scope = self._ask_scope_documents(request, tenant_id)
        conversation.selected_scope = self._conversation_scope(request, effective_scope)
        casual_responses = {
            "greeting": VisionAskResponse(
                answer=(
                    "Hello. Ask me about extracted fields, tables, clauses, validation "
                    "issues, risk indicators or relationships in the selected documents."
                ),
                answer_status="supported",
                query_type="greeting",
                reasoning_summary="Conversational greeting; no document retrieval was run.",
                requires_review=False,
                retriever="not_used",
            ),
            "help": VisionAskResponse(
                answer=(
                    "Ask about extracted fields, tables, clauses, calculations, visual "
                    "elements, validation issues, risk indicators or document relationships."
                ),
                answer_status="supported",
                query_type="help",
                reasoning_summary="Help response; no document retrieval was run.",
                requires_review=False,
                retriever="not_used",
            ),
            "ambiguous_question": VisionAskResponse(
                answer=(
                    "Do you want me to check document quality, financial consistency, "
                    "suspicious indicators, signature presence or approval status?"
                ),
                answer_status="clarification_needed",
                query_type="ambiguous_question",
                reasoning_summary="The requested meaning is too broad for an evidence-bound answer.",
                requires_review=False,
                retriever="not_used",
                clarification_options=list(CLARIFICATION_OPTIONS),
            ),
            "unsupported_question": VisionAskResponse(
                answer=(
                    "That request is outside the available document evidence. Ask about "
                    "fields, tables, clauses, calculations, visual elements, risks or relationships."
                ),
                answer_status="not_found",
                query_type="unsupported_question",
                reasoning_summary="No document retrieval was run for an unsupported request.",
                requires_review=False,
                retriever="not_used",
            ),
        }
        if query_type in casual_responses:
            if query_type == "ambiguous_question":
                conversation.pending_clarification = {
                    "clarification_id": f"clarification-{uuid4().hex[:12]}",
                    "original_question": question,
                    "options": [item.model_dump(mode="json") for item in CLARIFICATION_OPTIONS],
                    "selected_scope": self._conversation_scope(request, effective_scope),
                    "created_at": utc_now_iso(),
                }
            return self._finish_ask(
                request,
                casual_responses[query_type],
                documents,
                effective_scope,
                conversation,
                started_at,
                question,
                tenant_id,
            )
        if not documents:
            response = self._not_found_answer(requires_review=False)
            response.query_type = query_type
            return self._finish_ask(
                request,
                response,
                [],
                effective_scope,
                conversation,
                started_at,
                question,
                tenant_id,
            )

        resolved_tenant = tenant_id or documents[0].tenant_id
        for document in documents:
            self.retrieval_index.upsert_document(document.document_id, document.evidence_chunks)
        handlers = {
            "structured_field_lookup": self._answer_structured,
            "structured_lookup": self._answer_structured,
            "ocr_lookup": self._answer_structured,
            "semantic_question": self._answer_semantic,
            "relationship_question": self._answer_relationship,
            "arithmetic_question": self._answer_arithmetic,
            "visual_question": self._answer_visual,
            "comparison_question": self._answer_comparison,
            "risk_question": self._answer_risk,
            "quality_question": self._answer_quality,
            "approval_question": self._answer_approval,
            "safety_question": self._answer_safety,
        }
        response = handlers[query_type](
            question,
            documents,
            request.top_k,
            resolved_tenant,
        )
        if response.answer_status == "not_found" and query_type not in {
            "structured_field_lookup",
            "structured_lookup",
            "ocr_lookup",
            "arithmetic_question",
        }:
            cascade_response = self._answer_structured(
                question,
                documents,
                request.top_k,
                resolved_tenant,
            )
            # The cascade exists to find evidence the primary handler missed, not
            # to substitute something loosely related for an answer. If the
            # fallback does not actually address the question, abstaining is the
            # honest outcome.
            if cascade_response.answer_status != "not_found" and self._cascade_addresses_question(
                question,
                cascade_response,
            ):
                response = cascade_response
                response.reasoning_summary = (
                    f"{response.reasoning_summary} The primary {query_type.replace('_', ' ')} "
                    "source had no match, so the remaining evidence cascade was checked."
                )
        response.query_type = query_type
        response.resolved_question = question if question != request.question.strip() else None
        if response.answer_status == "not_found" and effective_scope in {
            "current_document",
            "selected_documents",
        }:
            broader = [
                document
                for document in self.list_documents(tenant_id)
                if document.document_id not in {item.document_id for item in documents}
            ]
            if self._question_has_document_evidence(question, broader):
                requested = self._requested_field_names(question)
                label = (
                    next(iter(sorted(requested))).replace("_", " ")
                    if requested
                    else "matching evidence"
                )
                response.answer = (
                    f"The selected document does not contain a {label} field. "
                    "Search all accessible documents instead?"
                )
                response.answer_status = "clarification_needed"
                response.reasoning_summary = (
                    "No evidence matched the selected scope; matching evidence exists outside it."
                )
                response.broader_scope_available = True
        return self._finish_ask(
            request,
            response,
            documents,
            effective_scope,
            conversation,
            started_at,
            question,
            tenant_id,
        )

    def _finish_ask(
        self,
        request: VisionAskRequest,
        response: VisionAskResponse,
        documents: list[VisionDocumentRecord],
        effective_scope: str,
        conversation: VisionConversationState,
        started_at: float,
        resolved_question: str,
        tenant_id: str | None,
    ) -> VisionAskResponse:
        response.scope = self._ask_scope_payload(effective_scope, documents)
        response.conversation_id = conversation.conversation_id
        if resolved_question != request.question.strip():
            response.resolved_question = resolved_question
        self._attach_retrieval_trace(
            response,
            resolved_question,
            documents,
            tenant_id or conversation.organisation_id,
            started_at,
        )
        conversation.last_query_type = response.query_type
        if response.answer_status != "clarification_needed":
            conversation.last_resolved_question = resolved_question
            conversation.last_answer = response.model_dump(mode="json")
        return self._record_question(
            request,
            response,
            documents,
            tenant_id=tenant_id,
            conversation=conversation,
        )

    def _conversation_state(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
    ) -> VisionConversationState:
        cache_key = self._conversation_cache_key(conversation_id, tenant_id, user_id)
        state = self.conversations.get(cache_key)
        if state is None:
            payload = self._read_json(self._state_file("conversations", cache_key))
            if payload is None:
                payload = self.persistence.get_conversation(
                    conversation_id,
                    tenant_id,
                    user_id,
                )
            try:
                state = VisionConversationState.model_validate(payload) if payload else None
            except ValueError:
                state = None
        if state is None or self._conversation_expired(state):
            now = datetime.now(UTC)
            state = VisionConversationState(
                conversation_id=conversation_id,
                user_id=user_id,
                organisation_id=tenant_id,
                expires_at=(now + timedelta(minutes=30)).isoformat(),
                updated_at=now.isoformat(),
            )
        self.conversations[cache_key] = state
        return state

    def _persist_conversation(self, state: VisionConversationState) -> None:
        now = datetime.now(UTC)
        state.updated_at = now.isoformat()
        state.expires_at = (now + timedelta(minutes=30)).isoformat()
        state.turns = state.turns[-100:]
        cache_key = self._conversation_cache_key(
            state.conversation_id,
            state.organisation_id,
            state.user_id,
        )
        self.conversations[cache_key] = state
        payload = state.model_dump(mode="json")
        self._write_json(self._state_file("conversations", cache_key), payload)
        self.persistence.save_conversation(payload)

    def _conversation_cache_key(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
    ) -> str:
        identity = f"{tenant_id}|{user_id}|{conversation_id}"
        return hash_document(identity.encode())

    def _conversation_expired(self, state: VisionConversationState) -> bool:
        try:
            expires_at = datetime.fromisoformat(state.expires_at.replace("Z", "+00:00"))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            return expires_at <= datetime.now(UTC)
        except ValueError:
            return True

    def _conversation_scope(
        self,
        request: VisionAskRequest,
        effective_scope: str,
    ) -> dict[str, Any]:
        return {
            "scope": effective_scope,
            "document_id": request.document_id,
            "document_ids": list(request.document_ids),
            "top_k": request.top_k,
        }

    def _request_with_conversation_scope(
        self,
        request: VisionAskRequest,
        pending: dict[str, Any],
    ) -> VisionAskRequest:
        selected = dict(pending.get("selected_scope") or {})
        if not selected:
            return request
        return request.model_copy(
            update={
                "scope": selected.get("scope", request.scope),
                "document_id": selected.get("document_id"),
                "document_ids": list(selected.get("document_ids") or []),
                "top_k": int(selected.get("top_k") or request.top_k),
            }
        )

    def _request_with_selected_conversation_scope(
        self,
        request: VisionAskRequest,
        conversation: VisionConversationState,
    ) -> VisionAskRequest:
        selected = dict(conversation.selected_scope or {})
        if not selected:
            return request
        return request.model_copy(
            update={
                "scope": selected.get("scope", request.scope),
                "document_id": selected.get("document_id"),
                "document_ids": list(selected.get("document_ids") or []),
                "top_k": int(selected.get("top_k") or request.top_k),
            }
        )

    def _conversation_follow_up_response(
        self,
        question: str,
        conversation: VisionConversationState,
    ) -> VisionAskResponse:
        previous = dict(conversation.last_answer or {})
        citations = [
            VisionAskCitation.model_validate(item)
            for item in previous.get("citations", [])
            if isinstance(item, dict)
        ]
        if not previous or not citations:
            response = self._not_found_answer(requires_review=False)
            response.query_type = "conversation_follow_up"
            response.reasoning_summary = (
                "No prior source-backed answer exists in this conversation."
            )
            return response
        normalized = " ".join(question.lower().split())
        if "page" in normalized or "where" in normalized:
            pages = ", ".join(
                f"{citation.filename or citation.document_id}, page {citation.page_number}"
                for citation in citations
            )
            answer = f"The cited evidence is on {pages}."
        elif "evidence" in normalized or normalized.rstrip("?.! ") == "source":
            answer = "Evidence: " + "; ".join(
                f"{citation.filename or citation.document_id}, page {citation.page_number}: "
                f"{citation.source_text}"
                for citation in citations
            )
        else:
            answer = (
                str(
                    previous.get("reasoning_summary")
                    or "The answer is supported by the cited evidence."
                )
                + " Evidence: "
                + "; ".join(citation.source_text for citation in citations)
            )
        return VisionAskResponse(
            answer=answer,
            answer_status=str(previous.get("answer_status") or "supported"),
            query_type="conversation_follow_up",
            citations=citations,
            confidence=float(previous.get("confidence") or 0.0),
            reasoning_summary="Resolved the follow-up against the preceding source-backed answer.",
            requires_review=bool(previous.get("requires_review")),
            generator_name="conversation-evidence-resolver",
            generator_version="v1",
            retriever="conversation_citation_state",
            retrieval_path="prior_turn_citations",
            resolved_question=conversation.last_resolved_question or None,
        )

    def _selected_clarification_option(
        self,
        requested_option: str | None,
        question: str,
        pending: dict[str, Any],
    ) -> VisionClarificationOption | None:
        allowed_ids = {
            str(item.get("option_id"))
            for item in pending.get("options", [])
            if isinstance(item, dict) and item.get("option_id")
        }
        candidate = " ".join((requested_option or question).lower().split()).rstrip(".! ")
        for option in CLARIFICATION_OPTIONS:
            if allowed_ids and option.option_id not in allowed_ids:
                continue
            if candidate in {
                option.option_id.lower(),
                option.label.lower(),
            }:
                return option
        return None

    def _answer_structured(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        requested_names = self._requested_field_names(question)
        matches = self._field_matches(
            question.lower(), self._documents_for_question(question, documents)
        )
        if requested_names:
            matches = self._filter_to_requested(matches, requested_names)
        matches.sort(
            key=lambda field: (
                field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited},
                field.confidence or 0.0,
            ),
            reverse=True,
        )
        matches = matches[:top_k]
        if matches:
            logical_name = self._logical_requested_field(question)
            if logical_name and len(matches) == 1:
                shown = self._mask_sensitive(
                    matches[0].name,
                    matches[0].reviewed_value or matches[0].value,
                )
                answer = f"The {logical_name.replace('_', ' ')} shown is {shown}."
            else:
                answer = ", ".join(
                    f"{field.name}: {self._mask_sensitive(field.name, field.reviewed_value or field.value)}"
                    for field in matches
                )
            structured = VisionAskResponse(
                answer=answer,
                answer_status="supported",
                citations=[self._field_citation(field) for field in matches],
                confidence=self._mean_available_field_confidence(matches) or 0.0,
                reasoning_summary="Returned extracted structured fields and their source evidence.",
                requires_review=any(
                    field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
                    or not field.calibrated
                    for field in matches
                ),
                retriever="persisted-structured-field-store",
                retrieval_path="structured_field_lookup",
            )
            return self._qualify_against_corpus(question, structured, documents)

        ocr_matches = self._ocr_lookup(question, documents, top_k)
        if ocr_matches:
            logical_name = self._logical_requested_field(question) or "value"
            citations = [
                VisionAskCitation(
                    document_id=document.document_id,
                    filename=document.filename,
                    page_number=line.page,
                    source_text=line.text,
                    bounding_boxes=[line.bounding_box],
                    retrieval_score=line.mean_confidence,
                    retrieval_backend="ocr-line-retriever-v1",
                    model_confidence=line.mean_confidence or 0.0,
                    model_name="ocr-line-retriever",
                    model_version="v1",
                    processing_timestamp=document.updated_at,
                    reviewer_status=ReviewerStatus.unverified,
                )
                for document, line, _ in ocr_matches
            ]
            values = list(dict.fromkeys(value for _, _, value in ocr_matches))
            return VisionAskResponse(
                answer=(
                    f"The {logical_name.replace('_', ' ')} shown is {values[0]}."
                    if len(values) == 1
                    else f"The {logical_name.replace('_', ' ')} values shown are {', '.join(values)}."
                ),
                answer_status="supported",
                citations=citations,
                confidence=round(
                    sum(line.mean_confidence or 0.0 for _, line, _ in ocr_matches)
                    / len(ocr_matches),
                    4,
                ),
                reasoning_summary="Matched a field alias to source-backed OCR label-value evidence.",
                requires_review=any(
                    (line.mean_confidence or 0.0) < settings.ocr_low_confidence_threshold
                    for _, line, _ in ocr_matches
                ),
                retriever="ocr-line-retriever-v1",
                retrieval_path="ocr_label_value_fallback",
            )

        graph = self._answer_relationship(
            question,
            documents,
            top_k,
            tenant_id,
            source_fields=requested_names or None,
        )
        if graph.answer_status != "not_found":
            graph.retrieval_path = "graph_lookup"
            return graph
        # Dense semantic similarity cannot establish an exact field value. It is
        # still probed by the retrieval trace, but a related prose chunk must not
        # be substituted for a missing requested field.
        if requested_names:
            response = self._not_found_answer(requires_review=False)
            response.retrieval_path = "cascade_exhausted"
            return response
        semantic = self._answer_semantic(question, documents, top_k, tenant_id)
        if semantic.answer_status != "not_found":
            semantic.retrieval_path = "local_chunk_retrieval"
            return semantic
        response = self._not_found_answer(requires_review=False)
        response.retrieval_path = "cascade_exhausted"
        return response

    def _answer_semantic(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        hits = self.retrieval_index.search(
            question,
            tenant_id=tenant_id,
            top_k=min(max(top_k, 5), 8),
            document_ids={document.document_id for document in documents},
        )
        hits = self._relevant_semantic_hits(question, hits)
        if not hits:
            return self._not_found_answer(requires_review=False)
        matched_tokens = self._matched_question_tokens(question, hits)
        citations = [
            VisionAskCitation(
                document_id=hit.chunk.document_id,
                filename=self.get_document(hit.chunk.document_id, tenant_id).filename,
                page_number=hit.chunk.page_number,
                source_text=hit.chunk.text,
                bounding_boxes=hit.chunk.bounding_boxes,
                chunk_id=hit.chunk.chunk_id,
                retrieval_score=round(max(0.0, min(1.0, hit.score)), 4),
                retrieval_backend=hit.backend,
                model_confidence=hit.chunk.confidence,
                model_name=hit.chunk.model_name,
                model_version=hit.chunk.model_version,
                processing_timestamp=self.get_document(hit.chunk.document_id, tenant_id).updated_at,
                reviewer_status=hit.chunk.review_status,
            )
            for hit in hits
        ]
        response = VisionAskResponse(
            answer=" ".join(hit.chunk.text for hit in hits)[:1200],
            answer_status="supported",
            citations=citations,
            confidence=round(sum(hit.chunk.confidence for hit in hits) / len(hits), 4),
            reasoning_summary="Returned the highest-ranked structure-aware evidence blocks.",
            requires_review=any(
                hit.chunk.review_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
                for hit in hits
            ),
            retriever=hits[0].backend,
        )
        response = self._apply_coverage_verdict(question, response, matched_tokens)
        if response.answer_status == "not_found":
            return response
        return self._apply_grounded_generator(question, hits, response)

    def _apply_grounded_generator(
        self,
        question: str,
        hits: list[Any],
        fallback: VisionAskResponse,
    ) -> VisionAskResponse:
        generator = self.generator_router.active()
        if generator is None:
            return fallback
        evidence = [
            {
                "id": hit.chunk.chunk_id,
                "text": hit.chunk.text,
                "document_id": hit.chunk.document_id,
                "page_number": hit.chunk.page_number,
                "review_status": hit.chunk.review_status.value,
            }
            for hit in hits
        ]
        try:
            generated = generator.generate(question, evidence)
        except Exception as exc:
            # A generator failure is a real degradation, not a no-op. The composed
            # evidence answer is still correct and cited, but the user is told that
            # the richer generator did not run.
            LOGGER.warning(
                "grounded generator failed; serving the composed evidence answer",
                extra={
                    "context": {
                        "component": "grounded_generator",
                        "generator": generator.adapter_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                },
            )
            fallback.degradations.append(
                VisionAnswerDegradation(
                    component=generator.adapter_id,
                    stage="answer_generation",
                    detail=(
                        f"The grounded generator raised {type(exc).__name__} and was "
                        "skipped. The answer below is composed directly from retrieved "
                        "evidence and remains fully cited."
                    ),
                    impact="generator_skipped",
                )
            )
            return fallback
        citation_by_id = {
            citation.chunk_id: citation
            for citation in fallback.citations
            if citation.chunk_id is not None
        }
        selected = [
            citation_by_id[item] for item in generated["citation_ids"] if item in citation_by_id
        ]
        if generated["answer_status"] != "not_found" and not selected:
            # The generator claimed support but cited nothing that resolves to
            # retrieved evidence. Rejecting it is the grounding contract working;
            # discarding it silently is not. This is also the signal that would fire
            # on a prompt-injection attempt from document content.
            LOGGER.warning(
                "generated answer rejected: no citation resolved to retrieved evidence",
                extra={
                    "context": {
                        "component": "grounding_validator",
                        "generator": generator.adapter_id,
                        "claimed_status": generated["answer_status"],
                        "claimed_citation_count": len(generated["citation_ids"]),
                        "resolved_citation_count": 0,
                    }
                },
            )
            fallback.degradations.append(
                VisionAnswerDegradation(
                    component="grounding_validator",
                    stage="grounding_validation",
                    detail=(
                        "The generated answer cited no evidence that resolves to the "
                        "retrieved set, so it was rejected. The answer below is composed "
                        "directly from retrieved evidence."
                    ),
                    impact="generated_answer_rejected",
                )
            )
            return fallback
        return VisionAskResponse(
            answer=generated["answer"]
            or "The available documents do not contain sufficient evidence to answer this question.",
            answer_status=generated["answer_status"],
            query_type=fallback.query_type,
            citations=selected,
            confidence=fallback.confidence if selected else 0.0,
            reasoning_summary=generated["reasoning_summary"],
            requires_review=fallback.requires_review or generated["requires_review"],
            retriever=fallback.retriever,
            generator_name=generator.adapter_id,
            generator_version=generator.model_version,
        )

    def _answer_visual(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del tenant_id
        detections = self._matching_visual_detections(question, documents)[:top_k]
        if not detections:
            return self._not_found_answer(requires_review=False)
        citations = [
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=detection.page_number,
                source_text=f"Visual candidate: {detection.detection_class}",
                bounding_boxes=[detection.bounding_box],
                model_confidence=detection.confidence,
                model_name=detection.model_name,
                model_version=detection.model_version,
                processing_timestamp=document.updated_at,
                reviewer_status=detection.reviewer_status,
                extraction_status=detection.reviewer_status.value,
                evidence_type="visual_detection",
            )
            for document, detection in detections
        ]
        return VisionAskResponse(
            answer=(
                f"Detected {len(detections)} visual candidate(s): "
                + ", ".join(detection.detection_class for _, detection in detections)
                + ". Detection does not establish authenticity."
            ),
            answer_status="supported",
            citations=citations,
            confidence=round(sum(item.confidence for _, item in detections) / len(detections), 4),
            reasoning_summary="Returned visual detector candidates and their page regions.",
            requires_review=True,
            retriever="stored-visual-detections",
            retrieval_path="visual_detection_lookup",
        )

    def _answer_quality(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del question, tenant_id
        unverified = [
            field
            for document in documents
            for field in document.extracted_fields
            if field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
        ]
        missing = sum(
            len(document.field_completeness.missing_required_fields) for document in documents
        )
        disputed = sum(len(document.field_completeness.disputed_fields) for document in documents)
        quality_scores = [
            report.image_quality_score for document in documents for report in document.quality
        ]
        warnings = [
            (document, warning)
            for document in documents
            for warning in document.processing_warnings
        ]
        citations = [self._field_citation(field) for field in unverified[:top_k]]
        citations.extend(
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=evidence.page_number,
                source_text=evidence.source_text,
                bounding_boxes=[evidence.bounding_box] if evidence.bounding_box else [],
                evidence_id=evidence.evidence_id,
                model_confidence=evidence.model_confidence,
                model_name=evidence.model_name,
                model_version=evidence.model_version,
                processing_timestamp=evidence.processing_timestamp,
                reviewer_status=evidence.reviewer_status,
                extraction_status=evidence.reviewer_status.value,
                evidence_type="processing_warning",
            )
            for document, warning in warnings
            for evidence in warning.evidence[:1]
        )
        citations.extend(
            self._document_state_citation(
                document,
                (
                    f"Persisted data-quality summary: {len(document.extracted_fields)} extracted "
                    f"field(s), {len(document.field_completeness.missing_required_fields)} "
                    "missing required field(s)."
                ),
                "data_quality_summary",
            )
            for document in documents
        )
        citations = self._deduplicate_citations(citations)[: max(top_k, 8)]
        if not citations:
            return self._not_found_answer(requires_review=False)
        quality_label = (
            f"mean page-quality score {sum(quality_scores) / len(quality_scores):.0%}"
            if quality_scores
            else "page-quality score not available"
        )
        requires_review = bool(unverified or missing or disputed or warnings)
        return VisionAskResponse(
            answer=(
                f"Processing quality: {quality_label}; {len(warnings)} source-linked warning(s). "
                f"Data quality: {len(unverified)} unverified field(s), {missing} unresolved "
                f"required field(s), and {disputed} disputed field(s). "
                + (
                    "Human review is required before relying on the document."
                    if requires_review
                    else "No unresolved data-quality item is recorded in the selected scope."
                )
            ),
            answer_status="partially_supported" if requires_review else "supported",
            citations=citations,
            confidence=self._mean_available_field_confidence(unverified) or 1.0,
            reasoning_summary=(
                "Combined persisted page-quality results, processing warnings, field verification "
                "states and schema completeness without making a fraud or legal determination."
            ),
            requires_review=requires_review,
            retriever="deterministic-document-quality-summary-v1",
            retrieval_path="multi_source_quality_summary",
        )

    def _answer_approval(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del question, tenant_id
        selected = documents[:top_k]
        if not selected:
            return self._not_found_answer(requires_review=False)
        citations = [
            self._document_state_citation(
                document,
                (
                    f"Persisted workflow status for {document.filename}: {document.status.value}; "
                    f"unresolved required fields: "
                    f"{len(document.field_completeness.missing_required_fields)}."
                ),
                "workflow_state",
            )
            for document in selected
        ]
        return VisionAskResponse(
            answer="; ".join(
                (
                    f"{document.filename}: {document.status.value.replace('_', ' ').title()}"
                    f" ({len(document.field_completeness.missing_required_fields)} unresolved "
                    "required field(s))"
                )
                for document in selected
            ),
            answer_status="supported",
            citations=citations,
            confidence=1.0,
            reasoning_summary="Returned persisted workflow state and deterministic completeness counts.",
            requires_review=any(
                document.status != VisionProcessingStatus.approved for document in selected
            ),
            retriever="persisted-approval-workflow-v1",
            retrieval_path="approval_status_lookup",
        )

    def _answer_safety(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del question, tenant_id
        if not documents:
            return self._not_found_answer(requires_review=False)
        unverified = [
            field
            for document in documents
            for field in document.extracted_fields
            if field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
        ]
        missing = sum(
            len(document.field_completeness.missing_required_fields) for document in documents
        )
        linked_risks = [
            (document, risk)
            for document in documents
            for risk in document.risk_indicators
            if risk.evidence
        ]
        manipulation_regions = [
            (document, analysis, region)
            for document in documents
            for analysis in document.tamper_analysis
            for region in analysis.suspicious_regions
        ]
        confirmed_fraud = any(
            document.status == VisionProcessingStatus.confirmed_fraud for document in documents
        )
        statuses = ", ".join(
            f"{document.filename}: {document.status.value.replace('_', ' ').title()}"
            for document in documents
        )
        citations = [self._field_citation(field) for field in unverified[:top_k]]
        citations.extend(
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=evidence.page_number,
                source_text=evidence.source_text,
                bounding_boxes=[evidence.bounding_box] if evidence.bounding_box else [],
                evidence_id=evidence.evidence_id,
                model_confidence=evidence.model_confidence,
                model_name=evidence.model_name,
                model_version=evidence.model_version,
                processing_timestamp=evidence.processing_timestamp,
                reviewer_status=evidence.reviewer_status,
                extraction_status=evidence.reviewer_status.value,
                evidence_type="risk_indicator",
            )
            for document, risk in linked_risks
            for evidence in risk.evidence[:1]
        )
        citations.extend(
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=analysis.page_number,
                source_text=(
                    f"Visual-manipulation screening candidate: {region.manipulation_class}. "
                    "This is not a fraud determination."
                ),
                bounding_boxes=[region.bounding_box],
                evidence_id=region.region_id,
                model_confidence=region.screening_score,
                model_name=analysis.model_name,
                model_version=analysis.model_version,
                processing_timestamp=document.updated_at,
                reviewer_status=analysis.reviewer_status,
                extraction_status=analysis.reviewer_status.value,
                evidence_type="visual_manipulation_screening",
            )
            for document, analysis, region in manipulation_regions
        )
        citations.extend(
            self._document_state_citation(
                document, f"Persisted workflow status: {document.status.value}.", "workflow_state"
            )
            for document in documents
        )
        citations = self._deduplicate_citations(citations)[: max(top_k, 8)]
        requires_review = any(
            document.status != VisionProcessingStatus.approved for document in documents
        ) or bool(unverified or missing or linked_risks or manipulation_regions)
        determination = (
            "A reviewer-confirmed fraud workflow status is recorded."
            if confirmed_fraud
            else "No confirmed fraud determination is available."
        )
        return VisionAskResponse(
            answer=(
                f"{determination} Processing quality: {sum(len(d.quality) for d in documents)} "
                f"page-quality record(s). Data-quality issues: {len(unverified)} unverified "
                f"field(s) and {missing} unresolved required field(s). Business-rule indicators: "
                f"{len(linked_risks)} source-linked indicator(s). Visual-manipulation indicators: "
                f"{len(manipulation_regions)} screening candidate(s). Approval status: {statuses}. "
                "This evidence summary is not a legal, authenticity or fraud conclusion."
            ),
            answer_status="partially_supported" if requires_review else "supported",
            citations=citations,
            confidence=self._mean_available_field_confidence(unverified) or 1.0,
            reasoning_summary=(
                "Separated processing quality, data quality, business-rule indicators, visual "
                "screening and workflow status using only persisted evidence and deterministic counts."
            ),
            requires_review=requires_review,
            retriever="multi-source-safety-summary-v1",
            retrieval_path="multi_source_safety_summary",
        )

    def _document_state_citation(
        self,
        document: VisionDocumentRecord,
        source_text: str,
        evidence_type: str,
    ) -> VisionAskCitation:
        return VisionAskCitation(
            document_id=document.document_id,
            filename=document.filename,
            page_number=1,
            source_text=source_text,
            bounding_boxes=[],
            evidence_id=f"state:{document.document_id}:{evidence_type}",
            retrieval_score=1.0,
            retrieval_backend="persisted-document-state-v1",
            model_confidence=1.0,
            model_name="persisted-document-state",
            model_version="v1",
            processing_timestamp=document.updated_at,
            reviewer_status=(
                ReviewerStatus.accepted
                if document.status == VisionProcessingStatus.approved
                else ReviewerStatus.unverified
            ),
            extraction_status="persisted",
            evidence_type=evidence_type,
        )

    def _deduplicate_citations(
        self,
        citations: list[VisionAskCitation],
    ) -> list[VisionAskCitation]:
        unique: dict[str, VisionAskCitation] = {}
        for citation in citations:
            key = str(
                citation.evidence_id
                or citation.chunk_id
                or f"{citation.document_id}:{citation.page_number}:{citation.source_text}"
            )
            unique.setdefault(key, citation)
        return list(unique.values())

    def _answer_risk(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del question, tenant_id
        matches = [(document, risk) for document in documents for risk in document.risk_indicators][
            :top_k
        ]
        if not matches:
            response = self._not_found_answer(requires_review=False)
            response.reasoning_summary = (
                "No stored business or visual risk indicator matched the scope."
            )
            response.retriever = "stored-risk-registry"
            return response
        citations = [
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=evidence.page_number,
                source_text=evidence.source_text,
                bounding_boxes=[evidence.bounding_box] if evidence.bounding_box else [],
                evidence_id=evidence.evidence_id,
                model_confidence=evidence.model_confidence,
                model_name=evidence.model_name,
                model_version=evidence.model_version,
                processing_timestamp=evidence.processing_timestamp,
                reviewer_status=evidence.reviewer_status,
            )
            for document, risk in matches
            for evidence in risk.evidence[:2]
        ]
        if not citations:
            response = self._not_found_answer(requires_review=False)
            response.reasoning_summary = "Stored risk records had no source-linked evidence."
            response.retriever = "stored-risk-registry"
            return response
        confidence_values = [risk.confidence for _, risk in matches if risk.confidence is not None]
        return VisionAskResponse(
            answer="; ".join(
                f"{risk.severity.value}: {risk.title}. {risk.recommended_action}"
                for _, risk in matches
            ),
            answer_status=(
                "supported"
                if all(risk.reviewer_status == ReviewerStatus.accepted for _, risk in matches)
                else "partially_supported"
            ),
            citations=citations,
            confidence=(
                round(sum(confidence_values) / len(confidence_values), 4)
                if confidence_values
                else 0.0
            ),
            reasoning_summary="Returned stored risk indicators and their source-linked evidence.",
            requires_review=any(
                risk.reviewer_status != ReviewerStatus.accepted for _, risk in matches
            ),
            retriever="stored-risk-registry",
        )

    def _answer_comparison(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del tenant_id
        if len(documents) < 2:
            return self._not_found_answer(requires_review=False)
        requested = self._requested_field_names(question)
        by_name: dict[str, list[ExtractedVisionField]] = {}
        for document in documents:
            for field in document.extracted_fields:
                if requested and field.name not in requested:
                    continue
                by_name.setdefault(field.name, []).append(field)
        comparable = [
            (name, fields)
            for name, fields in by_name.items()
            if len({field.evidence.document_id for field in fields}) >= 2
        ][:top_k]
        if not comparable:
            return self._not_found_answer(requires_review=False)
        fields = [field for _, values in comparable for field in values]
        return VisionAskResponse(
            answer="; ".join(
                f"{name}: "
                + ", ".join(
                    f"{field.evidence.document_id}={self._mask_sensitive(name, field.reviewed_value or field.value)}"
                    for field in values
                )
                for name, values in comparable
            ),
            answer_status="partially_supported",
            citations=[self._field_citation(field) for field in fields],
            confidence=self._mean_available_field_confidence(fields) or 0.0,
            reasoning_summary="Compared like-named structured fields across documents.",
            requires_review=any(
                field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
                for field in fields
            ),
            retriever="cross-document-field-comparison",
        )

    def _record_question(
        self,
        request: VisionAskRequest,
        response: VisionAskResponse,
        documents: list[VisionDocumentRecord],
        tenant_id: str | None = None,
        conversation: VisionConversationState | None = None,
    ) -> VisionAskResponse:
        question_id = f"vision-question-{uuid4().hex[:10]}"
        response.question_id = question_id
        self.questions[question_id] = {
            "question_id": question_id,
            "question": request.question,
            "document_id": request.document_id,
            "document_ids": [document.document_id for document in documents],
            "top_k": request.top_k,
            "answer": response.model_dump(mode="json"),
            "feedback": [],
            "tenant_id": tenant_id or (documents[0].tenant_id if documents else "tenant-local"),
            "created_at": utc_now_iso(),
        }
        self._persist_question(question_id, self.questions[question_id])
        if conversation is not None:
            conversation.turns.extend(
                [
                    {
                        "role": "user",
                        "question_id": question_id,
                        "content": request.question,
                        "clarification_option": request.clarification_option,
                        "created_at": self.questions[question_id]["created_at"],
                    },
                    {
                        "role": "assistant",
                        "question_id": question_id,
                        "answer_status": response.answer_status,
                        "query_type": response.query_type,
                        "content": response.answer,
                        "retrieval_trace_id": response.retrieval_trace_id,
                        "created_at": self.questions[question_id]["created_at"],
                    },
                ]
            )
            self._persist_conversation(conversation)
        return response

    def get_question(self, question_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        question = self.questions.get(question_id) or self._load_question(question_id)
        if not question and tenant_id:
            question = self.persistence.get_question(question_id, tenant_id)
            if question:
                self.questions[question_id] = question
        if not question or (tenant_id and question.get("tenant_id", "tenant-local") != tenant_id):
            raise ValueError("question_id not found")
        return question

    def record_question_feedback(
        self,
        question_id: str,
        *,
        rating: str,
        comment: str,
        actor: str = "system",
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        question = self.get_question(question_id, tenant_id)
        feedback = {
            "feedback_id": f"vision-feedback-{uuid4().hex[:10]}",
            "question_id": question_id,
            "rating": rating,
            "comment": comment,
            "actor": actor,
            "created_at": utc_now_iso(),
        }
        self.question_feedback.setdefault(question_id, []).append(feedback)
        question["feedback"] = list(self.question_feedback[question_id])
        self._persist_question(question_id, question)
        return feedback

    def _not_found_answer(self, *, requires_review: bool) -> VisionAskResponse:
        summary = "No extracted field, chunk, or graph evidence matched the question."
        # A non-semantic embedding backend cannot match a paraphrase, so an
        # abstention may mean "not reached" rather than "not present". The
        # reader is entitled to know which. This is a standing property of the
        # configured backend rather than a fallback taken during this answer,
        # so it belongs in the reasoning and the retrieval trace — putting it
        # in `degradations` would make every abstention read as degraded and
        # dilute the signal that a generator or vector backend actually failed.
        if not self._semantic_retrieval_available():
            summary = (
                f"{summary} Retrieval is served by the lexical/feature-hashing "
                "backend, which matches wording rather than meaning, so a "
                "question phrased differently from the document may not reach "
                "evidence that exists."
            )
        return VisionAskResponse(
            answer=UNSUPPORTED_ANSWER,
            answer_status="not_found",
            citations=[],
            confidence=0.0,
            reasoning_summary=summary,
            requires_review=requires_review,
        )

    def _semantic_retrieval_available(self) -> bool:
        try:
            return bool(self.retrieval_index.status().get("semantic_embeddings"))
        except Exception:  # pragma: no cover - capability probe is never required
            return False
