"""Human review queue, decisions, and active-learning capture.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import settings
from app.core.security import hash_document, utc_now_iso
from app.models.schemas import UserRole
from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VerifiedVisionFact,
    VisionDocumentRecord,
    VisionEvidence,
    VisionProcessingStatus,
    VisionReviewDecisionRequest,
    VisionReviewTask,
    VisionRiskSeverity,
    VisionSemanticType,
)
from app.services.vision.analytics import vision_analytics_service
from app.services.vision.approval_policy import (
    ApprovalPolicyEvaluation,
    ApprovalPolicyViolation,
    ReviewDecisionValidationError,
    vision_approval_policy_service,
)
from app.services.vision.legacy_audit import CURRENT_PIPELINE_VERSION
from app.services.vision.schema_registry import vision_schema_registry
from app.services.vision.semantic_facts import (
    normalize_semantic_value,
    semantic_type_for_name,
)


class ReviewMixin:
    def list_reviews(self, tenant_id: str | None = None) -> list[VisionReviewTask]:
        self._load_all_reviews()
        if tenant_id:
            for review in self.persistence.list_reviews(tenant_id):
                self.review_tasks[review.review_id] = review
        reviews = self.review_tasks.values()
        if tenant_id:
            reviews = [item for item in reviews if self._review_belongs_to_tenant(item, tenant_id)]
        return sorted(reviews, key=lambda item: item.created_at, reverse=True)

    def review_queue(
        self,
        tenant_id: str,
        *,
        search: str = "",
        document_type: str = "all",
        priority: str = "all",
        reason: str = "all",
        assignee: str = "all",
        status: str = "open",
        sort_by: str = "created_at",
        sort_direction: str = "desc",
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:
        all_reviews = self.list_reviews(tenant_id)
        needle = self._normalize(search)
        rows = []
        for review in all_reviews:
            document = self.get_document(review.document_id, tenant_id)
            searchable = self._normalize(
                " ".join(
                    [
                        review.review_id,
                        review.document_id,
                        review.filename or document.filename,
                        document.document_type.value,
                        *review.reasons,
                        *review.reason_labels,
                    ]
                )
            )
            if needle and needle not in searchable:
                continue
            if document_type != "all" and document.document_type.value != document_type:
                continue
            if priority != "all" and review.priority.value != priority:
                continue
            if reason != "all" and reason not in review.reasons:
                continue
            if assignee == "unassigned" and review.assigned_reviewer:
                continue
            if assignee not in {"all", "unassigned"} and review.assigned_reviewer != assignee:
                continue
            if status == "open" and review.status not in {
                VisionProcessingStatus.review_required,
                VisionProcessingStatus.under_review,
            }:
                continue
            if status not in {"all", "open"} and review.status.value != status:
                continue
            rows.append(review)

        priority_order = {
            VisionRiskSeverity.low: 1,
            VisionRiskSeverity.medium: 2,
            VisionRiskSeverity.high: 3,
            VisionRiskSeverity.critical: 4,
        }
        sorters = {
            "created_at": lambda item: item.created_at,
            "priority": lambda item: priority_order[item.priority],
            "filename": lambda item: (item.filename or "").lower(),
            "status": lambda item: item.status.value,
            "confidence": lambda item: item.confidence_value or -1.0,
        }
        rows.sort(
            key=sorters.get(sort_by, sorters["created_at"]),
            reverse=sort_direction != "asc",
        )
        total = len(rows)
        safe_page_size = min(max(page_size, 1), 100)
        safe_page = max(page, 1)
        start = (safe_page - 1) * safe_page_size
        paged = rows[start : start + safe_page_size]
        return {
            "total": total,
            "unfiltered_total": len(all_reviews),
            "page": safe_page,
            "page_size": safe_page_size,
            "total_pages": (total + safe_page_size - 1) // safe_page_size,
            "reviews": [item.model_dump(mode="json") for item in paged],
            "facets": {
                "document_types": sorted(
                    {
                        self.get_document(item.document_id, tenant_id).document_type.value
                        for item in all_reviews
                    }
                ),
                "priorities": sorted({item.priority.value for item in all_reviews}),
                "reasons": [
                    {"value": machine, "label": label}
                    for machine, label in sorted(
                        {
                            (machine, label)
                            for item in all_reviews
                            for machine, label in zip(
                                item.reasons,
                                item.reason_labels,
                                strict=False,
                            )
                        }
                    )
                ],
                "assignees": sorted(
                    {item.assigned_reviewer for item in all_reviews if item.assigned_reviewer}
                ),
                "statuses": sorted({item.status.value for item in all_reviews}),
            },
        }

    def get_review(self, review_id: str, tenant_id: str | None = None) -> VisionReviewTask:
        review = self.review_tasks.get(review_id) or self._load_review(review_id)
        if not review and tenant_id:
            review = self.persistence.get_review(review_id, tenant_id)
            if review:
                self.review_tasks[review.review_id] = review
        if not review or (tenant_id and not self._review_belongs_to_tenant(review, tenant_id)):
            raise ValueError("review_id not found")
        return review

    def patch_field(
        self,
        review_id: str,
        field_id: str,
        *,
        reviewed_value: str | None,
        reviewed_field_name: str | None = None,
        reviewer_status: ReviewerStatus,
        comment: str,
        actor: str,
        tenant_id: str | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> ExtractedVisionField:
        review = self.get_review(review_id, tenant_id)
        document = self.get_document(review.document_id, tenant_id)
        field = next(
            (item for item in document.extracted_fields if item.field_id == field_id), None
        )
        if not field:
            raise ValueError("field_id not found")
        old_name = field.name
        if reviewed_field_name and reviewed_field_name != field.name:
            if any(
                item.field_id != field.field_id and item.name == reviewed_field_name
                for item in document.extracted_fields
            ):
                raise ValueError("reviewed_field_name already exists on this document")
            field.name = reviewed_field_name
            field.schema_field = reviewed_field_name
            field.display_name = reviewed_field_name.replace("_", " ").title()
        old_value = field.reviewed_value or field.value
        field.reviewed_value = reviewed_value if reviewed_value is not None else field.value
        semantic_type = self._semantic_type_for_reviewed_field(document, field)
        field.normalized_value = normalize_semantic_value(
            field.reviewed_value,
            semantic_type,
            label=field.name,
            document_type=document.document_type.value,
        )
        field.reviewer_status = reviewer_status
        field.evidence.reviewer_status = reviewer_status
        field.verification_status = reviewer_status.value
        self._apply_field_review_to_generic_layers(
            document,
            field,
            semantic_type=semantic_type,
        )
        event = self._audit(
            document,
            "field_reviewed",
            "Reviewer updated an extracted field.",
            actor=actor,
            metadata={
                "field_id": field_id,
                "field_name": field.name,
                "old_field_name": old_name,
                "old_value": self._mask_sensitive(field.name, old_value),
                "new_value": self._mask_sensitive(field.name, field.reviewed_value or ""),
                "comment": comment,
                **(request_metadata or {}),
            },
        )
        review.history.append(event)
        review.updated_at = event.created_at
        if reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}:
            self._record_active_learning_correction(
                document,
                field,
                original_field_name=old_name,
                original_value=old_value,
                comment=comment,
                reviewer=actor,
                reviewed_at=event.created_at,
            )
        document.data_version += 1
        document.analytics_cache_version += 1
        self._refresh_verified_facts(document, reviewer=actor)
        self._resolve_semantic_entities(document)
        document.evidence_chunks = self._chunks(document)
        document.graph = self._graph(document)
        vision_analytics_service.refresh_document(document)
        self._index_document(document)
        self._persist_document(document)
        self._persist_review(review)
        return field

    def record_review_decision(
        self,
        review_id: str,
        *,
        decision: VisionProcessingStatus,
        request: VisionReviewDecisionRequest | None,
        actor: str,
        actor_role: UserRole,
        tenant_id: str | None = None,
        request_metadata: dict[str, Any] | None = None,
        reauthentication_payload: dict[str, Any] | None = None,
    ) -> VisionReviewTask:
        allowed = {
            VisionProcessingStatus.approved,
            VisionProcessingStatus.rejected,
            VisionProcessingStatus.confirmed_error,
            VisionProcessingStatus.confirmed_fraud,
            VisionProcessingStatus.under_review,
            VisionProcessingStatus.review_required,
        }
        if decision not in allowed:
            raise ValueError("Unsupported review decision")
        decision_request = request or VisionReviewDecisionRequest(decision=decision)
        if decision_request.decision != decision:
            raise ValueError("Request decision does not match the action endpoint")
        review = self.get_review(review_id, tenant_id)
        document = self.get_document(review.document_id, tenant_id)
        policy_evaluation: ApprovalPolicyEvaluation | None = None
        if decision == VisionProcessingStatus.approved:
            policy_evaluation = vision_approval_policy_service.evaluate(
                document,
                request=decision_request,
                actor_role=actor_role,
                confidence_threshold=self._mandatory_review_threshold(document),
            )
            if not policy_evaluation.allowed:
                raise ApprovalPolicyViolation(policy_evaluation)
            if policy_evaluation.mode == "manual_override":
                self._apply_manual_confirmations(
                    document,
                    decision_request.confirmed_fields,
                    actor=actor,
                )
        if decision == VisionProcessingStatus.confirmed_fraud:
            self._validate_fraud_confirmation(
                document,
                decision_request,
                actor=actor,
                actor_role=actor_role,
                reauthentication_payload=reauthentication_payload,
            )
        comment = decision_request.comment.strip()
        review.status = decision
        review.reviewer_comments = comment
        review.updated_at = utc_now_iso()
        document.status = decision
        document.updated_at = review.updated_at
        if decision == VisionProcessingStatus.approved:
            for field in document.extracted_fields:
                if field.reviewer_status == ReviewerStatus.unverified:
                    field.reviewer_status = ReviewerStatus.accepted
                    field.evidence.reviewer_status = ReviewerStatus.accepted
                    field.verification_status = ReviewerStatus.accepted.value
            for detection in document.visual_detections:
                if detection.reviewer_status == ReviewerStatus.unverified:
                    detection.reviewer_status = ReviewerStatus.accepted
            self._refresh_verified_facts(document, reviewer=actor)
        document.evidence_chunks = self._chunks(document)
        document.graph = self._graph(document)
        document.data_version += 1
        document.analytics_cache_version += 1
        vision_analytics_service.refresh_document(document)
        self._index_document(document)
        event_type = "review_decision_recorded"
        if decision == VisionProcessingStatus.confirmed_fraud:
            event_type = "confirmed_fraud_by_reviewer"
        decision_metadata = {
            "comment": comment,
            "approval_reason": decision_request.approval_reason.strip(),
            "approval_mode": policy_evaluation.mode if policy_evaluation else None,
            "confirmed_field_names": sorted(decision_request.confirmed_fields),
            "evidence_ids": sorted(set(decision_request.evidence_ids)),
            "decision_rationale": decision_request.decision_rationale.strip(),
            "automated_indicators_acknowledged": (
                decision_request.automated_indicators_acknowledged
            ),
            "reauthenticated_at": (
                reauthentication_payload.get("auth_time") if reauthentication_payload else None
            ),
            "overridden_policy_conditions": (
                [item.code for item in policy_evaluation.observed_conditions]
                if policy_evaluation and policy_evaluation.mode == "manual_override"
                else []
            ),
            **(request_metadata or {}),
        }
        event = self._audit(
            document,
            event_type,
            f"Reviewer set document status to {decision.value}.",
            actor=actor,
            metadata=decision_metadata,
        )
        review.history.append(event)
        if decision == VisionProcessingStatus.approved:
            self.persistence.upsert_graph(
                document.tenant_id,
                document.document_id,
                document.graph,
            )
        self._persist_document(document)
        self._persist_review(review)
        return review

    def approval_policy_for_review(
        self,
        review_id: str,
        *,
        actor_role: UserRole,
        tenant_id: str,
    ) -> ApprovalPolicyEvaluation:
        review = self.get_review(review_id, tenant_id)
        document = self.get_document(review.document_id, tenant_id)
        evaluation = vision_approval_policy_service.evaluate(
            document,
            request=None,
            actor_role=actor_role,
            confidence_threshold=self._mandatory_review_threshold(document),
        )
        evaluation.fraud_second_reviewer_required = settings.fraud_second_reviewer_required
        return evaluation

    def _apply_manual_confirmations(
        self,
        document: VisionDocumentRecord,
        confirmed_fields: dict[str, str],
        *,
        actor: str,
    ) -> None:
        schema = vision_schema_registry.get(document.document_type)
        schema_names = {item.name for item in schema.fields} if schema else set()
        by_name = {field.name: field for field in document.extracted_fields}
        confirmed_names: list[str] = []
        for raw_name, raw_value in confirmed_fields.items():
            name = raw_name.strip().lower()
            value = raw_value.strip()
            if not value or (schema_names and name not in schema_names):
                continue
            field = by_name.get(name)
            if field:
                field.reviewed_value = value
                field.reviewer_status = ReviewerStatus.accepted
                field.evidence.reviewer_status = ReviewerStatus.accepted
                field.verification_status = ReviewerStatus.accepted.value
            else:
                spec = schema.field(name) if schema else None
                identity = hash_document(f"{document.document_id}|manual-review|{name}".encode())
                evidence = VisionEvidence(
                    evidence_id=f"vis-ev-manual-{identity[:10]}",
                    document_id=document.document_id,
                    page_number=1,
                    source_text=f"Reviewer-entered field: {name.replace('_', ' ')}",
                    model_confidence=0.0,
                    model_name="human-manual-verification",
                    model_version="v1",
                    processing_timestamp=utc_now_iso(),
                    reviewer_status=ReviewerStatus.accepted,
                    metadata={
                        "confidence_available": False,
                        "source": "manual_approval_confirmation",
                        "actor": actor,
                    },
                )
                field = ExtractedVisionField(
                    field_id=f"field-manual-{identity[:10]}",
                    name=name,
                    schema_field=name,
                    display_name=(spec.display_name if spec else name.replace("_", " ").title()),
                    value=value,
                    normalized_value=self._normalize(value),
                    confidence=None,
                    evidence=evidence,
                    model_name="human-manual-verification",
                    model_version="v1",
                    extractor="manual-review",
                    verification_status=ReviewerStatus.accepted.value,
                    required=bool(spec and spec.required),
                    reviewer_status=ReviewerStatus.accepted,
                    reviewed_value=value,
                )
                document.extracted_fields.append(field)
                by_name[name] = field
            confirmed_names.append(name)
        vision_analytics_service.refresh_document(document)
        self._audit(
            document,
            "manual_approval_fields_confirmed",
            "Reviewer entered or confirmed required fields for manual approval.",
            actor=actor,
            metadata={"field_names": sorted(set(confirmed_names))},
        )

    def _validate_fraud_confirmation(
        self,
        document: VisionDocumentRecord,
        request: VisionReviewDecisionRequest,
        *,
        actor: str,
        actor_role: UserRole,
        reauthentication_payload: dict[str, Any] | None,
    ) -> None:
        if actor_role not in {
            UserRole.admin,
            UserRole.senior_reviewer,
            UserRole.compliance_officer,
        }:
            raise PermissionError(
                "Compliance Officer, Senior Reviewer, or Admin role is required to confirm fraud"
            )
        if (
            not reauthentication_payload
            or reauthentication_payload.get("email") != actor
            or reauthentication_payload.get("purpose") != "sensitive_review_decision"
        ):
            raise PermissionError("Recent reauthentication is required to confirm fraud")
        if len(request.comment.strip()) < 10:
            raise ReviewDecisionValidationError(
                "A reviewer comment of at least 10 characters is required"
            )
        if len(request.decision_rationale.strip()) < 20:
            raise ReviewDecisionValidationError(
                "A written fraud decision rationale of at least 20 characters is required"
            )
        if not request.automated_indicators_acknowledged:
            raise ReviewDecisionValidationError(
                "Acknowledge that automated indicators alone do not prove fraud"
            )
        if not request.confirmation_acknowledged:
            raise ReviewDecisionValidationError(
                "The fraud confirmation modal acknowledgement is required"
            )
        available_evidence = self._document_evidence_ids(document)
        selected_evidence = set(request.evidence_ids)
        if not selected_evidence:
            raise ReviewDecisionValidationError(
                "At least one supporting evidence item must be selected"
            )
        unavailable = sorted(selected_evidence - available_evidence)
        if unavailable:
            raise ReviewDecisionValidationError(f"Unknown evidence IDs: {', '.join(unavailable)}")
        if settings.fraud_second_reviewer_required:
            second_reviewer = (request.second_reviewer_email or "").strip().lower()
            if (
                not request.second_reviewer_approved
                or not second_reviewer
                or second_reviewer == actor.lower()
            ):
                raise ReviewDecisionValidationError(
                    "A distinct second reviewer approval is required"
                )

    def _document_evidence_ids(self, document: VisionDocumentRecord) -> set[str]:
        return {
            evidence.evidence_id
            for evidence in [
                *(field.evidence for field in document.extracted_fields),
                *(
                    evidence
                    for risk in [
                        *document.risk_indicators,
                        *document.processing_warnings,
                    ]
                    for evidence in risk.evidence
                ),
                *(
                    evidence
                    for duplicate in document.duplicate_candidates
                    for evidence in duplicate.evidence
                ),
            ]
        }

    def accept_high_confidence_fields(
        self,
        review_id: str,
        *,
        actor: str,
        tenant_id: str,
        request_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        review = self.get_review(review_id, tenant_id)
        document = self.get_document(review.document_id, tenant_id)
        threshold = self._review_threshold(
            "auto_approval_min_confidence",
            document,
        )
        accepted = []
        for field in document.extracted_fields:
            if (
                field.reviewer_status != ReviewerStatus.unverified
                or field.confidence is None
                or field.confidence < threshold
            ):
                continue
            field.reviewer_status = ReviewerStatus.accepted
            field.evidence.reviewer_status = ReviewerStatus.accepted
            field.verification_status = ReviewerStatus.accepted.value
            accepted.append(field.field_id)
        event = self._audit(
            document,
            "high_confidence_fields_accepted",
            f"Reviewer accepted {len(accepted)} field(s) at or above the configured threshold.",
            actor=actor,
            metadata={
                "field_ids": accepted,
                "threshold": threshold,
                **(request_metadata or {}),
            },
        )
        review.history.append(event)
        review.updated_at = document.updated_at
        document.data_version += 1
        document.analytics_cache_version += 1
        self._refresh_verified_facts(document, reviewer=actor)
        document.evidence_chunks = self._chunks(document)
        document.graph = self._graph(document)
        vision_analytics_service.refresh_document(document)
        self._index_document(document)
        self.persistence.upsert_graph(document.tenant_id, document.document_id, document.graph)
        self._persist_document(document)
        self._persist_review(review)
        return {"accepted_field_ids": accepted, "threshold": threshold}

    def active_learning_export(self, tenant_id: str | None = None) -> dict[str, Any]:
        resolved_tenant = str(tenant_id or "").strip()
        if not resolved_tenant:
            raise ValueError("tenant_id is required for active-learning export")
        corrections: list[dict[str, Any]] = []
        persisted_keys: set[tuple[str, str]] = set()
        for record in self.feedback_dataset.records(resolved_tenant):
            field_name = str(
                record.get("corrected_field_name") or record.get("original_field_name") or "unknown"
            )
            semantic_type = str(record.get("semantic_type") or "")
            sensitive = semantic_type == VisionSemanticType.bank_account.value or any(
                token in field_name.casefold()
                for token in ("bank", "account", "ifsc", "routing", "iban")
            )
            document_id = str(record.get("document_id") or "")
            field_id = str(record.get("field_id") or "")
            persisted_keys.add((document_id, field_id))
            corrections.append(
                {
                    "correction_id": record.get("correction_id"),
                    "schema_version": record.get("schema_version"),
                    "tenant_id": resolved_tenant,
                    "document_id": document_id,
                    "document_type": record.get("document_type"),
                    "field_id": field_id,
                    "field_name": field_name,
                    "original_field_name": record.get("original_field_name"),
                    "raw_label": record.get("raw_label"),
                    "semantic_type": record.get("semantic_type"),
                    "model_value": self._mask_sensitive(
                        field_name,
                        str(record.get("original_value") or ""),
                    ),
                    "reviewed_value": self._mask_sensitive(
                        field_name,
                        str(record.get("corrected_value") or ""),
                    ),
                    "evidence": {
                        "evidence_id": record.get("evidence_id"),
                        "page": record.get("page"),
                        "bounding_box": record.get("bounding_box"),
                        "source_text": (
                            "Sensitive source text withheld from export."
                            if sensitive
                            else record.get("source_text")
                        ),
                    },
                    "extraction_method": record.get("extraction_method"),
                    "layout_relation": record.get("layout_relation"),
                    "schema_mode": record.get("schema_mode"),
                    "schema_status": record.get("schema_status"),
                    "processing_version": record.get("processing_version"),
                    "status": record.get("status"),
                    "reviewed_at": record.get("reviewed_at"),
                    "training_policy": record.get("training_policy"),
                }
            )

        # Keep edited fields created before the durable v2 dataset exportable.
        legacy_count = 0
        for document in self.list_documents(resolved_tenant):
            for field in document.extracted_fields:
                if (
                    field.reviewer_status == ReviewerStatus.edited
                    and field.reviewed_value is not None
                    and (document.document_id, field.field_id) not in persisted_keys
                ):
                    legacy_count += 1
                    corrections.append(
                        {
                            "document_id": document.document_id,
                            "field_id": field.field_id,
                            "field_name": field.name,
                            "tenant_id": document.tenant_id,
                            "document_type": document.document_type.value,
                            "model_value": self._mask_sensitive(field.name, field.value),
                            "reviewed_value": self._mask_sensitive(
                                field.name, field.reviewed_value
                            ),
                            "evidence": field.evidence.model_dump(mode="json"),
                        }
                    )
        return {
            "total_corrections": len(corrections),
            "persisted_corrections": len(corrections) - legacy_count,
            "legacy_live_corrections": legacy_count,
            "format": "active_learning_jsonl_v2_export",
            "summary": self.feedback_dataset.summary(resolved_tenant),
            "corrections": corrections,
        }

    def _semantic_type_for_reviewed_field(
        self,
        document: VisionDocumentRecord,
        field: ExtractedVisionField,
    ) -> VisionSemanticType | None:
        generic = next(
            (
                item
                for item in document.generic_fields
                if item.evidence.evidence_id == field.evidence.evidence_id
            ),
            None,
        )
        return semantic_type_for_name(field.name) or (
            generic.semantic_type_if_known if generic else None
        )

    def _apply_field_review_to_generic_layers(
        self,
        document: VisionDocumentRecord,
        field: ExtractedVisionField,
        *,
        semantic_type: VisionSemanticType | None,
    ) -> None:
        approved = field.reviewer_status in {
            ReviewerStatus.accepted,
            ReviewerStatus.edited,
        }
        for generic in document.generic_fields:
            if generic.evidence.evidence_id != field.evidence.evidence_id:
                continue
            generic.evidence.reviewer_status = field.reviewer_status
            generic.review_required = not approved
            generic.metadata["reviewed_field_name"] = field.name
            generic.metadata["reviewed_value"] = field.reviewed_value
            if approved:
                generic.normalized_label = field.name
                generic.normalized_value = field.normalized_value or generic.normalized_value
                generic.semantic_type_if_known = semantic_type or generic.semantic_type_if_known
        for index, entity in enumerate(document.typed_entities):
            if field.evidence.evidence_id not in entity.evidence_ids:
                continue
            context = dict(entity.context)
            context["reviewed_field_name"] = field.name
            context["reviewer_status"] = field.reviewer_status.value
            document.typed_entities[index] = entity.model_copy(
                update={
                    "normalized_value": (
                        field.normalized_value or entity.normalized_value
                        if approved
                        else entity.normalized_value
                    ),
                    "semantic_type": semantic_type or entity.semantic_type,
                    "review_required": not approved,
                    "context": context,
                }
            )

    def _refresh_verified_facts(
        self,
        document: VisionDocumentRecord,
        *,
        reviewer: str | None = None,
    ) -> None:
        semantic_by_label = {
            field.normalized_label: field.semantic_type_if_known
            for field in document.generic_fields
            if field.semantic_type_if_known is not None
        }
        document.verified_facts = [
            VerifiedVisionFact(
                fact_id=f"fact-{field.field_id}",
                field_id=field.field_id,
                name=field.name,
                value=field.reviewed_value or field.value,
                normalized_value=field.normalized_value,
                semantic_type=semantic_by_label.get(field.name),
                evidence_id=field.evidence.evidence_id,
                reviewer_status=field.reviewer_status,
                reviewer=reviewer,
                verified_at=utc_now_iso(),
            )
            for field in document.extracted_fields
            if field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
        ]

    def _record_active_learning_correction(
        self,
        document: VisionDocumentRecord,
        field: ExtractedVisionField,
        *,
        original_field_name: str,
        original_value: str,
        comment: str,
        reviewer: str,
        reviewed_at: str,
    ) -> None:
        generic = next(
            (
                item
                for item in document.generic_fields
                if item.evidence.evidence_id == field.evidence.evidence_id
                or item.normalized_label == original_field_name
            ),
            None,
        )
        raw_label = generic.raw_label if generic else self._source_label(field)
        semantic_type = self._semantic_type_for_reviewed_field(document, field)
        record = {
            "correction_id": f"correction-{uuid4().hex[:12]}",
            "tenant_id": document.tenant_id,
            "document_id": document.document_id,
            "document_type": document.document_type.value,
            "field_id": field.field_id,
            "raw_label": raw_label,
            "original_field_name": original_field_name,
            "corrected_field_name": field.name,
            "original_value": original_value,
            "corrected_value": field.reviewed_value or field.value,
            "normalized_corrected_value": field.normalized_value,
            "semantic_type": semantic_type.value if semantic_type else None,
            "source_text": field.evidence.source_text,
            "evidence_id": field.evidence.evidence_id,
            "page": field.evidence.page_number,
            "bounding_box": (
                field.evidence.bounding_box.model_dump(mode="json")
                if field.evidence.bounding_box
                else None
            ),
            "extraction_method": field.extractor,
            "layout_relation": (generic.metadata.get("extraction_strategy") if generic else None),
            "source_line_ids": list(field.evidence.metadata.get("source_line_ids") or []),
            "schema_mode": document.schema_profile.mode,
            "schema_status": document.schema_profile.schema_status,
            "schema_id": document.schema_profile.schema_id,
            "processing_version": document.metadata.get(
                "pipeline_version",
                CURRENT_PIPELINE_VERSION,
            ),
            "reviewer": reviewer,
            "reviewer_comment": comment,
            "reviewed_at": reviewed_at,
            "status": field.reviewer_status.value,
            "dataset_usage": "controlled_alias_and_training_export",
        }
        self.feedback_dataset.append(record)

    def _feedback_aliases(self, tenant_id: str) -> dict[str, str]:
        return self.feedback_dataset.approved_label_aliases(
            tenant_id,
            normalizer=self._normalize_token,
        )

    def _active_learning_path(self, tenant_id: str) -> Path:
        return self.feedback_dataset.path_for(tenant_id)

    def _create_review_task(
        self,
        document: VisionDocumentRecord,
        *,
        review_id: str | None = None,
    ) -> VisionReviewTask:
        reasons = []
        if document.model_routing.get("fallback_required"):
            reasons.extend(document.model_routing.get("reasons", []))
        reasons.extend(
            risk.title
            for risk in document.risk_indicators
            if risk.severity in {VisionRiskSeverity.high, VisionRiskSeverity.critical}
        )
        reasons.extend(
            warning.title
            for warning in document.processing_warnings
            if warning.severity in {VisionRiskSeverity.high, VisionRiskSeverity.critical}
        )
        reasons.extend(
            f"missing_required_field:{name}"
            for name in document.field_completeness.missing_required_fields
        )
        if not reasons:
            reasons.append("field_confidence_requires_review")
        severity = self._review_priority(document)
        now = utc_now_iso()
        task_data: dict[str, Any] = {
            "document_id": document.document_id,
            "priority": severity,
            "reasons": sorted(set(reasons)),
            "filename": document.filename,
            "document_type": document.document_type,
            "page_count": document.page_count,
            "confidence_status": (
                "available" if document.confidence_summary.confidence_available else "not_available"
            ),
            "confidence_value": document.confidence_summary.mean_field_confidence,
            "expected_required_fields": document.field_completeness.expected_required_count,
            "unresolved_required_fields": document.field_completeness.unresolved_required_count,
            "created_at": now,
            "updated_at": now,
        }
        if review_id:
            task_data["review_id"] = review_id
        task = VisionReviewTask(**task_data)
        event = self._audit(
            document,
            "review_task_created",
            "Human review task created from evidence-backed routing rules.",
            metadata={
                "review_id": task.review_id,
                "priority": severity.value,
                "reasons": task.reasons,
            },
        )
        task.history.append(event)
        return task

    def _ensure_review_task(self, document: VisionDocumentRecord) -> None:
        if document.review_task or document.status not in {
            VisionProcessingStatus.review_required,
            VisionProcessingStatus.under_review,
        }:
            return
        document.review_task = self._create_review_task(document)
        self.review_tasks[document.review_task.review_id] = document.review_task
        self._persist_document(document)
        self._persist_review(document.review_task)

    def _requires_review(self, document: VisionDocumentRecord) -> bool:
        if not bool(self.confidence_policy.get("calibration", {}).get("calibrated")):
            return True
        if document.model_routing.get("fallback_required"):
            return True
        if not document.ocr_words:
            return True
        if (
            document.field_completeness.schema_available
            and document.field_completeness.unresolved_required_count
        ):
            return True
        if any(
            risk.severity in {VisionRiskSeverity.high, VisionRiskSeverity.critical}
            for risk in document.risk_indicators
        ):
            return True
        if any(
            field.confidence is None
            or field.confidence < self._mandatory_review_threshold(document)
            for field in document.extracted_fields
        ):
            return True
        return False
