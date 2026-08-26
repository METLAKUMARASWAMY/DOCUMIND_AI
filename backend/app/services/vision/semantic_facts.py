"""Schema-independent semantic facts used by risk and entity workflows.

The extraction pipeline intentionally keeps several evidence layers.  This
module provides a read-only, deterministic view over those layers so downstream
rules do not need to depend on a particular document schema or template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from app.models.vision import (
    ReviewerStatus,
    VisionDocumentRecord,
    VisionEvidence,
    VisionSemanticType,
)
from app.services.vision.semantic_values import (
    SemanticContext,
    SemanticValueType,
    semantic_value_registry,
)

VERIFIED_STATUSES = {ReviewerStatus.accepted, ReviewerStatus.edited}


@dataclass(frozen=True)
class SemanticFact:
    """One evidence-backed fact independent of its extraction layer."""

    name: str
    value: str
    normalized_value: str
    semantic_type: VisionSemanticType | None
    confidence: float
    evidence: VisionEvidence | None
    source_layers: tuple[str, ...]
    source_ids: tuple[str, ...]
    verified: bool = False

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return (self.evidence.evidence_id,) if self.evidence else ()


def normalize_label(value: str) -> str:
    normalized = value.casefold().replace("#", " number ").replace("&", " and ")
    normalized = re.sub(r"\b(?:no|num|nbr)\.?\b", " number ", normalized)
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def normalize_semantic_value(
    value: str,
    semantic_type: VisionSemanticType | None,
    *,
    label: str = "",
    document_type: str = "",
) -> str:
    """Normalize with the shared type system, falling back without inventing data."""

    cleaned = " ".join(str(value).split()).strip()
    if not cleaned or semantic_type is None:
        return cleaned
    try:
        kind = SemanticValueType(semantic_type.value.casefold())
        result = semantic_value_registry.validate(
            kind,
            cleaned,
            SemanticContext(label=label, document_type=document_type),
        )
    except (KeyError, ValueError):
        return cleaned
    return result.normalized_value if result.valid and result.normalized_value else cleaned


def semantic_type_for_name(name: str) -> VisionSemanticType | None:
    """Conservative type hints for established cross-schema concepts."""

    normalized = normalize_label(name)
    exact: dict[str, VisionSemanticType] = {
        "date": VisionSemanticType.date,
        "document_date": VisionSemanticType.date,
        "invoice_date": VisionSemanticType.date,
        "due_date": VisionSemanticType.date,
        "request_date": VisionSemanticType.date,
        "submission_date": VisionSemanticType.date,
        "effective_date": VisionSemanticType.date,
        "expiry_date": VisionSemanticType.date,
        "email": VisionSemanticType.email,
        "email_address": VisionSemanticType.email,
        "phone": VisionSemanticType.phone,
        "phone_number": VisionSemanticType.phone,
        "fax_number": VisionSemanticType.phone,
        "bank_account": VisionSemanticType.bank_account,
        "bank_account_number": VisionSemanticType.bank_account,
        "account_number": VisionSemanticType.bank_account,
        "requested_payment_account": VisionSemanticType.bank_account,
        "approved_bank_account": VisionSemanticType.bank_account,
        "organisation": VisionSemanticType.organisation,
        "organisation_name": VisionSemanticType.organisation,
        "organization": VisionSemanticType.organisation,
        "organization_name": VisionSemanticType.organisation,
        "company_name": VisionSemanticType.organisation,
        "vendor_name": VisionSemanticType.organisation,
        "supplier_name": VisionSemanticType.organisation,
        "person": VisionSemanticType.person,
        "person_name": VisionSemanticType.person,
        "contact_name": VisionSemanticType.person,
        "requested_by": VisionSemanticType.person,
        "invoice_number": VisionSemanticType.document_reference,
        "purchase_order_number": VisionSemanticType.document_reference,
        "reference_number": VisionSemanticType.document_reference,
        "document_reference": VisionSemanticType.document_reference,
        "total_amount": VisionSemanticType.money,
        "subtotal": VisionSemanticType.money,
        "tax": VisionSemanticType.money,
        "discount": VisionSemanticType.money,
        "amount": VisionSemanticType.money,
        "currency": VisionSemanticType.currency,
        "address": VisionSemanticType.address,
        "url": VisionSemanticType.url,
    }
    return exact.get(normalized)


def collect_semantic_facts(document: VisionDocumentRecord) -> list[SemanticFact]:
    """Collect and deduplicate all durable semantic layers for one document."""

    evidence_by_id: dict[str, VisionEvidence] = {}
    type_by_evidence: dict[str, VisionSemanticType] = {}
    name_by_evidence: dict[str, str] = {}
    for field in document.extracted_fields:
        evidence_by_id[field.evidence.evidence_id] = field.evidence
    for field in document.generic_fields:
        evidence_by_id[field.evidence.evidence_id] = field.evidence
        name_by_evidence[field.evidence.evidence_id] = field.normalized_label
        if field.semantic_type_if_known is not None:
            type_by_evidence[field.evidence.evidence_id] = field.semantic_type_if_known

    facts: list[SemanticFact] = []
    verified_field_ids = {fact.field_id for fact in document.verified_facts}
    for fact in document.verified_facts:
        evidence = evidence_by_id.get(fact.evidence_id)
        semantic_type = fact.semantic_type or type_by_evidence.get(fact.evidence_id)
        value = fact.value
        facts.append(
            SemanticFact(
                name=normalize_label(fact.name),
                value=value,
                normalized_value=(
                    fact.normalized_value
                    or normalize_semantic_value(
                        value,
                        semantic_type,
                        label=fact.name,
                        document_type=document.document_type.value,
                    )
                ),
                semantic_type=semantic_type,
                confidence=evidence.model_confidence if evidence else 0.0,
                evidence=evidence,
                source_layers=("verified_fact",),
                source_ids=(fact.fact_id,),
                verified=True,
            )
        )

    for field in document.extracted_fields:
        value = field.reviewed_value or field.value
        semantic_type = type_by_evidence.get(field.evidence.evidence_id) or semantic_type_for_name(
            field.name
        )
        reviewed = field.reviewer_status in VERIFIED_STATUSES
        facts.append(
            SemanticFact(
                name=normalize_label(field.name),
                value=value,
                normalized_value=(
                    normalize_semantic_value(
                        value,
                        semantic_type,
                        label=field.name,
                        document_type=document.document_type.value,
                    )
                    if field.reviewed_value is not None
                    else field.normalized_value
                    or normalize_semantic_value(
                        value,
                        semantic_type,
                        label=field.name,
                        document_type=document.document_type.value,
                    )
                ),
                semantic_type=semantic_type,
                confidence=field.confidence or field.evidence.model_confidence or 0.0,
                evidence=field.evidence,
                source_layers=("reviewed_field" if reviewed else "extracted_field",),
                source_ids=(field.field_id,),
                verified=reviewed or field.field_id in verified_field_ids,
            )
        )

    for field in document.generic_fields:
        semantic_confidence = field.semantic_mapping_confidence
        confidence = (
            min(field.confidence, semantic_confidence) if semantic_confidence else field.confidence
        )
        facts.append(
            SemanticFact(
                name=normalize_label(field.normalized_label),
                value=field.raw_value,
                normalized_value=(
                    field.normalized_value
                    or normalize_semantic_value(
                        field.raw_value,
                        field.semantic_type_if_known,
                        label=field.raw_label,
                        document_type=document.document_type.value,
                    )
                ),
                semantic_type=field.semantic_type_if_known,
                confidence=confidence,
                evidence=field.evidence,
                source_layers=("generic_field",),
                source_ids=(field.field_id,),
                verified=field.evidence.reviewer_status in VERIFIED_STATUSES,
            )
        )

    for entity in document.typed_entities:
        evidence = next(
            (evidence_by_id[item] for item in entity.evidence_ids if item in evidence_by_id),
            None,
        )
        contextual_label = str(entity.context.get("contextual_label") or "")
        name = name_by_evidence.get(evidence.evidence_id if evidence else "") or normalize_label(
            contextual_label
        )
        facts.append(
            SemanticFact(
                name=name or entity.semantic_type.value.casefold(),
                value=entity.raw_value,
                normalized_value=entity.normalized_value,
                semantic_type=entity.semantic_type,
                confidence=entity.confidence,
                evidence=evidence,
                source_layers=("typed_entity",),
                source_ids=(entity.entity_id,),
                verified=bool(evidence and evidence.reviewer_status in VERIFIED_STATUSES),
            )
        )

    return _deduplicate_facts(facts)


def _deduplicate_facts(facts: list[SemanticFact]) -> list[SemanticFact]:
    retained: dict[tuple[str, str, str, str], SemanticFact] = {}
    for fact in facts:
        evidence_id = fact.evidence.evidence_id if fact.evidence else ""
        key = (
            fact.name,
            fact.semantic_type.value if fact.semantic_type else "",
            fact.normalized_value.casefold(),
            evidence_id,
        )
        existing = retained.get(key)
        if existing is None:
            retained[key] = fact
            continue
        preferred = (
            fact
            if (fact.verified, fact.confidence) > (existing.verified, existing.confidence)
            else existing
        )
        retained[key] = replace(
            preferred,
            confidence=max(existing.confidence, fact.confidence),
            source_layers=tuple(dict.fromkeys((*existing.source_layers, *fact.source_layers))),
            source_ids=tuple(dict.fromkeys((*existing.source_ids, *fact.source_ids))),
            verified=existing.verified or fact.verified,
        )
    return sorted(
        retained.values(),
        key=lambda item: (item.name, item.normalized_value, item.source_layers),
    )


__all__ = [
    "SemanticFact",
    "collect_semantic_facts",
    "normalize_label",
    "normalize_semantic_value",
    "semantic_type_for_name",
]
