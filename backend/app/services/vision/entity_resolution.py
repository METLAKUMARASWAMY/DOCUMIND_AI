"""Conservative, tenant-scoped resolution for evidence-backed typed entities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import ClassVar

from app.core.security import hash_document
from app.models.vision import TypedVisionEntity, VisionDocumentRecord, VisionSemanticType


@dataclass(frozen=True)
class EntityResolutionDecision:
    entity_id: str
    matched_entity_id: str
    matched_document_id: str
    semantic_type: VisionSemanticType
    status: str
    method: str
    confidence: float
    canonical_entity_id: str | None
    automatic_merge: bool
    evidence_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "matched_entity_id": self.matched_entity_id,
            "matched_document_id": self.matched_document_id,
            "semantic_type": self.semantic_type.value,
            "status": self.status,
            "method": self.method,
            "confidence": self.confidence,
            "canonical_entity_id": self.canonical_entity_id,
            "automatic_merge": self.automatic_merge,
            "evidence_ids": list(self.evidence_ids),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class EntityResolutionReport:
    entities: tuple[TypedVisionEntity, ...]
    decisions: tuple[EntityResolutionDecision, ...]
    input_entity_count: int
    deduplicated_entity_count: int
    tenant_id: str
    model_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "tenant_id": self.tenant_id,
            "input_entity_count": self.input_entity_count,
            "deduplicated_entity_count": self.deduplicated_entity_count,
            "duplicates_removed": self.input_entity_count - self.deduplicated_entity_count,
            "resolved_matches": sum(item.status == "resolved" for item in self.decisions),
            "possible_matches": sum(item.status == "possible_match" for item in self.decisions),
            "shared_identifiers": sum(
                item.status == "shared_identifier" for item in self.decisions
            ),
            "decisions": [item.as_dict() for item in self.decisions],
        }


class ConservativeEntityResolver:
    """Resolve only strong identifiers or explicit reviewer-approved aliases.

    Similar names are reported as candidates.  They are never silently merged.
    Raw values are intentionally omitted from the report so bank/contact values
    are not copied into diagnostic metadata.
    """

    model_version = "tenant-entity-resolution-v1"
    _STRONG_TYPES: ClassVar[set[VisionSemanticType]] = {
        VisionSemanticType.bank_account,
        VisionSemanticType.email,
        VisionSemanticType.phone,
        VisionSemanticType.url,
    }
    _NAME_TYPES: ClassVar[set[VisionSemanticType]] = {
        VisionSemanticType.organisation,
        VisionSemanticType.person,
        VisionSemanticType.product,
    }
    _REFERENCE_TYPES: ClassVar[set[VisionSemanticType]] = {
        VisionSemanticType.document_reference,
        VisionSemanticType.identifier,
    }

    def resolve(
        self,
        document: VisionDocumentRecord,
        corpus: list[VisionDocumentRecord] | tuple[VisionDocumentRecord, ...],
        *,
        reviewed_value_aliases: dict[str, dict[str, str]] | None = None,
    ) -> EntityResolutionReport:
        input_count = len(document.typed_entities)
        aliases = reviewed_value_aliases or {}
        entities = self._deduplicate(document, aliases)
        decisions: list[EntityResolutionDecision] = []
        for current in entities:
            for candidate_document in corpus:
                if candidate_document.document_id == document.document_id:
                    continue
                if candidate_document.tenant_id != document.tenant_id:
                    continue
                for candidate in self._deduplicate(candidate_document, aliases):
                    decision = self._compare(
                        document.tenant_id,
                        current,
                        candidate,
                        candidate_document.document_id,
                        aliases,
                    )
                    if decision is not None:
                        decisions.append(decision)
        decisions = self._deduplicate_decisions(decisions)
        entities = self._attach_resolution_context(
            document.tenant_id,
            document.document_id,
            entities,
            decisions,
            aliases,
        )
        return EntityResolutionReport(
            entities=tuple(entities),
            decisions=tuple(decisions),
            input_entity_count=input_count,
            deduplicated_entity_count=len(entities),
            tenant_id=document.tenant_id,
            model_version=self.model_version,
        )

    def _deduplicate(
        self,
        document: VisionDocumentRecord,
        aliases: dict[str, dict[str, str]],
    ) -> list[TypedVisionEntity]:
        retained: dict[tuple[VisionSemanticType, str], TypedVisionEntity] = {}
        counts: dict[tuple[VisionSemanticType, str], int] = {}
        for entity in document.typed_entities:
            normalized = self._canonical_value(entity, aliases)
            if not normalized:
                normalized = self._normalize_entity_value(entity)
            key = (entity.semantic_type, normalized)
            counts[key] = counts.get(key, 0) + 1
            existing = retained.get(key)
            if existing is None:
                retained[key] = entity
                continue
            preferred = (
                entity
                if (not entity.review_required, entity.confidence)
                > (not existing.review_required, existing.confidence)
                else existing
            )
            retained[key] = preferred.model_copy(
                update={
                    "confidence": max(existing.confidence, entity.confidence),
                    "evidence_ids": list(
                        dict.fromkeys((*existing.evidence_ids, *entity.evidence_ids))
                    ),
                    "conflict_ids": list(
                        dict.fromkeys((*existing.conflict_ids, *entity.conflict_ids))
                    ),
                    "alternatives": [
                        *existing.alternatives,
                        *(
                            item
                            for item in entity.alternatives
                            if item not in existing.alternatives
                        ),
                    ],
                    "review_required": existing.review_required or entity.review_required,
                }
            )
        result: list[TypedVisionEntity] = []
        for key, entity in retained.items():
            context = dict(entity.context)
            context["within_document_deduplication"] = {
                "observations": counts[key],
                "duplicates_removed": counts[key] - 1,
                "method": "exact_typed_normalized_value",
                "evidence_ids": list(entity.evidence_ids),
            }
            result.append(entity.model_copy(update={"context": context}))
        return sorted(result, key=lambda item: (item.semantic_type.value, item.entity_id))

    def _compare(
        self,
        tenant_id: str,
        current: TypedVisionEntity,
        candidate: TypedVisionEntity,
        candidate_document_id: str,
        aliases: dict[str, dict[str, str]],
    ) -> EntityResolutionDecision | None:
        if current.semantic_type != candidate.semantic_type:
            return None
        if min(current.confidence, candidate.confidence) < 0.6:
            return None
        current_raw = self._normalize_entity_value(current)
        candidate_raw = self._normalize_entity_value(candidate)
        current_value = self._canonical_value(current, aliases)
        candidate_value = self._canonical_value(candidate, aliases)
        if not current_value or not candidate_value:
            return None
        alias_applied = current_value != current_raw or candidate_value != candidate_raw
        evidence_ids = tuple(dict.fromkeys((*current.evidence_ids, *candidate.evidence_ids)))
        base_confidence = min(current.confidence, candidate.confidence)

        status: str
        method: str
        confidence: float
        automatic_merge = False
        reasons: tuple[str, ...]
        canonical_entity_id: str | None = None
        if current_value == candidate_value and alias_applied:
            if current.review_required or candidate.review_required:
                status = "possible_match"
                method = "reviewed_value_alias_unreviewed_observation"
                confidence = round(base_confidence * 0.9, 4)
                reasons = (
                    "The alias is reviewer-approved, but at least one current observation still requires review.",
                )
            else:
                status = "resolved"
                method = "reviewed_value_alias"
                confidence = min(1.0, max(0.95, base_confidence))
                automatic_merge = True
                reasons = ("Reviewer-approved value alias maps both observations to one value.",)
                canonical_entity_id = self._canonical_id(
                    tenant_id,
                    current.semantic_type,
                    current_value,
                )
        elif current_value == candidate_value and current.semantic_type in self._STRONG_TYPES:
            if current.review_required or candidate.review_required:
                status = "possible_match"
                method = "exact_strong_identifier_unreviewed"
                confidence = round(base_confidence * 0.9, 4)
                reasons = (
                    "Exact identifier agrees, but at least one observation requires review.",
                )
            else:
                status = "resolved"
                method = "exact_strong_identifier"
                confidence = round(base_confidence * 0.98, 4)
                automatic_merge = True
                reasons = ("Exact normalized strong identifier agrees within the tenant.",)
                canonical_entity_id = self._canonical_id(
                    tenant_id,
                    current.semantic_type,
                    current_value,
                )
        elif current_value == candidate_value and current.semantic_type in self._REFERENCE_TYPES:
            status = "shared_identifier"
            method = "exact_document_identifier"
            confidence = round(base_confidence * 0.96, 4)
            reasons = (
                "Exact document identifier is shared; related documents remain distinct entities.",
            )
            canonical_entity_id = self._canonical_id(
                tenant_id,
                current.semantic_type,
                current_value,
            )
        elif current_value == candidate_value and current.semantic_type in self._NAME_TYPES:
            status = "possible_match"
            method = "exact_name_only"
            confidence = round(base_confidence * 0.84, 4)
            reasons = ("Name agreement alone is insufficient for automatic entity merging.",)
        elif current.semantic_type == VisionSemanticType.organisation:
            current_legal = self._organisation_normalize(current_value)
            candidate_legal = self._organisation_normalize(candidate_value)
            similarity = SequenceMatcher(None, current_legal, candidate_legal).ratio()
            if current_legal == candidate_legal:
                status = "possible_match"
                method = "legal_suffix_normalization"
                confidence = round(min(base_confidence, 0.82), 4)
                reasons = (
                    "Legal-suffix normalization agrees, but no independent identifier corroborates the name.",
                )
            elif similarity >= 0.9:
                status = "possible_match"
                method = "high_name_similarity"
                confidence = round(min(base_confidence * similarity, 0.74), 4)
                reasons = (
                    "High organisation-name similarity is a review candidate, not an automatic merge.",
                )
            else:
                return None
        else:
            return None
        return EntityResolutionDecision(
            entity_id=current.entity_id,
            matched_entity_id=candidate.entity_id,
            matched_document_id=candidate_document_id,
            semantic_type=current.semantic_type,
            status=status,
            method=method,
            confidence=confidence,
            canonical_entity_id=canonical_entity_id,
            automatic_merge=automatic_merge,
            evidence_ids=evidence_ids,
            reasons=reasons,
        )

    def _attach_resolution_context(
        self,
        tenant_id: str,
        document_id: str,
        entities: list[TypedVisionEntity],
        decisions: list[EntityResolutionDecision],
        aliases: dict[str, dict[str, str]],
    ) -> list[TypedVisionEntity]:
        by_entity: dict[str, list[EntityResolutionDecision]] = {}
        for decision in decisions:
            by_entity.setdefault(decision.entity_id, []).append(decision)
        enriched: list[TypedVisionEntity] = []
        for entity in entities:
            entity_decisions = by_entity.get(entity.entity_id, [])
            resolved = next(
                (item for item in entity_decisions if item.automatic_merge),
                None,
            )
            context = dict(entity.context)
            context["entity_resolution"] = {
                "model_version": self.model_version,
                "tenant_scoped": True,
                "canonical_entity_id": (resolved.canonical_entity_id if resolved else None),
                "observation_entity_id": self._observation_id(
                    tenant_id,
                    document_id,
                    entity.semantic_type,
                    entity.entity_id,
                ),
                "automatic_merge": bool(resolved),
                "candidate_count": len(entity_decisions),
                "decision_statuses": sorted({item.status for item in entity_decisions}),
                "resolution_confidence": max(
                    (item.confidence for item in entity_decisions),
                    default=None,
                ),
                "resolution_methods": sorted({item.method for item in entity_decisions}),
                "matched_document_ids": sorted(
                    {item.matched_document_id for item in entity_decisions}
                ),
                "evidence_ids": list(entity.evidence_ids),
            }
            enriched.append(entity.model_copy(update={"context": context}))
        return enriched

    def _deduplicate_decisions(
        self,
        decisions: list[EntityResolutionDecision],
    ) -> list[EntityResolutionDecision]:
        retained: dict[tuple[str, str, str], EntityResolutionDecision] = {}
        for decision in decisions:
            key = (
                decision.entity_id,
                decision.matched_document_id,
                decision.matched_entity_id,
            )
            existing = retained.get(key)
            if existing is None or decision.confidence > existing.confidence:
                retained[key] = decision
        return sorted(
            retained.values(),
            key=lambda item: (
                item.entity_id,
                item.matched_document_id,
                item.matched_entity_id,
            ),
        )

    def _canonical_value(
        self,
        entity: TypedVisionEntity,
        aliases: dict[str, dict[str, str]],
    ) -> str:
        value = self._normalize_entity_value(entity)
        typed_aliases = aliases.get(entity.semantic_type.value, {})
        seen: set[str] = set()
        while value in typed_aliases and value not in seen:
            seen.add(value)
            value = self._normalize_for_type(typed_aliases[value], entity.semantic_type)
        return value

    def _canonical_id(
        self,
        tenant_id: str,
        semantic_type: VisionSemanticType,
        value: str,
    ) -> str:
        identity = f"{tenant_id}|{semantic_type.value}|{value}".encode()
        return f"resolved-entity-{hash_document(identity)[:20]}"

    def _observation_id(
        self,
        tenant_id: str,
        document_id: str,
        semantic_type: VisionSemanticType,
        entity_id: str,
    ) -> str:
        identity = f"{tenant_id}|{document_id}|{semantic_type.value}|{entity_id}".encode()
        return f"entity-observation-{hash_document(identity)[:20]}"

    def _basic_normalize(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    def _normalize_entity_value(self, entity: TypedVisionEntity) -> str:
        return self._normalize_for_type(
            entity.normalized_value or entity.raw_value,
            entity.semantic_type,
        )

    def _normalize_for_type(
        self,
        value: str,
        semantic_type: VisionSemanticType,
    ) -> str:
        cleaned = value.strip()
        if semantic_type == VisionSemanticType.email:
            return cleaned.casefold()
        if semantic_type == VisionSemanticType.url:
            return cleaned.casefold().rstrip("/")
        if semantic_type == VisionSemanticType.bank_account:
            return re.sub(r"[^a-z0-9]+", "", cleaned.casefold())
        if semantic_type == VisionSemanticType.phone:
            prefix = "+" if cleaned.startswith("+") else ""
            return prefix + re.sub(r"\D+", "", cleaned)
        return self._basic_normalize(cleaned)

    def _organisation_normalize(self, value: str) -> str:
        tokens = self._basic_normalize(value).split()
        expanded: list[str] = []
        replacements = {
            "pvt": "private",
            "ltd": "limited",
            "corp": "corporation",
            "co": "company",
            "inc": "incorporated",
        }
        for token in tokens:
            expanded.append(replacements.get(token, token))
        return " ".join(expanded)


conservative_entity_resolver = ConservativeEntityResolver()


__all__ = [
    "ConservativeEntityResolver",
    "EntityResolutionDecision",
    "EntityResolutionReport",
    "conservative_entity_resolver",
]
