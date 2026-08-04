from __future__ import annotations

from typing import Any

CURRENT_PIPELINE_VERSION = "documind-vision-v3"


def legacy_document_flags(document: dict[str, Any]) -> list[str]:
    metadata = dict(document.get("metadata") or {})
    fields = list(document.get("extracted_fields") or [])
    events = list(document.get("audit_events") or [])
    flags: list[str] = []
    pipeline_version = str(metadata.get("pipeline_version") or "legacy")
    manual_verification = any(
        event.get("event_type")
        in {"manual_approval_fields_confirmed", "review_decision_recorded"}
        and (event.get("metadata") or {}).get("confirmed_field_names")
        for event in events
    )
    if document.get("status") == "APPROVED" and not fields and not manual_verification:
        flags.append("LEGACY_APPROVAL_REVIEW_REQUIRED")
    if pipeline_version != CURRENT_PIPELINE_VERSION:
        flags.append("OUTDATED_PIPELINE")
    if not fields and document.get("ocr_words"):
        flags.append("REPROCESS_REQUIRED")
    language = dict(document.get("language_analysis") or {})
    review = dict(document.get("review_task") or {})
    if not language.get("mixed_language_routing") and any(
        reason == "mixed_languages_detected" for reason in review.get("reasons", [])
    ):
        flags.append("STALE_LANGUAGE_REVIEW_REASON")
    return flags
