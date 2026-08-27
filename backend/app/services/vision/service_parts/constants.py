"""Module-level constants shared by the VisionAIService parts.

Extracted verbatim from service.py when the class was split by
responsibility (D-16). service.py re-exports them, so importers that
already reach for `app.services.vision.service.RETRIEVAL_SOURCE_SEQUENCE`
keep working.
"""

from __future__ import annotations

import logging

from app.models.vision import VisionClarificationOption

LOGGER = logging.getLogger("documind.vision")

CONTRACT_CLAUSE_FIELDS = {
    "parties",
    "agreement_date",
    "effective_date",
    "expiry_date",
    "governing_law",
    "renewal_term",
    "termination_for_convenience",
    "notice_period",
    "anti_assignment",
    "exclusivity",
    "non_compete",
    "audit_rights",
    "insurance",
    "cap_on_liability",
    "uncapped_liability",
    "perpetual_licence",
    "revenue_sharing",
    "change_of_control",
    "confidentiality",
    "indemnification",
    "payment_terms",
}

CLARIFICATION_OPTIONS = (
    VisionClarificationOption(
        option_id="document_quality",
        label="Check document quality",
        resolved_question="What document quality and unresolved data-quality issues require human review?",
        query_type="quality_question",
    ),
    VisionClarificationOption(
        option_id="suspicious_indicators",
        label="Check suspicious indicators",
        resolved_question="What suspicious business-rule or visual-manipulation indicators are recorded?",
        query_type="risk_question",
    ),
    VisionClarificationOption(
        option_id="financial_consistency",
        label="Check financial consistency",
        resolved_question="Are the extracted financial values consistent across the selected documents?",
        query_type="comparison_question",
    ),
    VisionClarificationOption(
        option_id="signature_or_stamp",
        label="Check signature or stamp presence",
        resolved_question="Is there a signature or stamp present?",
        query_type="visual_question",
    ),
    VisionClarificationOption(
        option_id="approval_status",
        label="Check approval status",
        resolved_question="What is the current approval status?",
        query_type="approval_question",
    ),
)

RETRIEVAL_SOURCE_SEQUENCE = (
    "verified_structured_fields",
    "reviewed_structured_fields",
    "unverified_extracted_fields",
    "field_aliases",
    "ocr_key_value_pairs",
    "reconstructed_ocr_lines",
    "lexical_bm25_chunks",
    "dense_vector_retrieval",
    "reranked_evidence",
    "graph_facts_and_paths",
    "visual_detections",
    "deterministic_calculations",
)
