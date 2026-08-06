from __future__ import annotations

import json

import pytest

from app.services.vision.generator_adapters import (
    GroundedGeneratorError,
    validate_grounded_output,
)

EVIDENCE = [
    {
        "id": "evidence-sample-number",
        "text": "Sample No. 6030",
        "document_id": "vision-doc-grounding",
        "page_number": 1,
        "review_status": "accepted",
    }
]


def test_grounded_generator_accepts_only_supplied_citation_ids() -> None:
    result = validate_grounded_output(
        json.dumps(
            {
                "answer": "The sample number is 6030.",
                "answer_status": "supported",
                "citation_ids": ["evidence-sample-number"],
                "reasoning_summary": "The cited field supplies the number.",
                "requires_review": False,
            }
        ),
        EVIDENCE,
    )

    assert result["citation_ids"] == ["evidence-sample-number"]


def test_grounded_generator_rejects_citation_id_not_supplied_to_model() -> None:
    with pytest.raises(GroundedGeneratorError, match="not supplied"):
        validate_grounded_output(
            json.dumps(
                {
                    "answer": "The sample number is 6030.",
                    "answer_status": "supported",
                    "citation_ids": ["invented-evidence-id"],
                    "reasoning_summary": "Unsupported citation.",
                    "requires_review": False,
                }
            ),
            EVIDENCE,
        )


def test_grounded_generator_rejects_number_absent_from_evidence() -> None:
    with pytest.raises(GroundedGeneratorError, match="number absent"):
        validate_grounded_output(
            json.dumps(
                {
                    "answer": "The sample number is 9999.",
                    "answer_status": "supported",
                    "citation_ids": ["evidence-sample-number"],
                    "reasoning_summary": "Invented value.",
                    "requires_review": True,
                }
            ),
            EVIDENCE,
        )
