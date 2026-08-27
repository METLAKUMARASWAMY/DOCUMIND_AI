"""Regression tests for D-12: silent generator fallbacks.

Directive §50 — "Do NOT silently switch to fake data / demo data / a random
document / a previous answer. Fallbacks are allowed only when explicitly designed
and disclosed."

Before the fix, ``_apply_grounded_generator`` swallowed every generator exception
and silently returned the composed-evidence answer, and silently discarded a
generated answer whose citations did not resolve. In both cases the caller could
not tell that anything had gone wrong.
"""

from __future__ import annotations

import pytest

from app.models.schemas import TextAnalysisRequest
from app.models.vision import VisionAskRequest
from app.services.vision.service import VisionAIService

TENANT = "tenant-degradation"


@pytest.fixture
def service_with_document() -> VisionAIService:
    service = VisionAIService()
    service.process_text(
        TextAnalysisRequest(
            filename="degradation-invoice.txt",
            text=(
                "Invoice INV-2026-5501\n"
                "Vendor: Northwind Trading\n"
                "Invoice Date: 2026-05-02\n"
                "Total Amount: INR 128,400.00\n"
                "Payment Terms: Net 45\n"
                "The vendor provides quarterly logistics services under this invoice.\n"
            ),
            metadata={},
        ),
        actor="degradation-test",
        tenant_id=TENANT,
    )
    return service


class _ExplodingGenerator:
    adapter_id = "exploding-generator"
    model_version = "v-test"

    def generate(self, question, evidence):
        raise RuntimeError("generator backend unreachable")


class _UngroundedGenerator:
    """Claims support but cites evidence that is not in the retrieved set."""

    adapter_id = "ungrounded-generator"
    model_version = "v-test"

    def generate(self, question, evidence):
        return {
            "answer": "The total is INR 999,999.00.",
            "answer_status": "supported",
            "citation_ids": ["chunk-that-was-never-retrieved"],
            "reasoning_summary": "fabricated",
            "requires_review": False,
        }


def _ask(service: VisionAIService, question: str):
    return service.ask(VisionAskRequest(question=question), tenant_id=TENANT)


def test_generator_failure_is_disclosed_not_silent(
    service_with_document: VisionAIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = service_with_document
    monkeypatch.setattr(service.generator_router, "active", lambda: _ExplodingGenerator())

    response = _ask(service, "describe the quarterly logistics arrangement")

    assert response.degraded is True, "a failed generator must be disclosed on the response"
    reported = [item for item in response.degradations if item.stage == "answer_generation"]
    assert reported, f"expected an answer_generation degradation, got {response.degradations}"
    assert reported[0].component == "exploding-generator"
    assert reported[0].impact == "generator_skipped"
    assert "RuntimeError" in reported[0].detail
    # The answer itself must still be usable and cited.
    assert response.answer
    assert response.citations


def test_ungrounded_generated_answer_is_rejected_and_disclosed(
    service_with_document: VisionAIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = service_with_document
    monkeypatch.setattr(service.generator_router, "active", lambda: _UngroundedGenerator())

    response = _ask(service, "describe the quarterly logistics arrangement")

    assert "999,999" not in response.answer, "an ungrounded claim must never be returned"
    assert response.degraded is True
    reported = [item for item in response.degradations if item.stage == "grounding_validation"]
    assert reported, f"expected a grounding_validation degradation, got {response.degradations}"
    assert reported[0].component == "grounding_validator"
    assert reported[0].impact == "generated_answer_rejected"


def test_healthy_answer_reports_no_degradation(
    service_with_document: VisionAIService,
) -> None:
    response = _ask(service_with_document, "what is the total amount")

    assert response.degradations == []
    assert response.degraded is False


def test_degradations_survive_api_serialisation(
    service_with_document: VisionAIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disclosure has to reach the client, not just the Python object."""

    service = service_with_document
    monkeypatch.setattr(service.generator_router, "active", lambda: _ExplodingGenerator())

    payload = _ask(service, "describe the quarterly logistics arrangement").model_dump(mode="json")

    assert payload["degraded"] is True
    assert payload["degradations"]
    assert payload["degradations"][0]["stage"] == "answer_generation"
