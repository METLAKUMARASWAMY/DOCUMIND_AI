"""A question the document cannot answer must not be answered anyway.

Found end to end: asked "what is the client's VAT registration number" of a
contract that records no VAT number at all, the structured-field lookup replied

    client: Ravensmoor Logistics PLC, notice_to_the_other_party: 6. Data
    Protection, the_united_kingdom: 7. Approval, purchase_order_number: PO-...

and labelled it ``partially_supported``. Two things were wrong. Function words
counted as matches, so ``notice_to_the_other_party`` and ``the_united_kingdom``
were selected purely on the token "the"; and a single shared token was enough,
so ``purchase_order_number`` matched on "number" and ``client`` on "client".

A field name must now cover at least half of the question's content words.
"""

from __future__ import annotations

import secrets

import pytest

from app.models.schemas import TextAnalysisRequest
from app.models.vision import VisionAskRequest
from app.services.vision.service import VisionAIService

TENANT = f"tenant-abstain-{secrets.token_hex(3)}"

CONTRACT = (
    "MANAGED SERVICES AGREEMENT\n"
    "Agreement Reference: MSA-2026-0447\n"
    "Service Provider: Thornbury Analytics Limited\n"
    "Client: Ravensmoor Logistics PLC\n"
    "Contract Manager: Priya Raghunathan\n"
    "Agreement Date: 2026-04-17\n"
    "Governing Law: England and Wales\n"
    "Purchase Order Number: PO-RAV-88213\n"
    "Notice To The Other Party: ninety days\n"
)


@pytest.fixture(scope="module")
def service() -> VisionAIService:
    svc = VisionAIService()
    svc.process_text(
        TextAnalysisRequest(filename="abstention-contract.txt", text=CONTRACT, metadata={}),
        actor="abstention",
        tenant_id=TENANT,
    )
    return svc


def ask(service: VisionAIService, question: str):
    return service.ask(VisionAskRequest(question=question), tenant_id=TENANT)


@pytest.mark.parametrize(
    "question",
    [
        "what is the client's VAT registration number",
        "what is the insurance policy number",
        "what is the termination penalty percentage",
    ],
)
def test_a_question_the_document_cannot_answer_is_not_claimed_as_supported(service, question):
    response = ask(service, question)
    if response.answer_status in {"supported", "partially_supported"}:
        answer = response.answer.lower()
        # If it claims support, it must not be a dump of unrelated fields.
        assert "notice_to_the_other_party" not in answer, response.answer
        assert "the_united_kingdom" not in answer, response.answer
        assert "po-rav-88213" not in answer, response.answer


def test_function_words_alone_never_select_a_field(service):
    """ "the" is in almost every field name; it must not make one a match."""
    response = ask(service, "what is the")
    assert response.answer_status not in {"supported"}, response.answer


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("what is the purchase order number", "po-rav-88213"),
        ("who is the service provider", "thornbury"),
        ("what is the agreement date", "2026-04-17"),
        ("what is the governing law", "england"),
    ],
)
def test_questions_the_document_does_answer_still_work(service, question, expected):
    """The coverage rule must not cost real answers."""
    response = ask(service, question)
    assert response.answer_status in {"supported", "partially_supported"}, response.answer
    assert expected in response.answer.lower(), response.answer


def test_field_name_coverage_rule():
    from app.services.vision.service_parts.retrieval import RetrievalMixin

    answers = RetrievalMixin._field_name_answers
    # 3 of 3 content words -> a real answer.
    assert answers({"purchase", "order", "number"}, {"purchase", "order", "number"})
    # 1 of 4 -> incidental.
    assert not answers({"purchase", "order", "number"}, {"client", "vat", "registration", "number"})
    assert not answers({"client"}, {"client", "vat", "registration", "number"})
    # 2 of 2 -> a real answer.
    assert answers({"service", "provider"}, {"service", "provider"})
    # Nothing shared, or nothing asked.
    assert not answers({"client"}, set())
    assert not answers(set(), {"vat"})
