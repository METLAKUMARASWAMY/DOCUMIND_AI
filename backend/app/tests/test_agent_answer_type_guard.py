"""A "who" question must never be answered with a value that cannot be a party.

Found by test_cross_domain_generalization: asked "who owns the property being
let", the graph path answered

    "<file> is connected to Property Address: 14 Harrow Court, Leeds by has attribute"

and labelled it ``supported``. The landlord edge scored zero lexical overlap
(the question shares no token with "Landlord: Brackenfield Estates"), while the
address edge matched the single generic token "property" -- so a coincidence on
one word outranked the real answer and was then presented as fact.

The guard uses the document's own contextual semantic typing: a value the
document typed ADDRESS, MONEY, DATE, IDENTIFIER and so on cannot be the agent a
"who" question asks for. Untyped values stay eligible, so real names -- which
the classifier usually leaves as TEXT -- are unaffected.
"""

from __future__ import annotations

import pytest

from app.models.schemas import TextAnalysisRequest
from app.models.vision import VisionAskRequest
from app.services.vision.service import VisionAIService

TENANT = "tenant-agent-guard"

TENANCY = (
    "TENANCY AGREEMENT\n"
    "Landlord: Brackenfield Estates\n"
    "Tenant: Mr J Patel\n"
    "Property Address: 14 Harrow Court, Leeds\n"
    "Monthly Rent: GBP 1450.00\n"
    "Deposit: GBP 1740.00\n"
    "Agreement Date: 2026-02-11\n"
)


@pytest.fixture(scope="module")
def service() -> VisionAIService:
    svc = VisionAIService()
    svc.process_text(
        TextAnalysisRequest(filename="agent-guard-tenancy.txt", text=TENANCY, metadata={}),
        actor="agent-guard",
        tenant_id=TENANT,
    )
    return svc


def ask(service: VisionAIService, question: str):
    return service.ask(VisionAskRequest(question=question), tenant_id=TENANT)


@pytest.mark.parametrize(
    "question",
    [
        "who owns the property being let",
        "who is responsible for the property",
        "whose property is this",
    ],
)
def test_an_agent_question_is_never_answered_with_a_non_agent_value(service, question):
    response = ask(service, question)
    answer = response.answer.lower()
    if response.answer_status in {"supported", "partially_supported"}:
        # If it claims support, the answer must not be the address or an amount.
        assert "harrow court" not in answer, response.answer
        assert "1450" not in answer, response.answer
        assert "2026-02-11" not in answer, response.answer


def test_a_non_agent_question_can_still_reach_those_values(service):
    """The guard must only apply to agent questions."""
    response = ask(service, "what is the property address")
    assert response.answer_status in {"supported", "partially_supported"}, response.answer
    assert "harrow court" in response.answer.lower(), response.answer


def test_the_rent_is_still_answerable(service):
    response = ask(service, "what is the monthly rent")
    assert response.answer_status in {"supported", "partially_supported"}, response.answer
    assert "1450" in response.answer, response.answer


def test_abstention_is_acceptable_and_must_not_invent_a_party(service):
    """Abstaining is fine. Naming a party the document does not record is not."""
    response = ask(service, "who is the mortgage lender for the property")
    answer = response.answer.lower()
    if response.answer_status in {"supported", "partially_supported"}:
        assert "brackenfield" in answer or "patel" in answer, response.answer
    assert "harrow court" not in answer, response.answer
