"""An answer must not repeat itself just because retrieval was thorough.

Retrieval deliberately returns the same fact from several sources — a chunk, a
reconstructed OCR line, an extracted field — because each is separate
provenance. Concatenating their text produced:

    'Landlord: Brackenfield Estates Landlord: Brackenfield Estates
     Landlord: Brackenfield Estates landlord: Brackenfield Estates
     Landlord: Brackenfield Estates'

Five citations for one fact is honest provenance and stays. Five copies of the
sentence is not an answer.
"""

from __future__ import annotations

import secrets

import pytest

from app.core.evidence_text import ANSWER_CHARACTER_LIMIT, compose_evidence_text
from app.models.schemas import TextAnalysisRequest
from app.models.vision import VisionAskRequest
from app.services.vision.service import VisionAIService

RUN = secrets.token_hex(6)


def test_identical_blocks_appear_once():
    text = compose_evidence_text(["Landlord: Brackenfield Estates"] * 5)
    assert text == "Landlord: Brackenfield Estates"


def test_case_differences_are_still_the_same_block():
    text = compose_evidence_text(
        ["Landlord: Brackenfield Estates", "landlord: Brackenfield Estates"]
    )
    assert text == "Landlord: Brackenfield Estates"


def test_whitespace_differences_are_still_the_same_block():
    text = compose_evidence_text(["Total:  4,200.00  USD", "Total: 4,200.00 USD"])
    assert text == "Total: 4,200.00 USD"


def test_distinct_facts_are_all_kept():
    text = compose_evidence_text(["Landlord: Brackenfield Estates", "Monthly Rent: 1,275.00 GBP"])
    assert text == "Landlord: Brackenfield Estates Monthly Rent: 1,275.00 GBP"


def test_retrieval_order_is_preserved():
    """Retrieval ranked these; the best block should still lead."""
    text = compose_evidence_text(["second best", "best", "third"])
    assert text.startswith("second best")


def test_a_block_contained_in_another_is_dropped():
    """The common case: a chunk and the OCR line inside it are both retrieved."""
    text = compose_evidence_text(
        ["Lease Agreement Landlord: Brackenfield Estates", "Landlord: Brackenfield Estates"]
    )
    assert text == "Lease Agreement Landlord: Brackenfield Estates"


def test_a_longer_block_arriving_later_subsumes_a_shorter_one():
    text = compose_evidence_text(
        ["Landlord: Brackenfield Estates", "Lease Agreement Landlord: Brackenfield Estates"]
    )
    assert text == "Lease Agreement Landlord: Brackenfield Estates"


def test_blank_and_whitespace_only_blocks_are_dropped():
    assert compose_evidence_text(["", "   ", "\n", "Real evidence"]) == "Real evidence"


def test_no_blocks_at_all_yields_empty_text():
    assert compose_evidence_text([]) == ""


def test_the_character_limit_is_respected():
    text = compose_evidence_text(f"block {index}" for index in range(2000))
    assert len(text) <= ANSWER_CHARACTER_LIMIT


def test_the_limit_is_configurable():
    assert len(compose_evidence_text(["abcdefghij"], limit=4)) == 4


# --------------------------------------------------------------------------
# End to end: the shape of the original finding
# --------------------------------------------------------------------------

LEASE = """Residential Lease Agreement
Landlord: Brackenfield Estates
Monthly Rent: 1,275.00 GBP
Property Address: 14 Harrow Court
"""


@pytest.fixture(scope="module")
def service_with_lease():
    service = VisionAIService()
    tenant = f"tenant-compose-{RUN}"
    service.process_text(
        TextAnalysisRequest(filename=f"lease-{RUN}.txt", text=LEASE, metadata={}),
        actor="compose",
        tenant_id=tenant,
    )
    return service, tenant


# The semantic path is where evidence blocks are concatenated, so the question
# used here must actually route there. An agent question now reaches the graph
# and returns a single precise citation instead, which is a better answer but
# does not exercise deduplication.
SEMANTIC_QUESTION = "what is the monthly rent"


def test_the_question_used_here_really_does_take_the_semantic_path(service_with_lease):
    """Otherwise the tests below would pass without testing anything."""
    service, tenant = service_with_lease
    response = service.ask(VisionAskRequest(question=SEMANTIC_QUESTION), tenant_id=tenant)

    assert response.query_type == "semantic_question", response.query_type
    assert len(response.citations) > 1, "not enough overlapping sources to deduplicate"


def test_an_answer_does_not_repeat_the_same_sentence(service_with_lease):
    service, tenant = service_with_lease
    response = service.ask(VisionAskRequest(question=SEMANTIC_QUESTION), tenant_id=tenant)

    occurrences = response.answer.lower().count("monthly rent")
    assert occurrences == 1, f"repeated {occurrences}x: {response.answer}"


def test_deduplication_does_not_discard_provenance(service_with_lease):
    """Several sources for one fact is good evidence, not noise. The citations
    must survive even though their text collapses to one sentence."""
    service, tenant = service_with_lease
    response = service.ask(VisionAskRequest(question=SEMANTIC_QUESTION), tenant_id=tenant)

    assert len(response.citations) > 1, "distinct provenance was lost"
    for citation in response.citations:
        assert citation.document_id
        assert citation.page_number >= 1


def test_the_answer_still_contains_the_value(service_with_lease):
    service, tenant = service_with_lease
    response = service.ask(VisionAskRequest(question=SEMANTIC_QUESTION), tenant_id=tenant)

    assert "1,275" in response.answer, response.answer
