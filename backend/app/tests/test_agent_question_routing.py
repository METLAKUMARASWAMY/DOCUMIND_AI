"""A "who" question must reach the graph, not raw OCR text.

`relationship_terms` carried literal phrases, including "who requested". So
"who requested this document" reached the graph and "Who made the original
request?" did not — it fell through to the semantic path, which answered from
raw chunk text and claimed *supported* for:

    original Request Made By___Dr. A. W. Spears on__September original
    Laboratory oGs Original Request Analysis to Dr, oGs Made Av By___Dr.
    W. Spear A. ocs W. Spears to 3.48 30 PMO mm butt add-on on__Sept

while the extracted field held exactly "Dr. A. W. Spears" and the graph held
`REQUESTED_BY -> Person`. This is the D-04 defect class in a path D-04 did not
cover: it works for the phrasing someone anticipated.

Agent detection is now structural rather than phrase-based, and applies only
where the question would otherwise have become `semantic_question`, so no
existing classification changes.
"""

from __future__ import annotations

import pytest

from app.services.vision.query_router import VisionQueryRouter


@pytest.fixture(scope="module")
def router() -> VisionQueryRouter:
    return VisionQueryRouter()


@pytest.mark.parametrize(
    "question",
    [
        "Who made the original request?",
        "who requested this document",
        "who wrote the specifications",
        "who is the landlord",
        "who authorised the payment",
        "whom should this be escalated to",
        "who prepared this report",
    ],
)
def test_an_agent_question_routes_to_the_graph(router, question):
    assert router.route(question) == "relationship_question", question


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("what is the total amount", "structured_field_lookup"),
        ("which vendor issued this invoice", "structured_field_lookup"),
        ("what is the sample number", "structured_field_lookup"),
        ("what is the weather today", "unsupported_question"),
        ("what is the relationship between these documents", "relationship_question"),
    ],
)
def test_existing_classifications_are_unchanged(router, question, expected):
    """The rule is a last resort, so nothing that already routed elsewhere moves."""
    assert router.route(question) == expected, question


@pytest.mark.parametrize(
    "question",
    [
        "what is the whole amount",
        "what is the wholesale price",
        "who's on first",  # apostrophe form still asks for an agent
    ],
)
def test_word_boundaries_are_respected(router, question):
    """ "whole" and "wholesale" contain "who" and must not trigger."""
    route = router.route(question)
    if question.startswith("who's"):
        assert route == "relationship_question"
    else:
        assert route != "relationship_question", (question, route)


@pytest.mark.parametrize(
    "question",
    [
        "whose signature is on the form",
        "Who signed off the laboratory results?",
    ],
)
def test_a_signature_question_still_goes_to_the_visual_handler(router, question):
    """Visual terms are matched before the agent rule, so anything about
    signing is answered from detections.

    "Who signed off ..." is arguably an agent question rather than a visual
    one, and this is the one case where the two readings conflict. It is left
    with the visual handler because moving visual-term precedence would change
    signature detection, and the outcome is safe either way: the visual handler
    abstains when there is no detection and the answer cascades to structured
    lookup rather than returning something wrong. Verified: on a text document
    with no signature it returns not_found, not a fabricated name."""
    assert router.route(question) == "visual_question"


def test_capitalisation_and_punctuation_do_not_matter(router):
    assert router.route("WHO MADE THE ORIGINAL REQUEST?") == "relationship_question"
    assert router.route("  who   made the original request  ") == "relationship_question"


def test_a_question_with_no_interrogative_is_not_an_agent_question(router):
    assert router.route("the document was signed") != "relationship_question"
