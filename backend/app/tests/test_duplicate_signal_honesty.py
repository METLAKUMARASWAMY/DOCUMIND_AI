"""A perceptual hash with no information must not be reported as similarity.

Found by the cross-domain generalization test: a clinical laboratory report and
a bill of lading — sharing no entity, party, reference or amount — were linked
`DUPLICATE_OF` in the graph and raised a "Duplicate Document Review" risk.

The cause is structural, not fixture-specific. A dHash bit records "this pixel
is darker than the next one". On a mostly-blank page nearly every comparison is
a tie, so nearly every bit is 0, and any two such pages agree on ~98% of their
bits — comfortably over the 0.92 duplicate threshold. Measured set-bit ratios:

    real content pages   0.24 - 0.40
    three-line page      0.008
    blank page           0.000

So the fix is not a higher threshold, which would only move the false positive:
it is refusing to report a measurement the hash cannot support.
"""

from __future__ import annotations

import secrets

import pytest

from app.core.config import settings
from app.models.schemas import TextAnalysisRequest
from app.services.vision.duplicate_detection import (
    perceptual_hash_bit_ratio,
    perceptual_hash_is_discriminative,
    perceptual_similarity,
)
from app.services.vision.service import VisionAIService

RUN = secrets.token_hex(6)

# Two different near-blank pages: 6 and 3 set bits out of 256.
SPARSE_LEFT = "a000a0000000a000000000000000000000000000000000000000000000000000"
SPARSE_RIGHT = "8000900000000000000000000000000000000000000000000000000000000000"
# A page with real ink: roughly a third of the bits set.
DENSE_LEFT = "a5c3f01e93b7c42d5e81aa39cd7f0264b8e1937fac52d80e6b34f19a7c0d58e2"
DENSE_RIGHT = "a5c3f01e93b7c42d5e81aa39cd7f0264b8e1937fac52d80e6b34f19a7c0d58e3"


def test_two_sparse_hashes_would_otherwise_look_like_a_duplicate():
    """The defect, stated as arithmetic. Without the guard this is 0.98."""
    left = int(SPARSE_LEFT, 16)
    right = int(SPARSE_RIGHT, 16)
    raw = 1.0 - (left ^ right).bit_count() / 256
    assert raw > settings.visual_duplicate_threshold, raw


def test_a_sparse_hash_is_not_discriminative():
    assert perceptual_hash_bit_ratio(SPARSE_LEFT) < 0.05
    assert not perceptual_hash_is_discriminative(SPARSE_LEFT)
    assert not perceptual_hash_is_discriminative(SPARSE_RIGHT)


def test_a_dense_hash_is_discriminative():
    ratio = perceptual_hash_bit_ratio(DENSE_LEFT)
    assert 0.2 < ratio < 0.6, ratio
    assert perceptual_hash_is_discriminative(DENSE_LEFT)


def test_an_all_zero_hash_is_not_discriminative():
    assert not perceptual_hash_is_discriminative("0" * 64)


def test_an_all_one_hash_is_not_discriminative():
    """An inverted page is as uninformative as a blank one."""
    assert not perceptual_hash_is_discriminative("f" * 64)


def test_similarity_between_two_sparse_pages_is_not_measurable():
    assert perceptual_similarity(SPARSE_LEFT, SPARSE_RIGHT) is None


def test_similarity_between_two_dense_pages_is_still_measured():
    """The guard must not disable genuine detection."""
    similarity = perceptual_similarity(DENSE_LEFT, DENSE_RIGHT)
    assert similarity is not None
    assert similarity > settings.visual_duplicate_threshold


def test_a_dense_page_against_a_sparse_page_is_not_measurable():
    assert perceptual_similarity(DENSE_LEFT, SPARSE_LEFT) is None


def test_a_multi_page_document_uses_only_the_pages_that_carry_information():
    left = f"{SPARSE_LEFT}:{DENSE_LEFT}"
    right = f"{SPARSE_RIGHT}:{DENSE_RIGHT}"

    similarity = perceptual_similarity(left, right)
    assert similarity is not None
    # Only the dense page pair contributed, so the score reflects that pair.
    assert similarity > settings.visual_duplicate_threshold


def test_a_malformed_hash_is_not_treated_as_a_match():
    assert perceptual_hash_bit_ratio("not-hex") is None
    assert not perceptual_hash_is_discriminative("not-hex")
    assert perceptual_similarity("not-hex", DENSE_LEFT) is None


# --------------------------------------------------------------------------
# The end-to-end shape of the original finding
# --------------------------------------------------------------------------

LAB_REPORT = """Clinical Laboratory Report
Accession: LAB-77301
Specimen Type: Whole blood
Reporting Pathologist: Dr Amara Osei
"""

BILL_OF_LADING = """Bill of Lading
Bill of Lading Number: BOL-55219
Vessel Name: MV Coral Meridian
Port of Discharge: Santos
"""


@pytest.fixture(scope="module")
def unrelated_pair():
    service = VisionAIService()
    tenant = f"tenant-dupfp-{RUN}"
    first = service.process_text(
        TextAnalysisRequest(filename=f"lab-{RUN}.txt", text=LAB_REPORT, metadata={}),
        actor="dup-fp",
        tenant_id=tenant,
    )
    second = service.process_text(
        TextAnalysisRequest(filename=f"bol-{RUN}.txt", text=BILL_OF_LADING, metadata={}),
        actor="dup-fp",
        tenant_id=tenant,
    )
    return first, second


def test_unrelated_documents_are_not_flagged_as_duplicates(unrelated_pair):
    _first, second = unrelated_pair
    statuses = {candidate.status for candidate in (second.duplicate_candidates or [])}
    assert statuses <= {"no_duplicate"}, statuses


def test_unrelated_documents_get_no_duplicate_graph_edge(unrelated_pair):
    first, second = unrelated_pair
    for edge in second.graph.edges:
        assert edge.get("type") != "DUPLICATE_OF", edge
        assert edge.get("target") != first.document_id, edge


def test_unrelated_documents_share_no_graph_node(unrelated_pair):
    first, second = unrelated_pair
    first_nodes = {node["id"] for node in first.graph.nodes}
    second_nodes = {node["id"] for node in second.graph.nodes}
    assert not (first_nodes & second_nodes), first_nodes & second_nodes


def test_no_spurious_duplicate_risk_is_raised(unrelated_pair):
    _first, second = unrelated_pair
    titles = {indicator.title for indicator in second.risk_indicators}
    assert "Duplicate Document Review" not in titles, titles


def test_an_actual_byte_identical_document_is_still_caught(unrelated_pair):
    """Exact duplicates never depended on the perceptual signal — prove it."""
    service = VisionAIService()
    tenant = f"tenant-dupexact-{RUN}"
    service.process_text(
        TextAnalysisRequest(filename=f"first-{RUN}.txt", text=LAB_REPORT, metadata={}),
        actor="dup-exact",
        tenant_id=tenant,
    )
    again = service.process_text(
        TextAnalysisRequest(filename=f"second-{RUN}.txt", text=LAB_REPORT, metadata={}),
        actor="dup-exact",
        tenant_id=tenant,
    )
    statuses = {candidate.status for candidate in (again.duplicate_candidates or [])}
    assert "exact_duplicate" in statuses, statuses
