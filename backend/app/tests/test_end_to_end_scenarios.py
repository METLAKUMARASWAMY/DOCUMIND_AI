"""The six end-to-end scenarios from directive §56, as executable tests.

These are deliberately written against *newly authored* documents, not the
shipped fixtures, so they exercise generalization rather than memorisation
(directive §57).

  A  upload one new document -> process -> extract -> embed -> index -> graph
     -> ask -> retrieve -> answer with citations
  B  upload several related documents -> build relationships -> ask a
     cross-document question -> answer citing more than one document
  C  ask an unsupported question -> abstain
  D  upload a low-confidence document -> flag -> route to review
  E  an unauthorised user attempts access -> reject
  F  disable an optional model -> the system continues through a disclosed
     fallback
"""

from __future__ import annotations

import secrets

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import create_app
from app.models.schemas import SignupRequest, TextAnalysisRequest, UserRole
from app.models.vision import VisionAskRequest, VisionProcessingStatus
from app.services.auth import auth_service
from app.services.vision.service import VisionAIService

RUN = secrets.token_hex(6)
TENANT = f"tenant-e2e-{RUN}"

# --- documents authored for this test, unlike anything in var/test_fixtures ---

INVOICE = """Invoice INV-E2E-4400
Vendor: Kestrel Instrumentation
Vendor Address: 18 Harbour Road, Cork
Invoice Date: 2026-02-09
Due Date: 2026-03-11
PO Number: PO-E2E-9120
Contract Number: CTR-E2E-3300
Payment Terms: Net 30
Total Amount: EUR 64,800.00
Kestrel supplied calibration instruments for the metrology laboratory.
"""

PURCHASE_ORDER = """Purchase Order: PO-E2E-9120
Supplier: Kestrel Instrumentation
Buyer: Northgate Analytics
Contract Number: CTR-E2E-3300
Effective Date: 2026-01-28
Payment Terms: Net 30
Total Amount: EUR 64,800.00
Approved By: R. Whelan
"""

CONTRACT = """Master Services Contract
Contract Number: CTR-E2E-3300
Party: Northgate Analytics
Contractor: Kestrel Instrumentation
Agreement Date: 2026-01-05
Expiry Date: 2027-01-04
Payment Terms: Net 30
Governing Law: Ireland
The contractor shall provide calibration and metrology services.
"""

# Sparse, ambiguous, low-signal: should not confidently auto-approve.
LOW_CONFIDENCE = """scanned copy
amt ???
vndr .....
dt //
illegible
"""


@pytest.fixture(scope="module")
def service() -> VisionAIService:
    return VisionAIService()


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def _ingest(service: VisionAIService, text: str, filename: str):
    return service.process_text(
        TextAnalysisRequest(filename=filename, text=text, metadata={}),
        actor="e2e",
        tenant_id=TENANT,
    )


def _ask(service: VisionAIService, question: str, **kwargs):
    return service.ask(VisionAskRequest(question=question, **kwargs), tenant_id=TENANT)


# ------------------------------------------------------------------ Scenario A


def test_scenario_a_single_new_document_end_to_end(service: VisionAIService) -> None:
    document = _ingest(service, INVOICE, f"e2e-invoice-{RUN}.txt")

    # process -> extract
    assert document.document_id
    assert document.extracted_fields, "extraction produced no fields"
    fields = {field.name: field.value for field in document.extracted_fields}
    assert any("Kestrel" in str(value) for value in fields.values()), fields

    # chunk + index
    assert document.evidence_chunks, "no evidence chunks were produced"
    for chunk in document.evidence_chunks:
        assert chunk.document_id == document.document_id
        assert chunk.tenant_id == TENANT
        assert chunk.page_number >= 1

    # graph
    assert document.graph.nodes, "no graph nodes were produced"
    assert document.graph.edges, "no graph edges were produced"

    # ask -> retrieve -> answer with citations
    response = _ask(service, "what is the total amount")
    assert response.answer_status == "supported", response.answer
    assert "64,800" in response.answer, response.answer
    assert response.citations, "a supported answer must carry citations"

    # every citation must be traceable back to a real document and page
    for citation in response.citations:
        assert citation.document_id
        assert citation.page_number >= 1
        assert citation.source_text or citation.evidence_id


def test_scenario_a_provenance_is_complete(service: VisionAIService) -> None:
    """Directive §49: a result must not be detached from its source."""

    document = _ingest(service, INVOICE, f"e2e-provenance-{RUN}.txt")

    for field in document.extracted_fields:
        assert field.evidence.document_id == document.document_id
        assert field.evidence.page_number >= 1
        assert field.evidence.model_name
        assert field.evidence.model_version
        assert field.evidence.processing_timestamp


# ------------------------------------------------------------------ Scenario B


@pytest.fixture(scope="module")
def related_corpus(service: VisionAIService) -> dict[str, str]:
    invoice = _ingest(service, INVOICE, f"e2e-b-invoice-{RUN}.txt")
    order = _ingest(service, PURCHASE_ORDER, f"e2e-b-po-{RUN}.txt")
    contract = _ingest(service, CONTRACT, f"e2e-b-contract-{RUN}.txt")
    return {
        "invoice": invoice.document_id,
        "purchase_order": order.document_id,
        "contract": contract.document_id,
    }


def test_scenario_b_cross_document_question_cites_multiple_documents(
    service: VisionAIService, related_corpus: dict[str, str]
) -> None:
    response = _ask(service, "compare the total amount across the documents", top_k=10)

    cited = {citation.document_id for citation in response.citations}
    assert response.answer_status in {"supported", "partially_supported"}, response.answer
    assert len(cited) >= 2, (
        f"a cross-document question cited only {len(cited)} document(s): {response.answer}"
    )


def test_scenario_b_shared_reference_links_the_documents(
    service: VisionAIService, related_corpus: dict[str, str]
) -> None:
    """The invoice, PO and contract share CTR-E2E-3300 and PO-E2E-9120."""

    documents = [
        service.get_document(document_id, TENANT) for document_id in related_corpus.values()
    ]
    contract_numbers = {
        field.value
        for document in documents
        for field in document.extracted_fields
        if field.name == "contract_number"
    }
    assert any("CTR-E2E-3300" in str(value) for value in contract_numbers), (
        f"the shared contract reference was not extracted: {contract_numbers}"
    )


def test_scenario_b_amounts_agree_between_invoice_and_order(
    service: VisionAIService, related_corpus: dict[str, str]
) -> None:
    invoice = service.get_document(related_corpus["invoice"], TENANT)
    order = service.get_document(related_corpus["purchase_order"], TENANT)

    def total(document) -> str | None:
        for field in document.extracted_fields:
            if field.name in {"total_amount", "amount"}:
                return field.value
        return None

    assert total(invoice) and total(order)
    digits = lambda value: "".join(character for character in value if character.isdigit())  # noqa: E731
    assert digits(total(invoice)) == digits(total(order)), (
        f"invoice {total(invoice)!r} vs order {total(order)!r}"
    )


# ------------------------------------------------------------------ Scenario C


@pytest.mark.parametrize(
    "question",
    [
        "what is the mean time between failures for the turbine",
        "what was the closing share price on the day of issue",
    ],
)
def test_scenario_c_unsupported_question_abstains(
    service: VisionAIService, related_corpus: dict[str, str], question: str
) -> None:
    response = _ask(service, question)

    assert response.answer_status == "not_found", (
        f"expected abstention, got {response.answer_status}: {response.answer}"
    )
    assert response.citations == [], "an abstention must not carry citations"
    assert "not contain sufficient evidence" in response.answer.lower()


def test_scenario_c_partially_answerable_question_is_not_overclaimed(
    service: VisionAIService, related_corpus: dict[str, str]
) -> None:
    """The corpus knows the contractor but nothing about employees.

    Purely lexical relevance cannot prove such a question is unanswerable -- a
    legitimate paraphrase also leaves words unmatched. What the system must never
    do is claim *full* support for evidence that does not address the question.
    Reporting partial support, and naming the part it could not cover, is the
    honest outcome.
    """

    response = _ask(service, "how many employees does the contractor have")

    assert response.answer_status != "supported", (
        f"overclaimed support: {response.answer}"
    )
    if response.answer_status == "partially_supported":
        assert "employees" in response.reasoning_summary, response.reasoning_summary


def test_scenario_c_abstention_still_reports_what_was_searched(
    service: VisionAIService, related_corpus: dict[str, str]
) -> None:
    """Directive §21: show retrieved count, documents searched, reason."""

    response = _ask(service, "what is the mean time between failures for the turbine")

    assert response.scope, "abstention must still report the scope that was searched"
    assert response.scope.get("document_count", 0) >= 1
    assert response.reasoning_summary, "abstention must give a reason"


def test_scenario_c_out_of_domain_request_is_refused(service: VisionAIService) -> None:
    response = _ask(service, "what is the weather in Cork")
    assert response.answer_status == "not_found"
    assert response.citations == []


# ------------------------------------------------------------------ Scenario D


def test_scenario_d_low_confidence_document_is_flagged_for_review(
    service: VisionAIService,
) -> None:
    document = _ingest(service, LOW_CONFIDENCE, f"e2e-lowconf-{RUN}.txt")

    assert document.status != VisionProcessingStatus.approved, (
        "an illegible document must not auto-approve"
    )
    assert document.review_task is not None, "a low-signal document must raise a review task"
    assert document.review_task.reasons, "the review task must say why"


def test_scenario_d_low_confidence_does_not_become_confident_information(
    service: VisionAIService,
) -> None:
    """Directive §8: low-confidence OCR must not silently become high confidence."""

    document = _ingest(service, LOW_CONFIDENCE, f"e2e-lowconf-2-{RUN}.txt")

    for field in document.extracted_fields:
        if field.confidence is not None:
            assert field.confidence <= 1.0
            # Nothing extracted from this text should read as verified.
            assert field.reviewer_status.value == "unverified"


# ------------------------------------------------------------------ Scenario E


def test_scenario_e_unauthorised_user_is_rejected(client: TestClient) -> None:
    response = client.get("/api/v1/documents")
    assert response.status_code in {401, 403}


def test_scenario_e_viewer_cannot_perform_reviewer_actions(client: TestClient) -> None:
    """Role separation must be enforced by the API, not only the UI."""

    email = f"pytest-e2e-viewer-{RUN}@documind.test"
    password = f"Viewer-{secrets.token_urlsafe(18)}-Aa1!"
    auth_service.create_bootstrap_user(
        SignupRequest(
            name="E2E Viewer",
            email=email,
            password=password,
            role=UserRole.viewer,
            organisation_name=f"E2E Viewer Org {RUN}",
        )
    )
    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    headers = {"Authorization": f"Bearer {login.json()['data']['access_token']}"}

    # A viewer may read...
    assert client.get("/api/v1/documents", headers=headers).status_code == 200

    # ...but must not seed, delete or approve.
    assert client.post("/api/v1/demo/vision/seed", headers=headers).status_code in {401, 403}


# ------------------------------------------------------------------ Scenario F


def test_scenario_f_disabled_optional_model_falls_back_and_discloses(
    service: VisionAIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turning an optional model off must degrade, not break."""

    monkeypatch.setattr(settings, "enable_heavy_models", False)
    monkeypatch.setattr(settings, "enable_bge_m3", False)

    document = _ingest(service, INVOICE, f"e2e-fallback-{RUN}.txt")
    assert document.extracted_fields, "core extraction must survive optional models being off"

    response = _ask(service, "what is the total amount")
    assert response.answer_status == "supported", response.answer
    assert response.citations

    status = service.retrieval_index.status()
    # The fallback is named, not hidden.
    assert status["embedding"]["adapter_id"] == "local-hashing-embedding"
    assert status["vector_backend"] in {"local_hybrid_only", "qdrant_mirrored_local_hybrid"}


def test_scenario_f_disabled_model_reports_disabled_not_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.vision.retrieval_adapters import BGEM3EmbeddingAdapter

    monkeypatch.setattr(settings, "enable_heavy_models", False)
    monkeypatch.setattr(settings, "enable_bge_m3", False)

    status = BGEM3EmbeddingAdapter().status()

    assert status.status == "disabled", f"a disabled model must not claim {status.status!r}"
    assert status.reason


def test_scenario_f_qdrant_outage_does_not_break_answering(
    service: VisionAIService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directive §41 case 12: Qdrant unavailable."""

    monkeypatch.setattr(settings, "enable_qdrant", True)
    monkeypatch.setattr(settings, "qdrant_url", "http://127.0.0.1:6399")
    service.retrieval_index.qdrant.invalidate_liveness_cache()

    document = _ingest(service, INVOICE, f"e2e-qdrant-out-{RUN}.txt")
    assert document.extracted_fields

    response = _ask(service, "what is the total amount")
    assert response.answer_status == "supported", response.answer
    assert service.retrieval_index.status()["vector_backend"] == "local_hybrid_only"
