from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def _headers() -> dict[str, str]:
    return auth_headers(client)


def _analyse_form() -> dict:
    unique = uuid4().hex[:8]
    response = client.post(
        "/api/v1/vision/documents/analyze",
        headers=_headers(),
        json={
            "filename": f"ask-registration-{unique}.txt",
            "text": (
                "Organisation Registration Form\n"
                f"Organisation Name: Northwind Research {unique}\n"
                "Contact Name: Priya Shah\n"
                "FAX NO.: 212-403-2211\n"
                "Email: priya@example.org\n"
                "Signature: Priya Shah\n"
            ),
            "metadata": {"document_type_hint": "form"},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]


def _ask(question: str, document_id: str | None = None) -> dict:
    payload: dict[str, str] = {"question": question}
    if document_id:
        payload["document_id"] = document_id
    response = client.post("/api/v1/ai/ask", headers=_headers(), json=payload)
    assert response.status_code == 200, response.text
    return response.json()["data"]


def test_casual_and_ambiguous_queries_do_not_create_review_work():
    greeting = _ask("Hi")
    assert greeting["answer"] == (
        "Hello. Ask me about extracted fields, tables, clauses, validation issues, "
        "risk indicators or relationships in the selected documents."
    )
    assert greeting["query_type"] == "greeting"
    assert greeting["answer_status"] == "supported"
    assert greeting["citations"] == []
    assert greeting["requires_review"] is False
    assert greeting["retriever"] == "not_used"

    ambiguous = _ask("Is this safe?")
    assert ambiguous["answer"] == (
        "Do you want me to check document quality, financial consistency, suspicious "
        "indicators, signature presence or approval status?"
    )
    assert ambiguous["query_type"] == "ambiguous_question"
    assert ambiguous["answer_status"] == "clarification_needed"
    assert ambiguous["citations"] == []
    assert ambiguous["requires_review"] is False


def test_structured_visual_and_absent_answers_are_evidence_bound():
    document = _analyse_form()
    document_id = document["document_id"]

    fax = _ask("What is the fax number?", document_id)
    assert fax["query_type"] == "structured_lookup"
    assert fax["answer_status"] == "supported"
    assert "212-403-2211" in fax["answer"]
    assert fax["retriever"] == "persisted-structured-field-store"
    assert fax["citations"]
    assert all(item["filename"] == document["filename"] for item in fax["citations"])
    assert all(item["page_number"] == 1 for item in fax["citations"])
    assert all(item["page"] == 1 for item in fax["citations"])
    assert all(item["bounding_boxes"] for item in fax["citations"])
    assert fax["generator"]

    signed = _ask("Is this signed?", document_id)
    assert signed["query_type"] == "visual_question"
    assert signed["answer_status"] == "supported"
    assert signed["citations"]
    assert signed["citations"][0]["filename"] == document["filename"]
    assert signed["citations"][0]["bounding_boxes"]

    absent = _ask("What is the employee birthplace?", document_id)
    assert absent["answer_status"] == "not_found"
    assert absent["citations"] == []
    assert absent["requires_review"] is False
    assert "birthplace" not in absent["answer"].lower()


def test_answer_record_and_feedback_are_available_at_contract_paths():
    document = _analyse_form()
    answer = _ask("What is the organisation name?", document["document_id"])
    assert answer["answer_status"] == "supported"
    assert answer["citations"]

    detail = client.get(
        f"/api/v1/ai/responses/{answer['question_id']}",
        headers=_headers(),
    )
    assert detail.status_code == 200, detail.text
    assert detail.json()["data"]["question_id"] == answer["question_id"]
    assert detail.json()["data"]["retriever"] == "persisted-structured-field-store"

    feedback = client.post(
        f"/api/v1/ai/responses/{answer['question_id']}/feedback",
        headers=_headers(),
        json={"rating": "helpful", "comment": "Citation opened the correct field."},
    )
    assert feedback.status_code == 200, feedback.text
    assert feedback.json()["data"]["rating"] == "helpful"
