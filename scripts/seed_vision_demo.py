#!/usr/bin/env python3
import json
import os
import urllib.error
import urllib.request


API_BASE = "http://127.0.0.1:8000/api/v1"
DEMO_EMAIL = os.environ.get("DOCUMIND_DEMO_EMAIL", "vision.demo@documind.local")
DEMO_PASSWORD = os.environ.get("DOCUMIND_DEMO_PASSWORD", "Vision-Demo-123!")


DEMO_TEXT = """Invoice INV-1042
Vendor: Vendor A
Invoice Date: 2026-03-01
Due Date: 2026-03-20
Bank Account: ACCT-99999999
Subtotal: INR 400000.00
Tax: INR 25000.00
Discount: INR 0.00
Total Amount: INR 430000.00
item | qty | price | tax | total
service | 1 | 400000 | 25000 | 430000
Signature: Finance Manager
"""


def request(method: str, path: str, payload: dict | None = None, token: str | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{API_BASE}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise SystemExit(f"{method} {path} failed: {exc.code} {detail}") from exc


def login_or_register() -> str:
    login_payload = {"email": DEMO_EMAIL, "password": DEMO_PASSWORD}
    try:
        login = request("POST", "/auth/login", login_payload)
        return login["data"]["access_token"]
    except SystemExit as exc:
        if "401" not in str(exc):
            raise
    request(
        "POST",
        "/auth/register",
        {
            "name": "Vision Demo User",
            "email": DEMO_EMAIL,
            "password": DEMO_PASSWORD,
            "organisation_name": "DocuMind Local Demo",
        },
    )
    login = request("POST", "/auth/login", login_payload)
    return login["data"]["access_token"]


def main() -> None:
    token = login_or_register()
    analysis = request(
        "POST",
        "/vision/documents/analyze",
        {
            "filename": "seeded-vision-invoice.txt",
            "text": DEMO_TEXT,
            "metadata": {
                "approved_bank_identifier": "ACCT-11112222",
                "required_signature": True,
            },
        },
        token,
    )
    document = analysis["data"]
    question = request(
        "POST",
        "/vision/ask",
        {"question": "What is the total amount?", "document_id": document["document_id"]},
        token,
    )
    print(
        json.dumps(
            {
                "document_id": document["document_id"],
                "status": document["status"],
                "fields": len(document["extracted_fields"]),
                "risk_indicators": [risk["title"] for risk in document["risk_indicators"]],
                "answer": question["data"]["answer"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
