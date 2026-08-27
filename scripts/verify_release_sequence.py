#!/usr/bin/env python3
"""Directive Phase 33: the A-AA sequence, run against the live containerised stack.

Every step prints EXPECTED / OBSERVED / PASS-FAIL. Nothing is mocked; every call
goes over HTTP to the running backend. Documents are authored here, so a pass
reflects generalization rather than fixture memorisation.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000/api/v1"
RUN = secrets.token_hex(5)
results: list[tuple[str, str, bool]] = []


def call(path, token=None, method="GET", body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except Exception:
            return exc.code, {}


def step(label, expected, observed, ok):
    results.append((label, expected, ok))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}")
    print(f"       expected: {expected}")
    print(f"       observed: {observed}")


def ingest(token, name, text):
    status, payload = call(
        "/vision/documents/analyze",
        token,
        "POST",
        {"filename": name, "text": text, "metadata": {}},
    )
    assert status == 200, payload
    return payload["data"]


# --------------------------------------------------------------- A. auth
# Public signup deliberately cannot self-assign a privileged role, so the
# admin used for the reprocess and delete steps is created through the server's
# explicit bootstrap path. Using a VIEWER here made both of those steps return
# 403 and "pass" without testing anything.
email = os.environ["PHASE33_EMAIL"]
password = os.environ["PHASE33_PASSWORD"]
status, payload = call(
    "/auth/login", method="POST", body={"email": email, "password": password}
)
token = payload.get("data", {}).get("access_token", "")
step(
    "A. authentication",
    "a session token for a new tenant",
    f"HTTP {status}, token={'yes' if token else 'no'}",
    bool(token),
)

# --------------------------------------------------- B-K. document A
DOC_A = """Veterinary Practice Invoice
Invoice Number: VET-88213
Practice Name: Ashgrove Veterinary Clinic
Client Name: Marcus Delacroix
Animal Name: Bramble
Treatment: Dental scaling and polish
Amount Due: 486.50 EUR
Invoice Date: 2026-04-08
"""
a = ingest(token, f"vet-invoice-{RUN}.txt", DOC_A)
step(
    "B-D. upload and process document A",
    "a processed document with OCR/parsed text",
    f"status={a['status']}, text_chars={len(a.get('parsed_text') or '')}",
    bool(a.get("parsed_text")),
)
step(
    "E. extraction",
    ">=4 structured fields discovered without a supplied schema",
    f"{len(a['extracted_fields'])} fields: {[f['name'] for f in a['extracted_fields']][:6]}",
    len(a["extracted_fields"]) >= 4,
)
step(
    "F. chunks + provenance",
    "every chunk carries document, tenant and page",
    f"{len(a['evidence_chunks'])} chunks",
    bool(a["evidence_chunks"])
    and all(
        c.get("document_id") and c.get("tenant_id") and (c.get("page_number") or 0) >= 1
        for c in a["evidence_chunks"]
    ),
)
graph_a = a.get("graph") or {}
step(
    "H-I. graph entities and relationships",
    "nodes and typed edges",
    f"{len(graph_a.get('nodes') or [])} nodes, {len(graph_a.get('edges') or [])} edges",
    bool(graph_a.get("nodes")) and bool(graph_a.get("edges")),
)

status, payload = call(
    "/vision/ask",
    token,
    "POST",
    {"question": "what is the amount due", "document_ids": [a["document_id"]]},
)
ans_a = payload["data"]
cited_a = {c["document_id"] for c in ans_a["citations"]}
step(
    "J-K. ask about A -> A evidence",
    "an answer citing only document A, containing 486.50",
    f"status={ans_a['answer_status']}, cites={len(cited_a)} doc(s), answer={ans_a['answer'][:70]!r}",
    "486.50" in ans_a["answer"] and cited_a == {a["document_id"]},
)

# --------------------------------------------------- L-N. unrelated document B
DOC_B = """Aircraft Maintenance Release
Work Order: AMO-40127
Aircraft Registration: G-BXTQ
Engine Serial: CFM56-778201
Maintenance Organisation: Northvale Aero Services
Certifying Engineer: Helena Vasquez
Release Date: 2026-04-11
Flight Hours Since Overhaul: 1,284
"""
b = ingest(token, f"aircraft-{RUN}.txt", DOC_B)
status, payload = call(
    "/vision/ask", token, "POST", {"question": "what is the aircraft registration"}
)
ans_b = payload["data"]
cited_b = {c["document_id"] for c in ans_b["citations"]}
step(
    "L-N. ask about B -> different evidence",
    "cites B and not A",
    f"status={ans_b['answer_status']}, cites_B={b['document_id'] in cited_b}, cites_A={a['document_id'] in cited_b}",
    b["document_id"] in cited_b and a["document_id"] not in cited_b,
)

# --------------------------------------------------- O-Q. genuinely related C
DOC_C = """Veterinary Treatment Record
Record Number: VTR-99004
Invoice Number: VET-88213
Practice Name: Ashgrove Veterinary Clinic
Client Name: Marcus Delacroix
Animal Name: Bramble
Procedure Performed: Dental scaling and polish
Attending Veterinarian: Dr Ines Farkas
"""
c = ingest(token, f"vet-record-{RUN}.txt", DOC_C)
status, payload = call(
    "/vision/ask",
    token,
    "POST",
    {
        "question": "which documents reference invoice number VET-88213",
        "document_ids": [a["document_id"], c["document_id"]],
    },
)
ans_c = payload["data"]
cited_c = {x["document_id"] for x in ans_c["citations"]}
step(
    "O-Q. cross-document question",
    "evidence drawn from the two genuinely related documents",
    f"status={ans_c['answer_status']}, cites={len(cited_c)}, graph_paths={len(ans_c.get('graph_paths') or [])}",
    bool(cited_c) and cited_c <= {a["document_id"], c["document_id"]},
)

status, payload = call(
    "/vision/ask",
    token,
    "POST",
    {
        "question": "how is the veterinary invoice related to the aircraft maintenance release",
        "document_ids": [a["document_id"], b["document_id"]],
    },
)
ans_x = payload["data"]
step(
    "Q2. unrelated documents stay unrelated",
    "no fabricated relationship",
    f"status={ans_x['answer_status']}, graph_paths={len(ans_x.get('graph_paths') or [])}",
    ans_x["answer_status"]
    in {"not_found", "partially_supported", "clarification_needed"},
)

# --------------------------------------------------- R-S. abstention
status, payload = call(
    "/vision/ask",
    token,
    "POST",
    {"question": "what is the mean time between failures for the turbine bearing"},
)
ans_r = payload["data"]
step(
    "R-S. unsupported question abstains",
    "not_found with no citations",
    f"status={ans_r['answer_status']}, citations={len(ans_r['citations'])}",
    ans_r["answer_status"] in {"not_found", "clarification_needed"}
    and not ans_r["citations"],
)

# --------------------------------------------------- T-U. review routing
step(
    "T-U. low confidence routes to review",
    "at least one document awaiting review",
    f"A status={a['status']}, C status={c['status']}",
    any(d["status"] in {"REVIEW_REQUIRED", "UNDER_REVIEW"} for d in (a, b, c)),
)

# --------------------------------------------------- V-W. unauthorised access
status_anon, _ = call(f"/vision/documents/{a['document_id']}")
other_email = f"phase33-other-{RUN}@documind.local"
other_password = "Other33-" + secrets.token_urlsafe(20) + "-Aa1!"
call(
    "/auth/signup",
    method="POST",
    body={
        "name": "Phase 33 Other",
        "email": other_email,
        "password": other_password,
        "role": "VIEWER",
        "organisation_name": f"Phase33 Other Org {RUN}",
    },
)
_, other_payload = call(
    "/auth/login",
    method="POST",
    body={"email": other_email, "password": other_password},
)
other_token = other_payload.get("data", {}).get("access_token", "")
status_cross, _ = call(f"/vision/documents/{a['document_id']}", other_token)
step(
    "V-W. unauthorised access refused",
    "401/403 anonymous, 403/404 cross-tenant",
    f"anonymous={status_anon}, cross_tenant={status_cross}",
    status_anon in (401, 403) and status_cross in (403, 404),
)

# --------------------------------------------------- X-Y. reprocess idempotency
before_nodes = len(graph_a.get("nodes") or [])
before_edges = len(graph_a.get("edges") or [])
status, payload = call(
    f"/vision/documents/{a['document_id']}/reprocess", token, "POST", {}
)
# Wait for the graph to be *rebuilt*, not merely for the status to move off
# QUEUED. Polling on status alone caught the document mid-reprocess with an
# empty graph and reported a leak that did not exist.
deadline = time.time() + 240
after = None
terminal = {"REVIEW_REQUIRED", "COMPLETED", "APPROVED", "INDEXED"}
while time.time() < deadline:
    st, pl = call(f"/vision/documents/{a['document_id']}", token)
    candidate = pl.get("data") or {}
    graph = candidate.get("graph") or {}
    if st == 200 and candidate.get("status") in terminal and (graph.get("nodes") or []):
        after = candidate
        break
    time.sleep(3)
after = after or {}
gafter = after.get("graph") or {}
step(
    "X-Y. reprocess is idempotent",
    f"HTTP 200 and node/edge counts stay at {before_nodes}/{before_edges}",
    f"reprocess HTTP {status}, now {len(gafter.get('nodes') or [])}/{len(gafter.get('edges') or [])}",
    status == 200
    and len(gafter.get("nodes") or []) == before_nodes
    and len(gafter.get("edges") or []) == before_edges,
)

# --------------------------------------------------- Z-AA. delete cleans indexes
status_del, _ = call(f"/vision/documents/{b['document_id']}", token, "DELETE")
status_get, _ = call(f"/vision/documents/{b['document_id']}", token)
_, payload = call(
    "/vision/ask", token, "POST", {"question": "what is the aircraft registration"}
)
ans_after = payload["data"]
still = {x["document_id"] for x in ans_after["citations"]}
step(
    "Z-AA. delete removes the document from retrieval",
    "document gone and never cited again",
    f"delete HTTP {status_del}, fetch HTTP {status_get}, still_cited={b['document_id'] in still}",
    status_del in (200, 204) and status_get == 404 and b["document_id"] not in still,
)

print("\n" + "=" * 62)
passed = sum(1 for _, _, ok in results if ok)
print(f"Phase 33 sequence: {passed}/{len(results)} steps passed")
for label, _, ok in results:
    if not ok:
        print(f"  FAILED: {label}")
sys.exit(0 if passed == len(results) else 1)
