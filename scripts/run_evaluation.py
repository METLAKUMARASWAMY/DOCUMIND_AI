#!/usr/bin/env python3
"""Measure DocuMind AI on a labelled evaluation set and report only what was measured.

Directive §44. The rule this script enforces on itself: **every number printed is
computed from an actual run in this process**. Anything that could not be measured
is reported as ``not_measured`` with a reason rather than being estimated,
defaulted to zero, or omitted.

Usage
-----
    python scripts/run_evaluation.py                     # human-readable report
    python scripts/run_evaluation.py --json out.json     # machine-readable
    python scripts/run_evaluation.py --corpus my.json    # your own labelled set

The bundled corpus is authored here rather than drawn from ``var/test_fixtures``
or ``dataset/``, so a passing score reflects generalization rather than the
pipeline having been tuned to these exact files (§57).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models.schemas import TextAnalysisRequest  # noqa: E402
from app.models.vision import VisionAskRequest  # noqa: E402
from app.services.vision.service import VisionAIService  # noqa: E402

TENANT = "tenant-evaluation-harness"

# --------------------------------------------------------------------------- corpus

#: Documents with ground-truth field values, plus questions whose correct
#: behaviour is known. ``expected_answer=None`` means the corpus cannot support
#: the question and the system is expected to abstain.
DEFAULT_CORPUS: dict[str, Any] = {
    "documents": [
        {
            "id": "eval-invoice-1",
            "filename": "eval-invoice-1.txt",
            "text": (
                "Invoice INV-EV-7001\n"
                "Vendor: Halden Precision Works\n"
                "Invoice Date: 2026-03-14\n"
                "Due Date: 2026-04-13\n"
                "PO Number: PO-EV-2210\n"
                "Payment Terms: Net 30\n"
                "Total Amount: GBP 42,150.00\n"
                "Halden supplied hardened tooling for the press line.\n"
            ),
            "fields": {
                "invoice_number": "INV-EV-7001",
                "vendor_name": "Halden Precision Works",
                "invoice_date": "2026-03-14",
                "total_amount": "GBP 42,150.00",
                "payment_terms": "Net 30",
            },
        },
        {
            "id": "eval-po-1",
            "filename": "eval-po-1.txt",
            "text": (
                "Purchase Order: PO-EV-2210\n"
                "Supplier: Halden Precision Works\n"
                "Buyer: Ravensworth Manufacturing\n"
                "Effective Date: 2026-03-01\n"
                "Payment Terms: Net 30\n"
                "Total Amount: GBP 42,150.00\n"
                "Approved By: T. Mbeki\n"
            ),
            "fields": {
                "purchase_order_number": "PO-EV-2210",
                "total_amount": "GBP 42,150.00",
                "payment_terms": "Net 30",
            },
        },
        {
            "id": "eval-statement-1",
            "filename": "eval-statement-1.txt",
            "text": (
                "Bank Account Confirmation\n"
                "Vendor: Halden Precision Works\n"
                "Bank Name: Lloyds\n"
                "Account Number: ACCT-4471902\n"
                "Approval Date: 2026-03-05\n"
                "Approved By: T. Mbeki\n"
            ),
            "fields": {
                "account_number": "ACCT-4471902",
            },
        },
    ],
    "questions": [
        {
            "question": "what is the total amount",
            "expected_answer": "42,150",
            "expected_documents": ["eval-invoice-1", "eval-po-1"],
        },
        {
            "question": "what are the payment terms",
            "expected_answer": "Net 30",
            "expected_documents": ["eval-invoice-1", "eval-po-1"],
        },
        {
            "question": "what is the invoice number",
            "expected_answer": "INV-EV-7001",
            "expected_documents": ["eval-invoice-1"],
        },
        {
            "question": "which vendor issued this",
            "expected_answer": "Halden",
            "expected_documents": ["eval-invoice-1"],
        },
        {
            # Bank identifiers are deliberately masked before they reach an
            # answer (directive §33), so the expectation is the visible suffix,
            # not the raw value. An evaluation that demanded the full number
            # would be scoring the system for leaking it.
            "question": "what is the bank account number",
            "expected_answer": "1902",
            "expected_documents": ["eval-statement-1"],
        },
        # Correct behaviour is abstention: nothing in the corpus supports these.
        {"question": "what is the mean time between failures", "expected_answer": None},
        {"question": "what was the share price on the invoice date", "expected_answer": None},
        {"question": "what is the weather in Leeds", "expected_answer": None},
    ],
}


@dataclass
class Outcome:
    """One measured observation, or an explicit reason it could not be measured."""

    name: str
    measured: bool
    value: Any = None
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        if not self.measured:
            return {"status": "not_measured", "reason": self.reason}
        return {"status": "measured", "value": self.value, **self.detail}


def _percent(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


#: Below this many observations a proportion says very little on its own, and
#: the report says so rather than letting "100.0" stand unqualified.
SMALL_SAMPLE_THRESHOLD = 30


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion, as percentages.

    A bare "100%" on five questions reads like a claim about the system. The
    interval is what makes the sample size impossible to overlook: 5/5 is
    100% with a 95% interval of roughly 56%-100%, which is an honest summary
    of what five questions can establish.

    Wilson rather than the normal approximation because the normal one gives a
    zero-width interval at 0% and 100% — exactly the cases here.
    """
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    spread = (
        z
        * ((proportion * (1 - proportion) / total + z**2 / (4 * total**2)) ** 0.5)
        / denominator
    )
    low = max(0.0, centre - spread) * 100
    high = min(1.0, centre + spread) * 100
    return (round(low, 1), round(high, 1))


def run_evaluation(corpus: dict[str, Any]) -> dict[str, Any]:
    service = VisionAIService()
    documents = corpus["documents"]
    questions = corpus["questions"]

    # ------------------------------------------------------------- ingestion
    ingest_latencies: list[float] = []
    ingested: dict[str, Any] = {}
    ingest_failures: list[dict[str, str]] = []
    for spec in documents:
        started = time.perf_counter()
        try:
            record = service.process_text(
                TextAnalysisRequest(
                    filename=spec["filename"], text=spec["text"], metadata={}
                ),
                actor="evaluation-harness",
                tenant_id=TENANT,
            )
        except Exception as exc:
            ingest_failures.append({"id": spec["id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        ingest_latencies.append((time.perf_counter() - started) * 1000.0)
        ingested[spec["id"]] = record

    # ------------------------------------------------- 1. extraction accuracy
    expected_total = 0
    correct = 0
    missing: list[str] = []
    wrong: list[dict[str, str]] = []
    for spec in documents:
        record = ingested.get(spec["id"])
        if record is None:
            continue
        actual = {item.name: item.value for item in record.extracted_fields}
        for name, expected in spec["fields"].items():
            expected_total += 1
            observed = actual.get(name)
            if observed is None:
                missing.append(f"{spec['id']}.{name}")
            elif _normalise(expected) in _normalise(observed):
                correct += 1
            else:
                wrong.append({"field": f"{spec['id']}.{name}", "expected": expected, "observed": observed})

    extraction = Outcome(
        "extraction_accuracy",
        measured=expected_total > 0,
        value=_percent(correct, expected_total),
        reason="" if expected_total else "no labelled fields in corpus",
        detail={
            "correct": correct,
            "total": expected_total,
            "expected": expected_total,
            "missing": missing,
            "incorrect": wrong,
        },
    )

    # -------------------------------- 2-6. answering, citations, abstention
    answerable = [item for item in questions if item.get("expected_answer") is not None]
    unanswerable = [item for item in questions if item.get("expected_answer") is None]

    answer_correct = 0
    citation_correct = 0
    citation_evaluated = 0
    grounded = 0
    overclaimed: list[dict[str, Any]] = []
    answer_latencies: list[float] = []
    per_question: list[dict[str, Any]] = []

    for item in answerable:
        started = time.perf_counter()
        response = service.ask(VisionAskRequest(question=item["question"], top_k=10), tenant_id=TENANT)
        answer_latencies.append((time.perf_counter() - started) * 1000.0)

        hit = _normalise(item["expected_answer"]) in _normalise(response.answer)
        answer_correct += int(hit)

        # A claimed answer must carry evidence.
        if response.answer_status in {"supported", "partially_supported"}:
            if response.citations:
                grounded += 1
            else:
                overclaimed.append(
                    {"question": item["question"], "reason": "claimed support with no citation"}
                )

        expected_documents = item.get("expected_documents")
        if expected_documents:
            citation_evaluated += 1
            cited_files = {citation.filename for citation in response.citations}
            expected_files = {
                spec["filename"]
                for spec in documents
                if spec["id"] in expected_documents
            }
            if cited_files & expected_files:
                citation_correct += 1

        per_question.append(
            {
                "question": item["question"],
                "answer_status": response.answer_status,
                "correct": hit,
                "citations": len(response.citations),
                "retriever": response.retriever,
            }
        )

    abstention_correct = 0
    for item in unanswerable:
        started = time.perf_counter()
        response = service.ask(VisionAskRequest(question=item["question"], top_k=10), tenant_id=TENANT)
        answer_latencies.append((time.perf_counter() - started) * 1000.0)
        # Correct behaviour is not to claim full support for a question the
        # corpus cannot answer.
        abstained = response.answer_status != "supported"
        abstention_correct += int(abstained)
        if not abstained:
            overclaimed.append(
                {"question": item["question"], "reason": f"answered {response.answer_status}"}
            )
        per_question.append(
            {
                "question": item["question"],
                "answer_status": response.answer_status,
                "correct": abstained,
                "citations": len(response.citations),
                "retriever": response.retriever,
            }
        )

    retrieval_status = service.retrieval_index.status()

    results = {
        "corpus": {
            "documents": len(documents),
            "ingested": len(ingested),
            "ingest_failures": ingest_failures,
            "answerable_questions": len(answerable),
            "unanswerable_questions": len(unanswerable),
        },
        "metrics": {
            "extraction_accuracy_pct": extraction.as_dict(),
            "answer_accuracy_pct": Outcome(
                "answer_accuracy",
                measured=bool(answerable),
                value=_percent(answer_correct, len(answerable)),
                reason="" if answerable else "no answerable questions in corpus",
                detail={"correct": answer_correct, "total": len(answerable)},
            ).as_dict(),
            "citation_correctness_pct": Outcome(
                "citation_correctness",
                measured=citation_evaluated > 0,
                value=_percent(citation_correct, citation_evaluated),
                reason="" if citation_evaluated else "no questions declared expected_documents",
                detail={"correct": citation_correct, "total": citation_evaluated},
            ).as_dict(),
            "grounding_fidelity_pct": Outcome(
                "grounding_fidelity",
                measured=bool(answerable),
                value=_percent(grounded, len(answerable)),
                reason="" if answerable else "no answerable questions",
                detail={
                    "correct": grounded,
                    "claims_with_evidence": grounded,
                    "total": len(answerable),
                    "note": "share of claimed answers that carried at least one citation",
                },
            ).as_dict(),
            "abstention_correctness_pct": Outcome(
                "abstention_correctness",
                measured=bool(unanswerable),
                value=_percent(abstention_correct, len(unanswerable)),
                reason="" if unanswerable else "no unanswerable questions in corpus",
                detail={"correct": abstention_correct, "total": len(unanswerable)},
            ).as_dict(),
            "overclaims": Outcome(
                "overclaims",
                measured=True,
                value=len(overclaimed),
                detail={"instances": overclaimed, "note": "lower is better; 0 is required"},
            ).as_dict(),
            "ingest_latency_ms": _latency_outcome(ingest_latencies, "no document was ingested"),
            "answer_latency_ms": _latency_outcome(answer_latencies, "no question was asked"),
            "failure_rate_pct": Outcome(
                "failure_rate",
                measured=bool(documents),
                value=_percent(len(ingest_failures), len(documents)),
                detail={"failures": len(ingest_failures), "total": len(documents)},
            ).as_dict(),
        },
        "runtime": {
            "vector_backend": retrieval_status.get("vector_backend"),
            "degraded": retrieval_status.get("degraded"),
            "embedding_adapter": retrieval_status.get("embedding", {}).get("adapter_id"),
            "reranker_status": retrieval_status.get("reranker", {}).get("status"),
            "qdrant_status": retrieval_status.get("qdrant", {}).get("status"),
            "indexed_chunks": retrieval_status.get("chunks"),
        },
        "per_question": per_question,
    }
    return results


def _latency_outcome(samples: list[float], empty_reason: str) -> dict[str, Any]:
    if not samples:
        return Outcome("latency", measured=False, reason=empty_reason).as_dict()
    ordered = sorted(samples)
    return Outcome(
        "latency",
        measured=True,
        value=round(statistics.mean(ordered), 2),
        detail={
            "median": round(statistics.median(ordered), 2),
            "p95": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 2),
            "min": round(ordered[0], 2),
            "max": round(ordered[-1], 2),
            "samples": len(ordered),
        },
    ).as_dict()


def _normalise(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def render(results: dict[str, Any]) -> str:
    lines = ["DocuMind AI — measured evaluation", "=" * 42, ""]
    corpus = results["corpus"]
    questions = corpus["answerable_questions"] + corpus["unanswerable_questions"]

    # The sample size goes first and in its own block. A reader who takes one
    # thing from this report should take away how much it was measured on.
    lines.append("SAMPLE SIZE")
    lines.append(
        f"  {corpus['ingested']}/{corpus['documents']} documents ingested, "
        f"{questions} questions "
        f"({corpus['answerable_questions']} answerable, "
        f"{corpus['unanswerable_questions']} unanswerable)"
    )
    if questions < SMALL_SAMPLE_THRESHOLD:
        lines.append(
            f"  This is a small controlled corpus (n={questions}). Percentages below"
        )
        lines.append(
            "  carry a 95% Wilson interval; read the interval, not the point estimate."
        )
        lines.append(
            "  A score here is evidence the pipeline works end to end on unseen"
        )
        lines.append(
            "  documents. It is not a measurement of accuracy in production."
        )
    if corpus["ingest_failures"]:
        lines.append(f"  ingest failures: {corpus['ingest_failures']}")
    lines.append("")

    for name, payload in results["metrics"].items():
        if payload["status"] != "measured":
            lines.append(f"  {name:32s} not measured ({payload['reason']})")
            continue
        value = payload["value"]
        correct = payload.get("correct")
        total = payload.get("total")
        if name.endswith("_pct") and isinstance(correct, int) and isinstance(total, int) and total:
            low, high = wilson_interval(correct, total)
            lines.append(
                f"  {name:32s} {value}%  (n={total}, {correct}/{total}, 95% CI {low}-{high}%)"
            )
        elif name.endswith("_pct") and isinstance(total, int) and total:
            lines.append(f"  {name:32s} {value}%  (n={total})")
        else:
            lines.append(f"  {name:32s} {value}")
        for key in ("median", "p95", "failures"):
            if key in payload:
                lines.append(f"      {key}: {payload[key]}")
        if payload.get("missing"):
            lines.append(f"      missing fields: {payload['missing']}")
        if payload.get("incorrect"):
            for item in payload["incorrect"]:
                lines.append(
                    f"      wrong: {item['field']} expected={item['expected']!r} "
                    f"observed={item['observed']!r}"
                )
        if payload.get("instances"):
            for item in payload["instances"]:
                lines.append(f"      overclaim: {item['question']!r} -> {item['reason']}")

    lines.append("")
    lines.append("runtime actually used:")
    for key, value in results["runtime"].items():
        lines.append(f"  {key:24s} {value}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="JSON corpus; defaults to the bundled set")
    parser.add_argument("--json", type=Path, help="write machine-readable results here")
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text()) if args.corpus else DEFAULT_CORPUS
    results = run_evaluation(corpus)

    print(render(results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")

    # A non-zero exit makes this usable as a gate: overclaiming is a hard failure.
    overclaims = results["metrics"]["overclaims"]
    return 1 if overclaims.get("value") else 0


if __name__ == "__main__":
    raise SystemExit(main())
