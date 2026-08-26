from app.legacy.retrieval import VectorIntelligenceLayer
from app.models.schemas import QueryCitation, QueryResponse


class RAGEngine:
    def __init__(self, retrieval: VectorIntelligenceLayer) -> None:
        self.retrieval = retrieval

    def answer(self, question: str, case_id: str | None = None, top_k: int = 5) -> QueryResponse:
        citations = self.retrieval.search(question, case_id=case_id, top_k=top_k)
        if not citations:
            return QueryResponse(
                answer="I could not find grounded evidence in the indexed document and finding evidence set.",
                citations=[],
                confidence=0.18,
                reasoning=[
                    "No document chunk or finding evidence matched the query.",
                    "Upload/analyze documents first, then retry with a specific question.",
                ],
            )

        answer = self._compose_grounded_answer(question, citations)
        confidence = min(0.95, 0.28 + sum(citation.confidence for citation in citations[:3]) / 3 * 0.68)
        return QueryResponse(
            answer=answer,
            citations=citations,
            confidence=round(confidence, 3),
            reasoning=[
                "Searched local hashed embeddings across document chunks and generated finding evidence.",
                "Applied keyword fallback and reranking for low-vector-score compliance/fraud terms.",
                "Composed the answer only from cited evidence.",
            ],
        )

    def _compose_grounded_answer(self, question: str, citations: list[QueryCitation]) -> str:
        lower_question = question.lower()
        if self._is_payment_query(lower_question):
            return self._compose_payment_answer(citations)

        if any(term in lower_question for term in ["exception", "review", "blocked", "fraud", "clause", "missing"]):
            items = self._exception_items(citations)
            if items:
                heading = "Exceptions requiring review:"
                if "blocked" in lower_question or "payment" in lower_question:
                    heading = "Payment is blocked because:"
                if "fraud" in lower_question:
                    heading = "Fraud signals detected:"
                if "clause" in lower_question or "missing" in lower_question:
                    heading = "Missing or risky clauses:"
                lines = [heading]
                for index, item in enumerate(items[:5], start=1):
                    lines.extend(
                        [
                            f"{index}. {item['title']} - {item['severity']}",
                            f"   Reason: {item['reason']}",
                            (
                                f"   Evidence: {item['source_document']}, page {item['page_number']}, "
                                f"chunk {item['chunk_id']}, confidence {item['confidence']:.0%}"
                            ),
                        ]
                    )
                return "\n".join(lines)

        lines = ["Grounded answer:"]
        for index, citation in enumerate(citations[:5], start=1):
            text = self._trim(citation.evidence_text, 220)
            lines.append(
                f"{index}. {text} "
                f"[{citation.source_document}, page {citation.page_number}, "
                f"chunk {citation.chunk_id}, confidence {citation.confidence:.0%}]"
            )
        return "\n".join(lines)

    def _is_payment_query(self, lower_question: str) -> bool:
        return any(
            term in lower_question
            for term in (
                "expected payout",
                "payout",
                "payment amount",
                "invoice amount",
                "amount due",
                "how much",
                "expected payment",
                "payment status",
                "payment due",
                "release payment",
                "pay vendor",
                "payable amount",
            )
        )

    def _compose_payment_answer(self, citations: list[QueryCitation]) -> str:
        combined = " ".join(citation.evidence_text for citation in citations)
        invoice_amount = self._extract_after(combined, "Amount Due:")
        po_amount = self._extract_after(combined, "Approved Amount:")
        payment_terms = self._extract_after(combined, "Payment Terms:")
        due_date = self._extract_after(combined, "Due Date:")
        blockers = []
        lower = combined.lower()
        if "bank account" in lower:
            blockers.append("bank account change")
        if "urgent" in lower:
            blockers.append("urgent wire request")
        if "manual override" in lower:
            blockers.append("manual override")
        if "indemnity" in lower:
            blockers.append("missing indemnity")
        lines = [
            "Expected payout:",
            f"- Invoice amount: {invoice_amount or 'Not found'}",
            f"- Purchase Order amount: {po_amount or 'Not found'}",
            "- Match status: Passed" if invoice_amount and po_amount and invoice_amount == po_amount else "- Match status: Review Required",
            f"- Payment terms: {payment_terms or 'Not found'}",
            f"- Due date: {due_date or 'Not found'}",
            "- Current payment status: Hold Payment" if blockers else "- Current payment status: Ready For Review",
            "- Approval gate: Blocked" if blockers else "- Approval gate: Open",
            "",
            "Reason:",
            f"- {'; '.join(blockers) if blockers else 'No blockers found in retrieved evidence.'}",
            "",
            "Evidence:",
        ]
        for index, citation in enumerate(citations[:5], start=1):
            lines.append(
                f"{index}. {citation.source_document}, page {citation.page_number}, "
                f"chunk {citation.chunk_id}, confidence {citation.confidence:.0%}"
            )
        lines.extend(
            [
                "",
                "Recommended Action:",
                "- Verify vendor bank details independently and resolve approval-blocking findings before releasing payment.",
            ]
        )
        return "\n".join(lines)

    def _extract_after(self, text: str, marker: str) -> str | None:
        lower = text.lower()
        index = lower.find(marker.lower())
        if index < 0:
            return None
        candidate = text[index + len(marker) :].strip()
        stop = candidate.find(" ")
        if marker.lower() in {"payment terms:", "due date:"}:
            parts = candidate.split()
            return " ".join(parts[:2]) if parts else None
        parts = candidate.split()
        if not parts:
            return None
        if parts[0].upper() in {"USD", "INR", "EUR", "GBP"} and len(parts) > 1:
            return f"{parts[0]} {parts[1].strip('.,')}"
        return candidate[:stop].strip(".,") if stop > 0 else candidate.strip(".,")

    def _exception_items(self, citations: list[QueryCitation]) -> list[dict]:
        items = []
        seen = set()
        for citation in citations:
            title, severity, reason = self._classify_citation(citation)
            if not title or title in seen:
                continue
            seen.add(title)
            items.append(
                {
                    "title": title,
                    "severity": severity,
                    "reason": reason,
                    "source_document": citation.source_document,
                    "page_number": citation.page_number,
                    "chunk_id": citation.chunk_id,
                    "confidence": citation.confidence,
                }
            )
        priority = {
            "Missing indemnity requirement": 0,
            "Manual override lacks approval evidence": 1,
            "Bank account change request": 2,
            "Urgent wire request language": 3,
        }
        return sorted(items, key=lambda item: priority.get(item["title"], 20))

    def _classify_citation(self, citation: QueryCitation) -> tuple[str | None, str, str]:
        text = citation.evidence_text.lower()
        linked = " ".join(citation.linked_findings).lower()
        combined = f"{linked} {text}"
        if "indemnity" in combined:
            return (
                "Missing indemnity requirement",
                "High severity",
                "Required indemnity language was not found.",
            )
        if "manual override" in combined:
            return (
                "Manual override lacks approval evidence",
                "High severity",
                "Manual override was applied without approval.",
            )
        if "bank account change" in combined or "bank_change" in combined or "new bank" in combined:
            return (
                "Bank account change request",
                "High fraud risk",
                "Payment instructions changed by email and require independent verification.",
            )
        if "urgent" in combined:
            return (
                "Urgent wire request language",
                "Medium fraud risk",
                "Urgent payment wording requires finance control review.",
            )
        if "duplicate invoice" in combined:
            return (
                "Possible duplicate invoice",
                "Critical fraud risk",
                "Duplicate invoice evidence requires payment block until review.",
            )
        if "amount does not match" in combined or "amount mismatch" in combined:
            return (
                "Invoice amount does not match purchase order",
                "High severity",
                "Invoice and purchase order amounts require reconciliation.",
            )
        return None, "Review required", self._trim(citation.evidence_text.strip(), 140)

    def _trim(self, value: str, limit: int) -> str:
        value = " ".join(value.split())
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."
