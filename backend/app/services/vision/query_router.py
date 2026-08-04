from __future__ import annotations

import re
from typing import ClassVar


class VisionQueryRouter:
    field_terms: ClassVar[set[str]] = {
        "address",
        "approval status",
        "bank account",
        "bank code",
        "branch",
        "contact",
        "currency",
        "due date",
        "effective date",
        "email",
        "expiry date",
        "fax",
        "form title",
        "governing law",
        "invoice date",
        "invoice number",
        "organisation",
        "organization",
        "payment terms",
        "phone",
        "purchase order",
        "tax",
        "total",
        "vendor",
    }
    unsafe_phrases: ClassVar[tuple[str, ...]] = (
        "ignore the evidence",
        "ignore previous instructions",
        "reveal chain of thought",
        "confirm this is fraud",
        "give a final legal judgement",
    )

    def route(self, question: str) -> str:
        lower = " ".join(question.lower().split())
        if not lower or any(phrase in lower for phrase in self.unsafe_phrases):
            return "unsupported_question"
        if re.fullmatch(r"(?:hi|hello|hey|good (?:morning|afternoon|evening))[!. ]*", lower):
            return "greeting"
        if re.fullmatch(r"(?:help|what can you do|how can you help|show me what to ask)[?.! ]*", lower):
            return "help"
        if lower in {
            "is this safe",
            "is this safe?",
            "is this okay",
            "is this ok",
            "anything wrong",
            "check this",
        }:
            return "ambiguous_question"
        if any(term in lower for term in ("sum", "average", "how much total", "calculate", "difference in total")):
            return "arithmetic_question"
        if any(term in lower for term in ("signed", "signature", "stamp", "seal", "checkbox", "logo", "handwritten")):
            return "visual_question"
        if any(term in lower for term in ("risk", "suspicious", "warning", "why was", "why is", "tamper", "anomaly")):
            return "risk_question"
        if any(term in lower for term in ("relationship", "linked", "connected", "shares account", "related to", "different from vendor master")):
            return "relationship_question"
        if any(term in lower for term in ("compare", "versus", "difference between", "same as")):
            return "comparison_question"
        if any(term in lower for term in self.field_terms) and re.search(
            r"\b(what|which|when|who|show|find|give|tell)\b",
            lower,
        ):
            return "structured_lookup"
        if self._looks_unsupported(lower):
            return "unsupported_question"
        return "semantic_question"

    def _looks_unsupported(self, question: str) -> bool:
        return any(
            phrase in question
            for phrase in (
                "weather",
                "sports score",
                "write a poem",
                "tell me a joke",
                "stock price",
                "capital of",
            )
        )
