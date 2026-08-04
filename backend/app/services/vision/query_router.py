from __future__ import annotations

import re
from typing import ClassVar


class VisionQueryRouter:
    field_terms: ClassVar[set[str]] = {
        "vendor",
        "invoice number",
        "invoice date",
        "due date",
        "total",
        "tax",
        "currency",
        "purchase order",
        "bank account",
        "payment terms",
        "governing law",
    }

    def route(self, question: str) -> str:
        lower = " ".join(question.lower().split())
        if not lower or any(
            phrase in lower
            for phrase in (
                "ignore the evidence",
                "ignore previous instructions",
                "reveal chain of thought",
                "confirm this is fraud",
                "give a final legal judgement",
            )
        ):
            return "no_answer_or_unsafe"
        if any(term in lower for term in ("sum", "average", "how much total", "calculate", "difference")):
            return "arithmetic_question"
        if any(term in lower for term in ("signed", "signature", "stamp", "seal", "checkbox", "logo", "handwritten")):
            return "visual_question"
        if any(term in lower for term in ("relationship", "linked", "connected", "shares account", "different from vendor master")):
            return "relationship_question"
        if any(term in lower for term in ("compare", "versus", "difference between", "same as")):
            return "comparison_question"
        if any(term in lower for term in self.field_terms) and re.search(r"\b(what|which|when|who|show|find)\b", lower):
            return "structured_lookup"
        return "semantic_question"
