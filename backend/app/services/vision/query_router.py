from __future__ import annotations

import re
from typing import ClassVar


class VisionQueryRouter:
    field_terms: ClassVar[set[str]] = {
        "address",
        "approval status",
        "batch size",
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
        "original request",
        "request date",
        "purpose",
        "organisation",
        "organization",
        "payment terms",
        "phone",
        "purchase order",
        "product type",
        "sample number",
        "sample specifications",
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
        tokens = set(re.findall(r"[a-z0-9]+", lower))
        if not lower or any(phrase in lower for phrase in self.unsafe_phrases):
            return "unsupported_question"
        if re.fullmatch(r"(?:hi|hello|hey|good (?:morning|afternoon|evening))[!. ]*", lower):
            return "greeting"
        if re.fullmatch(r"(?:help|what can you do|how can you help|show me what to ask)[?.! ]*", lower):
            return "help"
        if lower in {
            "yes",
            "yes.",
            "y",
            "no",
            "no.",
            "n",
            "check document quality",
            "check suspicious indicators",
            "check financial consistency",
            "check signature or stamp presence",
            "check approval status",
        }:
            return "clarification_response"
        if re.fullmatch(
            r"(?:why|why\?|show (?:me )?(?:the )?evidence|what page(?: is that on)?\??|"
            r"where is that(?: shown)?\??|explain that|source\??)[.! ]*",
            lower,
        ):
            return "conversation_follow_up"
        if lower in {
            "is this safe",
            "is this safe?",
            "is this okay",
            "is this ok",
            "anything wrong",
            "check this",
        }:
            return "ambiguous_question"
        if lower in {"is this document safe", "is this document safe?"}:
            return "safety_question"
        if any(
            term in lower
            for term in (
                "approval status",
                "approved",
                "ready for approval",
                "can this be approved",
            )
        ):
            return "approval_question"
        if (
            any(
                term in lower
                for term in (
                    "sum",
                    "average",
                    "how much total",
                    "calculate",
                    "difference in total",
                    "difference between invoice",
                    "invoice vs po",
                    "invoice versus po",
                    "percentage variance",
                    "percent variance",
                    "date interval",
                    "days between",
                )
            )
            or (
                bool(tokens & {"invoice", "invoices"})
                and ({"po"} <= tokens or "purchase order" in lower)
                and bool(
                    tokens
                    & {"difference", "delta", "differ", "different", "minus", "variance", "versus"}
                )
            )
            or (
                bool(tokens & {"invoice", "invoices"})
                and "total" in tokens
                and bool(tokens & {"all", "combined", "aggregate"})
            )
            or (
                any(term in lower for term in ("count", "how many", "number of"))
                and bool(tokens & {"invoice", "invoices"})
            )
        ):
            return "arithmetic_question"
        if any(term in lower for term in ("signed", "signature", "stamp", "seal", "checkbox", "logo", "handwritten")):
            return "visual_question"
        if any(term in lower for term in ("risk", "suspicious", "warning", "why was", "why is", "tamper", "anomaly")):
            return "risk_question"
        if any(
            term in lower
            for term in (
                "relationship",
                "linked",
                "connected",
                "shares account",
                "related to",
                "different from vendor master",
                "who requested",
                "made the original request",
                "who wrote the sample specifications",
                "specifications written by",
                "when was the request",
                "request made",
                "what was the purpose",
            )
        ):
            return "relationship_question"
        if any(term in lower for term in ("compare", "versus", "difference between", "same as")):
            return "comparison_question"
        if any(term in lower for term in self.field_terms) and re.search(
            r"\b(what|which|when|who|show|find|give|tell)\b",
            lower,
        ):
            return "structured_field_lookup"
        if any(term in lower for term in ("ocr", "read the line", "visible text", "what does the page say")):
            return "ocr_lookup"
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
