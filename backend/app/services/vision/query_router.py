from __future__ import annotations

import re
from functools import lru_cache
from typing import ClassVar


def _boundary_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    """Compile terms so they match whole words, never substrings.

    Substring matching was a real defect: ``"sum"`` matched *summarise*,
    *summary*, *consume* and *assumption*, routing plain summarisation requests
    into the arithmetic handler. Every vocabulary check in this router goes
    through a boundary-anchored pattern instead.
    """

    if not terms:
        return re.compile(r"(?!x)x")  # never matches
    alternatives = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])")


class VisionQueryRouter:
    """Routes a natural-language question to an evidence handler.

    The vocabulary below is a *routing hint*, not an answer. Anything it does not
    recognise falls through to ``semantic_question``, and ``VisionAIService.ask``
    cascades to the structured/graph/OCR handlers when a route returns no match,
    so an unseen field name is still answerable. Keeping this list honest matters
    only for choosing the cheapest correct handler first.
    """

    #: Domain-independent field vocabulary. Document-specific labels are NOT
    #: hardcoded here — they are discovered from the schema registry at runtime
    #: by :meth:`_registry_field_terms`, so a newly configured document schema
    #: routes correctly without editing this module.
    base_field_terms: ClassVar[tuple[str, ...]] = (
        "account number",
        "address",
        "amount",
        "approval status",
        "bank account",
        "bank code",
        "branch",
        "contact",
        "currency",
        "customer",
        "due date",
        "effective date",
        "email",
        "expiry date",
        "fax",
        "governing law",
        "invoice date",
        "invoice number",
        "issued by",
        "organisation",
        "organization",
        "payment terms",
        "phone",
        "purchase order",
        "quantity",
        "reference number",
        "subtotal",
        "supplier",
        "tax",
        "title",
        "total",
        "vendor",
    )

    unsafe_phrases: ClassVar[tuple[str, ...]] = (
        "ignore the evidence",
        "ignore previous instructions",
        "reveal chain of thought",
        "confirm this is fraud",
        "give a final legal judgement",
    )

    arithmetic_terms: ClassVar[tuple[str, ...]] = (
        "sum",
        "sum of",
        "total of",
        "average",
        "mean of",
        "how much total",
        "calculate",
        "add up",
        "difference in total",
        "difference between invoice",
        "invoice vs po",
        "invoice versus po",
        "percentage variance",
        "percent variance",
        "date interval",
        "days between",
    )

    visual_terms: ClassVar[tuple[str, ...]] = (
        "signed",
        "signature",
        "stamp",
        "seal",
        "checkbox",
        "logo",
        "handwritten",
    )

    risk_terms: ClassVar[tuple[str, ...]] = (
        "risk",
        "suspicious",
        "warning",
        "why was",
        "why is",
        "tamper",
        "tampered",
        "anomaly",
        "anomalies",
    )

    relationship_terms: ClassVar[tuple[str, ...]] = (
        "relationship",
        "related to",
        "linked",
        "linked to",
        "connected",
        "connected to",
        "shares account",
        "shared account",
        "same vendor",
        "same customer",
        "different from vendor master",
        "who requested",
        "requested by",
        "written by",
        "issued by",
        "refers to",
        "refer to",
        "associated with",
    )

    comparison_terms: ClassVar[tuple[str, ...]] = (
        "compare",
        "comparison",
        "versus",
        "vs",
        "difference between",
        "same as",
        "agree with",
        "match",
        "consistent with",
        "conflicting",
        "contradict",
    )

    approval_terms: ClassVar[tuple[str, ...]] = (
        "approval status",
        "approved",
        "ready for approval",
        "can this be approved",
        "approval",
    )

    ocr_terms: ClassVar[tuple[str, ...]] = (
        "ocr",
        "read the line",
        "visible text",
        "what does the page say",
        "raw text",
    )

    out_of_scope_terms: ClassVar[tuple[str, ...]] = (
        "weather",
        "sports score",
        "write a poem",
        "tell me a joke",
        "stock price",
        "capital of",
    )

    interrogatives: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(what|which|when|who|where|show|find|give|tell|list)\b"
    )

    def route(self, question: str) -> str:
        lower = " ".join(question.lower().split())
        if not lower or any(phrase in lower for phrase in self.unsafe_phrases):
            return "unsupported_question"
        if re.fullmatch(r"(?:hi|hello|hey|good (?:morning|afternoon|evening))[!. ]*", lower):
            return "greeting"
        if re.fullmatch(
            r"(?:help|what can you do|how can you help|show me what to ask)[?.! ]*", lower
        ):
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

        tokens = set(re.findall(r"[a-z0-9]+", lower))

        if self._matches(lower, self.approval_terms):
            return "approval_question"
        if self._is_arithmetic(lower, tokens):
            return "arithmetic_question"
        if self._matches(lower, self.visual_terms):
            return "visual_question"
        if self._matches(lower, self.risk_terms):
            return "risk_question"
        if self._matches(lower, self.relationship_terms):
            return "relationship_question"
        if self._matches(lower, self.comparison_terms):
            return "comparison_question"
        if self._matches(lower, self.field_terms()) and self.interrogatives.search(lower):
            return "structured_field_lookup"
        if self._matches(lower, self.ocr_terms):
            return "ocr_lookup"
        if self._matches(lower, self.out_of_scope_terms):
            return "unsupported_question"
        # Last resort before falling back to raw semantic chunks: a question
        # asking *who* wants an entity related to the document, which is what
        # the graph models. `relationship_terms` above carries literal phrases
        # like "who requested", so "who requested this document" reached the
        # graph while "Who made the original request?" did not — and the
        # semantic path answered it from raw OCR text, claiming support for
        # "original Request Made By___Dr. A. W. Spears on__September original
        # Laboratory oGs Original Request Analysis to Dr, oGs Made Av By___..."
        # while the extracted field held exactly "Dr. A. W. Spears".
        #
        # This only reclassifies questions that would otherwise be
        # `semantic_question`, and a relationship question with no matching
        # edge cascades to structured lookup, so nothing that worked stops
        # working.
        if self._asks_for_an_agent(lower):
            return "relationship_question"
        return "semantic_question"

    # ------------------------------------------------------------------ helpers

    def _matches(self, lower: str, terms: tuple[str, ...]) -> bool:
        return bool(_boundary_pattern(terms).search(lower))

    #: Word-bounded so "whose" and "whom" count but "whole" does not.
    _AGENT_INTERROGATIVE: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![\w-])(?:who|whom|whose)(?![\w-])"
    )

    def _asks_for_an_agent(self, lower: str) -> bool:
        """Whether the question asks which entity did or holds something."""
        return bool(self._AGENT_INTERROGATIVE.search(lower))

    def _is_arithmetic(self, lower: str, tokens: set[str]) -> bool:
        if self._matches(lower, self.arithmetic_terms):
            return True
        invoice = bool(tokens & {"invoice", "invoices"})
        if (
            invoice
            and ({"po"} <= tokens or "purchase order" in lower)
            and bool(
                tokens
                & {"difference", "delta", "differ", "different", "minus", "variance", "versus"}
            )
        ):
            return True
        if invoice and "total" in tokens and bool(tokens & {"all", "combined", "aggregate"}):
            return True
        if self._matches(lower, ("count", "how many", "number of")) and invoice:
            return True
        return False

    def field_terms(self) -> tuple[str, ...]:
        """Domain vocabulary: static basics plus whatever schemas are configured."""

        return self.base_field_terms + self._registry_field_terms()

    @staticmethod
    @lru_cache(maxsize=1)
    def _registry_field_terms() -> tuple[str, ...]:
        """Field labels and aliases discovered from the configured schemas.

        This is what makes routing generalise: adding a document schema (or a
        tenant-specific one) makes its field vocabulary routable immediately,
        with no change to this module. Failure here is non-fatal — the router
        degrades to :attr:`base_field_terms` and the ask() cascade still finds
        the field.
        """

        try:
            from app.services.vision.schema_registry import vision_schema_registry

            discovered: set[str] = set()
            for schema in vision_schema_registry.registered_schemas():
                for spec in (*schema.required_fields, *schema.optional_fields):
                    for label in (spec.name, *spec.aliases):
                        text = str(label).replace("_", " ").strip().lower()
                        # Single characters and pure numbers are not useful routing
                        # hints and would match far too broadly.
                        if len(text) >= 4 and not text.isdigit():
                            discovered.add(text)
            return tuple(sorted(discovered))
        except Exception:  # pragma: no cover - registry is optional at import time
            return ()

    @classmethod
    def reset_vocabulary_cache(cls) -> None:
        cls._registry_field_terms.cache_clear()
