"""Reusable semantic value validation, normalization, and classification.

The types in this module deliberately have no dependency on Pydantic or on a
particular document schema.  They can therefore be used by schema extraction,
generic discovery, table extraction, and retrieval indexing without coupling
those layers together.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import ClassVar


class SemanticValueType(str, Enum):
    DATE = "date"
    DATETIME = "datetime"
    PERSON = "person"
    ORGANISATION = "organisation"
    MONEY = "money"
    NUMBER = "number"
    QUANTITY = "quantity"
    PERCENTAGE = "percentage"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    BANK_ACCOUNT = "bank_account"
    IDENTIFIER = "identifier"
    DOCUMENT_REFERENCE = "document_reference"
    CURRENCY = "currency"
    PRODUCT = "product"
    TEXT = "text"
    BOOLEAN = "boolean"
    URL = "url"


@dataclass(frozen=True)
class SemanticContext:
    """Evidence surrounding a value; all properties are optional hints."""

    label: str = ""
    surrounding_text: str = ""
    document_type: str = ""
    section: str = ""

    @property
    def normalized_label(self) -> str:
        return _normalize_words(self.label)

    @property
    def normalized_context(self) -> str:
        return _normalize_words(
            " ".join(
                part
                for part in (
                    self.label,
                    self.surrounding_text,
                    self.document_type,
                    self.section,
                )
                if part
            )
        )


@dataclass(frozen=True)
class ValidationDecision:
    valid: bool
    confidence: float
    reason: str


@dataclass(frozen=True)
class SemanticValidation:
    semantic_type: SemanticValueType
    raw_value: str
    normalized_value: str | None
    valid: bool
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "semantic_type": self.semantic_type.value,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "valid": self.valid,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SemanticSpan:
    start: int
    end: int
    validation: SemanticValidation

    @property
    def text(self) -> str:
        return self.validation.raw_value


Validator = Callable[[str, SemanticContext], ValidationDecision]
Normalizer = Callable[[str, SemanticContext], str]


@dataclass(frozen=True)
class SemanticTypeDefinition:
    semantic_type: SemanticValueType
    validator: Validator
    normalizer: Normalizer


class SemanticValueRegistry:
    """Registry for independently testable semantic value definitions."""

    _TIE_BREAK_ORDER: ClassVar[dict[SemanticValueType, int]] = {
        SemanticValueType.DATETIME: 19,
        SemanticValueType.DATE: 18,
        SemanticValueType.EMAIL: 17,
        SemanticValueType.URL: 16,
        SemanticValueType.BANK_ACCOUNT: 15,
        SemanticValueType.DOCUMENT_REFERENCE: 14,
        SemanticValueType.PHONE: 13,
        SemanticValueType.MONEY: 12,
        SemanticValueType.QUANTITY: 11,
        SemanticValueType.PERCENTAGE: 10,
        SemanticValueType.CURRENCY: 9,
        SemanticValueType.IDENTIFIER: 8,
        SemanticValueType.NUMBER: 7,
        SemanticValueType.BOOLEAN: 6,
        SemanticValueType.ORGANISATION: 5,
        SemanticValueType.PERSON: 4,
        SemanticValueType.ADDRESS: 3,
        SemanticValueType.PRODUCT: 2,
        SemanticValueType.TEXT: 1,
    }

    def __init__(self) -> None:
        self._definitions: dict[SemanticValueType, SemanticTypeDefinition] = {}

    @property
    def registered_types(self) -> tuple[SemanticValueType, ...]:
        return tuple(self._definitions)

    def register(
        self,
        semantic_type: SemanticValueType,
        validator: Validator,
        normalizer: Normalizer,
        *,
        replace: bool = False,
    ) -> None:
        if semantic_type in self._definitions and not replace:
            raise ValueError(f"Semantic type already registered: {semantic_type.value}")
        self._definitions[semantic_type] = SemanticTypeDefinition(
            semantic_type=semantic_type,
            validator=validator,
            normalizer=normalizer,
        )

    def validate(
        self,
        semantic_type: SemanticValueType | str,
        value: str,
        context: SemanticContext | None = None,
    ) -> SemanticValidation:
        kind = SemanticValueType(semantic_type)
        definition = self._definitions.get(kind)
        if definition is None:
            raise KeyError(f"No validator registered for {kind.value}")

        raw_value = str(value).strip()
        observed_context = context or SemanticContext()
        if not raw_value:
            return SemanticValidation(kind, raw_value, None, False, 0.0, "empty_value")

        # This exclusion is global rather than label- or schema-specific.  OCR
        # frequently emits dates as phone-like digit runs, but a valid date token
        # must never be accepted by the PHONE validator.
        if kind is SemanticValueType.PHONE and _contains_date_token(raw_value):
            return SemanticValidation(
                kind,
                raw_value,
                None,
                False,
                0.0,
                "date_shape_excluded_from_phone",
            )

        decision = definition.validator(raw_value, observed_context)
        if not decision.valid:
            return SemanticValidation(
                kind,
                raw_value,
                None,
                False,
                _clamp(decision.confidence),
                decision.reason,
            )

        try:
            normalized = definition.normalizer(raw_value, observed_context)
        except (InvalidOperation, ValueError):
            return SemanticValidation(
                kind,
                raw_value,
                None,
                False,
                0.0,
                "normalization_failed",
            )
        confidence = _clamp(decision.confidence + self._context_adjustment(kind, observed_context))
        return SemanticValidation(
            kind,
            raw_value,
            normalized,
            True,
            confidence,
            decision.reason,
        )

    def classify(
        self,
        value: str,
        context: SemanticContext | None = None,
        *,
        preferred_types: Iterable[SemanticValueType | str] = (),
        include_text: bool = True,
        minimum_confidence: float = 0.5,
    ) -> tuple[SemanticValidation, ...]:
        observed_context = context or SemanticContext()
        preferred = tuple(SemanticValueType(item) for item in preferred_types)
        ordered_types = tuple(
            dict.fromkeys(
                (
                    *preferred,
                    *(
                        item
                        for item in self._definitions
                        if include_text or item is not SemanticValueType.TEXT
                    ),
                )
            )
        )
        results = [
            result
            for kind in ordered_types
            if (
                result := self.validate(kind, value, observed_context)
            ).valid
            and result.confidence >= minimum_confidence
        ]
        results.sort(
            key=lambda item: (
                item.confidence,
                item.semantic_type in preferred,
                self._TIE_BREAK_ORDER.get(item.semantic_type, 0),
            ),
            reverse=True,
        )
        return tuple(results)

    def classify_best(
        self,
        value: str,
        context: SemanticContext | None = None,
        *,
        preferred_types: Iterable[SemanticValueType | str] = (),
        include_text: bool = True,
        minimum_confidence: float = 0.5,
    ) -> SemanticValidation | None:
        matches = self.classify(
            value,
            context,
            preferred_types=preferred_types,
            include_text=include_text,
            minimum_confidence=minimum_confidence,
        )
        return matches[0] if matches else None

    def find_spans(
        self,
        text: str,
        context: SemanticContext | None = None,
    ) -> tuple[SemanticSpan, ...]:
        """Find conservative standalone typed spans in free text.

        Context-dependent PERSON/ORGANISATION/IDENTIFIER values are discovered
        from key/value pairs instead of guessed from arbitrary capitalized text.
        """

        patterns: tuple[tuple[SemanticValueType, re.Pattern[str]], ...] = (
            (SemanticValueType.EMAIL, _EMAIL_SEARCH),
            (SemanticValueType.URL, _URL_SEARCH),
            (SemanticValueType.DATETIME, _DATETIME_SEARCH),
            (SemanticValueType.DATE, _DATE_SEARCH),
            (SemanticValueType.MONEY, _MONEY_SEARCH),
            (SemanticValueType.PERCENTAGE, _PERCENTAGE_SEARCH),
            (SemanticValueType.QUANTITY, _QUANTITY_SEARCH),
            (SemanticValueType.PHONE, _PHONE_SEARCH),
        )
        candidates: list[SemanticSpan] = []
        for semantic_type, pattern in patterns:
            for match in pattern.finditer(text):
                raw_value = match.group(0).strip(" \t,;")
                start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
                result = self.validate(semantic_type, raw_value, context)
                if result.valid:
                    candidates.append(SemanticSpan(start, start + len(raw_value), result))

        accepted: list[SemanticSpan] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                item.validation.confidence,
                item.end - item.start,
                self._TIE_BREAK_ORDER.get(item.validation.semantic_type, 0),
            ),
            reverse=True,
        ):
            if any(candidate.start < item.end and candidate.end > item.start for item in accepted):
                continue
            accepted.append(candidate)
        return tuple(sorted(accepted, key=lambda item: (item.start, item.end)))

    def _context_adjustment(
        self,
        kind: SemanticValueType,
        context: SemanticContext,
    ) -> float:
        label = context.normalized_label
        if not label:
            return 0.0
        positive: dict[SemanticValueType, tuple[str, ...]] = {
            SemanticValueType.DATE: ("date", "dated", "due", "issued", "submitted"),
            SemanticValueType.DATETIME: ("datetime", "timestamp", "created at"),
            SemanticValueType.PHONE: ("phone", "telephone", "mobile", "fax", "tel"),
            SemanticValueType.EMAIL: ("email", "e mail"),
            SemanticValueType.BANK_ACCOUNT: (
                "account",
                "iban",
                "routing",
                "sort code",
                "swift",
            ),
            SemanticValueType.IDENTIFIER: (" id", "identifier", "code", "number", " no"),
            SemanticValueType.DOCUMENT_REFERENCE: (
                "invoice",
                "purchase order",
                " po ",
                "reference",
                "document number",
            ),
            SemanticValueType.PERSON: (
                "requested by",
                "prepared by",
                "approved by",
                "contact",
                "person",
                "name",
            ),
            SemanticValueType.ORGANISATION: (
                "organisation",
                "organization",
                "company",
                "vendor",
                "supplier",
                "customer",
            ),
            SemanticValueType.MONEY: ("amount", "total", "price", "balance", "cost"),
            SemanticValueType.QUANTITY: ("quantity", "qty", "units", "weight"),
            SemanticValueType.PERCENTAGE: ("percentage", "percent", "rate"),
            SemanticValueType.ADDRESS: ("address", "location"),
            SemanticValueType.CURRENCY: ("currency",),
            SemanticValueType.PRODUCT: ("product", "item", "description"),
            SemanticValueType.BOOLEAN: ("enabled", "active", "approved", "flag"),
            SemanticValueType.URL: ("url", "website", "web site"),
        }
        padded = f" {label} "
        if any(token in padded for token in positive.get(kind, ())):
            return 0.2
        if kind is SemanticValueType.PHONE and any(
            token in padded
            for token in (" account ", " invoice ", " reference ", " date ", " id ")
        ):
            return -0.35
        if kind is SemanticValueType.NUMBER and any(
            token in padded for token in (" account ", " id ", " reference ", " invoice ")
        ):
            return -0.2
        return 0.0


def _normalize_words(value: str) -> str:
    expanded = value.casefold().replace("#", " number ").replace("&", " and ")
    expanded = re.sub(r"\b(?:no|num|nbr)\.?\b", " number ", expanded)
    return re.sub(r"[^a-z0-9]+", " ", expanded).strip()


def _clean_text(value: str, _context: SemanticContext) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _decision(valid: bool, confidence: float, reason: str) -> ValidationDecision:
    return ValidationDecision(valid, confidence if valid else 0.0, reason)


def _expand_year(value: int) -> int:
    if value >= 100:
        return value
    return 2000 + value if value <= 68 else 1900 + value


def _parse_date(value: str) -> tuple[str, float, str] | None:
    cleaned = re.sub(r"\s+", " ", value).strip(" .,:;")
    iso_match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", cleaned)
    if iso_match:
        try:
            parsed = date(*(int(item) for item in iso_match.groups()))
        except ValueError:
            return None
        return parsed.isoformat(), 0.99, "iso_date"

    numeric = re.fullmatch(r"(\d{1,2})([/.-])(\d{1,2})\2(\d{2}|\d{4})", cleaned)
    if numeric:
        first, _, second, year_text = numeric.groups()
        first_number = int(first)
        second_number = int(second)
        year_number = _expand_year(int(year_text))
        if first_number > 31 or second_number > 31 or first_number == 0 or second_number == 0:
            return None
        if first_number > 12 and second_number <= 12:
            day_number, month_number = first_number, second_number
            reason = "numeric_date_day_first"
        elif second_number > 12 and first_number <= 12:
            month_number, day_number = first_number, second_number
            reason = "numeric_date_month_first"
        elif first_number <= 12 and second_number <= 12:
            # Preserve source order when locale cannot be established.  This is
            # normalized without asserting a false day/month interpretation.
            return (
                f"{first_number:02d}/{second_number:02d}/{year_number:04d}",
                0.84,
                "numeric_date_ambiguous_order",
            )
        else:
            return None
        try:
            parsed = date(year_number, month_number, day_number)
        except ValueError:
            return None
        return parsed.isoformat(), 0.96, reason

    for date_format in (
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ):
        try:
            parsed = datetime.strptime(cleaned, date_format).date()
            return parsed.isoformat(), 0.97, "named_month_date"
        except ValueError:
            continue
    return None


def _contains_date_token(value: str) -> bool:
    """Return true when all or part of a phone-shaped OCR value is a real date."""

    if _parse_date(value) is not None:
        return True
    candidates = re.findall(
        r"(?<!\d)(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})(?!\d)",
        value,
    )
    return any(_parse_date(item) is not None for item in candidates)


def _validate_date(value: str, _context: SemanticContext) -> ValidationDecision:
    parsed = _parse_date(value)
    return (
        _decision(True, parsed[1], parsed[2])
        if parsed
        else _decision(False, 0.0, "not_a_supported_date")
    )


def _normalize_date(value: str, _context: SemanticContext) -> str:
    parsed = _parse_date(value)
    if parsed is None:
        raise ValueError("Invalid date")
    return parsed[0]


def _parse_datetime(value: str) -> datetime | None:
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
        if "T" in cleaned or re.search(r"\d{1,2}:\d{2}", cleaned):
            return parsed
    except ValueError:
        pass
    for date_format in (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d %b %Y %H:%M",
        "%d %B %Y %H:%M",
    ):
        try:
            return datetime.strptime(cleaned, date_format)
        except ValueError:
            continue
    return None


def _validate_datetime(value: str, _context: SemanticContext) -> ValidationDecision:
    return _decision(
        _parse_datetime(value) is not None,
        0.98,
        "supported_datetime" if _parse_datetime(value) is not None else "not_a_supported_datetime",
    )


def _normalize_datetime(value: str, _context: SemanticContext) -> str:
    parsed = _parse_datetime(value)
    if parsed is None:
        raise ValueError("Invalid datetime")
    return parsed.isoformat()


def _validate_email(value: str, _context: SemanticContext) -> ValidationDecision:
    valid = bool(_EMAIL_FULL.fullmatch(value))
    return _decision(valid, 0.99, "email_syntax" if valid else "invalid_email_syntax")


def _normalize_email(value: str, _context: SemanticContext) -> str:
    local, separator, domain = value.strip().partition("@")
    return f"{local}{separator}{domain.casefold()}"


def _validate_url(value: str, _context: SemanticContext) -> ValidationDecision:
    valid = bool(_URL_FULL.fullmatch(value))
    return _decision(valid, 0.97, "url_syntax" if valid else "invalid_url_syntax")


def _normalize_url(value: str, _context: SemanticContext) -> str:
    cleaned = value.strip().rstrip(".,;)")
    scheme, separator, remainder = cleaned.partition("://")
    return f"{scheme.casefold()}{separator}{remainder}" if separator else cleaned


def _validate_phone(value: str, context: SemanticContext) -> ValidationDecision:
    if not _PHONE_FULL.fullmatch(value):
        return _decision(False, 0.0, "invalid_phone_characters")
    digits = re.sub(r"\D", "", value)
    if not 7 <= len(digits) <= 15:
        return _decision(False, 0.0, "phone_digit_count_out_of_range")
    contextual = any(
        token in f" {context.normalized_label} "
        for token in (" phone ", " telephone ", " mobile ", " fax ", " tel ")
    )
    if value.isdigit() and len(digits) < 10 and not contextual:
        return _decision(False, 0.0, "short_unformatted_number_without_phone_context")
    return _decision(True, 0.9 if contextual else 0.76, "phone_shape")


def _normalize_phone(value: str, _context: SemanticContext) -> str:
    digits = re.sub(r"\D", "", value)
    return f"+{digits}" if value.strip().startswith("+") else digits


def _decimal_text(value: str) -> str:
    number = Decimal(value.replace(",", ""))
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def _validate_number(value: str, _context: SemanticContext) -> ValidationDecision:
    valid = bool(_NUMBER_FULL.fullmatch(value.strip()))
    return _decision(valid, 0.82, "numeric_value" if valid else "not_a_number")


def _normalize_number(value: str, _context: SemanticContext) -> str:
    return _decimal_text(value.strip())


def _validate_money(value: str, context: SemanticContext) -> ValidationDecision:
    match = _MONEY_FULL.fullmatch(value.strip())
    amount_context = any(
        token in f" {context.normalized_label} "
        for token in (" amount ", " total ", " price ", " cost ", " balance ")
    )
    if not match and amount_context:
        match = _NUMBER_FULL.fullmatch(value.strip())
    return _decision(bool(match), 0.92 if match else 0.0, "monetary_value" if match else "not_money")


def _money_parts(value: str) -> tuple[str, str]:
    cleaned = value.strip()
    currency_match = re.search(r"(?i)(USD|EUR|GBP|INR|JPY|AUD|CAD)|[$€£₹¥]", cleaned)
    number_match = re.search(r"-?\d[\d,]*(?:\.\d+)?", cleaned)
    if number_match is None:
        raise ValueError("Missing amount")
    currency = _currency_code(currency_match.group(0)) if currency_match else ""
    return currency, _decimal_text(number_match.group(0))


def _normalize_money(value: str, _context: SemanticContext) -> str:
    currency, amount = _money_parts(value)
    return f"{currency} {amount}".strip()


def _validate_quantity(value: str, _context: SemanticContext) -> ValidationDecision:
    valid = bool(_QUANTITY_FULL.fullmatch(value.strip()))
    return _decision(valid, 0.94, "number_with_unit" if valid else "not_a_quantity")


def _normalize_quantity(value: str, _context: SemanticContext) -> str:
    match = _QUANTITY_FULL.fullmatch(value.strip())
    if match is None:
        raise ValueError("Invalid quantity")
    return f"{_decimal_text(match.group('number'))} {match.group('unit').casefold().rstrip('.')}"


def _validate_percentage(value: str, _context: SemanticContext) -> ValidationDecision:
    valid = bool(_PERCENTAGE_FULL.fullmatch(value.strip()))
    return _decision(valid, 0.98, "percentage_value" if valid else "not_a_percentage")


def _normalize_percentage(value: str, _context: SemanticContext) -> str:
    number = re.search(r"-?\d+(?:\.\d+)?", value)
    if number is None:
        raise ValueError("Invalid percentage")
    return f"{_decimal_text(number.group(0))}%"


def _validate_currency(value: str, _context: SemanticContext) -> ValidationDecision:
    valid = bool(_CURRENCY_FULL.fullmatch(value.strip()))
    return _decision(valid, 0.98, "currency_symbol_or_code" if valid else "unknown_currency")


def _currency_code(value: str) -> str:
    return {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "₹": "INR",
        "¥": "JPY",
    }.get(value.upper(), value.upper())


def _normalize_currency(value: str, _context: SemanticContext) -> str:
    return _currency_code(value.strip())


def _label_has(context: SemanticContext, *tokens: str) -> bool:
    padded = f" {context.normalized_label} "
    return any(token in padded for token in tokens)


def _account_shape(value: str) -> bool:
    compact = re.sub(r"[\s-]", "", value)
    return bool(re.fullmatch(r"[A-Za-z0-9]{4,34}", compact)) and any(
        character.isdigit() for character in compact
    )


def _validate_bank_account(value: str, context: SemanticContext) -> ValidationDecision:
    contextual = _label_has(
        context,
        " account ",
        " iban ",
        " routing ",
        " sort code ",
        " swift ",
    )
    compact = re.sub(r"\s", "", value)
    iban = bool(re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", compact, re.I))
    valid = _account_shape(value) and (contextual or iban)
    return _decision(valid, 0.9 if contextual else 0.87, "account_context" if valid else "not_account")


def _normalize_account(value: str, _context: SemanticContext) -> str:
    return re.sub(r"[\s-]", "", value).upper()


def _identifier_shape(value: str) -> bool:
    cleaned = value.strip()
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{1,63}", cleaned))


def _validate_identifier(value: str, context: SemanticContext) -> ValidationDecision:
    contextual = _label_has(context, " id ", " identifier ", " code ", " number ", " no ")
    valid = contextual and _identifier_shape(value)
    return _decision(valid, 0.86, "identifier_context" if valid else "not_identifier")


def _validate_document_reference(value: str, context: SemanticContext) -> ValidationDecision:
    contextual = _label_has(
        context,
        " invoice ",
        " purchase order ",
        " po ",
        " reference ",
        " document number ",
        " ref ",
    )
    valid = contextual and _identifier_shape(value)
    return _decision(valid, 0.91, "document_reference_context" if valid else "not_reference")


def _normalize_identifier(value: str, _context: SemanticContext) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _validate_person(value: str, context: SemanticContext) -> ValidationDecision:
    contextual = _label_has(
        context,
        " requested by ",
        " prepared by ",
        " approved by ",
        " contact ",
        " person ",
        " name ",
    )
    valid = contextual and bool(
        re.fullmatch(r"[\w.'-]+(?:\s+[\w.'-]+){0,6}", value.strip(), re.UNICODE)
    )
    return _decision(valid, 0.82, "person_label_context" if valid else "person_context_missing")


def _validate_organisation(value: str, context: SemanticContext) -> ValidationDecision:
    contextual = _label_has(
        context,
        " organisation ",
        " organization ",
        " company ",
        " vendor ",
        " supplier ",
        " customer ",
    )
    suffix = bool(
        re.search(
            r"(?i)\b(?:ltd|limited|inc|incorporated|llc|plc|corp|corporation|company|"
            r"pvt|private|gmbh|university|bank)\.?$",
            value.strip(),
        )
    )
    valid = (contextual or suffix) and 2 <= len(value.strip()) <= 160
    return _decision(valid, 0.83 if contextual else 0.78, "organisation_context" if valid else "not_organisation")


def _validate_address(value: str, context: SemanticContext) -> ValidationDecision:
    contextual = _label_has(context, " address ", " location ")
    address_shape = bool(
        re.search(
            r"(?i)\b(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|"
            r"suite|floor|postal|postcode|zip)\b",
            value,
        )
    )
    valid = (contextual or address_shape) and len(value.strip()) >= 5
    return _decision(valid, 0.79, "address_context" if valid else "not_address")


def _validate_product(value: str, context: SemanticContext) -> ValidationDecision:
    contextual = _label_has(context, " product ", " item ", " description ", " service ")
    valid = contextual and 1 < len(value.strip()) <= 200
    return _decision(valid, 0.72, "product_context" if valid else "product_context_missing")


def _validate_boolean(value: str, context: SemanticContext) -> ValidationDecision:
    normalized = _normalize_words(value)
    valid = normalized in {"yes", "no", "true", "false", "enabled", "disabled", "active", "inactive"}
    confidence = 0.94 if _label_has(context, " flag ", " enabled ", " active ", " approved ") else 0.82
    return _decision(valid, confidence, "boolean_lexeme" if valid else "not_boolean")


def _normalize_boolean(value: str, _context: SemanticContext) -> str:
    return (
        "true"
        if _normalize_words(value) in {"yes", "true", "enabled", "active"}
        else "false"
    )


def _validate_text(value: str, _context: SemanticContext) -> ValidationDecision:
    cleaned = value.strip()
    valid = bool(cleaned) and len(cleaned) <= 10_000
    return _decision(valid, 0.51, "non_empty_text" if valid else "empty_or_oversized_text")


_EMAIL_FULL = re.compile(r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9-]+(?:\.[A-Z0-9-]+)+", re.I)
_URL_FULL = re.compile(r"(?:(?:https?|ftp)://|www\.)[^\s<>{}\[\]]+", re.I)
_PHONE_FULL = re.compile(r"\+?[\d() ./-]+(?:\s*(?:x|ext\.?)[ ]*\d{1,6})?", re.I)
_NUMBER_FULL = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_CURRENCY_FULL = re.compile(r"(?:USD|EUR|GBP|INR|JPY|AUD|CAD|[$€£₹¥])", re.I)
_MONEY_FULL = re.compile(
    r"(?:(?:USD|EUR|GBP|INR|JPY|AUD|CAD|[$€£₹¥])\s*"
    r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|"
    r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*"
    r"(?:USD|EUR|GBP|INR|JPY|AUD|CAD))",
    re.I,
)
_UNIT = (
    r"kg|kgs?|g|grams?|mg|lb|lbs?|pounds?|oz|ounces?|l|litres?|liters?|ml|"
    r"units?|pcs?|pieces?|hours?|hrs?|days?|months?|years?|m|cm|mm|km"
)
_QUANTITY_FULL = re.compile(
    rf"(?P<number>-?\d+(?:[.,]\d+)?)\s*(?P<unit>{_UNIT})\.?",
    re.I,
)
_PERCENTAGE_FULL = re.compile(r"-?\d+(?:\.\d+)?\s*(?:%|percent(?:age)?)", re.I)

_EMAIL_SEARCH = re.compile(r"(?<![\w.+-])" + _EMAIL_FULL.pattern + r"(?![\w.-])", re.I)
_URL_SEARCH = re.compile(_URL_FULL.pattern, re.I)
_DATETIME_SEARCH = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}[T ]\d{1,2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?\b",
    re.I,
)
_DATE_SEARCH = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b",
    re.I,
)
_MONEY_SEARCH = re.compile(r"(?<!\w)" + _MONEY_FULL.pattern + r"(?!\w)", re.I)
_PERCENTAGE_SEARCH = re.compile(r"(?<!\w)" + _PERCENTAGE_FULL.pattern + r"(?!\w)", re.I)
_QUANTITY_SEARCH = re.compile(r"(?<!\w)" + _QUANTITY_FULL.pattern + r"(?!\w)", re.I)
_PHONE_SEARCH = re.compile(
    r"(?<!\w)(?:\+?\d[\d() ./-]{5,}\d)(?:\s*(?:x|ext\.?)\s*\d{1,6})?(?!\w)",
    re.I,
)


def build_default_semantic_value_registry() -> SemanticValueRegistry:
    registry = SemanticValueRegistry()
    definitions: tuple[tuple[SemanticValueType, Validator, Normalizer], ...] = (
        (SemanticValueType.DATE, _validate_date, _normalize_date),
        (SemanticValueType.DATETIME, _validate_datetime, _normalize_datetime),
        (SemanticValueType.PERSON, _validate_person, _clean_text),
        (SemanticValueType.ORGANISATION, _validate_organisation, _clean_text),
        (SemanticValueType.MONEY, _validate_money, _normalize_money),
        (SemanticValueType.NUMBER, _validate_number, _normalize_number),
        (SemanticValueType.QUANTITY, _validate_quantity, _normalize_quantity),
        (SemanticValueType.PERCENTAGE, _validate_percentage, _normalize_percentage),
        (SemanticValueType.PHONE, _validate_phone, _normalize_phone),
        (SemanticValueType.EMAIL, _validate_email, _normalize_email),
        (SemanticValueType.ADDRESS, _validate_address, _clean_text),
        (SemanticValueType.BANK_ACCOUNT, _validate_bank_account, _normalize_account),
        (SemanticValueType.IDENTIFIER, _validate_identifier, _normalize_identifier),
        (
            SemanticValueType.DOCUMENT_REFERENCE,
            _validate_document_reference,
            _normalize_identifier,
        ),
        (SemanticValueType.CURRENCY, _validate_currency, _normalize_currency),
        (SemanticValueType.PRODUCT, _validate_product, _clean_text),
        (SemanticValueType.TEXT, _validate_text, _clean_text),
        (SemanticValueType.BOOLEAN, _validate_boolean, _normalize_boolean),
        (SemanticValueType.URL, _validate_url, _normalize_url),
    )
    for semantic_type, validator, normalizer in definitions:
        registry.register(semantic_type, validator, normalizer)
    return registry


semantic_value_registry = build_default_semantic_value_registry()


__all__ = [
    "SemanticContext",
    "SemanticSpan",
    "SemanticTypeDefinition",
    "SemanticValidation",
    "SemanticValueRegistry",
    "SemanticValueType",
    "ValidationDecision",
    "build_default_semantic_value_registry",
    "semantic_value_registry",
]
