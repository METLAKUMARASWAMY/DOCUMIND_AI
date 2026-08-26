from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionDocumentType,
    VisionFieldCompleteness,
)
from app.services.vision.schema_config import (
    ConfiguredField,
    LoadedSchemaCatalog,
    load_schema_catalog,
)


@dataclass(frozen=True)
class SchemaFieldSpec:
    name: str
    display_name: str
    aliases: tuple[str, ...]
    required: bool = False
    value_kind: str = "text"
    multiline: bool = False
    extraction_strategies: tuple[str, ...] = ()
    semantic_entity_type: str | None = None
    allow_evidence_sharing_with: tuple[str, ...] = ()
    validators: tuple[str, ...] = ()
    normalizers: tuple[str, ...] = ()
    graph_relationship: str | None = None
    multi_value: bool = False
    incompatible_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentSchema:
    document_type: VisionDocumentType | None
    fields: tuple[SchemaFieldSpec, ...]
    schema_id: str = ""
    schema_kind: str = "exact"
    version: str = "v1"
    family: str = "enterprise_document"
    classification_hints: tuple[str, ...] = ()
    applicable_modules: tuple[str, ...] = ()
    catalog_version: str = "unversioned"

    @property
    def name(self) -> str:
        return self.schema_id or (
            self.document_type.value if self.document_type else "generic_document"
        )

    @property
    def required_fields(self) -> tuple[SchemaFieldSpec, ...]:
        return tuple(item for item in self.fields if item.required)

    @property
    def optional_fields(self) -> tuple[SchemaFieldSpec, ...]:
        return tuple(item for item in self.fields if not item.required)

    def field(self, name: str) -> SchemaFieldSpec | None:
        normalized = name.strip().lower()
        return next((item for item in self.fields if item.name == normalized), None)


@dataclass(frozen=True)
class SchemaSelection:
    schema: DocumentSchema
    exact_schema: str | None
    fallback_schema: str | None
    generic_schema: str
    selected_schema: str
    source: str
    path: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source not in {"exact", "parent", "generic_key_value", "generic_metadata"}:
            raise ValueError(f"Unsupported schema selection source: {self.source}")
        if self.selected_schema != self.schema.name:
            raise ValueError("Selected schema ID must match the selected schema")
        if self.source == "exact" and self.exact_schema != self.selected_schema:
            raise ValueError("Mode A selection requires the exact schema to be selected")
        if self.source == "parent" and self.fallback_schema != self.selected_schema:
            raise ValueError("Mode B parent selection requires the fallback schema to be selected")
        if self.source.startswith("generic") and (
            self.exact_schema is not None or self.fallback_schema is not None
        ):
            raise ValueError("Mode C selection cannot claim an exact or parent schema")
        if not self.path or self.selected_schema not in self.path:
            raise ValueError("Schema decision path must contain the selected schema")

    @property
    def fallback_used(self) -> bool:
        return self.source != "exact"

    @property
    def mode(self) -> str:
        if self.source == "exact":
            return "A"
        if self.source == "parent":
            return "B"
        return "C"

    @property
    def schema_status(self) -> str:
        return "registered" if self.mode == "A" else "inferred" if self.mode == "B" else "generic"

    def as_dict(self) -> dict[str, object]:
        return {
            "exact_schema": self.exact_schema,
            "fallback_schema": self.fallback_schema,
            "generic_schema": self.generic_schema,
            "selected_schema": self.selected_schema,
            "schema_source": self.source,
            "schema_path": list(self.path),
            "fallback_used": self.fallback_used,
            "mode": self.mode,
            "schema_status": self.schema_status,
            "schema_version": self.schema.version,
            "schema_family": self.schema.family,
            "schema_kind": self.schema.schema_kind,
            "schema_catalog_version": self.schema.catalog_version,
        }


def _runtime_field(item: ConfiguredField) -> SchemaFieldSpec:
    return SchemaFieldSpec(
        name=item.name,
        display_name=item.display_name,
        aliases=item.aliases,
        required=item.required,
        value_kind=item.value_kind,
        multiline=item.multiline,
        extraction_strategies=item.extraction_strategies,
        semantic_entity_type=item.semantic_entity_type,
        allow_evidence_sharing_with=item.allow_evidence_sharing_with,
        validators=item.validators,
        normalizers=item.normalizers,
        graph_relationship=item.graph_relationship,
        multi_value=item.multi_value,
        incompatible_with=item.incompatible_with,
    )


def _runtime_maps(
    catalog: LoadedSchemaCatalog,
) -> tuple[
    dict[VisionDocumentType, DocumentSchema],
    dict[str, DocumentSchema],
    dict[VisionDocumentType, str],
]:
    exact: dict[VisionDocumentType, DocumentSchema] = {}
    generic: dict[str, DocumentSchema] = {}
    for configured in catalog.schemas:
        fields = tuple(_runtime_field(item) for item in configured.fields)
        common = {
            "fields": fields,
            "schema_id": configured.schema_id,
            "schema_kind": configured.schema_kind,
            "version": configured.version,
            "family": configured.family,
            "classification_hints": configured.classification_hints,
            "applicable_modules": configured.applicable_modules,
            "catalog_version": catalog.catalog_version,
        }
        if configured.schema_kind == "exact":
            for document_type in configured.document_types:
                exact[document_type] = DocumentSchema(document_type=document_type, **common)
        else:
            generic[configured.schema_id] = DocumentSchema(document_type=None, **common)
    return exact, generic, dict(catalog.parent_schema_by_document_type)


_DEFAULT_CATALOG = load_schema_catalog()
SCHEMAS, GENERIC_SCHEMAS, PARENT_SCHEMA_BY_TYPE = _runtime_maps(_DEFAULT_CATALOG)


class VisionSchemaRegistry:
    def __init__(
        self,
        catalog: LoadedSchemaCatalog | None = None,
        *,
        catalog_path: str | Path | None = None,
    ) -> None:
        if catalog is not None and catalog_path is not None:
            raise ValueError("Provide either catalog or catalog_path, not both")
        self.catalog = catalog or (
            load_schema_catalog(catalog_path) if catalog_path is not None else _DEFAULT_CATALOG
        )
        self._schemas, self._generic_schemas, self._parents = _runtime_maps(self.catalog)

    @property
    def catalog_version(self) -> str:
        return self.catalog.catalog_version

    @property
    def catalog_fingerprint(self) -> str:
        return self.catalog.fingerprint

    @property
    def catalog_source(self) -> str:
        return self.catalog.source

    def registered_schemas(self) -> Iterator[DocumentSchema]:
        """Yield one configured schema for each registered document type."""

        yield from self._schemas.values()

    def classification_schemas(self) -> Iterator[DocumentSchema]:
        """Yield unambiguous schema/type pairs suitable for classifier hints."""

        schema_id_counts = Counter(item.name for item in self._schemas.values())
        yield from (
            item
            for item in self._schemas.values()
            if schema_id_counts[item.name] == 1 and item.classification_hints
        )

    def get(self, document_type: VisionDocumentType | str) -> DocumentSchema | None:
        resolved = self._document_type(document_type)
        return self._schemas.get(resolved) if resolved else None

    def get_by_id(self, schema_id: str) -> DocumentSchema | None:
        if schema_id in self._generic_schemas:
            return self._generic_schemas[schema_id]
        return next((item for item in self._schemas.values() if item.name == schema_id), None)

    def resolve(
        self,
        document_type: VisionDocumentType | str,
        *,
        prefer_key_value: bool = False,
    ) -> SchemaSelection:
        resolved = self._document_type(document_type) or VisionDocumentType.unknown
        exact = self._schemas.get(resolved)
        parent_id = self._parents.get(resolved)
        parent = self.get_by_id(parent_id) if parent_id and exact is None else None
        generic_id = "generic_key_value_form" if prefer_key_value else "generic_document"
        generic = self._generic_schemas[generic_id]
        selected = exact or parent or generic
        source = "exact" if exact else "parent" if parent else generic.schema_kind
        fallback_id = parent.name if parent is not None else None
        path = tuple(
            dict.fromkeys(
                item
                for item in (
                    resolved.value if resolved != VisionDocumentType.unknown else None,
                    exact.name if exact else None,
                    fallback_id,
                    generic_id,
                    "generic_document",
                )
                if item
            )
        )
        return SchemaSelection(
            schema=selected,
            exact_schema=exact.name if exact else None,
            fallback_schema=fallback_id,
            generic_schema=generic_id,
            selected_schema=selected.name,
            source=source,
            path=path,
        )

    def summarize(
        self,
        document_type: VisionDocumentType,
        fields: Iterable[ExtractedVisionField],
        *,
        selection: SchemaSelection | None = None,
        extraction_diagnostics: dict[str, object] | None = None,
    ) -> VisionFieldCompleteness:
        field_list = [item for item in fields if (item.reviewed_value or item.value).strip()]
        selected = selection or self.resolve(document_type, prefer_key_value=bool(field_list))
        schema = selected.schema
        by_name = {item.name: item for item in field_list}
        required = [item.name for item in schema.required_fields]
        optional = [item.name for item in schema.optional_fields]
        non_rejected_names = {
            name
            for name, item in by_name.items()
            if item.reviewer_status != ReviewerStatus.rejected
        }
        invalid = sorted(
            name
            for name, item in by_name.items()
            if item.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
            and (
                item.validation_status not in {"valid", "reviewer_verified"}
                or not self.value_is_valid(schema.field(name), item.reviewed_value or item.value)
            )
        )
        disputed = sorted(
            name
            for name, item in by_name.items()
            if item.reviewer_status == ReviewerStatus.rejected
        )
        manually_resolved = sorted(
            name
            for name, item in by_name.items()
            if item.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
        )
        resolved_names = non_rejected_names - set(invalid)
        required_matched = sorted(set(required) & resolved_names)
        missing = sorted(set(required) - set(required_matched))
        optional_extracted = sorted(set(optional) & resolved_names)
        name_counts = Counter(item.name for item in field_list)
        duplicate_fields = {name for name, count in name_counts.items() if count > 1}
        diagnostics = extraction_diagnostics or {}
        rejected_candidates = diagnostics.get("rejected_candidates") or []
        duplicate_rejections = [
            item
            for item in rejected_candidates
            if isinstance(item, dict)
            and str(item.get("reason", "")).startswith("duplicate_evidence")
        ]
        for item in duplicate_rejections:
            if item.get("field"):
                duplicate_fields.add(str(item["field"]))
            if item.get("conflicts_with"):
                duplicate_fields.add(str(item["conflicts_with"]))
        duplicate_rejected_count = int(
            diagnostics.get("duplicates_rejected", len(duplicate_rejections)) or 0
        )
        usable = not (
            selected.source == "generic_metadata"
            and not field_list
            and document_type == VisionDocumentType.unknown
        )
        return VisionFieldCompleteness(
            schema_available=usable,
            document_type=document_type,
            required_fields=required if usable else [],
            optional_fields=optional if usable else [],
            extracted_fields=sorted(by_name),
            missing_required_fields=missing if usable else [],
            invalid_fields=invalid,
            disputed_fields=disputed,
            manually_resolved_fields=manually_resolved,
            required_matched_fields=required_matched if usable else [],
            optional_extracted_fields=optional_extracted,
            duplicate_fields=sorted(duplicate_fields),
            required_expected_count=len(required) if usable else None,
            required_matched_count=len(required_matched) if usable else None,
            required_missing_count=len(missing) if usable else None,
            optional_extracted_count=len(optional_extracted),
            total_extracted_count=len(field_list),
            duplicates_rejected_count=duplicate_rejected_count,
            disputed_count=len(disputed),
            expected_required_count=len(required) if usable else None,
            extracted_count=len(field_list),
            unresolved_required_count=len(missing) if usable else None,
            exact_schema=selected.exact_schema,
            fallback_schema=selected.fallback_schema,
            generic_schema=selected.generic_schema,
            selected_schema=selected.selected_schema,
            schema_source=selected.source,
            schema_path=list(selected.path),
            fallback_used=selected.fallback_used,
        )

    def value_is_valid(self, spec: SchemaFieldSpec | None, value: str) -> bool:
        valid, _ = self.validate_value(spec, value)
        return valid

    def validate_value(
        self,
        spec: SchemaFieldSpec | None,
        value: str,
    ) -> tuple[bool, str]:
        cleaned = value.strip()
        if not cleaned:
            return False, "empty_value"
        validator = spec.validators[0] if spec and spec.validators else (
            spec.value_kind if spec else "visible_text"
        )
        validator = {
            "text": "visible_text",
            "organisation": "visible_text",
            "product": "visible_text",
            "name_list": "visible_text",
            "organisation_list": "visible_text",
            "email_list": "email",
            "phone_list": "phone",
            "money_list": "money",
            "date_list": "date",
            "reference_list": "reference",
            "account": "reference",
        }.get(validator, validator)
        if validator == "visible_text":
            valid = any(character.isalnum() for character in cleaned)
            return valid, "contains_visible_text" if valid else "missing_visible_text"
        if validator == "name":
            date_like = re.search(
                r"(?i)\b(?:date|dated|january|february|march|april|may|june|july|"
                r"august|september|october|november|december)\b.*\d",
                cleaned,
            )
            valid = bool(re.search(r"[A-Za-z]", cleaned)) and not date_like
            return valid, "name_text_preserved" if valid else "name_is_date_or_missing"
        if validator == "title":
            labelled_value = re.match(
                r"(?i)^\s*(?:date|dated|sample\s+(?:no\.?|number|id)|batch\s+size|"
                r"type\s+of\s+cigarette|original\s+request\s+made\s+by|"
                r"sample\s+specifications\s+written\s+by)\s*[:;,#=_-]",
                cleaned,
            )
            date_line = re.search(
                r"(?i)\b(?:january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\s+\d{1,2}.*\d",
                cleaned,
            )
            valid = bool(re.search(r"[A-Za-z]", cleaned)) and labelled_value is None and date_line is None
            return valid, "title_like_text" if valid else "labelled_value_is_not_a_title"
        if validator == "email":
            valid = re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", cleaned) is not None
            return valid, "valid_email" if valid else "invalid_email_format"
        if validator == "phone":
            date_token = re.compile(
                r"(?<!\d)(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?!\d)"
            )
            if date_token.search(cleaned):
                return False, "calendar_date_is_not_phone"
            digits = re.sub(r"\D", "", cleaned)
            phone_shape = re.fullmatch(
                r"(?x)\s*(?:\+\s*)?(?:\(?\d{2,4}\)?[ ./-]*)?\d{3,4}[ ./-]\d{3,4}\s*",
                cleaned,
            )
            valid = 7 <= len(digits) <= 15 and phone_shape is not None
            return valid, "valid_phone_shape" if valid else "invalid_phone_shape_or_length"
        if validator == "money":
            valid = re.search(r"-?\d[\d,]*(?:\.\d+)?", cleaned) is not None
            return valid, "numeric_amount_present" if valid else "amount_not_parseable"
        if validator == "currency":
            valid = re.fullmatch(
                r"(?:USD|INR|EUR|GBP|AUD|CAD|JPY|\$|€|£|₹)", cleaned, re.I
            ) is not None
            return valid, "known_currency" if valid else "unknown_currency"
        if validator == "date":
            date_cleaned = cleaned.rstrip(" .,:;")
            year_match = re.search(r"(?<!\d)(\d{4})(?!\d)", date_cleaned)
            if year_match and not 1800 <= int(year_match.group(1)) <= 2100:
                return False, "date_year_out_of_range"
            for date_format in (
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%d-%m-%Y",
                "%m-%d-%Y",
                "%d/%m/%y",
                "%m/%d/%y",
                "%d-%m-%y",
                "%m-%d-%y",
                "%b %d, %Y",
                "%B %d, %Y",
                "%b %d %Y",
                "%B %d %Y",
            ):
                try:
                    datetime.strptime(date_cleaned, date_format)
                    numeric = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", date_cleaned)
                    if numeric and int(numeric.group(1)) <= 12 and int(numeric.group(2)) <= 12:
                        return True, "ambiguous_numeric_date"
                    return True, "valid_calendar_date"
                except ValueError:
                    continue
            date_like = bool(
                re.search(
                    r"(?i)\b(?:january|february|march|april|may|june|july|"
                    r"august|september|october|november|december)\b",
                    date_cleaned,
                )
                and re.search(r"\d", date_cleaned)
            ) or bool(re.fullmatch(r"\d{1,4}(?:[/-]\d{1,4}){1,2}", date_cleaned))
            return (True, "date_uncertain_needs_review") if date_like else (False, "date_not_parseable")
        if validator == "time":
            valid = re.search(r"\b\d{1,2}:\d{2}(?:\s*[AP]M)?\b", cleaned, re.I) is not None
            return valid, "valid_time" if valid else "time_not_parseable"
        if validator == "integer":
            valid = re.fullmatch(r"\d{1,5}", cleaned) is not None
            return valid, "valid_integer" if valid else "integer_not_parseable"
        if validator == "boolean":
            valid = cleaned.lower() in {
                "yes",
                "no",
                "true",
                "false",
                "y",
                "n",
                "checked",
                "unchecked",
            }
            return valid, "valid_boolean" if valid else "boolean_not_parseable"
        if validator == "quantity":
            valid = re.fullmatch(
                r"(?i)\s*\d+(?:[.,]\d+)?\s*(?:lbs?\.?|pounds?|kg|kgs|g|grams?|"
                r"mg|oz|ounces?|litres?|liters?|ml|units?|pcs?\.?)\s*",
                cleaned,
            ) is not None
            return valid, "number_and_unit_present" if valid else "quantity_not_parseable"
        if validator == "reference":
            valid = bool(re.search(r"[A-Za-z0-9]", cleaned)) and len(cleaned) <= 100
            return valid, "valid_identifier" if valid else "identifier_not_parseable"
        raise ValueError(f"Schema field {spec.name if spec else '<unknown>'} uses unknown validator {validator}")

    def _document_type(
        self,
        value: VisionDocumentType | str,
    ) -> VisionDocumentType | None:
        if isinstance(value, VisionDocumentType):
            return value
        try:
            return VisionDocumentType(str(value))
        except ValueError:
            return None


vision_schema_registry = VisionSchemaRegistry()
