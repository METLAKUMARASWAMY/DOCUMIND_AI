"""Strict loader for the versioned, data-driven Vision schema catalog."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from app.models.vision import VisionDocumentType, VisionSemanticType

SUPPORTED_FORMAT_VERSION = 1
DEFAULT_RESOURCE_PACKAGE = "app.services.vision.schemas"
DEFAULT_RESOURCE_NAME = "catalog.v1.json"

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"^v\d+(?:\.\d+)*$")
_CATALOG_VERSION = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")
_RELATIONSHIP = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SCHEMA_KINDS = {"exact", "generic_key_value", "generic_metadata"}
_VALIDATORS = {
    "visible_text",
    "name",
    "title",
    "email",
    "phone",
    "money",
    "currency",
    "date",
    "time",
    "integer",
    "boolean",
    "quantity",
    "reference",
}
_NORMALIZERS = {
    "casefold_whitespace",
    "preserve",
    "phone",
    "lowercase",
    "iso_date",
    "sensitive_hash",
    "decimal_amount",
    "currency_code",
    "integer",
    "boolean",
}
_EXTRACTION_STRATEGIES = {
    "same_line_label_value",
    "nearest_right_value",
    "nearest_lower_value",
    "multi_line_continuation",
    "table_cell_relationship",
    "roi_reocr_same_line",
    "source_matched_pattern",
    "measured_title_region",
    "strong_title_ocr_line",
    "top_ocr_line",
}
DEFAULT_EXTRACTION_STRATEGIES = (
    "same_line_label_value",
    "nearest_right_value",
    "nearest_lower_value",
    "table_cell_relationship",
    "roi_reocr_same_line",
    "multi_line_continuation",
    "source_matched_pattern",
    "measured_title_region",
    "strong_title_ocr_line",
    "top_ocr_line",
)


class SchemaConfigurationError(ValueError):
    """Raised when schema configuration cannot be trusted at startup."""


@dataclass(frozen=True)
class ConfiguredValueKind:
    name: str
    validator: str
    normalizer: str
    semantic_entity_type: str


@dataclass(frozen=True)
class ConfiguredField:
    name: str
    display_name: str
    aliases: tuple[str, ...]
    required: bool
    value_kind: str
    multiline: bool
    extraction_strategies: tuple[str, ...]
    semantic_entity_type: str
    validators: tuple[str, ...]
    normalizers: tuple[str, ...]
    graph_relationship: str | None
    multi_value: bool
    allow_evidence_sharing_with: tuple[str, ...]
    incompatible_with: tuple[str, ...]


@dataclass(frozen=True)
class ConfiguredSchema:
    schema_id: str
    schema_kind: str
    version: str
    family: str
    document_types: tuple[VisionDocumentType, ...]
    fields: tuple[ConfiguredField, ...]
    classification_hints: tuple[str, ...]
    applicable_modules: tuple[str, ...]


@dataclass(frozen=True)
class LoadedSchemaCatalog:
    format_version: int
    catalog_version: str
    schemas: tuple[ConfiguredSchema, ...]
    parent_schema_by_document_type: dict[VisionDocumentType, str]
    source: str
    fingerprint: str


def load_schema_catalog(path: str | Path | None = None) -> LoadedSchemaCatalog:
    """Load and validate a catalog, failing explicitly instead of silently falling back."""

    if path is None:
        resource = resources.files(DEFAULT_RESOURCE_PACKAGE).joinpath(DEFAULT_RESOURCE_NAME)
        source = f"package:{DEFAULT_RESOURCE_PACKAGE}/{DEFAULT_RESOURCE_NAME}"
        try:
            raw_bytes = resource.read_bytes()
        except OSError as exc:
            raise SchemaConfigurationError(f"Unable to read {source}: {exc}") from exc
    else:
        observed = Path(path).expanduser().resolve()
        source = str(observed)
        try:
            raw_bytes = observed.read_bytes()
        except OSError as exc:
            raise SchemaConfigurationError(f"Unable to read {source}: {exc}") from exc

    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaConfigurationError(f"Invalid JSON in {source}: {exc}") from exc
    catalog = _parse_catalog(payload, source)
    return LoadedSchemaCatalog(
        format_version=catalog["format_version"],
        catalog_version=catalog["catalog_version"],
        schemas=catalog["schemas"],
        parent_schema_by_document_type=catalog["parents"],
        source=source,
        fingerprint=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _parse_catalog(payload: Any, source: str) -> dict[str, Any]:
    root = _object(payload, "$", source)
    _only_keys(
        root,
        {
            "format_version",
            "catalog_version",
            "value_kinds",
            "field_sets",
            "schemas",
            "parent_schema_by_document_type",
        },
        "$",
        source,
    )
    format_version = root.get("format_version")
    if type(format_version) is not int or format_version != SUPPORTED_FORMAT_VERSION:
        _fail(
            source,
            "$.format_version",
            f"must equal supported integer {SUPPORTED_FORMAT_VERSION}",
        )
    catalog_version = _required_string(root, "catalog_version", "$", source)
    if not _CATALOG_VERSION.fullmatch(catalog_version):
        _fail(source, "$.catalog_version", "must use YYYY.MM.DD")

    value_kinds = _parse_value_kinds(root.get("value_kinds"), source)
    field_sets_payload = _object(root.get("field_sets"), "$.field_sets", source)
    field_sets: dict[str, tuple[ConfiguredField, ...]] = {}
    for name, fields_payload in field_sets_payload.items():
        _identifier(name, f"$.field_sets.{name}", source)
        field_sets[name] = _parse_fields(
            fields_payload,
            value_kinds,
            f"$.field_sets.{name}",
            source,
        )

    schemas_payload = root.get("schemas")
    if not isinstance(schemas_payload, list) or not schemas_payload:
        _fail(source, "$.schemas", "must be a non-empty array")
    schemas = tuple(
        _parse_schema(item, index, field_sets, value_kinds, source)
        for index, item in enumerate(schemas_payload)
    )
    schema_ids = [item.schema_id for item in schemas]
    _unique(schema_ids, "$.schemas[*].schema_id", source)

    exact_types: dict[VisionDocumentType, str] = {}
    generic_ids: set[str] = set()
    for schema in schemas:
        if schema.schema_kind == "exact":
            if not schema.document_types:
                _fail(source, f"$.schemas[{schema.schema_id}]", "exact schema needs document_types")
            for document_type in schema.document_types:
                previous = exact_types.get(document_type)
                if previous is not None:
                    _fail(
                        source,
                        "$.schemas",
                        f"document type {document_type.value!r} is assigned to both "
                        f"{previous!r} and {schema.schema_id!r}",
                    )
                exact_types[document_type] = schema.schema_id
        else:
            if schema.document_types:
                _fail(
                    source,
                    f"$.schemas[{schema.schema_id}]",
                    "generic schema cannot bind a document type",
                )
            generic_ids.add(schema.schema_id)
    for required_generic in ("generic_key_value_form", "generic_document"):
        if required_generic not in generic_ids:
            _fail(source, "$.schemas", f"missing required generic schema {required_generic!r}")

    parents_payload = _object(
        root.get("parent_schema_by_document_type"),
        "$.parent_schema_by_document_type",
        source,
    )
    parents: dict[VisionDocumentType, str] = {}
    for raw_type, schema_id in parents_payload.items():
        try:
            document_type = VisionDocumentType(raw_type)
        except ValueError:
            _fail(source, f"$.parent_schema_by_document_type.{raw_type}", "unknown document type")
        parent_id = _nonempty_string(
            schema_id,
            f"$.parent_schema_by_document_type.{raw_type}",
            source,
        )
        if parent_id not in schema_ids:
            _fail(
                source,
                f"$.parent_schema_by_document_type.{raw_type}",
                f"references missing schema {parent_id!r}",
            )
        parents[document_type] = parent_id

    return {
        "format_version": format_version,
        "catalog_version": catalog_version,
        "schemas": schemas,
        "parents": parents,
    }


def _parse_value_kinds(payload: Any, source: str) -> dict[str, ConfiguredValueKind]:
    values = _object(payload, "$.value_kinds", source)
    if not values:
        _fail(source, "$.value_kinds", "must not be empty")
    parsed: dict[str, ConfiguredValueKind] = {}
    semantic_types = {item.value for item in VisionSemanticType}
    for name, raw in values.items():
        path = f"$.value_kinds.{name}"
        _identifier(name, path, source)
        item = _object(raw, path, source)
        _only_keys(item, {"validator", "normalizer", "semantic_entity_type"}, path, source)
        validator = _required_string(item, "validator", path, source)
        normalizer = _required_string(item, "normalizer", path, source)
        semantic_type = _required_string(item, "semantic_entity_type", path, source)
        if validator not in _VALIDATORS:
            _fail(source, f"{path}.validator", f"unsupported validator {validator!r}")
        if normalizer not in _NORMALIZERS:
            _fail(source, f"{path}.normalizer", f"unsupported normalizer {normalizer!r}")
        if semantic_type not in semantic_types:
            _fail(
                source, f"{path}.semantic_entity_type", f"unknown semantic type {semantic_type!r}"
            )
        parsed[name] = ConfiguredValueKind(name, validator, normalizer, semantic_type)
    return parsed


def _parse_schema(
    payload: Any,
    index: int,
    field_sets: dict[str, tuple[ConfiguredField, ...]],
    value_kinds: dict[str, ConfiguredValueKind],
    source: str,
) -> ConfiguredSchema:
    path = f"$.schemas[{index}]"
    item = _object(payload, path, source)
    _only_keys(
        item,
        {
            "schema_id",
            "schema_kind",
            "version",
            "family",
            "document_types",
            "field_set",
            "fields",
            "classification_hints",
            "applicable_modules",
        },
        path,
        source,
    )
    schema_id = _required_string(item, "schema_id", path, source)
    _identifier(schema_id, f"{path}.schema_id", source)
    schema_kind = _required_string(item, "schema_kind", path, source)
    if schema_kind not in _SCHEMA_KINDS:
        _fail(source, f"{path}.schema_kind", f"unsupported schema kind {schema_kind!r}")
    version = _required_string(item, "version", path, source)
    if not _VERSION.fullmatch(version):
        _fail(source, f"{path}.version", "must be vN or vN.N")
    family = _required_string(item, "family", path, source)
    _identifier(family, f"{path}.family", source)

    raw_types = _string_list(item.get("document_types"), f"{path}.document_types", source)
    document_types: list[VisionDocumentType] = []
    for raw_type in raw_types:
        try:
            document_types.append(VisionDocumentType(raw_type))
        except ValueError:
            _fail(source, f"{path}.document_types", f"unknown document type {raw_type!r}")
    _unique([item.value for item in document_types], f"{path}.document_types", source)

    has_field_set = "field_set" in item
    has_fields = "fields" in item
    if has_field_set == has_fields:
        _fail(source, path, "must define exactly one of field_set or fields")
    if has_field_set:
        field_set = _nonempty_string(item["field_set"], f"{path}.field_set", source)
        if field_set not in field_sets:
            _fail(source, f"{path}.field_set", f"references missing field set {field_set!r}")
        fields = field_sets[field_set]
    else:
        fields = _parse_fields(item["fields"], value_kinds, f"{path}.fields", source)

    field_names = {field.name for field in fields}
    for field in fields:
        missing_share = set(field.allow_evidence_sharing_with) - field_names
        if missing_share:
            _fail(
                source,
                f"{path}.fields.{field.name}.allow_evidence_sharing_with",
                f"references fields outside this schema: {sorted(missing_share)}",
            )
    return ConfiguredSchema(
        schema_id=schema_id,
        schema_kind=schema_kind,
        version=version,
        family=family,
        document_types=tuple(document_types),
        fields=fields,
        classification_hints=_string_list(
            item.get("classification_hints"),
            f"{path}.classification_hints",
            source,
        ),
        applicable_modules=_string_list(
            item.get("applicable_modules"),
            f"{path}.applicable_modules",
            source,
        ),
    )


def _parse_fields(
    payload: Any,
    value_kinds: dict[str, ConfiguredValueKind],
    path: str,
    source: str,
) -> tuple[ConfiguredField, ...]:
    if not isinstance(payload, list) or not payload:
        _fail(source, path, "must be a non-empty array")
    fields = tuple(
        _parse_field(item, index, value_kinds, path, source) for index, item in enumerate(payload)
    )
    _unique([item.name for item in fields], f"{path}[*].name", source)
    return fields


def _parse_field(
    payload: Any,
    index: int,
    value_kinds: dict[str, ConfiguredValueKind],
    parent_path: str,
    source: str,
) -> ConfiguredField:
    path = f"{parent_path}[{index}]"
    item = _object(payload, path, source)
    _only_keys(
        item,
        {
            "name",
            "display_name",
            "aliases",
            "required",
            "value_kind",
            "multiline",
            "extraction_strategies",
            "semantic_entity_type",
            "validators",
            "normalizers",
            "graph_relationship",
            "multi_value",
            "allow_evidence_sharing_with",
            "incompatible_with",
        },
        path,
        source,
    )
    name = _required_string(item, "name", path, source)
    _identifier(name, f"{path}.name", source)
    display_name = _nonempty_string(
        item.get("display_name", name.replace("_", " ").title()),
        f"{path}.display_name",
        source,
    )
    kind_name = _nonempty_string(item.get("value_kind", "text"), f"{path}.value_kind", source)
    kind = value_kinds.get(kind_name)
    if kind is None:
        _fail(source, f"{path}.value_kind", f"references missing value kind {kind_name!r}")

    aliases = _string_list(item.get("aliases", []), f"{path}.aliases", source)
    aliases = tuple(dict.fromkeys((name.replace("_", " "), *aliases)))
    validators = _string_list(
        item.get("validators", [kind.validator]),
        f"{path}.validators",
        source,
    )
    normalizers = _string_list(
        item.get("normalizers", [kind.normalizer]),
        f"{path}.normalizers",
        source,
    )
    strategies = _string_list(
        item.get("extraction_strategies", DEFAULT_EXTRACTION_STRATEGIES),
        f"{path}.extraction_strategies",
        source,
    )
    for validator in validators:
        if validator not in _VALIDATORS:
            _fail(source, f"{path}.validators", f"unsupported validator {validator!r}")
    for normalizer in normalizers:
        if normalizer not in _NORMALIZERS:
            _fail(source, f"{path}.normalizers", f"unsupported normalizer {normalizer!r}")
    for strategy in strategies:
        if strategy not in _EXTRACTION_STRATEGIES:
            _fail(source, f"{path}.extraction_strategies", f"unsupported strategy {strategy!r}")

    semantic_type = _nonempty_string(
        item.get("semantic_entity_type", kind.semantic_entity_type),
        f"{path}.semantic_entity_type",
        source,
    )
    if semantic_type not in {item.value for item in VisionSemanticType}:
        _fail(source, f"{path}.semantic_entity_type", f"unknown semantic type {semantic_type!r}")
    graph_relationship = item.get("graph_relationship")
    if graph_relationship is not None:
        graph_relationship = _nonempty_string(
            graph_relationship,
            f"{path}.graph_relationship",
            source,
        )
        if not _RELATIONSHIP.fullmatch(graph_relationship):
            _fail(source, f"{path}.graph_relationship", "must be upper snake case")

    sharing = _identifier_list(
        item.get("allow_evidence_sharing_with", []),
        f"{path}.allow_evidence_sharing_with",
        source,
    )
    incompatible = _identifier_list(
        item.get("incompatible_with", []),
        f"{path}.incompatible_with",
        source,
    )
    overlap = set(sharing) & set(incompatible)
    if overlap:
        _fail(source, path, f"fields cannot be both compatible and incompatible: {sorted(overlap)}")
    return ConfiguredField(
        name=name,
        display_name=display_name,
        aliases=aliases,
        required=_boolean(item.get("required", False), f"{path}.required", source),
        value_kind=kind_name,
        multiline=_boolean(item.get("multiline", False), f"{path}.multiline", source),
        extraction_strategies=strategies,
        semantic_entity_type=semantic_type,
        validators=validators,
        normalizers=normalizers,
        graph_relationship=graph_relationship,
        multi_value=_boolean(item.get("multi_value", False), f"{path}.multi_value", source),
        allow_evidence_sharing_with=sharing,
        incompatible_with=incompatible,
    )


def _object(value: Any, path: str, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(source, path, "must be an object")
    return value


def _only_keys(value: dict[str, Any], allowed: set[str], path: str, source: str) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        _fail(source, path, f"contains unsupported keys: {unexpected}")


def _required_string(value: dict[str, Any], key: str, path: str, source: str) -> str:
    if key not in value:
        _fail(source, f"{path}.{key}", "is required")
    return _nonempty_string(value[key], f"{path}.{key}", source)


def _nonempty_string(value: Any, path: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(source, path, "must be a non-empty string")
    return value.strip()


def _string_list(value: Any, path: str, source: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        _fail(source, path, "must be an array of strings")
    parsed = tuple(
        _nonempty_string(item, f"{path}[{index}]", source) for index, item in enumerate(value)
    )
    _unique(list(parsed), path, source)
    return parsed


def _identifier_list(value: Any, path: str, source: str) -> tuple[str, ...]:
    parsed = _string_list(value, path, source)
    for item in parsed:
        _identifier(item, path, source)
    return parsed


def _identifier(value: str, path: str, source: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        _fail(source, path, "must be lower snake case")


def _boolean(value: Any, path: str, source: str) -> bool:
    if type(value) is not bool:
        _fail(source, path, "must be a boolean")
    return value


def _unique(values: list[str], path: str, source: str) -> None:
    duplicates = sorted({item for item in values if values.count(item) > 1})
    if duplicates:
        _fail(source, path, f"contains duplicates: {duplicates}")


def _fail(source: str, path: str, reason: str) -> None:
    raise SchemaConfigurationError(f"Invalid schema catalog {source} at {path}: {reason}")
