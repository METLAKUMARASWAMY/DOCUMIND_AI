from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionDocumentType,
    VisionFieldCompleteness,
)


@dataclass(frozen=True)
class SchemaFieldSpec:
    name: str
    display_name: str
    aliases: tuple[str, ...]
    required: bool = False
    value_kind: str = "text"
    multiline: bool = False
    allow_evidence_sharing_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentSchema:
    document_type: VisionDocumentType | None
    fields: tuple[SchemaFieldSpec, ...]
    schema_id: str = ""
    schema_kind: str = "exact"

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

    @property
    def fallback_used(self) -> bool:
        return self.source != "exact"

    def as_dict(self) -> dict[str, object]:
        return {
            "exact_schema": self.exact_schema,
            "fallback_schema": self.fallback_schema,
            "generic_schema": self.generic_schema,
            "selected_schema": self.selected_schema,
            "schema_source": self.source,
            "schema_path": list(self.path),
            "fallback_used": self.fallback_used,
        }


def _field(
    name: str,
    *aliases: str,
    required: bool = False,
    value_kind: str = "text",
    multiline: bool = False,
    allow_evidence_sharing_with: tuple[str, ...] = (),
) -> SchemaFieldSpec:
    return SchemaFieldSpec(
        name=name,
        display_name=name.replace("_", " ").title(),
        aliases=tuple(dict.fromkeys((name.replace("_", " "), *aliases))),
        required=required,
        value_kind=value_kind,
        multiline=multiline,
        allow_evidence_sharing_with=allow_evidence_sharing_with,
    )


ORGANISATION_FORM_FIELDS = (
    _field("document_title", "form title", "form name", "title", value_kind="title"),
    _field(
        "organisation_name",
        "organization name",
        "organisation",
        "organization",
        "company name",
        required=True,
    ),
    _field("address", "postal address", "office address", multiline=True),
    _field("contact_name", "contact person", "contact", required=True),
    _field(
        "phone_number",
        "phone",
        "phone no",
        "telephone",
        "telephone number",
        "tel",
        "contact number",
        value_kind="phone",
    ),
    _field(
        "fax_number",
        "fax",
        "fax no",
        "fax no.",
        "facsimile",
        "facsimile number",
        "telefax",
        value_kind="phone",
    ),
    _field("email", "email address", "e-mail", value_kind="email"),
    _field(
        "date",
        "dated",
        "document date",
        "form date",
        "request date",
        "transmission date",
        value_kind="date",
    ),
    _field("reference_number", "reference no", "ref no", value_kind="reference"),
    _field("subject", "re"),
    _field("message", "comments", "notes", multiline=True),
)


# A business-form classification is used for labelled operational request forms.
# Keep this schema distinct from the legacy generic ``form`` type so that an
# organisation-registration form is not incorrectly required to contain sample data.
BUSINESS_FORM_FIELDS = (
    _field(
        "date",
        "dated",
        "document date",
        "form date",
        "request date",
        required=True,
        value_kind="date",
    ),
    _field(
        "sample_number",
        "sample no",
        "sample no.",
        "sample number",
        "sample id",
        required=True,
        value_kind="reference",
    ),
    _field(
        "product_type",
        "type of cigarette",
        "product type",
        "item type",
        "sample type",
        required=True,
    ),
    _field(
        "batch_size",
        "quantity",
        "lot size",
        required=True,
        value_kind="quantity",
    ),
    _field(
        "requested_by",
        "original request made by",
        "requested by",
        "requestor",
        "requester",
        required=True,
        value_kind="name",
    ),
    _field(
        "specifications_written_by",
        "sample specifications written by",
        "specifications by",
        "prepared by",
        required=True,
        value_kind="name",
    ),
    _field(
        "document_title",
        "title",
        "document title",
        "form title",
        "heading",
        value_kind="title",
    ),
    _field(
        "organisation_name", "organization name", "organisation", "organization", "company name"
    ),
    _field("additional_notes", "additional notes", "notes", "comments", multiline=True),
    # Preserve useful generic fields without making them requirements of every
    # operational request form.
    _field("address", "postal address", "office address", multiline=True),
    _field("contact_name", "contact person", "contact", value_kind="name"),
    _field(
        "phone_number",
        "phone",
        "phone no",
        "telephone",
        "telephone number",
        "tel",
        "contact number",
        value_kind="phone",
    ),
    _field(
        "fax_number",
        "fax",
        "fax no",
        "fax no.",
        "facsimile",
        "facsimile number",
        "telefax",
        value_kind="phone",
    ),
    _field("email", "email address", "e-mail", value_kind="email"),
    _field("reference_number", "reference no", "ref no", value_kind="reference"),
    _field("subject", "re"),
)

FAX_FIELDS = (
    _field("document_title", "title", required=True, value_kind="title"),
    _field("date", "dated", "transmission date", required=True, value_kind="date"),
    _field("time", "transmission time", value_kind="time"),
    _field("sender_name", "from", "sender", "sent by", value_kind="name"),
    _field("sender_company", "sender company", "from company", value_kind="organisation"),
    _field("sender_phone", "sender phone", "from phone", "phone", "telephone", value_kind="phone"),
    _field(
        "sender_fax", "sender fax", "from fax", "fax", "fax number", "facsimile", value_kind="phone"
    ),
    _field(
        "recipient_name",
        "to",
        "message to",
        "recipient",
        "attention",
        "attn",
        required=True,
        value_kind="name",
    ),
    _field(
        "recipient_company",
        "company",
        "recipient company",
        "organisation",
        "organization",
        "firm",
        value_kind="organisation",
    ),
    _field(
        "recipient_phone",
        "phone",
        "phone number",
        "telephone",
        "telephone number",
        "tel",
        "contact number",
        value_kind="phone",
    ),
    _field(
        "recipient_fax",
        "fax number",
        "fax no",
        "fax no.",
        "fax",
        "facsimile",
        "facsimile number",
        "telefax",
        required=True,
        value_kind="phone",
    ),
    _field(
        "client_number",
        "client no",
        "client number",
        "reference no",
        "reference number",
        "ref no",
        value_kind="reference",
    ),
    _field("subject", "re"),
    _field("page_count", "pages", "number of pages", "page count", value_kind="integer"),
    _field("hard_copy_follow_up", "hard copy", "hardcopy", value_kind="boolean"),
    _field("message", "comments", "notes", multiline=True),
    _field("confidentiality_notice", "confidentiality", "confidential notice", multiline=True),
)

GENERIC_KEY_VALUE_FIELDS = (
    _field(
        "document_title", "form title", "document title", "heading", "title", value_kind="title"
    ),
    _field("organisation_name", "organization", "organisation"),
    _field("person_name", "name", "contact", value_kind="name"),
    _field("company_name", "company", "firm", value_kind="organisation"),
    _field("date", "dated", value_kind="date"),
    _field("reference_number", "reference no", "ref no", "client number", value_kind="reference"),
    _field("account_number", "account no", "acct no", value_kind="account"),
    _field("phone_number", "phone", "telephone", "tel", "contact number", value_kind="phone"),
    _field("fax_number", "fax", "fax no", "facsimile", "telefax", value_kind="phone"),
    _field("email", "email address", "e-mail", value_kind="email"),
    _field("address", "postal address", multiline=True),
    _field("amount", "total", "amount due", value_kind="money"),
    _field("subject", "re"),
    _field("message", "comments", "notes", multiline=True),
    _field("signature_name", "signed by", "signature", value_kind="name"),
)

GENERIC_METADATA_FIELDS = (
    _field("title", "document title", value_kind="title"),
    _field("visible_dates", value_kind="date_list"),
    _field("visible_names", value_kind="name_list"),
    _field("visible_organisations", value_kind="organisation_list"),
    _field("visible_phone_numbers", value_kind="phone_list"),
    _field("visible_fax_numbers", value_kind="phone_list"),
    _field("visible_emails", value_kind="email_list"),
    _field("visible_currency_amounts", value_kind="money_list"),
    _field("visible_reference_numbers", value_kind="reference_list"),
)

REPORT_FIELDS = (
    _field("document_title", "report title", "title", "heading", value_kind="title"),
    _field("report_date", "report date", "date", "dated", value_kind="date"),
    _field(
        "organisation_name", "organisation", "organization", "company", value_kind="organisation"
    ),
    _field("prepared_by", "prepared by", "written by", "author", value_kind="name"),
    _field(
        "reference_number", "reference number", "reference no", "ref no", value_kind="reference"
    ),
    _field("additional_notes", "summary", "notes", "comments", multiline=True),
)

CORRESPONDENCE_FIELDS = (
    _field("document_title", "subject", "title", "heading", value_kind="title"),
    _field("date", "dated", "letter date", value_kind="date"),
    _field("sender_name", "from", "sender", "sent by", value_kind="name"),
    _field("recipient_name", "to", "recipient", "attention", "attn", value_kind="name"),
    _field(
        "organisation_name", "company", "organisation", "organization", value_kind="organisation"
    ),
    _field(
        "reference_number", "reference number", "reference no", "ref no", value_kind="reference"
    ),
    _field("subject", "re"),
    _field("message", "body", "message", "comments", multiline=True),
)


SCHEMAS: dict[VisionDocumentType, DocumentSchema] = {
    VisionDocumentType.form: DocumentSchema(
        VisionDocumentType.form, ORGANISATION_FORM_FIELDS, "organisation_form"
    ),
    VisionDocumentType.business_form: DocumentSchema(
        VisionDocumentType.business_form, BUSINESS_FORM_FIELDS, "business_form"
    ),
    VisionDocumentType.fax_transmission: DocumentSchema(
        VisionDocumentType.fax_transmission, FAX_FIELDS, "fax_transmission"
    ),
    VisionDocumentType.invoice: DocumentSchema(
        VisionDocumentType.invoice,
        (
            _field(
                "vendor_name",
                "vendor",
                "supplier",
                "seller",
                required=True,
                value_kind="organisation",
            ),
            _field("vendor_address", "supplier address", "seller address", multiline=True),
            _field("vendor_tax_id", "tax id", "gstin", "vat", "tin", value_kind="reference"),
            _field(
                "invoice_number",
                "invoice no",
                "invoice #",
                "inv no",
                required=True,
                value_kind="reference",
            ),
            _field("invoice_date", "date of invoice", required=True, value_kind="date"),
            _field("due_date", "payment due", value_kind="date"),
            _field(
                "purchase_order_number",
                "purchase order",
                "po number",
                "po no",
                "po reference",
                value_kind="reference",
            ),
            _field("subtotal", "sub total", value_kind="money"),
            _field("tax", "tax amount", "gst", "vat amount", value_kind="money"),
            _field("discount", "discount amount", value_kind="money"),
            _field(
                "total_amount",
                "grand total",
                "amount due",
                "invoice total",
                required=True,
                value_kind="money",
                allow_evidence_sharing_with=("currency",),
            ),
            _field(
                "currency",
                "currency code",
                required=True,
                value_kind="currency",
                allow_evidence_sharing_with=("total_amount",),
            ),
            _field("payment_terms", "terms of payment", "payment term"),
            _field("bank_name", "beneficiary bank", value_kind="organisation"),
            _field(
                "bank_account_number",
                "account number",
                "account no",
                "bank account",
                "acct no",
                value_kind="account",
            ),
            _field(
                "bank_code",
                "ifsc",
                "swift",
                "bic",
                "routing number",
                "sort code",
                value_kind="reference",
            ),
        ),
        "invoice",
    ),
    VisionDocumentType.purchase_order: DocumentSchema(
        VisionDocumentType.purchase_order,
        (
            _field(
                "purchase_order_number",
                "purchase order",
                "po number",
                "po no",
                required=True,
                value_kind="reference",
            ),
            _field(
                "buyer_name",
                "buyer",
                "customer",
                "bill to",
                required=True,
                value_kind="organisation",
            ),
            _field(
                "vendor_name",
                "vendor",
                "supplier",
                "seller",
                required=True,
                value_kind="organisation",
            ),
            _field("order_date", "po date", "purchase date", required=True, value_kind="date"),
            _field("delivery_date", "required date", "ship date", value_kind="date"),
            _field("line_items", "items", "description", multiline=True),
            _field("subtotal", "sub total", value_kind="money"),
            _field("tax", "tax amount", value_kind="money"),
            _field(
                "total_amount",
                "grand total",
                "po total",
                required=True,
                value_kind="money",
                allow_evidence_sharing_with=("currency",),
            ),
            _field(
                "currency",
                "currency code",
                required=True,
                value_kind="currency",
                allow_evidence_sharing_with=("total_amount",),
            ),
            _field("approval_status", "status", "approved"),
        ),
        "purchase_order",
    ),
    VisionDocumentType.contract: DocumentSchema(
        VisionDocumentType.contract,
        (
            _field("parties", "contracting parties", "between", required=True, multiline=True),
            _field("agreement_date", "date of agreement", value_kind="date"),
            _field("effective_date", "effective as of", required=True, value_kind="date"),
            _field("expiry_date", "expiration date", "end date", value_kind="date"),
            _field("governing_law", "governed by"),
            _field("renewal_term", "renewal", "automatic renewal"),
            _field("termination_clause", "termination", "terminate", multiline=True),
            _field("payment_terms", "payment shall be due", required=True),
            _field("liability_clause", "liability", "limitation of liability", multiline=True),
            _field("confidentiality_clause", "confidentiality", "non-disclosure", multiline=True),
            _field(
                "indemnification_clause",
                "indemnification",
                "indemnity",
                "hold harmless",
                multiline=True,
            ),
            _field("signatures", "signed by", "signature", required=True),
        ),
        "contract",
    ),
    VisionDocumentType.bank_document: DocumentSchema(
        VisionDocumentType.bank_document,
        (
            _field(
                "account_holder",
                "account holder name",
                "beneficiary",
                required=True,
                value_kind="name",
            ),
            _field(
                "bank_name", "bank", "beneficiary bank", required=True, value_kind="organisation"
            ),
            _field(
                "account_number",
                "bank account number",
                "account no",
                "acct no",
                required=True,
                value_kind="account",
            ),
            _field(
                "bank_code",
                "ifsc",
                "swift",
                "bic",
                "routing number",
                required=True,
                value_kind="reference",
            ),
            _field("branch", "branch name", "bank branch"),
            _field("effective_date", "effective from", value_kind="date"),
            _field(
                "authorised_signer",
                "authorized signer",
                "signed by",
                "authorised by",
                value_kind="name",
            ),
        ),
        "bank_document",
    ),
    VisionDocumentType.bank_confirmation: DocumentSchema(
        VisionDocumentType.bank_confirmation,
        (
            _field("document_title", "title", required=True),
            _field(
                "account_holder",
                "account holder",
                "customer name",
                required=True,
                value_kind="name",
            ),
            _field("bank_name", "bank", required=True, value_kind="organisation"),
            _field(
                "account_number",
                "account no",
                "bank account number",
                required=True,
                value_kind="account",
            ),
            _field("bank_code", "swift", "bic", "routing number", "ifsc", value_kind="reference"),
            _field("confirmation_date", "date", "confirmed on", value_kind="date"),
            _field("authorised_signer", "authorized signer", "signed by", value_kind="name"),
        ),
        "bank_confirmation",
    ),
    VisionDocumentType.receipt: DocumentSchema(
        VisionDocumentType.receipt,
        (
            _field("merchant_name", "merchant", "store", value_kind="organisation"),
            _field("receipt_number", "receipt no", "transaction no", value_kind="reference"),
            _field("date", "transaction date", value_kind="date"),
            _field(
                "total_amount",
                "total",
                "amount paid",
                required=True,
                value_kind="money",
                allow_evidence_sharing_with=("currency",),
            ),
            _field(
                "currency",
                "currency code",
                value_kind="currency",
                allow_evidence_sharing_with=("total_amount",),
            ),
        ),
        "receipt",
    ),
    VisionDocumentType.report: DocumentSchema(
        VisionDocumentType.report,
        REPORT_FIELDS,
        "report",
    ),
    VisionDocumentType.general_report: DocumentSchema(
        VisionDocumentType.general_report,
        REPORT_FIELDS,
        "report",
    ),
    VisionDocumentType.financial_report: DocumentSchema(
        VisionDocumentType.financial_report,
        REPORT_FIELDS,
        "report",
    ),
    VisionDocumentType.correspondence: DocumentSchema(
        VisionDocumentType.correspondence,
        CORRESPONDENCE_FIELDS,
        "correspondence",
    ),
    VisionDocumentType.letter: DocumentSchema(
        VisionDocumentType.letter,
        CORRESPONDENCE_FIELDS,
        "correspondence",
    ),
    VisionDocumentType.insurance_document: DocumentSchema(
        VisionDocumentType.insurance_document,
        (
            _field("document_title", "title"),
            _field("policy_number", "policy no", required=True, value_kind="reference"),
            _field("insured_name", "insured", "policy holder", value_kind="name"),
            _field("insurer_name", "insurer", "insurance company", value_kind="organisation"),
            _field("effective_date", "policy start", value_kind="date"),
            _field("expiry_date", "policy end", value_kind="date"),
        ),
        "insurance_document",
    ),
}

GENERIC_SCHEMAS: dict[str, DocumentSchema] = {
    "generic_key_value_form": DocumentSchema(
        None, GENERIC_KEY_VALUE_FIELDS, "generic_key_value_form", "generic_key_value"
    ),
    "generic_document": DocumentSchema(
        None, GENERIC_METADATA_FIELDS, "generic_document", "generic_metadata"
    ),
}

PARENT_SCHEMA_BY_TYPE: dict[VisionDocumentType, str] = {
    VisionDocumentType.fax_transmission: "business_form",
    VisionDocumentType.form: "business_form",
    VisionDocumentType.business_form: "business_form",
    VisionDocumentType.legal_form: "generic_key_value_form",
}


class VisionSchemaRegistry:
    def get(self, document_type: VisionDocumentType | str) -> DocumentSchema | None:
        resolved = self._document_type(document_type)
        return SCHEMAS.get(resolved) if resolved else None

    def get_by_id(self, schema_id: str) -> DocumentSchema | None:
        if schema_id in GENERIC_SCHEMAS:
            return GENERIC_SCHEMAS[schema_id]
        return next((item for item in SCHEMAS.values() if item.name == schema_id), None)

    def resolve(
        self,
        document_type: VisionDocumentType | str,
        *,
        prefer_key_value: bool = False,
    ) -> SchemaSelection:
        resolved = self._document_type(document_type) or VisionDocumentType.unknown
        exact = SCHEMAS.get(resolved)
        parent_id = PARENT_SCHEMA_BY_TYPE.get(resolved)
        parent = self.get_by_id(parent_id) if parent_id else None
        generic_id = "generic_key_value_form" if prefer_key_value else "generic_document"
        generic = GENERIC_SCHEMAS[generic_id]
        selected = exact or parent or generic
        source = "exact" if exact else "parent" if parent else generic.schema_kind
        path = tuple(
            dict.fromkeys(
                item
                for item in (
                    resolved.value if exact or resolved != VisionDocumentType.unknown else None,
                    parent_id,
                    "generic_key_value_form",
                    "generic_document",
                )
                if item
            )
        )
        return SchemaSelection(
            schema=selected,
            exact_schema=exact.name if exact else None,
            fallback_schema=parent_id,
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
        kind = spec.value_kind if spec else "text"
        if kind in {"text", "organisation", "name_list", "organisation_list"}:
            return (any(character.isalnum() for character in cleaned), "contains_visible_text")
        if kind == "name":
            date_like = re.search(
                r"(?i)\b(?:date|dated|january|february|march|april|may|june|july|"
                r"august|september|october|november|december)\b.*\d",
                cleaned,
            )
            valid = bool(re.search(r"[A-Za-z]", cleaned)) and not date_like
            return valid, "name_text_preserved" if valid else "name_is_date_or_missing"
        if kind == "title":
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
            valid = (
                bool(re.search(r"[A-Za-z]", cleaned))
                and labelled_value is None
                and date_line is None
            )
            return valid, "title_like_text" if valid else "labelled_value_is_not_a_title"
        if kind in {"email", "email_list"}:
            valid = re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", cleaned) is not None
            return valid, "valid_email" if valid else "invalid_email_format"
        if kind in {"phone", "phone_list"}:
            digits = re.sub(r"\D", "", cleaned)
            valid = 7 <= len(digits) <= 15
            return valid, "valid_phone_length" if valid else "phone_digit_count_out_of_range"
        if kind in {"money", "money_list"}:
            valid = re.search(r"-?\d[\d,]*(?:\.\d+)?", cleaned) is not None
            return valid, "numeric_amount_present" if valid else "amount_not_parseable"
        if kind == "currency":
            valid = (
                re.fullmatch(r"(?:USD|INR|EUR|GBP|AUD|CAD|JPY|\$|€|£|₹)", cleaned, re.I) is not None
            )
            return valid, "known_currency" if valid else "unknown_currency"
        if kind in {"date", "date_list"}:
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
            return (
                (True, "date_uncertain_needs_review")
                if date_like
                else (False, "date_not_parseable")
            )
        if kind == "time":
            valid = re.search(r"\b\d{1,2}:\d{2}(?:\s*[AP]M)?\b", cleaned, re.I) is not None
            return valid, "valid_time" if valid else "time_not_parseable"
        if kind == "integer":
            valid = re.fullmatch(r"\d{1,5}", cleaned) is not None
            return valid, "valid_integer" if valid else "integer_not_parseable"
        if kind == "boolean":
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
        if kind == "quantity":
            valid = (
                re.fullmatch(
                    r"(?i)\s*\d+(?:[.,]\d+)?\s*(?:lbs?\.?|pounds?|kg|kgs|g|grams?|"
                    r"mg|oz|ounces?|litres?|liters?|ml|units?|pcs?\.?)\s*",
                    cleaned,
                )
                is not None
            )
            return valid, "number_and_unit_present" if valid else "quantity_not_parseable"
        if kind in {"reference", "reference_list", "account"}:
            valid = bool(re.search(r"[A-Za-z0-9]", cleaned)) and len(cleaned) <= 100
            return valid, "valid_identifier" if valid else "identifier_not_parseable"
        return True, "no_special_validation"

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
