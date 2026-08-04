from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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


@dataclass(frozen=True)
class DocumentSchema:
    document_type: VisionDocumentType
    fields: tuple[SchemaFieldSpec, ...]

    @property
    def required_fields(self) -> tuple[SchemaFieldSpec, ...]:
        return tuple(item for item in self.fields if item.required)

    @property
    def optional_fields(self) -> tuple[SchemaFieldSpec, ...]:
        return tuple(item for item in self.fields if not item.required)

    def field(self, name: str) -> SchemaFieldSpec | None:
        normalized = name.strip().lower()
        return next((item for item in self.fields if item.name == normalized), None)


def _field(
    name: str,
    *aliases: str,
    required: bool = False,
    value_kind: str = "text",
) -> SchemaFieldSpec:
    return SchemaFieldSpec(
        name=name,
        display_name=name.replace("_", " ").title(),
        aliases=tuple(dict.fromkeys((name.replace("_", " "), *aliases))),
        required=required,
        value_kind=value_kind,
    )


SCHEMAS: dict[VisionDocumentType, DocumentSchema] = {
    VisionDocumentType.form: DocumentSchema(
        VisionDocumentType.form,
        (
            _field("organisation_name", "organization name", "organisation", "organization", "company name", required=True),
            _field("address", "postal address", "office address"),
            _field("contact_name", "contact person", "contact", required=True),
            _field("phone_number", "phone", "telephone", "tel", value_kind="phone"),
            _field("fax_number", "fax", "fax no", "fax number", value_kind="phone"),
            _field("email", "email address", "e-mail", value_kind="email"),
            _field("form_title", "form name", "title", required=True),
        ),
    ),
    VisionDocumentType.invoice: DocumentSchema(
        VisionDocumentType.invoice,
        (
            _field("vendor_name", "vendor", "supplier", "seller", required=True),
            _field("vendor_address", "supplier address", "seller address"),
            _field("vendor_tax_id", "tax id", "gstin", "vat", "tin"),
            _field("invoice_number", "invoice no", "invoice #", "inv no", required=True),
            _field("invoice_date", "date of invoice", required=True, value_kind="date"),
            _field("due_date", "payment due", value_kind="date"),
            _field("purchase_order_number", "purchase order", "po number", "po no", "po reference"),
            _field("subtotal", "sub total", value_kind="money"),
            _field("tax", "tax amount", "gst", "vat amount", value_kind="money"),
            _field("discount", "discount amount", value_kind="money"),
            _field("total_amount", "grand total", "amount due", "invoice total", required=True, value_kind="money"),
            _field("currency", "currency code", required=True, value_kind="currency"),
            _field("payment_terms", "terms of payment", "payment term"),
            _field("bank_name", "beneficiary bank"),
            _field("bank_account_number", "account number", "account no", "bank account", "acct no"),
            _field("bank_code", "ifsc", "swift", "bic", "routing number", "sort code"),
        ),
    ),
    VisionDocumentType.purchase_order: DocumentSchema(
        VisionDocumentType.purchase_order,
        (
            _field("purchase_order_number", "purchase order", "po number", "po no", required=True),
            _field("buyer_name", "buyer", "customer", "bill to", required=True),
            _field("vendor_name", "vendor", "supplier", "seller", required=True),
            _field("order_date", "po date", "purchase date", required=True, value_kind="date"),
            _field("delivery_date", "required date", "ship date", value_kind="date"),
            _field("line_items", "items", "description"),
            _field("subtotal", "sub total", value_kind="money"),
            _field("tax", "tax amount", value_kind="money"),
            _field("total_amount", "grand total", "po total", required=True, value_kind="money"),
            _field("currency", "currency code", required=True, value_kind="currency"),
            _field("approval_status", "status", "approved"),
        ),
    ),
    VisionDocumentType.contract: DocumentSchema(
        VisionDocumentType.contract,
        (
            _field("parties", "contracting parties", "between", required=True),
            _field("agreement_date", "date of agreement", value_kind="date"),
            _field("effective_date", "effective as of", required=True, value_kind="date"),
            _field("expiry_date", "expiration date", "end date", value_kind="date"),
            _field("governing_law", "governed by"),
            _field("renewal_term", "renewal", "automatic renewal"),
            _field("termination_clause", "termination", "terminate"),
            _field("payment_terms", "payment shall be due", required=True),
            _field("liability_clause", "liability", "limitation of liability"),
            _field("confidentiality_clause", "confidentiality", "non-disclosure"),
            _field("indemnification_clause", "indemnification", "indemnity", "hold harmless"),
            _field("signatures", "signed by", "signature", required=True),
        ),
    ),
    VisionDocumentType.bank_document: DocumentSchema(
        VisionDocumentType.bank_document,
        (
            _field("account_holder", "account holder name", "beneficiary", required=True),
            _field("bank_name", "bank", "beneficiary bank", required=True),
            _field("account_number", "bank account number", "account no", "acct no", required=True),
            _field("bank_code", "ifsc", "swift", "bic", "routing number", required=True),
            _field("branch", "branch name", "bank branch"),
            _field("effective_date", "effective from", value_kind="date"),
            _field("authorised_signer", "authorized signer", "signed by", "authorised by"),
        ),
    ),
}


class VisionSchemaRegistry:
    def get(self, document_type: VisionDocumentType | str) -> DocumentSchema | None:
        try:
            resolved = (
                document_type
                if isinstance(document_type, VisionDocumentType)
                else VisionDocumentType(str(document_type))
            )
        except ValueError:
            return None
        return SCHEMAS.get(resolved)

    def summarize(
        self,
        document_type: VisionDocumentType,
        fields: Iterable[ExtractedVisionField],
    ) -> VisionFieldCompleteness:
        schema = self.get(document_type)
        field_list = list(fields)
        if schema is None:
            return VisionFieldCompleteness(
                schema_available=False,
                document_type=document_type,
                extracted_fields=sorted({item.name for item in field_list}),
                extracted_count=len(field_list),
            )

        by_name = {item.name: item for item in field_list if (item.reviewed_value or item.value).strip()}
        required = [item.name for item in schema.required_fields]
        optional = [item.name for item in schema.optional_fields]
        resolved = {
            name
            for name, item in by_name.items()
            if item.reviewer_status != ReviewerStatus.rejected
        }
        invalid = sorted(
            name
            for name, item in by_name.items()
            if not self.value_is_valid(schema.field(name), item.reviewed_value or item.value)
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
        missing = sorted(set(required) - resolved | (set(required) & set(invalid)))
        return VisionFieldCompleteness(
            schema_available=True,
            document_type=document_type,
            required_fields=required,
            optional_fields=optional,
            extracted_fields=sorted(by_name),
            missing_required_fields=missing,
            invalid_fields=invalid,
            disputed_fields=disputed,
            manually_resolved_fields=manually_resolved,
            expected_required_count=len(required),
            extracted_count=len(field_list),
            unresolved_required_count=len(missing),
        )

    def value_is_valid(self, spec: SchemaFieldSpec | None, value: str) -> bool:
        import re

        cleaned = value.strip()
        if not cleaned:
            return False
        if spec is None or spec.value_kind == "text":
            return any(character.isalnum() for character in cleaned)
        if spec.value_kind == "email":
            return re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", cleaned) is not None
        if spec.value_kind == "phone":
            return len(re.sub(r"\D", "", cleaned)) >= 7
        if spec.value_kind == "money":
            return re.search(r"-?\d[\d,]*(?:\.\d+)?", cleaned) is not None
        if spec.value_kind == "currency":
            return re.fullmatch(r"(?:USD|INR|EUR|GBP|AUD|CAD|JPY|\$|€|£|₹)", cleaned, re.I) is not None
        if spec.value_kind == "date":
            return len(cleaned) >= 6 and any(character.isdigit() for character in cleaned)
        return True


vision_schema_registry = VisionSchemaRegistry()
