from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from app.core.security import hash_document
from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionDocumentRecord,
    VisionEvidence,
    VisionGraphProjection,
    VisionSemanticType,
)

VERIFIED_REVIEW_STATUSES = {ReviewerStatus.accepted, ReviewerStatus.edited}


class EvidenceGraphBuilder:
    """Build a deterministic, source-linked graph projection for one document."""

    def build(self, document: VisionDocumentRecord) -> VisionGraphProjection:
        self.document = document
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.fields = {field.name: field for field in document.extracted_fields}
        self.entities_by_field: dict[str, str] = {}

        document_node = self._add_node(
            document.document_id,
            "Document",
            document.filename,
            value=document.filename,
            status=document.status.value.lower(),
            review_status=(
                document.review_task.status.value.lower()
                if document.review_task
                else document.status.value.lower()
            ),
            source_page=1,
        )
        self._add_pages(document_node)
        self._add_fields()
        self._add_generic_attributes(document_node)
        self._add_entity_relationships(document_node)
        self._add_visual_signatures(document_node)
        self._add_risks(document_node)
        self._add_duplicates(document_node)

        return VisionGraphProjection(
            nodes=list(self.nodes.values()),
            edges=list(self.edges.values()),
            evidence_policy=(
                "Relationships are deterministic and source-linked. Unreviewed evidence "
                "remains proposed until a reviewer verifies it."
            ),
            schema_version="documind-evidence-graph-v4",
            preferred_backend="neo4j",
            active_backend="local_evidence_graph",
            legend={
                "node_types": [
                    "Document",
                    "Organisation",
                    "Person",
                    "Product",
                    "Form",
                    "ContactMethod",
                    "Date",
                    "Amount",
                    "Identifier",
                    "BankAccount",
                    "Signature",
                    "Stamp",
                    "ExtractedField",
                    "Attribute",
                    "RiskIndicator",
                ],
                "relationships": {
                    "verified": "green solid",
                    "proposed": "orange dashed",
                    "conflict": "red",
                },
            },
        )

    def _add_generic_attributes(self, document_node: str) -> None:
        """Keep every discovered label/value pair graph-accessible.

        Recognised schema fields continue to receive richer relationships below;
        unseen labels use a stable generic Attribute node instead of disappearing.
        """
        schema_evidence = {
            field.evidence.evidence_id
            for field in self.document.extracted_fields
            if field.extractor != "generic-open-set"
        }
        schema_names = {
            field.name
            for field in self.document.extracted_fields
            if field.extractor != "generic-open-set"
        }
        for field in self.document.generic_fields:
            if (
                field.evidence.evidence_id in schema_evidence
                or field.normalized_label in schema_names
            ):
                continue
            value = self._mask(field.normalized_label, field.raw_value)
            identity = (
                f"{self.document.tenant_id}:Attribute:{field.normalized_label}:"
                f"{self._normalize(field.normalized_value)}"
            )
            attribute_id = f"Attribute:{hash_document(identity.encode())[:18]}"
            self._add_node(
                attribute_id,
                "Attribute",
                f"{field.raw_label}: {value}",
                value=value,
                status="proposed" if field.review_required else "machine_observed",
                review_status=ReviewerStatus.unverified.value,
                source_page=field.page,
                evidence_id=field.evidence.evidence_id,
                bounding_boxes=self._boxes(field.evidence),
                confidence=field.confidence,
                field_name=field.normalized_label,
                source_text=field.evidence.source_text,
                extractor_version=field.source_engine,
                semantic_type=(
                    field.semantic_type_if_known.value
                    if isinstance(field.semantic_type_if_known, VisionSemanticType)
                    else None
                ),
            )
            self._add_edge(
                document_node,
                attribute_id,
                "HAS_ATTRIBUTE",
                evidence=field.evidence,
                status="proposed",
                provenance=field.source_engine,
                source_field=field.normalized_label,
            )

    def _add_pages(self, document_node: str) -> None:
        page_count = max(
            self.document.page_count,
            max((artifact.page_number for artifact in self.document.page_artifacts), default=0),
            1,
        )
        artifacts = {item.page_number: item for item in self.document.page_artifacts}
        for page_number in range(1, page_count + 1):
            artifact = artifacts.get(page_number)
            page_id = f"Page:{self.document.document_id}:{page_number}"
            self._add_node(
                page_id,
                "Page",
                f"Page {page_number}",
                value=str(page_number),
                status="verified",
                review_status=ReviewerStatus.accepted.value,
                source_page=page_number,
                width=artifact.width if artifact else None,
                height=artifact.height if artifact else None,
            )
            self._add_edge(
                document_node,
                page_id,
                "CONTAINS_PAGE",
                status="verified",
                provenance="document_structure",
                source_page=page_number,
                evidence_id=f"page:{self.document.document_id}:{page_number}",
                evidence_text=f"{self.document.filename}, page {page_number}",
                confidence=1.0,
                review_status=ReviewerStatus.accepted,
                structural=True,
            )

    def _add_fields(self) -> None:
        for field in self.document.extracted_fields:
            value = field.reviewed_value or field.value
            shown_value = self._mask(field.name, value)
            field_node = self._add_node(
                f"ExtractedField:{self.document.document_id}:{field.field_id}",
                "ExtractedField",
                f"{field.display_name}: {shown_value}",
                value=shown_value,
                status=self._status_for(field),
                review_status=field.reviewer_status.value,
                source_page=field.evidence.page_number,
                evidence_id=field.evidence.evidence_id,
                bounding_boxes=self._boxes(field.evidence),
                field_name=field.name,
                required=field.required,
                confidence=field.confidence,
                source_field_id=field.field_id,
                source_text=field.evidence.source_text,
                extractor_version=field.extractor,
            )
            page_id = f"Page:{self.document.document_id}:{field.evidence.page_number}"
            self._add_edge(
                page_id,
                field_node,
                "HAS_FIELD",
                evidence=field.evidence,
                status=self._status_for(field),
                provenance=field.extractor,
                source_field=field.name,
            )
            entity = self._entity_for_field(field)
            if entity is None or not self._promotable_enterprise_entity(field):
                continue
            node_type, label = entity
            entity_node = self._add_entity_node(node_type, label, field)
            self.entities_by_field[field.name] = entity_node
            self._add_edge(
                entity_node,
                field_node,
                "MATCHES",
                evidence=field.evidence,
                status=self._status_for(field),
                provenance="deterministic_field_mapping",
                source_field=field.name,
            )

    def _add_entity_relationships(self, document_node: str) -> None:
        organisation = self._first_entity(
            "organisation_name",
            "company_name",
            "recipient_company",
            "sender_company",
        )
        vendor = self._first_entity("vendor_name")
        customer = self._first_entity("buyer_name", "customer_name")
        person = self._first_entity(
            "contact_name",
            "person_name",
            "recipient_name",
            "sender_name",
            "authorised_signer",
            "authorized_signer",
            "signature_name",
        )
        invoice = self._first_entity("invoice_number")
        purchase_order = self._first_entity("purchase_order_number")
        contract = self._first_entity("contract_number", "parties")
        bank = self._first_entity("bank_account_number", "account_number", "IFSC_or_bank_code")
        tax = self._first_entity("vendor_tax_id", "tax_identifier")
        address = self._first_entity("address", "vendor_address")
        payment = self._first_entity("payment_reference", "total_amount")
        contacts = self._entities(
            "phone_number",
            "fax_number",
            "recipient_phone",
            "sender_phone",
            "recipient_fax",
            "sender_fax",
            "visible_phone_numbers",
            "visible_fax_numbers",
            "email",
        )
        title = self._first_entity("document_title", "form_title", "title")
        date = self._first_entity(
            "date",
            "visible_dates",
            "invoice_date",
            "effective_date",
        )
        recipient = self._first_entity("recipient_name")
        sender = self._first_entity("sender_name")
        recipient_company = self._first_entity("recipient_company")
        client_number = self._first_entity("client_number", "reference_number")
        sample_number = self._first_entity("sample_number")
        product_type = self._first_entity("product_type")
        batch_size = self._first_entity("batch_size")
        requested_by = self._first_entity("requested_by")
        specifications_written_by = self._first_entity("specifications_written_by")
        request_date = self._first_entity("request_date")
        purpose = self._first_entity("purpose")

        if title:
            self._field_relationship(
                document_node,
                title,
                "HAS_TITLE",
                "document_title",
                "form_title",
                "title",
            )
        if date:
            self._field_relationship(
                document_node,
                date,
                "HAS_DATE",
                "date",
                "visible_dates",
                "invoice_date",
                "effective_date",
            )
        if recipient:
            self._field_relationship(
                document_node,
                recipient,
                "SENT_TO",
                "recipient_name",
            )
        if sender:
            self._field_relationship(
                sender,
                document_node,
                "SENT",
                "sender_name",
            )
        if recipient_company:
            self._field_relationship(
                document_node,
                recipient_company,
                "ASSOCIATED_WITH",
                "recipient_company",
            )
        if client_number:
            self._field_relationship(
                document_node,
                client_number,
                "HAS_CLIENT_NUMBER",
                "client_number",
                "reference_number",
            )
        if sample_number:
            self._field_relationship(
                document_node,
                sample_number,
                "HAS_SAMPLE_NUMBER",
                "sample_number",
            )
        if product_type:
            self._field_relationship(
                document_node,
                product_type,
                "HAS_PRODUCT_TYPE",
                "product_type",
            )
        if batch_size:
            self._field_relationship(
                document_node,
                batch_size,
                "HAS_BATCH_SIZE",
                "batch_size",
            )
        if requested_by:
            self._field_relationship(
                document_node,
                requested_by,
                "REQUESTED_BY",
                "requested_by",
            )
        if specifications_written_by:
            self._field_relationship(
                document_node,
                specifications_written_by,
                "SPECIFICATIONS_WRITTEN_BY",
                "specifications_written_by",
            )
        if request_date:
            self._field_relationship(
                document_node,
                request_date,
                "HAS_REQUEST_DATE",
                "request_date",
            )
        if purpose:
            self._field_relationship(
                document_node,
                purpose,
                "HAS_PURPOSE",
                "purpose",
            )

        if organisation:
            self._field_relationship(
                organisation,
                document_node,
                "SUBMITTED",
                "organisation_name",
            )
        if vendor:
            self._field_relationship(document_node, vendor, "ISSUED_BY", "vendor_name")
        if customer:
            self._field_relationship(
                document_node,
                customer,
                "ISSUED_TO",
                "buyer_name",
                "customer_name",
            )
        if invoice:
            self._field_relationship(document_node, invoice, "REFERENCES", "invoice_number")
        if purchase_order:
            self._field_relationship(
                invoice or document_node,
                purchase_order,
                "REFERENCES",
                "purchase_order_number",
            )
        if contract:
            self._field_relationship(
                document_node,
                contract,
                "GOVERNED_BY",
                "contract_number",
                "parties",
            )
        if bank:
            self._field_relationship(
                invoice or document_node,
                bank,
                "PAID_TO",
                "bank_account_number",
                "account_number",
                "IFSC_or_bank_code",
            )
        if payment:
            self._field_relationship(
                invoice or document_node,
                payment,
                "CONTAINS",
                "payment_reference",
                "total_amount",
            )

        owner = organisation or vendor or customer or person or document_node
        if address:
            self._field_relationship(
                owner,
                address,
                "LOCATED_AT",
                "address",
                "vendor_address",
            )
        if tax:
            self._field_relationship(
                vendor or organisation or document_node,
                tax,
                "REGISTERED_WITH",
                "vendor_tax_id",
                "tax_identifier",
            )
        if person and organisation and person != organisation:
            self._field_relationship(
                organisation,
                person,
                "HAS_CONTACT",
                "contact_name",
            )
        for field_name, contact in contacts:
            relationship = (
                "HAS_FAX"
                if "fax" in field_name
                else "HAS_PHONE"
                if "phone" in field_name
                else "HAS_CONTACT"
            )
            self._field_relationship(
                organisation or vendor or customer or person or document_node,
                contact,
                relationship,
                field_name,
            )

    def _add_visual_signatures(self, document_node: str) -> None:
        signer = self._first_entity(
            "signature_name",
            "authorised_signer",
            "authorized_signer",
            "contact_name",
        )
        for detection in self.document.visual_detections:
            if detection.detection_class not in {"signature", "stamp"}:
                continue
            node_type = "Signature" if detection.detection_class == "signature" else "Stamp"
            signature_id = f"{node_type}:{self.document.document_id}:{detection.detection_id}"
            self._add_node(
                signature_id,
                node_type,
                f"{node_type} candidate",
                value=detection.detection_class,
                status=(
                    "verified"
                    if detection.reviewer_status in VERIFIED_REVIEW_STATUSES
                    else "proposed"
                ),
                review_status=detection.reviewer_status.value,
                source_page=detection.page_number,
                evidence_id=detection.detection_id,
                bounding_boxes=[detection.bounding_box.model_dump(mode="json")],
                confidence=detection.confidence,
                source_text=f"Visual {detection.detection_class} candidate",
                extractor_version=detection.model_version,
            )
            status = (
                "verified" if detection.reviewer_status in VERIFIED_REVIEW_STATUSES else "proposed"
            )
            self._add_edge(
                document_node,
                signature_id,
                "HAS_FIELD",
                status=status,
                provenance="visual_detection",
                source_page=detection.page_number,
                evidence_id=detection.detection_id,
                evidence_text=f"Visual {detection.detection_class} candidate",
                bounding_boxes=[detection.bounding_box.model_dump(mode="json")],
                confidence=detection.confidence,
                review_status=detection.reviewer_status,
            )
            if signer and detection.detection_class == "signature":
                self._add_edge(
                    signature_id,
                    signer,
                    "SIGNED_BY",
                    status=status,
                    provenance="deterministic_signature_association",
                    source_page=detection.page_number,
                    evidence_id=detection.detection_id,
                    evidence_text="Visual signature candidate",
                    bounding_boxes=[detection.bounding_box.model_dump(mode="json")],
                    confidence=detection.confidence,
                    review_status=detection.reviewer_status,
                )

    def _add_risks(self, document_node: str) -> None:
        for risk in self.document.risk_indicators:
            evidence = risk.evidence[0] if risk.evidence else None
            risk_node = self._add_node(
                f"RiskIndicator:{self.document.document_id}:{risk.risk_id}",
                "RiskIndicator",
                risk.title,
                value=risk.description,
                status=self._status_for(risk),
                review_status=risk.reviewer_status.value,
                source_page=evidence.page_number if evidence else 1,
                evidence_id=evidence.evidence_id if evidence else None,
                bounding_boxes=self._boxes(evidence),
                severity=risk.severity.value,
                category=risk.category,
                confidence=risk.confidence,
            )
            if evidence:
                self._add_edge(
                    document_node,
                    risk_node,
                    "HAS_RISK",
                    evidence=evidence,
                    status=self._status_for(risk),
                    provenance=risk.rule_id,
                )

    def _add_duplicates(self, document_node: str) -> None:
        for duplicate in self.document.duplicate_candidates:
            if not duplicate.matched_document_id:
                continue
            target = self._add_node(
                duplicate.matched_document_id,
                "Document",
                duplicate.matched_filename or duplicate.matched_document_id,
                value=duplicate.matched_filename or duplicate.matched_document_id,
                status="proposed",
                review_status=ReviewerStatus.unverified.value,
                source_page=1,
            )
            evidence = duplicate.evidence[0] if duplicate.evidence else None
            relationship = "MATCHES" if duplicate.status == "related_document" else "DUPLICATE_OF"
            self._add_edge(
                document_node,
                target,
                relationship,
                evidence=evidence,
                status="proposed",
                provenance="deterministic_duplicate_detection",
                source_page=evidence.page_number if evidence else 1,
                evidence_id=evidence.evidence_id if evidence else duplicate.duplicate_id,
                evidence_text=evidence.source_text if evidence else duplicate.reason,
                bounding_boxes=self._boxes(evidence),
                confidence=duplicate.similarity,
                review_status=ReviewerStatus.unverified,
            )

    def _entity_for_field(self, field: ExtractedVisionField) -> tuple[str, str] | None:
        value = field.reviewed_value or field.value
        mapping = {
            "organisation_name": "Organisation",
            "company_name": "Organisation",
            "recipient_company": "Organisation",
            "sender_company": "Organisation",
            "vendor_name": "Vendor",
            "buyer_name": "Customer",
            "customer_name": "Customer",
            "contact_name": "Person",
            "person_name": "Person",
            "recipient_name": "Person",
            "sender_name": "Person",
            "authorised_signer": "Person",
            "authorized_signer": "Person",
            "signature_name": "Person",
            "invoice_number": "Invoice",
            "purchase_order_number": "PurchaseOrder",
            "contract_number": "Contract",
            "parties": "Contract",
            "bank_account_number": "BankAccount",
            "account_number": "BankAccount",
            "IFSC_or_bank_code": "BankAccount",
            "vendor_tax_id": "TaxIdentifier",
            "tax_identifier": "TaxIdentifier",
            "address": "Address",
            "vendor_address": "Address",
            "phone_number": "ContactMethod",
            "fax_number": "ContactMethod",
            "recipient_phone": "ContactMethod",
            "sender_phone": "ContactMethod",
            "recipient_fax": "ContactMethod",
            "sender_fax": "ContactMethod",
            "visible_phone_numbers": "ContactMethod",
            "visible_fax_numbers": "ContactMethod",
            "email": "ContactMethod",
            "document_title": "Form",
            "form_title": "Form",
            "title": "Form",
            "date": "Date",
            "visible_dates": "Date",
            "invoice_date": "Date",
            "effective_date": "Date",
            "request_date": "Date",
            "client_number": "Identifier",
            "reference_number": "Identifier",
            "sample_number": "Identifier",
            "product_type": "Product",
            "batch_size": "Amount",
            "requested_by": "Person",
            "specifications_written_by": "Person",
            "purpose": "Identifier",
            "payment_reference": "Identifier",
            "total_amount": "Amount",
            "amount": "Amount",
            "signatures": "Signature",
        }
        node_type = mapping.get(field.name)
        if node_type is None:
            return None
        if node_type in {"BankAccount", "TaxIdentifier"}:
            value = self._mask(field.name, value)
        if node_type == "ContactMethod":
            value = f"{field.display_name}: {value}"
        return node_type, value

    def _add_entity_node(
        self,
        node_type: str,
        label: str,
        field: ExtractedVisionField,
    ) -> str:
        raw_value = field.reviewed_value or field.value
        stable_value, resolution = self._entity_identity(node_type, field, raw_value)
        node_id = f"{node_type}:{hash_document(stable_value.encode())[:18]}"
        return self._add_node(
            node_id,
            node_type,
            label,
            value=label,
            status=self._status_for(field),
            review_status=field.reviewer_status.value,
            source_page=field.evidence.page_number,
            evidence_id=field.evidence.evidence_id,
            bounding_boxes=self._boxes(field.evidence),
            confidence=field.confidence,
            field_name=field.name,
            source_field_id=field.field_id,
            source_text=field.evidence.source_text,
            extractor_version=field.extractor,
            entity_resolution_status=resolution.get("status"),
            entity_resolution_confidence=resolution.get("confidence"),
            entity_resolution_method=resolution.get("method"),
            canonical_entity_id=resolution.get("canonical_entity_id"),
            observation_entity_id=resolution.get("observation_entity_id"),
            automatic_entity_merge=resolution.get("automatic_merge", False),
            resolution_evidence_ids=resolution.get("evidence_ids", []),
        )

    def _entity_identity(
        self,
        node_type: str,
        field: ExtractedVisionField,
        raw_value: str,
    ) -> tuple[str, dict[str, Any]]:
        """Use canonical IDs only after the conservative resolver approved a merge."""
        identity_types = {
            "Organisation": {VisionSemanticType.organisation},
            "Vendor": {VisionSemanticType.organisation},
            "Customer": {VisionSemanticType.organisation},
            "Person": {VisionSemanticType.person},
            "Product": {VisionSemanticType.product},
            "Address": {VisionSemanticType.address},
            "ContactMethod": {
                VisionSemanticType.email,
                VisionSemanticType.phone,
                VisionSemanticType.url,
            },
            "BankAccount": {VisionSemanticType.bank_account},
        }
        expected_types = identity_types.get(node_type, set())
        normalized_value = self._normalize(raw_value)
        candidates = [
            entity
            for entity in self.document.typed_entities
            if entity.semantic_type in expected_types
            and (
                field.evidence.evidence_id in entity.evidence_ids
                or self._normalize(entity.normalized_value or entity.raw_value) == normalized_value
            )
        ]
        entity = max(candidates, key=lambda item: item.confidence, default=None)
        resolution_payload = entity.context.get("entity_resolution") if entity is not None else None
        context = dict(resolution_payload) if isinstance(resolution_payload, dict) else {}
        canonical_id = str(context.get("canonical_entity_id") or "").strip()
        automatic_merge = context.get("automatic_merge") is True
        if automatic_merge and canonical_id:
            stable = f"{self.document.tenant_id}:{node_type}:{canonical_id}"
            return stable, {
                "status": "resolved",
                "confidence": context.get("resolution_confidence"),
                "method": ",".join(context.get("resolution_methods") or []),
                "canonical_entity_id": canonical_id,
                "observation_entity_id": context.get("observation_entity_id"),
                "automatic_merge": True,
                "evidence_ids": list(entity.evidence_ids) if entity else [],
            }

        if expected_types:
            stable = (
                f"{self.document.tenant_id}:{self.document.document_id}:"
                f"{node_type}:{normalized_value}"
            )
            statuses = list(context.get("decision_statuses") or [])
            return stable, {
                "status": statuses[0] if len(statuses) == 1 else "unresolved",
                "confidence": context.get("resolution_confidence"),
                "method": ",".join(context.get("resolution_methods") or [])
                or "document_scoped_unresolved_entity",
                "canonical_entity_id": None,
                "observation_entity_id": context.get("observation_entity_id"),
                "automatic_merge": False,
                "evidence_ids": list(entity.evidence_ids) if entity else [],
            }

        # Dates, amounts, and document-reference value nodes retain their existing
        # semantic-value identity; they do not assert that two people or businesses
        # are the same real-world entity.
        stable = f"{self.document.tenant_id}:{node_type}:{normalized_value}"
        return stable, {
            "status": "semantic_value",
            "method": "exact_normalized_semantic_value",
            "automatic_merge": False,
            "evidence_ids": [],
        }

    def _promotable_enterprise_entity(self, field: ExtractedVisionField) -> bool:
        """Keep uncertain fields visible without turning them into trusted business facts."""
        source = " ".join(field.evidence.source_text.lower().split())
        if field.name in {"document_title", "form_title", "title"} and re.match(
            r"^(?:date|bate|dated)\s*[:#=-]",
            source,
        ):
            return False
        if (
            field.validation_status == "invalid"
            and field.reviewer_status not in VERIFIED_REVIEW_STATUSES
        ):
            return False
        if (
            field.reviewer_status not in VERIFIED_REVIEW_STATUSES
            and (field.confidence or 0.0) < 0.55
        ):
            return False
        return True

    def _add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        **properties: Any,
    ) -> str:
        cleaned_label = " ".join(str(label).split()) or node_type
        boxes = list(properties.get("bounding_boxes") or [])
        status = str(properties.get("status") or "proposed")
        verification_status = (
            status
            if status in {"verified", "proposed", "rejected", "conflict"}
            else "verified"
            if node_type in {"Document", "Page"}
            else "proposed"
        )
        self.nodes.setdefault(
            node_id,
            {
                "id": node_id,
                "type": node_type,
                "label": cleaned_label,
                "display_label": self._truncate(cleaned_label),
                "tenant_id": self.document.tenant_id,
                "document_id": self.document.document_id,
                "source_document": self.document.document_id,
                "source_route": f"/documents/{self.document.document_id}",
                "evidence_count": 1 if properties.get("evidence_id") else 0,
                "source_page": int(properties.get("source_page") or 1),
                "source_field_id": properties.get("source_field_id"),
                "source_text": str(properties.get("source_text") or cleaned_label),
                "bounding_box": boxes,
                "bounding_boxes": boxes,
                "confidence": properties.get("confidence"),
                "verification_status": verification_status,
                "extractor_version": str(
                    properties.get("extractor_version") or "document-structure-v1"
                ),
                "pipeline_version": str(self.document.metadata.get("pipeline_version", "legacy")),
                **{key: value for key, value in properties.items() if value is not None},
            },
        )
        return node_id

    def _add_edge(
        self,
        source: str,
        target: str,
        relationship: str,
        *,
        evidence: VisionEvidence | None = None,
        status: str,
        provenance: str,
        source_page: int | None = None,
        evidence_id: str | None = None,
        evidence_text: str | None = None,
        bounding_boxes: list[dict[str, float]] | None = None,
        confidence: float | None = None,
        review_status: ReviewerStatus | None = None,
        structural: bool = False,
        source_field: str | None = None,
        source_field_id: str | None = None,
    ) -> str | None:
        if source not in self.nodes or target not in self.nodes:
            return None
        if evidence:
            source_page = evidence.page_number
            evidence_id = evidence.evidence_id
            evidence_text = evidence.source_text
            bounding_boxes = self._boxes(evidence)
            confidence = evidence.model_confidence
            review_status = evidence.reviewer_status
        if source_field and source_field in self.fields:
            source_field_id = source_field_id or self.fields[source_field].field_id
        if not structural and not evidence_text:
            return None
        identity = (
            f"{self.document.document_id}|{source}|{relationship}|{target}|"
            f"{evidence_id or evidence_text or 'structural'}"
        )
        edge_id = f"Relationship:{hash_document(identity.encode())[:20]}"
        self.edges[edge_id] = {
            "id": edge_id,
            "relationship_id": edge_id,
            "source": source,
            "target": target,
            "type": relationship,
            "label": relationship.replace("_", " "),
            "status": status,
            "verification_status": status,
            "tenant_id": self.document.tenant_id,
            "source_document": self.document.document_id,
            "document_id": self.document.document_id,
            "source_page": source_page or 1,
            "evidence_id": evidence_id,
            "evidence_text": evidence_text or "Document structure",
            "source_text": evidence_text or "Document structure",
            "bounding_boxes": bounding_boxes or [],
            "bounding_box": bounding_boxes or [],
            "confidence": confidence,
            "review_status": (
                review_status.value
                if isinstance(review_status, ReviewerStatus)
                else str(review_status or "unverified")
            ),
            "provenance": provenance,
            "extractor_version": (evidence.model_version if evidence else provenance),
            "pipeline_version": str(self.document.metadata.get("pipeline_version", "legacy")),
            "model_name": evidence.model_name if evidence else provenance,
            "model_version": evidence.model_version if evidence else "v1",
            "deterministic": True,
            "source_linked": bool(evidence_id or evidence_text),
            "verification_eligible": status == "proposed" and not structural,
            "source_field": source_field,
            "source_field_id": source_field_id,
        }
        return edge_id

    def _field_relationship(
        self,
        source: str,
        target: str,
        relationship: str,
        *field_names: str,
    ) -> str | None:
        fields = [self.fields[name] for name in field_names if name in self.fields]
        if not fields:
            return None
        evidence_field = fields[0]
        status = (
            "verified"
            if all(field.reviewer_status in VERIFIED_REVIEW_STATUSES for field in fields)
            else "proposed"
        )
        return self._add_edge(
            source,
            target,
            relationship,
            evidence=evidence_field.evidence,
            status=status,
            provenance="deterministic_entity_relationship",
            source_field=evidence_field.name,
        )

    def _first_entity(self, *field_names: str) -> str | None:
        return next(
            (
                self.entities_by_field[name]
                for name in field_names
                if name in self.entities_by_field
            ),
            None,
        )

    def _entities(self, *field_names: str) -> Iterable[tuple[str, str]]:
        return (
            (name, self.entities_by_field[name])
            for name in field_names
            if name in self.entities_by_field
        )

    def _status_for(self, item: ExtractedVisionField | Any) -> str:
        return "verified" if item.reviewer_status in VERIFIED_REVIEW_STATUSES else "proposed"

    def _boxes(self, evidence: VisionEvidence | None) -> list[dict[str, float]]:
        return (
            [evidence.bounding_box.model_dump(mode="json")]
            if evidence and evidence.bounding_box
            else []
        )

    def _mask(self, field_name: str, value: str) -> str:
        if field_name not in {
            "account_number",
            "bank_account_number",
            "IFSC_or_bank_code",
            "tax_identifier",
            "vendor_tax_id",
        }:
            return value
        compact = value.strip()
        if len(compact) <= 4:
            return "*" * len(compact)
        return f"{'*' * (len(compact) - 4)}{compact[-4:]}"

    def _normalize(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _truncate(self, value: str, limit: int = 30) -> str:
        return value if len(value) <= limit else f"{value[: limit - 3]}..."


evidence_graph_builder = EvidenceGraphBuilder()
