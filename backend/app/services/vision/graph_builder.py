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
            schema_version="documind-evidence-graph-v2",
            preferred_backend="neo4j",
            active_backend="local_evidence_graph",
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
                "CONTAINS",
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
            )
            page_id = f"Page:{self.document.document_id}:{field.evidence.page_number}"
            self._add_edge(
                page_id,
                field_node,
                "HAS_FIELD",
                evidence=field.evidence,
                status=self._status_for(field),
                provenance=field.extractor,
            )
            entity = self._entity_for_field(field)
            if entity is None:
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
            )

    def _add_entity_relationships(self, document_node: str) -> None:
        organisation = self._first_entity("organisation_name")
        vendor = self._first_entity("vendor_name")
        customer = self._first_entity("buyer_name", "customer_name")
        person = self._first_entity(
            "contact_name",
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
        contacts = self._entities("phone_number", "fax_number", "email")

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
            self._field_relationship(
                organisation or vendor or customer or person or document_node,
                contact,
                "HAS_CONTACT",
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
            if detection.detection_class != "signature":
                continue
            signature_id = f"Signature:{self.document.document_id}:{detection.detection_id}"
            self._add_node(
                signature_id,
                "Signature",
                "Signature candidate",
                value="signature",
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
            )
            status = (
                "verified"
                if detection.reviewer_status in VERIFIED_REVIEW_STATUSES
                else "proposed"
            )
            self._add_edge(
                document_node,
                signature_id,
                "CONTAINS",
                status=status,
                provenance="visual_detection",
                source_page=detection.page_number,
                evidence_id=detection.detection_id,
                evidence_text="Visual signature candidate",
                bounding_boxes=[detection.bounding_box.model_dump(mode="json")],
                confidence=detection.confidence,
                review_status=detection.reviewer_status,
            )
            if signer:
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
            relationship = (
                "MATCHES" if duplicate.status == "related_document" else "DUPLICATE_OF"
            )
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
            "vendor_name": "Vendor",
            "buyer_name": "Customer",
            "customer_name": "Customer",
            "contact_name": "Person",
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
            "email": "ContactMethod",
            "payment_reference": "Payment",
            "total_amount": "Payment",
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
        stable_value = f"{self.document.tenant_id}:{node_type}:{self._normalize(raw_value)}"
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
        )

    def _add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        **properties: Any,
    ) -> str:
        cleaned_label = " ".join(str(label).split()) or node_type
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
            "source_document": self.document.document_id,
            "document_id": self.document.document_id,
            "source_page": source_page or 1,
            "evidence_id": evidence_id,
            "evidence_text": evidence_text or "Document structure",
            "bounding_boxes": bounding_boxes or [],
            "confidence": confidence,
            "review_status": (
                review_status.value if isinstance(review_status, ReviewerStatus) else str(review_status or "unverified")
            ),
            "provenance": provenance,
            "deterministic": True,
            "source_linked": bool(evidence_id or evidence_text),
            "verification_eligible": status == "proposed" and not structural,
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
        )

    def _first_entity(self, *field_names: str) -> str | None:
        return next(
            (self.entities_by_field[name] for name in field_names if name in self.entities_by_field),
            None,
        )

    def _entities(self, *field_names: str) -> Iterable[tuple[str, str]]:
        return (
            (name, self.entities_by_field[name])
            for name in field_names
            if name in self.entities_by_field
        )

    def _status_for(self, item: ExtractedVisionField | Any) -> str:
        return (
            "verified"
            if item.reviewer_status in VERIFIED_REVIEW_STATUSES
            else "proposed"
        )

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
