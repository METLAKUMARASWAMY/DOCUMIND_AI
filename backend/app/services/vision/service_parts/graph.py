"""Neo4j-backed relationship retrieval and graph projection.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.security import hash_document, utc_now_iso
from app.models.vision import (
    ReviewerStatus,
    VisionAskCitation,
    VisionAskResponse,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionGraphProjection,
)
from app.services.vision.graph_builder import evidence_graph_builder


class GraphMixin:
    def _answer_relationship(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
        *,
        source_fields: set[str] | None = None,
    ) -> VisionAskResponse:
        neo4j_paths = self.persistence.query_graph_paths(
            tenant_id,
            question,
            [document.document_id for document in documents],
            limit=top_k,
            # What the corpus holds decides what can be asked for; deriving the
            # candidates from the registry alone left open-set fields
            # unreachable.
            present_relationships=self._relationships_in_scope(documents),
        )
        matches = self._neo4j_graph_matches(neo4j_paths, documents)
        graph_backend = "neo4j-evidence-path-v1" if matches else "local-semantic-graph-path-v1"
        if not matches:
            matches = self._matching_graph_edges(question, documents)
        if source_fields:
            matches = [
                item for item in matches if str(item[1].get("source_field") or "") in source_fields
            ]
        matches = matches[:top_k]
        if not matches:
            return self._not_found_answer(requires_review=False)
        citations = [
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=int(edge.get("source_page", 1)),
                source_text=str(edge.get("evidence_text", "")),
                bounding_boxes=[
                    VisionBoundingBox.model_validate(box) for box in edge.get("bounding_boxes", [])
                ],
                evidence_id=edge.get("evidence_id"),
                retrieval_score=float(edge.get("confidence", 0.0) or 0.0),
                retrieval_backend=graph_backend,
                model_confidence=float(edge.get("confidence", 0.0) or 0.0),
                model_name=str(edge.get("model_name", "deterministic-graph-builder")),
                model_version=str(edge.get("model_version", "v1")),
                processing_timestamp=document.updated_at,
                reviewer_status=self._graph_reviewer_status(edge),
                extraction_status=str(edge.get("status", "proposed")),
                evidence_type="graph_path",
            )
            for document, edge, _, _ in matches
        ]
        paths = [
            {
                "path_id": f"graph-path-{edge.get('id')}",
                "document_id": document.document_id,
                "node_ids": [str(edge.get("source")), str(edge.get("target"))],
                "relationships": [str(edge.get("type"))],
                "display_path": (
                    f"{source_node.get('label', edge.get('source'))} → "
                    f"{str(edge.get('type', '')).replace('_', ' ')} → "
                    f"{target_node.get('label', edge.get('target'))}"
                ),
                "explanation": self._graph_path_explanation(
                    str(edge.get("type", "")),
                    str(source_node.get("label", edge.get("source", ""))),
                    str(target_node.get("label", edge.get("target", ""))),
                ),
                "verification_status": str(edge.get("status", "proposed")),
                "evidence_id": edge.get("evidence_id"),
                "source_page": int(edge.get("source_page", 1)),
                "source_text": str(edge.get("evidence_text", "")),
                "source_field": edge.get("source_field"),
            }
            for document, edge, source_node, target_node in matches
        ]
        answer = "; ".join(path["explanation"] for path in paths)
        return VisionAskResponse(
            answer=answer,
            answer_status=(
                "supported"
                if all(edge.get("status") == "verified" for _, edge, _, _ in matches)
                else "partially_supported"
            ),
            citations=citations,
            confidence=round(
                sum(float(edge.get("confidence", 0.0) or 0.0) for _, edge, _, _ in matches)
                / len(matches),
                4,
            ),
            reasoning_summary=(
                "Traversed semantic graph relationships and returned each path with its "
                "originating field or OCR evidence."
            ),
            requires_review=any(edge.get("status") != "verified" for _, edge, _, _ in matches),
            retriever=graph_backend,
            retrieval_path="graph_path_query",
            graph_paths=paths,
        )

    @staticmethod
    def _relationships_in_scope(documents: list[VisionDocumentRecord]) -> list[str]:
        """Every edge type present on the documents the question is scoped to."""
        seen: set[str] = set()
        for document in documents:
            graph = getattr(document, "graph", None)
            for edge in getattr(graph, "edges", None) or []:
                relationship = str((edge or {}).get("type") or "").strip()
                if relationship:
                    seen.add(relationship)
        return sorted(seen)

    def _neo4j_graph_matches(
        self,
        paths: list[dict[str, Any]],
        documents: list[VisionDocumentRecord],
    ) -> list[
        tuple[
            VisionDocumentRecord,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]
    ]:
        by_document = {document.document_id: document for document in documents}
        matches = []
        for path in paths:
            document = by_document.get(str(path.get("document_id") or ""))
            if document is None:
                continue
            properties = dict(path.get("relationship_properties") or {})
            bounding_boxes = self._neo4j_json_value(
                properties.get("bounding_boxes"),
                [],
            )
            bounding_box = self._neo4j_json_value(
                properties.get("bounding_box"),
                None,
            )
            if not bounding_boxes and isinstance(bounding_box, dict):
                bounding_boxes = [bounding_box]
            relationship = str(path.get("relationship_type") or "")
            edge = {
                **properties,
                "id": properties.get("edge_id") or f"neo4j:{document.document_id}:{relationship}",
                "source": document.document_id,
                "target": str(path.get("target_id") or ""),
                "type": relationship,
                "source_page": int(properties.get("source_page") or 1),
                "evidence_text": str(properties.get("evidence_text") or ""),
                "status": str(properties.get("status") or "proposed"),
                "confidence": float(properties.get("confidence") or 0.0),
                "model_name": "neo4j-evidence-path",
                "model_version": "v1",
                "bounding_boxes": (bounding_boxes if isinstance(bounding_boxes, list) else []),
            }
            source_node = {
                "id": document.document_id,
                "label": str(path.get("source_label") or document.filename),
            }
            target_properties = dict(path.get("target_properties") or {})
            target_node = {
                **target_properties,
                "id": str(path.get("target_id") or ""),
                "label": str(
                    path.get("target_label")
                    or target_properties.get("display_label")
                    or path.get("target_id")
                    or "related entity"
                ),
            }
            matches.append((document, edge, source_node, target_node))
        return matches

    def _neo4j_json_value(self, value: object, default: object) -> object:
        if not isinstance(value, str):
            return value if value is not None else default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def _matching_graph_edges(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> list[
        tuple[
            VisionDocumentRecord,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]
    ]:
        lower = question.lower()
        intents: set[str] = set()
        intent_terms = {
            "REQUESTED_BY": ("who requested", "original request", "requested by", "requester"),
            "SPECIFICATIONS_WRITTEN_BY": (
                "who wrote",
                "sample specifications",
                "specifications written by",
            ),
            "HAS_SAMPLE_NUMBER": ("sample number", "sample no", "sample id"),
            "HAS_PRODUCT_TYPE": ("product type", "type of product", "type of cigarette"),
            "HAS_BATCH_SIZE": ("batch size", "lot size", "quantity"),
            "HAS_DATE": ("what date", "document date", "form date"),
            "HAS_REQUEST_DATE": ("request date", "when was the request", "request made"),
            "HAS_PURPOSE": ("purpose", "what was the purpose"),
            "HAS_FAX": ("fax number", "fax no", "facsimile", "telefax"),
            "ISSUED_BY": ("issued by", "vendor", "supplier"),
            "ISSUED_TO": ("issued to", "buyer", "customer"),
            "SIGNED_BY": ("signed by", "who signed"),
            "REFERENCES": ("references", "linked", "related"),
            "HAS_BANK_ACCOUNT": ("bank account", "paid to"),
            "HAS_RISK": ("risk", "suspicious", "warning"),
        }
        for relationship, terms in intent_terms.items():
            if any(term in lower for term in terms):
                intents.add(relationship)
        stopwords = {
            "are",
            "does",
            "document",
            "for",
            "from",
            "how",
            "is",
            "of",
            "the",
            "this",
            "to",
            "what",
            "which",
            "who",
            "with",
        }
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", lower)
            if len(token) >= 3 and token not in stopwords
        }
        ranked: list[
            tuple[
                int,
                VisionDocumentRecord,
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
            ]
        ] = []
        for document in documents:
            nodes = {str(node.get("id")): node for node in document.graph.nodes}
            for edge in document.graph.edges:
                relationship = str(edge.get("type", ""))
                if relationship in {"HAS_FIELD", "MATCHES", "CONTAINS_PAGE", "CONTAINS"}:
                    continue
                source_node = nodes.get(str(edge.get("source")), {})
                target_node = nodes.get(str(edge.get("target")), {})
                haystack = " ".join(
                    (
                        relationship,
                        str(source_node.get("label", "")),
                        str(target_node.get("label", "")),
                        str(edge.get("evidence_text", "")),
                        str(edge.get("source_field", "")),
                    )
                ).lower()
                edge_tokens = set(re.findall(r"[a-z0-9]+", haystack))
                if intents and relationship not in intents:
                    continue
                overlap = len(tokens & edge_tokens)
                if not intents and not overlap:
                    continue
                score = (100 if relationship in intents else 0) + overlap * 5
                score += 3 if edge.get("status") == "verified" else 0
                ranked.append((score, document, edge, source_node, target_node))
        ranked.sort(
            key=lambda item: (item[0], float(item[2].get("confidence", 0.0) or 0.0)),
            reverse=True,
        )
        return [(document, edge, source, target) for _, document, edge, source, target in ranked]

    def _graph_reviewer_status(self, edge: dict[str, Any]) -> ReviewerStatus:
        try:
            return ReviewerStatus(str(edge.get("review_status", "unverified")))
        except ValueError:
            return (
                ReviewerStatus.accepted
                if edge.get("status") == "verified"
                else ReviewerStatus.unverified
            )

    def _graph_path_explanation(
        self,
        relationship: str,
        source_label: str,
        target_label: str,
    ) -> str:
        templates = {
            "REQUESTED_BY": f"{target_label} is recorded as the person who requested the sample",
            "SPECIFICATIONS_WRITTEN_BY": (
                f"{target_label} is recorded as the person who wrote the sample specifications"
            ),
            "HAS_SAMPLE_NUMBER": f"{source_label} has sample number {target_label}",
            "HAS_PRODUCT_TYPE": f"{source_label} identifies the product as {target_label}",
            "HAS_BATCH_SIZE": f"{source_label} records a batch size of {target_label}",
            "HAS_DATE": f"{source_label} records the date as {target_label}",
            "HAS_REQUEST_DATE": f"{source_label} records the request date as {target_label}",
            "HAS_PURPOSE": f"{source_label} records the purpose as {target_label}",
            "ISSUED_BY": f"{source_label} was issued by {target_label}",
            "ISSUED_TO": f"{source_label} was issued to {target_label}",
            "SIGNED_BY": f"{source_label} is linked to signer {target_label}",
            "HAS_BANK_ACCOUNT": f"{source_label} is associated with bank account {target_label}",
            "HAS_RISK": f"{source_label} has the source-linked risk indicator {target_label}",
        }
        return templates.get(
            relationship,
            f"{source_label} is connected to {target_label} by {relationship.replace('_', ' ').lower()}",
        )

    def verify_graph_relationship(
        self,
        relationship_id: str,
        *,
        tenant_id: str,
        actor: str,
        comment: str,
        evidence_acknowledged: bool,
    ) -> dict[str, Any]:
        if not evidence_acknowledged:
            raise ValueError("Relationship evidence must be acknowledged before verification")
        if len(comment.strip()) < 10:
            raise ValueError("A reviewer comment of at least 10 characters is required")

        document: VisionDocumentRecord | None = None
        relationship: dict[str, Any] | None = None
        for candidate in self.list_documents(tenant_id):
            relationship = next(
                (edge for edge in candidate.graph.edges if edge.get("id") == relationship_id),
                None,
            )
            if relationship:
                document = candidate
                break
        if document is None or relationship is None:
            raise ValueError("graph relationship not found")
        if not relationship.get("source_linked") or not relationship.get("evidence_text"):
            raise ValueError("A relationship without source evidence cannot be verified")
        if not relationship.get("verification_eligible"):
            raise ValueError("Relationship is not eligible for manual verification")

        now = utc_now_iso()
        relationship.update(
            {
                "status": "verified",
                "review_status": ReviewerStatus.accepted.value,
                "verification_eligible": False,
                "verified_by": actor,
                "verified_at": now,
                "verification_comment": comment.strip(),
            }
        )
        document.updated_at = now
        self._audit(
            document,
            "graph_relationship_verified",
            "A reviewer verified a source-linked graph relationship.",
            actor=actor,
            metadata={
                "relationship_id": relationship_id,
                "relationship_type": relationship.get("type"),
                "evidence_id": relationship.get("evidence_id"),
                "comment": comment.strip(),
            },
        )
        self.persistence.upsert_graph(
            document.tenant_id,
            document.document_id,
            document.graph,
        )
        self._persist_document(document)
        return relationship

    def _graph(self, document: VisionDocumentRecord) -> VisionGraphProjection:
        return evidence_graph_builder.build(document)

    def _legacy_graph(self, document: VisionDocumentRecord) -> VisionGraphProjection:
        nodes: dict[str, dict[str, Any]] = {
            document.document_id: {
                "id": document.document_id,
                "label": document.filename,
                "type": "document",
                "tenant_id": document.tenant_id,
                "status": document.status.value,
            }
        }
        edges: list[dict[str, Any]] = []
        fields = {field.name: field for field in document.extracted_fields}

        def add_field_edge(
            field_name: str,
            node_type: str,
            relationship: str,
            *,
            reverse: bool = False,
            masked: bool = False,
        ) -> str | None:
            field = fields.get(field_name)
            if not field:
                return None
            value = field.reviewed_value or field.value
            node_value = self._mask_sensitive(field_name, value) if masked else value
            node_id = (
                f"{node_type}:{hash_document(value.encode('utf-8'))[:16]}"
                if masked
                else f"{node_type}:{self._normalize(value)}"
            )
            nodes[node_id] = {
                "id": node_id,
                "label": node_value,
                "type": node_type,
                "review_status": field.reviewer_status.value,
                "tenant_id": document.tenant_id,
            }
            source, target = (
                (node_id, document.document_id) if reverse else (document.document_id, node_id)
            )
            verified = field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
            edges.append(
                {
                    "id": f"edge:{document.document_id}:{relationship}:{node_id}",
                    "source": source,
                    "target": target,
                    "type": relationship,
                    "status": "verified" if verified else "proposed",
                    "source_document": document.document_id,
                    "source_page": field.evidence.page_number,
                    "evidence_id": field.evidence.evidence_id,
                    "evidence_text": field.evidence.source_text,
                    "bounding_boxes": [field.evidence.bounding_box.model_dump(mode="json")]
                    if field.evidence.bounding_box
                    else [],
                    "confidence": field.confidence,
                    "review_status": field.reviewer_status.value,
                    "model_name": field.model_name,
                    "model_version": field.model_version,
                    "deterministic": True,
                }
            )
            return node_id

        vendor_id = add_field_edge("vendor_name", "vendor", "SUBMITTED", reverse=True)
        add_field_edge("invoice_number", "invoice", "REFERENCES")
        add_field_edge("purchase_order_number", "purchase_order", "REFERENCES")
        add_field_edge("contract_number", "contract", "GOVERNED_BY")
        bank_id = add_field_edge("bank_account_number", "bank_account", "PAID_TO", masked=True)
        if bank_id is None:
            bank_id = add_field_edge("IFSC_or_bank_code", "bank_account", "PAID_TO", masked=True)
        tax_id = add_field_edge("vendor_tax_id", "tax_identifier", "REGISTERED_WITH", masked=True)
        add_field_edge("signature_name", "approver", "APPROVED_BY")
        add_field_edge("party_name", "party", "REFERENCES")
        if vendor_id and bank_id:
            evidence_field = fields.get("bank_account_number") or fields.get("IFSC_or_bank_code")
            if evidence_field:
                edges.append(
                    {
                        "id": f"edge:{vendor_id}:PAID_TO:{bank_id}",
                        "source": vendor_id,
                        "target": bank_id,
                        "type": "PAID_TO",
                        "status": "verified"
                        if evidence_field.reviewer_status
                        in {ReviewerStatus.accepted, ReviewerStatus.edited}
                        else "proposed",
                        "source_document": document.document_id,
                        "source_page": evidence_field.evidence.page_number,
                        "evidence_id": evidence_field.evidence.evidence_id,
                        "evidence_text": evidence_field.evidence.source_text,
                        "bounding_boxes": [
                            evidence_field.evidence.bounding_box.model_dump(mode="json")
                        ]
                        if evidence_field.evidence.bounding_box
                        else [],
                        "confidence": evidence_field.confidence,
                        "review_status": evidence_field.reviewer_status.value,
                        "model_name": evidence_field.model_name,
                        "model_version": evidence_field.model_version,
                        "deterministic": True,
                    }
                )
        if vendor_id and tax_id:
            evidence_field = fields["vendor_tax_id"]
            edges.append(
                {
                    "id": f"edge:{vendor_id}:REGISTERED_WITH:{tax_id}",
                    "source": vendor_id,
                    "target": tax_id,
                    "type": "REGISTERED_WITH",
                    "status": "verified"
                    if evidence_field.reviewer_status
                    in {ReviewerStatus.accepted, ReviewerStatus.edited}
                    else "proposed",
                    "source_document": document.document_id,
                    "source_page": evidence_field.evidence.page_number,
                    "evidence_id": evidence_field.evidence.evidence_id,
                    "evidence_text": evidence_field.evidence.source_text,
                    "bounding_boxes": [evidence_field.evidence.bounding_box.model_dump(mode="json")]
                    if evidence_field.evidence.bounding_box
                    else [],
                    "confidence": evidence_field.confidence,
                    "review_status": evidence_field.reviewer_status.value,
                    "model_name": evidence_field.model_name,
                    "model_version": evidence_field.model_version,
                    "deterministic": True,
                }
            )

        for duplicate in document.duplicate_candidates:
            if not duplicate.matched_document_id:
                continue
            nodes.setdefault(
                duplicate.matched_document_id,
                {
                    "id": duplicate.matched_document_id,
                    "label": duplicate.matched_filename or duplicate.matched_document_id,
                    "type": "document",
                    "tenant_id": document.tenant_id,
                },
            )
            evidence = duplicate.evidence[0] if duplicate.evidence else None
            relationship = "DUPLICATE_OF" if duplicate.status != "related_document" else "MATCHES"
            edges.append(
                {
                    "id": f"edge:{document.document_id}:{relationship}:{duplicate.matched_document_id}",
                    "source": document.document_id,
                    "target": duplicate.matched_document_id,
                    "type": relationship,
                    "status": "proposed",
                    "source_document": document.document_id,
                    "source_page": evidence.page_number if evidence else 1,
                    "evidence_id": evidence.evidence_id if evidence else None,
                    "evidence_text": evidence.source_text if evidence else duplicate.reason,
                    "bounding_boxes": [evidence.bounding_box.model_dump(mode="json")]
                    if evidence and evidence.bounding_box
                    else [],
                    "confidence": duplicate.similarity,
                    "review_status": ReviewerStatus.unverified.value,
                    "model_name": evidence.model_name if evidence else "duplicate-detector",
                    "model_version": evidence.model_version if evidence else "v1",
                    "deterministic": True,
                }
            )

        for risk in document.risk_indicators:
            if not risk.rule_id.startswith("cross_document."):
                continue
            target_id = f"trusted_reference:{self._normalize(risk.title)}"
            nodes[target_id] = {
                "id": target_id,
                "label": risk.title,
                "type": "trusted_reference",
                "tenant_id": document.tenant_id,
            }
            evidence = risk.evidence[0] if risk.evidence else None
            edges.append(
                {
                    "id": f"edge:{document.document_id}:CONFLICTS_WITH:{target_id}",
                    "source": document.document_id,
                    "target": target_id,
                    "type": "CONFLICTS_WITH",
                    "status": "proposed",
                    "source_document": document.document_id,
                    "source_page": evidence.page_number if evidence else 1,
                    "evidence_id": evidence.evidence_id if evidence else None,
                    "evidence_text": evidence.source_text if evidence else risk.title,
                    "bounding_boxes": [evidence.bounding_box.model_dump(mode="json")]
                    if evidence and evidence.bounding_box
                    else [],
                    "confidence": evidence.model_confidence if evidence else 1.0,
                    "review_status": risk.reviewer_status.value,
                    "model_name": risk.model_name,
                    "model_version": risk.model_version,
                    "deterministic": True,
                }
            )
        return VisionGraphProjection(nodes=list(nodes.values()), edges=edges)
