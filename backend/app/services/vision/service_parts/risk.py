"""Risk indicators: validation, tamper, contract, duplicate and anomaly signals.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.config import settings
from app.core.security import hash_document
from app.models.vision import (
    DuplicateCandidate,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidence,
    VisionRiskIndicator,
    VisionRiskSeverity,
)
from app.services.vision.duplicate_detection import (
    cosine_similarity,
    perceptual_similarity,
)
from app.services.vision.semantic_risk import (
    SemanticRiskFinding,
    generic_semantic_risk_engine,
)


class RiskMixin:
    def _generic_semantic_risks(
        self,
        document: VisionDocumentRecord,
        metadata: dict[str, Any],
    ) -> list[VisionRiskIndicator]:
        self._load_all_documents(document.tenant_id)
        corpus = [item for item in self.documents.values() if item.tenant_id == document.tenant_id]
        findings = generic_semantic_risk_engine.evaluate(document, metadata, corpus)
        return [self._semantic_risk_indicator(document, finding) for finding in findings]

    def _semantic_risk_indicator(
        self,
        document: VisionDocumentRecord,
        finding: SemanticRiskFinding,
    ) -> VisionRiskIndicator:
        indicator = self._risk(
            document,
            title=finding.title,
            severity=finding.severity,
            category=finding.category,
            description=finding.description,
            recommended_action=finding.recommended_action,
            source_text=finding.source_text,
            field_names=list(finding.affected_fields),
            evidence=list(finding.evidence),
            rule_id=finding.rule_id,
        )
        indicator.confidence = finding.confidence
        indicator.model_name = "generic-semantic-risk-engine"
        indicator.model_version = generic_semantic_risk_engine.model_version
        indicator.model_or_rule = finding.rule_id
        return indicator

    def _validation_risks(
        self,
        document: VisionDocumentRecord,
        metadata: dict[str, Any],
    ) -> list[VisionRiskIndicator]:
        risks: list[VisionRiskIndicator] = []
        fields = self._field_map(document)
        subtotal = self._money(fields.get("subtotal"))
        tax = self._money(fields.get("tax"))
        discount = self._money(fields.get("discount")) or 0.0
        total = self._money(fields.get("total_amount"))
        if subtotal is not None and tax is not None and total is not None:
            expected = subtotal + tax - discount
            tolerance = max(1.0, abs(total) * 0.01)
            if abs(expected - total) > tolerance:
                risks.append(
                    self._risk(
                        document,
                        title="Amount Arithmetic Mismatch",
                        severity=VisionRiskSeverity.high,
                        category="validation",
                        description="The document contains suspicious indicators and requires human review.",
                        recommended_action="Review subtotal, tax, discount, and total before approval.",
                        source_text=f"subtotal + tax - discount = {expected:.2f}; total = {total:.2f}",
                        field_names=["subtotal", "tax", "discount", "total_amount"],
                    )
                )
        invoice_date = self._date(fields.get("invoice_date"))
        due_date = self._date(fields.get("due_date"))
        if invoice_date and due_date and due_date < invoice_date:
            risks.append(
                self._risk(
                    document,
                    title="Due Date Before Invoice Date",
                    severity=VisionRiskSeverity.medium,
                    category="validation",
                    description="The document contains suspicious indicators and requires human review.",
                    recommended_action="Verify invoice and due dates.",
                    source_text=f"invoice_date={fields.get('invoice_date')}; due_date={fields.get('due_date')}",
                    field_names=["invoice_date", "due_date"],
                )
            )
        approved_bank = metadata.get("approved_bank_identifier") or metadata.get(
            "approved_bank_account"
        )
        bank_identifier = fields.get("bank_account_number") or fields.get("IFSC_or_bank_code")
        if (
            approved_bank
            and bank_identifier
            and self._normalize(approved_bank) != self._normalize(bank_identifier)
        ):
            risks.append(
                self._risk(
                    document,
                    title="Bank Detail Mismatch",
                    severity=VisionRiskSeverity.critical,
                    category="cross_document",
                    description="The document contains suspicious indicators and requires human review.",
                    recommended_action="Pause payment until vendor bank details are independently verified.",
                    source_text=f"document={self._mask_account(bank_identifier)}; approved={self._mask_account(str(approved_bank))}",
                    field_names=["bank_account_number", "IFSC_or_bank_code"],
                )
            )
        required_signature = metadata.get("required_signature") is True
        if required_signature and not any(
            item.detection_class == "signature" for item in document.visual_detections
        ):
            risks.append(
                self._risk(
                    document,
                    title="Required Signature Missing",
                    severity=VisionRiskSeverity.high,
                    category="visual_validation",
                    description="The document contains suspicious indicators and requires human review.",
                    recommended_action="Request a signed copy or reviewer confirmation.",
                    source_text="No signature visual cue detected by the local adapter.",
                )
            )
        tax_id = fields.get("vendor_tax_id")
        if tax_id and not re.fullmatch(r"[A-Z0-9-]{5,30}", tax_id.strip(), re.IGNORECASE):
            risks.append(
                self._risk(
                    document,
                    title="Vendor Tax ID Format Review",
                    severity=VisionRiskSeverity.medium,
                    category="format_validation",
                    description="The document contains suspicious indicators and requires human review.",
                    recommended_action="Verify vendor tax identifier format.",
                    source_text=tax_id,
                    field_names=["vendor_tax_id"],
                )
            )
        contract_terms = metadata.get("contract_payment_terms")
        payment_terms = fields.get("payment_terms")
        if (
            contract_terms
            and payment_terms
            and self._normalize(contract_terms) != self._normalize(payment_terms)
        ):
            risks.append(
                self._risk(
                    document,
                    title="Payment Terms Conflict",
                    severity=VisionRiskSeverity.medium,
                    category="cross_document",
                    description="The document contains suspicious indicators and requires human review.",
                    recommended_action="Compare invoice terms against the governing contract.",
                    source_text=f"document={payment_terms}; contract={contract_terms}",
                    field_names=["payment_terms"],
                )
            )
        return risks

    def _tamper_risks(self, document: VisionDocumentRecord) -> list[VisionRiskIndicator]:
        risks: list[VisionRiskIndicator] = []
        threshold = settings.tamper_screening_review_threshold
        for analysis in document.tamper_analysis:
            score = analysis.tampering_probability or analysis.screening_score
            regions = [
                region
                for region in analysis.suspicious_regions
                if region.screening_score >= threshold
            ]
            if score < threshold or not regions:
                continue
            evidence = [
                self._evidence(
                    document,
                    f"Possible visual inconsistency: {region.manipulation_class}",
                    analysis.model_name,
                    region.screening_score,
                    bounding_box=region.bounding_box,
                    page_number=analysis.page_number,
                    metadata={
                        "page_number": analysis.page_number,
                        "heatmap_path": analysis.heatmap_path,
                        "calibrated": analysis.calibrated,
                        "screening_only": True,
                    },
                )
                for region in regions[:6]
            ]
            risks.append(
                self._risk(
                    document,
                    title="Possible Visual Manipulation",
                    severity=VisionRiskSeverity.high,
                    category="tamper_screening",
                    description=(
                        "Measured visual inconsistencies crossed the configured screening threshold; "
                        "this is not a finding of forgery."
                    ),
                    recommended_action=(
                        "Inspect the heatmap and source regions, then obtain human forensic verification."
                    ),
                    source_text=analysis.explanation,
                    evidence=evidence,
                )
            )
        return risks

    def _contract_risks(self, document: VisionDocumentRecord) -> list[VisionRiskIndicator]:
        if document.document_type != VisionDocumentType.contract:
            return []
        fields = {field.name: field for field in document.extracted_fields}
        policies = {
            "uncapped_liability": (
                "Uncapped Liability Clause Review",
                VisionRiskSeverity.high,
                "Review liability scope, exceptions, and organisational policy with qualified counsel.",
            ),
            "perpetual_licence": (
                "Perpetual Licence Clause Review",
                VisionRiskSeverity.high,
                "Review licence duration, termination effects, and approved contracting policy.",
            ),
            "renewal_term": (
                "Renewal Clause Review",
                VisionRiskSeverity.medium,
                "Review renewal dates, notice windows, and ownership before the deadline.",
            ),
            "non_compete": (
                "Non-Compete Clause Review",
                VisionRiskSeverity.medium,
                "Review scope, duration, and applicable policy with qualified counsel.",
            ),
            "exclusivity": (
                "Exclusivity Clause Review",
                VisionRiskSeverity.medium,
                "Review exclusivity scope and commercial approval requirements.",
            ),
        }
        risks: list[VisionRiskIndicator] = []
        for field_name, (title, severity, action) in policies.items():
            field = fields.get(field_name)
            if not field:
                continue
            risks.append(
                self._risk(
                    document,
                    title=title,
                    severity=severity,
                    category="contract_clause_review",
                    description=(
                        "A contract clause matched a configured review category. "
                        "This is an evidence-location aid, not a legal judgement."
                    ),
                    recommended_action=action,
                    source_text=field.evidence.source_text,
                    field_names=[field_name],
                )
            )
        return risks

    def _duplicate_candidates(self, document: VisionDocumentRecord) -> list[DuplicateCandidate]:
        self._load_all_documents(document.tenant_id)
        candidates: list[DuplicateCandidate] = []
        exact_key = f"{document.tenant_id}:{document.file_hash}"
        existing_id = self._hash_index.get(exact_key)
        if existing_id and existing_id != document.document_id:
            existing = self.documents.get(existing_id) or self._load_document(existing_id)
            if existing and existing.tenant_id == document.tenant_id:
                candidates.append(
                    DuplicateCandidate(
                        status="exact_duplicate",
                        matched_document_id=existing_id,
                        matched_filename=existing.filename,
                        similarity=1.0,
                        evidence=[
                            self._evidence(
                                document,
                                "Exact SHA-256 content hash match.",
                                "sha256-duplicate-detector",
                                1.0,
                                metadata={
                                    "matched_document_id": existing_id,
                                    "algorithm": "SHA-256",
                                },
                            )
                        ],
                        reason="The source bytes have the same SHA-256 hash.",
                    )
                )

        for existing in self.documents.values():
            if (
                existing.document_id == document.document_id
                or existing.tenant_id != document.tenant_id
                or existing.document_id == existing_id
            ):
                continue
            dhash_similarity = perceptual_similarity(
                document.perceptual_hash,
                existing.perceptual_hash,
            )
            dino_similarity = cosine_similarity(
                document.visual_embedding,
                existing.visual_embedding,
            )
            dhash_match = (
                dhash_similarity is not None
                and dhash_similarity >= settings.visual_duplicate_threshold
            )
            dino_match = (
                dino_similarity is not None
                and dino_similarity >= settings.dinov2_duplicate_threshold
            )
            if not dhash_match and not dino_match:
                continue
            measured = [value for value in (dhash_similarity, dino_similarity) if value is not None]
            similarity = max(measured)
            evidence: list[VisionEvidence] = []
            if dhash_similarity is not None:
                evidence.append(
                    self._evidence(
                        document,
                        f"Document-image dHash similarity: {dhash_similarity:.4f}",
                        "document-dhash-detector",
                        dhash_similarity,
                        metadata={
                            "matched_document_id": existing.document_id,
                            "algorithm": "multi-page-dHash",
                            "threshold": settings.visual_duplicate_threshold,
                            "crossed_threshold": dhash_match,
                        },
                    )
                )
            if dino_similarity is not None:
                evidence.append(
                    self._evidence(
                        document,
                        f"DINOv2 document-image cosine similarity: {dino_similarity:.4f}",
                        self.dinov2_duplicate.adapter_id,
                        max(0.0, dino_similarity),
                        metadata={
                            "matched_document_id": existing.document_id,
                            "algorithm": "DINOv2-cosine",
                            "threshold": settings.dinov2_duplicate_threshold,
                            "crossed_threshold": dino_match,
                            "embedding_model": document.visual_embedding_model,
                        },
                    )
                )
            candidates.append(
                DuplicateCandidate(
                    status="probable_duplicate",
                    matched_document_id=existing.document_id,
                    matched_filename=existing.filename,
                    similarity=round(similarity, 4),
                    evidence=evidence,
                    reason=(
                        "The rendered page images crossed a configured visual-similarity threshold "
                        "using perceptual hash, DINOv2, or both."
                    ),
                )
            )

        semantic_key = self._semantic_key(document)
        if semantic_key:
            matched_id = self._semantic_index.get(f"{document.tenant_id}:{semantic_key}")
            if matched_id and matched_id != document.document_id:
                existing = self.documents.get(matched_id) or self._load_document(matched_id)
                similarity, matches, compared = self._structured_duplicate_similarity(
                    document, existing
                )
                if (
                    existing
                    and existing.tenant_id == document.tenant_id
                    and compared >= 2
                    and similarity >= 0.8
                    and not any(item.matched_document_id == matched_id for item in candidates)
                ):
                    candidates.append(
                        DuplicateCandidate(
                            status="probable_duplicate",
                            matched_document_id=matched_id,
                            matched_filename=existing.filename,
                            similarity=round(similarity, 4),
                            evidence=[
                                self._evidence(
                                    document,
                                    f"Matching structured fields: {', '.join(matches)}",
                                    "structured-duplicate-detector",
                                    similarity,
                                    metadata={
                                        "matched_document_id": matched_id,
                                        "matched_fields": matches,
                                        "compared_fields": compared,
                                    },
                                )
                            ],
                            reason="Identity and financial fields agree across the two documents.",
                        )
                    )

        related_match = self._related_document(document)
        if related_match:
            related, similarity = related_match
            if not any(
                candidate.matched_document_id == related.document_id for candidate in candidates
            ):
                candidates.append(
                    DuplicateCandidate(
                        status="related_document",
                        matched_document_id=related.document_id,
                        matched_filename=related.filename,
                        similarity=round(similarity, 4),
                        evidence=[
                            self._evidence(
                                document,
                                f"Normalized token Jaccard similarity: {similarity:.4f}",
                                "token-jaccard-related-document-detector",
                                similarity,
                                metadata={
                                    "matched_document_id": related.document_id,
                                    "threshold": settings.related_document_threshold,
                                },
                            )
                        ],
                        reason="The document text crossed the configured related-document threshold.",
                    )
                )
        if not candidates:
            candidates.append(
                DuplicateCandidate(
                    status="no_duplicate",
                    similarity=0.0,
                    reason="No measured duplicate signal crossed a configured threshold.",
                )
            )
        return candidates

    def _duplicate_risks(self, document: VisionDocumentRecord) -> list[VisionRiskIndicator]:
        risks = []
        for candidate in document.duplicate_candidates:
            if candidate.status in {"exact_duplicate", "probable_duplicate"}:
                risks.append(
                    self._risk(
                        document,
                        title="Duplicate Document Review",
                        severity=VisionRiskSeverity.high
                        if candidate.status == "exact_duplicate"
                        else VisionRiskSeverity.medium,
                        category="duplicate_detection",
                        description="The document contains suspicious indicators and requires human review.",
                        recommended_action="Compare duplicate match before approval.",
                        source_text=candidate.reason,
                        extra_evidence=candidate.evidence,
                    )
                )
        return risks

    def _run_anomaly_models(self, document: VisionDocumentRecord) -> None:
        self._load_all_documents(document.tenant_id)
        corpus = [item for item in self.documents.values() if item.tenant_id == document.tenant_id]
        unsupervised = self.isolation_forest.analyze(document, corpus)
        document.anomaly_analysis.append(unsupervised)
        document.model_runs.append(
            self._model_run(
                self.isolation_forest.stage,
                self.isolation_forest.adapter_id,
                self.isolation_forest.model_version,
                unsupervised.status,
                unsupervised.anomaly_score,
                notes=[
                    f"Corpus samples: {unsupervised.sample_count}; minimum: {settings.anomaly_min_samples}.",
                    "Anomaly percentile is an empirical rank, not a fraud probability.",
                ],
            )
        )
        if (
            unsupervised.status == "completed"
            and unsupervised.anomaly_score is not None
            and unsupervised.anomaly_score >= settings.anomaly_review_percentile
        ):
            evidence = self._evidence(
                document,
                "Unsupervised financial and document feature anomaly.",
                self.isolation_forest.adapter_id,
                unsupervised.anomaly_score,
                metadata={
                    "features": unsupervised.features,
                    "explanations": unsupervised.explanations,
                    "sample_count": unsupervised.sample_count,
                    "score_semantics": "empirical_anomaly_percentile_not_probability",
                },
            )
            document.risk_indicators.append(
                self._risk(
                    document,
                    title="Unusual Document Pattern",
                    severity=VisionRiskSeverity.medium,
                    category="unsupervised_anomaly",
                    description=(
                        "Measured document and financial features are unusual relative to this "
                        "organisation's processed corpus."
                    ),
                    recommended_action="Review the contributing features and supporting documents.",
                    source_text="Isolation Forest empirical anomaly percentile crossed the review threshold.",
                    evidence=[evidence],
                )
            )

        supervised_status = self.xgboost_risk.status()
        if supervised_status.status != "active":
            document.model_runs.append(
                self._model_run(
                    self.xgboost_risk.stage,
                    self.xgboost_risk.adapter_id,
                    self.xgboost_risk.model_version,
                    f"skipped_{supervised_status.status}",
                    None,
                    notes=[supervised_status.reason],
                )
            )
            return
        try:
            supervised = self.xgboost_risk.analyze(unsupervised.features)
        except Exception as exc:
            document.model_runs.append(
                self._model_run(
                    self.xgboost_risk.stage,
                    self.xgboost_risk.adapter_id,
                    self.xgboost_risk.model_version,
                    "failed",
                    None,
                    notes=[str(exc)],
                )
            )
            return
        document.anomaly_analysis.append(supervised)
        document.model_runs.append(
            self._model_run(
                self.xgboost_risk.stage,
                self.xgboost_risk.adapter_id,
                self.xgboost_risk.model_version,
                supervised.status,
                supervised.supervised_probability,
                notes=[
                    "XGBoost prediction is uncalibrated until validation calibration is supplied."
                ],
            )
        )
        if (
            supervised.supervised_probability is not None
            and supervised.supervised_probability >= settings.xgboost_review_threshold
        ):
            evidence = self._evidence(
                document,
                "Supervised risk-model review signal.",
                self.xgboost_risk.adapter_id,
                supervised.supervised_probability,
                metadata={
                    "features": supervised.features,
                    "shap_explanations": supervised.explanations,
                    "calibrated": supervised.calibrated,
                },
            )
            document.risk_indicators.append(
                self._risk(
                    document,
                    title="Supervised Risk Model Review",
                    severity=VisionRiskSeverity.high,
                    category="supervised_anomaly",
                    description="The trained risk model crossed the configured human-review threshold.",
                    recommended_action="Review SHAP feature contributions and source evidence.",
                    source_text="XGBoost review score crossed the configured threshold.",
                    evidence=[evidence],
                )
            )

    def _risk(
        self,
        document: VisionDocumentRecord,
        *,
        title: str,
        severity: VisionRiskSeverity,
        category: str,
        description: str,
        recommended_action: str,
        source_text: str,
        field_names: list[str] | None = None,
        evidence: list[VisionEvidence] | None = None,
        extra_evidence: list[VisionEvidence] | None = None,
        rule_id: str | None = None,
    ) -> VisionRiskIndicator:
        evidence_items = evidence or []
        if not evidence_items:
            fields = [
                field
                for field in document.extracted_fields
                if not field_names or field.name in field_names
            ]
            evidence_items.extend(field.evidence for field in fields[:6])
        if extra_evidence:
            evidence_items.extend(extra_evidence)
        if not evidence_items:
            evidence_items.append(
                self._evidence(
                    document,
                    source_text,
                    "local-risk-rule-engine",
                    1.0,
                    metadata={
                        "category": category,
                        "confidence_semantics": "deterministic_rule_condition_observed",
                    },
                )
            )
        resolved_category = self._indicator_category(category)
        resolved_rule_id = rule_id or self._rule_id(category, title)
        risk_identity = hash_document(
            "|".join(
                (
                    document.document_id,
                    resolved_rule_id,
                    resolved_category,
                    title.strip().lower(),
                    severity.value,
                    ",".join(sorted(field_names or [])),
                    self._normalize(source_text),
                )
            ).encode()
        )
        return VisionRiskIndicator(
            risk_id=f"vis-risk-{risk_identity[:12]}",
            title=title,
            severity=severity,
            category=resolved_category,
            description=description,
            recommended_action=recommended_action,
            evidence=evidence_items,
            rule_id=resolved_rule_id,
            affected_fields=field_names or [],
        )
