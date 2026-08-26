from collections import defaultdict
from uuid import uuid4

from app.models.schemas import CrossDocumentCheck, DocumentRecord, Evidence, FraudSignal, RiskLevel


class FraudDetectionEngine:
    def evaluate(
        self,
        documents: list[DocumentRecord],
        cross_document_checks: list[CrossDocumentCheck] | None = None,
    ) -> tuple[float, list[FraudSignal], str]:
        cross_document_checks = cross_document_checks or []
        signals: list[FraudSignal] = []
        signals.extend(self._duplicate_invoice_signals(documents))
        signals.extend(self._cross_document_signals(cross_document_checks))
        signals.extend(self._payment_anomaly_signals(documents))
        signals.extend(self._fake_document_signals(documents))

        if not signals:
            confidence = "low" if len(documents) < 2 or not cross_document_checks else "medium"
            return (0.04 if confidence == "low" else 0.08), [], confidence

        weighted = sum(signal.probability * self._severity_weight(signal.severity) for signal in signals)
        max_weight = max(1.0, sum(self._severity_weight(signal.severity) for signal in signals))
        coverage_bonus = min(0.16, len(cross_document_checks) * 0.035)
        probability = min(0.97, weighted / max_weight + coverage_bonus)
        confidence = "high" if cross_document_checks and len(documents) >= 3 else "medium"
        return round(probability, 3), signals, confidence

    def _duplicate_invoice_signals(self, documents: list[DocumentRecord]) -> list[FraudSignal]:
        groups: dict[tuple[str, str], list[DocumentRecord]] = defaultdict(list)
        for document in documents:
            invoice_numbers = [f.normalized_value or f.value.lower() for f in document.extracted_fields if f.name == "invoice_number"]
            amounts = [f.normalized_value or f.value.lower() for f in document.extracted_fields if f.name == "amount"]
            for invoice_number in invoice_numbers:
                groups[(invoice_number, (amounts[:1] or ["unknown"])[0])].append(document)

        signals = []
        for (invoice_number, amount), members in groups.items():
            if len(members) <= 1:
                continue
            signal_id = f"frd-{uuid4().hex[:10]}"
            signals.append(
                FraudSignal(
                    signal_id=signal_id,
                    title=f"Duplicate invoice number: {invoice_number}",
                    severity=RiskLevel.high,
                    probability=0.82,
                    evidence=[
                        self._evidence(member, f"Invoice {invoice_number} repeats with amount {amount}.", [signal_id])
                        for member in members
                    ],
                    recommendation="Hold payment and request vendor confirmation.",
                    score_impact=22,
                )
            )
        return signals

    def _cross_document_signals(self, checks: list[CrossDocumentCheck]) -> list[FraudSignal]:
        signals = []
        for check in checks:
            status = check.status.lower()
            if status not in {"mismatch", "conflict", "changed", "failed", "review required"}:
                continue
            severity = RiskLevel.high if "amount" in check.title.lower() or "bank" in check.title.lower() else RiskLevel.medium
            probability = 0.74 if severity == RiskLevel.high else 0.54
            signal_id = f"frd-{uuid4().hex[:10]}"
            evidence = [item.model_copy(update={"linked_findings": [*item.linked_findings, signal_id]}) for item in check.evidence]
            signals.append(
                FraudSignal(
                    signal_id=signal_id,
                    title=f"Cross-document verification issue: {check.title}",
                    severity=severity,
                    probability=probability,
                    evidence=evidence,
                    recommendation=check.recommendation or "Investigate conflicting document evidence before approval.",
                    score_impact=18 if severity == RiskLevel.high else 11,
                )
            )
        return signals

    def _payment_anomaly_signals(self, documents: list[DocumentRecord]) -> list[FraudSignal]:
        signals = []
        rules = [
            ("new bank account", "Vendor bank account change", RiskLevel.high, 0.76, "Verify bank details through an independent channel."),
            ("bank account change", "Vendor bank account change", RiskLevel.high, 0.76, "Verify bank details through an independent channel."),
            ("urgent wire", "Urgent payment wording", RiskLevel.medium, 0.58, "Require finance manager confirmation before release."),
            ("off-cycle payment", "Off-cycle payment request", RiskLevel.medium, 0.55, "Route to finance operations review."),
            ("manual override", "Manual override risk", RiskLevel.high, 0.71, "Require approval evidence and audit justification."),
            ("suspicious vendor", "Suspicious vendor anomaly", RiskLevel.high, 0.78, "Escalate vendor due diligence."),
        ]
        by_title: dict[str, FraudSignal] = {}
        for document in documents:
            lower = document.text.lower()
            for marker, title, severity, probability, recommendation in rules:
                if marker not in lower:
                    continue
                if title in by_title:
                    by_title[title].evidence.append(
                        self._evidence(document, f"Detected '{marker}' payment-risk wording.", [by_title[title].signal_id])
                    )
                    continue
                signal_id = f"frd-{uuid4().hex[:10]}"
                signal = (
                    FraudSignal(
                        signal_id=signal_id,
                        title=title,
                        severity=severity,
                        probability=probability,
                        evidence=[self._evidence(document, f"Detected '{marker}' payment-risk wording.", [signal_id])],
                        recommendation=recommendation,
                        score_impact=18 if severity == RiskLevel.high else 10,
                    )
                )
                by_title[title] = signal
                signals.append(signal)
        return signals

    def _fake_document_signals(self, documents: list[DocumentRecord]) -> list[FraudSignal]:
        signals = []
        for document in documents:
            lower = document.text.lower()
            if "photoshopped" not in lower and "template mismatch" not in lower and "fake document" not in lower:
                continue
            signal_id = f"frd-{uuid4().hex[:10]}"
            signals.append(
                FraudSignal(
                    signal_id=signal_id,
                    title="Fake document indicator",
                    severity=RiskLevel.critical,
                    probability=0.88,
                    evidence=[self._evidence(document, "Fake document indicator found in source text.", [signal_id])],
                    recommendation="Escalate to fraud operations and pause automated processing.",
                    score_impact=30,
                )
            )
        return signals

    def _evidence(self, document: DocumentRecord, reasoning: str, linked_findings: list[str]) -> Evidence:
        return Evidence(
            document_id=document.document_id,
            source_document=document.filename,
            page_number=1,
            text=document.text[:520] or "No extracted text available.",
            confidence=0.8,
            extraction_method="fraud_rule",
            reasoning=reasoning,
            linked_findings=linked_findings,
        )

    def _severity_weight(self, severity: RiskLevel) -> float:
        return {
            RiskLevel.low: 0.35,
            RiskLevel.medium: 0.55,
            RiskLevel.high: 0.82,
            RiskLevel.critical: 1.0,
        }[severity]
