from typing import ClassVar
from uuid import uuid4

from app.models.schemas import (
    ComplianceFinding,
    ComplianceFramework,
    DocumentRecord,
    DocumentType,
    Evidence,
    RiskLevel,
)


class ComplianceEngine:
    required_clauses: ClassVar[dict[DocumentType, dict[str, dict[str, object]]]] = {
        DocumentType.contract: {
            "indemnity": {
                "keywords": ("indemnify", "indemnification", "hold harmless"),
                "severity": RiskLevel.high,
                "framework": ComplianceFramework.custom,
                "impact": 18,
            },
            "confidentiality": {
                "keywords": ("confidential", "non-disclosure", "nda"),
                "severity": RiskLevel.medium,
                "framework": ComplianceFramework.iso27001,
                "impact": 10,
            },
            "termination": {
                "keywords": ("termination", "terminate", "notice period"),
                "severity": RiskLevel.medium,
                "framework": ComplianceFramework.custom,
                "impact": 10,
            },
            "governing law": {
                "keywords": ("governing law", "jurisdiction", "venue"),
                "severity": RiskLevel.medium,
                "framework": ComplianceFramework.custom,
                "impact": 10,
            },
            "audit rights": {
                "keywords": ("audit rights", "right to audit", "audit cooperation"),
                "severity": RiskLevel.high,
                "framework": ComplianceFramework.soc2,
                "impact": 18,
            },
        },
        DocumentType.invoice: {
            "tax identifier": {
                "keywords": ("tax id", "gst", "vat", "ein"),
                "severity": RiskLevel.medium,
                "framework": ComplianceFramework.custom,
                "impact": 10,
            },
            "payment terms": {
                "keywords": ("payment terms", "net 30", "due date", "due on receipt"),
                "severity": RiskLevel.medium,
                "framework": ComplianceFramework.custom,
                "impact": 10,
            },
        },
    }

    def evaluate(self, documents: list[DocumentRecord]) -> tuple[int, list[ComplianceFinding]]:
        findings: list[ComplianceFinding] = []
        for document in documents:
            findings.extend(self._missing_clause_findings(document))
            findings.extend(self._policy_mismatch_findings(document))
            findings.extend(self._regulatory_risk_findings(document))
        score = max(0, 100 - sum(finding.score_impact for finding in findings))
        return score, findings

    def _missing_clause_findings(self, document: DocumentRecord) -> list[ComplianceFinding]:
        required = self.required_clauses.get(document.document_type, {})
        lower = document.text.lower()
        findings = []
        for clause, rule in required.items():
            if self._clause_present(lower, clause, rule["keywords"]):
                continue
            finding_id = f"cmp-{uuid4().hex[:10]}"
            evidence = self._evidence(
                document,
                f"Required {clause} language was not found in extracted text.",
                linked_findings=[finding_id],
            )
            severity = rule["severity"]
            findings.append(
                ComplianceFinding(
                    finding_id=finding_id,
                    title=f"Missing {clause} requirement",
                    severity=severity,
                    framework=rule["framework"],
                    clause_or_policy=clause,
                    evidence=[evidence],
                    recommendation=f"Add or verify the {clause} clause before approval.",
                    approval_blocking=severity in {RiskLevel.high, RiskLevel.critical},
                    score_impact=rule["impact"],
                )
            )
        return findings

    def _clause_present(self, lower_text: str, clause: str, keywords: tuple[str, ...]) -> bool:
        clause_negations = (
            f"does not include {clause}",
            f"doesn't include {clause}",
            f"does not contain {clause}",
            f"not include {clause}",
            f"no {clause}",
            f"without {clause}",
            f"missing {clause}",
            f"lacks {clause}",
        )
        if any(negation in lower_text for negation in clause_negations):
            return False
        return any(
            keyword in lower_text and not self._keyword_negated(lower_text, keyword)
            for keyword in keywords
        )

    def _keyword_negated(self, lower_text: str, keyword: str) -> bool:
        index = lower_text.find(keyword)
        if index < 0:
            return False
        context = lower_text[max(0, index - 90) : index + len(keyword) + 50]
        return any(
            marker in context
            for marker in (
                "does not include",
                "doesn't include",
                "does not contain",
                "not contain",
                "not include",
                "without",
                "missing",
                "not found",
                "lacks",
            )
        )

    def _policy_mismatch_findings(self, document: DocumentRecord) -> list[ComplianceFinding]:
        lower = document.text.lower()
        findings = []
        if "manual override" in lower and (
            "without approval" in lower or "approval not found" in lower or "no approval" in lower
        ):
            finding_id = f"cmp-{uuid4().hex[:10]}"
            findings.append(
                ComplianceFinding(
                    finding_id=finding_id,
                    title="Manual override lacks approval evidence",
                    severity=RiskLevel.high,
                    framework=ComplianceFramework.soc2,
                    clause_or_policy="approval policy",
                    evidence=[
                        self._evidence(
                            document,
                            "Manual override was detected without supporting approval evidence.",
                            linked_findings=[finding_id],
                        )
                    ],
                    recommendation="Route to compliance review and attach the policy exception approval.",
                    approval_blocking=True,
                    score_impact=18,
                )
            )
        if "sanctioned" in lower or "blocked vendor" in lower:
            finding_id = f"cmp-{uuid4().hex[:10]}"
            findings.append(
                ComplianceFinding(
                    finding_id=finding_id,
                    title="Potential restricted-party policy conflict",
                    severity=RiskLevel.critical,
                    framework=ComplianceFramework.custom,
                    clause_or_policy="restricted party policy",
                    evidence=[
                        self._evidence(
                            document,
                            "Restricted-party language was found in the source text.",
                            linked_findings=[finding_id],
                        )
                    ],
                    recommendation="Stop processing until compliance clears the counterparty.",
                    approval_blocking=True,
                    score_impact=25,
                )
            )
        return findings

    def _regulatory_risk_findings(self, document: DocumentRecord) -> list[ComplianceFinding]:
        lower = document.text.lower()
        framework_hits = [
            ("personal data", ComplianceFramework.gdpr, "privacy handling"),
            ("protected health", ComplianceFramework.hipaa, "PHI handling"),
            ("cardholder", ComplianceFramework.pci, "payment data handling"),
        ]
        findings = []
        for marker, framework, policy in framework_hits:
            if marker not in lower:
                continue
            if "redacted" in lower or "encrypted" in lower:
                continue
            finding_id = f"cmp-{uuid4().hex[:10]}"
            findings.append(
                ComplianceFinding(
                    finding_id=finding_id,
                    title=f"{framework.value} regulatory risk requires controls",
                    severity=RiskLevel.medium,
                    framework=framework,
                    clause_or_policy=policy,
                    evidence=[
                        self._evidence(
                            document,
                            f"{framework.value} sensitive data indicator found without explicit controls.",
                            linked_findings=[finding_id],
                        )
                    ],
                    recommendation="Confirm redaction, encryption, retention, and access controls.",
                    approval_blocking=False,
                    score_impact=10,
                )
            )
        return findings

    def _evidence(
        self,
        document: DocumentRecord,
        reasoning: str,
        linked_findings: list[str] | None = None,
    ) -> Evidence:
        return Evidence(
            document_id=document.document_id,
            source_document=document.filename,
            page_number=1,
            text=document.text[:520] or "No extracted text available.",
            confidence=0.84,
            extraction_method="compliance_rule",
            reasoning=reasoning,
            linked_findings=linked_findings or [],
        )
