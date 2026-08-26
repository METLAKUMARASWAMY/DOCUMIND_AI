from typing import ClassVar

from app.models.schemas import (
    AgentDecision,
    ComplianceFinding,
    DocumentRecord,
    Evidence,
    EvidenceChunk,
    FraudSignal,
    RiskLevel,
)


class MultiAgentReasoningSystem:
    agent_names: ClassVar[list[str]] = [
        "Document Understanding Agent",
        "Retrieval Agent",
        "Legal Agent",
        "Financial Agent",
        "Compliance Agent",
        "Risk Agent",
        "Summary Agent",
    ]

    def decide(
        self,
        documents: list[DocumentRecord],
        compliance_findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
        evidence_chunks: list[EvidenceChunk] | None = None,
    ) -> list[AgentDecision]:
        evidence_chunks = evidence_chunks or []
        evidence = self._core_evidence(documents, compliance_findings, fraud_signals)
        extraction_quality = self._extraction_quality(documents)
        evidence_coverage = min(1.0, len(evidence_chunks) / max(len(documents), 1) / 2)
        high_compliance = [f for f in compliance_findings if f.severity in {RiskLevel.high, RiskLevel.critical}]
        high_fraud = [s for s in fraud_signals if s.severity in {RiskLevel.high, RiskLevel.critical}]

        decisions = []
        for agent in self.agent_names:
            decision, next_action, status, certainty = self._decision_for_agent(
                agent,
                documents,
                compliance_findings,
                fraud_signals,
                high_compliance,
                high_fraud,
                evidence_coverage,
            )
            confidence = self._confidence(agent, extraction_quality, evidence_coverage, certainty)
            decisions.append(
                AgentDecision(
                    agent_name=agent,
                    decision=decision,
                    confidence=confidence,
                    evidence=evidence[:4],
                    evidence_ids=[item.evidence_id for item in evidence[:4]],
                    reasoning_summary=(
                        f"{agent} used extraction quality {extraction_quality:.0%}, "
                        f"evidence coverage {evidence_coverage:.0%}, and rule certainty {certainty:.0%}."
                    ),
                    reasoning_chain=[
                        "Reviewed classified documents, extracted entities, and generated evidence.",
                        "Checked compliance, fraud, and cross-document verification outputs.",
                        "Produced a bounded recommendation with linked evidence IDs.",
                    ],
                    next_action=next_action,
                    status=status,
                )
            )
        return decisions

    def _decision_for_agent(
        self,
        agent: str,
        documents: list[DocumentRecord],
        compliance_findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
        high_compliance: list[ComplianceFinding],
        high_fraud: list[FraudSignal],
        evidence_coverage: float,
    ) -> tuple[str, str, str, float]:
        has_contract = any(doc.document_type.value in {"contract", "legal_document"} for doc in documents)
        if agent == "Document Understanding Agent":
            return (
                f"Classified {len(documents)} document(s) and extracted enterprise entities",
                "Review low-confidence extracted fields",
                "completed",
                0.78,
            )
        if agent == "Retrieval Agent":
            return (
                "Evidence index ready for citation-grounded search" if evidence_coverage else "Evidence index has limited coverage",
                "Run search against generated findings and document chunks",
                "completed" if evidence_coverage else "needs_attention",
                0.72 + evidence_coverage * 0.2,
            )
        if agent == "Legal Agent":
            if has_contract and high_compliance:
                return (
                    "Legal review required before approval",
                    "Escalate to legal and remediate missing clauses",
                    "blocked",
                    0.86,
                )
            return (
                "No legal blocker detected",
                "Continue business review",
                "completed",
                0.72 if has_contract else 0.64,
            )
        if agent == "Financial Agent":
            if high_fraud:
                return (
                    "Payment hold recommended",
                    "Hold payment and verify vendor/payment evidence",
                    "blocked",
                    0.88,
                )
            return (
                "Financial checks passed with current evidence",
                "Continue finance approval",
                "completed",
                0.7,
            )
        if agent == "Compliance Agent":
            if compliance_findings:
                return (
                    f"{len(compliance_findings)} compliance exception(s) open",
                    "Resolve approval-blocking findings",
                    "blocked" if high_compliance else "needs_attention",
                    0.84 if high_compliance else 0.72,
                )
            return ("Compliance baseline satisfied", "Continue approval", "completed", 0.76)
        if agent == "Risk Agent":
            risk_count = len(high_compliance) + len(high_fraud)
            if risk_count:
                return ("Escalate case to risk operations", "Route case to risk queue", "blocked", 0.87)
            return ("Monitor under normal risk workflow", "No escalation required", "completed", 0.68)
        return (
            f"Prepared executive summary for {len(documents)} document(s)",
            "Generate final report package",
            "completed",
            0.74,
        )

    def _core_evidence(
        self,
        documents: list[DocumentRecord],
        compliance_findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
    ) -> list[Evidence]:
        evidence = []
        for finding in compliance_findings:
            evidence.extend(finding.evidence)
        for signal in fraud_signals:
            evidence.extend(signal.evidence)
        if evidence:
            return self._dedupe(evidence)
        return [
            Evidence(
                document_id=document.document_id,
                source_document=document.filename,
                page_number=1,
                text=document.text[:320] or "No extracted text available.",
                confidence=0.72,
                extraction_method="agent_context",
                reasoning="Representative document evidence selected for agent deliberation.",
            )
            for document in documents[:3]
        ]

    def _extraction_quality(self, documents: list[DocumentRecord]) -> float:
        fields = [field.confidence for document in documents for field in document.extracted_fields]
        if not fields:
            return 0.45
        return sum(fields) / len(fields)

    def _confidence(self, agent: str, extraction_quality: float, evidence_coverage: float, certainty: float) -> float:
        agent_weight = {
            "Document Understanding Agent": 0.12,
            "Retrieval Agent": 0.08,
            "Legal Agent": 0.05,
            "Financial Agent": 0.06,
            "Compliance Agent": 0.07,
            "Risk Agent": 0.04,
            "Summary Agent": 0.03,
        }[agent]
        score = 0.22 + extraction_quality * 0.28 + evidence_coverage * 0.22 + certainty * 0.22 + agent_weight
        return round(min(0.96, max(0.35, score)), 3)

    def _dedupe(self, evidence: list[Evidence]) -> list[Evidence]:
        seen = set()
        result = []
        for item in evidence:
            key = (item.document_id, item.text, tuple(item.linked_findings))
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
