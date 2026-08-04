import json
import re
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import ValidationError

from app.core.config import settings
from app.models.schemas import (
    AnalysisResponse,
    AuthUser,
    DashboardMetrics,
    DemoRunRequest,
    Evidence,
    EvidenceChunk,
    ModelRoutingPreviewRequest,
    QueryCitation,
    QueryRequest,
    QueryResponse,
    TextAnalysisRequest,
    WorkflowDecisionRequest,
)
from app.services.auth import require_user
from app.services.pipeline import pipeline
from app.services.rag import RAGEngine
from app.services.readiness import PhaseReadinessService
from app.services.vision import vision_ai_service

router = APIRouter(tags=["documind"])
rag_engine = RAGEngine(pipeline.retrieval)
readiness = PhaseReadinessService()


@router.get("/health")
async def health() -> dict:
    return _envelope(
        {
            "status": "ok",
            "service": "documind-ai",
            "version": "0.1.0",
            "tenant_id": "tenant-local",
            "roles": settings.role_labels,
            "dataset_path": str(pipeline.dataset_cases.loader.dataset_path),
        },
    )


@router.get("/capabilities")
async def capabilities() -> dict:
    return {
        "ingestion": ["pdf", "docx", "pptx-adapter", "images", "emails", "scanned-documents"],
        "ai": ["ocr-layout", "classification", "ner-extraction", "hybrid-retrieval", "rag", "multi-agent"],
        "risk": ["compliance", "fraud", "cross-document-verification", "explainability"],
        "enterprise": ["workflow", "reports", "knowledge-graph", "mlops-ready"],
    }


@router.get("/models/registry")
async def model_registry() -> dict:
    return _envelope(pipeline.model_registry.registry())


@router.post("/models/routing-preview")
async def model_routing_preview(request: ModelRoutingPreviewRequest) -> dict:
    return _envelope(pipeline.model_registry.routing_preview(request))


@router.post("/documents/analyze")
async def analyze_text(request: TextAnalysisRequest):
    analysis = pipeline.process_text(request)
    return _envelope(analysis.model_dump(mode="json"), analysis.case_id)


@router.post("/documents/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    metadata: str = Form(default="{}"),
):
    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc

    payload = []
    for file in files:
        content = await file.read()
        payload.append((file.filename or "uploaded-document", content, parsed_metadata))
    try:
        analysis = pipeline.process_files(payload)
        return _envelope(analysis.model_dump(mode="json"), analysis.case_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/demo/run")
async def run_demo_case(request: DemoRunRequest | None = None) -> dict:
    try:
        analysis = pipeline.process_demo_case(request.case_id if request else None)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(analysis.model_dump(mode="json"), analysis.case_id)


@router.get("/cases")
async def list_cases() -> dict:
    cases = pipeline.dataset_case_summaries()
    return _envelope({"total": len(cases), "cases": cases})


@router.get("/cases/{case_id}/dashboard")
async def case_dashboard(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    data = {
        "summary": pipeline.case_summary(analysis),
        "executive_summary": analysis.executive_summary,
        "scores": analysis.scores or {
            "compliance_score": analysis.compliance_score,
            "risk_score": analysis.risk_score,
            "fraud_probability": analysis.fraud_probability,
            "approval_blocked": analysis.workflow.approval_blocked,
        },
        "metrics": {
            "document_count": len(analysis.documents),
            "evidence_count": len(analysis.evidence_chunks) + len(pipeline.collect_evidence(analysis)),
            "open_findings": len(analysis.compliance_findings) + len(analysis.fraud_signals),
            "agent_count": len(analysis.agent_decisions),
            "approval_gate": "Blocked" if analysis.workflow.approval_blocked else "Open",
        },
        "workflow": analysis.workflow.model_dump(mode="json"),
        "agent_decisions": [decision.model_dump(mode="json") for decision in analysis.agent_decisions],
    }
    return _envelope(data, case_id)


@router.get("/cases/{case_id}/documents")
async def case_documents(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    return _envelope([document.model_dump(mode="json") for document in analysis.documents], case_id)


@router.get("/cases/{case_id}/entities")
async def case_entities(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    return _envelope(analysis.entities, case_id)


@router.get("/cases/{case_id}/evidence")
async def case_evidence(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    evidence = pipeline.collect_evidence(analysis)
    data = {
        "evidence_count": len(evidence),
        "chunks": [chunk.model_dump(mode="json") for chunk in analysis.evidence_chunks],
        "evidence": [item.model_dump(mode="json") for item in evidence],
    }
    return _envelope(data, case_id)


@router.get("/cases/{case_id}/search", response_model=None)
async def case_search(case_id: str, q: str = Query(..., min_length=1), top_k: int = Query(default=5, ge=1, le=12)):
    analysis = _get_case_or_load(case_id)
    structured = _structured_case_search(analysis, q, top_k)
    if structured:
        return _envelope(structured.model_dump(mode="json"), case_id)
    result = rag_engine.answer(q, case_id, top_k)
    return _envelope(result.model_dump(mode="json"), case_id)


@router.get("/cases/{case_id}/risk")
async def case_risk(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    data = {
        "risk_score": analysis.risk_score,
        "risk_severity": analysis.workflow.status,
        "scores": analysis.scores,
        "blockers": analysis.workflow.blocked_reasons,
        "compliance_findings": [finding.model_dump(mode="json") for finding in analysis.compliance_findings],
        "fraud_signals": [signal.model_dump(mode="json") for signal in analysis.fraud_signals],
        "cross_document_checks": [check.model_dump(mode="json") for check in analysis.cross_document_checks],
        "workflow": analysis.workflow.model_dump(mode="json"),
    }
    return _envelope(data, case_id)


@router.get("/cases/{case_id}/compliance")
async def case_compliance(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    evidence_ids = {
        evidence.evidence_id
        for finding in analysis.compliance_findings
        for evidence in finding.evidence
    }
    data = {
        "compliance_score": analysis.compliance_score,
        "open_findings": len(analysis.compliance_findings),
        "approval_gate": "Blocked" if analysis.workflow.approval_blocked else "Open",
        "evidence_count": len(evidence_ids),
        "findings": [finding.model_dump(mode="json") for finding in analysis.compliance_findings],
    }
    return _envelope(data, case_id)


@router.get("/cases/{case_id}/fraud")
async def case_fraud(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    data = {
        "fraud_probability": analysis.fraud_probability,
        "fraud_probability_score": round(analysis.fraud_probability * 100),
        "fraud_confidence": analysis.fraud_confidence,
        "risk_score": analysis.risk_score,
        "signal_count": len(analysis.fraud_signals),
        "verification_checks": len(analysis.cross_document_checks),
        "coverage": _fraud_coverage(analysis),
        "signals": [signal.model_dump(mode="json") for signal in analysis.fraud_signals],
    }
    return _envelope(data, case_id)


@router.get("/cases/{case_id}/graph")
async def case_graph_page(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    data = {
        "nodes": analysis.graph_nodes or analysis.knowledge_graph.get("nodes", []),
        "edges": analysis.graph_edges or analysis.knowledge_graph.get("edges", []),
        "graph": analysis.knowledge_graph,
    }
    return _envelope(data, case_id)


@router.get("/cases/{case_id}/workflow")
async def case_workflow(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    return _envelope(analysis.workflow.model_dump(mode="json"), case_id)


@router.post("/cases/{case_id}/workflow/decision")
async def case_workflow_decision(
    case_id: str,
    payload: dict[str, Any],
    user: AuthUser = Depends(require_user),
) -> dict:
    try:
        request = WorkflowDecisionRequest(
            case_id=case_id,
            approver=payload.get("approver") or user.name,
            decision=payload.get("decision", "request_changes"),
            comment=payload.get("comment"),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    result = _record_workflow_decision(request, user)
    return _envelope(result, case_id)


@router.get("/cases/{case_id}/audit")
async def case_audit(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    return _envelope({"events": [event.model_dump(mode="json") for event in analysis.audit_events]}, case_id)


@router.get("/cases/{case_id}/agents")
async def case_agents(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    return _envelope([decision.model_dump(mode="json") for decision in analysis.agent_decisions], case_id)


@router.get("/cases/{case_id}", response_model=AnalysisResponse)
async def case_detail(case_id: str) -> AnalysisResponse:
    return _get_case_or_load(case_id)


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    if request.case_id:
        analysis = _get_case_or_load(request.case_id)
        structured = _structured_case_search(analysis, request.question, request.top_k)
        if structured:
            return structured
    return rag_engine.answer(request.question, request.case_id, request.top_k)


@router.get("/dashboard/metrics", response_model=DashboardMetrics)
async def dashboard_metrics() -> DashboardMetrics:
    return DashboardMetrics(**pipeline.dashboard_metrics())


@router.get("/compliance/findings")
async def compliance_findings(case_id: str | None = Query(default=None)) -> dict:
    cases = _case_scope(case_id)
    return {
        "total": sum(len(case.compliance_findings) for case in cases),
        "findings": [
            _case_payload(case.case_id, finding.model_dump(mode="json"))
            for case in cases
            for finding in case.compliance_findings
        ],
    }


@router.get("/fraud/findings")
async def fraud_findings(case_id: str | None = Query(default=None)) -> dict:
    cases = _case_scope(case_id)
    return {
        "total": sum(len(case.fraud_signals) for case in cases),
        "findings": [
            _case_payload(case.case_id, signal.model_dump(mode="json"))
            for case in cases
            for signal in case.fraud_signals
        ],
    }


@router.get("/agents/runs/{case_id}")
async def agent_run(case_id: str) -> dict:
    analysis = _get_case_or_404(case_id)
    return {
        "case_id": case_id,
        "agent_count": len(analysis.agent_decisions),
        "decisions": [decision.model_dump(mode="json") for decision in analysis.agent_decisions],
    }


@router.post("/workflow/decisions")
async def workflow_decision(
    request: WorkflowDecisionRequest,
    user: AuthUser = Depends(require_user),
) -> dict:
    return _record_workflow_decision(request, user)


@router.get("/graph/cases/{case_id}")
async def case_graph(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    return analysis.knowledge_graph


@router.get("/graph/risk-paths")
async def graph_risk_paths(min_risk_score: int = Query(default=50, ge=0, le=100)) -> dict:
    paths = pipeline.knowledge_graph.risk_paths(list(pipeline.cases.values()), min_risk_score)
    return {"min_risk_score": min_risk_score, "total": len(paths), "risk_paths": paths}


@router.post("/graph/project/{case_id}")
async def project_graph(case_id: str) -> dict:
    analysis = _get_case_or_load(case_id)
    graph = pipeline.knowledge_graph.build(
        case_id,
        analysis.documents,
        compliance_findings=analysis.compliance_findings,
        fraud_signals=analysis.fraud_signals,
        cross_document_checks=analysis.cross_document_checks,
        agent_decisions=analysis.agent_decisions,
        workflow=analysis.workflow,
        risk_score=analysis.risk_score,
        tenant_id=analysis.tenant_id,
    )
    graph["explainability"] = pipeline.explainability.package(
        analysis.agent_decisions,
        analysis.compliance_findings,
        analysis.fraud_signals,
    )
    analysis.knowledge_graph = graph
    return {
        "case_id": case_id,
        "status": "projected",
        "node_count": len(graph.get("nodes", [])),
        "edge_count": len(graph.get("edges", [])),
        "graph": graph,
    }


@router.get("/graph/entities/{entity_id}")
async def graph_entity(entity_id: str, case_id: str | None = Query(default=None)) -> dict:
    cases = _case_scope(case_id)
    neighborhoods = [
        pipeline.knowledge_graph.entity_neighborhood(case.knowledge_graph, entity_id) for case in cases
    ]
    matches = [neighborhood for neighborhood in neighborhoods if neighborhood["nodes"]]
    if not matches:
        raise HTTPException(status_code=404, detail="entity_id not found")
    return {
        "entity_id": entity_id,
        "case_id": case_id,
        "matches": matches,
    }


@router.get("/mlops/readiness")
async def mlops_readiness() -> dict:
    return readiness.mlops()


@router.get("/security/readiness")
async def security_readiness() -> dict:
    return readiness.security()


@router.get("/deployment/readiness")
async def deployment_readiness() -> dict:
    return readiness.deployment()


@router.get("/testing/readiness")
async def testing_readiness() -> dict:
    return readiness.testing()


@router.get("/research/readiness")
async def research_readiness() -> dict:
    return readiness.research()


@router.get("/portfolio/readiness")
async def portfolio_readiness() -> dict:
    return readiness.portfolio()


@router.get("/reports/{case_id}")
async def report(
    case_id: str,
    format: str = Query(default="json", pattern="^(md|json|pdf)$"),
    user: AuthUser = Depends(require_user),
):
    if case_id.startswith("report-"):
        try:
            stored_report = vision_ai_service.get_report(case_id, user.tenant_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if format == "md":
            return PlainTextResponse(
                vision_ai_service.render_stored_report_markdown(case_id, user.tenant_id)
            )
        if format == "pdf":
            return Response(
                content=vision_ai_service.render_stored_report_pdf(case_id, user.tenant_id),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{case_id}.pdf"'},
            )
        return JSONResponse(content=_envelope(stored_report.model_dump(mode="json")))
    analysis = _get_case_or_load(case_id)
    event = pipeline.audit.record(
        case_id,
        "report_exported",
        f"{user.name} exported {format.upper()} report.",
        actor=user.email,
        metadata={"format": format},
    )
    analysis.audit_events.append(event)
    if format == "json":
        return JSONResponse(content=_envelope(pipeline.reports.render_payload(analysis), case_id))
    if format == "pdf":
        return Response(
            content=pipeline.reports.render_pdf_bytes(analysis),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{case_id}-report.pdf"'},
        )
    return PlainTextResponse(pipeline.reports.render_markdown(analysis))


def _get_case_or_404(case_id: str) -> AnalysisResponse:
    analysis = pipeline.cases.get(case_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="case_id not found")
    return analysis


def _get_case_or_load(case_id: str) -> AnalysisResponse:
    analysis = pipeline.cases.get(case_id)
    if analysis:
        return analysis
    try:
        return pipeline.process_demo_case(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _case_scope(case_id: str | None) -> list[AnalysisResponse]:
    if case_id:
        return [_get_case_or_load(case_id)]
    return list(pipeline.cases.values())


def _case_payload(case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"case_id": case_id, **payload}


def _envelope(data: Any, case_id: str | None = None, errors: list[str] | None = None) -> dict[str, Any]:
    return {
        "success": not errors,
        "case_id": case_id,
        "data": data,
        "errors": errors or [],
    }


def _record_workflow_decision(request: WorkflowDecisionRequest, user: AuthUser | None = None) -> dict:
    analysis = _get_case_or_load(request.case_id)
    result = pipeline.workflow.record_decision(request, analysis.workflow)
    event = pipeline.audit.record(
        request.case_id,
        "workflow_decision_recorded",
        f"{request.approver} submitted {request.decision.value}.",
        actor=user.email if user else request.approver,
        metadata={**result, "role": user.role.value if user else None},
    )
    analysis.audit_events.append(event)
    return result


def _fraud_coverage(analysis: AnalysisResponse) -> list[dict[str, str]]:
    signal_text = " ".join(
        [signal.title for signal in analysis.fraud_signals]
        + [evidence.text for signal in analysis.fraud_signals for evidence in signal.evidence]
    ).lower()
    amount_check_review = any(
        "amount" in check.title.lower()
        and check.status.lower() in {"failed", "changed", "review required", "mismatch", "conflict"}
        for check in analysis.cross_document_checks
    )

    def review_if(condition: bool) -> str:
        return "Review" if condition else "Clear"

    return [
        {
            "name": "Duplicate invoices",
            "status": review_if("duplicate invoice" in signal_text),
            "rule": "Review only when duplicate invoice evidence exists.",
        },
        {
            "name": "Amount mismatches",
            "status": review_if("amount mismatch" in signal_text or "does not match" in signal_text or amount_check_review),
            "rule": "Review when invoice and PO values fail comparison.",
        },
        {
            "name": "Vendor anomalies",
            "status": review_if("bank account change" in signal_text or "vendor anomaly" in signal_text or "new bank" in signal_text),
            "rule": "Review for vendor bank changes, anomalies, or new bank details.",
        },
        {
            "name": "Payment anomalies",
            "status": review_if(
                any(term in signal_text for term in ["urgent", "manual override", "off-cycle", "bank account change"])
            ),
            "rule": "Review urgent, manual override, off-cycle, or bank-change evidence.",
        },
        {
            "name": "Fake document review",
            "status": review_if("fake document" in signal_text),
            "rule": "Review only when fake-document indicators are detected.",
        },
    ]


def _structured_case_search(analysis: AnalysisResponse, question: str, top_k: int) -> QueryResponse | None:
    lower = question.lower()
    if _is_payment_query(lower):
        return _payment_answer(analysis, top_k)

    if any(term in lower for term in ["exception", "review", "fraud signal", "clause", "missing"]):
        include_compliance = not ("fraud signal" in lower or (lower.startswith("what fraud") or lower.startswith("which fraud")))
        include_fraud = not ("clause" in lower or "which clause" in lower)
        finding_rows = []
        evidence_items: list[Evidence] = []
        if include_compliance:
            for finding in analysis.compliance_findings:
                evidence = finding.evidence[0] if finding.evidence else None
                if evidence:
                    evidence_items.append(evidence)
                title = _display_finding_title(finding.title)
                finding_rows.append(
                    {
                        "title": title,
                        "severity": f"{finding.severity.value.title()} severity",
                        "reason": evidence.reasoning if evidence and evidence.reasoning else finding.recommendation,
                        "evidence": evidence,
                    }
                )
        if include_fraud:
            for signal in analysis.fraud_signals:
                evidence = signal.evidence[0] if signal.evidence else None
                if evidence:
                    evidence_items.append(evidence)
                title = _display_finding_title(signal.title)
                finding_rows.append(
                    {
                        "title": title,
                        "severity": _fraud_severity_label(title, signal.severity.value),
                        "reason": evidence.reasoning if evidence and evidence.reasoning else signal.recommendation,
                        "evidence": evidence,
                    }
                )
        if finding_rows:
            priority = {
                "Missing indemnity requirement": 0,
                "Manual override lacks approval evidence": 1,
                "Bank account change request": 2,
                "Urgent wire request language detected": 3,
            }
            finding_rows.sort(key=lambda row: priority.get(row["title"], 20))
            evidence_items = []
            seen_evidence_ids = set()
            for row in finding_rows:
                evidence = row["evidence"]
                if not evidence:
                    continue
                key = evidence.evidence_id or evidence.chunk_id
                if key in seen_evidence_ids:
                    continue
                seen_evidence_ids.add(key)
                evidence_items.append(evidence)
            heading = "Exceptions requiring review:"
            if "fraud" in lower:
                heading = "Fraud signals detected:"
            if "clause" in lower or "missing" in lower:
                heading = "Missing or risky clauses:"
            lines = [heading]
            for index, row in enumerate(finding_rows[:top_k], start=1):
                evidence = row["evidence"]
                lines.append(f"{index}. {row['title']} - {row['severity']}")
                lines.append(f"   Reason: {row['reason']}")
                if evidence:
                    lines.append(
                        f"   Evidence: {evidence.source_document}, page {evidence.page_number}, "
                        f"chunk {evidence.chunk_id or evidence.evidence_id}, confidence {evidence.confidence:.0%}"
                    )
            return QueryResponse(
                answer="\n".join(lines),
                citations=_evidence_citations(evidence_items[:top_k]),
                confidence=0.95,
                reasoning=["Answered from backend compliance, fraud, and evidence-link records."],
            )

    if "document" in lower and ("uploaded" in lower or "load" in lower or "ingested" in lower):
        citations = _chunk_citations(analysis.evidence_chunks[:top_k])
        lines = ["Uploaded documents:"]
        for index, document in enumerate(analysis.documents, start=1):
            lines.append(
                f"{index}. {document.filename} - {document.document_type.value}, "
                f"{document.pages} page(s), checksum {document.checksum[:12]}"
            )
        return QueryResponse(
            answer="\n".join(lines),
            citations=citations,
            confidence=0.94,
            reasoning=["Answered from the backend case document registry."],
        )

    if "workflow" in lower or "recommended action" in lower or "payment blocked" in lower or "approval" in lower:
        evidence = pipeline.collect_evidence(analysis)[:top_k]
        citations = _evidence_citations(evidence)
        blockers = "; ".join(analysis.workflow.blocked_reasons) or "No approval blockers are active."
        answer = (
            f"Recommended workflow action: {_humanize_label(analysis.workflow.recommended_action.value)}.\n"
            f"Approval gate: {'Blocked' if analysis.workflow.approval_blocked else 'Open'}.\n"
            f"Assigned team: {analysis.workflow.assigned_team}.\n"
            f"Reason: {_humanize_sentence(blockers)}"
        )
        return QueryResponse(
            answer=answer,
            citations=citations,
            confidence=0.93,
            reasoning=["Answered from workflow state, approval gate, and linked evidence."],
        )
    return None


def _is_payment_query(lower_question: str) -> bool:
    return any(
        term in lower_question
        for term in (
            "expected payout",
            "payout",
            "payment amount",
            "invoice amount",
            "amount due",
            "how much",
            "expected payment",
            "payment status",
            "payment due",
            "release payment",
            "pay vendor",
            "payable amount",
            "amount matching",
            "amount match",
        )
    )


def _payment_answer(analysis: AnalysisResponse, top_k: int) -> QueryResponse:
    invoice_amount = _entity_value(analysis, "invoice_amount") or _document_field_value(analysis, "invoice", "amount")
    po_amount = _entity_value(analysis, "po_amount") or _document_field_value(analysis, "purchase_order", "amount")
    payment_terms = _entity_value(analysis, "payment_terms") or _document_field_value(analysis, "invoice", "payment_terms")
    amount_check = next(
        (
            check
            for check in analysis.cross_document_checks
            if "amount" in check.title.lower()
        ),
        None,
    )
    amount_status = _humanize_label(amount_check.status if amount_check else "review_required")
    status = "Hold Payment" if analysis.workflow.approval_blocked else "Ready For Review"
    approval_gate = "Blocked" if analysis.workflow.approval_blocked else "Open"
    reason_text = _payment_reason_text(analysis)
    citations = _payment_citations(analysis, top_k)
    answer = "\n".join(
        [
            "Expected payout:",
            f"- Invoice amount: {invoice_amount or 'Not found'}",
            f"- Purchase Order amount: {po_amount or 'Not found'}",
            f"- Match status: {amount_status}",
            f"- Payment terms: {payment_terms or 'Not found'}",
            f"- Current payment status: {status}",
            f"- Approval gate: {approval_gate}",
            f"- Reason: {reason_text}",
            f"- Recommended action: {_payment_recommended_action(analysis)}",
            "",
            "Evidence:",
            *[
                (
                    f"{index}. {citation.source_document}, page {citation.page_number}, "
                    f"chunk {citation.chunk_id}, confidence {citation.confidence:.0%}"
                )
                for index, citation in enumerate(citations, start=1)
            ],
            "",
            "Recommended Action:",
            f"- {_payment_recommended_action(analysis)}",
        ]
    )
    return QueryResponse(
        answer=answer,
        citations=citations,
        confidence=0.96,
        reasoning=["Answered from invoice, purchase order, workflow, fraud, compliance, and evidence chunks."],
    )


def _entity_value(analysis: AnalysisResponse, entity_type: str) -> str | None:
    for entity in analysis.entities:
        if entity.get("entity_type") == entity_type:
            return entity.get("entity_value")
    return None


def _document_field_value(analysis: AnalysisResponse, document_type: str, field_name: str) -> str | None:
    for document in analysis.documents:
        if document.document_type.value != document_type:
            continue
        for field in document.extracted_fields:
            if field.name == field_name:
                return field.value
    return None


def _invoice_due_date(analysis: AnalysisResponse) -> str | None:
    invoice_text = " ".join(
        document.text
        for document in analysis.documents
        if document.document_type.value == "invoice"
    )
    match = re.search(r"\bDue Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", invoice_text, re.IGNORECASE)
    return match.group(1) if match else None


def _payment_citations(analysis: AnalysisResponse, top_k: int) -> list[QueryCitation]:
    priority = ["invoice", "purchase_order", "email", "bank_statement", "policy"]
    selected = []
    seen = set()
    for document_type in priority:
        for chunk in analysis.evidence_chunks:
            key = chunk.chunk_id
            if key in seen or document_type not in chunk.document_id.lower():
                continue
            selected.append(chunk)
            seen.add(key)
            break
    return _chunk_citations(selected[: max(top_k, 5)])


def _display_titles(titles) -> list[str]:
    return [_display_finding_title(title) for title in titles]


def _payment_reason_text(analysis: AnalysisResponse) -> str:
    reason_flags = []
    combined = " ".join(
        [finding.title for finding in analysis.compliance_findings]
        + [signal.title for signal in analysis.fraud_signals]
    ).lower()
    if "bank account" in combined or "bank change" in combined:
        reason_flags.append("vendor bank account change")
    if "urgent" in combined or "wire" in combined:
        reason_flags.append("urgent wire request")
    if "manual override" in combined:
        reason_flags.append("manual override without approval")
    if "indemnity" in combined:
        reason_flags.append("missing indemnity requirement")
    if not reason_flags:
        return "no payment blockers are active."
    if len(reason_flags) == 1:
        return f"{reason_flags[0]}."
    return f"{', '.join(reason_flags[:-1])}, and {reason_flags[-1]}."


def _payment_recommended_action(analysis: AnalysisResponse) -> str:
    if analysis.workflow.approval_blocked:
        return (
            "verify vendor bank details independently and resolve approval-blocking compliance finding "
            "before releasing payment."
        )
    return _humanize_sentence(analysis.workflow.next_action)


def _chunk_citations(chunks: list[EvidenceChunk]) -> list[QueryCitation]:
    return [
        QueryCitation(
            document_id=chunk.document_id,
            source_document=chunk.source_document,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
            evidence_id=chunk.chunk_id,
            evidence_text=chunk.text,
            snippet=chunk.text[:420],
            score=1.0,
            confidence=chunk.confidence,
            retrieval_method="case_registry",
            linked_findings=chunk.linked_findings,
        )
        for chunk in chunks
    ]


def _evidence_citations(evidence: list[Evidence]) -> list[QueryCitation]:
    return [
        QueryCitation(
            document_id=item.document_id,
            source_document=item.source_document,
            page_number=item.page_number,
            chunk_id=item.chunk_id or item.evidence_id,
            evidence_id=item.evidence_id,
            evidence_text=item.text,
            snippet=item.snippet or item.text[:420],
            score=1.0,
            confidence=item.confidence,
            retrieval_method="case_evidence",
            linked_findings=item.linked_findings,
        )
        for item in evidence
    ]


def _display_finding_title(title: str) -> str:
    lower = title.lower()
    if "manual override" in lower:
        return "Manual override lacks approval evidence"
    if "bank account change" in lower:
        return "Bank account change request"
    if "urgent" in lower:
        return "Urgent wire request language detected"
    return title


def _humanize_label(value: str | None) -> str:
    if value is None:
        return "-"
    raw = str(value).strip()
    if not raw:
        return "-"
    overrides = {
        "hold_payment": "Hold Payment",
        "review_required": "Review Required",
        "approval_blocking": "Approval Blocking",
        "purchase_order": "Purchase Order",
        "bank_statement": "Bank Statement",
        "compliance_policy": "Compliance Policy",
    }
    key = raw.lower()
    if key in overrides:
        return overrides[key]
    label = re.sub(r"[_-]+", " ", raw)
    label = re.sub(r"\s+", " ", label).strip().title()
    for source, target in {
        "Api": "API",
        "Pdf": "PDF",
        "Json": "JSON",
        "Rag": "RAG",
        "Ocr": "OCR",
        "Pii": "PII",
        "Rbac": "RBAC",
        "Sla": "SLA",
        "Soc2": "SOC2",
        "Hipaa": "HIPAA",
        "Gdpr": "GDPR",
        "Xai": "XAI",
        "Mlops": "MLOps",
    }.items():
        label = re.sub(rf"\b{source}\b", target, label)
    return label


def _humanize_sentence(value: str | None) -> str:
    if not value:
        return "-"
    text = re.sub(r"[_-]+", " ", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:]


def _fraud_severity_label(title: str, severity: str) -> str:
    lower = title.lower()
    if "bank account change" in lower:
        return f"{severity.title()} fraud risk"
    if "manual override" in lower or "amount" in lower:
        return f"{severity.title()} severity"
    return f"{severity.title()} fraud risk"
