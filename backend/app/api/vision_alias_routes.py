from __future__ import annotations

from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.api.vision_routes import (
    accept_high_confidence_vision_fields,
    download_vision_report,
    generate_finance_narrative,
    generate_vision_report,
    get_finance_narrative,
    get_vision_report,
    list_finance_narratives,
    list_vision_reports,
    upload_vision_documents,
    vision_document_artifact,
    vision_document_events,
    vision_document_page_image,
    vision_document_source,
    vision_model_comparison,
)
from app.models.schemas import AuthUser, UserRole
from app.models.vision import (
    VisionAskRequest,
    VisionFieldReviewRequest,
    VisionFinanceNarrativeRequest,
    VisionProcessingStatus,
    VisionReportRequest,
    VisionReviewDecisionRequest,
)
from app.services.auth import require_user
from app.services.vision import vision_ai_service
from app.workers.vision_tasks import dispatch_vision_job

router = APIRouter(tags=["documind-vision-ai-aliases"])


@router.post("/demo/vision/seed")
async def alias_seed_connected_vision_demo(user: AuthUser = Depends(require_user)) -> dict:
    _require_reviewer(user)
    return _envelope(
        vision_ai_service.seed_connected_demo(
            tenant_id=user.tenant_id,
            actor=user.email,
        )
    )


@router.post("/documents/upload")
async def alias_upload_vision_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    metadata: str = Form(default="{}"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    user: AuthUser = Depends(require_user),
) -> dict:
    return await upload_vision_documents(
        background_tasks,
        files,
        metadata,
        idempotency_key,
        user,
    )


@router.get("/documents")
async def alias_list_vision_documents(user: AuthUser = Depends(require_user)) -> dict:
    documents = vision_ai_service.list_documents(user.tenant_id)
    return _envelope({"total": len(documents), "documents": [item.model_dump(mode="json") for item in documents]})


@router.get("/documents/{document_id}")
async def alias_vision_document_detail(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(document.model_dump(mode="json"), document.case_id)


@router.delete("/documents/{document_id}")
async def alias_delete_vision_document(document_id: str, user: AuthUser = Depends(require_user)) -> dict:
    _require_reviewer(user)
    try:
        document = vision_ai_service.delete_document(document_id, user.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope({"status": "deleted", "document_id": document_id}, document.case_id)


@router.post("/documents/{document_id}/reprocess")
async def alias_reprocess_vision_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    document = _document_or_404(document_id, user.tenant_id)
    if not document.original_storage_path:
        raise HTTPException(status_code=400, detail="Original file is not available for reprocessing")
    try:
        with open(document.original_storage_path, "rb") as source:
            content = source.read()
        response = vision_ai_service.queue_documents(
            [(document.filename, content, {**document.metadata, "reprocessed_from": document_id})],
            actor=user.email,
            tenant_id=user.tenant_id,
        )
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = response.model_dump(mode="json")
    payload["dispatch"] = _dispatch_or_background(
        background_tasks,
        response.job_id,
        user.tenant_id,
    )
    return _envelope(payload, response.case_id)


@router.get("/documents/{document_id}/status")
async def alias_vision_document_status(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(
        {
            "document_id": document.document_id,
            "case_id": document.case_id,
            "status": document.status.value,
            "updated_at": document.updated_at,
            "model_routing": document.model_routing,
            "review_task": document.review_task.model_dump(mode="json") if document.review_task else None,
        },
        document.case_id,
    )


@router.get("/documents/{document_id}/events")
async def alias_vision_document_events(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(require_user),
) -> StreamingResponse:
    return await vision_document_events(document_id, request, user)


@router.get("/documents/{document_id}/pages")
async def alias_vision_document_pages(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(
        {
            "document_id": document_id,
            "quality": [item.model_dump(mode="json") for item in document.quality],
            "ocr_words": [item.model_dump(mode="json") for item in document.ocr_words],
            "layout_regions": [item.model_dump(mode="json") for item in document.layout_regions],
            "visual_detections": [item.model_dump(mode="json") for item in document.visual_detections],
        },
        document.case_id,
    )


@router.get("/documents/{document_id}/pages/{page_number}/image")
async def alias_vision_document_page_image(
    document_id: str,
    page_number: int,
    variant: str = Query(default="processed", pattern="^(processed|original)$"),
    user: AuthUser = Depends(require_user),
) -> FileResponse:
    return await vision_document_page_image(document_id, page_number, variant, user)


@router.get("/documents/{document_id}/source")
async def alias_vision_document_source(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> FileResponse:
    return await vision_document_source(document_id, user)


@router.get("/documents/{document_id}/artifacts/{artifact_kind}/{artifact_id}")
async def alias_vision_document_artifact(
    document_id: str,
    artifact_kind: str,
    artifact_id: str,
    user: AuthUser = Depends(require_user),
) -> FileResponse:
    return await vision_document_artifact(
        document_id,
        artifact_kind,
        artifact_id,
        user,
    )


@router.get("/documents/{document_id}/fields")
async def alias_vision_document_fields(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope({"document_id": document_id, "fields": [item.model_dump(mode="json") for item in document.extracted_fields]}, document.case_id)


@router.get("/documents/{document_id}/tables")
async def alias_vision_document_tables(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope({"document_id": document_id, "tables": [item.model_dump(mode="json") for item in document.tables]}, document.case_id)


@router.get("/documents/{document_id}/risk")
async def alias_vision_document_risk(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(
        {
            "document_id": document_id,
            "risk_indicators": [item.model_dump(mode="json") for item in document.risk_indicators],
            "duplicate_candidates": [item.model_dump(mode="json") for item in document.duplicate_candidates],
            "review_required": document.review_task is not None,
        },
        document.case_id,
    )


@router.get("/documents/{document_id}/related")
async def alias_vision_document_related(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(
        {
            "document_id": document_id,
            "duplicates": [item.model_dump(mode="json") for item in document.duplicate_candidates],
            "graph_edges": [
                edge for edge in document.graph.edges if edge.get("type") in {"DUPLICATE_OF", "MATCHES", "CONFLICTS_WITH"}
            ],
        },
        document.case_id,
    )


@router.get("/reviews")
async def alias_list_vision_reviews(user: AuthUser = Depends(require_user)) -> dict:
    reviews = vision_ai_service.list_reviews(user.tenant_id)
    return _envelope({"total": len(reviews), "reviews": [item.model_dump(mode="json") for item in reviews]})


@router.get("/reviews/{review_id}")
async def alias_vision_review_detail(
    review_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    review = _review_or_404(review_id, user.tenant_id)
    document = _document_or_404(review.document_id, user.tenant_id)
    return _envelope({"review": review.model_dump(mode="json"), "document": document.model_dump(mode="json")}, document.case_id)


@router.patch("/reviews/{review_id}/fields/{field_id}")
async def alias_patch_vision_field(
    review_id: str,
    field_id: str,
    field_request: VisionFieldReviewRequest,
    request: Request,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    try:
        field = vision_ai_service.patch_field(
            review_id,
            field_id,
            reviewed_value=field_request.reviewed_value,
            reviewer_status=field_request.reviewer_status,
            comment=field_request.comment,
            actor=user.email,
            tenant_id=user.tenant_id,
            request_metadata=_request_metadata(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(field.model_dump(mode="json"))


@router.post("/reviews/{review_id}/accept-high-confidence")
async def alias_accept_high_confidence_vision_fields(
    review_id: str,
    request: Request,
    user: AuthUser = Depends(require_user),
) -> dict:
    return await accept_high_confidence_vision_fields(review_id, request, user)


@router.post("/reviews/{review_id}/approve")
async def alias_approve_vision_review(
    review_id: str,
    http_request: Request,
    request: VisionReviewDecisionRequest | None = None,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    return _review_decision(review_id, VisionProcessingStatus.approved, request, user, http_request)


@router.post("/reviews/{review_id}/reject")
async def alias_reject_vision_review(
    review_id: str,
    http_request: Request,
    request: VisionReviewDecisionRequest | None = None,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    return _review_decision(review_id, VisionProcessingStatus.rejected, request, user, http_request)


@router.post("/reviews/{review_id}/escalate")
async def alias_escalate_vision_review(
    review_id: str,
    http_request: Request,
    request: VisionReviewDecisionRequest | None = None,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    return _review_decision(review_id, VisionProcessingStatus.under_review, request, user, http_request)


@router.post("/reviews/{review_id}/confirm-error")
async def alias_confirm_vision_error(
    review_id: str,
    http_request: Request,
    request: VisionReviewDecisionRequest | None = None,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    return _review_decision(review_id, VisionProcessingStatus.confirmed_error, request, user, http_request)


@router.post("/reviews/{review_id}/confirm-fraud")
async def alias_confirm_vision_fraud(
    review_id: str,
    http_request: Request,
    request: VisionReviewDecisionRequest | None = None,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    return _review_decision(review_id, VisionProcessingStatus.confirmed_fraud, request, user, http_request)


@router.post("/ask")
async def alias_ask_vision_documents(
    request: VisionAskRequest,
    user: AuthUser = Depends(require_user),
) -> dict:
    try:
        answer = vision_ai_service.ask(request, user.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(answer.model_dump(mode="json"))


@router.get("/questions/{question_id}")
async def alias_vision_question_detail(
    question_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    try:
        return _envelope(vision_ai_service.get_question(question_id, user.tenant_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/questions/{question_id}/feedback")
async def alias_vision_question_feedback(
    question_id: str,
    payload: dict[str, Any],
    user: AuthUser = Depends(require_user),
) -> dict:
    try:
        feedback = vision_ai_service.record_question_feedback(
            question_id,
            rating=str(payload.get("rating", "helpful")),
            comment=str(payload.get("comment", "")),
            actor=user.email,
            tenant_id=user.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(feedback)


@router.get("/graph/document/{document_id}")
async def alias_vision_graph_for_document(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(document.graph.model_dump(mode="json"), document.case_id)


@router.get("/graph/entity/{entity_id}")
async def alias_vision_graph_entity(
    entity_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    needle = entity_id.lower()
    matches = []
    for document in vision_ai_service.list_documents(user.tenant_id):
        nodes = [
            node
            for node in document.graph.nodes
            if needle in str(node.get("id", "")).lower()
            or needle in str(node.get("label", "")).lower()
            or needle in str(node.get("value", "")).lower()
        ]
        if nodes:
            node_ids = {node.get("id") for node in nodes}
            matches.append(
                {
                    "document_id": document.document_id,
                    "nodes": nodes,
                    "edges": [
                        edge
                        for edge in document.graph.edges
                        if edge.get("source") in node_ids or edge.get("target") in node_ids
                    ],
                }
            )
    if not matches:
        raise HTTPException(status_code=404, detail="entity_id not found")
    return _envelope({"entity_id": entity_id, "matches": matches})


@router.post("/graph/query")
async def alias_vision_graph_query(
    payload: dict[str, Any],
    user: AuthUser = Depends(require_user),
) -> dict:
    answer = vision_ai_service.ask(
        VisionAskRequest(
            question=str(payload.get("question", "")),
            document_id=payload.get("document_id"),
        ),
        user.tenant_id,
    )
    return _envelope({"query_type": "vision_graph_or_evidence_search", "answer": answer.model_dump(mode="json")})


@router.get("/analytics/overview")
async def alias_vision_analytics_overview(user: AuthUser = Depends(require_user)) -> dict:
    return _envelope(vision_ai_service.analytics_overview(user.tenant_id))


@router.get("/analytics/vendors")
async def alias_vision_analytics_vendors(user: AuthUser = Depends(require_user)) -> dict:
    return _envelope(vision_ai_service.analytics_vendors(user.tenant_id))


@router.get("/analytics/risk")
async def alias_vision_analytics_risk(user: AuthUser = Depends(require_user)) -> dict:
    return _envelope(vision_ai_service.analytics_risk(user.tenant_id))


@router.get("/analytics/review")
async def alias_vision_analytics_review(user: AuthUser = Depends(require_user)) -> dict:
    return _envelope(vision_ai_service.analytics_review(user.tenant_id))


@router.get("/analytics/models")
async def alias_vision_analytics_models(user: AuthUser = Depends(require_user)) -> dict:
    return _envelope(vision_ai_service.analytics_models(user.tenant_id))


@router.get("/models/comparison/{document_id}")
async def alias_vision_model_comparison(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    return await vision_model_comparison(document_id, user)


@router.post("/finance-narratives")
async def alias_generate_finance_narrative(
    request: VisionFinanceNarrativeRequest,
    user: AuthUser = Depends(require_user),
) -> dict:
    return await generate_finance_narrative(request, user)


@router.get("/finance-narratives")
async def alias_list_finance_narratives(user: AuthUser = Depends(require_user)) -> dict:
    return await list_finance_narratives(user)


@router.get("/finance-narratives/{narrative_id}")
async def alias_get_finance_narrative(
    narrative_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    return await get_finance_narrative(narrative_id, user)


@router.post("/reports/generate")
async def alias_generate_vision_report(
    request: VisionReportRequest,
    user: AuthUser = Depends(require_user),
) -> dict:
    return await generate_vision_report(request, user)


@router.get("/reports")
async def alias_list_vision_reports(user: AuthUser = Depends(require_user)) -> dict:
    return await list_vision_reports(user)


@router.get("/reports/{report_id}")
async def alias_get_vision_report(
    report_id: str,
    format: str = Query(default="json", pattern="^(json|md|pdf)$"),
    user: AuthUser = Depends(require_user),
) -> Response:
    return await get_vision_report(report_id, format, user)


@router.get("/reports/{report_id}/download")
async def alias_download_vision_report(
    report_id: str,
    user: AuthUser = Depends(require_user),
) -> Response:
    return await download_vision_report(report_id, user)


@router.get("/admin/models")
async def alias_vision_admin_models(user: AuthUser = Depends(require_user)) -> dict:
    _require_admin(user)
    return _envelope(vision_ai_service.admin_models())


@router.patch("/admin/models/{model_id}")
async def alias_vision_admin_model_patch(
    model_id: str,
    payload: dict[str, Any],
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_admin(user)
    return _envelope(vision_ai_service.patch_admin_model(model_id, payload, actor=user.email))


@router.get("/admin/rules")
async def alias_vision_admin_rules(user: AuthUser = Depends(require_user)) -> dict:
    _require_admin(user)
    return _envelope(vision_ai_service.admin_rules())


@router.patch("/admin/rules/{rule_id}")
async def alias_vision_admin_rule_patch(
    rule_id: str,
    payload: dict[str, Any],
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_admin(user)
    return _envelope(vision_ai_service.patch_admin_rule(rule_id, payload, actor=user.email))


@router.get("/admin/audit")
async def alias_vision_admin_audit(user: AuthUser = Depends(require_user)) -> dict:
    _require_auditor(user)
    return _envelope(vision_ai_service.audit_history(user.tenant_id))


def _review_decision(
    review_id: str,
    decision: VisionProcessingStatus,
    request: VisionReviewDecisionRequest | None,
    user: AuthUser,
    http_request: Request,
) -> dict:
    try:
        review = vision_ai_service.record_review_decision(
            review_id,
            decision=decision,
            comment=request.comment if request else "",
            actor=user.email,
            tenant_id=user.tenant_id,
            request_metadata=_request_metadata(http_request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(review.model_dump(mode="json"), review.document_id)


def _request_metadata(request: Request) -> dict[str, Any]:
    return {
        "client_ip": request.client.host if request.client else "unknown",
        "session_id": request.headers.get("x-session-id", "not-provided")[:128],
        "user_agent": request.headers.get("user-agent", "unknown")[:256],
    }


def _document_or_404(document_id: str, tenant_id: str | None = None):
    try:
        return vision_ai_service.get_document(document_id, tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _review_or_404(review_id: str, tenant_id: str | None = None):
    try:
        return vision_ai_service.get_review(review_id, tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _require_reviewer(user: AuthUser) -> None:
    if user.role not in {UserRole.admin, UserRole.reviewer}:
        raise HTTPException(status_code=403, detail="Reviewer role required")


def _require_admin(user: AuthUser) -> None:
    if user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Admin role required")


def _require_auditor(user: AuthUser) -> None:
    if user.role not in {UserRole.admin, UserRole.auditor}:
        raise HTTPException(status_code=403, detail="Auditor role required")


def _dispatch_or_background(
    background_tasks: BackgroundTasks,
    job_id: str,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    dispatch = dispatch_vision_job(job_id, tenant_id)
    vision_ai_service.record_job_dispatch(job_id, dispatch, tenant_id)
    if dispatch["mode"] != "celery":
        background_tasks.add_task(vision_ai_service.process_job, job_id, tenant_id)
    return dispatch


def _envelope(data: Any, case_id: str | None = None, errors: list[str] | None = None) -> dict[str, Any]:
    return {
        "success": not errors,
        "case_id": case_id,
        "data": data,
        "errors": errors or [],
    }
