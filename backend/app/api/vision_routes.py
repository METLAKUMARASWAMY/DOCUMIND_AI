import asyncio
import json
from pathlib import Path
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
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from app.models.schemas import AuthUser, TextAnalysisRequest, UserRole
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
from app.services.vision.evaluation import vision_evaluation_service
from app.services.vision.infrastructure import vision_infrastructure_readiness_service
from app.workers.vision_tasks import dispatch_vision_job

router = APIRouter(prefix="/vision", tags=["documind-vision-ai"])


@router.post("/documents/upload")
async def upload_vision_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    metadata: str = Form(default="{}"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    user: AuthUser = Depends(require_user),
) -> dict:
    parsed_metadata = _json_metadata(metadata)
    payload = []
    for file in files:
        payload.append(
            (
                file.filename or "uploaded-document",
                await file.read(),
                {**parsed_metadata, "declared_mime": file.content_type},
            )
        )
    try:
        response = vision_ai_service.queue_documents(
            payload,
            actor=user.email,
            idempotency_key=idempotency_key,
            tenant_id=user.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = response.model_dump(mode="json")
    payload["dispatch"] = _dispatch_or_background(
        background_tasks,
        response.job_id,
        user.tenant_id,
    )
    return _envelope(payload, response.case_id)


@router.post("/documents/analyze")
async def analyze_vision_text(
    request: TextAnalysisRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    user: AuthUser = Depends(require_user),
) -> dict:
    try:
        document = vision_ai_service.process_text(
            request,
            actor=user.email,
            idempotency_key=idempotency_key,
            tenant_id=user.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(document.model_dump(mode="json"), document.case_id)


@router.get("/jobs/{job_id}")
async def vision_job_status(job_id: str, user: AuthUser = Depends(require_user)) -> dict:
    job = vision_ai_service.get_job(job_id, user.tenant_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return _envelope(job, job.get("case_id"))


@router.post("/jobs/{job_id}/retry")
async def retry_vision_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    try:
        job = vision_ai_service.retry_job(job_id, user.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="job_id not found") from exc
    job["dispatch"] = _dispatch_or_background(background_tasks, job_id, user.tenant_id)
    return _envelope(job, job.get("case_id"))


@router.get("/documents")
async def list_vision_documents(user: AuthUser = Depends(require_user)) -> dict:
    documents = vision_ai_service.list_documents(user.tenant_id)
    return _envelope(
        {
            "total": len(documents),
            "documents": [document.model_dump(mode="json") for document in documents],
        }
    )


@router.get("/documents/{document_id}")
async def vision_document_detail(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(document.model_dump(mode="json"), document.case_id)


@router.delete("/documents/{document_id}")
async def delete_vision_document(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    try:
        document = vision_ai_service.delete_document(document_id, user.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope({"status": "deleted", "document_id": document_id}, document.case_id)


@router.post("/documents/{document_id}/reprocess")
async def reprocess_vision_document(
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
async def vision_document_status(
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
            "audit_events": [event.model_dump(mode="json") for event in document.audit_events[-12:]],
        },
        document.case_id,
    )


@router.get("/documents/{document_id}/events")
async def vision_document_events(
    document_id: str,
    request: Request,
    user: AuthUser = Depends(require_user),
) -> StreamingResponse:
    _document_or_404(document_id, user.tenant_id)

    async def event_stream():
        sent: set[str] = set()
        last_status = ""
        idle_ticks = 0
        terminal = {
            VisionProcessingStatus.approved,
            VisionProcessingStatus.rejected,
            VisionProcessingStatus.confirmed_error,
            VisionProcessingStatus.confirmed_fraud,
            VisionProcessingStatus.completed,
            VisionProcessingStatus.review_required,
        }
        while not await request.is_disconnected():
            current = vision_ai_service.get_document(document_id, user.tenant_id)
            for event in current.audit_events[-80:]:
                if event.event_id in sent:
                    continue
                sent.add(event.event_id)
                yield (
                    f"id: {event.event_id}\n"
                    f"event: {event.event_type}\n"
                    f"data: {json.dumps(event.model_dump(mode='json'))}\n\n"
                )
            if current.status.value != last_status:
                last_status = current.status.value
                idle_ticks = 0
                payload = {
                    "document_id": document_id,
                    "status": last_status,
                    "updated_at": current.updated_at,
                }
                yield f"event: status\ndata: {json.dumps(payload)}\n\n"
            else:
                idle_ticks += 1
            if current.status in terminal and idle_ticks >= 1:
                break
            if idle_ticks and idle_ticks % 20 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.75)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/documents/{document_id}/pages")
async def vision_document_pages(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(
        {
            "document_id": document_id,
            "page_count": document.page_count,
            "page_artifacts": [
                {
                    **item.model_dump(mode="json"),
                    "original_image_url": (
                        f"/api/v1/vision/documents/{document_id}/pages/"
                        f"{item.page_number}/image?variant=original"
                    ),
                    "processed_image_url": (
                        f"/api/v1/vision/documents/{document_id}/pages/"
                        f"{item.page_number}/image?variant=processed"
                    ),
                }
                for item in document.page_artifacts
            ],
            "quality": [item.model_dump(mode="json") for item in document.quality],
            "ocr_words": [word.model_dump(mode="json") for word in document.ocr_words],
            "layout_regions": [region.model_dump(mode="json") for region in document.layout_regions],
            "visual_detections": [item.model_dump(mode="json") for item in document.visual_detections],
        },
        document.case_id,
    )


@router.get("/documents/{document_id}/pages/{page_number}/image")
async def vision_document_page_image(
    document_id: str,
    page_number: int,
    variant: str = Query(default="processed", pattern="^(processed|original)$"),
    user: AuthUser = Depends(require_user),
) -> FileResponse:
    document = _document_or_404(document_id, user.tenant_id)
    artifact = next(
        (item for item in document.page_artifacts if item.page_number == page_number),
        None,
    )
    if not artifact:
        raise HTTPException(status_code=404, detail="page_number not found")
    path = (
        artifact.processed_image_path
        if variant == "processed"
        else artifact.original_image_path
    )
    if not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Page artifact is not available")
    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{document_id}-page-{page_number}.png",
    )


@router.get("/documents/{document_id}/source")
async def vision_document_source(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> FileResponse:
    document = _document_or_404(document_id, user.tenant_id)
    if not document.original_storage_path or not Path(document.original_storage_path).is_file():
        raise HTTPException(status_code=404, detail="Original source is not available")
    return FileResponse(
        document.original_storage_path,
        filename=document.filename,
        content_disposition_type="inline",
    )


@router.get("/documents/{document_id}/artifacts/{artifact_kind}/{artifact_id}")
async def vision_document_artifact(
    document_id: str,
    artifact_kind: str,
    artifact_id: str,
    user: AuthUser = Depends(require_user),
) -> FileResponse:
    document = _document_or_404(document_id, user.tenant_id)
    path: str | None = None
    if artifact_kind == "heatmap":
        analysis = next(
            (item for item in document.tamper_analysis if item.analysis_id == artifact_id),
            None,
        )
        path = analysis.heatmap_path if analysis else None
    elif artifact_kind == "visual-crop":
        detection = next(
            (item for item in document.visual_detections if item.detection_id == artifact_id),
            None,
        )
        path = detection.cropped_evidence_path if detection else None
    elif artifact_kind == "tamper-crop":
        region = next(
            (
                region
                for analysis in document.tamper_analysis
                for region in analysis.suspicious_regions
                if region.region_id == artifact_id
            ),
            None,
        )
        path = region.cropped_evidence_path if region else None
    else:
        raise HTTPException(status_code=404, detail="artifact_kind not found")
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Artifact is not available")
    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{document_id}-{artifact_kind}-{artifact_id}.png",
        content_disposition_type="inline",
    )


@router.get("/documents/{document_id}/fields")
async def vision_document_fields(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(
        {
            "document_id": document_id,
            "fields": [field.model_dump(mode="json") for field in document.extracted_fields],
        },
        document.case_id,
    )


@router.get("/documents/{document_id}/tables")
async def vision_document_tables(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(
        {
            "document_id": document_id,
            "tables": [table.model_dump(mode="json") for table in document.tables],
        },
        document.case_id,
    )


@router.get("/documents/{document_id}/risk")
async def vision_document_risk(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(
        {
            "document_id": document_id,
            "risk_indicators": [risk.model_dump(mode="json") for risk in document.risk_indicators],
            "duplicate_candidates": [candidate.model_dump(mode="json") for candidate in document.duplicate_candidates],
            "review_required": document.review_task is not None,
        },
        document.case_id,
    )


@router.get("/documents/{document_id}/related")
async def vision_document_related(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(
        {
            "document_id": document_id,
            "duplicates": [candidate.model_dump(mode="json") for candidate in document.duplicate_candidates],
            "graph_edges": [
                edge for edge in document.graph.edges if edge.get("type") in {"DUPLICATE_OF", "MATCHES", "CONFLICTS_WITH"}
            ],
        },
        document.case_id,
    )


@router.get("/reviews")
async def list_vision_reviews(user: AuthUser = Depends(require_user)) -> dict:
    reviews = vision_ai_service.list_reviews(user.tenant_id)
    return _envelope({"total": len(reviews), "reviews": [review.model_dump(mode="json") for review in reviews]})


@router.get("/reviews/{review_id}")
async def vision_review_detail(
    review_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    review = _review_or_404(review_id, user.tenant_id)
    document = _document_or_404(review.document_id, user.tenant_id)
    return _envelope(
        {
            "review": review.model_dump(mode="json"),
            "document": document.model_dump(mode="json"),
        },
        document.case_id,
    )


@router.patch("/reviews/{review_id}/fields/{field_id}")
async def patch_vision_field(
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
async def accept_high_confidence_vision_fields(
    review_id: str,
    request: Request,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    try:
        result = vision_ai_service.accept_high_confidence_fields(
            review_id,
            actor=user.email,
            tenant_id=user.tenant_id,
            request_metadata=_request_metadata(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(result)


@router.post("/reviews/{review_id}/approve")
async def approve_vision_review(
    review_id: str,
    http_request: Request,
    request: VisionReviewDecisionRequest | None = None,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    return _review_decision(review_id, VisionProcessingStatus.approved, request, user, http_request)


@router.post("/reviews/{review_id}/reject")
async def reject_vision_review(
    review_id: str,
    http_request: Request,
    request: VisionReviewDecisionRequest | None = None,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    return _review_decision(review_id, VisionProcessingStatus.rejected, request, user, http_request)


@router.post("/reviews/{review_id}/escalate")
async def escalate_vision_review(
    review_id: str,
    http_request: Request,
    request: VisionReviewDecisionRequest | None = None,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    return _review_decision(review_id, VisionProcessingStatus.under_review, request, user, http_request)


@router.post("/reviews/{review_id}/confirm-error")
async def confirm_vision_error(
    review_id: str,
    http_request: Request,
    request: VisionReviewDecisionRequest | None = None,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    return _review_decision(review_id, VisionProcessingStatus.confirmed_error, request, user, http_request)


@router.post("/reviews/{review_id}/confirm-fraud")
async def confirm_vision_fraud(
    review_id: str,
    http_request: Request,
    request: VisionReviewDecisionRequest | None = None,
    user: AuthUser = Depends(require_user),
) -> dict:
    _require_reviewer(user)
    return _review_decision(review_id, VisionProcessingStatus.confirmed_fraud, request, user, http_request)


@router.post("/ask")
async def ask_vision_documents(
    request: VisionAskRequest,
    user: AuthUser = Depends(require_user),
) -> dict:
    try:
        answer = vision_ai_service.ask(request, user.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(answer.model_dump(mode="json"))


@router.get("/graph/document/{document_id}")
async def vision_graph_for_document(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    document = _document_or_404(document_id, user.tenant_id)
    return _envelope(document.graph.model_dump(mode="json"), document.case_id)


@router.post("/graph/query")
async def vision_graph_query(
    payload: dict[str, Any],
    user: AuthUser = Depends(require_user),
) -> dict:
    question = str(payload.get("question", ""))
    answer = vision_ai_service.ask(
        VisionAskRequest(question=question, document_id=payload.get("document_id")),
        user.tenant_id,
    )
    return _envelope(
        {
            "query_type": "local_graph_or_evidence_search",
            "answer": answer.model_dump(mode="json"),
        }
    )


@router.get("/analytics/overview")
async def vision_analytics_overview(user: AuthUser = Depends(require_user)) -> dict:
    return _envelope(vision_ai_service.analytics_overview(user.tenant_id))


@router.get("/infrastructure/readiness")
async def vision_infrastructure_readiness() -> dict:
    return _envelope(vision_infrastructure_readiness_service.readiness())


@router.post("/evaluation/run")
async def vision_evaluation_run(payload: dict[str, Any]) -> dict:
    return _envelope(vision_evaluation_service.evaluate(payload))


@router.get("/models/comparison/{document_id}")
async def vision_model_comparison(
    document_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    try:
        return _envelope(
            vision_ai_service.model_comparison(document_id, user.tenant_id),
            document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/reports/generate")
async def generate_vision_report(
    request: VisionReportRequest,
    user: AuthUser = Depends(require_user),
) -> dict:
    try:
        report = vision_ai_service.generate_report(
            request,
            tenant_id=user.tenant_id,
            actor=user.email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(report.model_dump(mode="json"))


@router.get("/reports")
async def list_vision_reports(user: AuthUser = Depends(require_user)) -> dict:
    reports = vision_ai_service.list_reports(user.tenant_id)
    return _envelope(
        {"total": len(reports), "reports": [item.model_dump(mode="json") for item in reports]}
    )


@router.get("/reports/{report_id}")
async def get_vision_report(
    report_id: str,
    format: str = Query(default="json", pattern="^(json|md|pdf)$"),
    user: AuthUser = Depends(require_user),
) -> Response:
    try:
        report = vision_ai_service.get_report(report_id, user.tenant_id)
        if format == "md":
            return PlainTextResponse(
                vision_ai_service.render_stored_report_markdown(report_id, user.tenant_id)
            )
        if format == "pdf":
            return Response(
                content=vision_ai_service.render_stored_report_pdf(report_id, user.tenant_id),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{report_id}.pdf"'},
            )
        return JSONResponse(content=_envelope(report.model_dump(mode="json")))
    except ValueError as exc:
        # Compatibility for clients that used a document ID before reports became resources.
        try:
            document = vision_ai_service.get_document(report_id, user.tenant_id)
        except ValueError:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if format == "md":
            return PlainTextResponse(vision_ai_service.render_report_markdown(report_id, user.tenant_id))
        if format == "pdf":
            return Response(
                content=vision_ai_service.render_pdf_bytes(report_id, user.tenant_id),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{report_id}-vision-report.pdf"'},
            )
        return JSONResponse(
            content=_envelope(
                vision_ai_service.render_report_payload(report_id, user.tenant_id),
                document.case_id,
            )
        )


@router.get("/reports/{report_id}/download")
async def download_vision_report(
    report_id: str,
    user: AuthUser = Depends(require_user),
) -> Response:
    try:
        content = vision_ai_service.render_stored_report_pdf(report_id, user.tenant_id)
        filename = f"{report_id}.pdf"
    except ValueError:
        _document_or_404(report_id, user.tenant_id)
        content = vision_ai_service.render_pdf_bytes(report_id, user.tenant_id)
        filename = f"{report_id}-vision-report.pdf"
    return Response(content=content, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.post("/finance-narratives")
async def generate_finance_narrative(
    request: VisionFinanceNarrativeRequest,
    user: AuthUser = Depends(require_user),
) -> dict:
    try:
        narrative = vision_ai_service.generate_finance_narrative(
            request,
            tenant_id=user.tenant_id,
            actor=user.email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(narrative.model_dump(mode="json"))


@router.get("/finance-narratives")
async def list_finance_narratives(user: AuthUser = Depends(require_user)) -> dict:
    narratives = vision_ai_service.list_finance_narratives(user.tenant_id)
    return _envelope(
        {
            "total": len(narratives),
            "narratives": [item.model_dump(mode="json") for item in narratives],
        }
    )


@router.get("/finance-narratives/{narrative_id}")
async def get_finance_narrative(
    narrative_id: str,
    user: AuthUser = Depends(require_user),
) -> dict:
    try:
        narrative = vision_ai_service.get_finance_narrative(narrative_id, user.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(narrative.model_dump(mode="json"))


@router.get("/active-learning/export")
async def vision_active_learning_export(user: AuthUser = Depends(require_user)) -> dict:
    _require_reviewer(user)
    return _envelope(vision_ai_service.active_learning_export(user.tenant_id))


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


def _json_metadata(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    return parsed


def _require_reviewer(user: AuthUser) -> None:
    if user.role not in {UserRole.admin, UserRole.reviewer}:
        raise HTTPException(status_code=403, detail="Reviewer role required")


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
