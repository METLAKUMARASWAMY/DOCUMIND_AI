import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.legacy.finance_workflow_service import FinanceReviewGuardrailError, finance_workflow
from app.models.schemas import (
    EvaluationRunRequest,
    FinanceDocumentType,
    FinanceExtractionRequest,
    FinanceNarrativeRequest,
    HumanReviewRequest,
    RiskLevel,
)

router = APIRouter(prefix="/finance", tags=["finance-project6"])


@router.post("/upload")
async def finance_upload(
    file: UploadFile = File(...),
    document_type: str | None = Form(default=None),
    uploaded_by: str = Form(default="analyst"),
    metadata: str = Form(default="{}"),
) -> dict[str, Any]:
    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
    try:
        doc_type = FinanceDocumentType(document_type) if document_type else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid finance document_type") from exc
    try:
        document = finance_workflow.upload(
            file.filename or "finance-document.txt",
            await file.read(),
            document_type=doc_type,
            uploaded_by=uploaded_by,
            metadata=parsed_metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _envelope(document.model_dump(mode="json"), document.document_id)


@router.post("/extract")
async def finance_extract(request: FinanceExtractionRequest) -> dict[str, Any]:
    try:
        extraction = finance_workflow.extract(request.document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(extraction.model_dump(mode="json"), request.document_id)


@router.post("/narrative")
async def finance_narrative(request: FinanceNarrativeRequest) -> dict[str, Any]:
    try:
        narrative = finance_workflow.generate_narrative(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(narrative.model_dump(mode="json"), request.document_id)


@router.get("/documents")
async def finance_documents() -> dict[str, Any]:
    documents = [
        document.model_dump(mode="json") for document in finance_workflow.documents.values()
    ]
    return _envelope({"total": len(documents), "documents": documents})


@router.get("/documents/{document_id}")
async def finance_document(document_id: str) -> dict[str, Any]:
    document = finance_workflow.documents.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="finance document not found")
    return _envelope(
        {
            "document": document.model_dump(mode="json"),
            "extraction": finance_workflow.extractions.get(document_id).model_dump(mode="json")
            if document_id in finance_workflow.extractions
            else None,
            "narrative": finance_workflow.generated_narratives.get(document_id).model_dump(
                mode="json"
            )
            if document_id in finance_workflow.generated_narratives
            else None,
            "review": finance_workflow.reviews.get(document_id).model_dump(mode="json")
            if document_id in finance_workflow.reviews
            else None,
        },
        document_id,
    )


@router.get("/documents/{document_id}/chunks")
async def finance_document_chunks(document_id: str) -> dict[str, Any]:
    records = [
        record.model_dump(mode="json")
        for record in finance_workflow.vector_store.document_records(document_id)
        if record.source_type == "document_chunk"
    ]
    return _envelope({"total": len(records), "chunks": records}, document_id)


@router.get("/documents/{document_id}/extraction")
async def finance_document_extraction(document_id: str) -> dict[str, Any]:
    extraction = finance_workflow.extractions.get(document_id)
    if not extraction:
        raise HTTPException(status_code=404, detail="extraction not found")
    return _envelope(extraction.model_dump(mode="json"), document_id)


@router.get("/documents/{document_id}/narrative")
async def finance_document_narrative(document_id: str) -> dict[str, Any]:
    narrative = finance_workflow.generated_narratives.get(document_id)
    if not narrative:
        raise HTTPException(status_code=404, detail="narrative not found")
    return _envelope(narrative.model_dump(mode="json"), document_id)


@router.get("/documents/{document_id}/risk-flags")
async def finance_document_risk_flags(document_id: str) -> dict[str, Any]:
    extraction = finance_workflow.extractions.get(document_id)
    narrative = finance_workflow.generated_narratives.get(document_id)
    flags = []
    if extraction:
        flags.extend(flag.model_dump(mode="json") for flag in extraction.risk_flags)
    if narrative:
        known = {flag["flag_id"] for flag in flags}
        flags.extend(
            flag.model_dump(mode="json")
            for flag in narrative.risk_flags
            if flag.flag_id not in known
        )
    return _envelope({"total": len(flags), "risk_flags": flags}, document_id)


@router.get("/review/pending")
async def finance_pending_reviews() -> dict[str, Any]:
    pending = finance_workflow.pending_reviews()
    return _envelope({"total": len(pending), "reviews": pending})


@router.get("/review/completed")
async def finance_completed_reviews() -> dict[str, Any]:
    completed = finance_workflow.completed_reviews()
    return _envelope({"total": len(completed), "reviews": completed})


@router.get("/review/{document_id}")
async def finance_review_detail(document_id: str) -> dict[str, Any]:
    review = finance_workflow.reviews.get(document_id)
    if not review:
        raise HTTPException(status_code=404, detail="review not found")
    return _envelope(finance_workflow._review_payload(review), document_id)


@router.post("/review/{document_id}")
async def finance_send_or_update_review(
    document_id: str, request: HumanReviewRequest | None = None
) -> dict[str, Any]:
    try:
        if request is None:
            review = finance_workflow.send_to_review(document_id)
        else:
            review = finance_workflow.review(document_id, request, action=request.decision)
    except FinanceReviewGuardrailError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(review.model_dump(mode="json"), document_id)


@router.put("/review/{document_id}")
async def finance_update_review(document_id: str, request: HumanReviewRequest) -> dict[str, Any]:
    try:
        review = finance_workflow.review(document_id, request, action=request.decision)
    except FinanceReviewGuardrailError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(review.model_dump(mode="json"), document_id)


@router.post("/review/{document_id}/approve")
async def finance_approve_review(document_id: str, request: HumanReviewRequest) -> dict[str, Any]:
    try:
        review = finance_workflow.review(document_id, request, action="approve")
    except FinanceReviewGuardrailError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(review.model_dump(mode="json"), document_id)


@router.post("/review/{document_id}/reject")
async def finance_reject_review(document_id: str, request: HumanReviewRequest) -> dict[str, Any]:
    try:
        review = finance_workflow.review(document_id, request, action="reject")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(review.model_dump(mode="json"), document_id)


@router.post("/review/{document_id}/escalate")
async def finance_escalate_review(document_id: str, request: HumanReviewRequest) -> dict[str, Any]:
    try:
        review = finance_workflow.review(document_id, request, action="escalate")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(review.model_dump(mode="json"), document_id)


@router.post("/evaluation/run")
async def finance_evaluation_run(request: EvaluationRunRequest | None = None) -> dict[str, Any]:
    evaluation = finance_workflow.evaluate(request.document_id if request else None)
    return _envelope(evaluation.model_dump(mode="json"), request.document_id if request else None)


@router.get("/evaluation")
async def finance_evaluation() -> dict[str, Any]:
    evaluation = finance_workflow.evaluations.get("all") or finance_workflow.evaluate()
    return _envelope(evaluation.model_dump(mode="json"))


@router.get("/evaluation/{document_id}")
async def finance_document_evaluation(document_id: str) -> dict[str, Any]:
    evaluation = finance_workflow.evaluate(document_id)
    return _envelope(evaluation.model_dump(mode="json"), document_id)


@router.get("/metrics")
async def finance_metrics() -> dict[str, Any]:
    return _envelope(finance_workflow.metrics())


@router.get("/vector/search")
async def finance_vector_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(default=5, ge=1, le=20),
    document_id: str | None = Query(default=None),
    source_type: str | None = Query(default=None),
) -> dict[str, Any]:
    started = time_marker()
    hits = finance_workflow.vector_store.search(
        q, top_k=top_k, document_id=document_id, source_type=source_type
    )
    if document_id:
        finance_workflow.latency_by_document.setdefault(document_id, {})[
            "vector_retrieval_latency"
        ] = time_marker() - started
    return _envelope({"query": q, "total": len(hits), "hits": hits}, document_id)


@router.get("/vector/document/{document_id}")
async def finance_vector_document(document_id: str) -> dict[str, Any]:
    records = [
        record.model_dump(mode="json")
        for record in finance_workflow.vector_store.document_records(document_id)
    ]
    return _envelope(
        {
            "total": len(records),
            "records": records,
            "facets": finance_workflow.vector_store.facets(),
        },
        document_id,
    )


@router.get("/vector/similar-risk-cases")
async def finance_vector_similar_risk_cases(
    risk_level: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        parsed_level = RiskLevel(risk_level) if risk_level else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid risk_level") from exc
    cases = finance_workflow.vector_store.similar_risk_cases(parsed_level)
    return _envelope({"total": len(cases), "cases": cases})


@router.get("/demo-cases")
async def finance_demo_cases() -> dict[str, Any]:
    cases = [demo.model_dump(mode="json") for demo in finance_workflow.demo_cases]
    return _envelope({"total": len(cases), "cases": cases})


@router.post("/demo-cases/{case_id}/run")
async def finance_run_demo_case(case_id: str) -> dict[str, Any]:
    try:
        result = finance_workflow.run_demo_case(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _envelope(result, result["document"]["document_id"])


def _envelope(
    data: Any, document_id: str | None = None, errors: list[str] | None = None
) -> dict[str, Any]:
    return {
        "success": not errors,
        "document_id": document_id,
        "data": data,
        "errors": errors or [],
    }


def time_marker() -> float:
    import time

    return time.perf_counter()
