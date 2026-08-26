import pytest

from app.models.schemas import TextAnalysisRequest
from app.models.vision import VisionAskRequest, VisionProcessingStatus
from app.services.vision.deterministic_extraction import deterministic_schema_extractor
from app.services.vision.repositories import LocalVisionRepository
from app.services.vision.service import VisionAIService
from app.services.vision.worker import VisionProcessingWorker
from app.workers.vision_tasks import dispatch_vision_job


def test_local_vision_repository_binds_documents_jobs_reviews_and_idempotency():
    service = VisionAIService()
    document = service.process_text(
        TextAnalysisRequest(
            filename="repo-invoice.txt",
            text="Invoice INV-REPO\nVendor: Repository Vendor\nTotal Amount: USD 40.00",
        )
    )
    repository = LocalVisionRepository()
    repository.save_document(document)
    repository.save_job({"job_id": "job-1", "status": "COMPLETED"})
    repository.bind_idempotency_key("idem-1", "job-1")

    assert repository.get_document(document.document_id).filename == "repo-invoice.txt"
    assert repository.get_job("job-1")["status"] == "COMPLETED"
    assert repository.resolve_idempotency_key("idem-1") == "job-1"

    if document.review_task:
        repository.save_review(document.review_task)
        assert (
            repository.get_review(document.review_task.review_id).document_id
            == document.document_id
        )


def test_vision_processing_worker_executes_idempotent_local_job():
    service = VisionAIService()
    response = service.queue_documents(
        [
            (
                "worker-invoice.txt",
                b"Invoice INV-WORKER\nVendor: Worker Vendor\nTotal Amount: USD 30.00",
                {},
            )
        ],
        idempotency_key="worker-idempotent",
    )
    duplicate = service.queue_documents(
        [
            (
                "worker-invoice.txt",
                b"Invoice INV-WORKER\nVendor: Worker Vendor\nTotal Amount: USD 30.00",
                {},
            )
        ],
        idempotency_key="worker-idempotent",
    )

    worker = VisionProcessingWorker(service)
    result = worker.process(response.job_id)

    assert response.job_id == duplicate.job_id
    assert result.status == "COMPLETED"
    assert service.get_document(response.document_ids[0]).ocr_words


def test_disk_state_allows_distinct_worker_instance_to_process_queued_job(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.vision.service.settings.upload_dir", str(tmp_path / "uploads")
    )
    api_service = VisionAIService()
    response = api_service.queue_documents(
        [
            (
                "distributed-worker-invoice.txt",
                b"Invoice INV-DIST\nVendor: Distributed Worker LLC\nTotal Amount: USD 77.00\nSignature: Queue",
                {},
            )
        ]
    )

    worker_service = VisionAIService()
    result = VisionProcessingWorker(worker_service).process(response.job_id)
    document = worker_service.get_document(response.document_ids[0])

    assert result.status in {
        VisionProcessingStatus.completed.value,
        VisionProcessingStatus.review_required.value,
    }
    assert document.ocr_words
    assert document.extracted_fields

    # The original long-lived API service must see the worker's durable update,
    # not return its stale pre-queue in-memory object.
    refreshed_job = api_service.get_job(response.job_id)
    refreshed_document = next(
        item
        for item in api_service.list_documents(document.tenant_id)
        if item.document_id == document.document_id
    )
    assert refreshed_job is not None
    assert refreshed_job["status"] == result.status
    assert refreshed_document.ocr_words
    assert refreshed_document.extracted_fields

    api_reader = VisionAIService()
    assert api_reader.get_job(response.job_id)["status"] == result.status
    assert api_reader.get_document(document.document_id).ocr_words


def test_dispatcher_uses_background_mode_when_celery_is_not_enabled(monkeypatch):
    monkeypatch.setattr("app.workers.vision_tasks.settings.enable_redis", False)

    dispatch = dispatch_vision_job("vision-job-test")

    assert dispatch["mode"] == "fastapi_background_task"
    assert dispatch["queue"] == VisionProcessingWorker.queue_name


def test_structured_extraction_failure_preserves_ocr_index_and_grounded_ask(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.services.vision.service.settings.upload_dir", str(tmp_path / "uploads")
    )
    monkeypatch.setattr(
        deterministic_schema_extractor,
        "extract",
        lambda _document: (_ for _ in ()).throw(RuntimeError("forced extraction failure")),
    )
    monkeypatch.setattr(
        VisionAIService,
        "_discover_generic_intelligence",
        lambda _service, _document: (_ for _ in ()).throw(
            RuntimeError("forced generic extraction failure")
        ),
    )
    service = VisionAIService()
    tenant_id = "tenant-extraction-failure-continuity"

    document = service.process_text(
        TextAnalysisRequest(
            filename="ocr-survives.txt",
            text="Reference Code: OCR-KEEP-731\nContact Desk: North Wing",
        ),
        tenant_id=tenant_id,
    )

    assert document.metadata["extraction"]["status"] == "EXTRACTION_FAILED"
    assert document.metadata["extraction"]["ocr_retrieval_preserved"] is True
    assert document.extracted_fields == []
    assert document.evidence_chunks
    assert any(
        run.stage == "extraction" and run.status == "failed_ocr_retrieval_preserved"
        for run in document.model_runs
    )
    answer = service.ask(
        VisionAskRequest(
            question="What is the reference code?",
            document_id=document.document_id,
            scope="current_document",
        ),
        tenant_id,
    )
    assert answer.answer_status == "supported"
    assert "OCR-KEEP-731" in answer.answer
    assert answer.citations
    assert answer.citations[0].document_id == document.document_id


def test_durable_reads_do_not_reindex_an_unchanged_document(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.vision.service.settings.upload_dir", str(tmp_path / "uploads")
    )
    service = VisionAIService()
    document = service.process_text(
        TextAnalysisRequest(filename="stable.txt", text="Case Label: CACHE-1"),
        tenant_id="tenant-stable-read",
    )
    calls = 0
    original = service.retrieval_index.upsert_document

    def count_upsert(document_id, chunks):
        nonlocal calls
        calls += 1
        return original(document_id, chunks)

    monkeypatch.setattr(service.retrieval_index, "upsert_document", count_upsert)
    assert service.get_document(document.document_id, "tenant-stable-read").document_id == (
        document.document_id
    )
    assert service.get_document(document.document_id, "tenant-stable-read").document_id == (
        document.document_id
    )
    assert calls == 0


def test_terminal_processing_failure_can_be_audited_retried_and_completed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.services.vision.service.settings.upload_dir", str(tmp_path / "uploads")
    )
    service = VisionAIService()
    response = service.queue_documents(
        [("retryable.txt", b"Reference: RETRY-42", {})],
        tenant_id="tenant-retry-audit",
    )
    original_parse = service.ingestion.parse
    monkeypatch.setattr(
        service.ingestion,
        "parse",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced parse outage")),
    )

    service.process_job(response.job_id, "tenant-retry-audit")
    failed = service.get_job(response.job_id, "tenant-retry-audit")
    assert failed is not None
    assert failed["status"] == VisionProcessingStatus.review_required.value
    assert failed["errors"][0]["error"] == "forced parse outage"
    failed_document = service.get_document(
        response.document_ids[0],
        "tenant-retry-audit",
    )
    assert failed_document.status == VisionProcessingStatus.review_required
    assert any(event.event_type == "processing_failed" for event in failed_document.audit_events)

    monkeypatch.setattr(service.ingestion, "parse", original_parse)
    service.retry_job(
        response.job_id,
        "tenant-retry-audit",
        actor="reviewer@documind.test",
    )
    service.process_job(response.job_id, "tenant-retry-audit")

    completed = service.get_job(response.job_id, "tenant-retry-audit")
    assert completed is not None
    assert completed["status"] in {
        VisionProcessingStatus.completed.value,
        VisionProcessingStatus.review_required.value,
    }
    assert completed["errors"] == []
    assert any(
        event["event_type"] == "job_retry_queued" and event["actor"] == "reviewer@documind.test"
        for event in service.system_audit_events
    )
    service.delete_document(
        response.document_ids[0],
        "tenant-retry-audit",
        actor="reviewer@documind.test",
    )
    assert any(
        event["event_type"] == "document_deleted"
        and event["metadata"]["document_id"] == response.document_ids[0]
        for event in service.system_audit_events
    )


def test_delete_persistence_failure_does_not_emit_success_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.vision.service.settings.upload_dir", str(tmp_path / "uploads")
    )
    service = VisionAIService()
    tenant_id = "tenant-delete-failure-audit"
    document = service.process_text(
        TextAnalysisRequest(
            filename="delete-failure.txt",
            text="Reference: DELETE-42",
        ),
        tenant_id=tenant_id,
    )
    monkeypatch.setattr(
        service.persistence,
        "delete_document",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("forced persistence delete failure")
        ),
    )

    with pytest.raises(RuntimeError, match="forced persistence delete failure"):
        service.delete_document(
            document.document_id,
            tenant_id,
            actor="reviewer@documind.test",
        )

    assert not any(
        event["event_type"] == "document_deleted"
        and event["metadata"].get("document_id") == document.document_id
        for event in service.system_audit_events
    )
