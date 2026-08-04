from app.models.schemas import TextAnalysisRequest
from app.models.vision import VisionProcessingStatus
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
        assert repository.get_review(document.review_task.review_id).document_id == document.document_id


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
    monkeypatch.setattr("app.services.vision.service.settings.upload_dir", str(tmp_path / "uploads"))
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

    assert result.status in {VisionProcessingStatus.completed.value, VisionProcessingStatus.review_required.value}
    assert document.ocr_words
    assert document.extracted_fields

    api_reader = VisionAIService()
    assert api_reader.get_job(response.job_id)["status"] == result.status
    assert api_reader.get_document(document.document_id).ocr_words


def test_dispatcher_uses_background_mode_when_celery_is_not_enabled(monkeypatch):
    monkeypatch.setattr("app.workers.vision_tasks.settings.enable_redis", False)

    dispatch = dispatch_vision_job("vision-job-test")

    assert dispatch["mode"] == "fastapi_background_task"
    assert dispatch["queue"] == VisionProcessingWorker.queue_name
