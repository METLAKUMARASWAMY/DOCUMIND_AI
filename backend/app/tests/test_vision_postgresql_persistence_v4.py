from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import settings
from app.models.vision import (
    ExtractedVisionField,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidence,
    VisionFieldCompleteness,
)
from app.services.vision.persistence import VisionPersistenceCoordinator


class FakeConnection:
    def __init__(self, recorder: AsyncpgRecorder) -> None:
        self.recorder = recorder
        self.executions: list[tuple[str, tuple[Any, ...]]] = []
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False

    async def execute(self, statement: str, *parameters: Any) -> str:
        self.executions.append((statement, parameters))
        return "OK"

    async def fetchrow(self, statement: str, *parameters: Any) -> dict[str, Any] | None:
        self.fetches.append((statement, parameters))
        return self.recorder.fetchrow_value

    async def close(self) -> None:
        self.closed = True


class AsyncpgRecorder:
    def __init__(self) -> None:
        self.connect_calls: list[dict[str, Any]] = []
        self.connections: list[FakeConnection] = []
        self.fetchrow_value: dict[str, Any] | None = None
        self.connect_error: Exception | None = None

    async def connect(self, **kwargs: Any) -> FakeConnection:
        self.connect_calls.append(kwargs)
        if self.connect_error is not None:
            raise self.connect_error
        connection = FakeConnection(self)
        self.connections.append(connection)
        return connection


@pytest.fixture
def fake_asyncpg(monkeypatch: pytest.MonkeyPatch) -> AsyncpgRecorder:
    recorder = AsyncpgRecorder()
    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(connect=recorder.connect))
    monkeypatch.setattr(
        settings,
        "database_url",
        "postgresql+asyncpg://app:test-password@database.example/documind",
    )
    for name in ("enable_mongodb", "enable_redis", "enable_minio", "enable_neo4j"):
        monkeypatch.setattr(settings, name, False)
    return recorder


def _document() -> VisionDocumentRecord:
    evidence = VisionEvidence(
        evidence_id="evidence-sample-number",
        document_id="document-postgresql",
        page_number=1,
        source_text="Sample No. 6030",
        bounding_box=VisionBoundingBox(x1=10, y1=20, x2=80, y2=40),
        model_confidence=0.91,
        model_name="deterministic-schema-extractor",
        model_version="v3",
        processing_timestamp="2026-08-06T10:00:02+00:00",
        metadata={
            "candidate_score_components": {
                "label_match": 1.0,
                "spatial_relationship": 0.95,
            }
        },
    )
    field = ExtractedVisionField(
        field_id="field-sample-number",
        name="sample_number",
        value="6030",
        validation_status="valid",
        confidence=0.91,
        evidence=evidence,
        model_name="deterministic-schema-extractor",
        model_version="v3",
    )
    return VisionDocumentRecord(
        document_id="document-postgresql",
        case_id="case-postgresql",
        filename="00093726.png",
        tenant_id="tenant-alpha",
        document_type=VisionDocumentType.business_form,
        file_hash="a" * 64,
        extracted_fields=[field],
        field_completeness=VisionFieldCompleteness(
            schema_available=True,
            document_type=VisionDocumentType.business_form,
            required_fields=["sample_number", "date"],
            optional_fields=["notes"],
            missing_required_fields=["date"],
        ),
        created_at="2026-08-06T10:00:00+00:00",
        updated_at="2026-08-06T10:00:03+00:00",
        metadata={
            "pipeline_version": "documind-vision-v4",
            "processing_timing": {
                "pipeline_version": "documind-vision-v4",
                "queued_at": "2026-08-06T10:00:00+00:00",
                "processing_started_at": "2026-08-06T10:00:01+00:00",
                "processing_completed_at": "2026-08-06T10:00:03+00:00",
                "queued_duration_ms": 1000.2,
                "processing_duration_ms": 2000.4,
                "stage_durations_ms": {"ocr": 750.3},
                "reliable": True,
                "exclusion_reason": None,
            },
            "extraction_diagnostics": {
                "extractor": "deterministic-schema-extractor-v3",
                "rejected_candidates": [
                    {
                        "field": "sample_number",
                        "value": "603O",
                        "page": 1,
                        "strategy": "roi_reocr_same_line",
                        "reason": "duplicate_evidence_conflict",
                        "score": 0.61,
                        "evidence_box": {"x1": 10, "y1": 20, "x2": 80, "y2": 40},
                    }
                ],
            },
        },
    )


def test_document_artifacts_use_parameterized_idempotent_postgresql_upserts(
    fake_asyncpg: AsyncpgRecorder,
) -> None:
    persistence = VisionPersistenceCoordinator()
    document = _document()

    persistence.save_document(document)
    persistence.save_document(document)

    assert len(fake_asyncpg.connections) == 2
    assert all(connection.closed for connection in fake_asyncpg.connections)
    assert all(
        call["dsn"] == "postgresql://app:test-password@database.example/documind"
        for call in fake_asyncpg.connect_calls
    )
    first = fake_asyncpg.connections[0].executions
    second = fake_asyncpg.connections[1].executions
    assert len(first) == len(second) == 4
    assert [parameters for _, parameters in first] == [parameters for _, parameters in second]
    assert all("ON CONFLICT" in statement for statement, _ in first)
    assert all("tenant-alpha" not in statement for statement, _ in first)

    completeness = next(item for item in first if "vision_document_completeness" in item[0])
    assert completeness[1][:3] == (
        "tenant-alpha",
        "document-postgresql",
        "business_form",
    )
    assert json.loads(completeness[1][4]) == ["sample_number", "date"]

    timing = next(item for item in first if "vision_processing_timings" in item[0])
    assert timing[1][6:8] == (1000, 2000)
    candidates = [item for item in first if "vision_field_candidate_resolutions" in item[0]]
    assert {item[1][6] for item in candidates} == {"selected", "rejected_duplicate"}
    assert candidates[0][1][2] == "field-sample-number"
    assert str(candidates[1][1][2]).startswith("candidate-")


def test_conversation_round_trip_is_tenant_and_user_scoped_inside_running_loop(
    fake_asyncpg: AsyncpgRecorder,
) -> None:
    persistence = VisionPersistenceCoordinator()
    now = datetime.now(UTC)
    conversation = {
        "organisation_id": "tenant-alpha",
        "user_id": "user-alpha",
        "conversation_id": "conversation-alpha",
        "selected_scope": {"scope": "current_document", "document_id": "doc-1"},
        "pending_clarification": {"kind": "safety_scope"},
        "last_query_type": "clarification_required",
        "last_resolved_question": "What suspicious indicators are recorded?",
        "last_answer": {"answer": "No confirmed finding.", "citations": []},
        "turns": [{"role": "user", "content": "Is this safe?"}],
        "expires_at": (now + timedelta(minutes=30)).isoformat(),
        "updated_at": now.isoformat(),
    }
    fake_asyncpg.fetchrow_value = {
        "tenant_id": "tenant-alpha",
        "user_id": "user-alpha",
        "conversation_id": "conversation-alpha",
        "selected_scope": json.dumps(conversation["selected_scope"]),
        "pending_clarification": json.dumps(conversation["pending_clarification"]),
        "last_query_type": conversation["last_query_type"],
        "last_resolved_question": conversation["last_resolved_question"],
        "last_answer": json.dumps(conversation["last_answer"]),
        "turns": json.dumps(conversation["turns"]),
        "expires_at": now + timedelta(minutes=30),
        "updated_at": now,
    }

    async def call_from_running_loop() -> dict[str, Any] | None:
        persistence.save_conversation(conversation)
        return persistence.get_conversation(
            "conversation-alpha",
            "tenant-alpha",
            "user-alpha",
        )

    loaded = asyncio.run(call_from_running_loop())

    save_statement, save_parameters = fake_asyncpg.connections[0].executions[0]
    assert "ON CONFLICT (tenant_id, user_id, conversation_id)" in save_statement
    assert save_parameters[:3] == (
        "tenant-alpha",
        "user-alpha",
        "conversation-alpha",
    )
    fetch_statement, fetch_parameters = fake_asyncpg.connections[1].fetches[0]
    assert "tenant_id = $1" in fetch_statement
    assert "user_id = $2" in fetch_statement
    assert "conversation_id = $3" in fetch_statement
    assert fetch_parameters == ("tenant-alpha", "user-alpha", "conversation-alpha")
    assert loaded is not None
    assert loaded["organisation_id"] == "tenant-alpha"
    assert loaded["selected_scope"]["document_id"] == "doc-1"
    assert loaded["last_resolved_question"] == conversation["last_resolved_question"]
    assert loaded["last_answer"]["answer"] == "No confirmed finding."


def test_question_retrieval_trace_is_upserted_by_tenant_and_trace_identity(
    fake_asyncpg: AsyncpgRecorder,
) -> None:
    persistence = VisionPersistenceCoordinator()
    question = {
        "tenant_id": "tenant-alpha",
        "question_id": "question-alpha",
        "created_at": "2026-08-06T10:10:00+00:00",
        "answer": {
            "conversation_id": "conversation-alpha",
            "retrieval_trace_id": "trace-alpha",
            "retrieval_trace": {
                "trace_id": "trace-alpha",
                "sources_checked": ["verified_structured_fields", "lexical_bm25"],
                "candidate_counts": {"verified_structured_fields": 1},
                "selected_evidence": ["evidence-sample-number"],
                "retriever_versions": {"lexical_bm25": "v1"},
                "retrieval_duration_ms": 12.7,
            },
        },
    }

    persistence.save_question(question)
    persistence.save_question(question)

    first_statement, first_parameters = fake_asyncpg.connections[0].executions[0]
    _, second_parameters = fake_asyncpg.connections[1].executions[0]
    assert "ON CONFLICT (tenant_id, trace_id)" in first_statement
    assert "tenant-alpha" not in first_statement
    assert first_parameters == second_parameters
    assert first_parameters[:4] == (
        "tenant-alpha",
        "trace-alpha",
        "conversation-alpha",
        "question-alpha",
    )
    assert first_parameters[8] == 13


def test_postgresql_health_failure_is_sanitized_and_does_not_break_local_save(
    fake_asyncpg: AsyncpgRecorder,
) -> None:
    fake_asyncpg.connect_error = RuntimeError(
        "connection failed for postgresql://app:super-secret@database.example/documind "
        "password=hunter2"
    )
    persistence = VisionPersistenceCoordinator()

    health = persistence.initialize()
    persistence.save_document(_document())

    assert health["enabled"]["postgresql"] is True
    assert health["status"] == "degraded"
    assert "postgresql" in health["errors"]
    assert "super-secret" not in health["errors"]["postgresql"]
    assert "hunter2" not in health["errors"]["postgresql"]
    assert "****" in health["errors"]["postgresql"]
