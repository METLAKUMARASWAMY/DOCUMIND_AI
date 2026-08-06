from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import re
import threading
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.core.config import settings
from app.models.vision import (
    VisionDocumentRecord,
    VisionFinanceNarrative,
    VisionGraphProjection,
    VisionReportRecord,
    VisionReviewTask,
)

GRAPH_NODE_LABELS = frozenset(
    {
        "Organisation",
        "Document",
        "Page",
        "Vendor",
        "Customer",
        "Person",
        "Invoice",
        "PurchaseOrder",
        "Contract",
        "BankAccount",
        "TaxIdentifier",
        "Address",
        "ContactMethod",
        "Date",
        "Amount",
        "Identifier",
        "Product",
        "Form",
        "Payment",
        "Signature",
        "Stamp",
        "RiskIndicator",
        "ExtractedField",
    }
)
GRAPH_RELATIONSHIP_TYPES = frozenset(
    {
        "OWNS",
        "CONTAINS",
        "CONTAINS_PAGE",
        "SUBMITTED",
        "ISSUED_BY",
        "ISSUED_TO",
        "REFERENCES",
        "PAID_TO",
        "APPROVED_BY",
        "GOVERNED_BY",
        "REGISTERED_WITH",
        "LOCATED_AT",
        "HAS_CONTACT",
        "HAS_FIELD",
        "HAS_DATE",
        "HAS_TITLE",
        "HAS_SAMPLE_NUMBER",
        "HAS_PRODUCT_TYPE",
        "HAS_BATCH_SIZE",
        "REQUESTED_BY",
        "SPECIFICATIONS_WRITTEN_BY",
        "ASSOCIATED_WITH",
        "HAS_BANK_ACCOUNT",
        "HAS_CLIENT_NUMBER",
        "HAS_FAX",
        "HAS_PHONE",
        "SENT",
        "SENT_TO",
        "HAS_RISK",
        "SIGNED_BY",
        "DUPLICATE_OF",
        "MATCHES",
        "CONFLICTS_WITH",
        "AMENDS",
    }
)


class VisionPersistenceCoordinator:
    def __init__(self) -> None:
        self._mongo_client: Any | None = None
        self._mongo_database: Any | None = None
        self._redis: Any | None = None
        self._minio: Any | None = None
        self._neo4j: Any | None = None
        self._lock = threading.RLock()
        self._initialized = False
        self._errors: dict[str, str] = {}

    def initialize(self) -> dict[str, Any]:
        with self._lock:
            if settings.database_url:
                self._initialize_postgresql()
            if settings.enable_mongodb:
                self._initialize_mongodb()
            if settings.enable_redis:
                self._initialize_redis()
            if settings.enable_minio:
                self._initialize_minio()
            if settings.enable_neo4j:
                self._initialize_neo4j()
            self._initialized = True
        return self.health()

    def close(self) -> None:
        if self._mongo_client is not None:
            self._mongo_client.close()
        if self._redis is not None:
            self._redis.close()
        if self._neo4j is not None:
            self._neo4j.close()

    def store_original(
        self,
        document: VisionDocumentRecord,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> str | None:
        if not settings.enable_minio:
            return None
        try:
            client = self._minio_client()
            object_key = f"{document.tenant_id}/{document.document_id}/original/{Path(filename).name}"
            client.put_object(
                settings.minio_bucket,
                object_key,
                BytesIO(content),
                length=len(content),
                content_type=content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
            )
            self._errors.pop("minio", None)
            return f"s3://{settings.minio_bucket}/{object_key}"
        except Exception as exc:
            self._record_error("minio", exc)
            return None

    def save_document(self, document: VisionDocumentRecord) -> None:
        if settings.enable_mongodb:
            try:
                database = self._mongo_database_handle()
                payload = document.model_dump(mode="json")
                protected = self._protect(payload)
                database.documents.replace_one(
                    {
                        "tenant_id": document.tenant_id,
                        "document_id": document.document_id,
                    },
                    protected,
                    upsert=True,
                )
                self._replace_many(database.pages, document, payload.get("page_artifacts", []), "page_number")
                self._replace_many(database.ocr_results, document, payload.get("ocr_words", []), "word_id")
                self._replace_many(database.ocr_lines, document, payload.get("ocr_lines", []), "line_id")
                self._replace_many(
                    database.extracted_fields,
                    document,
                    self._field_records(document),
                    "field_id",
                )
                self._replace_many(database.tables, document, payload.get("tables", []), "table_id")
                self._replace_many(
                    database.risk_indicators,
                    document,
                    payload.get("risk_indicators", []),
                    "risk_id",
                )
                self._replace_many(
                    database.processing_warnings,
                    document,
                    payload.get("processing_warnings", []),
                    "risk_id",
                )
                database.document_completeness.replace_one(
                    {
                        "tenant_id": document.tenant_id,
                        "document_id": document.document_id,
                    },
                    self._protect(
                        {
                            "tenant_id": document.tenant_id,
                            "document_id": document.document_id,
                            **payload.get("field_completeness", {}),
                            "updated_at": document.updated_at,
                        }
                    ),
                    upsert=True,
                )
                self._replace_many(database.model_runs, document, payload.get("model_runs", []), "run_id")
                self._insert_audits(database, document)
                self._errors.pop("mongodb", None)
            except Exception as exc:
                self._record_error("mongodb", exc)
        if settings.enable_redis:
            try:
                client = self._redis_client()
                status_payload = {
                    "tenant_id": document.tenant_id,
                    "document_id": document.document_id,
                    "status": document.status.value,
                    "updated_at": document.updated_at,
                }
                client.hset(
                    f"vision:documents:{document.tenant_id}:{document.document_id}",
                    mapping=status_payload,
                )
                client.publish(
                    f"vision:events:{document.tenant_id}:{document.document_id}",
                    json.dumps(status_payload),
                )
                self._errors.pop("redis", None)
            except Exception as exc:
                self._record_error("redis", exc)
        if settings.enable_minio and document.page_artifacts:
            self._store_artifacts(document)
        if settings.database_url:
            self._run_postgresql(
                lambda connection: self._postgresql_save_document(connection, document)
            )

    def save_auth_user(self, user: dict[str, Any], password_hash: str) -> None:
        if not settings.enable_mongodb:
            return
        try:
            tenant_id = str(user["tenant_id"])
            email = str(user["email"]).strip().lower()
            payload = {
                **user,
                "password_hash": password_hash,
                "email_hash": self._tenant_hash(tenant_id, email),
            }
            database = self._mongo_database_handle()
            database.users.replace_one(
                {"tenant_id": tenant_id, "user_id": user["user_id"]},
                self._protect(payload),
                upsert=True,
            )
            database.organisations.update_one(
                {"tenant_id": tenant_id},
                {
                    "$setOnInsert": {
                        "tenant_id": tenant_id,
                        "created_from_user": user["user_id"],
                    },
                    "$set": {"updated_at": user.get("created_at")},
                },
                upsert=True,
            )
            self._errors.pop("mongodb", None)
        except Exception as exc:
            self._record_error("mongodb", exc)

    def load_auth_users(self) -> list[dict[str, Any]]:
        return self._find_many("users", {}, "email_hash")

    def save_refresh_sessions(self, sessions: dict[str, dict[str, Any]]) -> None:
        if not settings.enable_mongodb:
            return
        try:
            collection = self._mongo_database_handle().refresh_sessions
            active_keys = list(sessions)
            if active_keys:
                collection.delete_many({"session_key": {"$nin": active_keys}})
            else:
                collection.delete_many({})
            for session_key, session in sessions.items():
                collection.replace_one(
                    {"session_key": session_key},
                    self._protect({"session_key": session_key, **session}),
                    upsert=True,
                )
            self._errors.pop("mongodb", None)
        except Exception as exc:
            self._record_error("mongodb", exc)

    def load_refresh_sessions(self) -> dict[str, dict[str, Any]]:
        return {
            str(item.pop("session_key")): item
            for item in self._find_many("refresh_sessions", {}, "created_at")
            if item.get("session_key")
        }

    def get_document(self, document_id: str, tenant_id: str) -> VisionDocumentRecord | None:
        payload = self._find_one(
            "documents",
            {"tenant_id": tenant_id, "document_id": document_id},
        )
        return VisionDocumentRecord.model_validate(payload) if payload else None

    def list_documents(self, tenant_id: str) -> list[VisionDocumentRecord]:
        records = self._find_many("documents", {"tenant_id": tenant_id}, "created_at")
        return [VisionDocumentRecord.model_validate(item) for item in records]

    def delete_document(self, document_id: str, tenant_id: str) -> None:
        if settings.enable_mongodb:
            try:
                database = self._mongo_database_handle()
                query = {"tenant_id": tenant_id, "document_id": document_id}
                for collection_name in (
                    "documents",
                    "pages",
                    "ocr_results",
                    "ocr_lines",
                    "extracted_fields",
                    "tables",
                    "risk_indicators",
                    "model_runs",
                    "review_tasks",
                    "questions",
                ):
                    database[collection_name].delete_many(query)
                self._errors.pop("mongodb", None)
            except Exception as exc:
                self._record_error("mongodb", exc)
        if settings.enable_redis:
            try:
                self._redis_client().delete(f"vision:documents:{tenant_id}:{document_id}")
                self._errors.pop("redis", None)
            except Exception as exc:
                self._record_error("redis", exc)
        if settings.enable_minio:
            try:
                client = self._minio_client()
                prefix = f"{tenant_id}/{document_id}/"
                for item in client.list_objects(settings.minio_bucket, prefix=prefix, recursive=True):
                    client.remove_object(settings.minio_bucket, item.object_name)
                self._errors.pop("minio", None)
            except Exception as exc:
                self._record_error("minio", exc)

    def save_job(self, job: dict[str, Any]) -> None:
        if settings.enable_mongodb:
            try:
                self._mongo_database_handle().vision_jobs.replace_one(
                    {"tenant_id": job.get("tenant_id"), "job_id": job["job_id"]},
                    self._protect(job),
                    upsert=True,
                )
                self._errors.pop("mongodb", None)
            except Exception as exc:
                self._record_error("mongodb", exc)
        if settings.enable_redis:
            try:
                self._redis_client().hset(
                    f"vision:jobs:{job.get('tenant_id')}:{job['job_id']}",
                    mapping={
                        key: json.dumps(value)
                        if isinstance(value, dict | list)
                        else str(value)
                        for key, value in job.items()
                    },
                )
                self._errors.pop("redis", None)
            except Exception as exc:
                self._record_error("redis", exc)

    def get_job(self, job_id: str, tenant_id: str) -> dict[str, Any] | None:
        return self._find_one(
            "vision_jobs",
            {"tenant_id": tenant_id, "job_id": job_id},
        )

    def save_review(self, review: VisionReviewTask, tenant_id: str) -> None:
        if not settings.enable_mongodb:
            return
        try:
            payload = {"tenant_id": tenant_id, **review.model_dump(mode="json")}
            self._mongo_database_handle().review_tasks.replace_one(
                {"tenant_id": tenant_id, "review_id": review.review_id},
                self._protect(payload),
                upsert=True,
            )
            self._errors.pop("mongodb", None)
        except Exception as exc:
            self._record_error("mongodb", exc)

    def get_review(self, review_id: str, tenant_id: str) -> VisionReviewTask | None:
        payload = self._find_one(
            "review_tasks",
            {"tenant_id": tenant_id, "review_id": review_id},
        )
        if payload:
            payload.pop("tenant_id", None)
        return VisionReviewTask.model_validate(payload) if payload else None

    def list_reviews(self, tenant_id: str) -> list[VisionReviewTask]:
        records = self._find_many("review_tasks", {"tenant_id": tenant_id}, "created_at")
        for record in records:
            record.pop("tenant_id", None)
        return [VisionReviewTask.model_validate(item) for item in records]

    def save_question(self, question: dict[str, Any]) -> None:
        if settings.enable_mongodb:
            try:
                self._mongo_database_handle().questions.replace_one(
                    {
                        "tenant_id": question.get("tenant_id"),
                        "question_id": question["question_id"],
                    },
                    self._protect(question),
                    upsert=True,
                )
                self._errors.pop("mongodb", None)
            except Exception as exc:
                self._record_error("mongodb", exc)
        if settings.database_url:
            self._run_postgresql(
                lambda connection: self._postgresql_save_retrieval_trace(
                    connection,
                    question,
                )
            )

    def get_question(self, question_id: str, tenant_id: str) -> dict[str, Any] | None:
        return self._find_one(
            "questions",
            {"tenant_id": tenant_id, "question_id": question_id},
        )

    def save_conversation(self, conversation: dict[str, Any]) -> None:
        """Persist tenant- and user-scoped conversational retrieval state."""
        if settings.enable_mongodb:
            try:
                payload = {
                    **conversation,
                    "tenant_id": conversation.get("organisation_id"),
                }
                expires_at = payload.get("expires_at")
                if isinstance(expires_at, str):
                    payload["expires_at"] = _utc_datetime(expires_at)
                self._mongo_database_handle().conversations.replace_one(
                    {
                        "tenant_id": conversation.get("organisation_id"),
                        "user_id": conversation.get("user_id"),
                        "conversation_id": conversation["conversation_id"],
                    },
                    self._protect(payload),
                    upsert=True,
                )
                self._errors.pop("mongodb", None)
            except Exception as exc:
                self._record_error("mongodb", exc)
        if settings.database_url:
            self._run_postgresql(
                lambda connection: self._postgresql_save_conversation(
                    connection,
                    conversation,
                )
            )

    def get_conversation(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        payload = self._find_one(
            "conversations",
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
        )
        if payload and isinstance(payload.get("expires_at"), datetime):
            payload["expires_at"] = payload["expires_at"].isoformat()
        if payload is None and settings.database_url:
            payload = self._run_postgresql(
                lambda connection: self._postgresql_get_conversation(
                    connection,
                    conversation_id,
                    tenant_id,
                    user_id,
                )
            )
        return payload

    def save_report(self, report: dict[str, Any], content: bytes | None = None) -> str | None:
        if settings.enable_mongodb:
            try:
                self._mongo_database_handle().reports.replace_one(
                    {
                        "tenant_id": report.get("tenant_id"),
                        "report_id": report["report_id"],
                    },
                    self._protect(report),
                    upsert=True,
                )
            except Exception as exc:
                self._record_error("mongodb", exc)
        if content is not None and settings.enable_minio:
            try:
                key = f"{report.get('tenant_id')}/reports/{report['report_id']}.pdf"
                self._minio_client().put_object(
                    settings.minio_bucket,
                    key,
                    BytesIO(content),
                    length=len(content),
                    content_type="application/pdf",
                )
                return f"s3://{settings.minio_bucket}/{key}"
            except Exception as exc:
                self._record_error("minio", exc)
        return None

    def get_report(self, report_id: str, tenant_id: str) -> VisionReportRecord | None:
        payload = self._find_one(
            "reports",
            {"tenant_id": tenant_id, "report_id": report_id},
        )
        return VisionReportRecord.model_validate(payload) if payload else None

    def list_reports(self, tenant_id: str) -> list[VisionReportRecord]:
        return [
            VisionReportRecord.model_validate(item)
            for item in self._find_many("reports", {"tenant_id": tenant_id}, "created_at")
        ]

    def save_narrative(self, narrative: VisionFinanceNarrative) -> None:
        if not settings.enable_mongodb:
            return
        try:
            self._mongo_database_handle().finance_narratives.replace_one(
                {
                    "tenant_id": narrative.tenant_id,
                    "narrative_id": narrative.narrative_id,
                },
                self._protect(narrative.model_dump(mode="json")),
                upsert=True,
            )
            self._errors.pop("mongodb", None)
        except Exception as exc:
            self._record_error("mongodb", exc)

    def get_narrative(self, narrative_id: str, tenant_id: str) -> VisionFinanceNarrative | None:
        payload = self._find_one(
            "finance_narratives",
            {"tenant_id": tenant_id, "narrative_id": narrative_id},
        )
        return VisionFinanceNarrative.model_validate(payload) if payload else None

    def list_narratives(self, tenant_id: str) -> list[VisionFinanceNarrative]:
        return [
            VisionFinanceNarrative.model_validate(item)
            for item in self._find_many(
                "finance_narratives",
                {"tenant_id": tenant_id},
                "created_at",
            )
        ]

    def latest_status(self, document_id: str, tenant_id: str) -> dict[str, str]:
        if not settings.enable_redis:
            return {}
        try:
            value = self._redis_client().hgetall(
                f"vision:documents:{tenant_id}:{document_id}"
            )
            self._errors.pop("redis", None)
            return dict(value)
        except Exception as exc:
            self._record_error("redis", exc)
            return {}

    def upsert_graph(
        self,
        tenant_id: str,
        document_id: str,
        graph: VisionGraphProjection,
    ) -> None:
        if not settings.enable_neo4j:
            return
        try:
            with self._neo4j_driver().session(database=settings.neo4j_database) as session:
                session.execute_write(
                    self._upsert_graph_tx,
                    tenant_id,
                    document_id,
                    graph.model_dump(mode="json"),
                )
            self._errors.pop("neo4j", None)
        except Exception as exc:
            self._record_error("neo4j", exc)

    def health(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "enabled": {
                "postgresql": bool(settings.database_url),
                "mongodb": settings.enable_mongodb,
                "redis": settings.enable_redis,
                "minio": settings.enable_minio,
                "neo4j": settings.enable_neo4j,
                "qdrant": settings.enable_qdrant,
            },
            "errors": dict(self._errors),
            "status": "degraded" if self._errors else "ready",
        }

    def _initialize_postgresql(self) -> None:
        self._run_postgresql(lambda connection: connection.execute("SELECT 1"))

    def _run_postgresql(self, operation: Any) -> Any:
        """Run one asyncpg unit of work without borrowing the caller's event loop."""
        if not settings.database_url:
            return None

        async def invoke() -> Any:
            import asyncpg

            connection = await asyncpg.connect(
                dsn=self._postgresql_dsn(),
                timeout=5,
                command_timeout=5,
            )
            try:
                return await operation(connection)
            finally:
                await connection.close()

        try:
            result = _run_async_in_thread(invoke)
            self._errors.pop("postgresql", None)
            return result
        except Exception as exc:
            self._record_error("postgresql", exc)
            return None

    def _postgresql_dsn(self) -> str:
        value = str(settings.database_url or "").strip()
        return re.sub(r"^postgresql\+asyncpg://", "postgresql://", value, count=1)

    async def _postgresql_save_document(
        self,
        connection: Any,
        document: VisionDocumentRecord,
    ) -> None:
        completeness = document.field_completeness.model_dump(mode="json")
        await connection.execute(
            """
            INSERT INTO vision_document_completeness (
                tenant_id, document_id, document_type, schema_available,
                required_fields, optional_fields, missing_required_fields,
                invalid_fields, disputed_fields, manually_resolved_fields, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb,
                $8::jsonb, $9::jsonb, $10::jsonb, $11
            )
            ON CONFLICT (tenant_id, document_id) DO UPDATE SET
                document_type = EXCLUDED.document_type,
                schema_available = EXCLUDED.schema_available,
                required_fields = EXCLUDED.required_fields,
                optional_fields = EXCLUDED.optional_fields,
                missing_required_fields = EXCLUDED.missing_required_fields,
                invalid_fields = EXCLUDED.invalid_fields,
                disputed_fields = EXCLUDED.disputed_fields,
                manually_resolved_fields = EXCLUDED.manually_resolved_fields,
                updated_at = EXCLUDED.updated_at
            """,
            document.tenant_id,
            document.document_id,
            document.document_type.value,
            bool(completeness.get("schema_available")),
            _json_parameter(completeness.get("required_fields", [])),
            _json_parameter(completeness.get("optional_fields", [])),
            _json_parameter(completeness.get("missing_required_fields", [])),
            _json_parameter(completeness.get("invalid_fields", [])),
            _json_parameter(completeness.get("disputed_fields", [])),
            _json_parameter(completeness.get("manually_resolved_fields", [])),
            _optional_utc_datetime(document.updated_at),
        )

        timing = document.metadata.get("processing_timing")
        if isinstance(timing, dict):
            pipeline_version = str(
                timing.get("pipeline_version")
                or document.metadata.get("pipeline_version")
                or ""
            ).strip()
            if pipeline_version:
                await connection.execute(
                    """
                    INSERT INTO vision_processing_timings (
                        tenant_id, document_id, pipeline_version, queued_at,
                        processing_started_at, processing_completed_at,
                        queued_duration_ms, processing_duration_ms,
                        stage_durations_ms, reliable, exclusion_reason, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8,
                        $9::jsonb, $10, $11, $12
                    )
                    ON CONFLICT (tenant_id, document_id, pipeline_version) DO UPDATE SET
                        queued_at = EXCLUDED.queued_at,
                        processing_started_at = EXCLUDED.processing_started_at,
                        processing_completed_at = EXCLUDED.processing_completed_at,
                        queued_duration_ms = EXCLUDED.queued_duration_ms,
                        processing_duration_ms = EXCLUDED.processing_duration_ms,
                        stage_durations_ms = EXCLUDED.stage_durations_ms,
                        reliable = EXCLUDED.reliable,
                        exclusion_reason = EXCLUDED.exclusion_reason,
                        updated_at = EXCLUDED.updated_at
                    """,
                    document.tenant_id,
                    document.document_id,
                    pipeline_version,
                    _optional_utc_datetime(timing.get("queued_at")),
                    _optional_utc_datetime(timing.get("processing_started_at")),
                    _optional_utc_datetime(timing.get("processing_completed_at")),
                    _nonnegative_milliseconds(timing.get("queued_duration_ms")),
                    _nonnegative_milliseconds(timing.get("processing_duration_ms")),
                    _json_parameter(timing.get("stage_durations_ms", {})),
                    bool(timing.get("reliable", False)),
                    _optional_text(timing.get("exclusion_reason")),
                    _optional_utc_datetime(document.updated_at),
                )

        for candidate in self._field_candidate_resolutions(document):
            await connection.execute(
                """
                INSERT INTO vision_field_candidate_resolutions (
                    tenant_id, document_id, candidate_id, schema_field,
                    evidence_fingerprint, score_components, resolution,
                    reason, extractor_version
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9)
                ON CONFLICT (tenant_id, document_id, candidate_id) DO UPDATE SET
                    schema_field = EXCLUDED.schema_field,
                    evidence_fingerprint = EXCLUDED.evidence_fingerprint,
                    score_components = EXCLUDED.score_components,
                    resolution = EXCLUDED.resolution,
                    reason = EXCLUDED.reason,
                    extractor_version = EXCLUDED.extractor_version
                """,
                document.tenant_id,
                document.document_id,
                candidate["candidate_id"],
                candidate["schema_field"],
                candidate["evidence_fingerprint"],
                _json_parameter(candidate["score_components"]),
                candidate["resolution"],
                candidate["reason"],
                candidate["extractor_version"],
            )

    def _field_candidate_resolutions(
        self,
        document: VisionDocumentRecord,
    ) -> list[dict[str, Any]]:
        diagnostics = document.metadata.get("extraction_diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        default_extractor = str(
            diagnostics.get("extractor")
            or document.metadata.get("pipeline_version")
            or "unknown"
        )
        resolutions: list[dict[str, Any]] = []
        for field in document.extracted_fields:
            evidence = field.evidence.model_dump(mode="json")
            evidence_metadata = evidence.get("metadata")
            evidence_metadata = evidence_metadata if isinstance(evidence_metadata, dict) else {}
            score_components = evidence_metadata.get("candidate_score_components")
            if not isinstance(score_components, dict):
                score_components = field.confidence_components
            disputed = field.validation_status not in {"valid", "not_evaluated"}
            fingerprint_payload = {
                "evidence_id": evidence.get("evidence_id"),
                "page": evidence.get("page_number"),
                "bounding_box": evidence.get("bounding_box"),
                "source_text": evidence.get("source_text"),
            }
            resolutions.append(
                {
                    "candidate_id": field.field_id,
                    "schema_field": field.schema_field or field.name,
                    "evidence_fingerprint": _fingerprint(fingerprint_payload),
                    "score_components": score_components,
                    "resolution": "disputed" if disputed else "selected",
                    "reason": (
                        field.validation_reason
                        if disputed and field.validation_reason
                        else "selected_by_extractor"
                    ),
                    "extractor_version": f"{field.model_name}-{field.model_version}",
                }
            )

        rejected = diagnostics.get("rejected_candidates")
        for item in rejected if isinstance(rejected, list) else []:
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason") or "candidate_not_selected")
            resolution = (
                "rejected_duplicate"
                if "duplicate" in reason
                else "rejected_conflict"
                if "conflict" in reason
                else "disputed"
            )
            identity_payload = {
                "field": item.get("field"),
                "value": item.get("value"),
                "page": item.get("page"),
                "strategy": item.get("strategy"),
                "evidence_box": item.get("evidence_box"),
                "reason": reason,
            }
            digest = _fingerprint(identity_payload)
            resolutions.append(
                {
                    "candidate_id": f"candidate-{digest[:20]}",
                    "schema_field": str(item.get("field") or "unknown"),
                    "evidence_fingerprint": digest,
                    "score_components": {"candidate_score": item.get("score")},
                    "resolution": resolution,
                    "reason": reason,
                    "extractor_version": default_extractor,
                }
            )
        return resolutions

    async def _postgresql_save_conversation(
        self,
        connection: Any,
        conversation: dict[str, Any],
    ) -> None:
        tenant_id = str(
            conversation.get("organisation_id") or conversation.get("tenant_id") or ""
        ).strip()
        user_id = str(conversation.get("user_id") or "").strip()
        conversation_id = str(conversation.get("conversation_id") or "").strip()
        if not tenant_id or not user_id or not conversation_id:
            raise ValueError("Conversation tenant, user and conversation identifiers are required")
        selected_scope = self._protect(conversation.get("selected_scope") or {})
        pending = self._protect(conversation.get("pending_clarification") or {})
        turns = self._protect(conversation.get("turns") or [])
        updated_at = _optional_utc_datetime(conversation.get("updated_at")) or datetime.now(UTC)
        expires_at = _optional_utc_datetime(conversation.get("expires_at"))
        if expires_at is None:
            raise ValueError("Conversation expiry is required")
        await connection.execute(
            """
            INSERT INTO vision_conversations (
                tenant_id, user_id, conversation_id, selected_scope,
                pending_clarification, last_query_type, turns, expires_at, updated_at
            ) VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7::jsonb, $8, $9)
            ON CONFLICT (tenant_id, conversation_id) DO UPDATE SET
                selected_scope = EXCLUDED.selected_scope,
                pending_clarification = EXCLUDED.pending_clarification,
                last_query_type = EXCLUDED.last_query_type,
                turns = EXCLUDED.turns,
                expires_at = EXCLUDED.expires_at,
                updated_at = EXCLUDED.updated_at
            WHERE vision_conversations.user_id = EXCLUDED.user_id
            """,
            tenant_id,
            user_id,
            conversation_id,
            _json_parameter(selected_scope),
            _json_parameter(pending),
            _optional_text(conversation.get("last_query_type")),
            _json_parameter(turns),
            expires_at,
            updated_at,
        )

    async def _postgresql_get_conversation(
        self,
        connection: Any,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        row = await connection.fetchrow(
            """
            SELECT tenant_id, user_id, conversation_id, selected_scope,
                   pending_clarification, last_query_type, turns, expires_at, updated_at
            FROM vision_conversations
            WHERE tenant_id = $1
              AND user_id = $2
              AND conversation_id = $3
              AND expires_at > now()
            """,
            tenant_id,
            user_id,
            conversation_id,
        )
        if row is None:
            return None
        payload = dict(row)
        expires_at = payload.get("expires_at")
        updated_at = payload.get("updated_at")
        return {
            "conversation_id": str(payload["conversation_id"]),
            "user_id": str(payload["user_id"]),
            "organisation_id": str(payload["tenant_id"]),
            "selected_scope": self._unprotect(
                _decoded_json(payload.get("selected_scope"), {})
            ),
            "pending_clarification": self._unprotect(
                _decoded_json(payload.get("pending_clarification"), {})
            ),
            "last_query_type": str(payload.get("last_query_type") or ""),
            "turns": self._unprotect(_decoded_json(payload.get("turns"), [])),
            "expires_at": (
                expires_at.isoformat() if isinstance(expires_at, datetime) else str(expires_at)
            ),
            "updated_at": (
                updated_at.isoformat() if isinstance(updated_at, datetime) else str(updated_at)
            ),
        }

    async def _postgresql_save_retrieval_trace(
        self,
        connection: Any,
        question: dict[str, Any],
    ) -> None:
        answer = question.get("answer")
        answer = answer if isinstance(answer, dict) else {}
        trace = answer.get("retrieval_trace")
        trace = trace if isinstance(trace, dict) else {}
        trace_id = str(trace.get("trace_id") or answer.get("retrieval_trace_id") or "").strip()
        tenant_id = str(question.get("tenant_id") or "").strip()
        if not trace_id or not tenant_id:
            return
        await connection.execute(
            """
            INSERT INTO vision_retrieval_traces (
                tenant_id, trace_id, conversation_id, question_id,
                sources_checked, candidate_counts, selected_evidence,
                retriever_versions, retrieval_duration_ms, created_at
            ) VALUES (
                $1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb,
                $8::jsonb, $9, COALESCE($10::timestamptz, now())
            )
            ON CONFLICT (tenant_id, trace_id) DO UPDATE SET
                conversation_id = EXCLUDED.conversation_id,
                question_id = EXCLUDED.question_id,
                sources_checked = EXCLUDED.sources_checked,
                candidate_counts = EXCLUDED.candidate_counts,
                selected_evidence = EXCLUDED.selected_evidence,
                retriever_versions = EXCLUDED.retriever_versions,
                retrieval_duration_ms = EXCLUDED.retrieval_duration_ms,
                created_at = EXCLUDED.created_at
            """,
            tenant_id,
            trace_id,
            _optional_text(answer.get("conversation_id")),
            _optional_text(question.get("question_id")),
            _json_parameter(trace.get("sources_checked", [])),
            _json_parameter(trace.get("candidate_counts", {})),
            _json_parameter(trace.get("selected_evidence", [])),
            _json_parameter(trace.get("retriever_versions", {})),
            _nonnegative_milliseconds(trace.get("retrieval_duration_ms")) or 0,
            _optional_utc_datetime(question.get("created_at")),
        )

    def _initialize_mongodb(self) -> None:
        try:
            database = self._mongo_database_handle()
            self._mongo_client.admin.command("ping")
            database.documents.create_index([("tenant_id", 1), ("document_id", 1)], unique=True)
            database.documents.create_index([("tenant_id", 1), ("status", 1), ("created_at", -1)])
            database.documents.create_index([("tenant_id", 1), ("filename", 1)])
            database.documents.create_index([("tenant_id", 1), ("document_type", 1)])
            database.documents.create_index([("tenant_id", 1), ("updated_at", -1)])
            database.extracted_fields.create_index([("tenant_id", 1), ("document_id", 1), ("name", 1)])
            database.ocr_lines.create_index(
                [("tenant_id", 1), ("document_id", 1), ("page", 1), ("reading_order", 1)]
            )
            database.extracted_fields.create_index([("tenant_id", 1), ("name", 1), ("value_hash", 1)])
            database.review_tasks.create_index([("tenant_id", 1), ("priority", -1), ("status", 1)])
            database.review_tasks.create_index([("tenant_id", 1), ("filename", 1)])
            database.risk_indicators.create_index(
                [("tenant_id", 1), ("document_id", 1), ("category", 1), ("severity", 1)]
            )
            database.processing_warnings.create_index(
                [("tenant_id", 1), ("document_id", 1), ("severity", 1)]
            )
            database.document_completeness.create_index(
                [("tenant_id", 1), ("document_id", 1)],
                unique=True,
            )
            database.audit_events.create_index([("tenant_id", 1), ("event_id", 1)], unique=True)
            database.questions.create_index([("tenant_id", 1), ("question_id", 1)], unique=True)
            database.conversations.create_index(
                [("tenant_id", 1), ("user_id", 1), ("conversation_id", 1)],
                unique=True,
            )
            database.conversations.create_index("expires_at", expireAfterSeconds=0)
            database.reports.create_index([("tenant_id", 1), ("report_id", 1)], unique=True)
            database.reports.create_index(
                [("tenant_id", 1), ("identity_hash", 1), ("version", 1)],
                unique=True,
                partialFilterExpression={"identity_hash": {"$gt": ""}},
            )
            database.finance_narratives.create_index(
                [("tenant_id", 1), ("narrative_id", 1)],
                unique=True,
            )
            database.finance_narratives.create_index(
                [("tenant_id", 1), ("identity_hash", 1), ("version", 1)],
                unique=True,
                partialFilterExpression={"identity_hash": {"$gt": ""}},
            )
            database.vision_jobs.create_index([("tenant_id", 1), ("job_id", 1)], unique=True)
            database.users.create_index([("tenant_id", 1), ("user_id", 1)], unique=True)
            database.users.create_index([("tenant_id", 1), ("email_hash", 1)], unique=True)
            database.organisations.create_index("tenant_id", unique=True)
            database.refresh_sessions.create_index("session_key", unique=True)
            database.refresh_sessions.create_index("expires_at")
            self._errors.pop("mongodb", None)
        except Exception as exc:
            self._record_error("mongodb", exc)

    def _initialize_redis(self) -> None:
        try:
            self._redis_client().ping()
            self._errors.pop("redis", None)
        except Exception as exc:
            self._record_error("redis", exc)

    def _initialize_minio(self) -> None:
        try:
            client = self._minio_client()
            if not client.bucket_exists(settings.minio_bucket):
                client.make_bucket(settings.minio_bucket)
            self._errors.pop("minio", None)
        except Exception as exc:
            self._record_error("minio", exc)

    def _initialize_neo4j(self) -> None:
        try:
            driver = self._neo4j_driver()
            driver.verify_connectivity()
            with driver.session(database=settings.neo4j_database) as session:
                session.run(
                    "CREATE CONSTRAINT vision_entity IF NOT EXISTS "
                    "FOR (n:VisionEntity) REQUIRE (n.tenant_id, n.entity_id) IS UNIQUE"
                ).consume()
                session.run(
                    "CREATE CONSTRAINT vision_document IF NOT EXISTS "
                    "FOR (d:VisionDocument) REQUIRE (d.tenant_id, d.document_id) IS UNIQUE"
                ).consume()
            self._errors.pop("neo4j", None)
        except Exception as exc:
            self._record_error("neo4j", exc)

    def _mongo_database_handle(self) -> Any:
        if self._mongo_database is None:
            from pymongo import MongoClient

            self._mongo_client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
            database_name = urlsplit(settings.mongodb_uri).path.strip("/").split("/", 1)[0] or "documind"
            self._mongo_database = self._mongo_client[database_name]
        return self._mongo_database

    def _redis_client(self) -> Any:
        if self._redis is None:
            from redis import Redis

            self._redis = Redis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    def _minio_client(self) -> Any:
        if self._minio is None:
            from minio import Minio

            if not settings.minio_access_key or not settings.minio_secret_key:
                raise ValueError("MinIO credentials are not configured")
            endpoint = settings.minio_endpoint.removeprefix("http://").removeprefix("https://")
            self._minio = Minio(
                endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_endpoint.startswith("https://"),
            )
            if not self._minio.bucket_exists(settings.minio_bucket):
                self._minio.make_bucket(settings.minio_bucket)
        return self._minio

    def _neo4j_driver(self) -> Any:
        if self._neo4j is None:
            from neo4j import GraphDatabase

            auth = (
                (settings.neo4j_user, settings.neo4j_password)
                if settings.neo4j_user and settings.neo4j_password
                else None
            )
            self._neo4j = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        return self._neo4j

    def _replace_many(
        self,
        collection: Any,
        document: VisionDocumentRecord,
        records: list[dict[str, Any]],
        identity_key: str,
    ) -> None:
        collection.delete_many(
            {"tenant_id": document.tenant_id, "document_id": document.document_id}
        )
        if records:
            prepared = [
                    self._protect(
                        {
                            "tenant_id": document.tenant_id,
                            "document_id": document.document_id,
                            **record,
                        }
                    )
                    for record in records
                    if record.get(identity_key) is not None
                ]
            if prepared:
                collection.insert_many(prepared)

    def _field_records(self, document: VisionDocumentRecord) -> list[dict[str, Any]]:
        records = []
        for field in document.extracted_fields:
            payload = field.model_dump(mode="json")
            value = field.reviewed_value or field.value
            payload["value_hash"] = hashlib.sha256(
                f"{document.tenant_id}:{value}".encode()
            ).hexdigest()
            if "bank" in field.name.lower() or "account" in field.name.lower():
                payload["masked_value"] = self._mask(value)
            records.append(payload)
        return records

    def _insert_audits(self, database: Any, document: VisionDocumentRecord) -> None:
        for event in document.audit_events:
            payload = {
                "tenant_id": document.tenant_id,
                **event.model_dump(mode="json"),
                "document_version": document.updated_at,
                "model_versions": sorted({run.model_version for run in document.model_runs}),
            }
            database.audit_events.update_one(
                {"tenant_id": document.tenant_id, "event_id": event.event_id},
                {"$setOnInsert": self._protect(payload)},
                upsert=True,
            )

    def _store_artifacts(self, document: VisionDocumentRecord) -> None:
        try:
            client = self._minio_client()
            stored = document.metadata.setdefault("object_storage", {})
            for artifact in document.page_artifacts:
                for kind, raw_path in (
                    ("original_page", artifact.original_image_path),
                    ("processed_page", artifact.processed_image_path),
                ):
                    path = Path(raw_path)
                    if not path.is_file():
                        continue
                    key = (
                        f"{document.tenant_id}/{document.document_id}/pages/"
                        f"{kind}/{path.name}"
                    )
                    marker = f"{kind}:{artifact.page_number}"
                    if marker in stored:
                        continue
                    content = path.read_bytes()
                    client.put_object(
                        settings.minio_bucket,
                        key,
                        BytesIO(content),
                        length=len(content),
                        content_type="image/png",
                    )
                    stored[marker] = f"s3://{settings.minio_bucket}/{key}"
            self._errors.pop("minio", None)
        except Exception as exc:
            self._record_error("minio", exc)

    def _protect(self, value: Any, field_name: str | None = None) -> Any:
        if not settings.data_encryption_key:
            return value
        if isinstance(value, dict):
            return {key: self._protect(child, key) for key, child in value.items()}
        if isinstance(value, list):
            return [self._protect(child, field_name) for child in value]
        sensitive_fields = {
            "email",
            "name",
            "parsed_text",
            "text",
            "source_text",
            "value",
            "reviewed_value",
            "old_value",
            "new_value",
            "comment",
            "reviewer_comments",
            "question",
            "answer",
        }
        if isinstance(value, str) and len(value) >= 1 and field_name in sensitive_fields:
            return {"__encrypted__": self._encrypt(value)}
        return value

    def _unprotect(self, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"__encrypted__"}:
                return self._decrypt(str(value["__encrypted__"]))
            return {
                key: self._unprotect(child)
                for key, child in value.items()
                if key != "_id"
            }
        if isinstance(value, list):
            return [self._unprotect(child) for child in value]
        return value

    def _encrypt(self, value: str) -> str:
        import os

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = base64.urlsafe_b64decode(settings.data_encryption_key + "===")
        if len(key) != 32:
            raise ValueError("DOCUMIND_DATA_ENCRYPTION_KEY must encode exactly 32 bytes")
        nonce = os.urandom(12)
        encrypted = AESGCM(key).encrypt(nonce, value.encode("utf-8"), None)
        return base64.urlsafe_b64encode(nonce + encrypted).decode("ascii")

    def _decrypt(self, value: str) -> str:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = base64.urlsafe_b64decode(settings.data_encryption_key + "===")
        payload = base64.urlsafe_b64decode(value.encode("ascii"))
        return AESGCM(key).decrypt(payload[:12], payload[12:], None).decode("utf-8")

    def _find_one(self, collection_name: str, query: dict[str, Any]) -> dict[str, Any] | None:
        if not settings.enable_mongodb:
            return None
        try:
            payload = self._mongo_database_handle()[collection_name].find_one(query, {"_id": 0})
            self._errors.pop("mongodb", None)
            return self._unprotect(payload) if payload else None
        except Exception as exc:
            self._record_error("mongodb", exc)
            return None

    def _find_many(
        self,
        collection_name: str,
        query: dict[str, Any],
        sort_key: str,
    ) -> list[dict[str, Any]]:
        if not settings.enable_mongodb:
            return []
        try:
            cursor = self._mongo_database_handle()[collection_name].find(query, {"_id": 0})
            payloads = [self._unprotect(item) for item in cursor.sort(sort_key, -1)]
            self._errors.pop("mongodb", None)
            return payloads
        except Exception as exc:
            self._record_error("mongodb", exc)
            return []

    @staticmethod
    def _upsert_graph_tx(
        tx: Any,
        tenant_id: str,
        document_id: str,
        graph: dict[str, Any],
    ) -> None:
        tx.run(
            """
            MERGE (d:VisionDocument {tenant_id: $tenant_id, document_id: $document_id})
            SET d.updated_at = datetime()
            """,
            tenant_id=tenant_id,
            document_id=document_id,
        ).consume()

        edge_ids = [str(edge.get("id", "")) for edge in graph.get("edges", [])]
        node_ids = [
            str(node.get("id", ""))
            for node in graph.get("nodes", [])
            if str(node.get("type", "")) in GRAPH_NODE_LABELS
        ]
        tx.run(
            """
            MATCH ()-[r]->()
            WHERE r.tenant_id = $tenant_id
              AND r.document_id = $document_id
              AND r.edge_id IS NOT NULL
              AND NOT r.edge_id IN $edge_ids
            DELETE r
            """,
            tenant_id=tenant_id,
            document_id=document_id,
            edge_ids=edge_ids,
        ).consume()

        tx.run(
            """
            MATCH (d:VisionDocument {tenant_id: $tenant_id, document_id: $document_id})
                  -[projection:PROJECTS]->(node:VisionEntity)
            WHERE NOT node.entity_id IN $node_ids
            DELETE projection
            WITH DISTINCT node
            WHERE NOT (node)<-[:PROJECTS]-(:VisionDocument)
            DETACH DELETE node
            """,
            tenant_id=tenant_id,
            document_id=document_id,
            node_ids=node_ids,
        ).consume()

        for node in graph.get("nodes", []):
            node_label = str(node.get("type", ""))
            if node_label not in GRAPH_NODE_LABELS:
                continue
            properties = _neo4j_properties(node)
            tx.run(
                f"""
                MATCH (d:VisionDocument {{tenant_id: $tenant_id, document_id: $document_id}})
                MERGE (n:VisionEntity:{node_label} {{tenant_id: $tenant_id, entity_id: $entity_id}})
                SET n += $properties, n.tenant_id = $tenant_id, n.entity_id = $entity_id
                MERGE (d)-[:PROJECTS]->(n)
                """,
                tenant_id=tenant_id,
                document_id=document_id,
                entity_id=str(node["id"]),
                properties=properties,
            ).consume()

        for edge in graph.get("edges", []):
            relationship_type = str(edge.get("type", ""))
            if relationship_type not in GRAPH_RELATIONSHIP_TYPES:
                continue
            properties = _neo4j_properties(edge)
            tx.run(
                f"""
                MATCH (source:VisionEntity {{tenant_id: $tenant_id, entity_id: $source}})
                MATCH (target:VisionEntity {{tenant_id: $tenant_id, entity_id: $target}})
                MERGE (source)-[r:{relationship_type} {{tenant_id: $tenant_id, edge_id: $edge_id}}]->(target)
                SET r += $properties, r.tenant_id = $tenant_id,
                    r.edge_id = $edge_id, r.relationship_type = $relationship_type
                """,
                tenant_id=tenant_id,
                source=str(edge["source"]),
                target=str(edge["target"]),
                edge_id=str(edge["id"]),
                relationship_type=relationship_type,
                properties=properties,
            ).consume()

    def _record_error(self, backend: str, exc: Exception) -> None:
        message = re.sub(
            r"(?i)(password|pwd|secret|token|key)=([^&\s]+)",
            r"\1=****",
            str(exc),
        )
        message = re.sub(r"(?i)://([^:@/\s]+):([^@/\s]+)@", "://****:****@", message)
        self._errors[backend] = f"{type(exc).__name__}: {message[:500]}"

    def _mask(self, value: str) -> str:
        return "*" * max(0, len(value) - 4) + value[-4:]

    def _tenant_hash(self, tenant_id: str, value: str) -> str:
        return hashlib.sha256(f"{tenant_id}:{value}".encode()).hexdigest()


def _run_async_in_thread(factory: Any) -> Any:
    result: list[Any] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(asyncio.run(factory()))
        except BaseException as exc:  # pragma: no cover - re-raised on the caller thread
            errors.append(exc)

    thread = threading.Thread(target=worker, name="vision-postgresql", daemon=True)
    thread.start()
    thread.join(timeout=15)
    if thread.is_alive():
        raise TimeoutError("PostgreSQL operation exceeded the bounded execution window")
    if errors:
        raise errors[0]
    return result[0] if result else None


def _json_parameter(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _decoded_json(value: Any, default: Any) -> Any:
    if isinstance(value, str | bytes | bytearray):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default
    return value if isinstance(value, dict | list) else default


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json_parameter(value).encode()).hexdigest()


def _optional_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _utc_datetime(value)
    except ValueError:
        return None


def _nonnegative_milliseconds(value: Any) -> int | None:
    try:
        milliseconds = float(value)
        if milliseconds < 0:
            return None
        return round(milliseconds)
    except (OverflowError, TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _neo4j_properties(payload: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None or isinstance(value, str | int | float | bool):
            properties[key] = value
        elif isinstance(value, list) and all(
            item is None or isinstance(item, str | int | float | bool)
            for item in value
        ):
            properties[key] = value
        else:
            properties[key] = json.dumps(value, sort_keys=True)
    return properties


vision_persistence = VisionPersistenceCoordinator()
