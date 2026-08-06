from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.util import find_spec
from io import BytesIO
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_URL, uuid5

from app.core.config import settings
from app.models.vision import (
    VisionDocumentRecord,
    VisionEvidenceChunk,
    VisionGraphProjection,
    VisionReviewTask,
)
from app.workers.celery_app import celery_readiness


class InfrastructureUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class InfrastructureSpec:
    backend_id: str
    role: str
    enabled: bool
    package: str
    endpoint: str
    enable_hint: str
    safe_config: dict[str, Any]
    notes: list[str]


class VisionInfrastructureReadinessService:
    """Readiness and adapter-gating for optional production backends.

    The local Vision AI runtime intentionally stays runnable without external
    services. PostgreSQL is the exception to the configuration-only checks: when
    ``DATABASE_URL`` is present, readiness performs an authenticated ``SELECT 1``
    so operators can distinguish a configured URL from a usable database.
    """

    def __init__(
        self,
        package_checker: Callable[[str], bool] | None = None,
        postgres_probe: Callable[[str], Awaitable[dict[str, str]]] | None = None,
    ) -> None:
        self._package_checker = package_checker or self._installed
        self._postgres_probe = postgres_probe or self._probe_postgresql

    def readiness(self) -> dict[str, Any]:
        """Synchronous facade for scripts and tests without an active event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.readiness_async())
        raise RuntimeError("Use 'await readiness_async()' from asynchronous code")

    async def readiness_async(self) -> dict[str, Any]:
        services = [await self.service_status_async(spec.backend_id) for spec in self._specs()]
        services.append(celery_readiness())
        postgresql = next(
            service for service in services if service["backend_id"] == "postgresql"
        )
        enabled_services = [service for service in services if service["enabled"]]
        blocked = [
            service
            for service in enabled_services
            if service["status"] not in {"ready_for_connection", "active"}
        ]
        if blocked:
            status = "blocked"
        elif enabled_services:
            status = "external_backends_ready_for_connection"
        else:
            status = "local_runtime_only"

        return {
            "status": status,
            "default_runtime": "atomic_local_json_artifacts_and_fastapi_background_tasks",
            "network_probe": postgresql.get("network_probe", "not_executed"),
            "secret_policy": "Endpoints and credentials are redacted; bank/account values must be masked in logs and UI.",
            "services": services,
            "local_fallbacks": [
                "Atomic local JSON state repository",
                "VisionProcessingWorker",
                "local-hashing-v1",
                "local-graph-projection-v1",
            ],
            "production_gate": (
                "Configure DATABASE_URL for PostgreSQL; for other external services set the matching "
                "ENABLE_* flag. Install each driver and run deployment smoke tests before switching "
                "a tenant from local fallbacks."
            ),
        }

    def service_status(self, backend_id: str) -> dict[str, Any]:
        """Synchronous facade for callers outside an event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.service_status_async(backend_id))
        raise RuntimeError("Use 'await service_status_async()' from asynchronous code")

    async def service_status_async(self, backend_id: str) -> dict[str, Any]:
        spec = next((item for item in self._specs() if item.backend_id == backend_id), None)
        if not spec:
            raise ValueError(f"Unknown Vision infrastructure backend: {backend_id}")

        installed = self._package_checker(spec.package)
        if backend_id == "mongodb" and not installed:
            installed = self._package_checker("motor")
        probe: dict[str, str] = {}
        network_probe = "not_executed"
        if not spec.enabled:
            status = "disabled"
            reason = spec.enable_hint
        elif not installed:
            status = "unavailable"
            reason = f"Feature flag is enabled but Python package is missing: {spec.package}."
        elif backend_id == "minio" and (
            not settings.minio_access_key or not settings.minio_secret_key
        ):
            status = "misconfigured"
            reason = "MinIO is enabled but access credentials are not configured."
        elif backend_id == "neo4j" and (
            not settings.neo4j_user or not settings.neo4j_password
        ):
            status = "misconfigured"
            reason = "Neo4j is enabled but database credentials are not configured."
        elif not spec.endpoint:
            status = "misconfigured"
            reason = "Feature flag is enabled but no endpoint is configured."
        elif backend_id == "postgresql":
            try:
                probe = await self._postgres_probe(_normalise_postgresql_url(spec.endpoint))
            except (TypeError, ValueError):
                status = "misconfigured"
                reason = "DATABASE_URL must use the postgresql or postgresql+asyncpg scheme."
            except Exception as exc:
                network_probe = "authenticated_postgresql_select_1_failed"
                status = "unavailable"
                reason = (
                    "Authenticated PostgreSQL SELECT 1 failed "
                    f"({type(exc).__name__}); verify the safe endpoint and database service."
                )
            else:
                network_probe = "authenticated_postgresql_select_1_succeeded"
                status = "active"
                reason = "Authenticated PostgreSQL SELECT 1 succeeded."
        else:
            status = "ready_for_connection"
            reason = "Driver package is importable and configuration is present; no network probe was executed."

        result = {
            "backend_id": spec.backend_id,
            "role": spec.role,
            "enabled": spec.enabled,
            "package": spec.package,
            "installed": installed,
            "status": status,
            "reason": reason,
            "endpoint": _redact_url(spec.endpoint),
            "safe_config": spec.safe_config,
            "notes": spec.notes,
        }
        if probe:
            result["server"] = {
                "database": probe["database"],
                "version": probe["version"],
                "select_1": True,
            }
        if backend_id == "postgresql":
            result["network_probe"] = network_probe
        return result

    def require_ready(self, backend_id: str) -> None:
        status = self.service_status(backend_id)
        if status["status"] not in {"ready_for_connection", "active"}:
            raise InfrastructureUnavailableError(
                f"{status['backend_id']} is {status['status']}: {status['reason']}"
            )

    def _specs(self) -> list[InfrastructureSpec]:
        return [
            InfrastructureSpec(
                backend_id="postgresql",
                role="Operational metadata, document lineage, evidence, workflows, audits, and reports.",
                enabled=bool(settings.database_url),
                package="asyncpg",
                endpoint=settings.database_url or "",
                enable_hint="Configure DATABASE_URL in backend/.env to enable PostgreSQL.",
                safe_config=_postgresql_safe_config(settings.database_url),
                notes=[
                    "Readiness performs an authenticated SELECT 1 with a bounded timeout.",
                    "Run scripts/migrate_postgres.py --execute to apply repository migrations.",
                ],
            ),
            InfrastructureSpec(
                backend_id="mongodb",
                role="Persistent documents, pages, OCR, fields, tables, risks, reviews, reports, and audits.",
                enabled=settings.enable_mongodb,
                package="pymongo",
                endpoint=settings.mongodb_uri,
                enable_hint="Set ENABLE_MONGODB=true and configure MONGODB_URI.",
                safe_config={
                    "uri": _redact_url(settings.mongodb_uri),
                    "database": _mongo_database_name(settings.mongodb_uri),
                    "indexes": [
                        "organisation_id",
                        "document_id",
                        "vendor_id",
                        "invoice_number",
                        "purchase_order_number",
                        "contract_number",
                        "bank_account_hash",
                        "status",
                        "created_at",
                    ],
                },
                notes=[
                    "Async adapter contract uses the maintained PyMongo asynchronous API.",
                    "Sensitive bank identifiers should be hashed before indexing.",
                ],
            ),
            InfrastructureSpec(
                backend_id="redis",
                role="Celery broker, processing status cache, retry locks, rate-limit counters, and transient events.",
                enabled=settings.enable_redis,
                package="redis",
                endpoint=settings.redis_url,
                enable_hint="Set ENABLE_REDIS=true and configure REDIS_URL.",
                safe_config={"url": _redact_url(settings.redis_url), "queue_namespace": "vision"},
                notes=["FastAPI background tasks remain the local default until Redis/Celery is enabled."],
            ),
            InfrastructureSpec(
                backend_id="minio",
                role="Original documents, rendered pages, processed images, cropped evidence, heatmaps, and reports.",
                enabled=settings.enable_minio,
                package="minio",
                endpoint=settings.minio_endpoint,
                enable_hint="Set ENABLE_MINIO=true and configure MINIO_ENDPOINT/MINIO_BUCKET.",
                safe_config={
                    "endpoint": _redact_endpoint(settings.minio_endpoint),
                    "bucket": settings.minio_bucket,
                    "access_key": _mask_identifier(settings.minio_access_key),
                },
                notes=["Secret keys are never returned by readiness endpoints."],
            ),
            InfrastructureSpec(
                backend_id="qdrant",
                role="Dense embeddings, sparse vectors, metadata filters, and evidence chunk retrieval.",
                enabled=settings.enable_qdrant,
                package="qdrant_client",
                endpoint=settings.qdrant_url,
                enable_hint="Set ENABLE_QDRANT=true and configure QDRANT_URL.",
                safe_config={
                    "url": _redact_url(settings.qdrant_url),
                    "collection": "vision_evidence_chunks",
                    "payload_indexes": ["tenant_id", "document_id", "document_type", "review_status"],
                },
                notes=["BGE-M3 embedding generation is separately gated by the model registry."],
            ),
            InfrastructureSpec(
                backend_id="neo4j",
                role="Verified entities, trusted relationships, proposed edges, graph evidence, and GraphRAG paths.",
                enabled=settings.enable_neo4j,
                package="neo4j",
                endpoint=settings.neo4j_uri,
                enable_hint="Set ENABLE_NEO4J=true and configure NEO4J_URI plus deployment credentials.",
                safe_config={
                    "uri": _redact_url(settings.neo4j_uri),
                    "instance_id": settings.neo4j_instance_id,
                    "node_labels": [
                        "Vendor",
                        "Document",
                        "Invoice",
                        "PurchaseOrder",
                        "Contract",
                        "BankAccount",
                        "Reviewer",
                    ],
                    "relationship_types": [
                        "SUBMITTED",
                        "REFERENCES",
                        "PAID_TO",
                        "APPROVED_BY",
                        "GOVERNED_BY",
                        "DUPLICATE_OF",
                        "MATCHES",
                        "CONFLICTS_WITH",
                    ],
                },
                notes=["LLM-created graph edges must enter with proposed status and evidence."],
            ),
        ]

    async def _probe_postgresql(self, database_url: str) -> dict[str, str]:
        import asyncpg

        connection = await asyncpg.connect(
            dsn=database_url,
            timeout=5,
            command_timeout=10,
        )
        try:
            select_one = await connection.fetchval("SELECT 1")
            if select_one != 1:
                raise RuntimeError("PostgreSQL SELECT 1 returned an unexpected result")
            version = str(await connection.fetchval("SHOW server_version"))
            database = str(await connection.fetchval("SELECT current_database()"))
            return {"version": version, "database": database}
        finally:
            await connection.close()

    def _installed(self, package: str) -> bool:
        try:
            return find_spec(package) is not None
        except (ImportError, ValueError):
            return False


class AsyncMongoVisionRepository:
    """Optional PyMongo asynchronous repository for production persistence."""

    def __init__(self, database: Any | None = None) -> None:
        if database is None:
            vision_infrastructure_readiness_service.require_ready("mongodb")
            from pymongo import AsyncMongoClient

            client = AsyncMongoClient(settings.mongodb_uri)
            database = client[_mongo_database_name(settings.mongodb_uri)]
        self.database = database

    async def save_document(self, document: VisionDocumentRecord) -> None:
        payload = document.model_dump(mode="json")
        await self.database.documents.replace_one(
            {"tenant_id": document.tenant_id, "document_id": document.document_id},
            payload,
            upsert=True,
        )

    async def get_document(self, document_id: str, tenant_id: str) -> VisionDocumentRecord | None:
        payload = await self.database.documents.find_one(
            {"tenant_id": tenant_id, "document_id": document_id},
            {"_id": 0},
        )
        return VisionDocumentRecord.model_validate(payload) if payload else None

    async def list_documents(self, tenant_id: str | None = None) -> list[VisionDocumentRecord]:
        query = {"tenant_id": tenant_id} if tenant_id else {}
        cursor = self.database.documents.find(query, {"_id": 0}).sort("created_at", -1)
        return [VisionDocumentRecord.model_validate(item) async for item in cursor]

    async def save_job(self, job: dict[str, Any]) -> None:
        await self.database.vision_jobs.replace_one(
            {"tenant_id": job["tenant_id"], "job_id": job["job_id"]},
            job,
            upsert=True,
        )

    async def get_job(self, job_id: str, tenant_id: str) -> dict[str, Any] | None:
        return await self.database.vision_jobs.find_one(
            {"tenant_id": tenant_id, "job_id": job_id},
            {"_id": 0},
        )

    async def save_review(self, review: VisionReviewTask, tenant_id: str) -> None:
        payload = {"tenant_id": tenant_id, **review.model_dump(mode="json")}
        await self.database.review_tasks.replace_one(
            {"tenant_id": tenant_id, "review_id": review.review_id},
            payload,
            upsert=True,
        )

    async def get_review(self, review_id: str, tenant_id: str) -> VisionReviewTask | None:
        payload = await self.database.review_tasks.find_one(
            {"tenant_id": tenant_id, "review_id": review_id},
            {"_id": 0},
        )
        if payload:
            payload.pop("tenant_id", None)
        return VisionReviewTask.model_validate(payload) if payload else None


class MinIOVisionObjectStorage:
    """Optional S3-compatible object storage for source and evidence artefacts."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            vision_infrastructure_readiness_service.require_ready("minio")
            from minio import Minio

            endpoint = settings.minio_endpoint.removeprefix("http://").removeprefix("https://")
            secure = settings.minio_endpoint.startswith("https://")
            client = Minio(
                endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=secure,
            )
        self.client = client

    def put_bytes(self, object_key: str, content: bytes, content_type: str = "application/octet-stream") -> str:
        if not self.client.bucket_exists(settings.minio_bucket):
            self.client.make_bucket(settings.minio_bucket)
        self.client.put_object(
            settings.minio_bucket,
            object_key,
            BytesIO(content),
            length=len(content),
            content_type=content_type,
        )
        return f"s3://{settings.minio_bucket}/{object_key}"


class RedisVisionStatusStore:
    """Optional Redis status store matching the Celery processing contract."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            vision_infrastructure_readiness_service.require_ready("redis")
            from redis import Redis

            client = Redis.from_url(settings.redis_url, decode_responses=True)
        self.client = client

    def set_job_status(self, job: dict[str, Any]) -> None:
        self.client.hset(f"vision:jobs:{job['job_id']}", mapping={key: str(value) for key, value in job.items()})

    def get_job_status(self, job_id: str) -> dict[str, str]:
        return dict(self.client.hgetall(f"vision:jobs:{job_id}"))


class QdrantVisionVectorStore:
    """Optional Qdrant evidence index; vectors are supplied by a gated embedding adapter."""

    def __init__(self, client: Any | None = None, collection_name: str = "vision_evidence_chunks") -> None:
        if client is None:
            vision_infrastructure_readiness_service.require_ready("qdrant")
            from qdrant_client import QdrantClient

            client = QdrantClient(url=settings.qdrant_url)
        self.client = client
        self.collection_name = collection_name

    def upsert_chunks(self, chunks: list[VisionEvidenceChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)


class Neo4jVisionGraphStore:
    """Optional Neo4j projection store for reviewed graph evidence."""

    def __init__(self, driver: Any | None = None) -> None:
        if driver is None:
            vision_infrastructure_readiness_service.require_ready("neo4j")
            from neo4j import GraphDatabase

            auth = (
                (settings.neo4j_user, settings.neo4j_password)
                if settings.neo4j_user and settings.neo4j_password
                else None
            )
            driver = GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        self.driver = driver

    def upsert_graph(self, tenant_id: str, document_id: str, graph: VisionGraphProjection) -> None:
        with self.driver.session(database=settings.neo4j_database) as session:
            session.execute_write(
                self._upsert_graph_tx,
                tenant_id,
                document_id,
                graph.model_dump(mode="json"),
            )

    @staticmethod
    def _upsert_graph_tx(
        tx: Any,
        tenant_id: str,
        document_id: str,
        graph: dict[str, Any],
    ) -> None:
        tx.run(
            """
            MERGE (d:Document {tenant_id: $tenant_id, document_id: $document_id})
            SET d.updated_at = datetime()
            WITH d
            UNWIND $nodes AS node
            MERGE (n:VisionEntity {tenant_id: $tenant_id, entity_id: node.id})
            SET n += node, n.tenant_id = $tenant_id
            MERGE (d)-[:CONTAINS_ENTITY]->(n)
            """,
            tenant_id=tenant_id,
            document_id=document_id,
            nodes=graph.get("nodes", []),
        )
        tx.run(
            """
            UNWIND $edges AS edge
            MATCH (source:VisionEntity {tenant_id: $tenant_id, entity_id: edge.source})
            MATCH (target:VisionEntity {tenant_id: $tenant_id, entity_id: edge.target})
            MERGE (source)-[r:VISION_RELATIONSHIP {tenant_id: $tenant_id, edge_id: edge.id}]->(target)
            SET r += edge, r.tenant_id = $tenant_id
            """,
            tenant_id=tenant_id,
            edges=graph.get("edges", []),
        )


def _redact_url(value: str) -> str:
    if not value:
        return ""
    redacted = re.sub(r"(?i)(password|pwd|secret|token|key)=([^&]+)", r"\1=****", value)
    parts = urlsplit(redacted)
    if parts.username or parts.password:
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        netloc = f"****:****@{host}{port}" if parts.password else f"****@{host}{port}"
        redacted = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return redacted


def _redact_endpoint(value: str) -> str:
    return re.sub(r"(?i)(secret|password|token|key)[^/]*", "****", value or "")


def _normalise_postgresql_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"postgresql", "postgresql+asyncpg"}:
        raise ValueError("Unsupported PostgreSQL URL scheme")
    if not parts.hostname or not parts.path.strip("/"):
        raise ValueError("PostgreSQL URL requires a host and database")
    return urlunsplit(("postgresql", parts.netloc, parts.path, parts.query, parts.fragment))


def _postgresql_safe_config(value: str | None) -> dict[str, Any]:
    if not value:
        return {
            "configured": False,
            "scheme": "",
            "host": "",
            "port": None,
            "database": "",
        }
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return {
            "configured": True,
            "scheme": "invalid",
            "host": "",
            "port": None,
            "database": "",
        }
    return {
        "configured": True,
        "scheme": parts.scheme,
        "host": parts.hostname or "",
        "port": port,
        "database": parts.path.strip("/").split("/", maxsplit=1)[0],
        "credentials": "configured" if parts.username or parts.password else "not_in_url",
    }


def _mask_identifier(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}***{value[-2:]}"


def _mongo_database_name(uri: str) -> str:
    path = urlsplit(uri).path.strip("/")
    if not path:
        return "documind"
    return path.split("/", maxsplit=1)[0] or "documind"


vision_infrastructure_readiness_service = VisionInfrastructureReadinessService()
