from __future__ import annotations

import importlib.util
import logging
import math
import re
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from app.core.config import settings
from app.models.vision import ReviewerStatus, VisionEvidenceChunk
from app.services.vision.adapters import AdapterStatus, VisionAdapter
from app.services.vision.checkpoint_utils import local_or_hf_cached_checkpoint

LOGGER = logging.getLogger("documind.retrieval")

TOKEN_PATTERN = re.compile(r"[\w][\w.-]*", re.UNICODE)


@dataclass
class HybridVector:
    dense: list[float]
    sparse: dict[str, float]


@dataclass
class RetrievalHit:
    chunk: VisionEvidenceChunk
    score: float
    dense_score: float
    sparse_score: float
    rerank_score: float | None = None
    backend: str = "local-hybrid"


class LocalHashingEmbeddingAdapter(VisionAdapter):
    adapter_id = "local-hashing-embedding"
    stage = "retrieval_embedding"
    model_version = "feature-hashing-v1"
    collection_version = "local_hashing_v1"

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active",
            reason="Deterministic signed feature hashing and lexical weights are available.",
            enabled=True,
            installed=True,
            model_version=self.model_version,
            metadata={"dimensions": self.dimensions},
        )

    def embed(self, text: str) -> HybridVector:
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        counts = Counter(tokens)
        counts.update(" ".join(pair) for pair in pairwise(tokens))
        dense = np.zeros(self.dimensions, dtype=np.float32)
        for token, count in counts.items():
            import hashlib

            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            dense[index] += sign * float(count)
        norm = float(np.linalg.norm(dense))
        if norm:
            dense /= norm
        sparse_norm = math.sqrt(sum(value * value for value in counts.values()))
        sparse = {
            token: float(value / sparse_norm) if sparse_norm else 0.0
            for token, value in counts.items()
        }
        return HybridVector(dense=dense.tolist(), sparse=sparse)


class BGEM3EmbeddingAdapter(VisionAdapter):
    adapter_id = "bge-m3-embedding"
    stage = "retrieval_embedding"
    model_version = "BAAI/bge-m3"
    collection_version = "bge_m3"
    dimensions = 1024

    def __init__(self) -> None:
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = importlib.util.find_spec("FlagEmbedding") is not None
        enabled = settings.enable_heavy_models and settings.enable_bge_m3
        checkpoint = settings.bge_m3_checkpoint
        local_checkpoint = local_or_hf_cached_checkpoint(checkpoint)
        assets_available = local_checkpoint is not None or settings.bge_allow_download
        active = enabled and installed and assets_available
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "FlagEmbedding BGE-M3 adapter is enabled."
                if active
                else "Install FlagEmbedding and provide a local checkpoint, or explicitly allow download."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={
                "checkpoint": checkpoint,
                "resolved_checkpoint": str(local_checkpoint) if local_checkpoint else None,
                "local_checkpoint": local_checkpoint is not None,
                "allow_download": settings.bge_allow_download,
            },
        )

    def embed(self, text: str) -> HybridVector:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[HybridVector]:
        if not texts:
            return []
        self.ensure_ready()
        model = self._load()
        output = model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = np.asarray(output["dense_vecs"], dtype=np.float32)
        lexical_vectors = output.get("lexical_weights", [{} for _ in texts])
        return [
            HybridVector(
                dense=dense.tolist(),
                sparse={str(key): float(value) for key, value in dict(lexical).items()},
            )
            for dense, lexical in zip(dense_vectors, lexical_vectors, strict=True)
        ]

    def _load(self) -> Any:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            local_checkpoint = local_or_hf_cached_checkpoint(settings.bge_m3_checkpoint)
            checkpoint = str(local_checkpoint or settings.bge_m3_checkpoint)
            self._model = BGEM3FlagModel(
                checkpoint,
                use_fp16=settings.paddleocr_device != "cpu",
                trust_remote_code=False,
            )
        return self._model


class BGERerankerAdapter(VisionAdapter):
    adapter_id = "bge-reranker-v2-m3"
    stage = "retrieval_reranking"
    model_version = "configured-bge-reranker"

    def __init__(self) -> None:
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = importlib.util.find_spec("FlagEmbedding") is not None
        enabled = settings.enable_heavy_models and settings.enable_bge_m3
        checkpoint = settings.bge_reranker_checkpoint
        local_checkpoint = local_or_hf_cached_checkpoint(checkpoint)
        active = (
            enabled and installed and (local_checkpoint is not None or settings.bge_allow_download)
        )
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "BGE reranker adapter is enabled."
                if active
                else "Install FlagEmbedding and provide a local reranker checkpoint, or explicitly allow download."
            ),
            enabled=enabled,
            installed=installed,
            model_version=checkpoint,
            metadata={
                "checkpoint": checkpoint,
                "resolved_checkpoint": str(local_checkpoint) if local_checkpoint else None,
                "local_checkpoint": local_checkpoint is not None,
                "allow_download": settings.bge_allow_download,
            },
        )

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.ensure_ready()
        model = self._load()
        values = model.compute_score(
            [[query, passage] for passage in passages],
            normalize=True,
        )
        if isinstance(values, (float, int)):
            return [float(values)]
        return [float(value) for value in values]

    def _load(self) -> Any:
        if self._model is None:
            local_checkpoint = local_or_hf_cached_checkpoint(settings.bge_reranker_checkpoint)
            checkpoint = str(local_checkpoint or settings.bge_reranker_checkpoint)
            self._model = _LocalSequenceReranker(checkpoint)
        return self._model


class _LocalSequenceReranker:
    """Minimal safe reranker compatible with current Transformers tokenizers.

    FlagEmbedding 1.x calls the removed ``prepare_for_model`` tokenizer API on
    Transformers 5.  Loading the same sequence-classification checkpoint through
    the public paired-text tokenizer path preserves exact model inference while
    allowing patched Transformers releases.  Cached assets are passed as concrete
    paths and remote model code is never trusted.
    """

    def __init__(self, checkpoint: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            local_files_only=Path(checkpoint).exists(),
            trust_remote_code=False,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            local_files_only=Path(checkpoint).exists(),
            trust_remote_code=False,
        )
        self._model.eval()

    def compute_score(
        self,
        sentence_pairs: list[list[str]],
        *,
        normalize: bool = True,
    ) -> list[float]:
        if not sentence_pairs:
            return []
        queries = [item[0] for item in sentence_pairs]
        passages = [item[1] for item in sentence_pairs]
        encoded = self._tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            logits = self._model(**encoded, return_dict=True).logits.reshape(-1).float()
        if normalize:
            logits = self._torch.sigmoid(logits)
        return [float(value) for value in logits.cpu().tolist()]


class LocalHybridRetrievalIndex:
    def __init__(self, *, force_local: bool = False) -> None:
        bge = BGEM3EmbeddingAdapter()
        self.embedding = (
            bge
            if not force_local and bge.status().status == "active"
            else LocalHashingEmbeddingAdapter()
        )
        self.reranker = BGERerankerAdapter()
        self.force_local = force_local
        self._chunks: dict[str, VisionEvidenceChunk] = {}
        self._vectors: dict[str, HybridVector] = {}
        self._document_chunks: dict[str, set[str]] = {}
        self.qdrant = QdrantHybridBackend(
            vector_namespace=self.embedding.collection_version,
            expected_vector_size=self.embedding.dimensions,
        )
        self._external_error: str | None = None
        self._external_error_at: float | None = None
        self._external_failures = 0

    def upsert_document(self, document_id: str, chunks: list[VisionEvidenceChunk]) -> None:
        tenant_ids = {chunk.tenant_id for chunk in chunks}
        if len(tenant_ids) > 1:
            raise ValueError("A document replacement cannot span multiple tenants")
        for chunk_id in self._document_chunks.get(document_id, set()):
            self._chunks.pop(chunk_id, None)
            self._vectors.pop(chunk_id, None)
        ids: set[str] = set()
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = self.embedding.embed(chunk.text)
            ids.add(chunk.chunk_id)
        self._document_chunks[document_id] = ids
        if chunks and not self.force_local and self.qdrant.status().status == "active":
            try:
                self.qdrant.replace_document(
                    document_id,
                    tenant_id=next(iter(tenant_ids)),
                    chunks=chunks,
                    vectors=[self._vectors[chunk.chunk_id] for chunk in chunks],
                )
                self._record_external_success()
            except Exception as exc:
                self._record_external_failure("upsert_document", exc)

    def remove_document(self, document_id: str) -> None:
        existing_ids = self._document_chunks.pop(document_id, set())
        tenant_ids = {
            self._chunks[chunk_id].tenant_id
            for chunk_id in existing_ids
            if chunk_id in self._chunks
        }
        for chunk_id in existing_ids:
            self._chunks.pop(chunk_id, None)
            self._vectors.pop(chunk_id, None)
        if not self.force_local and self.qdrant.status().status == "active":
            try:
                for tenant_id in tenant_ids:
                    self.qdrant.delete_document(document_id, tenant_id=tenant_id)
                self._record_external_success()
            except Exception as exc:
                self._record_external_failure("delete_document", exc)

    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 5,
        document_ids: set[str] | None = None,
        document_type: str | None = None,
        review_status: ReviewerStatus | None = None,
    ) -> list[RetrievalHit]:
        query_vector = self.embedding.embed(query)
        candidates: list[RetrievalHit] = []
        external_scores: dict[str, float] = {}
        if (
            not self.force_local
            and self._external_backoff_elapsed()
            and self.qdrant.status().status == "active"
        ):
            try:
                external_scores = self.qdrant.candidate_scores(
                    query_vector.dense,
                    tenant_id=tenant_id,
                    document_ids=document_ids,
                    limit=30,
                )
                self._record_external_success()
            except Exception as exc:
                self._record_external_failure("candidate_scores", exc)
        source_items = (
            (
                (chunk_id, self._chunks[chunk_id])
                for chunk_id in external_scores
                if chunk_id in self._chunks
            )
            if external_scores
            else self._chunks.items()
        )
        for chunk_id, chunk in source_items:
            if chunk.tenant_id != tenant_id:
                continue
            if document_ids is not None and chunk.document_id not in document_ids:
                continue
            if document_type and chunk.document_type.value != document_type:
                continue
            if review_status and chunk.review_status != review_status:
                continue
            vector = self._vectors[chunk_id]
            dense_score = external_scores.get(
                chunk_id,
                self._dense_cosine(query_vector.dense, vector.dense),
            )
            sparse_score = self._sparse_cosine(query_vector.sparse, vector.sparse)
            score = 0.65 * dense_score + 0.35 * sparse_score
            if chunk.review_status in {ReviewerStatus.accepted, ReviewerStatus.edited}:
                score *= 1.05
            if score > 0:
                candidates.append(
                    RetrievalHit(
                        chunk=chunk,
                        score=score,
                        dense_score=dense_score,
                        sparse_score=sparse_score,
                    )
                )
        candidates.sort(key=lambda item: item.score, reverse=True)
        first_stage = candidates[:30]
        rerank_candidates = first_stage[:15]
        if rerank_candidates and not self.force_local and self.reranker.status().status == "active":
            scores = self.reranker.rerank(
                query,
                [item.chunk.text for item in rerank_candidates],
            )
            for item, score in zip(rerank_candidates, scores, strict=True):
                item.rerank_score = score
                item.score = 0.45 * item.score + 0.55 * score
            rerank_candidates.sort(key=lambda item: item.score, reverse=True)
            ranked_ids = {item.chunk.chunk_id for item in rerank_candidates}
            first_stage = [
                *rerank_candidates,
                *(item for item in first_stage if item.chunk.chunk_id not in ranked_ids),
            ]
        return first_stage[:top_k]

    #: A failed Qdrant call must not disable the mirror for the process lifetime.
    #: Retries are held off for this long, then attempted again.
    external_retry_backoff_seconds: float = 30.0

    def _external_backoff_elapsed(self) -> bool:
        if self._external_error is None:
            return True
        if self._external_error_at is None:
            return True
        return (time.monotonic() - self._external_error_at) >= self.external_retry_backoff_seconds

    def _record_external_success(self) -> None:
        if self._external_error is not None:
            LOGGER.info(
                "qdrant mirror recovered",
                extra={
                    "context": {
                        "component": "retrieval_index",
                        "previous_error": self._external_error,
                        "consecutive_failures": self._external_failures,
                    }
                },
            )
        self._external_error = None
        self._external_error_at = None
        self._external_failures = 0

    def _record_external_failure(self, operation: str, exc: Exception) -> None:
        self._external_error = f"{type(exc).__name__}: {exc}"
        self._external_error_at = time.monotonic()
        self._external_failures += 1
        # The mirror is optional, but a silent downgrade is not acceptable: the
        # degradation is logged here and reported by status()/retrieval traces.
        LOGGER.warning(
            "qdrant mirror unavailable; serving retrieval from the local hybrid index",
            extra={
                "context": {
                    "component": "retrieval_index",
                    "operation": operation,
                    "error": self._external_error,
                    "consecutive_failures": self._external_failures,
                    "retry_after_seconds": self.external_retry_backoff_seconds,
                }
            },
        )
        self.qdrant.invalidate_liveness_cache()

    def status(self) -> dict[str, Any]:
        qdrant_status = self.qdrant.status()
        degraded = self._external_error is not None
        return {
            "embedding": self.embedding.status().__dict__,
            "reranker": self.reranker.status().__dict__,
            "chunks": len(self._chunks),
            "documents": len(self._document_chunks),
            "pipeline": "top30_hybrid_then_top15_rerank_then_top5_to_8",
            "qdrant": qdrant_status.__dict__,
            "external_error": self._external_error,
            "external_consecutive_failures": self._external_failures,
            "external_retry_after_seconds": (
                None
                if not degraded
                else max(
                    0.0,
                    round(
                        self.external_retry_backoff_seconds
                        - (time.monotonic() - (self._external_error_at or 0.0)),
                        2,
                    ),
                )
            ),
            # The authoritative statement of what actually served this index.
            "vector_backend": (
                "qdrant_mirrored_local_hybrid"
                if qdrant_status.status == "active" and not degraded
                else "local_hybrid_only"
            ),
            "degraded": degraded,
        }

    def _dense_cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        left_array = np.asarray(left, dtype=np.float32)
        right_array = np.asarray(right, dtype=np.float32)
        denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
        return (
            max(0.0, float(np.dot(left_array, right_array) / denominator)) if denominator else 0.0
        )

    def _sparse_cosine(self, left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        common = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class QdrantHybridBackend(VisionAdapter):
    adapter_id = "qdrant-hybrid-retrieval"
    stage = "vector_store"
    model_version = "Qdrant"

    #: A configured client proves nothing about the server. ``status()`` is called on
    #: every upsert and search, so the liveness result is cached for this many seconds
    #: rather than probing per call.
    liveness_cache_seconds: float = 5.0
    liveness_timeout_seconds: float = 1.5

    def __init__(
        self,
        collection_name: str = "vision_evidence_chunks",
        *,
        vector_namespace: str = "generic",
        expected_vector_size: int | None = None,
    ) -> None:
        self.base_collection_name = collection_name
        self.vector_namespace = self._slug(vector_namespace)
        self.expected_vector_size = expected_vector_size
        self._active_vector_size = expected_vector_size
        self.collection_name = (
            self._collection_for_dimension(expected_vector_size)
            if expected_vector_size is not None
            else collection_name
        )
        self._client: Any | None = None
        self._liveness: tuple[float, bool, str] | None = None

    def status(self) -> AdapterStatus:
        installed = importlib.util.find_spec("qdrant_client") is not None
        enabled = settings.enable_qdrant
        metadata: dict[str, Any] = {
            "base_collection": self.base_collection_name,
            "collection": self.collection_name,
            "collection_template": (
                f"{self.base_collection_name}__{self.vector_namespace}__d{{dimensions}}_v1"
            ),
            "expected_vector_size": self.expected_vector_size,
            "schema_policy": "non-destructive-dimension-versioned-v1",
        }
        if not enabled:
            return AdapterStatus(
                adapter_id=self.adapter_id,
                stage=self.stage,
                status="disabled",
                reason="Enable Qdrant with ENABLE_QDRANT=true.",
                enabled=enabled,
                installed=installed,
                model_version=self.model_version,
                metadata={**metadata, "server_reachable": None},
            )
        if not installed:
            return AdapterStatus(
                adapter_id=self.adapter_id,
                stage=self.stage,
                status="unavailable",
                reason="Qdrant is enabled but the qdrant-client package is not installed.",
                enabled=enabled,
                installed=installed,
                model_version=self.model_version,
                metadata={**metadata, "server_reachable": None},
            )
        reachable, detail = self._server_reachable()
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if reachable else "unavailable",
            reason=(
                f"Qdrant client is configured and the server answered a liveness probe ({detail})."
                if reachable
                else f"Qdrant is configured but the server did not answer a liveness probe: {detail}."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={
                **metadata,
                "server_reachable": reachable,
                "liveness_detail": detail,
                "liveness_probe": "GET /readyz",
                "liveness_cache_seconds": self.liveness_cache_seconds,
            },
        )

    def _server_reachable(self) -> tuple[bool, str]:
        """Bounded, cached ``GET /readyz`` against the configured Qdrant endpoint.

        Configuration alone must never be reported as ``active``: a badge that
        outlives the server it describes is worse than no badge at all.
        """

        now = time.monotonic()
        cached = self._liveness
        if cached is not None and now - cached[0] < self.liveness_cache_seconds:
            return cached[1], cached[2]
        endpoint = f"{settings.qdrant_url.rstrip('/')}/readyz"
        try:
            request = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(  # operator-configured endpoint
                request,
                timeout=self.liveness_timeout_seconds,
            ) as response:
                code = int(response.status)
            reachable = 200 <= code < 300
            detail = f"HTTP {code}"
        except Exception as exc:
            reachable = False
            detail = type(exc).__name__
        self._liveness = (now, reachable, detail)
        return reachable, detail

    def invalidate_liveness_cache(self) -> None:
        self._liveness = None

    def ensure_collection(self, vector_size: int) -> None:
        self.ensure_ready()
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if vector_size <= 0:
            raise ValueError("Qdrant vectors must have a positive dimension")
        if self.expected_vector_size is not None and vector_size != self.expected_vector_size:
            raise ValueError(
                "Vector dimension does not match the configured embedding profile: "
                f"expected {self.expected_vector_size}, received {vector_size}"
            )
        self._active_vector_size = vector_size
        self.collection_name = self._collection_for_dimension(vector_size)
        client = self._load()
        collections = {item.name for item in client.get_collections().collections}
        if self.collection_name not in collections:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        collection = client.get_collection(self.collection_name)
        configured_vectors = collection.config.params.vectors
        configured_size = getattr(configured_vectors, "size", None)
        configured_distance = getattr(configured_vectors, "distance", None)
        if configured_size != vector_size or configured_distance != Distance.COSINE:
            raise RuntimeError(
                f"Qdrant collection {self.collection_name!r} has an incompatible vector schema; "
                f"expected cosine/{vector_size}, found {configured_distance}/{configured_size}. "
                "The collection was preserved and must be migrated explicitly."
            )
        payload_schema = dict(getattr(collection, "payload_schema", {}) or {})
        for field_name in ("tenant_id", "document_id"):
            if field_name in payload_schema:
                continue
            client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def upsert(self, chunks: list[VisionEvidenceChunk], vectors: list[HybridVector]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("Qdrant upsert requires one vector per chunk")
        vector_size = len(vectors[0].dense)
        if any(len(vector.dense) != vector_size for vector in vectors):
            raise ValueError("Qdrant upsert vectors must have one consistent dimension")
        self.ensure_collection(vector_size)
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector.dense,
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._load().upsert(collection_name=self.collection_name, points=points, wait=True)

    def replace_document(
        self,
        document_id: str,
        *,
        tenant_id: str,
        chunks: list[VisionEvidenceChunk],
        vectors: list[HybridVector],
    ) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError(
                "A document replacement requires matching non-empty chunks and vectors"
            )
        if any(
            chunk.document_id != document_id or chunk.tenant_id != tenant_id for chunk in chunks
        ):
            raise ValueError("Replacement chunks must match the scoped tenant and document")
        vector_size = len(vectors[0].dense)
        if any(len(vector.dense) != vector_size for vector in vectors):
            raise ValueError("Replacement vectors must have one consistent dimension")
        self.ensure_collection(vector_size)
        self.delete_document(document_id, tenant_id=tenant_id)
        self.upsert(chunks, vectors)

    def delete_document(self, document_id: str, *, tenant_id: str) -> None:
        self.ensure_ready()
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        client = self._load()
        if self._active_vector_size is None:
            return
        collections = {item.name for item in client.get_collections().collections}
        if self.collection_name not in collections:
            return
        client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                        FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    ]
                )
            ),
            wait=True,
        )

    def candidate_scores(
        self,
        dense: list[float],
        *,
        tenant_id: str,
        document_ids: set[str] | None,
        limit: int,
    ) -> dict[str, float]:
        self.ensure_ready()
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        self.ensure_collection(len(dense))
        must = [FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        if document_ids:
            must.append(
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=sorted(document_ids)),
                )
            )
        result = self._load().query_points(
            collection_name=self.collection_name,
            query=dense,
            query_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return {
            str(point.payload["chunk_id"]): float(point.score)
            for point in result.points
            if point.payload and point.payload.get("chunk_id")
        }

    def _collection_for_dimension(self, vector_size: int) -> str:
        return f"{self.base_collection_name}__{self.vector_namespace}__d{vector_size}_v1"

    def _slug(self, value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        return normalized or "generic"

    def _load(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=settings.qdrant_url)
        return self._client
