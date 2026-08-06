from __future__ import annotations

import importlib.util
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from app.core.config import settings
from app.models.vision import ReviewerStatus, VisionEvidenceChunk
from app.services.vision.adapters import AdapterStatus, VisionAdapter
from app.services.vision.checkpoint_utils import local_or_hf_cached_checkpoint

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
        self.ensure_ready()
        model = self._load()
        output = model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(output["dense_vecs"][0], dtype=np.float32)
        lexical = output.get("lexical_weights", [{}])[0]
        return HybridVector(
            dense=dense.tolist(),
            sparse={str(key): float(value) for key, value in dict(lexical).items()},
        )

    def _load(self) -> Any:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                settings.bge_m3_checkpoint,
                use_fp16=settings.paddleocr_device != "cpu",
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
        active = enabled and installed and (
            local_checkpoint is not None or settings.bge_allow_download
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
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                settings.bge_reranker_checkpoint,
                use_fp16=settings.paddleocr_device != "cpu",
            )
        return self._model


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
                self._external_error = None
            except Exception as exc:
                self._external_error = f"{type(exc).__name__}: {exc}"

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
                self._external_error = None
            except Exception as exc:
                self._external_error = f"{type(exc).__name__}: {exc}"

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
            and self.qdrant.status().status == "active"
            and self._external_error is None
        ):
            try:
                external_scores = self.qdrant.candidate_scores(
                    query_vector.dense,
                    tenant_id=tenant_id,
                    document_ids=document_ids,
                    limit=30,
                )
            except Exception as exc:
                self._external_error = f"{type(exc).__name__}: {exc}"
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
        if (
            rerank_candidates
            and not self.force_local
            and self.reranker.status().status == "active"
        ):
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

    def status(self) -> dict[str, Any]:
        return {
            "embedding": self.embedding.status().__dict__,
            "reranker": self.reranker.status().__dict__,
            "chunks": len(self._chunks),
            "documents": len(self._document_chunks),
            "pipeline": "top30_hybrid_then_top15_rerank_then_top5_to_8",
            "qdrant": self.qdrant.status().__dict__,
            "external_error": self._external_error,
        }

    def _dense_cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        left_array = np.asarray(left, dtype=np.float32)
        right_array = np.asarray(right, dtype=np.float32)
        denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
        return max(0.0, float(np.dot(left_array, right_array) / denominator)) if denominator else 0.0

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

    def status(self) -> AdapterStatus:
        installed = importlib.util.find_spec("qdrant_client") is not None
        enabled = settings.enable_qdrant
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if enabled and installed else ("disabled" if not enabled else "unavailable"),
            reason=(
                "Qdrant client is configured."
                if enabled and installed
                else "Enable Qdrant and install qdrant-client."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={
                "base_collection": self.base_collection_name,
                "collection": self.collection_name,
                "collection_template": (
                    f"{self.base_collection_name}__{self.vector_namespace}__d{{dimensions}}_v1"
                ),
                "expected_vector_size": self.expected_vector_size,
                "schema_policy": "non-destructive-dimension-versioned-v1",
            },
        )

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
            raise ValueError("A document replacement requires matching non-empty chunks and vectors")
        if any(
            chunk.document_id != document_id or chunk.tenant_id != tenant_id
            for chunk in chunks
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
        return (
            f"{self.base_collection_name}__{self.vector_namespace}__d{vector_size}_v1"
        )

    def _slug(self, value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        return normalized or "generic"

    def _load(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=settings.qdrant_url)
        return self._client
