from __future__ import annotations

import hashlib
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings
from app.models.vision import (
    OCRWord,
    ReviewerStatus,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidenceChunk,
    VisionPageArtifact,
    VisionProcessingStatus,
)
from app.services.model_registry import ModelRegistryService
from app.services.vision.runtime_health import (
    ModelSmokeStore,
    model_configuration_fingerprint,
    model_smoke_store,
)


class ModelSmokeService:
    def __init__(
        self,
        *,
        registry: ModelRegistryService | None = None,
        store: ModelSmokeStore | None = None,
    ) -> None:
        self.registry = registry or ModelRegistryService()
        self.store = store or model_smoke_store

    def run(self, model_id: str) -> dict[str, Any]:
        backend = next(
            (
                item
                for item in self.registry.backends(include_runtime_health=False)
                if str(item["model_id"]) == model_id or str(item.get("adapter_id")) == model_id
            ),
            None,
        )
        if backend is None:
            raise ValueError(f"Model is not registered: {model_id}")

        started_at = self._now()
        started = time.perf_counter()
        configured_status = str(backend.get("configured_status", backend.get("status")))
        result: dict[str, Any] = {
            "model_id": str(backend["model_id"]),
            "adapter_id": str(backend.get("adapter_id") or backend["model_id"]),
            "stage": str(backend["stage"]),
            "configured_status": configured_status,
            "status": "blocked",
            "health_status": "unknown",
            "started_at": started_at,
            "completed_at": None,
            "duration_ms": None,
            "device": self._device(backend),
            "package_version": backend.get("package_version"),
            "checkpoint": (backend.get("runtime") or {}).get("checkpoint"),
            "source_document": "synthetic-smoke-invoice.png",
            "source_pages": [1],
            "configuration_fingerprint": model_configuration_fingerprint(backend),
            "initialization": {"status": "not_run", "duration_ms": None},
            "inference": {"status": "not_run", "duration_ms": None},
            "output_schema": {
                "status": "not_evaluated",
                "valid": None,
                "name": None,
                "observed_keys": [],
            },
            "error": None,
            "blocker": None,
        }
        if configured_status != "active":
            result["blocker"] = self._blocker(backend)
            result["error"] = str(
                backend.get("reason") or "Model is not configured for runtime use."
            )
            return self._finish(result, started)

        adapter: Any = None
        try:
            init_started = time.perf_counter()
            adapter = self._construct(str(backend.get("adapter_id") or backend["model_id"]))
            self._initialize(str(backend.get("adapter_id") or backend["model_id"]), adapter)
            result["initialization"] = {
                "status": "passed",
                "duration_ms": self._duration(init_started),
            }
            result["device"] = self._device(backend, adapter)

            inference_started = time.perf_counter()
            with TemporaryDirectory(prefix="documind-model-smoke-") as directory:
                document = self._fixture(Path(directory))
                try:
                    output = self._infer(
                        str(backend.get("adapter_id") or backend["model_id"]),
                        adapter,
                        document,
                    )
                    schema = self._validate_output(
                        str(backend.get("adapter_id") or backend["model_id"]),
                        output,
                    )
                finally:
                    self._cleanup_fixture(document)
            result["inference"] = {
                "status": "passed",
                "duration_ms": self._duration(inference_started),
                "summary": output,
            }
            result["output_schema"] = schema
            result["status"] = "passed"
            result["health_status"] = "healthy"
        except Exception as exc:
            result["status"] = "failed"
            result["health_status"] = "unhealthy"
            result["error"] = f"{type(exc).__name__}: {exc}"
            if result["initialization"]["status"] != "passed":
                result["initialization"] = {
                    "status": "failed",
                    "duration_ms": self._duration(init_started),
                    "error": result["error"],
                }
            else:
                result["inference"] = {
                    "status": "failed",
                    "duration_ms": self._duration(inference_started),
                    "error": result["error"],
                }
        finally:
            self._close(adapter)
        return self._finish(result, started)

    def run_all(self) -> dict[str, Any]:
        results = [
            self.run(str(item["model_id"]))
            for item in self.registry.backends(include_runtime_health=False)
        ]
        return {
            "generated_at": self._now(),
            "total": len(results),
            "passed": sum(item["status"] == "passed" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "blocked": sum(item["status"] == "blocked" for item in results),
            "results": results,
        }

    def _construct(self, adapter_id: str) -> Any:
        from app.services.ingestion import DocumentIngestionService
        from app.services.vision.advanced_adapters import PaddleOCRVLAdapter
        from app.services.vision.anomaly_adapters import (
            IsolationForestRiskAdapter,
            XGBoostRiskAdapter,
        )
        from app.services.vision.contract_adapters import DebertaContractClauseAdapter
        from app.services.vision.deterministic_extraction import deterministic_schema_extractor
        from app.services.vision.duplicate_detection import DinoV2DuplicateAdapter
        from app.services.vision.extraction_adapters import (
            DonutExtractionAdapter,
            LayoutLMv3ExtractionAdapter,
        )
        from app.services.vision.generator_adapters import (
            GroundedTemplateGeneratorAdapter,
            OpenAICompatibleGroundedGenerator,
            QwenGroundedGeneratorAdapter,
        )
        from app.services.vision.graph_builder import evidence_graph_builder
        from app.services.vision.layout_adapters import (
            GeometryLayoutAdapter,
            PaddleStructureV3Adapter,
        )
        from app.services.vision.ocr_adapters import PaddleOCRV5Adapter, TesseractOCRAdapter
        from app.services.vision.retrieval_adapters import (
            BGEM3EmbeddingAdapter,
            BGERerankerAdapter,
            LocalHashingEmbeddingAdapter,
            LocalHybridRetrievalIndex,
            QdrantHybridBackend,
        )
        from app.services.vision.tamper_adapters import (
            LocalForensicsAdapter,
            SegFormerTamperAdapter,
        )
        from app.services.vision.visual_adapters import (
            OpenCVVisualDetectionAdapter,
            Yolo26VisualDetectionAdapter,
        )

        factories: dict[str, Any] = {
            "local-parser-v1": DocumentIngestionService,
            "tesseract-local-ocr": TesseractOCRAdapter,
            "geometry-layout-table": GeometryLayoutAdapter,
            "paddleocr-pp-ocrv5": PaddleOCRV5Adapter,
            "paddleocr-pp-structurev3": PaddleStructureV3Adapter,
            "layoutlmv3-extraction": LayoutLMv3ExtractionAdapter,
            "deterministic-schema-extractor-v2": lambda: deterministic_schema_extractor,
            "paddleocr-vl-parser": PaddleOCRVLAdapter,
            "donut-ocr-free": DonutExtractionAdapter,
            "dinov2-document-embedding": DinoV2DuplicateAdapter,
            "yolo26-visual-detection": Yolo26VisualDetectionAdapter,
            "opencv-visual-evidence": OpenCVVisualDetectionAdapter,
            "deberta-cuad-clause-extraction": DebertaContractClauseAdapter,
            "ela-noise-forensics": LocalForensicsAdapter,
            "segformer-document-forensics": SegFormerTamperAdapter,
            "isolation-forest-risk": IsolationForestRiskAdapter,
            "xgboost-supervised-risk": XGBoostRiskAdapter,
            "local-hashing-embedding": LocalHashingEmbeddingAdapter,
            "bge-m3-embedding": BGEM3EmbeddingAdapter,
            "bge-reranker-v2-m3": BGERerankerAdapter,
            "local-vector-store-v1": lambda: LocalHybridRetrievalIndex(force_local=True),
            "qdrant-hybrid-retrieval": lambda: QdrantHybridBackend(
                collection_name=f"documind_model_smoke_{uuid4().hex[:10]}"
            ),
            "local-graph-projection-v1": lambda: evidence_graph_builder,
            "grounded-template-generator": GroundedTemplateGeneratorAdapter,
            "qwen-grounded-generator": QwenGroundedGeneratorAdapter,
            "openai-compatible-grounded-generator": OpenAICompatibleGroundedGenerator,
        }
        if adapter_id == "Neo4j":
            from neo4j import GraphDatabase

            auth = (
                (settings.neo4j_user, settings.neo4j_password)
                if settings.neo4j_user and settings.neo4j_password
                else None
            )
            return GraphDatabase.driver(settings.neo4j_uri, auth=auth)
        try:
            return factories[adapter_id]()
        except KeyError as exc:
            raise RuntimeError(f"No smoke probe is implemented for adapter {adapter_id}") from exc

    def _initialize(self, adapter_id: str, adapter: Any) -> None:
        if adapter_id == "tesseract-local-ocr":
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_command
            pytesseract.get_tesseract_version()
            available_languages = set(pytesseract.get_languages(config=""))
            requested_languages = set(settings.tesseract_languages.split("+"))
            if not requested_languages.issubset(available_languages):
                raise RuntimeError(
                    "Tesseract language data is unavailable: "
                    f"{', '.join(sorted(requested_languages - available_languages))}"
                )
            return
        loader_names = {
            "paddleocr-pp-ocrv5": "_load_pipeline",
            "paddleocr-pp-structurev3": "_load_pipeline",
            "layoutlmv3-extraction": "_load",
            "paddleocr-vl-parser": "_load_pipeline",
            "donut-ocr-free": "_load",
            "dinov2-document-embedding": "_load",
            "yolo26-visual-detection": "_load_model",
            "deberta-cuad-clause-extraction": "_load",
            "segformer-document-forensics": "_load",
            "xgboost-supervised-risk": "_load",
            "bge-m3-embedding": "_load",
            "bge-reranker-v2-m3": "_load",
            "qwen-grounded-generator": "_load",
        }
        loader_name = loader_names.get(adapter_id)
        if loader_name:
            loaded = getattr(adapter, loader_name)()
            if adapter_id == "yolo26-visual-detection":
                names = set(dict(getattr(loaded, "names", {}) or {}).values())
                required = {"signature", "stamp", "company_logo", "handwritten_note"}
                checkbox_ready = bool(
                    names & {"checkbox", "checkbox_checked", "checkbox_unchecked"}
                )
                if not required.issubset(names) or not checkbox_ready:
                    raise RuntimeError(
                        "YOLO project checkpoint lacks required signature, stamp, logo, checkbox, "
                        "and handwritten-note classes."
                    )
            return
        if adapter_id == "qdrant-hybrid-retrieval":
            adapter._load().get_collections()
            return
        if adapter_id == "Neo4j":
            adapter.verify_connectivity()
            return
        status_method = getattr(adapter, "status", None)
        if callable(status_method):
            status = status_method()
            if isinstance(status, dict):
                return
            if status.status != "active":
                raise RuntimeError(f"{status.adapter_id} is {status.status}: {status.reason}")

    def _infer(
        self,
        adapter_id: str,
        adapter: Any,
        document: VisionDocumentRecord,
    ) -> dict[str, Any]:
        if adapter_id == "local-parser-v1":
            content = b"Invoice Number: INV-SMOKE-1042\nTotal Amount: INR 1250.00"
            adapter.validate_file("smoke.txt", content)
            text = adapter._decode_text(content)
            return {"text": text, "page_count": 1}
        if adapter_id in {"tesseract-local-ocr", "paddleocr-pp-ocrv5"}:
            words = adapter.extract_words(document)
            return {
                "word_count": len(words),
                "words": [item.model_dump(mode="json") for item in words[:3]],
            }
        if adapter_id in {"geometry-layout-table", "paddleocr-pp-structurev3"}:
            regions, tables = adapter.analyze_layout(document)
            return {
                "region_count": len(regions),
                "table_count": len(tables),
                "regions": [item.model_dump(mode="json") for item in regions[:3]],
            }
        if adapter_id in {"layoutlmv3-extraction", "deterministic-schema-extractor-v2"}:
            if adapter_id == "layoutlmv3-extraction":
                fields = adapter.extract_fields(document)
            else:
                fields = adapter.extract(document)
            return {
                "field_count": len(fields),
                "fields": [item.model_dump(mode="json") for item in fields[:5]],
            }
        if adapter_id == "paddleocr-vl-parser":
            regions, tables, blocks = adapter.parse(document)
            return {
                "region_count": len(regions),
                "table_count": len(tables),
                "block_count": len(blocks),
            }
        if adapter_id == "donut-ocr-free":
            return {"document_json": adapter.extract_json(document)}
        if adapter_id == "dinov2-document-embedding":
            vector = adapter.embed(document)
            return {"dimensions": len(vector), "finite": all(_finite(value) for value in vector)}
        if adapter_id in {"yolo26-visual-detection", "opencv-visual-evidence"}:
            detections = adapter.detect(document)
            return {
                "detection_count": len(detections),
                "detections": [item.model_dump(mode="json") for item in detections[:5]],
            }
        if adapter_id == "deberta-cuad-clause-extraction":
            document.document_type = VisionDocumentType.contract
            fields = adapter.extract_fields(document)
            return {
                "field_count": len(fields),
                "fields": [item.model_dump(mode="json") for item in fields[:5]],
            }
        if adapter_id in {"ela-noise-forensics", "segformer-document-forensics"}:
            analyses = adapter.analyze(document)
            return {
                "analysis_count": len(analyses),
                "analyses": [item.model_dump(mode="json") for item in analyses[:2]],
            }
        if adapter_id == "isolation-forest-risk":
            analysis = adapter.analyze(document, [document])
            return {"analysis": analysis.model_dump(mode="json")}
        if adapter_id == "xgboost-supervised-risk":
            from app.services.vision.anomaly_adapters import FEATURE_NAMES

            analysis = adapter.analyze({name: 0.0 for name in FEATURE_NAMES})
            return {"analysis": analysis.model_dump(mode="json")}
        if adapter_id in {"local-hashing-embedding", "bge-m3-embedding"}:
            vector = adapter.embed("Invoice INV-SMOKE-1042 total INR 1250.00")
            return {
                "dense_dimensions": len(vector.dense),
                "sparse_terms": len(vector.sparse),
                "finite": all(_finite(value) for value in vector.dense),
            }
        if adapter_id == "bge-reranker-v2-m3":
            scores = adapter.rerank(
                "What is the invoice total?",
                ["The total is INR 1250.00.", "The vendor address is Pune."],
            )
            return {"score_count": len(scores), "scores": scores}
        if adapter_id == "local-vector-store-v1":
            chunk = self._chunk(document)
            adapter.upsert_document(document.document_id, [chunk])
            hits = adapter.search("invoice total", tenant_id=document.tenant_id, top_k=1)
            return {
                "hit_count": len(hits),
                "hits": [
                    {"chunk_id": item.chunk.chunk_id, "score": item.score, "backend": item.backend}
                    for item in hits
                ],
            }
        if adapter_id == "qdrant-hybrid-retrieval":
            from app.services.vision.retrieval_adapters import LocalHashingEmbeddingAdapter

            chunk = self._chunk(document)
            vector = LocalHashingEmbeddingAdapter(dimensions=64).embed(chunk.text)
            adapter.upsert([chunk], [vector])
            scores = adapter.candidate_scores(
                vector.dense,
                tenant_id=document.tenant_id,
                document_ids={document.document_id},
                limit=1,
            )
            return {"hit_count": len(scores), "scores": scores}
        if adapter_id == "local-graph-projection-v1":
            from app.services.vision.deterministic_extraction import deterministic_schema_extractor

            document.extracted_fields = deterministic_schema_extractor.extract(document)
            graph = adapter.build(document)
            return {"node_count": len(graph.nodes), "edge_count": len(graph.edges)}
        if adapter_id == "Neo4j":
            with adapter.session(database=settings.neo4j_database) as session:
                record = session.run("RETURN 1 AS value").single()
            return {"value": int(record["value"]) if record else None}
        if adapter_id in {
            "grounded-template-generator",
            "qwen-grounded-generator",
            "openai-compatible-grounded-generator",
        }:
            generated = adapter.generate(
                "What is the invoice total?",
                [
                    {
                        "id": "smoke-evidence-1",
                        "text": "The invoice total is INR 1250.00.",
                        "document_id": document.document_id,
                        "page_number": 1,
                        "review_status": ReviewerStatus.accepted.value,
                    }
                ],
            )
            return {"response": generated}
        raise RuntimeError(f"No inference probe is implemented for adapter {adapter_id}")

    def _validate_output(self, adapter_id: str, output: dict[str, Any]) -> dict[str, Any]:
        validators: dict[str, tuple[str, Any]] = {
            "local-parser-v1": ("parser-v1", lambda value: bool(value.get("text"))),
            "tesseract-local-ocr": (
                "ocr-words-v1",
                lambda value: (
                    value.get("word_count", 0) > 0 and _valid_ocr_words(value.get("words"))
                ),
            ),
            "paddleocr-pp-ocrv5": (
                "ocr-words-v1",
                lambda value: (
                    value.get("word_count", 0) > 0 and _valid_ocr_words(value.get("words"))
                ),
            ),
            "geometry-layout-table": ("layout-v1", lambda value: value.get("region_count", 0) > 0),
            "paddleocr-pp-structurev3": (
                "layout-v1",
                lambda value: isinstance(value.get("regions"), list),
            ),
            "layoutlmv3-extraction": (
                "extracted-fields-v1",
                lambda value: isinstance(value.get("fields"), list),
            ),
            "deterministic-schema-extractor-v2": (
                "extracted-fields-v1",
                lambda value: value.get("field_count", 0) > 0,
            ),
            "paddleocr-vl-parser": (
                "advanced-layout-v1",
                lambda value: all(
                    key in value for key in ("region_count", "table_count", "block_count")
                ),
            ),
            "donut-ocr-free": (
                "document-json-v1",
                lambda value: isinstance(value.get("document_json"), dict),
            ),
            "dinov2-document-embedding": (
                "dense-vector-v1",
                lambda value: value.get("dimensions", 0) > 0 and value.get("finite") is True,
            ),
            "yolo26-visual-detection": (
                "visual-detections-v1",
                lambda value: isinstance(value.get("detections"), list),
            ),
            "opencv-visual-evidence": (
                "visual-detections-v1",
                lambda value: isinstance(value.get("detections"), list),
            ),
            "deberta-cuad-clause-extraction": (
                "extracted-fields-v1",
                lambda value: isinstance(value.get("fields"), list),
            ),
            "ela-noise-forensics": (
                "forensics-v1",
                lambda value: value.get("analysis_count", 0) > 0,
            ),
            "segformer-document-forensics": (
                "forensics-v1",
                lambda value: value.get("analysis_count", 0) > 0,
            ),
            "isolation-forest-risk": (
                "anomaly-analysis-v1",
                lambda value: isinstance(value.get("analysis"), dict),
            ),
            "xgboost-supervised-risk": (
                "anomaly-analysis-v1",
                lambda value: isinstance(value.get("analysis"), dict),
            ),
            "local-hashing-embedding": (
                "hybrid-vector-v1",
                lambda value: value.get("dense_dimensions", 0) > 0 and value.get("finite") is True,
            ),
            "bge-m3-embedding": (
                "hybrid-vector-v1",
                lambda value: value.get("dense_dimensions", 0) > 0 and value.get("finite") is True,
            ),
            "bge-reranker-v2-m3": (
                "reranker-scores-v1",
                lambda value: value.get("score_count") == 2,
            ),
            "local-vector-store-v1": (
                "retrieval-hits-v1",
                lambda value: value.get("hit_count", 0) > 0,
            ),
            "qdrant-hybrid-retrieval": (
                "retrieval-hits-v1",
                lambda value: value.get("hit_count", 0) > 0,
            ),
            "local-graph-projection-v1": (
                "evidence-graph-v2",
                lambda value: value.get("node_count", 0) > 0 and value.get("edge_count", 0) > 0,
            ),
            "Neo4j": ("connectivity-query-v1", lambda value: value.get("value") == 1),
            "grounded-template-generator": ("grounded-answer-v1", _valid_generated),
            "qwen-grounded-generator": ("grounded-answer-v1", _valid_generated),
            "openai-compatible-grounded-generator": ("grounded-answer-v1", _valid_generated),
        }
        try:
            schema_name, validator = validators[adapter_id]
        except KeyError as exc:
            raise RuntimeError(f"No output schema is registered for adapter {adapter_id}") from exc
        valid = bool(validator(output))
        if not valid:
            raise ValueError(f"Smoke inference failed output schema {schema_name}")
        return {
            "status": "evaluated",
            "valid": True,
            "name": schema_name,
            "observed_keys": sorted(output),
        }

    def _fixture(self, root: Path) -> VisionDocumentRecord:
        document_id = f"smoke-doc-{uuid4().hex[:10]}"
        image_path = root / "synthetic-smoke-invoice.png"
        image = Image.new("RGB", (1200, 820), "white")
        draw = ImageDraw.Draw(image)
        font = self._font(38)
        lines = [
            "INVOICE",
            "Vendor: Meridian Office Systems",
            "Invoice Number: INV-SMOKE-1042",
            "Invoice Date: 2026-08-04",
            "Due Date: 2026-09-03",
            "Total Amount: INR 1250.00",
            "Currency: INR",
            "Payment Terms: Net 30",
        ]
        words: list[OCRWord] = []
        reading_order = 1
        y = 60
        for line_index, line in enumerate(lines, start=1):
            x = 65
            draw.text((x, y), line, fill="black", font=font)
            for token in line.split():
                left, top, right, bottom = draw.textbbox((x, y), token, font=font)
                box = VisionBoundingBox(
                    x1=float(left),
                    y1=float(top),
                    x2=float(right),
                    y2=float(bottom),
                )
                words.append(
                    OCRWord(
                        text=token,
                        confidence=0.99,
                        polygon=[
                            [box.x1, box.y1],
                            [box.x2, box.y1],
                            [box.x2, box.y2],
                            [box.x1, box.y2],
                        ],
                        bounding_box=box,
                        page=1,
                        line_id=f"p1-line-{line_index}",
                        block_id="p1-block-1",
                        reading_order=reading_order,
                        model_name="smoke-fixture",
                        model_version="v1",
                    )
                )
                reading_order += 1
                x = right + int(draw.textlength(" ", font=font))
            y += 78
        image.save(image_path, format="PNG")
        now = self._now()
        return VisionDocumentRecord(
            document_id=document_id,
            case_id="smoke-case",
            filename=image_path.name,
            tenant_id="smoke-tenant",
            status=VisionProcessingStatus.extraction_completed,
            document_type=VisionDocumentType.invoice,
            file_hash=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            parsed_text="\n".join(lines),
            page_count=1,
            page_artifacts=[
                VisionPageArtifact(
                    page_number=1,
                    original_image_path=str(image_path),
                    processed_image_path=str(image_path),
                    width=image.width,
                    height=image.height,
                    dpi=150,
                    source_kind="synthetic_smoke_fixture",
                )
            ],
            ocr_words=words,
            created_at=now,
            updated_at=now,
        )

    def _chunk(self, document: VisionDocumentRecord) -> VisionEvidenceChunk:
        return VisionEvidenceChunk(
            chunk_id=f"smoke-chunk-{uuid4().hex[:8]}",
            document_id=document.document_id,
            tenant_id=document.tenant_id,
            page_number=1,
            document_type=VisionDocumentType.invoice,
            reviewer_status=ReviewerStatus.accepted,
            text="Invoice INV-SMOKE-1042 has a total of INR 1250.00.",
            confidence=0.99,
        )

    def _cleanup_fixture(self, document: VisionDocumentRecord) -> None:
        artifact_root = Path(settings.vision_artifact_dir) / document.document_id
        if artifact_root.exists():
            shutil.rmtree(artifact_root)

    def _close(self, adapter: Any) -> None:
        if adapter is None:
            return
        if adapter.__class__.__module__.startswith("neo4j"):
            adapter.close()
            return
        if hasattr(adapter, "collection_name") and hasattr(adapter, "_client"):
            client = getattr(adapter, "_client", None)
            if client is not None and str(adapter.collection_name).startswith(
                "documind_model_smoke_"
            ):
                try:
                    client.delete_collection(adapter.collection_name)
                except Exception:
                    pass
                close = getattr(client, "close", None)
                if callable(close):
                    close()

    def _blocker(self, backend: dict[str, Any]) -> dict[str, Any]:
        requirements = dict(backend.get("requirements") or {})
        category = backend.get("blocker_category") or "configuration"
        return {
            "category": category,
            "reason": backend.get("reason"),
            "missing_packages": requirements.get("missing_packages", []),
            "missing_checkpoint": requirements.get("missing_checkpoint", False),
            "required_environment_variables": requirements.get("environment_variables", []),
            "preparation_command": requirements.get("preparation_command"),
        }

    def _finish(self, result: dict[str, Any], started: float) -> dict[str, Any]:
        result["completed_at"] = self._now()
        result["duration_ms"] = self._duration(started)
        self.store.record(result)
        return result

    def _device(self, backend: dict[str, Any], adapter: Any | None = None) -> str:
        runtime = dict(backend.get("runtime") or {})
        configured = runtime.get("device")
        if configured:
            return str(configured)
        for attribute in ("_model", "model"):
            model = getattr(adapter, attribute, None) if adapter is not None else None
            if model is not None and getattr(model, "device", None) is not None:
                return str(model.device)
        return "cpu"

    def _font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = (
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _duration(self, started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()


def _valid_ocr_words(words: Any) -> bool:
    required = {
        "text",
        "polygon",
        "bounding_box",
        "confidence",
        "line_id",
        "block_id",
        "reading_order",
    }
    return (
        isinstance(words, list)
        and bool(words)
        and all(isinstance(item, dict) and required.issubset(item) for item in words)
    )


def _valid_generated(value: dict[str, Any]) -> bool:
    response = value.get("response")
    return (
        isinstance(response, dict)
        and response.get("answer_status") in {"supported", "partially_supported", "not_found"}
        and isinstance(response.get("citation_ids"), list)
    )


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
