from __future__ import annotations

import importlib.metadata
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models.schemas import ModelRoutingPreviewRequest


class ModelRegistryService:
    """Runtime registry for target Vision AI models and local fallbacks.

    The project can name production adapters without pretending they are active.
    A heavy backend is active only when its feature flag is enabled, the
    corresponding Python package can be imported, and an adapter is wired into
    the current business pipeline.
    """

    def registry(self) -> dict[str, Any]:
        backends = self.backends()
        return {
            "project_name": "DOCUMIND VISION AI",
            "model_profile": self._profile(),
            "heavy_models_enabled": settings.enable_heavy_models,
            "generated_at": self._now(),
            "evidence_policy": {
                "fraud_language": (
                    "Suspicious indicators require human review; only Compliance Officers, "
                    "Senior Reviewers, or Admins may confirm fraud."
                ),
                "unsupported_answer": (
                    "The available documents do not contain sufficient evidence to answer this question."
                ),
                "fake_metrics_policy": "Metrics are shown only when calculated from labelled validation or test data.",
            },
            "active_local_pipeline": [
                item["model_id"] for item in backends if item["status"] == "active"
            ],
            "backends": backends,
            "adapter_roles": self._adapter_roles(backends),
            "routing_policy": self.routing_preview().get("policy"),
        }

    def backends(self, *, include_runtime_health: bool = True) -> list[dict[str, Any]]:
        from app.services.vision.advanced_adapters import PaddleOCRVLAdapter
        from app.services.vision.anomaly_adapters import (
            IsolationForestRiskAdapter,
            XGBoostRiskAdapter,
        )
        from app.services.vision.contract_adapters import DebertaContractClauseAdapter
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
        from app.services.vision.layout_adapters import (
            GeometryLayoutAdapter,
            PaddleStructureV3Adapter,
        )
        from app.services.vision.ocr_adapters import PaddleOCRV5Adapter, TesseractOCRAdapter
        from app.services.vision.retrieval_adapters import (
            BGEM3EmbeddingAdapter,
            BGERerankerAdapter,
            LocalHashingEmbeddingAdapter,
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

        adapter_status = {
            "paddleocr": PaddleOCRV5Adapter().status(),
            "structure": PaddleStructureV3Adapter().status(),
            "layout_local": GeometryLayoutAdapter().status(),
            "layoutlmv3": LayoutLMv3ExtractionAdapter().status(),
            "paddleocr_vl": PaddleOCRVLAdapter().status(),
            "donut": DonutExtractionAdapter().status(),
            "dinov2": DinoV2DuplicateAdapter().status(),
            "yolo": Yolo26VisualDetectionAdapter().status(),
            "visual_local": OpenCVVisualDetectionAdapter().status(),
            "tesseract": TesseractOCRAdapter().status(),
            "contract": DebertaContractClauseAdapter().status(),
            "forensics_local": LocalForensicsAdapter().status(),
            "forensics_model": SegFormerTamperAdapter().status(),
            "isolation_forest": IsolationForestRiskAdapter().status(),
            "xgboost": XGBoostRiskAdapter().status(),
            "bge_m3": BGEM3EmbeddingAdapter().status(),
            "bge_reranker": BGERerankerAdapter().status(),
            "hashing": LocalHashingEmbeddingAdapter().status(),
            "qdrant": QdrantHybridBackend().status(),
            "qwen": QwenGroundedGeneratorAdapter().status(),
            "external_generator": OpenAICompatibleGroundedGenerator().status(),
            "template_generator": GroundedTemplateGeneratorAdapter().status(),
        }
        backends = [
            self._local_backend(
                stage="ingestion",
                role="native PDF/DOCX/TXT parser and image fallback",
                model_id="local-parser-v1",
                packages=["fitz", "pypdf", "docx"],
                notes="Current default parser before OCR/layout routing.",
            ),
            self._local_backend(
                stage="ocr",
                role="word-level OCR fallback with measured boxes and confidence",
                model_id="Tesseract-5",
                packages=["pytesseract"],
                notes=adapter_status["tesseract"].reason,
                runtime_status=adapter_status["tesseract"],
            ),
            self._local_backend(
                stage="ocr_layout",
                role="geometry-based structure and table fallback",
                model_id="local-layout-adapter-v1",
                packages=["cv2"],
                notes="Builds regions and table structure from measured OCR geometry.",
                runtime_status=adapter_status["layout_local"],
            ),
            self._target_backend(
                stage="ocr",
                role="primary word-level OCR",
                model_id=self._ocr_model_id(),
                provider="PaddleOCR",
                enabled=settings.enable_heavy_models and settings.enable_paddleocr,
                packages=["paddleocr", "paddle"],
                enable_hint="Set ENABLE_HEAVY_MODELS=true and ENABLE_PADDLEOCR=true.",
                notes="Target adapter for word polygons, bounding boxes, confidence, line IDs, and reading order.",
                runtime_status=adapter_status["paddleocr"],
            ),
            self._target_backend(
                stage="layout",
                role="document structure and table parsing",
                model_id="PP-StructureV3 / PP-DocLayoutV3",
                provider="PaddleOCR",
                enabled=settings.enable_heavy_models and settings.enable_paddleocr_structure,
                packages=["paddleocr", "paddle"],
                enable_hint="Set ENABLE_HEAVY_MODELS=true and ENABLE_PADDLEOCR_STRUCTURE=true.",
                notes="Target adapter for structure-aware regions, tables, merged cells, and reading order.",
                runtime_status=adapter_status["structure"],
            ),
            self._target_backend(
                stage="classification_extraction",
                role="schema-specific classification and key information extraction",
                model_id=self._layoutlmv3_model_id(),
                provider="Hugging Face Transformers",
                enabled=settings.enable_heavy_models and settings.enable_layoutlmv3,
                packages=["transformers"],
                enable_hint="Set ENABLE_HEAVY_MODELS=true and ENABLE_LAYOUTLMV3=true.",
                notes="Target adapter consumes OCR words, normalized 0-1000 boxes, page image, and token mappings.",
                runtime_status=adapter_status["layoutlmv3"],
            ),
            self._local_backend(
                stage="classification_extraction",
                role="schema-aware deterministic field extraction fallback",
                model_id="deterministic-schema-extractor-v3",
                packages=[],
                notes="Extracts only source-backed values and reports missing schema fields explicitly.",
                runtime_version="v3",
            ),
            self._target_backend(
                stage="advanced_parsing",
                role="low-confidence visual parsing fallback",
                model_id="PaddleOCR-VL-1.6",
                provider="PaddleOCR",
                enabled=settings.enable_heavy_models and settings.enable_paddleocr_vl,
                packages=["paddleocr", "paddle"],
                enable_hint="Set ENABLE_HEAVY_MODELS=true and ENABLE_PADDLEOCR_VL=true.",
                notes="Run only for low confidence, complex layout, conflict, mixed-language, or manual advanced mode.",
                runtime_status=adapter_status["paddleocr_vl"],
            ),
            self._target_backend(
                stage="ocr_free_baseline",
                role="OCR-free extraction and disagreement verification",
                model_id="Donut",
                provider="Hugging Face Transformers",
                enabled=settings.enable_heavy_models and settings.enable_donut,
                packages=["transformers"],
                enable_hint="Set ENABLE_HEAVY_MODELS=true and ENABLE_DONUT=true.",
                notes="Run only for model comparison, low-confidence OCR, or critical-field disagreement.",
                runtime_status=adapter_status["donut"],
            ),
            self._target_backend(
                stage="visual_duplicate_embedding",
                role="document-image near-duplicate embedding",
                model_id="DINOv2",
                provider="Hugging Face Transformers",
                enabled=settings.enable_heavy_models and settings.enable_dinov2,
                packages=["transformers", "torch"],
                enable_hint="Provide DINOV2_CHECKPOINT and enable DINOv2.",
                notes="Cosine similarity complements SHA-256 and multi-page perceptual hash.",
                runtime_status=adapter_status["dinov2"],
            ),
            self._target_backend(
                stage="visual_detection",
                role="signatures, stamps, logos, checkboxes, handwriting, and altered-region candidates",
                model_id=self._yolo_model_id(),
                provider="Ultralytics",
                enabled=settings.enable_heavy_models and settings.enable_yolo26,
                packages=["ultralytics"],
                enable_hint="Set ENABLE_HEAVY_MODELS=true and ENABLE_YOLO26=true.",
                notes="Detection does not claim signature authenticity.",
                runtime_status=adapter_status["yolo"],
            ),
            self._local_backend(
                stage="visual_detection",
                role="deterministic visual-cue fallback",
                model_id="OpenCV-visual-cues-v1",
                packages=["cv2"],
                notes=adapter_status["visual_local"].reason,
                runtime_status=adapter_status["visual_local"],
            ),
            self._target_backend(
                stage="contract_intelligence",
                role="CUAD-style contract clause classification",
                model_id="DeBERTa-v3-large-CUAD",
                provider="Hugging Face Transformers",
                enabled=settings.enable_heavy_models and bool(settings.contract_checkpoint),
                packages=["transformers", "torch"],
                enable_hint="Provide CONTRACT_CHECKPOINT and enable heavy models.",
                notes="Clause output is evidence-localized and never presented as final legal judgement.",
                runtime_status=adapter_status["contract"],
            ),
            self._local_backend(
                stage="tamper_screening",
                role="ELA, noise residual, and high-pass visual inconsistency screening",
                model_id="ELA-noise-forensics-v1",
                packages=["cv2", "PIL"],
                notes=adapter_status["forensics_local"].reason,
                runtime_status=adapter_status["forensics_local"],
            ),
            self._target_backend(
                stage="tamper_screening",
                role="fine-tuned document manipulation segmentation",
                model_id="SegFormer-B2-document-forensics",
                provider="Hugging Face Transformers",
                enabled=settings.enable_heavy_models and bool(settings.tamper_checkpoint),
                packages=["transformers", "torch"],
                enable_hint="Provide TAMPER_CHECKPOINT and enable heavy models.",
                notes="Outputs possible manipulation regions; human verification remains mandatory.",
                runtime_status=adapter_status["forensics_model"],
            ),
            self._local_backend(
                stage="anomaly",
                role="unsupervised financial and document anomaly ranking",
                model_id="IsolationForest",
                packages=["sklearn"],
                notes="Runs only after the configured minimum organisation corpus size is reached.",
                runtime_status=adapter_status["isolation_forest"],
            ),
            self._target_backend(
                stage="anomaly",
                role="supervised risk classification with SHAP contributions",
                model_id="XGBoost-project-risk-model",
                provider="XGBoost",
                enabled=settings.enable_heavy_models and bool(settings.xgboost_risk_checkpoint),
                packages=["xgboost"],
                enable_hint="Train and provide XGBOOST_RISK_CHECKPOINT.",
                notes="Activation requires labelled data and an explicit trained checkpoint.",
                runtime_status=adapter_status["xgboost"],
            ),
            self._local_backend(
                stage="retrieval",
                role="local hashed embeddings",
                model_id="local-hashing-v1",
                packages=[],
                notes="Current default embedding path for runnable local retrieval.",
                runtime_status=adapter_status["hashing"],
            ),
            self._target_backend(
                stage="retrieval",
                role="hybrid dense/sparse/multi-vector embeddings",
                model_id="BAAI/bge-m3",
                provider="FlagEmbedding or sentence-transformers",
                enabled=settings.enable_heavy_models and settings.enable_bge_m3,
                packages=["FlagEmbedding"],
                enable_hint="Set ENABLE_HEAVY_MODELS=true and ENABLE_BGE_M3=true.",
                notes="Target retriever for long multilingual document chunks.",
                runtime_status=adapter_status["bge_m3"],
            ),
            self._target_backend(
                stage="retrieval_reranking",
                role="cross-encoder evidence reranking",
                model_id=self._bge_reranker_model_id(),
                provider="FlagEmbedding",
                enabled=settings.enable_heavy_models and settings.enable_bge_m3,
                packages=["FlagEmbedding"],
                enable_hint="Enable BGE-M3 and install FlagEmbedding.",
                notes="Reranks the top 15 hybrid candidates before evidence selection.",
                runtime_status=adapter_status["bge_reranker"],
            ),
            self._local_backend(
                stage="vector_store",
                role="in-memory vector store",
                model_id="local-vector-store-v1",
                packages=[],
                notes="Current default store for tests and local demos.",
            ),
            self._target_backend(
                stage="vector_store",
                role="production vector database",
                model_id="Qdrant",
                provider="Qdrant",
                enabled=settings.enable_qdrant,
                packages=["qdrant_client"],
                enable_hint="Set ENABLE_QDRANT=true and configure Qdrant service.",
                notes="Target backend for dense vectors, sparse vectors, and metadata filters.",
                runtime_status=adapter_status["qdrant"],
            ),
            self._local_backend(
                stage="graph",
                role="local graph projection",
                model_id="local-graph-projection-v1",
                packages=[],
                notes="Current default knowledge graph representation in API responses.",
            ),
            self._target_backend(
                stage="graph",
                role="persistent graph database",
                model_id="Neo4j",
                provider="Neo4j",
                enabled=settings.enable_neo4j,
                packages=["neo4j"],
                enable_hint="Set ENABLE_NEO4J=true and configure NEO4J_URI.",
                notes="Target backend for verified entities, proposed relationships, and GraphRAG paths.",
                adapter_implemented=True,
            ),
            self._local_backend(
                stage="generator",
                role="deterministic grounded response composer",
                model_id="grounded-template-generator-v1",
                packages=[],
                notes="Current default: answers are assembled from retrieved evidence.",
                runtime_status=adapter_status["template_generator"],
            ),
            self._target_backend(
                stage="generator",
                role="local provider-independent LLM adapter",
                model_id=self._local_llm_model_id(),
                provider="local-transformers",
                enabled=settings.enable_heavy_models and settings.enable_qwen_local,
                packages=["transformers"],
                enable_hint="Set ENABLE_HEAVY_MODELS=true and ENABLE_QWEN_LOCAL=true.",
                notes=(
                    "Exact configured local Transformers checkpoint; generator receives only "
                    "retrieved evidence, verified facts, and deterministic calculations."
                ),
                runtime_status=adapter_status["qwen"],
            ),
            self._target_backend(
                stage="generator",
                role="optional OpenAI-compatible evidence-grounded generator",
                model_id=settings.external_llm_model or "external-provider-model",
                provider="configured OpenAI-compatible endpoint",
                enabled=settings.enable_external_llm,
                packages=["httpx"],
                enable_hint="Configure and enable the external LLM provider.",
                notes="Receives only retrieved evidence and is subject to citation/numeric grounding checks.",
                runtime_status=adapter_status["external_generator"],
            ),
        ]
        enriched = [self._enrich_requirements(item) for item in backends]
        if not include_runtime_health:
            return enriched
        return [self._attach_runtime_health(item) for item in enriched]

    def route_document(
        self,
        *,
        filename: str,
        text: str,
        parse_confidence: float,
        layout_result: dict[str, Any],
        mixed_languages: bool = False,
        language_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        regions = set(layout_result.get("regions", []))
        table_separators = text.count("|") + text.count("\t")
        request = ModelRoutingPreviewRequest(
            ocr_mean_confidence=parse_confidence,
            reading_order_confidence=layout_result.get("evidence", {}).get("confidence", 0.74),
            has_complex_table="table" in regions and table_separators >= 6,
            has_chart="chart" in text.lower(),
            mixed_languages=mixed_languages,
            model_outputs_conflict=False,
            manual_advanced=False,
            model_comparison_mode=False,
        )
        route = self.routing_preview(request)
        route["source"] = {
            "filename": filename,
            "layout_engine": layout_result.get("engine", "unknown"),
            "detected_regions": sorted(regions),
            "language_analysis": language_analysis or {},
        }
        return route

    def routing_preview(self, request: ModelRoutingPreviewRequest | None = None) -> dict[str, Any]:
        request = request or ModelRoutingPreviewRequest()
        reasons = self._fallback_reasons(request)
        fallback_required = bool(reasons)
        standard_models = [
            self._model_state(self._ocr_model_id()),
            self._model_state("PP-StructureV3 / PP-DocLayoutV3"),
            self._model_state(self._layoutlmv3_model_id()),
        ]
        requested_fallbacks = []
        if fallback_required:
            requested_fallbacks.append(self._model_state("PaddleOCR-VL-1.6"))
        if (
            request.model_comparison_mode
            or request.model_outputs_conflict
            or request.ocr_mean_confidence < settings.ocr_low_confidence_threshold
        ):
            requested_fallbacks.append(self._model_state("Donut"))

        return {
            "model_profile": self._profile(),
            "fallback_required": fallback_required,
            "reasons": reasons,
            "standard_models": standard_models,
            "fallback_models": requested_fallbacks,
            "unavailable_requested_models": [
                model
                for model in [*standard_models, *requested_fallbacks]
                if model["status"] != "active"
            ],
            "policy": {
                "normal_pages": [
                    self._ocr_model_id(),
                    "PP-StructureV3 / PP-DocLayoutV3",
                    self._layoutlmv3_model_id(),
                ],
                "fallback_triggers": [
                    "ocr_mean_confidence_below_threshold",
                    "reading_order_confidence_below_threshold",
                    "complex_table_or_chart_detected",
                    "mixed_languages_detected",
                    "model_outputs_conflict",
                    "manual_advanced_requested",
                ],
                "ocr_low_confidence_threshold": settings.ocr_low_confidence_threshold,
                "reading_order_low_confidence_threshold": settings.reading_order_low_confidence_threshold,
                "never_auto_confirm_fraud": True,
            },
            "generated_at": self._now(),
        }

    def _fallback_reasons(self, request: ModelRoutingPreviewRequest) -> list[str]:
        reasons = []
        if request.ocr_mean_confidence < settings.ocr_low_confidence_threshold:
            reasons.append("ocr_mean_confidence_below_threshold")
        if request.reading_order_confidence < settings.reading_order_low_confidence_threshold:
            reasons.append("reading_order_confidence_below_threshold")
        if request.has_complex_table or request.has_chart:
            reasons.append("complex_table_or_chart_detected")
        if request.mixed_languages:
            reasons.append("mixed_languages_detected")
        if request.model_outputs_conflict:
            reasons.append("model_outputs_conflict")
        if request.manual_advanced:
            reasons.append("manual_advanced_requested")
        return reasons

    def _model_state(self, model_id: str) -> dict[str, Any]:
        matches = [backend for backend in self.backends() if backend["model_id"] == model_id]
        if matches:
            backend = matches[0]
            return {
                "model_id": backend["model_id"],
                "stage": backend["stage"],
                "status": backend["status"],
                "reason": backend["reason"],
            }
        return {
            "model_id": model_id,
            "stage": "unknown",
            "status": "unavailable",
            "reason": "Model is not registered in the runtime registry.",
        }

    def _local_backend(
        self,
        *,
        stage: str,
        role: str,
        model_id: str,
        packages: list[str],
        notes: str,
        runtime_status: Any | None = None,
        runtime_version: str | None = None,
    ) -> dict[str, Any]:
        if runtime_status is not None:
            return {
                "stage": stage,
                "role": role,
                "model_id": model_id,
                "adapter_id": runtime_status.adapter_id,
                "provider": "local",
                "profile": "LITE",
                "enabled": runtime_status.enabled,
                "installed": runtime_status.installed,
                "status": runtime_status.status,
                "configured_status": runtime_status.status,
                "reason": runtime_status.reason,
                "packages": packages,
                "notes": notes,
                "runtime": runtime_status.metadata,
            }
        missing = [package for package in packages if not self._installed(package)]
        return {
            "stage": stage,
            "role": role,
            "model_id": model_id,
            "adapter_id": model_id,
            "provider": "local",
            "profile": "LITE",
            "enabled": True,
            "installed": not missing,
            "status": "active" if not missing else "degraded",
            "configured_status": "active" if not missing else "degraded",
            "reason": "Active local runtime path."
            if not missing
            else f"Missing optional packages: {', '.join(missing)}.",
            "packages": packages,
            "notes": notes,
            "runtime": {"model_version": runtime_version} if runtime_version else {},
        }

    def _target_backend(
        self,
        *,
        stage: str,
        role: str,
        model_id: str,
        provider: str,
        enabled: bool,
        packages: list[str],
        enable_hint: str,
        notes: str,
        adapter_implemented: bool = False,
        runtime_status: Any | None = None,
    ) -> dict[str, Any]:
        if runtime_status is not None:
            return {
                "stage": stage,
                "role": role,
                "model_id": model_id,
                "adapter_id": runtime_status.adapter_id,
                "provider": provider,
                "profile": self._profile(),
                "enabled": runtime_status.enabled,
                "installed": runtime_status.installed,
                "adapter_implemented": True,
                "status": runtime_status.status,
                "configured_status": runtime_status.status,
                "reason": runtime_status.reason,
                "packages": packages,
                "notes": notes,
                "runtime": runtime_status.metadata,
            }
        missing = [package for package in packages if not self._installed(package)]
        if enabled and not missing and adapter_implemented:
            status = "active"
            reason = "Feature flag is enabled, required package imports are available, and adapter is wired."
        elif enabled:
            if missing:
                status = "unavailable"
                reason = f"Feature flag is enabled but packages are missing: {', '.join(missing)}."
            else:
                status = "adapter_pending"
                reason = "Required packages are importable, but this adapter is not yet wired into the runtime pipeline."
        else:
            status = "disabled"
            reason = enable_hint
        return {
            "stage": stage,
            "role": role,
            "model_id": model_id,
            "adapter_id": model_id,
            "provider": provider,
            "profile": self._profile(),
            "enabled": enabled,
            "installed": not missing,
            "adapter_implemented": adapter_implemented,
            "status": status,
            "configured_status": status,
            "reason": reason,
            "packages": packages,
            "notes": notes,
        }

    def _attach_runtime_health(self, backend: dict[str, Any]) -> dict[str, Any]:
        from app.services.vision.runtime_health import (
            model_configuration_fingerprint,
            model_smoke_store,
        )

        item = dict(backend)
        configured_status = str(item.get("configured_status", item.get("status", "unavailable")))
        fingerprint = model_configuration_fingerprint(item)
        smoke = model_smoke_store.latest(str(item["model_id"]))
        smoke_current = bool(smoke and smoke.get("configuration_fingerprint") == fingerprint)
        smoke_status = str(smoke.get("status")) if smoke_current and smoke else "not_run"

        health_status = "unknown"
        last_error = None
        runtime_status = configured_status
        if configured_status == "active":
            if smoke_status == "passed":
                runtime_status = "active"
                health_status = "healthy"
            elif smoke_status == "failed":
                runtime_status = "error"
                health_status = "unhealthy"
                last_error = smoke.get("error") if smoke else None
            else:
                runtime_status = "unavailable"
                item["reason"] = (
                    "Configuration is present, but this exact runtime has not passed an "
                    "initialisation and inference smoke test."
                )
        elif configured_status == "disabled":
            runtime_status = "disabled"
        elif configured_status in {"adapter_pending", "degraded", "misconfigured"}:
            runtime_status = "unavailable"
        else:
            runtime_status = configured_status

        if smoke_current and smoke:
            last_error = last_error or smoke.get("error")
        runtime_metadata = dict(item.get("runtime") or {})
        item.update(
            {
                "status": runtime_status,
                "runtime_status": runtime_status,
                "health_status": health_status,
                "requested": bool(item.get("enabled")),
                "last_smoke_test": smoke.get("completed_at") if smoke_current and smoke else None,
                "last_error": last_error,
                "smoke_test_status": smoke_status,
                "smoke_test": smoke if smoke_current else None,
                "checkpoint": runtime_metadata.get("checkpoint"),
                "device": (
                    smoke.get("device")
                    if smoke_current and smoke and smoke.get("device")
                    else runtime_metadata.get("device", "cpu")
                ),
                "configuration_fingerprint": fingerprint,
            }
        )
        if smoke and not smoke_current:
            item["stale_smoke_test"] = {
                "completed_at": smoke.get("completed_at"),
                "status": smoke.get("status"),
            }
        return item

    def _enrich_requirements(self, backend: dict[str, Any]) -> dict[str, Any]:
        item = dict(backend)
        requirements = self._requirements_for(str(item["model_id"]))
        missing_packages = [
            package for package in item.get("packages", []) if not self._installed(package)
        ]
        runtime = dict(item.get("runtime") or {})
        checkpoint_value = runtime.get("checkpoint")
        checkpoint_required = bool(requirements.get("checkpoint_required"))
        checkpoint_present = self._checkpoint_present(checkpoint_value)
        missing_checkpoint = checkpoint_required and not checkpoint_present
        blocker_category = None
        if not item.get("enabled"):
            blocker_category = "environment"
        elif missing_packages:
            blocker_category = "package"
        elif missing_checkpoint:
            blocker_category = "checkpoint"
        elif str(item.get("configured_status")) != "active":
            blocker_category = "preparation"
        item.update(
            {
                "package_version": self._package_version(item.get("packages", [])),
                "package_versions": {
                    package: version
                    for package in item.get("packages", [])
                    if (version := self._package_version([package])) is not None
                },
                "requirements": {
                    **requirements,
                    "missing_packages": missing_packages,
                    "missing_checkpoint": missing_checkpoint,
                    "checkpoint_present": checkpoint_present,
                },
                "blocker_category": blocker_category,
            }
        )
        return item

    def _requirements_for(self, model_id: str) -> dict[str, Any]:
        profile_ocr = "PP-OCRv5_mobile" if self._profile() == "LITE" else "PP-OCRv5_server"
        profile_layoutlm = "LayoutLMv3-base" if self._profile() == "LITE" else "LayoutLMv3-large"
        profile_yolo = self._yolo_model_id()
        values: dict[str, dict[str, Any]] = {
            profile_ocr: {
                "environment_variables": ["ENABLE_HEAVY_MODELS", "ENABLE_PADDLEOCR"],
                "preparation_command": "scripts/bootstrap_ai_core.sh --allow-paddle-model-downloads",
            },
            "PP-StructureV3 / PP-DocLayoutV3": {
                "environment_variables": ["ENABLE_HEAVY_MODELS", "ENABLE_PADDLEOCR_STRUCTURE"],
                "preparation_command": "scripts/bootstrap_ai_core.sh --allow-paddle-model-downloads",
            },
            profile_layoutlm: {
                "environment_variables": [
                    "ENABLE_HEAVY_MODELS",
                    "ENABLE_LAYOUTLMV3",
                    "LAYOUTLMV3_CHECKPOINT",
                ],
                "checkpoint_required": True,
                "preparation_command": "python -m ml.training.layoutlmv3.train_token_classification --config ml/configs/layoutlmv3_token.example.yml",
            },
            "PaddleOCR-VL-1.6": {
                "environment_variables": ["ENABLE_HEAVY_MODELS", "ENABLE_PADDLEOCR_VL"],
                "preparation_command": "python -m pip install -r backend/requirements/ai-full.txt",
            },
            "Donut": {
                "environment_variables": [
                    "ENABLE_HEAVY_MODELS",
                    "ENABLE_DONUT",
                    "DONUT_CHECKPOINT",
                ],
                "checkpoint_required": True,
                "preparation_command": "python -m ml.training.donut.train --config <donut-training-config.yml>",
            },
            "DINOv2": {
                "environment_variables": [
                    "ENABLE_HEAVY_MODELS",
                    "ENABLE_DINOV2",
                    "DINOV2_CHECKPOINT",
                ],
                "checkpoint_required": True,
                "preparation_command": "Place a local DINOv2 checkpoint and set DINOV2_CHECKPOINT.",
            },
            profile_yolo: {
                "environment_variables": [
                    "ENABLE_HEAVY_MODELS",
                    "ENABLE_YOLO26",
                    "YOLO26_CHECKPOINT",
                ],
                "checkpoint_required": True,
                "preparation_command": "python -m ml.training.yolo26.train --config <visual-detection-training-config.yml>",
            },
            "DeBERTa-v3-large-CUAD": {
                "environment_variables": ["ENABLE_HEAVY_MODELS", "CONTRACT_CHECKPOINT"],
                "checkpoint_required": True,
                "preparation_command": "python -m ml.training.contract_model.train --config <contract-training-config.yml>",
            },
            "SegFormer-B2-document-forensics": {
                "environment_variables": ["ENABLE_HEAVY_MODELS", "TAMPER_CHECKPOINT"],
                "checkpoint_required": True,
                "preparation_command": "python -m ml.training.tamper_model.train --config <tamper-training-config.yml>",
            },
            "XGBoost-project-risk-model": {
                "environment_variables": ["ENABLE_HEAVY_MODELS", "XGBOOST_RISK_CHECKPOINT"],
                "checkpoint_required": True,
                "preparation_command": "python -m ml.training.anomaly_models.train_xgboost --config <risk-training-config.yml>",
            },
            "BAAI/bge-m3": {
                "environment_variables": [
                    "ENABLE_HEAVY_MODELS",
                    "ENABLE_BGE_M3",
                    "BGE_M3_CHECKPOINT",
                ],
                "checkpoint_required": not settings.bge_allow_download,
                "preparation_command": "scripts/bootstrap_ai_core.sh --allow-bge-downloads",
            },
            self._bge_reranker_model_id(): {
                "environment_variables": [
                    "ENABLE_HEAVY_MODELS",
                    "ENABLE_BGE_M3",
                    "BGE_RERANKER_CHECKPOINT",
                ],
                "checkpoint_required": not settings.bge_allow_download,
                "preparation_command": "scripts/bootstrap_ai_core.sh --allow-bge-downloads",
            },
            "Qdrant": {
                "environment_variables": ["ENABLE_QDRANT", "QDRANT_URL"],
                "preparation_command": "docker compose up -d qdrant",
            },
            "Neo4j": {
                "environment_variables": [
                    "ENABLE_NEO4J",
                    "NEO4J_URI",
                    "NEO4J_USER",
                    "NEO4J_PASSWORD",
                ],
                "preparation_command": "docker compose up -d neo4j",
            },
            self._local_llm_model_id(): {
                "environment_variables": [
                    "ENABLE_HEAVY_MODELS",
                    "ENABLE_QWEN_LOCAL",
                    "QWEN_CHECKPOINT",
                ],
                "checkpoint_required": not settings.qwen_allow_download,
                "preparation_command": (
                    "scripts/bootstrap_ai_core.sh --local-llm-checkpoint "
                    f"{settings.qwen_checkpoint!r}"
                ),
            },
            settings.external_llm_model or "external-provider-model": {
                "environment_variables": [
                    "ENABLE_EXTERNAL_LLM",
                    "EXTERNAL_LLM_BASE_URL",
                    "EXTERNAL_LLM_API_KEY",
                    "EXTERNAL_LLM_MODEL",
                ],
                "preparation_command": "Configure the OpenAI-compatible provider variables, then run the model smoke test.",
            },
        }
        return values.get(
            model_id,
            {
                "environment_variables": [],
                "checkpoint_required": False,
                "preparation_command": f"python scripts/smoke_test_models.py --model {model_id!r}",
            },
        )

    def _adapter_roles(self, backends: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id = {str(item["model_id"]): item for item in backends}
        pairs = [
            ("ocr", self._ocr_model_id(), "Tesseract-5"),
            ("layout", "PP-StructureV3 / PP-DocLayoutV3", "local-layout-adapter-v1"),
            ("field_extraction", self._layoutlmv3_model_id(), "deterministic-schema-extractor-v2"),
            ("visual_detection", self._yolo_model_id(), "OpenCV-visual-cues-v1"),
            ("retrieval", "BAAI/bge-m3", "local-hashing-v1"),
            ("vector_store", "Qdrant", "local-vector-store-v1"),
            ("graph", "Neo4j", "local-graph-projection-v1"),
            ("generator", self._local_llm_model_id(), "grounded-template-generator-v1"),
        ]
        output = []
        for stage, primary_id, fallback_id in pairs:
            primary = by_id.get(primary_id)
            fallback = by_id.get(fallback_id)
            active = primary if primary and primary.get("status") == "active" else fallback
            if active and active.get("status") != "active":
                active = None
            output.append(
                {
                    "stage": stage,
                    "primary": self._role_summary(primary_id, primary),
                    "fallback": self._role_summary(fallback_id, fallback),
                    "active_adapter": self._role_summary(str(active["model_id"]), active)
                    if active
                    else None,
                }
            )
        return output

    def _role_summary(self, model_id: str, backend: dict[str, Any] | None) -> dict[str, Any]:
        if backend is None:
            return {"model_id": model_id, "status": "unavailable", "reason": "Not registered."}
        return {
            "model_id": model_id,
            "adapter_id": backend.get("adapter_id"),
            "status": backend.get("status"),
            "health_status": backend.get("health_status"),
            "reason": backend.get("reason"),
        }

    def _checkpoint_present(self, value: Any) -> bool:
        if not value:
            return False
        from app.services.vision.checkpoint_utils import local_or_hf_cached_checkpoint

        return local_or_hf_cached_checkpoint(str(value)) is not None

    def _package_version(self, packages: list[str]) -> str | None:
        for package in packages:
            distribution_names = {
                "fitz": ["PyMuPDF"],
                "docx": ["python-docx"],
                "cv2": ["opencv-python-headless", "opencv-python"],
                "PIL": ["Pillow"],
                "sklearn": ["scikit-learn"],
                "paddle": ["paddlepaddle"],
                "qdrant_client": ["qdrant-client"],
            }
            candidates = [
                *distribution_names.get(package, []),
                package,
                package.replace("_", "-"),
                package.lower(),
            ]
            for candidate in candidates:
                try:
                    return importlib.metadata.version(candidate)
                except importlib.metadata.PackageNotFoundError:
                    continue
        return None

    def _ocr_model_id(self) -> str:
        return "PP-OCRv5_mobile" if self._profile() == "LITE" else "PP-OCRv5_server"

    def _layoutlmv3_model_id(self) -> str:
        return "LayoutLMv3-base" if self._profile() == "LITE" else "LayoutLMv3-large"

    def _yolo_model_id(self) -> str:
        profile = self._profile()
        if profile == "FULL":
            return "YOLO26m"
        if profile == "STANDARD":
            return "YOLO26s"
        return "YOLO26n"

    def _local_llm_model_id(self) -> str:
        configured = str(settings.qwen_checkpoint or "").strip()
        if configured and not Path(configured).expanduser().exists():
            return configured
        if configured:
            config_path = Path(configured).expanduser() / "config.json"
            try:
                import json

                payload = json.loads(config_path.read_text(encoding="utf-8"))
                source_name = str(payload.get("_name_or_path") or "").strip()
                if source_name:
                    return source_name
            except (OSError, ValueError, TypeError):
                pass
            return f"local:{Path(configured).expanduser().name}"
        return "local-transformers-model"

    def _bge_reranker_model_id(self) -> str:
        return str(settings.bge_reranker_checkpoint or "BAAI/bge-reranker-v2-m3")

    def _profile(self) -> str:
        value = (settings.model_profile or "LITE").upper()
        return value if value in {"LITE", "STANDARD", "FULL"} else "LITE"

    def _installed(self, package: str) -> bool:
        try:
            return find_spec(package) is not None
        except (ImportError, ValueError):
            return False

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()
