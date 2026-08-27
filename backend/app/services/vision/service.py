from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models.vision import (
    VisionConversationState,
    VisionDocumentRecord,
    VisionFinanceNarrative,
    VisionReportRecord,
    VisionReviewTask,
)
from app.services.classification import DocumentClassificationService
from app.services.ingestion import DocumentIngestionService
from app.services.model_registry import ModelRegistryService
from app.services.vision.advanced_adapters import PaddleOCRVLAdapter
from app.services.vision.anomaly_adapters import IsolationForestRiskAdapter, XGBoostRiskAdapter
from app.services.vision.contract_adapters import DebertaContractClauseAdapter
from app.services.vision.duplicate_detection import (
    DinoV2DuplicateAdapter,
)
from app.services.vision.extraction_adapters import (
    DonutExtractionAdapter,
    LayoutLMv3ClassificationAdapter,
    LayoutLMv3ExtractionAdapter,
)
from app.services.vision.generator_adapters import GroundedGeneratorRouter
from app.services.vision.layout_adapters import GeometryLayoutAdapter, PaddleStructureV3Adapter
from app.services.vision.model_smoke import ModelSmokeService
from app.services.vision.ocr_adapters import PaddleOCRV5Adapter, TesseractOCRAdapter
from app.services.vision.persistence import vision_persistence
from app.services.vision.preprocessing import DocumentPreprocessor
from app.services.vision.query_router import VisionQueryRouter
from app.services.vision.retrieval_adapters import LocalHybridRetrievalIndex
from app.services.vision.reviewer_feedback import ReviewerFeedbackDataset
from app.services.vision.service_parts import (
    AdminMixin,
    AnalyticsMixin,
    AnsweringMixin,
    ArithmeticMixin,
    ConfidenceMixin,
    DemoSeedMixin,
    DocumentMixin,
    ExtractionMixin,
    GraphMixin,
    IngestionMixin,
    NormalizationMixin,
    ProvenanceMixin,
    ReportingMixin,
    RetrievalMixin,
    ReviewMixin,
    RiskMixin,
    StateMixin,
)
from app.services.vision.service_parts.constants import (
    CLARIFICATION_OPTIONS,
    CONTRACT_CLAUSE_FIELDS,
    LOGGER,
    RETRIEVAL_SOURCE_SEQUENCE,
)
from app.services.vision.tamper_adapters import LocalForensicsAdapter, SegFormerTamperAdapter
from app.services.vision.visual_adapters import (
    OpenCVVisualDetectionAdapter,
    Yolo26VisualDetectionAdapter,
)

__all__ = [
    "CLARIFICATION_OPTIONS",
    "CONTRACT_CLAUSE_FIELDS",
    "LOGGER",
    "RETRIEVAL_SOURCE_SEQUENCE",
    "VisionAIService",
    "vision_ai_service",
]


class VisionAIService(
    IngestionMixin,
    DocumentMixin,
    ReviewMixin,
    DemoSeedMixin,
    AnsweringMixin,
    ArithmeticMixin,
    GraphMixin,
    RetrievalMixin,
    ExtractionMixin,
    RiskMixin,
    AnalyticsMixin,
    AdminMixin,
    ReportingMixin,
    ProvenanceMixin,
    ConfidenceMixin,
    NormalizationMixin,
    StateMixin,
):
    """Document intelligence service.

    The behaviour lives in the mixins under ``service_parts``; this class
    owns construction and composes them. Splitting the former 9,500-line
    body did not change any method, so ``self`` still reaches every
    collaborator exactly as before.
    """

    def __init__(self) -> None:
        self.ingestion = DocumentIngestionService()
        self.classifier = DocumentClassificationService()
        self.model_registry = ModelRegistryService()
        self.model_smoke = ModelSmokeService(registry=self.model_registry)
        self.preprocessor = DocumentPreprocessor()
        self.local_ocr = TesseractOCRAdapter()
        self.paddle_ocr = PaddleOCRV5Adapter()
        self.local_layout = GeometryLayoutAdapter()
        self.paddle_structure = PaddleStructureV3Adapter()
        self.local_visual_detector = OpenCVVisualDetectionAdapter()
        self.yolo_detector = Yolo26VisualDetectionAdapter()
        self.layoutlmv3_classifier = LayoutLMv3ClassificationAdapter()
        self.layoutlmv3_extractor = LayoutLMv3ExtractionAdapter()
        self.donut_extractor = DonutExtractionAdapter()
        self.dinov2_duplicate = DinoV2DuplicateAdapter()
        self.paddleocr_vl = PaddleOCRVLAdapter()
        self.local_forensics = LocalForensicsAdapter()
        self.segformer_forensics = SegFormerTamperAdapter()
        self.isolation_forest = IsolationForestRiskAdapter()
        self.xgboost_risk = XGBoostRiskAdapter()
        self.contract_extractor = DebertaContractClauseAdapter()
        self.retrieval_index = LocalHybridRetrievalIndex()
        self.query_router = VisionQueryRouter()
        self.generator_router = GroundedGeneratorRouter()
        self.persistence = vision_persistence
        self.documents: dict[str, VisionDocumentRecord] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.review_tasks: dict[str, VisionReviewTask] = {}
        self._pending_content: dict[str, tuple[str, bytes, dict[str, Any]]] = {}
        self._idempotency_index: dict[str, str] = {}
        self._hash_index: dict[str, str] = {}
        self._semantic_index: dict[str, str] = {}
        self._document_tenant_index: dict[str, str] = {}
        self._semantic_label_embedding_cache: dict[str, dict[str, list[float]]] = {}
        self.questions: dict[str, dict[str, Any]] = {}
        self.question_feedback: dict[str, list[dict[str, Any]]] = {}
        self.conversations: dict[str, VisionConversationState] = {}
        self.finance_narratives: dict[str, VisionFinanceNarrative] = {}
        self.reports: dict[str, VisionReportRecord] = {}
        self.model_overrides: dict[str, dict[str, Any]] = {}
        self.rule_overrides: dict[str, dict[str, Any]] = {}
        self.system_audit_events: list[dict[str, Any]] = []
        self._artifact_lock = threading.RLock()
        self._state_root = Path(settings.upload_dir).parent / "vision_state"
        self.feedback_dataset = ReviewerFeedbackDataset(self._state_root / "active_learning")
        self.confidence_policy = self._load_confidence_policy()
        self._ensure_state_dirs()
        self._load_admin_overrides()
        self._load_indexes()


vision_ai_service = VisionAIService()
