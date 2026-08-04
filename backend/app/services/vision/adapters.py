from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.models.vision import (
    ExtractedVisionField,
    LayoutRegion,
    OCRWord,
    TableResult,
    VisionAskResponse,
    VisionDocumentRecord,
    VisionEvidenceChunk,
    VisionGraphProjection,
    VisualDetection,
)


class AdapterUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdapterStatus:
    adapter_id: str
    stage: str
    status: str
    reason: str
    enabled: bool = False
    installed: bool = False
    model_version: str = "target"
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionAdapter(ABC):
    adapter_id: str
    stage: str
    model_version: str = "target"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="adapter_pending",
            reason="Adapter contract exists, but runtime implementation is not wired.",
            model_version=self.model_version,
        )

    def ensure_ready(self) -> None:
        current = self.status()
        if current.status != "active":
            raise AdapterUnavailableError(f"{current.adapter_id} is {current.status}: {current.reason}")


class OCRAdapter(VisionAdapter):
    stage = "ocr"

    @abstractmethod
    def extract_words(
        self,
        document: VisionDocumentRecord,
        native_words: list[dict[str, Any]] | None = None,
    ) -> list[OCRWord]:
        raise NotImplementedError


class LayoutAdapter(VisionAdapter):
    stage = "layout"

    @abstractmethod
    def analyze_layout(self, document: VisionDocumentRecord) -> tuple[list[LayoutRegion], list[TableResult]]:
        raise NotImplementedError


class ExtractionAdapter(VisionAdapter):
    stage = "classification_extraction"

    @abstractmethod
    def extract_fields(self, document: VisionDocumentRecord) -> list[ExtractedVisionField]:
        raise NotImplementedError


class OCRFreeAdapter(VisionAdapter):
    stage = "ocr_free_baseline"

    @abstractmethod
    def extract_json(self, document: VisionDocumentRecord) -> dict[str, Any]:
        raise NotImplementedError


class VisualDetectionAdapter(VisionAdapter):
    stage = "visual_detection"

    @abstractmethod
    def detect(self, document: VisionDocumentRecord) -> list[VisualDetection]:
        raise NotImplementedError


class VectorSearchAdapter(VisionAdapter):
    stage = "retrieval"

    @abstractmethod
    def index_chunks(self, chunks: list[VisionEvidenceChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[VisionEvidenceChunk]:
        raise NotImplementedError


class GraphAdapter(VisionAdapter):
    stage = "graph"

    @abstractmethod
    def upsert_graph(self, graph: VisionGraphProjection) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, question: str) -> VisionGraphProjection:
        raise NotImplementedError


class GeneratorAdapter(VisionAdapter):
    stage = "generator"

    @abstractmethod
    def answer(self, question: str, evidence: list[VisionEvidenceChunk]) -> VisionAskResponse:
        raise NotImplementedError

