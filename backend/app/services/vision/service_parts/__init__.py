"""The responsibility parts of VisionAIService.

service.py was a single 9,500-line class (D-16). Its methods now live in
cohesive mixins here and are composed back into one class, so the public
API, the call graph and `self` state are all exactly as they were.
"""

from __future__ import annotations

from app.services.vision.service_parts.administration import AdminMixin
from app.services.vision.service_parts.analytics import AnalyticsMixin
from app.services.vision.service_parts.answering import AnsweringMixin
from app.services.vision.service_parts.arithmetic import ArithmeticMixin
from app.services.vision.service_parts.confidence import ConfidenceMixin
from app.services.vision.service_parts.demo import DemoSeedMixin
from app.services.vision.service_parts.documents import DocumentMixin
from app.services.vision.service_parts.extraction import ExtractionMixin
from app.services.vision.service_parts.graph import GraphMixin
from app.services.vision.service_parts.ingestion import IngestionMixin
from app.services.vision.service_parts.normalization import NormalizationMixin
from app.services.vision.service_parts.provenance import ProvenanceMixin
from app.services.vision.service_parts.reporting import ReportingMixin
from app.services.vision.service_parts.retrieval import RetrievalMixin
from app.services.vision.service_parts.review import ReviewMixin
from app.services.vision.service_parts.risk import RiskMixin
from app.services.vision.service_parts.state import StateMixin

__all__ = [
    "AdminMixin",
    "AnalyticsMixin",
    "AnsweringMixin",
    "ArithmeticMixin",
    "ConfidenceMixin",
    "DemoSeedMixin",
    "DocumentMixin",
    "ExtractionMixin",
    "GraphMixin",
    "IngestionMixin",
    "NormalizationMixin",
    "ProvenanceMixin",
    "ReportingMixin",
    "RetrievalMixin",
    "ReviewMixin",
    "RiskMixin",
    "StateMixin",
]
