from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.models.vision import (
    ReviewerStatus,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisualDetection,
)
from app.services.vision.adapters import AdapterStatus, VisualDetectionAdapter
from app.services.vision.checkpoint_validation import validate_project_file_checkpoint

ALLOWED_VISUAL_CLASSES = {
    "signature",
    "stamp",
    "company_logo",
    "checkbox_checked",
    "checkbox_unchecked",
    "handwritten_note",
    "approval_mark",
    "seal",
    "QR_code",
    "barcode",
    "altered_region_candidate",
}


class OpenCVVisualDetectionAdapter(VisualDetectionAdapter):
    adapter_id = "opencv-visual-evidence"
    model_version = "v1"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active",
            reason="OpenCV visual heuristics are available for evidence candidates.",
            enabled=True,
            installed=True,
            model_version=self.model_version,
            metadata={
                "limitations": "Detections identify visual candidates, not authenticity.",
            },
        )

    def detect(self, document: VisionDocumentRecord) -> list[VisualDetection]:
        detections: list[VisualDetection] = []
        for artifact in document.page_artifacts:
            image = cv2.imread(artifact.original_image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            page_detections = [
                *self._detect_checkboxes(image, artifact.page_number),
                *self._detect_qr(image, artifact.page_number),
                *self._detect_coloured_seals(image, artifact.page_number),
                *self._detect_text_anchored_regions(document, artifact.page_number),
            ]
            for detection in self._deduplicate(page_detections):
                detection.cropped_evidence_path = self._save_crop(
                    document.document_id,
                    detection,
                    image,
                )
                detections.append(detection)
        return detections

    def _detect_checkboxes(self, image: np.ndarray, page_number: int) -> list[VisualDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[VisualDetection] = []
        page_scale = max(image.shape[0], image.shape[1]) / 1600.0
        minimum = max(8, int(12 * page_scale))
        maximum = max(36, int(70 * page_scale))
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if not (minimum <= width <= maximum and minimum <= height <= maximum):
                continue
            aspect = width / max(height, 1)
            if not 0.78 <= aspect <= 1.22:
                continue
            area = cv2.contourArea(contour)
            rectangularity = area / max(width * height, 1)
            if not 0.35 <= rectangularity <= 0.98:
                continue
            inner = binary[
                y + max(2, height // 5) : y + height - max(2, height // 5),
                x + max(2, width // 5) : x + width - max(2, width // 5),
            ]
            fill_ratio = float(np.count_nonzero(inner)) / max(inner.size, 1)
            detection_class = "checkbox_checked" if fill_ratio > 0.22 else "checkbox_unchecked"
            detections.append(
                self._detection(
                    detection_class,
                    page_number,
                    x,
                    y,
                    x + width,
                    y + height,
                    confidence=min(0.92, 0.55 + abs(1 - aspect) * -0.2 + rectangularity * 0.35),
                    notes=["Candidate found from square contour and interior ink density."],
                )
            )
            if len(detections) >= 30:
                break
        return detections

    def _detect_qr(self, image: np.ndarray, page_number: int) -> list[VisualDetection]:
        detector = cv2.QRCodeDetector()
        detections: list[VisualDetection] = []
        try:
            found, _, points, _ = detector.detectAndDecodeMulti(image)
            if found and points is not None:
                for polygon in points:
                    xs = polygon[:, 0]
                    ys = polygon[:, 1]
                    detections.append(
                        self._detection(
                            "QR_code",
                            page_number,
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max()),
                            float(ys.max()),
                            confidence=0.95,
                            notes=["Decoded or detected with OpenCV QRCodeDetector."],
                        )
                    )
        except cv2.error:
            pass
        return detections

    def _detect_coloured_seals(self, image: np.ndarray, page_number: int) -> list[VisualDetection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, np.array([0, 65, 40]), np.array([12, 255, 255]))
        red |= cv2.inRange(hsv, np.array([165, 65, 40]), np.array([180, 255, 255]))
        blue = cv2.inRange(hsv, np.array([90, 55, 35]), np.array([140, 255, 255]))
        mask = cv2.morphologyEx(red | blue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[VisualDetection] = []
        image_area = image.shape[0] * image.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if not image_area * 0.0002 <= area <= image_area * 0.08:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
            x, y, width, height = cv2.boundingRect(contour)
            detections.append(
                self._detection(
                    "stamp" if circularity > 0.35 else "approval_mark",
                    page_number,
                    x,
                    y,
                    x + width,
                    y + height,
                    confidence=min(0.88, 0.50 + float(circularity) * 0.35),
                    notes=["Candidate found from saturated red/blue document ink."],
                )
            )
        return detections[:12]

    def _detect_text_anchored_regions(
        self,
        document: VisionDocumentRecord,
        page_number: int,
    ) -> list[VisualDetection]:
        detections: list[VisualDetection] = []
        anchors = {
            "signature": ("signature", "signed", "signatory"),
            "stamp": ("stamp", "seal"),
            "approval_mark": ("approved", "approval"),
        }
        page_words = [word for word in document.ocr_words if word.page == page_number]
        for detection_class, terms in anchors.items():
            matching = [
                word for word in page_words if any(term in word.text.lower() for term in terms)
            ]
            if not matching:
                continue
            x1 = min(word.bounding_box.x1 for word in matching)
            y1 = min(word.bounding_box.y1 for word in matching)
            x2 = max(word.bounding_box.x2 for word in matching)
            y2 = max(word.bounding_box.y2 for word in matching)
            width = max(160.0, (x2 - x1) * 3.0)
            height = max(55.0, (y2 - y1) * 2.0)
            detections.append(
                self._detection(
                    detection_class,
                    page_number,
                    x1,
                    y1,
                    x1 + width,
                    y1 + height,
                    confidence=0.58,
                    notes=[
                        "Text-anchored visual candidate; presence and authenticity require reviewer confirmation."
                    ],
                )
            )
        return detections

    def _detection(
        self,
        detection_class: str,
        page_number: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        confidence: float,
        notes: list[str],
    ) -> VisualDetection:
        return VisualDetection(
            detection_class=detection_class,
            page_number=page_number,
            bounding_box=VisionBoundingBox(
                x1=max(0.0, x1),
                y1=max(0.0, y1),
                x2=max(x1 + 1.0, x2),
                y2=max(y1 + 1.0, y2),
            ),
            confidence=max(0.0, min(1.0, confidence)),
            model_name=self.adapter_id,
            model_version=self.model_version,
            reviewer_status=ReviewerStatus.unverified,
            notes=[
                *notes,
                "Detection does not establish signature, stamp, or document authenticity.",
            ],
        )

    def _save_crop(
        self,
        document_id: str,
        detection: VisualDetection,
        image: np.ndarray,
    ) -> str | None:
        height, width = image.shape[:2]
        box = detection.bounding_box
        x1 = max(0, min(width - 1, int(box.x1)))
        y1 = max(0, min(height - 1, int(box.y1)))
        x2 = max(x1 + 1, min(width, int(box.x2)))
        y2 = max(y1 + 1, min(height, int(box.y2)))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        target_root = Path(settings.vision_artifact_dir) / document_id / "evidence"
        target_root.mkdir(parents=True, exist_ok=True)
        target = target_root / f"{detection.detection_id}.png"
        cv2.imwrite(str(target), crop)
        return str(target)

    def _deduplicate(self, detections: list[VisualDetection]) -> list[VisualDetection]:
        retained: list[VisualDetection] = []
        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(
                candidate.detection_class == existing.detection_class
                and self._iou(candidate.bounding_box, existing.bounding_box) > 0.55
                for existing in retained
            ):
                continue
            retained.append(candidate)
        return retained

    def _iou(self, left: VisionBoundingBox, right: VisionBoundingBox) -> float:
        x1 = max(left.x1, right.x1)
        y1 = max(left.y1, right.y1)
        x2 = min(left.x2, right.x2)
        y2 = min(left.y2, right.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
        right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
        return intersection / max(left_area + right_area - intersection, 1.0)


class Yolo26VisualDetectionAdapter(VisualDetectionAdapter):
    adapter_id = "yolo26-visual-detection"
    model_version = "YOLO26-project-checkpoint"

    def __init__(self) -> None:
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = importlib.util.find_spec("ultralytics") is not None
        checkpoint = (
            Path(settings.yolo26_checkpoint).expanduser() if settings.yolo26_checkpoint else None
        )
        checkpoint_validation = validate_project_file_checkpoint(
            checkpoint,
            expected_tasks={"visual_detection", "yolo26_visual_detection"},
        )
        checkpoint_ready = checkpoint_validation.ready
        enabled = settings.enable_heavy_models and settings.enable_yolo26
        active = enabled and installed and checkpoint_ready
        reason = (
            "YOLO26 trained project checkpoint is ready for smoke verification."
            if active
            else checkpoint_validation.reason
        )
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=reason,
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={
                "checkpoint": str(checkpoint) if checkpoint else None,
                "checkpoint_validation": checkpoint_validation.metadata,
            },
        )

    def detect(self, document: VisionDocumentRecord) -> list[VisualDetection]:
        self.ensure_ready()
        model = self._load_model()
        detections: list[VisualDetection] = []
        for artifact in document.page_artifacts:
            results = model.predict(
                source=artifact.processed_image_path,
                conf=0.25,
                verbose=False,
            )
            for result in results:
                names = result.names
                for box in result.boxes:
                    class_name = str(names[int(box.cls.item())])
                    if class_name not in ALLOWED_VISUAL_CLASSES:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    detections.append(
                        VisualDetection(
                            detection_class=class_name,
                            page_number=artifact.page_number,
                            bounding_box=VisionBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                            confidence=float(box.conf.item()),
                            model_name="YOLO26",
                            model_version=self.model_version,
                            notes=["Detection does not establish authenticity."],
                        )
                    )
        return detections

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(Path(settings.yolo26_checkpoint).expanduser()))
        return self._model
