from __future__ import annotations

import importlib.util
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models.vision import OCRWord, VisionBoundingBox, VisionDocumentRecord
from app.services.vision.adapters import AdapterStatus, OCRAdapter


class TesseractOCRAdapter(OCRAdapter):
    adapter_id = "tesseract-local-ocr"
    model_version = "Tesseract-5"

    def status(self) -> AdapterStatus:
        package_installed = importlib.util.find_spec("pytesseract") is not None
        executable = shutil.which(settings.tesseract_command)
        active = package_installed and executable is not None
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else "unavailable",
            reason=(
                f"Using {executable}."
                if active
                else "Install pytesseract and the Tesseract executable for image OCR."
            ),
            enabled=True,
            installed=active,
            model_version=self.model_version,
            metadata={"languages": settings.tesseract_languages},
        )

    def extract_words(
        self,
        document: VisionDocumentRecord,
        native_words: list[dict[str, Any]] | None = None,
    ) -> list[OCRWord]:
        native_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in native_words or []:
            native_by_page[int(item.get("page", 1))].append(item)

        words: list[OCRWord] = []
        reading_order = 1
        for artifact in document.page_artifacts:
            native = native_by_page.get(artifact.page_number, [])
            if native:
                for item in native:
                    words.append(self._native_word(item, reading_order))
                    reading_order += 1
                continue
            page_words = self._tesseract_page(
                Path(artifact.processed_image_path),
                artifact.page_number,
                reading_order,
            )
            words.extend(page_words)
            reading_order += len(page_words)
        return words

    def _tesseract_page(
        self,
        image_path: Path,
        page_number: int,
        reading_order_start: int,
    ) -> list[OCRWord]:
        self.ensure_ready()
        import pytesseract
        from PIL import Image
        from pytesseract import Output

        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_command
        data = pytesseract.image_to_data(
            Image.open(image_path),
            lang=settings.tesseract_languages,
            config="--oem 3 --psm 3",
            output_type=Output.DICT,
        )
        words: list[OCRWord] = []
        for index, text in enumerate(data.get("text", [])):
            text = str(text).strip()
            if not text:
                continue
            try:
                raw_confidence = float(data["conf"][index])
            except (KeyError, TypeError, ValueError):
                raw_confidence = -1
            confidence = max(0.0, min(1.0, raw_confidence / 100.0))
            x1 = float(data["left"][index])
            y1 = float(data["top"][index])
            x2 = x1 + float(data["width"][index])
            y2 = y1 + float(data["height"][index])
            block_number = int(data.get("block_num", [1] * len(data["text"]))[index])
            line_number = int(data.get("line_num", [1] * len(data["text"]))[index])
            words.append(
                OCRWord(
                    text=text,
                    confidence=confidence,
                    polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    bounding_box=VisionBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    page=page_number,
                    line_id=f"p{page_number}-line-{line_number}",
                    block_id=f"p{page_number}-block-{block_number}",
                    reading_order=reading_order_start + len(words),
                    review_required=confidence < settings.ocr_low_confidence_threshold,
                    model_name="Tesseract",
                    model_version=self._tesseract_version(pytesseract),
                )
            )
        return words

    def _native_word(self, item: dict[str, Any], reading_order: int) -> OCRWord:
        x1 = float(item["x1"])
        y1 = float(item["y1"])
        x2 = float(item["x2"])
        y2 = float(item["y2"])
        confidence = float(item.get("confidence", 1.0))
        return OCRWord(
            text=str(item["text"]),
            confidence=confidence,
            polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            bounding_box=VisionBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            page=int(item.get("page", 1)),
            line_id=str(item.get("line_id", "line-1")),
            block_id=str(item.get("block_id", "block-1")),
            reading_order=reading_order,
            review_required=confidence < settings.ocr_low_confidence_threshold,
            model_name=str(item.get("model_name", "native-text-parser")),
            model_version=str(item.get("model_version", "v1")),
        )

    def _tesseract_version(self, pytesseract: Any) -> str:
        try:
            return str(pytesseract.get_tesseract_version()).splitlines()[0]
        except Exception:
            return self.model_version


class PaddleOCRV5Adapter(OCRAdapter):
    adapter_id = "paddleocr-pp-ocrv5"
    model_version = "PP-OCRv5"

    def __init__(self) -> None:
        self._pipeline: Any | None = None

    def status(self) -> AdapterStatus:
        installed = importlib.util.find_spec("paddleocr") is not None
        enabled = settings.enable_heavy_models and settings.enable_paddleocr
        active = installed and enabled
        if not enabled:
            reason = "Set ENABLE_HEAVY_MODELS=true and ENABLE_PADDLEOCR=true to activate."
        elif not installed:
            reason = "Install a compatible PaddlePaddle runtime and paddleocr package."
        else:
            reason = "PaddleOCR is enabled; model assets are validated on first load."
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=reason,
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={"device": settings.paddleocr_device, "profile": settings.model_profile},
        )

    def extract_words(
        self,
        document: VisionDocumentRecord,
        native_words: list[dict[str, Any]] | None = None,
    ) -> list[OCRWord]:
        self.ensure_ready()
        pipeline = self._load_pipeline()
        words: list[OCRWord] = []
        reading_order = 1
        for artifact in document.page_artifacts:
            predictions = pipeline.predict(str(artifact.processed_image_path))
            for prediction in predictions:
                payload = self._prediction_payload(prediction)
                texts = payload.get("rec_texts", [])
                scores = payload.get("rec_scores", [])
                polygons = payload.get("rec_polys") or payload.get("dt_polys") or []
                for index, text in enumerate(texts):
                    polygon = polygons[index] if index < len(polygons) else []
                    if len(polygon) < 4:
                        continue
                    points = [[float(point[0]), float(point[1])] for point in polygon]
                    xs = [point[0] for point in points]
                    ys = [point[1] for point in points]
                    confidence = float(scores[index]) if index < len(scores) else 0.0
                    words.append(
                        OCRWord(
                            text=str(text),
                            confidence=confidence,
                            polygon=points,
                            bounding_box=VisionBoundingBox(
                                x1=min(xs),
                                y1=min(ys),
                                x2=max(xs),
                                y2=max(ys),
                            ),
                            page=artifact.page_number,
                            line_id=f"p{artifact.page_number}-line-{index + 1}",
                            block_id=f"p{artifact.page_number}-block-1",
                            reading_order=reading_order,
                            review_required=confidence < settings.ocr_low_confidence_threshold,
                            model_name="PP-OCRv5",
                            model_version=self.model_version,
                        )
                    )
                    reading_order += 1
        return words

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PaddleOCR

            self._pipeline = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det"
                if settings.model_profile.upper() != "LITE"
                else "PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_server_rec"
                if settings.model_profile.upper() != "LITE"
                else "PP-OCRv5_mobile_rec",
                device=settings.paddleocr_device,
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_textline_orientation=True,
            )
        return self._pipeline

    def _prediction_payload(self, prediction: Any) -> dict[str, Any]:
        value = getattr(prediction, "json", prediction)
        if callable(value):
            value = value()
        if isinstance(value, str):
            import json

            value = json.loads(value)
        if isinstance(value, dict) and isinstance(value.get("res"), dict):
            return value["res"]
        return value if isinstance(value, dict) else {}


# Backward-compatible import name without placeholder behavior.
PendingPaddleOCRAdapter = PaddleOCRV5Adapter
