from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops

from app.core.config import settings
from app.models.vision import (
    TamperAnalysisResult,
    TamperRegion,
    VisionBoundingBox,
    VisionDocumentRecord,
)
from app.services.vision.adapters import AdapterStatus, VisionAdapter


class LocalForensicsAdapter(VisionAdapter):
    adapter_id = "ela-noise-forensics"
    stage = "tamper_screening"
    model_version = "v1"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active",
            reason="Measured ELA, noise-residual, and high-pass screening is available.",
            enabled=True,
            installed=True,
            model_version=self.model_version,
            metadata={"calibrated_probability": False},
        )

    def analyze(self, document: VisionDocumentRecord) -> list[TamperAnalysisResult]:
        results: list[TamperAnalysisResult] = []
        output_root = Path(settings.vision_artifact_dir) / document.document_id / "forensics"
        output_root.mkdir(parents=True, exist_ok=True)
        for artifact in document.page_artifacts:
            image = Image.open(artifact.original_image_path).convert("RGB")
            rgb = np.asarray(image)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            ela = self._ela(image)
            residual = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 1.2))
            high_pass = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 3.0))
            combined = np.mean(
                [self._robust_normalize(item) for item in (ela, residual, high_pass)],
                axis=0,
            ).astype(np.uint8)
            regions = self._regions(combined, rgb, output_root, artifact.page_number)
            top_count = max(1, int(combined.size * 0.01))
            top_values = np.partition(combined.reshape(-1), -top_count)[-top_count:]
            screening_score = float(top_values.mean() / 255.0)
            heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_TURBO)
            heatmap_path = output_root / f"page-{artifact.page_number:04d}-heatmap.png"
            cv2.imwrite(str(heatmap_path), heatmap)
            results.append(
                TamperAnalysisResult(
                    page_number=artifact.page_number,
                    screening_score=round(screening_score, 4),
                    tampering_probability=None,
                    calibrated=False,
                    heatmap_path=str(heatmap_path),
                    suspicious_regions=regions,
                    explanation=(
                        "Possible visual manipulation candidates were measured from compression, noise, "
                        "and high-frequency inconsistencies. This screening score is not a probability "
                        "and does not establish forgery."
                    ),
                    model_name=self.adapter_id,
                    model_version=self.model_version,
                )
            )
        return results

    def _ela(self, image: Image.Image) -> np.ndarray:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")
        difference = ImageChops.difference(image, recompressed)
        return cv2.cvtColor(np.asarray(difference), cv2.COLOR_RGB2GRAY)

    def _robust_normalize(self, image: np.ndarray) -> np.ndarray:
        low, high = np.percentile(image, (50.0, 99.5))
        if high <= low:
            return np.zeros_like(image, dtype=np.uint8)
        normalized = (image.astype(np.float32) - float(low)) * (255.0 / float(high - low))
        return np.clip(normalized, 0, 255).astype(np.uint8)

    def _regions(
        self,
        combined: np.ndarray,
        rgb: np.ndarray,
        output_root: Path,
        page_number: int,
    ) -> list[TamperRegion]:
        median = float(np.median(combined))
        mad = float(np.median(np.abs(combined.astype(np.float32) - median)))
        threshold = max(float(np.percentile(combined, 99.7)), median + 6.0 * max(mad, 1.0))
        binary = (combined >= threshold).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_area = combined.shape[0] * combined.shape[1]
        regions: list[TamperRegion] = []
        for index, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
            area = cv2.contourArea(contour)
            if not page_area * 0.00008 <= area <= page_area * 0.20:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            local_score = float(combined[y : y + height, x : x + width].mean() / 255.0)
            crop_path = output_root / f"page-{page_number:04d}-candidate-{index + 1:02d}.png"
            cv2.imwrite(str(crop_path), cv2.cvtColor(rgb[y : y + height, x : x + width], cv2.COLOR_RGB2BGR))
            regions.append(
                TamperRegion(
                    bounding_box=VisionBoundingBox(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + width),
                        y2=float(y + height),
                    ),
                    screening_score=round(local_score, 4),
                    cropped_evidence_path=str(crop_path),
                )
            )
            if len(regions) >= 12:
                break
        return regions


class SegFormerTamperAdapter(VisionAdapter):
    adapter_id = "segformer-document-forensics"
    stage = "tamper_screening"
    model_version = "project-checkpoint"

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = all(
            importlib.util.find_spec(package) is not None
            for package in ("transformers", "torch")
        )
        checkpoint = Path(settings.tamper_checkpoint).expanduser() if settings.tamper_checkpoint else None
        ready = bool(checkpoint and checkpoint.is_dir() and (checkpoint / "config.json").is_file())
        enabled = settings.enable_heavy_models and bool(settings.tamper_checkpoint)
        active = enabled and installed and ready
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "Fine-tuned document-forensics checkpoint is ready."
                if active
                else "Provide TAMPER_CHECKPOINT with a fine-tuned SegFormer model."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={"checkpoint": str(checkpoint) if checkpoint else None},
        )

    def analyze(self, document: VisionDocumentRecord) -> list[TamperAnalysisResult]:
        self.ensure_ready()
        import torch
        from torch.nn import functional

        processor, model = self._load()
        output_root = Path(settings.vision_artifact_dir) / document.document_id / "forensics-model"
        output_root.mkdir(parents=True, exist_ok=True)
        results: list[TamperAnalysisResult] = []
        for artifact in document.page_artifacts:
            image = Image.open(artifact.original_image_path).convert("RGB")
            encoded = processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**encoded).logits
                logits = functional.interpolate(
                    logits,
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                )
                probabilities = torch.softmax(logits, dim=1)[0]
            class_map = probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8)
            foreground = probabilities[1:].amax(dim=0).cpu().numpy() if probabilities.shape[0] > 1 else np.zeros_like(class_map)
            regions = self._model_regions(class_map, foreground, model.config.id2label)
            heatmap = cv2.applyColorMap(np.clip(foreground * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            heatmap_path = output_root / f"page-{artifact.page_number:04d}-heatmap.png"
            cv2.imwrite(str(heatmap_path), heatmap)
            probability = float(foreground.max()) if foreground.size else 0.0
            results.append(
                TamperAnalysisResult(
                    page_number=artifact.page_number,
                    screening_score=round(probability, 4),
                    tampering_probability=round(probability, 4),
                    calibrated=False,
                    heatmap_path=str(heatmap_path),
                    suspicious_regions=regions,
                    explanation=(
                        "Possible visual manipulation from the fine-tuned segmentation checkpoint; "
                        "the softmax score is uncalibrated and requires human verification."
                    ),
                    model_name=self.adapter_id,
                    model_version=self.model_version,
                )
            )
        return results

    def _load(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

            checkpoint = str(Path(settings.tamper_checkpoint).expanduser())
            self._processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
            self._model = AutoModelForSemanticSegmentation.from_pretrained(
                checkpoint,
                local_files_only=True,
            )
            self._model.eval()
        return self._processor, self._model

    def _model_regions(
        self,
        class_map: np.ndarray,
        foreground: np.ndarray,
        id2label: dict[int, str] | dict[str, str],
    ) -> list[TamperRegion]:
        regions: list[TamperRegion] = []
        for class_id in sorted(int(value) for value in np.unique(class_map) if int(value) != 0):
            mask = (class_map == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if width * height < 36:
                    continue
                score = float(foreground[y : y + height, x : x + width].mean())
                regions.append(
                    TamperRegion(
                        manipulation_class=str(id2label.get(class_id, id2label.get(str(class_id), "unknown"))),
                        bounding_box=VisionBoundingBox(
                            x1=float(x),
                            y1=float(y),
                            x2=float(x + width),
                            y2=float(y + height),
                        ),
                        screening_score=round(score, 4),
                    )
                )
        return regions
