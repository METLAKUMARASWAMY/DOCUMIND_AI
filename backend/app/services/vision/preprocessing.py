from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps, ImageSequence

from app.core.config import settings
from app.models.vision import PageQualityReport, VisionBoundingBox, VisionPageArtifact


@dataclass
class PreprocessingResult:
    artifacts: list[VisionPageArtifact] = field(default_factory=list)
    quality: list[PageQualityReport] = field(default_factory=list)
    native_words: list[dict[str, Any]] = field(default_factory=list)
    page_text: dict[int, str] = field(default_factory=dict)


class DocumentPreprocessor:
    """Renders immutable source pages and creates measured, reversible derivatives."""

    image_extensions: ClassVar[set[str]] = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    text_extensions: ClassVar[set[str]] = {".txt", ".md", ".csv", ".eml"}

    def __init__(self, artifact_root: str | Path | None = None) -> None:
        self.artifact_root = Path(artifact_root or settings.vision_artifact_dir)

    def process(
        self,
        document_id: str,
        filename: str,
        content: bytes,
        parsed_text: str = "",
    ) -> PreprocessingResult:
        extension = Path(filename).suffix.lower()
        document_root = self.artifact_root / document_id
        original_root = document_root / "pages" / "original"
        enhanced_root = document_root / "pages" / "enhanced"
        grayscale_root = document_root / "pages" / "grayscale"
        binary_root = document_root / "pages" / "binary"
        original_root.mkdir(parents=True, exist_ok=True)
        enhanced_root.mkdir(parents=True, exist_ok=True)
        grayscale_root.mkdir(parents=True, exist_ok=True)
        binary_root.mkdir(parents=True, exist_ok=True)

        if extension == ".pdf":
            pages = self._render_pdf(content)
        elif extension in self.image_extensions:
            pages = self._load_images(content)
        else:
            pages = self._render_text(parsed_text or self._decode_text(content))

        result = PreprocessingResult()
        for index, page in enumerate(pages, start=1):
            image = ImageOps.exif_transpose(page["image"]).convert("RGB")
            original_path = original_root / f"page-{index:04d}.png"
            enhanced_path = enhanced_root / f"page-{index:04d}.png"
            grayscale_path = grayscale_root / f"page-{index:04d}.png"
            binary_path = binary_root / f"page-{index:04d}.png"
            image.save(original_path, format="PNG")

            enhanced, grayscale, binary, report, crop_box = self._preprocess_page(
                image,
                page_number=index,
                dpi=int(page.get("dpi") or settings.pdf_render_dpi),
                preserve_geometry=True,
            )
            enhanced.save(enhanced_path, format="PNG")
            grayscale.save(grayscale_path, format="PNG")
            binary.save(binary_path, format="PNG")
            report.processed_image_path = str(binary_path)
            result.quality.append(report)
            result.artifacts.append(
                VisionPageArtifact(
                    page_number=index,
                    original_image_path=str(original_path),
                    processed_image_path=str(binary_path),
                    enhanced_image_path=str(enhanced_path),
                    grayscale_image_path=str(grayscale_path),
                    binary_image_path=str(binary_path),
                    width=binary.width,
                    height=binary.height,
                    dpi=int(page.get("dpi") or settings.pdf_render_dpi),
                    source_kind=str(page.get("source_kind", "image")),
                    native_text_available=bool(page.get("native_words")),
                    crop_box=crop_box,
                )
            )
            result.native_words.extend(page.get("native_words", []))
            result.page_text[index] = str(page.get("text", ""))
        return result

    def ensure_variant(
        self,
        artifact: VisionPageArtifact,
        variant: str,
    ) -> Path:
        configured = {
            "original": artifact.original_image_path,
            "enhanced": artifact.enhanced_image_path,
            "grayscale": artifact.grayscale_image_path,
            "binary": artifact.binary_image_path or artifact.processed_image_path,
            "processed": artifact.processed_image_path,
        }.get(variant)
        if configured and Path(configured).is_file():
            return Path(configured)
        if variant not in {"enhanced", "grayscale", "binary"}:
            raise ValueError("Unsupported page-image variant")
        source_path = Path(artifact.original_image_path)
        if not source_path.is_file():
            raise FileNotFoundError("Original page image is not available")
        target = source_path.parent.parent / variant / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Image.open(source_path).convert("RGB")
        if variant == "enhanced":
            image = ImageOps.autocontrast(source)
            image = ImageEnhance.Contrast(image).enhance(1.08)
            image = ImageEnhance.Sharpness(image).enhance(1.12)
        elif variant == "grayscale":
            image = ImageOps.grayscale(source).convert("RGB")
        else:
            gray = np.asarray(ImageOps.grayscale(source))
            thresholded = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
            image = Image.fromarray(thresholded).convert("RGB")
        image.save(target, format="PNG")
        return target

    def _render_pdf(self, content: bytes) -> list[dict[str, Any]]:
        import pymupdf

        document = pymupdf.open(stream=content, filetype="pdf")
        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported")
        pages: list[dict[str, Any]] = []
        zoom = settings.pdf_render_dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False, colorspace=pymupdf.csRGB)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            native_words = []
            for reading_order, word in enumerate(page.get_text("words", sort=True), start=1):
                x1, y1, x2, y2, text, block_id, line_id, _ = word[:8]
                native_words.append(
                    {
                        "text": str(text),
                        "confidence": 1.0,
                        "x1": float(x1 * zoom),
                        "y1": float(y1 * zoom),
                        "x2": float(x2 * zoom),
                        "y2": float(y2 * zoom),
                        "page": page_number,
                        "line_id": f"p{page_number}-line-{int(line_id) + 1}",
                        "block_id": f"p{page_number}-block-{int(block_id) + 1}",
                        "reading_order": reading_order,
                        "model_name": "PyMuPDF-native-text",
                        "model_version": getattr(pymupdf, "__version__", "unknown"),
                    }
                )
            pages.append(
                {
                    "image": image,
                    "dpi": settings.pdf_render_dpi,
                    "native_words": native_words,
                    "text": page.get_text("text", sort=True) or "",
                    "source_kind": "pdf_render",
                }
            )
        document.close()
        if not pages:
            raise ValueError("PDF contains no pages")
        return pages

    def _load_images(self, content: bytes) -> list[dict[str, Any]]:
        source = Image.open(BytesIO(content))
        dpi_value = source.info.get("dpi", (150, 150))
        dpi = int(dpi_value[0] if isinstance(dpi_value, tuple) else dpi_value or 150)
        return [
            {
                "image": frame.copy(),
                "dpi": max(1, dpi),
                "native_words": [],
                "text": "",
                "source_kind": "uploaded_image",
            }
            for frame in ImageSequence.Iterator(source)
        ]

    def _render_text(self, text: str) -> list[dict[str, Any]]:
        logical_pages = text.split("\f") or [""]
        pages: list[dict[str, Any]] = []
        for page_number, page_text in enumerate(logical_pages, start=1):
            image = Image.new("RGB", (1654, 2339), "white")
            draw = ImageDraw.Draw(image)
            font = self._font(28)
            x, y = 90, 90
            line_height = 40
            native_words: list[dict[str, Any]] = []
            reading_order = 1
            for line_index, raw_line in enumerate(page_text.splitlines() or [""]):
                line = raw_line.expandtabs(4)[:150]
                if y + line_height >= image.height - 80:
                    break
                draw.text((x, y), line, fill="black", font=font)
                cursor = x
                for token in re.findall(r"\S+", line):
                    token_box = draw.textbbox((cursor, y), token, font=font)
                    width = max(1, token_box[2] - token_box[0])
                    native_words.append(
                        {
                            "text": token,
                            "confidence": 1.0,
                            "x1": float(cursor),
                            "y1": float(y),
                            "x2": float(cursor + width),
                            "y2": float(y + line_height - 5),
                            "page": page_number,
                            "line_id": f"p{page_number}-line-{line_index + 1}",
                            "block_id": f"p{page_number}-block-1",
                            "reading_order": reading_order,
                            "model_name": "source-text-renderer",
                            "model_version": "v1",
                        }
                    )
                    reading_order += 1
                    cursor += width + int(draw.textlength(" ", font=font))
                y += line_height
            pages.append(
                {
                    "image": image,
                    "dpi": 150,
                    "native_words": native_words,
                    "text": page_text,
                    "source_kind": "rendered_text_source",
                }
            )
        return pages

    def _preprocess_page(
        self,
        image: Image.Image,
        *,
        page_number: int,
        dpi: int,
        preserve_geometry: bool,
    ) -> tuple[
        Image.Image,
        Image.Image,
        Image.Image,
        PageQualityReport,
        VisionBoundingBox | None,
    ]:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast_std = float(gray.std())
        block_mean = cv2.blur(gray, (max(9, gray.shape[1] // 20 | 1), max(9, gray.shape[0] // 20 | 1)))
        illumination_variation = float(block_mean.std())
        rotation = self._estimate_rotation(gray)
        crop_box = self._content_box(gray)

        working = rgb
        steps = ["original_preserved", "rotation_detected"]
        notes: list[str] = []
        if not preserve_geometry and 0.5 <= abs(rotation) <= 12:
            working = self._rotate(working, rotation)
            steps.append("deskewed")
        else:
            steps.append("deskew_not_required")

        working, perspective_applied = self._perspective_correct(working, preserve_geometry)
        steps.append("perspective_corrected" if perspective_applied else "perspective_not_required")
        denoised = cv2.fastNlMeansDenoisingColored(working, None, 5, 5, 7, 21)
        steps.append("denoised")
        enhanced_gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
        background = cv2.dilate(enhanced_gray, np.ones((7, 7), np.uint8))
        background = cv2.medianBlur(background, 21)
        shadow_removed = cv2.divide(enhanced_gray, background, scale=255)
        steps.append("shadow_removed")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(shadow_removed)
        steps.append("contrast_enhanced")
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        enhanced_color = cv2.cvtColor(
            cv2.merge((clahe.apply(lightness), channel_a, channel_b)),
            cv2.COLOR_LAB2RGB,
        )
        thresholded = cv2.adaptiveThreshold(
            contrast_enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        steps.append("adaptive_thresholded")
        enhanced = Image.fromarray(enhanced_color)
        grayscale = Image.fromarray(cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2RGB))
        binary = Image.fromarray(cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB))

        if crop_box:
            steps.append("page_crop_detected")
            notes.append("Crop recorded but not applied so every viewer mode shares coordinates.")

        sharpness = min(1.0, math.log1p(blur_variance) / math.log1p(1000.0))
        contrast_score = min(1.0, contrast_std / 64.0)
        shadow_risk = min(1.0, illumination_variation / 75.0)
        quality_score = max(
            0.0,
            min(1.0, 0.50 * sharpness + 0.35 * contrast_score + 0.15 * (1 - shadow_risk)),
        )
        report = PageQualityReport(
            page_number=page_number,
            width=binary.width,
            height=binary.height,
            dpi_estimate=dpi,
            rotation_degrees=round(rotation),
            blur_score=round(1 - sharpness, 4),
            contrast_score=round(contrast_score, 4),
            shadow_risk=round(shadow_risk, 4),
            image_quality_score=round(quality_score, 4),
            preprocessing_steps=steps,
            notes=notes,
        )
        return enhanced, grayscale, binary, report, crop_box

    def _estimate_rotation(self, gray: np.ndarray) -> float:
        inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coordinates = np.column_stack(np.where(inverted > 0))
        if coordinates.shape[0] < 50:
            return 0.0
        angle = float(cv2.minAreaRect(coordinates[:, ::-1].astype(np.float32))[-1])
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle -= 90
        return round(angle, 3)

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

    def _perspective_correct(
        self,
        image: np.ndarray,
        preserve_geometry: bool,
    ) -> tuple[np.ndarray, bool]:
        if preserve_geometry:
            return image, False
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        page_area = image.shape[0] * image.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) != 4 or cv2.contourArea(polygon) < page_area * 0.70:
                continue
            points = self._order_points(polygon.reshape(4, 2).astype(np.float32))
            width = int(max(np.linalg.norm(points[2] - points[3]), np.linalg.norm(points[1] - points[0])))
            height = int(max(np.linalg.norm(points[1] - points[2]), np.linalg.norm(points[0] - points[3])))
            if width < 100 or height < 100:
                break
            destination = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )
            transform = cv2.getPerspectiveTransform(points, destination)
            return cv2.warpPerspective(image, transform, (width, height)), True
        return image, False

    def _content_box(self, gray: np.ndarray) -> VisionBoundingBox | None:
        binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
        coordinates = cv2.findNonZero(binary)
        if coordinates is None:
            return None
        x, y, width, height = cv2.boundingRect(coordinates)
        return VisionBoundingBox(x1=x, y1=y, x2=x + width, y2=y + height)

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).reshape(-1)
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(differences)]
        ordered[3] = points[np.argmax(differences)]
        return ordered

    def _font(self, size: int) -> ImageFont.ImageFont:
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _decode_text(self, content: bytes) -> str:
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="ignore")
