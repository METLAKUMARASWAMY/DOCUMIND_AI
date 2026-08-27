from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import settings
from app.models.vision import VisionDocumentRecord
from app.services.vision.adapters import AdapterStatus, VisionAdapter


class DinoV2DuplicateAdapter(VisionAdapter):
    adapter_id = "dinov2-document-embedding"
    stage = "visual_duplicate_embedding"
    model_version = "DINOv2-project-checkpoint"

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = all(
            importlib.util.find_spec(package) is not None for package in ("transformers", "torch")
        )
        checkpoint = (
            Path(settings.dinov2_checkpoint).expanduser() if settings.dinov2_checkpoint else None
        )
        checkpoint_ready = bool(
            checkpoint and checkpoint.is_dir() and (checkpoint / "config.json").is_file()
        )
        enabled = settings.enable_heavy_models and settings.enable_dinov2
        active = enabled and installed and checkpoint_ready
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "A local DINOv2 checkpoint is ready for document-image embeddings."
                if active
                else "Provide DINOV2_CHECKPOINT and install torch/transformers to enable DINOv2."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={"checkpoint": str(checkpoint) if checkpoint else None},
        )

    def embed(self, document: VisionDocumentRecord) -> list[float]:
        self.ensure_ready()
        import torch

        page_images = []
        for artifact in sorted(document.page_artifacts, key=lambda item: item.page_number):
            path = Path(artifact.original_image_path)
            if not path.is_file():
                continue
            with Image.open(path) as image:
                page_images.append(image.convert("RGB").copy())
        if not page_images:
            raise ValueError("No rendered page image is available for DINOv2 embedding")
        processor, model = self._load()
        encoded = processor(images=page_images, return_tensors="pt")
        with torch.inference_mode():
            output = model(**encoded)
        vectors = output.last_hidden_state[:, 0, :].float()
        document_vector = vectors.mean(dim=0)
        norm = torch.linalg.vector_norm(document_vector)
        if float(norm.item()) <= 0:
            raise ValueError("DINOv2 returned a zero-length document embedding")
        document_vector = document_vector / norm
        return [float(value) for value in document_vector.cpu().tolist()]

    def _load(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            from transformers import AutoImageProcessor, AutoModel

            checkpoint = str(Path(settings.dinov2_checkpoint or "").expanduser())
            self._processor = AutoImageProcessor.from_pretrained(
                checkpoint,
                local_files_only=True,
            )
            self._model = AutoModel.from_pretrained(checkpoint, local_files_only=True)
            self._model.eval()
        return self._processor, self._model


def document_dhash(document: VisionDocumentRecord, hash_size: int = 16) -> str | None:
    page_hashes: list[str] = []
    for artifact in document.page_artifacts:
        path = Path(artifact.original_image_path)
        if not path.is_file():
            continue
        with Image.open(path) as image:
            resized = image.convert("L").resize(
                (hash_size + 1, hash_size), Image.Resampling.LANCZOS
            )
            pixels = list(resized.getdata())
        value = 0
        for row in range(hash_size):
            offset = row * (hash_size + 1)
            for column in range(hash_size):
                value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
        page_hashes.append(f"{value:0{hash_size * hash_size // 4}x}")
    return ":".join(page_hashes) if page_hashes else None


def perceptual_hash_bit_ratio(page_hash: str) -> float | None:
    """Fraction of set bits in one page's dHash, or None if it cannot be read."""
    try:
        value = int(page_hash, 16)
    except (TypeError, ValueError):
        return None
    bits = len(page_hash) * 4
    if bits <= 0:
        return None
    return value.bit_count() / bits


def perceptual_hash_is_discriminative(page_hash: str) -> bool:
    """Whether a page's dHash carries enough information to compare.

    Each bit records "this pixel is darker than the next one". On a page that is
    mostly blank almost every comparison is a tie, so almost every bit is 0 —
    and two such pages then agree on ~98% of their bits while sharing no
    content whatsoever. Measured here: pages with real content score 0.24-0.40,
    a blank page scores 0.00 and a three-line page scores 0.008.

    A hash outside the usable band is not evidence of similarity or of
    difference. It is an absent measurement, and saying so is the honest
    result.
    """
    ratio = perceptual_hash_bit_ratio(page_hash)
    if ratio is None:
        return False
    floor = settings.perceptual_hash_min_bit_ratio
    return floor <= ratio <= (1.0 - floor)


def perceptual_similarity(left: str | None, right: str | None) -> float | None:
    if not left or not right:
        return None
    left_pages = left.split(":")
    right_pages = right.split(":")
    common = min(len(left_pages), len(right_pages))
    if common == 0:
        return None
    similarities: list[float] = []
    for index in range(common):
        left_page = left_pages[index]
        right_page = right_pages[index]
        # A comparison is only meaningful when both hashes carry information.
        if not perceptual_hash_is_discriminative(
            left_page
        ) or not perceptual_hash_is_discriminative(right_page):
            continue
        left_value = int(left_page, 16)
        right_value = int(right_page, 16)
        bits = max(len(left_page), len(right_page)) * 4
        distance = (left_value ^ right_value).bit_count()
        similarities.append(1.0 - distance / max(bits, 1))
    if not similarities:
        return None
    page_count_penalty = common / max(len(left_pages), len(right_pages))
    return max(0.0, min(1.0, sum(similarities) / len(similarities) * page_count_penalty))


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm <= 0 or right_norm <= 0:
        return None
    raw = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, raw))
