from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.security import utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    OCRWord,
    ReviewerStatus,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidence,
)
from app.services.vision.adapters import AdapterStatus, ExtractionAdapter, OCRFreeAdapter


class LayoutLMv3ExtractionAdapter(ExtractionAdapter):
    adapter_id = "layoutlmv3-extraction"
    model_version = "LayoutLMv3-project-checkpoint"

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = all(
            importlib.util.find_spec(package) is not None
            for package in ("transformers", "torch")
        )
        checkpoint = (
            Path(settings.layoutlmv3_checkpoint).expanduser()
            if settings.layoutlmv3_checkpoint
            else None
        )
        checkpoint_ready = bool(
            checkpoint
            and checkpoint.is_dir()
            and (checkpoint / "config.json").is_file()
        )
        enabled = settings.enable_heavy_models and settings.enable_layoutlmv3
        active = enabled and installed and checkpoint_ready
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "Fine-tuned LayoutLMv3 token-classification checkpoint is ready."
                if active
                else "Provide a fine-tuned LAYOUTLMV3_CHECKPOINT and install torch/transformers."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={"checkpoint": str(checkpoint) if checkpoint else None},
        )

    def extract_fields(self, document: VisionDocumentRecord) -> list[ExtractedVisionField]:
        self.ensure_ready()
        processor, model = self._load()
        import torch
        from PIL import Image

        artifacts = {item.page_number: item for item in document.page_artifacts}
        all_fields: list[ExtractedVisionField] = []
        for page_number, artifact in artifacts.items():
            words = [word for word in document.ocr_words if word.page == page_number]
            if not words:
                continue
            image = Image.open(artifact.processed_image_path).convert("RGB")
            boxes = [self._normalise_box(word.bounding_box, artifact.width, artifact.height) for word in words]
            encoded = processor(
                image,
                [word.text for word in words],
                boxes=boxes,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            with torch.inference_mode():
                output = model(**encoded)
                probabilities = torch.softmax(output.logits, dim=-1)[0]
                labels = probabilities.argmax(dim=-1).tolist()
            word_ids = encoded.word_ids(batch_index=0)
            all_fields.extend(
                self._decode_fields(
                    document,
                    page_number,
                    words,
                    word_ids,
                    labels,
                    probabilities,
                    model.config.id2label,
                )
            )
        return self._deduplicate(all_fields)

    def _decode_fields(
        self,
        document: VisionDocumentRecord,
        page_number: int,
        words: list[OCRWord],
        word_ids: list[int | None],
        labels: list[int],
        probabilities: Any,
        id2label: dict[int, str] | dict[str, str],
    ) -> list[ExtractedVisionField]:
        predictions: dict[int, tuple[str, float]] = {}
        for token_index, word_index in enumerate(word_ids):
            if word_index is None or word_index >= len(words):
                continue
            label_id = labels[token_index]
            raw_label = str(id2label.get(label_id, id2label.get(str(label_id), "O")))
            label = raw_label.split("-", 1)[-1].strip().lower()
            if raw_label.upper() == "O" or label in {"o", "other"}:
                continue
            confidence = float(probabilities[token_index, label_id].item())
            current = predictions.get(word_index)
            if current is None or confidence > current[1]:
                predictions[word_index] = (label, confidence)

        grouped: list[tuple[str, list[tuple[OCRWord, float]]]] = []
        for word_index, word in enumerate(words):
            prediction = predictions.get(word_index)
            if not prediction:
                continue
            label, confidence = prediction
            if grouped and grouped[-1][0] == label:
                grouped[-1][1].append((word, confidence))
            else:
                grouped.append((label, [(word, confidence)]))

        fields: list[ExtractedVisionField] = []
        for label, labelled_words in grouped:
            value = " ".join(word.text for word, _ in labelled_words).strip()
            if not value:
                continue
            confidence = sum(score for _, score in labelled_words) / len(labelled_words)
            box = self._union([word.bounding_box for word, _ in labelled_words])
            source_text = value
            evidence = VisionEvidence(
                document_id=document.document_id,
                page_number=page_number,
                source_text=source_text,
                bounding_box=box,
                visual_region=f"page_{page_number}",
                model_confidence=confidence,
                model_name="LayoutLMv3",
                model_version=self.model_version,
                processing_timestamp=utc_now_iso(),
                reviewer_status=ReviewerStatus.unverified,
                metadata={"word_ids": [word.word_id for word, _ in labelled_words]},
            )
            fields.append(
                ExtractedVisionField(
                    name=label,
                    value=value,
                    normalized_value=value.strip().lower(),
                    confidence=confidence,
                    evidence=evidence,
                    model_name="LayoutLMv3",
                    model_version=self.model_version,
                    target_model="LayoutLMv3",
                    confidence_components={"layoutlmv3_probability": confidence},
                    calibrated=False,
                )
            )
        return fields

    def _load(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            from transformers import AutoModelForTokenClassification, AutoProcessor

            checkpoint = str(Path(settings.layoutlmv3_checkpoint).expanduser())
            self._processor = AutoProcessor.from_pretrained(
                checkpoint,
                apply_ocr=False,
                local_files_only=True,
            )
            self._model = AutoModelForTokenClassification.from_pretrained(
                checkpoint,
                local_files_only=True,
            )
            self._model.eval()
        return self._processor, self._model

    def _normalise_box(self, box: VisionBoundingBox, width: int, height: int) -> list[int]:
        return [
            max(0, min(1000, int(1000 * box.x1 / width))),
            max(0, min(1000, int(1000 * box.y1 / height))),
            max(0, min(1000, int(1000 * box.x2 / width))),
            max(0, min(1000, int(1000 * box.y2 / height))),
        ]

    def _union(self, boxes: list[VisionBoundingBox]) -> VisionBoundingBox:
        return VisionBoundingBox(
            x1=min(box.x1 for box in boxes),
            y1=min(box.y1 for box in boxes),
            x2=max(box.x2 for box in boxes),
            y2=max(box.y2 for box in boxes),
        )

    def _deduplicate(self, fields: list[ExtractedVisionField]) -> list[ExtractedVisionField]:
        retained: dict[tuple[str, str], ExtractedVisionField] = {}
        for field in fields:
            key = (field.name, field.normalized_value or field.value.lower())
            if key not in retained or field.confidence > retained[key].confidence:
                retained[key] = field
        return list(retained.values())


class LayoutLMv3ClassificationAdapter:
    adapter_id = "layoutlmv3-classification"
    model_version = "LayoutLMv3-project-classifier"

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = all(
            importlib.util.find_spec(package) is not None
            for package in ("transformers", "torch")
        )
        checkpoint = (
            Path(settings.layoutlmv3_classification_checkpoint).expanduser()
            if settings.layoutlmv3_classification_checkpoint
            else None
        )
        ready = bool(checkpoint and checkpoint.is_dir() and (checkpoint / "config.json").is_file())
        enabled = settings.enable_heavy_models and settings.enable_layoutlmv3
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage="document_classification",
            status="active" if enabled and installed and ready else (
                "disabled" if not enabled else "unavailable"
            ),
            reason=(
                "Fine-tuned LayoutLMv3 classification checkpoint is ready."
                if enabled and installed and ready
                else "Provide LAYOUTLMV3_CLASSIFICATION_CHECKPOINT for learned classification."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
        )

    def classify(self, document: VisionDocumentRecord) -> tuple[VisionDocumentType, float]:
        if self.status().status != "active":
            raise RuntimeError(self.status().reason)
        import torch
        from PIL import Image

        processor, model = self._load()
        artifact = document.page_artifacts[0]
        image = Image.open(artifact.processed_image_path).convert("RGB")
        encoded = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
        label_id = int(probabilities.argmax().item())
        raw_label = str(model.config.id2label.get(label_id, "unknown")).lower()
        label = raw_label.replace("-", "_").replace(" ", "_")
        try:
            document_type = VisionDocumentType(label)
        except ValueError:
            document_type = VisionDocumentType.unknown
        return document_type, float(probabilities[label_id].item())

    def _load(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoProcessor

            checkpoint = str(Path(settings.layoutlmv3_classification_checkpoint).expanduser())
            self._processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint,
                local_files_only=True,
            )
            self._model.eval()
        return self._processor, self._model


class DonutExtractionAdapter(OCRFreeAdapter):
    adapter_id = "donut-ocr-free"
    model_version = "Donut-project-checkpoint"

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = all(
            importlib.util.find_spec(package) is not None
            for package in ("transformers", "torch")
        )
        checkpoint = Path(settings.donut_checkpoint).expanduser() if settings.donut_checkpoint else None
        ready = bool(checkpoint and checkpoint.is_dir() and (checkpoint / "config.json").is_file())
        enabled = settings.enable_heavy_models and settings.enable_donut
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if enabled and installed and ready else (
                "disabled" if not enabled else "unavailable"
            ),
            reason=(
                "Fine-tuned Donut checkpoint is ready."
                if enabled and installed and ready
                else "Provide DONUT_CHECKPOINT and enable Donut for selective comparison."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
        )

    def extract_json(self, document: VisionDocumentRecord) -> dict[str, Any]:
        self.ensure_ready()
        import torch
        from PIL import Image

        processor, model = self._load()
        artifact = document.page_artifacts[0]
        image = Image.open(artifact.processed_image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        task_prompt = "<s_documind>"
        decoder_input_ids = processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        with torch.inference_mode():
            generated = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        sequence = processor.batch_decode(generated.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token,
            "",
        )
        sequence = sequence.replace(task_prompt, "").strip()
        parsed = processor.token2json(sequence)
        return parsed if isinstance(parsed, dict) else {"value": parsed}

    def _load(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            checkpoint = str(Path(settings.donut_checkpoint).expanduser())
            self._processor = DonutProcessor.from_pretrained(checkpoint, local_files_only=True)
            self._model = VisionEncoderDecoderModel.from_pretrained(
                checkpoint,
                local_files_only=True,
            )
            self._model.eval()
        return self._processor, self._model
