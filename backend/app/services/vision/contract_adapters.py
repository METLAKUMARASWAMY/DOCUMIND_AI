from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.security import utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionEvidence,
)
from app.services.vision.adapters import AdapterStatus, ExtractionAdapter
from app.services.vision.checkpoint_validation import validate_transformer_checkpoint


class DebertaContractClauseAdapter(ExtractionAdapter):
    adapter_id = "deberta-cuad-clause-extraction"
    stage = "contract_intelligence"
    model_version = "project-checkpoint"

    def __init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = all(
            importlib.util.find_spec(package) is not None for package in ("transformers", "torch")
        )
        checkpoint = (
            Path(settings.contract_checkpoint).expanduser()
            if settings.contract_checkpoint
            else None
        )
        checkpoint_validation = validate_transformer_checkpoint(
            checkpoint,
            expected_tasks={"contract_clause_extraction", "contract_classification"},
            require_domain_labels=True,
        )
        ready = checkpoint_validation.ready
        enabled = settings.enable_heavy_models and bool(checkpoint)
        active = enabled and installed and ready
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "Fine-tuned CUAD-style DeBERTa checkpoint is ready."
                if active
                else checkpoint_validation.reason
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={
                "checkpoint": str(checkpoint) if checkpoint else None,
                "checkpoint_validation": checkpoint_validation.metadata,
            },
        )

    def extract_fields(self, document: VisionDocumentRecord) -> list[ExtractedVisionField]:
        self.ensure_ready()
        import torch

        tokenizer, model = self._load()
        fields: list[ExtractedVisionField] = []
        for paragraph in self._paragraphs(document.parsed_text):
            encoded = tokenizer(
                paragraph,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = model(**encoded).logits[0]
            if str(getattr(model.config, "problem_type", "")) == "single_label_classification":
                scores = torch.softmax(logits, dim=-1)
            else:
                scores = torch.sigmoid(logits)
            for label_id, score_tensor in enumerate(scores):
                score = float(score_tensor.item())
                if score < settings.contract_clause_threshold:
                    continue
                raw_label = str(
                    model.config.id2label.get(
                        label_id,
                        model.config.id2label.get(str(label_id), f"clause_{label_id}"),
                    )
                )
                label = self._normalise_label(raw_label)
                box, page_number, word_ids = self._evidence_geometry(paragraph, document)
                evidence = VisionEvidence(
                    document_id=document.document_id,
                    page_number=page_number,
                    source_text=paragraph,
                    bounding_box=box,
                    visual_region=f"page_{page_number}",
                    model_confidence=score,
                    model_name=self.adapter_id,
                    model_version=self.model_version,
                    processing_timestamp=utc_now_iso(),
                    reviewer_status=ReviewerStatus.unverified,
                    metadata={"word_ids": word_ids, "calibrated": False},
                )
                fields.append(
                    ExtractedVisionField(
                        name=label,
                        value=paragraph,
                        normalized_value=" ".join(paragraph.lower().split()),
                        confidence=score,
                        evidence=evidence,
                        model_name=self.adapter_id,
                        model_version=self.model_version,
                        target_model="DeBERTa-v3-large-CUAD",
                        confidence_components={"deberta_probability": score},
                        calibrated=False,
                    )
                )
        return self._deduplicate(fields)

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            checkpoint = str(Path(settings.contract_checkpoint).expanduser())
            self._tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint,
                local_files_only=True,
            )
            self._model.eval()
        return self._tokenizer, self._model

    def _paragraphs(self, text: str) -> list[str]:
        paragraphs = [
            " ".join(item.split())
            for item in re.split(r"\n\s*\n|(?<=[.;])\s+(?=[A-Z])", text)
            if len(item.split()) >= 5
        ]
        return paragraphs[:300]

    def _normalise_label(self, label: str) -> str:
        value = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        aliases = {
            "document_name": "parties",
            "parties_to_the_agreement": "parties",
            "effective_date": "effective_date",
            "expiration_date": "expiry_date",
            "governing_law": "governing_law",
        }
        return aliases.get(value, value)

    def _evidence_geometry(
        self,
        paragraph: str,
        document: VisionDocumentRecord,
    ) -> tuple[VisionBoundingBox | None, int, list[str]]:
        targets = {re.sub(r"[^a-z0-9]+", "", token.lower()) for token in paragraph.split()}
        matches = [
            word
            for word in document.ocr_words
            if re.sub(r"[^a-z0-9]+", "", word.text.lower()) in targets
        ]
        if not matches:
            return None, 1, []
        page_number = matches[0].page
        page_matches = [word for word in matches if word.page == page_number][:40]
        box = VisionBoundingBox(
            x1=min(word.bounding_box.x1 for word in page_matches),
            y1=min(word.bounding_box.y1 for word in page_matches),
            x2=max(word.bounding_box.x2 for word in page_matches),
            y2=max(word.bounding_box.y2 for word in page_matches),
        )
        return box, page_number, [word.word_id for word in page_matches]

    def _deduplicate(self, fields: list[ExtractedVisionField]) -> list[ExtractedVisionField]:
        retained: dict[tuple[str, str], ExtractedVisionField] = {}
        for field in fields:
            key = (field.name, field.normalized_value or field.value.lower())
            if key not in retained or field.confidence > retained[key].confidence:
                retained[key] = field
        return list(retained.values())
