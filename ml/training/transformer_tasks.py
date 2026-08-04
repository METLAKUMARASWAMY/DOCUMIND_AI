from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ml.training.common import (
    ExperimentTracker,
    TrainingConfig,
    class_weights,
    read_manifest,
    seed_everything,
    transformer_training_arguments,
    validate_training_inputs,
    write_run_manifest,
)


def train_layoutlmv3_classification(config: TrainingConfig) -> dict[str, Any]:
    validation = validate_training_inputs(config)
    seed_everything(config.seed)
    _require_packages("torch", "transformers", "PIL")
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoProcessor,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    split_records = _split_records(config)
    labels = sorted({str(item["label"]) for item in split_records["train"]})
    if len(labels) < 2:
        raise ValueError("Document classification requires at least two training labels")
    label_to_id = {label: index for index, label in enumerate(labels)}
    processor = AutoProcessor.from_pretrained(config.base_model, apply_ocr=True)

    class DocumentDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            with Image.open(_image_path(record)) as image:
                encoded = processor(
                    images=image.convert("RGB"),
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt",
                )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = torch.tensor(label_to_id[str(record["label"])])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )
    weights = class_weights(
        [label_to_id[str(item["label"])] for item in split_records["train"]],
        len(labels),
    )

    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels_tensor = inputs.pop("labels")
            outputs = model(**inputs)
            weight = torch.tensor(weights, device=outputs.logits.device) if config.weighted_loss else None
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels_tensor, weight=weight)
            return (loss, outputs) if return_outputs else loss

    def metrics(result: Any) -> dict[str, float]:
        predicted = np.asarray(result.predictions).argmax(axis=-1).tolist()
        expected = np.asarray(result.label_ids).tolist()
        return _classification_metrics(expected, predicted, len(labels))

    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(**transformer_training_arguments(config)),
        train_dataset=DocumentDataset(split_records["train"]),
        eval_dataset=DocumentDataset(split_records["validation"]),
        processing_class=processor,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(config.early_stopping_patience)],
    )
    return _train_and_record(
        config,
        validation,
        trainer,
        processor,
        DocumentDataset(split_records["test"]),
        extra={"labels": labels, "class_weights": weights},
    )


def train_layoutlmv3_token_classification(config: TrainingConfig) -> dict[str, Any]:
    validation = validate_training_inputs(config)
    seed_everything(config.seed)
    _require_packages("torch", "transformers", "PIL")
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoProcessor,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    split_records = _split_records(config)
    entity_labels = sorted(
        {
            str(annotation.get("label", annotation.get("category", "other"))).lower()
            for item in split_records["train"]
            for annotation in _annotation_records(item)
            if str(annotation.get("label", annotation.get("category", "other"))).lower()
            not in {"", "other", "o"}
        }
    )
    labels = ["O", *[value for entity in entity_labels for value in (f"B-{entity}", f"I-{entity}")]]
    label_to_id = {label: index for index, label in enumerate(labels)}
    processor = AutoProcessor.from_pretrained(config.base_model, apply_ocr=False)

    class TokenDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            words, boxes, word_labels = _layout_words(record, label_to_id)
            with Image.open(_image_path(record)) as source:
                image = source.convert("RGB")
                normalized_boxes = [_normalize_box(box, image.width, image.height) for box in boxes]
                encoded = processor(
                    images=image,
                    text=words,
                    boxes=normalized_boxes,
                    word_labels=word_labels,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt",
                )
            return {key: value.squeeze(0) for key, value in encoded.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        config.base_model,
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )
    weights = _token_weights(split_records["train"], label_to_id)

    class WeightedTokenTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels_tensor = inputs.pop("labels")
            outputs = model(**inputs)
            weight = torch.tensor(weights, device=outputs.logits.device) if config.weighted_loss else None
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.reshape(-1, len(labels)),
                labels_tensor.reshape(-1),
                weight=weight,
                ignore_index=-100,
            )
            return (loss, outputs) if return_outputs else loss

    def metrics(result: Any) -> dict[str, float]:
        predicted = np.asarray(result.predictions).argmax(axis=-1)
        expected = np.asarray(result.label_ids)
        mask = expected != -100
        return _classification_metrics(
            expected[mask].tolist(),
            predicted[mask].tolist(),
            len(labels),
        )

    trainer = WeightedTokenTrainer(
        model=model,
        args=TrainingArguments(**transformer_training_arguments(config)),
        train_dataset=TokenDataset(split_records["train"]),
        eval_dataset=TokenDataset(split_records["validation"]),
        processing_class=processor,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(config.early_stopping_patience)],
    )
    return _train_and_record(
        config,
        validation,
        trainer,
        processor,
        TokenDataset(split_records["test"]),
        extra={"labels": labels, "class_weights": weights},
    )


def train_donut(config: TrainingConfig) -> dict[str, Any]:
    validation = validate_training_inputs(config)
    seed_everything(config.seed)
    _require_packages("torch", "transformers", "PIL")
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import (
        DonutProcessor,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        VisionEncoderDecoderModel,
    )

    split_records = _split_records(config)
    processor = DonutProcessor.from_pretrained(config.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(config.base_model)
    prompt = str(config.parameters.get("task_prompt", "<s_documind>"))
    if prompt not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [prompt]})
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(prompt)
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    class DonutDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            with Image.open(_image_path(record)) as source:
                pixels = processor(source.convert("RGB"), return_tensors="pt").pixel_values.squeeze(0)
            target = json.dumps(
                record.get("target_json", record.get("label", record.get("annotations", {}))),
                sort_keys=True,
                separators=(",", ":"),
            )
            tokens = processor.tokenizer(
                f"{prompt}{target}{processor.tokenizer.eos_token}",
                add_special_tokens=False,
                max_length=int(config.parameters.get("max_target_length", 768)),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            tokens[tokens == processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixels, "labels": tokens}

    def metrics(result: Any) -> dict[str, float]:
        predictions = processor.batch_decode(result.predictions, skip_special_tokens=True)
        labels_array = np.where(result.label_ids == -100, processor.tokenizer.pad_token_id, result.label_ids)
        references = processor.batch_decode(labels_array, skip_special_tokens=True)
        exact = [float(_normalize_text(left) == _normalize_text(right)) for left, right in zip(references, predictions, strict=True)]
        return {"exact_match": round(float(np.mean(exact)), 4) if exact else 0.0}

    arguments = transformer_training_arguments(config)
    arguments["predict_with_generate"] = True
    arguments["generation_max_length"] = int(config.parameters.get("max_target_length", 768))
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(**arguments),
        train_dataset=DonutDataset(split_records["train"]),
        eval_dataset=DonutDataset(split_records["validation"]),
        processing_class=processor,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(config.early_stopping_patience)],
    )
    return _train_and_record(
        config,
        validation,
        trainer,
        processor,
        DonutDataset(split_records["test"]),
    )


def train_contract_model(config: TrainingConfig) -> dict[str, Any]:
    validation = validate_training_inputs(config, required_record_keys={"text", "annotations"})
    seed_everything(config.seed)
    _require_packages("torch", "transformers")
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    split_records = _split_records(config)
    categories = sorted(
        {
            _category(annotation)
            for item in split_records["train"]
            for annotation in _annotation_records(item)
            if _category(annotation)
        }
    )
    labels = ["O", *[value for category in categories for value in (f"B-{category}", f"I-{category}")]]
    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)

    class ContractDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            text = str(record["text"])
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=int(config.parameters.get("max_length", 512)),
                return_offsets_mapping=True,
            )
            spans = _contract_spans(text, _annotation_records(record))
            encoded["labels"] = _span_labels(encoded.pop("offset_mapping"), spans, label_to_id)
            return encoded

    model = AutoModelForTokenClassification.from_pretrained(
        config.base_model,
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )

    def metrics(result: Any) -> dict[str, float]:
        predicted = np.asarray(result.predictions).argmax(axis=-1)
        expected = np.asarray(result.label_ids)
        mask = expected != -100
        return _classification_metrics(expected[mask].tolist(), predicted[mask].tolist(), len(labels))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**transformer_training_arguments(config)),
        train_dataset=ContractDataset(split_records["train"]),
        eval_dataset=ContractDataset(split_records["validation"]),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(config.early_stopping_patience)],
    )
    return _train_and_record(
        config,
        validation,
        trainer,
        tokenizer,
        ContractDataset(split_records["test"]),
        extra={"clause_categories": categories},
    )


def train_tamper_localization(config: TrainingConfig) -> dict[str, Any]:
    validation = validate_training_inputs(config)
    seed_everything(config.seed)
    _require_packages("torch", "transformers", "PIL")
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import (
        AutoImageProcessor,
        AutoModelForSemanticSegmentation,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    split_records = _split_records(config)
    labels = list(config.parameters.get("labels", ["background", "manipulated_region"]))
    processor = AutoImageProcessor.from_pretrained(config.base_model)

    class SegmentationDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            mask_path = record.get("mask_path") or record.get("metadata", {}).get("mask_path")
            if not mask_path or not Path(mask_path).is_file():
                raise ValueError(f"Tamper mask is missing for {record['page_id']}")
            with Image.open(_image_path(record)) as image, Image.open(mask_path) as mask:
                encoded = processor(
                    images=image.convert("RGB"),
                    segmentation_maps=mask.convert("L"),
                    return_tensors="pt",
                )
            return {key: value.squeeze(0) for key, value in encoded.items()}

    model = AutoModelForSemanticSegmentation.from_pretrained(
        config.base_model,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id={label: index for index, label in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )

    def metrics(result: Any) -> dict[str, float]:
        logits = torch.from_numpy(np.asarray(result.predictions))
        expected = torch.from_numpy(np.asarray(result.label_ids))
        resized = torch.nn.functional.interpolate(
            logits,
            size=expected.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        ious = []
        for label_id in range(len(labels)):
            predicted_mask = resized == label_id
            expected_mask = expected == label_id
            intersection = int((predicted_mask & expected_mask).sum())
            union = int((predicted_mask | expected_mask).sum())
            if union:
                ious.append(intersection / union)
        return {"mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**transformer_training_arguments(config)),
        train_dataset=SegmentationDataset(split_records["train"]),
        eval_dataset=SegmentationDataset(split_records["validation"]),
        processing_class=processor,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(config.early_stopping_patience)],
    )
    return _train_and_record(
        config,
        validation,
        trainer,
        processor,
        SegmentationDataset(split_records["test"]),
        extra={"mask_labels": labels},
    )


def _train_and_record(
    config: TrainingConfig,
    validation: dict[str, Any],
    trainer: Any,
    processor: Any,
    test_dataset: Any,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with ExperimentTracker(config, validation) as tracker:
        train_result = trainer.train()
        validation_metrics = trainer.evaluate(metric_key_prefix="validation")
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        checkpoint = config.output_dir / "best"
        trainer.save_model(str(checkpoint))
        processor.save_pretrained(str(checkpoint))
        result = {
            "status": "trained",
            "checkpoint": str(checkpoint.resolve()),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "train": _numeric_metrics(train_result.metrics),
            "validation": _numeric_metrics(validation_metrics),
            "test": _numeric_metrics(test_metrics),
            "early_stopping_patience": config.early_stopping_patience,
            "validation_based_model_selection": True,
            **(extra or {}),
        }
        manifest = write_run_manifest(config, validation, result)
        tracker.log_metrics({"validation": result["validation"], "test": result["test"]})
        tracker.log_artifact(manifest, "run")
        for artifact in config.output_dir.glob("**/confusion_matrix*.json"):
            tracker.log_artifact(artifact, "evaluation")
    return result


def _split_records(config: TrainingConfig) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": read_manifest(config.train_manifest),
        "validation": read_manifest(config.validation_manifest),
        "test": read_manifest(config.test_manifest),
    }


def _image_path(record: dict[str, Any]) -> Path:
    value = record.get("image_path") or record.get("source_path")
    if not value:
        raise ValueError(f"No image path is available for {record.get('page_id')}")
    path = Path(str(value)).expanduser()
    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise ValueError(f"A rendered image is required for {record.get('page_id')}: {path}")
    return path


def _annotation_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("annotations") or []
    return [dict(item) for item in value if isinstance(item, dict)]


def _layout_words(
    record: dict[str, Any],
    label_to_id: dict[str, int],
) -> tuple[list[str], list[list[float]], list[int]]:
    annotations = _annotation_records(record)
    words = record.get("words") or annotations
    if not isinstance(words, list) or not words:
        raise ValueError(f"OCR word annotations are required for {record.get('page_id')}")
    output_words = []
    boxes = []
    labels = []
    previous_label = ""
    for item in words:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        box = item.get("bounding_box", item.get("box"))
        if not text or not isinstance(box, list) or len(box) != 4:
            continue
        raw_label = str(item.get("label", "other")).lower()
        if raw_label in {"", "other", "o"}:
            encoded_label = "O"
        else:
            prefix = "I" if raw_label == previous_label else "B"
            encoded_label = f"{prefix}-{raw_label}"
        output_words.append(text)
        boxes.append([float(value) for value in box])
        labels.append(label_to_id.get(encoded_label, 0))
        previous_label = raw_label
    if not output_words:
        raise ValueError(f"No valid OCR word boxes found for {record.get('page_id')}")
    return output_words, boxes, labels


def _normalize_box(box: list[float], width: int, height: int) -> list[int]:
    return [
        max(0, min(1000, round(1000 * box[0] / width))),
        max(0, min(1000, round(1000 * box[1] / height))),
        max(0, min(1000, round(1000 * box[2] / width))),
        max(0, min(1000, round(1000 * box[3] / height))),
    ]


def _token_weights(records: list[dict[str, Any]], label_to_id: dict[str, int]) -> list[float]:
    labels = []
    for record in records:
        try:
            labels.extend(_layout_words(record, label_to_id)[2])
        except ValueError:
            continue
    return class_weights(labels, len(label_to_id))


def _category(annotation: dict[str, Any]) -> str:
    value = str(annotation.get("category", annotation.get("label", ""))).strip().lower()
    return value.replace(" ", "_").replace("-", "_")


def _contract_spans(
    text: str,
    annotations: list[dict[str, Any]],
) -> list[tuple[int, int, str]]:
    spans = []
    for annotation in annotations:
        category = _category(annotation)
        answer = str(annotation.get("text", ""))
        start = annotation.get("start")
        if start is None and answer:
            start = text.find(answer)
        if category and isinstance(start, int) and start >= 0 and answer:
            spans.append((start, start + len(answer), category))
    return spans


def _span_labels(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int, str]],
    label_to_id: dict[str, int],
) -> list[int]:
    labels = []
    seen: set[tuple[int, int, str]] = set()
    for start, end in offsets:
        if start == end:
            labels.append(-100)
            continue
        match = next((span for span in spans if start < span[1] and end > span[0]), None)
        if not match:
            labels.append(label_to_id["O"])
            continue
        prefix = "I" if match in seen else "B"
        seen.add(match)
        labels.append(label_to_id[f"{prefix}-{match[2]}"])
    return labels


def _classification_metrics(
    expected: list[int],
    predicted: list[int],
    label_count: int,
) -> dict[str, float]:
    accuracy = sum(left == right for left, right in zip(expected, predicted, strict=True)) / max(len(expected), 1)
    precisions = []
    recalls = []
    f1_values = []
    for label in range(label_count):
        tp = sum(left == label and right == label for left, right in zip(expected, predicted, strict=True))
        fp = sum(left != label and right == label for left, right in zip(expected, predicted, strict=True))
        fn = sum(left == label and right != label for left, right in zip(expected, predicted, strict=True))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_f1": round(float(np.mean(f1_values)), 4),
    }


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _require_packages(*packages: str) -> None:
    import importlib.util

    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    if missing:
        raise RuntimeError(f"Install the training dependencies before running this task: {missing}")
