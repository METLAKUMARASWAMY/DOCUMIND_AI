#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.split import split_manifest  # noqa: E402


def prepare(
    annotations: Path,
    output: Path,
    *,
    seed: int = 42,
    audit_only: bool = False,
) -> dict[str, Any]:
    records = _records(annotations)
    errors: list[str] = []
    canonical: list[dict[str, Any]] = []
    labels: set[str] = set()
    for index, record in enumerate(records, start=1):
        validated, record_errors = _validate_record(record, annotations.parent, index)
        errors.extend(record_errors)
        if validated:
            canonical.append(validated)
            labels.update(
                str(word["label"])
                for word in validated["words"]
                if str(word["label"]).lower() not in {"o", "other"}
            )
    document_count = len({item["document_id"] for item in canonical})
    audit = {
        "source": str(annotations.resolve()),
        "record_count": len(records),
        "valid_record_count": len(canonical),
        "document_count": document_count,
        "domain_labels": sorted(labels),
        "ready_for_layoutlmv3": not errors and document_count >= 7 and bool(labels),
        "errors": errors[:50],
        "requirements": {
            "minimum_documents": 7,
            "split_unit": "complete_document",
            "required_geometry": "one absolute [x1,y1,x2,y2] box per OCR word",
            "required_assets": "rendered page image for every page record",
        },
    }
    if audit_only:
        return audit
    if errors:
        raise ValueError("Annotation audit failed: " + " | ".join(errors[:5]))
    if document_count < 7:
        raise ValueError(
            "At least seven complete documents are required for non-empty train/validation/test splits"
        )
    if not labels:
        raise ValueError("At least one non-background field label is required")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(item, sort_keys=True, default=str) + "\n" for item in canonical),
        encoding="utf-8",
    )
    split = split_manifest(manifest, output / "splits", seed=seed)
    result = {**audit, "manifest": str(manifest.resolve()), "split": split}
    (output / "preparation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload if isinstance(payload, list) else payload.get("records", [])
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise ValueError("Annotations must be a JSON array or JSONL objects")
    return [dict(item) for item in values]


def _validate_record(
    record: dict[str, Any],
    source_root: Path,
    index: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = []
    prefix = f"record {index}"
    document_id = str(record.get("document_id", "")).strip()
    page_id = str(record.get("page_id", "")).strip()
    image_value = str(record.get("image_path", record.get("source_path", ""))).strip()
    image_path = Path(image_value).expanduser() if image_value else None
    if image_path and not image_path.is_absolute():
        image_path = source_root / image_path
    width = record.get("width", dict(record.get("metadata") or {}).get("width"))
    height = record.get("height", dict(record.get("metadata") or {}).get("height"))
    words = record.get("words")
    if not document_id:
        errors.append(f"{prefix}: document_id is missing")
    if not page_id:
        errors.append(f"{prefix}: page_id is missing")
    if image_path is None or not image_path.is_file():
        errors.append(f"{prefix}: rendered image is missing ({image_value or 'not supplied'})")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        errors.append(f"{prefix}: positive integer width and height are required")
    if not isinstance(words, list) or not words:
        errors.append(f"{prefix}: OCR words with boxes and field labels are required")
        return None, errors
    normalized_words = []
    for word_index, word in enumerate(words, start=1):
        if not isinstance(word, dict):
            errors.append(f"{prefix} word {word_index}: expected an object")
            continue
        text = str(word.get("text", "")).strip()
        label = str(word.get("label", "other")).strip().lower()
        box = word.get("bounding_box", word.get("box"))
        if not text:
            errors.append(f"{prefix} word {word_index}: text is missing")
            continue
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, (int, float)) for value in box)
        ):
            errors.append(f"{prefix} word {word_index}: bounding_box must contain four numbers")
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        if x2 <= x1 or y2 <= y1:
            errors.append(f"{prefix} word {word_index}: bounding_box has no positive area")
            continue
        if isinstance(width, int) and isinstance(height, int) and (x2 > width or y2 > height):
            errors.append(f"{prefix} word {word_index}: bounding_box exceeds page dimensions")
            continue
        normalized_words.append(
            {
                **word,
                "text": text,
                "label": label,
                "bounding_box": [x1, y1, x2, y2],
            }
        )
    if errors:
        return None, errors
    return {
        "document_id": document_id,
        "page_id": page_id,
        "task": "token_classification",
        "image_path": str(image_path.resolve()),
        "source_path": str(image_path.resolve()),
        "text": " ".join(item["text"] for item in normalized_words),
        "words": normalized_words,
        "annotations": normalized_words,
        "metadata": {"width": width, "height": height, **dict(record.get("metadata") or {})},
    }, []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and prepare LayoutLMv3 word-box annotations"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "dataset/documind_ai_synthetic_dataset/labels/ner_records.jsonl",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "ml/prepared/layoutlmv3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    try:
        result = prepare(
            arguments.annotations,
            arguments.output,
            seed=arguments.seed,
            audit_only=arguments.audit_only,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ready_for_layoutlmv3") else 2


if __name__ == "__main__":
    raise SystemExit(main())
