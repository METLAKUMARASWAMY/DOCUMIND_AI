from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def split_manifest(
    manifest: Path,
    output: Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> dict[str, Any]:
    test_ratio = 1.0 - train_ratio - validation_ratio
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("train, validation, and test ratios must all be positive")
    records = _read_manifest(manifest)
    by_document: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_document.setdefault(str(record["document_id"]), []).append(record)
    document_ids = sorted(by_document)
    random.Random(seed).shuffle(document_ids)
    train_end = round(len(document_ids) * train_ratio)
    validation_end = train_end + round(len(document_ids) * validation_ratio)
    partitions = {
        "train": set(document_ids[:train_end]),
        "validation": set(document_ids[train_end:validation_end]),
        "test": set(document_ids[validation_end:]),
    }
    if any(not values for values in partitions.values()):
        raise ValueError("At least seven complete documents are recommended for a 70/15/15 split")
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "seed": seed,
        "split_unit": "complete_document",
        "source_manifest": str(manifest.resolve()),
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "splits": {},
    }
    for split_name, identifiers in partitions.items():
        selected = [record for record in records if record["document_id"] in identifiers]
        path = output / f"{split_name}.jsonl"
        _write_manifest(path, selected)
        summary["splits"][split_name] = {
            "manifest": str(path.resolve()),
            "documents": len(identifiers),
            "records": len(selected),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    assert_no_document_leakage(output)
    (output / "split.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def assert_no_document_leakage(split_root: Path) -> None:
    document_sets = {
        name: {str(item["document_id"]) for item in _read_manifest(split_root / f"{name}.jsonl")}
        for name in ("train", "validation", "test")
    }
    intersections = {
        "train_validation": document_sets["train"] & document_sets["validation"],
        "train_test": document_sets["train"] & document_sets["test"],
        "validation_test": document_sets["validation"] & document_sets["test"],
    }
    leaked = {name: sorted(values) for name, values in intersections.items() if values}
    if leaked:
        raise ValueError(f"Document leakage detected: {leaked}")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("document_id"):
            raise ValueError(f"Invalid manifest record at {path}:{line_number}")
        records.append(value)
    return records


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a manifest by complete document")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    arguments = parser.parse_args()
    print(
        json.dumps(
            split_manifest(
                arguments.manifest,
                arguments.output,
                seed=arguments.seed,
                train_ratio=arguments.train_ratio,
                validation_ratio=arguments.validation_ratio,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
