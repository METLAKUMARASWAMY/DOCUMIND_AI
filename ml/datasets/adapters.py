from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


class DatasetAdapter(ABC):
    """Converts a source dataset into document-keyed canonical records."""

    name: str

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {self.root}")

    @abstractmethod
    def records(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    def _record(
        self,
        *,
        document_id: str,
        task: str,
        source_path: Path | None = None,
        page_id: str | None = None,
        text: str = "",
        label: Any = None,
        annotations: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "document_id": document_id,
            "page_id": page_id or document_id,
            "task": task,
            "source_path": str(source_path.resolve()) if source_path else None,
            "text": text,
            "label": label,
            "annotations": annotations or [],
            "dataset": self.name,
            "metadata": metadata or {},
        }


class SyntheticEnterpriseAdapter(DatasetAdapter):
    name = "documind_synthetic_enterprise"

    def records(self) -> Iterator[dict[str, Any]]:
        labels = self.root / "labels" / "documents.csv"
        documents_root = self.root / "documents"
        if not labels.is_file():
            raise FileNotFoundError(f"Expected {labels}")
        with labels.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                relative = Path(str(row.get("filename", "")))
                source_path = documents_root / relative
                text = source_path.read_text(encoding="utf-8", errors="replace")
                yield self._record(
                    document_id=str(row["document_id"]),
                    page_id=f"{row['document_id']}:1",
                    task="document_classification",
                    source_path=source_path,
                    text=text,
                    label=row.get("document_type", "unknown"),
                    metadata={
                        "case_id": row.get("case_id"),
                        "language": row.get("language", "unknown"),
                        "pages": int(row.get("pages", 1)),
                    },
                )

    def ner_records(self) -> Iterator[dict[str, Any]]:
        path = self.root / "labels" / "ner_records.jsonl"
        for item in _jsonl(path):
            yield self._record(
                document_id=str(item["document_id"]),
                task="token_classification",
                text=str(item.get("text", "")),
                annotations=item.get("entities", []),
                metadata={"case_id": item.get("case_id")},
            )


class RVLCDIPAdapter(DatasetAdapter):
    name = "rvl_cdip"

    def records(self) -> Iterator[dict[str, Any]]:
        label_files = sorted(self.root.glob("labels/*.txt")) or sorted(self.root.glob("*.txt"))
        if not label_files:
            raise FileNotFoundError("RVL-CDIP label files were not found; prepare labels/*.txt first")
        for label_file in label_files:
            split = label_file.stem
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.rsplit(maxsplit=1)
                if len(parts) != 2:
                    continue
                relative_path, label = parts
                image_path = self.root / "images" / relative_path
                document_id = Path(relative_path).stem
                yield self._record(
                    document_id=document_id,
                    page_id=f"{document_id}:1",
                    task="document_classification",
                    source_path=image_path,
                    label=label,
                    metadata={"source_split": split},
                )


class FUNSDAdapter(DatasetAdapter):
    name = "funsd"

    def records(self) -> Iterator[dict[str, Any]]:
        annotation_files = sorted(self.root.glob("**/annotations/*.json"))
        if not annotation_files:
            raise FileNotFoundError("FUNSD annotations/*.json files were not found")
        for annotation_path in annotation_files:
            payload = _json(annotation_path)
            document_id = annotation_path.stem
            image_path = annotation_path.parent.parent / "images" / f"{document_id}.png"
            words = []
            for item in payload.get("form", []):
                for word in item.get("words", []):
                    words.append(
                        {
                            "text": word.get("text", ""),
                            "bounding_box": word.get("box", [0, 0, 0, 0]),
                            "label": item.get("label", "other"),
                            "linking": item.get("linking", []),
                        }
                    )
            yield self._record(
                document_id=document_id,
                page_id=f"{document_id}:1",
                task="token_classification",
                source_path=image_path,
                text=" ".join(str(word["text"]) for word in words),
                annotations=words,
            )


class ReceiptFieldAdapter(DatasetAdapter):
    name = "cord_sroie"

    def records(self) -> Iterator[dict[str, Any]]:
        json_files = sorted(self.root.glob("**/*.json"))
        text_annotations = sorted(self.root.glob("**/*.txt"))
        if not json_files and not text_annotations:
            raise FileNotFoundError("CORD/SROIE JSON or text annotations were not found")
        for annotation_path in json_files:
            payload = _json(annotation_path)
            document_id = annotation_path.stem
            image_path = _matching_image(annotation_path)
            yield self._record(
                document_id=document_id,
                page_id=f"{document_id}:1",
                task="receipt_field_extraction",
                source_path=image_path,
                text=str(payload.get("text", payload.get("valid_line", ""))),
                annotations=payload,
            )
        for annotation_path in text_annotations:
            document_id = annotation_path.stem
            image_path = _matching_image(annotation_path)
            lines = annotation_path.read_text(encoding="utf-8", errors="replace").splitlines()
            annotations = []
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 9 and all(part.strip().lstrip("-").isdigit() for part in parts[:8]):
                    coordinates = [int(part) for part in parts[:8]]
                    annotations.append({"polygon": coordinates, "text": ",".join(parts[8:])})
            if annotations:
                yield self._record(
                    document_id=document_id,
                    page_id=f"{document_id}:1",
                    task="receipt_field_extraction",
                    source_path=image_path,
                    text=" ".join(str(item["text"]) for item in annotations),
                    annotations=annotations,
                )


class PubLayNetAdapter(DatasetAdapter):
    name = "publaynet"

    def records(self) -> Iterator[dict[str, Any]]:
        annotation_files = sorted(self.root.glob("**/*.json"))
        annotation_files = [path for path in annotation_files if "coco" in path.name.lower() or "train" in path.name.lower() or "val" in path.name.lower()]
        if not annotation_files:
            raise FileNotFoundError("PubLayNet COCO annotation JSON was not found")
        for annotation_file in annotation_files:
            payload = _json(annotation_file)
            images = {int(item["id"]): item for item in payload.get("images", [])}
            categories = {int(item["id"]): item["name"] for item in payload.get("categories", [])}
            grouped: dict[int, list[dict[str, Any]]] = {}
            for item in payload.get("annotations", []):
                grouped.setdefault(int(item["image_id"]), []).append(
                    {
                        "region_type": categories.get(int(item["category_id"]), "unknown"),
                        "bbox_xywh": item.get("bbox", []),
                        "segmentation": item.get("segmentation", []),
                    }
                )
            for image_id, image in images.items():
                filename = str(image.get("file_name", ""))
                document_id = Path(filename).stem
                yield self._record(
                    document_id=document_id,
                    page_id=f"{document_id}:1",
                    task="layout_detection",
                    source_path=self.root / filename,
                    annotations=grouped.get(image_id, []),
                    metadata={"width": image.get("width"), "height": image.get("height")},
                )


class DocumentQAAdapter(DatasetAdapter):
    name = "docvqa_dude"

    def records(self) -> Iterator[dict[str, Any]]:
        annotation_files = sorted(self.root.glob("**/*.json")) + sorted(self.root.glob("**/*.jsonl"))
        if not annotation_files:
            raise FileNotFoundError("DocVQA/DUDE JSON annotations were not found")
        for annotation_file in annotation_files:
            payloads = list(_jsonl(annotation_file)) if annotation_file.suffix == ".jsonl" else [_json(annotation_file)]
            for payload in payloads:
                items = payload.get("data", payload.get("questions", [payload]))
                if isinstance(items, dict):
                    items = [items]
                for index, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    document_id = str(item.get("document_id", item.get("docId", annotation_file.stem)))
                    question_id = str(item.get("question_id", item.get("questionId", index)))
                    image_name = item.get("image", item.get("image_path"))
                    yield self._record(
                        document_id=document_id,
                        page_id=str(item.get("page_id", f"{document_id}:1")),
                        task="document_qa",
                        source_path=self.root / str(image_name) if image_name else None,
                        text=str(item.get("question", "")),
                        label=item.get("answers", item.get("answer", [])),
                        metadata={"question_id": question_id},
                    )


class CUADAdapter(DatasetAdapter):
    name = "cuad"

    def records(self) -> Iterator[dict[str, Any]]:
        candidates = sorted(self.root.glob("**/*.json"))
        if not candidates:
            raise FileNotFoundError("CUAD SQuAD-style JSON was not found")
        for source in candidates:
            payload = _json(source)
            for contract in payload.get("data", []):
                title = str(contract.get("title", source.stem))
                for paragraph_index, paragraph in enumerate(contract.get("paragraphs", [])):
                    context = str(paragraph.get("context", ""))
                    annotations = []
                    for qa in paragraph.get("qas", []):
                        for answer in qa.get("answers", []):
                            annotations.append(
                                {
                                    "category": qa.get("question", ""),
                                    "text": answer.get("text", ""),
                                    "start": answer.get("answer_start"),
                                }
                            )
                    yield self._record(
                        document_id=title,
                        page_id=f"{title}:{paragraph_index + 1}",
                        task="contract_clause_extraction",
                        text=context,
                        annotations=annotations,
                    )


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "synthetic": SyntheticEnterpriseAdapter,
    "rvl-cdip": RVLCDIPAdapter,
    "funsd": FUNSDAdapter,
    "cord": ReceiptFieldAdapter,
    "sroie": ReceiptFieldAdapter,
    "publaynet": PubLayNetAdapter,
    "docvqa": DocumentQAAdapter,
    "dude": DocumentQAAdapter,
    "cuad": CUADAdapter,
}


def adapter_for(name: str, root: Path) -> DatasetAdapter:
    try:
        adapter = ADAPTERS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown adapter {name!r}; choose from {sorted(ADAPTERS)}") from exc
    return adapter(root)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        yield value


def _matching_image(annotation_path: Path) -> Path | None:
    roots = [annotation_path.parent, annotation_path.parent.parent / "images", annotation_path.parent.parent / "img"]
    for root in roots:
        for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = root / f"{annotation_path.stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None
