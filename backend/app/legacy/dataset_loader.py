import csv
import json
import os
from functools import cached_property
from pathlib import Path
from typing import Any

from app.core.config import settings


class DatasetLoader:
    def __init__(self, dataset_path: str | None = None) -> None:
        raw_path = (
            dataset_path or os.getenv("DOCUMIND_DATASET_PATH") or settings.documind_dataset_path
        )
        self.dataset_path = self._resolve_path(raw_path)
        self.labels_path = self.dataset_path / "labels"
        self.documents_path = self.dataset_path / "documents"

    def load_all_cases(self) -> list[dict[str, Any]]:
        return self._rows("cases.csv")

    def load_case(self, case_id: str) -> dict[str, Any] | None:
        return next((row for row in self.load_all_cases() if row["case_id"] == case_id), None)

    def load_case_documents(self, case_id: str) -> list[dict[str, Any]]:
        documents = [row for row in self._rows("documents.csv") if row["case_id"] == case_id]
        for document in documents:
            document["text"] = self.load_document_text(document["filename"])
            document["path"] = str(self.documents_path / document["filename"])
        return documents

    def load_case_entities(self, case_id: str) -> list[dict[str, Any]]:
        return [row for row in self._rows("entities.csv") if row["case_id"] == case_id]

    def load_case_evidence(self, case_id: str) -> list[dict[str, Any]]:
        return [row for row in self._rows("evidence_chunks.csv") if row["case_id"] == case_id]

    def load_case_compliance_findings(self, case_id: str) -> list[dict[str, Any]]:
        return [row for row in self._rows("compliance_findings.csv") if row["case_id"] == case_id]

    def load_case_fraud_findings(self, case_id: str) -> list[dict[str, Any]]:
        return [row for row in self._rows("fraud_findings.csv") if row["case_id"] == case_id]

    def load_case_cross_checks(self, case_id: str) -> list[dict[str, Any]]:
        return [row for row in self._rows("cross_document_checks.csv") if row["case_id"] == case_id]

    def load_case_workflow(self, case_id: str) -> dict[str, Any] | None:
        return next(
            (row for row in self._rows("workflow_decisions.csv") if row["case_id"] == case_id), None
        )

    def load_case_agents(self, case_id: str) -> list[dict[str, Any]]:
        return [row for row in self._rows("agent_decisions.csv") if row["case_id"] == case_id]

    def load_case_graph(self, case_id: str) -> list[dict[str, Any]]:
        return [row for row in self._rows("knowledge_graph_edges.csv") if row["case_id"] == case_id]

    def load_case_rag_questions(self, case_id: str) -> list[dict[str, Any]]:
        return [row for row in self._rows("rag_questions.csv") if row["case_id"] == case_id]

    def load_ner_records(self) -> list[dict[str, Any]]:
        path = self.labels_path / "ner_records.jsonl"
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def load_document_text(self, relative_filename: str) -> str:
        path = self.documents_path / relative_filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")

    @cached_property
    def _tables(self) -> dict[str, list[dict[str, Any]]]:
        return {}

    def _rows(self, filename: str) -> list[dict[str, Any]]:
        if filename not in self._tables:
            path = self.labels_path / filename
            if not path.exists():
                self._tables[filename] = []
            else:
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    self._tables[filename] = [dict(row) for row in csv.DictReader(handle)]
        return list(self._tables[filename])

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        backend_root = Path(__file__).resolve().parents[2]
        project_root = backend_root.parent
        candidates = [project_root / path, backend_root / path, Path.cwd() / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return project_root / path


dataset_loader = DatasetLoader()


def load_all_cases() -> list[dict[str, Any]]:
    return dataset_loader.load_all_cases()


def load_case(case_id: str) -> dict[str, Any] | None:
    return dataset_loader.load_case(case_id)


def load_case_documents(case_id: str) -> list[dict[str, Any]]:
    return dataset_loader.load_case_documents(case_id)


def load_case_entities(case_id: str) -> list[dict[str, Any]]:
    return dataset_loader.load_case_entities(case_id)


def load_case_evidence(case_id: str) -> list[dict[str, Any]]:
    return dataset_loader.load_case_evidence(case_id)


def load_case_compliance_findings(case_id: str) -> list[dict[str, Any]]:
    return dataset_loader.load_case_compliance_findings(case_id)


def load_case_fraud_findings(case_id: str) -> list[dict[str, Any]]:
    return dataset_loader.load_case_fraud_findings(case_id)


def load_case_cross_checks(case_id: str) -> list[dict[str, Any]]:
    return dataset_loader.load_case_cross_checks(case_id)


def load_case_workflow(case_id: str) -> dict[str, Any] | None:
    return dataset_loader.load_case_workflow(case_id)


def load_case_agents(case_id: str) -> list[dict[str, Any]]:
    return dataset_loader.load_case_agents(case_id)


def load_case_graph(case_id: str) -> list[dict[str, Any]]:
    return dataset_loader.load_case_graph(case_id)


def load_case_rag_questions(case_id: str) -> list[dict[str, Any]]:
    return dataset_loader.load_case_rag_questions(case_id)


def load_ner_records() -> list[dict[str, Any]]:
    return dataset_loader.load_ner_records()
