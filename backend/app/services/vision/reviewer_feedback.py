"""Durable, tenant-scoped reviewer feedback for controlled future reuse."""

from __future__ import annotations

import json
import os
import re
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.security import hash_document
from app.models.vision import ReviewerStatus, VisionSemanticType

APPROVED_REVIEW_STATUSES = {
    ReviewerStatus.accepted.value,
    ReviewerStatus.edited.value,
}
_FIELD_NAME = re.compile(r"[a-z][a-z0-9_]{0,79}")
_DATASET_LOCK = threading.RLock()


class ReviewerFeedbackDataset:
    """Append-only correction dataset; never retrains a production model implicitly."""

    schema_version = "review-feedback-v2"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, tenant_id: str) -> Path:
        tenant_key = hash_document(tenant_id.encode())[:20]
        return self.root / f"{tenant_key}.jsonl"

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        tenant_id = str(record.get("tenant_id") or "").strip()
        if not tenant_id:
            raise ValueError("tenant_id is required for reviewer feedback")
        stored = {
            "schema_version": self.schema_version,
            "feedback_kind": "field_correction",
            "correction_id": str(record.get("correction_id") or f"correction-{uuid4().hex[:12]}"),
            "training_policy": "curated_export_only_no_automatic_retraining",
            **record,
            "tenant_id": tenant_id,
        }
        digest_payload = json.dumps(
            stored,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        stored["record_hash"] = hash_document(digest_payload)
        path = self.path_for(tenant_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(stored, sort_keys=True, ensure_ascii=False) + "\n"
        with _DATASET_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return stored

    def records(self, tenant_id: str) -> tuple[dict[str, Any], ...]:
        path = self.path_for(tenant_id)
        if not path.exists():
            return ()
        retained: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ()
        for line in lines:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(record, dict):
                continue
            # Exact tenant validation prevents a copied or corrupted shard from
            # becoming cross-tenant training/alias input.
            if str(record.get("tenant_id") or "") != tenant_id:
                continue
            if not self._integrity_valid(record):
                continue
            retained.append(record)
        return tuple(retained)

    def approved_label_aliases(
        self,
        tenant_id: str,
        *,
        normalizer: Callable[[str], str],
    ) -> dict[str, str]:
        candidates: dict[str, set[str]] = defaultdict(set)
        for record in self.records(tenant_id):
            if record.get("status") not in APPROVED_REVIEW_STATUSES:
                continue
            raw_label = normalizer(str(record.get("raw_label") or ""))
            corrected_name = str(record.get("corrected_field_name") or "")
            if raw_label and _FIELD_NAME.fullmatch(corrected_name):
                candidates[raw_label].add(corrected_name)
        # Conflicting corrections abstain instead of silently selecting whichever
        # record happened to be written last.
        return {
            label: next(iter(destinations))
            for label, destinations in candidates.items()
            if len(destinations) == 1
        }

    def reviewed_value_aliases(self, tenant_id: str) -> dict[str, dict[str, str]]:
        candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
        supported_types = {item.value for item in VisionSemanticType}
        for record in self.records(tenant_id):
            if record.get("status") not in APPROVED_REVIEW_STATUSES:
                continue
            semantic_type = str(record.get("semantic_type") or "")
            if semantic_type not in supported_types:
                continue
            original = self._normalize_value(
                str(record.get("original_value") or ""),
                semantic_type,
            )
            corrected = self._normalize_value(
                str(record.get("corrected_value") or ""),
                semantic_type,
            )
            if original and corrected and original != corrected:
                candidates[(semantic_type, original)].add(corrected)
        aliases: dict[str, dict[str, str]] = defaultdict(dict)
        for (semantic_type, original), corrected_values in candidates.items():
            if len(corrected_values) == 1:
                aliases[semantic_type][original] = next(iter(corrected_values))
        return dict(aliases)

    def summary(self, tenant_id: str) -> dict[str, Any]:
        records = self.records(tenant_id)
        statuses: dict[str, int] = defaultdict(int)
        document_types: dict[str, int] = defaultdict(int)
        corrected_fields: dict[str, int] = defaultdict(int)
        semantic_types: dict[str, int] = defaultdict(int)
        document_ids: set[str] = set()
        for record in records:
            statuses[str(record.get("status") or "unknown")] += 1
            document_types[str(record.get("document_type") or "unknown")] += 1
            corrected_fields[str(record.get("corrected_field_name") or "unknown")] += 1
            semantic_types[str(record.get("semantic_type") or "unknown")] += 1
            if record.get("document_id"):
                document_ids.add(str(record["document_id"]))
        return {
            "schema_version": self.schema_version,
            "tenant_id": tenant_id,
            "record_count": len(records),
            "document_count": len(document_ids),
            "status_counts": dict(sorted(statuses.items())),
            "document_type_counts": dict(sorted(document_types.items())),
            "corrected_field_counts": dict(sorted(corrected_fields.items())),
            "semantic_type_counts": dict(sorted(semantic_types.items())),
            "automatic_retraining": False,
            "training_policy": "curated_export_only_no_automatic_retraining",
        }

    def _normalize_value(self, value: str, semantic_type: str) -> str:
        if semantic_type == VisionSemanticType.email.value:
            return value.strip().casefold()
        if semantic_type == VisionSemanticType.url.value:
            return value.strip().casefold().rstrip("/")
        if semantic_type == VisionSemanticType.bank_account.value:
            return re.sub(r"[^a-z0-9]+", "", value.casefold())
        if semantic_type == VisionSemanticType.phone.value:
            prefix = "+" if value.strip().startswith("+") else ""
            return prefix + re.sub(r"\D+", "", value)
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    def _integrity_valid(self, record: dict[str, Any]) -> bool:
        stored_hash = str(record.get("record_hash") or "")
        if not stored_hash:
            # Legacy v1 rows predate record hashes and remain importable.
            return record.get("schema_version") in {None, "review-feedback-v1"}
        payload = {key: value for key, value in record.items() if key != "record_hash"}
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hash_document(encoded) == stored_hash


__all__ = ["ReviewerFeedbackDataset"]
