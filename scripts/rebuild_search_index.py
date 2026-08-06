#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
os.chdir(BACKEND_ROOT)

from app.services.vision.service import vision_ai_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild tenant-scoped document chunks and vector indexes idempotently"
    )
    parser.add_argument("--organisation", required=True)
    parser.add_argument("--document-id", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    tenant_id = (
        arguments.organisation
        if arguments.organisation.startswith("tenant-")
        else f"tenant-{arguments.organisation}"
    )
    requested = set(arguments.document_id)
    documents = [
        document
        for document in vision_ai_service.list_documents(tenant_id)
        if not requested or document.document_id in requested
    ]
    result: dict[str, object] = {
        "status": "executed" if arguments.execute else "dry_run",
        "organisation": arguments.organisation,
        "tenant_id": tenant_id,
        "document_count": len(documents),
        "documents": [],
    }
    rows: list[dict[str, object]] = []
    for document in documents:
        before = {chunk.chunk_id for chunk in document.evidence_chunks}
        rebuilt = vision_ai_service._chunks(document)
        after = {chunk.chunk_id for chunk in rebuilt}
        row: dict[str, object] = {
            "document_id": document.document_id,
            "filename": document.filename,
            "before_count": len(before),
            "after_count": len(after),
            "obsolete_count": len(before - after),
            "duplicate_chunk_ids": len(rebuilt) - len(after),
        }
        if arguments.execute:
            document.evidence_chunks = rebuilt
            vision_ai_service.retrieval_index.remove_document(document.document_id)
            vision_ai_service._index_document(document)
            document.data_version += 1
            document.analytics_cache_version += 1
            vision_ai_service._persist_document(document)
            row["persisted"] = True
        rows.append(row)
    result["documents"] = rows
    result["duplicate_chunk_ids"] = sum(
        int(row["duplicate_chunk_ids"]) for row in rows
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result["duplicate_chunk_ids"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
