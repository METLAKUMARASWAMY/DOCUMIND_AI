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
        description="Selectively reprocess legacy DocuMind documents in place"
    )
    parser.add_argument("--organisation", required=True)
    parser.add_argument("--pipeline-version", default="current")
    parser.add_argument("--document-id", action="append", default=[])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute queued reprocessing. Without this flag the command is a dry run.",
    )
    arguments = parser.parse_args()
    expected = (
        "documind-vision-v3"
        if arguments.pipeline_version == "current"
        else arguments.pipeline_version
    )
    requested_ids = set(arguments.document_id)
    candidates = [
        document
        for document in vision_ai_service.list_documents(arguments.organisation)
        if (not requested_ids or document.document_id in requested_ids)
        and document.metadata.get("pipeline_version") != expected
    ]
    result = {
        "status": "executed" if arguments.execute else "dry_run",
        "organisation": arguments.organisation,
        "target_pipeline": expected,
        "candidate_count": len(candidates),
        "documents": [],
    }
    for document in candidates:
        source = Path(str(document.original_storage_path or ""))
        if source and not source.is_absolute():
            source = BACKEND_ROOT / source
        item = {
            "document_id": document.document_id,
            "filename": document.filename,
            "source": str(source),
            "status": "ready" if source.is_file() else "blocked_source_missing",
        }
        if arguments.execute and source.is_file():
            response = vision_ai_service.queue_reprocess(
                document.document_id,
                source.read_bytes(),
                actor="legacy-reprocess-script",
                tenant_id=arguments.organisation,
            )
            vision_ai_service.process_job(response.job_id, arguments.organisation)
            item["status"] = "reprocessed"
            item["job_id"] = response.job_id
        result["documents"].append(item)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
