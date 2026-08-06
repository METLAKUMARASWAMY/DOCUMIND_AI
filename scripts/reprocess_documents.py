#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
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
        "--force",
        action="store_true",
        help="Reprocess selected documents even when they already use the target pipeline.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=BACKEND_ROOT / "var/vision_state/archive/reprocess",
        help="Versioned evidence archive written before an executed reprocess.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute queued reprocessing. Without this flag the command is a dry run.",
    )
    arguments = parser.parse_args()
    tenant_id = (
        arguments.organisation
        if arguments.organisation.startswith("tenant-")
        else f"tenant-{arguments.organisation}"
    )
    expected = (
        "documind-vision-v4"
        if arguments.pipeline_version == "current"
        else arguments.pipeline_version
    )
    requested_ids = set(arguments.document_id)
    candidates = [
        document
        for document in vision_ai_service.list_documents(tenant_id)
        if (not requested_ids or document.document_id in requested_ids)
        and (
            arguments.force
            or document.metadata.get("pipeline_version") != expected
        )
    ]
    result = {
        "status": "executed" if arguments.execute else "dry_run",
        "organisation": arguments.organisation,
        "tenant_id": tenant_id,
        "target_pipeline": expected,
        "candidate_count": len(candidates),
        "documents": [],
    }
    archive_run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
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
            archive_directory = (
                arguments.archive_root.expanduser().resolve()
                / archive_run
                / document.document_id
            )
            archive_directory.mkdir(parents=True, exist_ok=True)
            (archive_directory / "document.json").write_text(
                json.dumps(document.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if document.review_task:
                (archive_directory / "review.json").write_text(
                    json.dumps(
                        document.review_task.model_dump(mode="json"),
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            (archive_directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "organisation": arguments.organisation,
                        "tenant_id": tenant_id,
                        "document_id": document.document_id,
                        "filename": document.filename,
                        "source": str(source),
                        "pipeline_before": document.metadata.get(
                            "pipeline_version", "legacy"
                        ),
                        "data_version_before": document.data_version,
                        "archived_at": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            response = vision_ai_service.queue_reprocess(
                document.document_id,
                source.read_bytes(),
                actor="legacy-reprocess-script",
                tenant_id=tenant_id,
            )
            vision_ai_service.process_job(response.job_id, tenant_id)
            item["status"] = "reprocessed"
            item["job_id"] = response.job_id
            item["archive_path"] = str(archive_directory)
        result["documents"].append(item)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
