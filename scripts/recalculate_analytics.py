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

from app.services.vision.analytics import vision_analytics_service  # noqa: E402
from app.services.vision.legacy_audit import CURRENT_PIPELINE_VERSION  # noqa: E402
from app.services.vision.service import vision_ai_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recalculate tenant analytics and classify unreliable legacy timing"
    )
    parser.add_argument("--organisation", required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    tenant_id = (
        arguments.organisation
        if arguments.organisation.startswith("tenant-")
        else f"tenant-{arguments.organisation}"
    )
    documents = vision_ai_service.list_documents(tenant_id)
    excluded: list[dict[str, str]] = []
    included: list[str] = []
    for document in documents:
        vision_analytics_service.refresh_document(document)
        sample = vision_analytics_service._document_processing_sample(document)
        if sample is None:
            reason = (
                "legacy_or_incompatible_pipeline_version"
                if str(document.metadata.get("pipeline_version") or "legacy")
                != CURRENT_PIPELINE_VERSION
                else "missing_or_invalid_processing_timestamps"
            )
            excluded.append(
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "reason": reason,
                }
            )
        else:
            included.append(document.document_id)
        if arguments.execute:
            document.metadata["analytics_timing_status"] = {
                "calculation_version": vision_analytics_service.calculation_version,
                "included": sample is not None,
                "exclusion_reason": None if sample is not None else excluded[-1]["reason"],
            }
            document.analytics_cache_version += 1
            vision_ai_service._persist_document(document)
    overview = vision_ai_service.analytics_overview(tenant_id)
    print(
        json.dumps(
            {
                "status": "executed" if arguments.execute else "dry_run",
                "organisation": arguments.organisation,
                "tenant_id": tenant_id,
                "document_count": len(documents),
                "included_timing_samples": len(included),
                "excluded_timing_samples": excluded,
                "calculation_version": overview["calculation_version"],
                "processing_timing": overview["processing_timing"],
                "structured_extraction_coverage": overview[
                    "structured_extraction_coverage"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
