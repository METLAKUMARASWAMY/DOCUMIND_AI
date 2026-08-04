#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
DEFAULT_STATE = PROJECT_ROOT / "backend/var/vision_state/documents"
from app.services.vision.legacy_audit import (  # noqa: E402
    CURRENT_PIPELINE_VERSION,
    legacy_document_flags,
)


def audit(
    state_directory: Path,
    *,
    organisation: str | None = None,
    apply_flags: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for path in sorted(state_directory.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if organisation and document.get("tenant_id") != organisation:
            continue
        flags = legacy_document_flags(document)
        if not flags:
            continue
        counts.update(flags)
        records.append(
            {
                "document_id": document.get("document_id"),
                "filename": document.get("filename"),
                "status": document.get("status"),
                "pipeline_version": (document.get("metadata") or {}).get(
                    "pipeline_version", "legacy"
                ),
                "flags": flags,
            }
)
        if apply_flags:
            metadata = dict(document.get("metadata") or {})
            metadata["legacy_flags"] = flags
            metadata["reprocess_required"] = "REPROCESS_REQUIRED" in flags
            if "LEGACY_APPROVAL_REVIEW_REQUIRED" in flags:
                metadata["legacy_approval_message"] = (
                    "Approved under an earlier policy. Re-verification is recommended."
                )
            document["metadata"] = metadata
            path.write_text(
                json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True),
                encoding="utf-8",
            )
    return {
        "status": "applied" if apply_flags else "audit_only",
        "current_pipeline": CURRENT_PIPELINE_VERSION,
        "state_directory": str(state_directory),
        "organisation": organisation,
        "flag_counts": dict(sorted(counts.items())),
        "record_count": len(records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit DocuMind records created under earlier pipeline policies"
    )
    parser.add_argument("--state-directory", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--organisation")
    parser.add_argument(
        "--apply-flags",
        action="store_true",
        help="Persist non-destructive legacy and re-verification flags.",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(
                arguments.state_directory,
                organisation=arguments.organisation,
                apply_flags=arguments.apply_flags,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
