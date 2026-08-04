#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = PROJECT_ROOT / "backend/var/vision_state"


def canonical_key(payload: dict[str, Any], artifact_type: str) -> str:
    identity = payload.get("identity_hash")
    if identity:
        return str(identity)
    scope = {
        "artifact_type": artifact_type,
        "tenant_id": payload.get("tenant_id"),
        "type": payload.get("narrative_type") or payload.get("report_type"),
        "request_scope": payload.get("request_scope") or {},
        "document_ids": sorted(set(payload.get("document_ids") or [])),
        "source_data_version": payload.get("source_data_version") or "legacy",
    }
    return hashlib.sha256(
        json.dumps(scope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def archive(
    state_root: Path,
    *,
    organisation: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "status": "applied" if apply else "dry_run",
        "organisation": organisation,
    }
    for directory_name, artifact_type in (
        ("narratives", "narrative"),
        ("reports", "report"),
    ):
        groups: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
        for path in sorted((state_root / directory_name).glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if organisation and payload.get("tenant_id") != organisation:
                continue
            groups[canonical_key(payload, artifact_type)].append((path, payload))
        archived = []
        for identity, items in groups.items():
            if len(items) < 2:
                continue
            ordered = sorted(
                items,
                key=lambda item: (
                    int(item[1].get("version") or 1),
                    str(item[1].get("created_at") or ""),
                ),
                reverse=True,
            )
            keeper = ordered[0][1]
            for path, payload in ordered[1:]:
                if payload.get("status") == "archived":
                    continue
                archived.append(
                    {
                        "artifact_id": payload.get("narrative_id")
                        or payload.get("report_id"),
                        "kept_id": keeper.get("narrative_id") or keeper.get("report_id"),
                        "identity_hash": identity,
                    }
                )
                if apply:
                    payload["status"] = "archived"
                    payload["archived_reason"] = "duplicate_canonical_identity"
                    payload["canonical_artifact_id"] = keeper.get(
                        "narrative_id"
                    ) or keeper.get("report_id")
                    path.write_text(
                        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
                        encoding="utf-8",
                    )
        results[directory_name] = {
            "duplicate_count": len(archived),
            "artifacts": archived,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archive duplicate narrative and report artifacts without deleting evidence"
    )
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--organisation")
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            archive(
                arguments.state_root,
                organisation=arguments.organisation,
                apply=arguments.apply,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
