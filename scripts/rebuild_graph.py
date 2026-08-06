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
        description="Rebuild tenant-scoped semantic graph projections idempotently"
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
    rows: list[dict[str, object]] = []
    for document in documents:
        rebuilt = vision_ai_service._graph(document)
        node_ids = [str(node.get("id")) for node in rebuilt.nodes]
        edge_ids = [str(edge.get("id")) for edge in rebuilt.edges]
        row: dict[str, object] = {
            "document_id": document.document_id,
            "filename": document.filename,
            "before_nodes": len(document.graph.nodes),
            "after_nodes": len(node_ids),
            "before_edges": len(document.graph.edges),
            "after_edges": len(edge_ids),
            "duplicate_node_ids": len(node_ids) - len(set(node_ids)),
            "duplicate_edge_ids": len(edge_ids) - len(set(edge_ids)),
        }
        if arguments.execute:
            document.graph = rebuilt
            vision_ai_service.persistence.upsert_graph(
                document.tenant_id,
                document.document_id,
                rebuilt,
            )
            document.data_version += 1
            document.analytics_cache_version += 1
            vision_ai_service._persist_document(document)
            row["persisted"] = True
        rows.append(row)
    duplicate_count = sum(
        int(row["duplicate_node_ids"]) + int(row["duplicate_edge_ids"])
        for row in rows
    )
    print(
        json.dumps(
            {
                "status": "executed" if arguments.execute else "dry_run",
                "organisation": arguments.organisation,
                "tenant_id": tenant_id,
                "document_count": len(documents),
                "duplicate_graph_ids": duplicate_count,
                "documents": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if duplicate_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
