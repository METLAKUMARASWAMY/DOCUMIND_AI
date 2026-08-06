#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.vision.model_smoke import ModelSmokeService  # noqa: E402

CORE_LOCAL_MODELS = (
    "local-parser-v1",
    "Tesseract-5",
    "local-layout-adapter-v1",
    "deterministic-schema-extractor-v3",
    "OpenCV-visual-cues-v1",
    "ELA-noise-forensics-v1",
    "IsolationForest",
    "local-hashing-v1",
    "local-vector-store-v1",
    "local-graph-projection-v1",
    "grounded-template-generator-v1",
)


def _aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "total": len(results),
        "passed": sum(item.get("status") == "passed" for item in results),
        "failed": sum(item.get("status") == "failed" for item in results),
        "blocked": sum(item.get("status") == "blocked" for item in results),
        "results": results,
    }


def _compact(payload: dict[str, object]) -> dict[str, object]:
    results = payload.get("results", [payload])
    if not isinstance(results, list):
        results = [payload]
    return {
        key: payload[key]
        for key in ("generated_at", "total", "passed", "failed", "blocked")
        if key in payload
    } | {
        "results": [
            {
                key: item.get(key)
                for key in (
                    "model_id",
                    "adapter_id",
                    "configured_status",
                    "status",
                    "health_status",
                    "duration_ms",
                    "completed_at",
                    "device",
                    "package_version",
                    "checkpoint",
                    "output_schema",
                    "error",
                    "blocker",
                )
            }
            for item in results
            if isinstance(item, dict)
        ]
    }


def _write_json(path_value: str, payload: dict[str, object]) -> None:
    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify DocuMind model initialization, inference, and output contracts"
    )
    parser.add_argument(
        "--model",
        help="Registered model ID or adapter ID. Omit to inspect every registered model.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failing exit code when any configured model fails or is blocked.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--core-local",
        action="store_true",
        help="Probe every adapter used by the dependency-light live pipeline.",
    )
    selection.add_argument(
        "--configured-only",
        action="store_true",
        help="Probe only adapters whose current configured status is active.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print lifecycle status without verbose synthetic inference output.",
    )
    parser.add_argument(
        "--output",
        help="Atomically write the complete JSON result to this path.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered model IDs and configured statuses without running probes.",
    )
    arguments = parser.parse_args()
    service = ModelSmokeService()
    try:
        if arguments.list:
            result = {
                "results": [
                    {
                        "model_id": item["model_id"],
                        "adapter_id": item.get("adapter_id"),
                        "configured_status": item.get("configured_status"),
                    }
                    for item in service.registry.backends(include_runtime_health=False)
                ]
            }
        elif arguments.model:
            result = service.run(arguments.model)
        elif arguments.core_local:
            result = _aggregate([service.run(model_id) for model_id in CORE_LOCAL_MODELS])
        elif arguments.configured_only:
            configured = [
                str(item["model_id"])
                for item in service.registry.backends(include_runtime_health=False)
                if item.get("configured_status") == "active"
            ]
            result = _aggregate([service.run(model_id) for model_id in configured])
        else:
            result = service.run_all()
    except ValueError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    if arguments.output:
        _write_json(arguments.output, result)
    printable = _compact(result) if arguments.compact else result
    print(json.dumps(printable, indent=2, sort_keys=True, default=str))
    if arguments.strict:
        results = result.get("results", [result])
        configured_failures = [
            item
            for item in results
            if item.get("configured_status") == "active" and item.get("status") != "passed"
        ]
        return 1 if configured_failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
