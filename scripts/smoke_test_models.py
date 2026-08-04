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
    arguments = parser.parse_args()
    service = ModelSmokeService()
    try:
        result = service.run(arguments.model) if arguments.model else service.run_all()
    except ValueError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
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
