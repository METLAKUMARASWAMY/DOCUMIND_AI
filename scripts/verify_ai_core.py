#!/usr/bin/env python3
"""Truthful platform, dependency, checkpoint, service, and model-smoke audit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for candidate in (BACKEND_ROOT, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.core.config import settings  # noqa: E402
from app.services.model_registry import ModelRegistryService  # noqa: E402
from app.services.vision.checkpoint_utils import (  # noqa: E402
    checkpoint_assets_complete,
)

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

PACKAGE_PROBES = {
    "fastapi": ("fastapi", "fastapi"),
    "pymupdf": ("PyMuPDF", "fitz"),
    "pypdf": ("pypdf", "pypdf"),
    "python-docx": ("python-docx", "docx"),
    "pytesseract": ("pytesseract", "pytesseract"),
    "opencv": ("opencv-python-headless", "cv2"),
    "torch": ("torch", "torch"),
    "transformers": ("transformers", "transformers"),
    "sentence-transformers": ("sentence-transformers", "sentence_transformers"),
    "FlagEmbedding": ("FlagEmbedding", "FlagEmbedding"),
    "paddlepaddle": ("paddlepaddle", "paddle"),
    "paddleocr": ("paddleocr", "paddleocr"),
    "qdrant-client": ("qdrant-client", "qdrant_client"),
    "neo4j": ("neo4j", "neo4j"),
}

CHECKPOINT_SUFFIXES = {
    ".bin",
    ".gguf",
    ".onnx",
    ".pdparams",
    ".pt",
    ".pth",
    ".safetensors",
}

SECRET_PATTERN = re.compile(
    r"(?i)(authorization|api[_-]?key|password|secret|token)(\s*[:=]\s*)([^\s,;]+)"
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _redact(value: str) -> str:
    text = SECRET_PATTERN.sub(r"\1\2[REDACTED]", value)
    return re.sub(r"(\w+://)[^/@\s]+@", r"\1[REDACTED]@", text)[:4000]


def _safe_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "configured" if value else "not_configured"
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))


def _run(arguments: list[str], *, timeout: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            arguments,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "returncode": None,
            "error": _redact(f"{type(exc).__name__}: {exc}"),
        }
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": _redact(output),
    }


def _memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        result = _run(["sysctl", "-n", "hw.memsize"])
        try:
            return int(str(result.get("output", "")).strip())
        except ValueError:
            return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _package_status(distribution: str, module: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    probe = _run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                f"module=importlib.import_module({module!r}); "
                "print(getattr(module, '__version__', 'import-ok'))"
            ),
        ],
        timeout=45,
    )
    return {
        "distribution": distribution,
        "module": module,
        "installed": version is not None,
        "version": version,
        "import_ok": bool(probe["ok"]),
        "import_detail": probe.get("output") if not probe["ok"] else None,
    }


def _hf_cached_snapshot(model_id: str) -> Path | None:
    if "/" not in model_id or model_id.startswith(("./", "../", "/")):
        return None
    repository = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + model_id.replace("/", "--")
    )
    reference = repository / "refs" / "main"
    try:
        revision = reference.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    snapshot = repository / "snapshots" / revision
    return snapshot if checkpoint_assets_complete(snapshot) else None


def _checkpoint_status(value: str | None, *, allow_download: bool) -> dict[str, Any]:
    if not value:
        return {
            "configured": False,
            "identifier": None,
            "available": False,
            "source": "not_configured",
            "download_allowed": allow_download,
        }
    expanded = Path(value).expanduser()
    cached = _hf_cached_snapshot(value)
    path = expanded if checkpoint_assets_complete(expanded) else cached
    return {
        "configured": True,
        "identifier": value,
        "available": bool(path and path.exists()),
        "source": (
            "local_path"
            if checkpoint_assets_complete(expanded)
            else ("huggingface_cache" if cached else "missing")
        ),
        "resolved_path": str(path) if path else None,
        "download_allowed": allow_download,
    }


def _project_checkpoints() -> list[dict[str, Any]]:
    ignored_parts = {".git", ".venv", ".venv-ai", "node_modules", "var"}
    values: list[dict[str, Any]] = []
    for root, directories, filenames in os.walk(PROJECT_ROOT):
        directories[:] = [name for name in directories if name not in ignored_parts]
        root_path = Path(root)
        for filename in filenames:
            path = root_path / filename
            if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
                continue
            values.append(
                {"path": str(path.relative_to(PROJECT_ROOT)), "bytes": path.stat().st_size}
            )
    return values


def _docker_status() -> dict[str, Any]:
    executable = shutil.which("docker")
    if not executable:
        return {"cli_installed": False, "daemon_reachable": False, "compose_services": []}
    version = _run([executable, "version", "--format", "{{.Server.Version}}"])
    compose = _run(
        [
            executable,
            "compose",
            "ps",
            "--format",
            "{{.Service}}|{{.State}}|{{.Health}}",
        ]
    )
    services: list[dict[str, Any]] = []
    if compose["ok"] and compose.get("output"):
        for line in str(compose["output"]).splitlines():
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            services.append(
                {
                    "service": parts[0],
                    "state": parts[1],
                    "health": parts[2],
                }
            )
    return {
        "cli_installed": True,
        "daemon_reachable": bool(version["ok"]),
        "server_version": str(version.get("output", "")).strip() if version["ok"] else None,
        "error": version.get("output") if not version["ok"] else None,
        "compose_services": services,
    }


def _qdrant_status() -> dict[str, Any]:
    endpoint = settings.qdrant_url.rstrip("/")
    try:
        with urlopen(endpoint + "/healthz", timeout=5) as response:
            body = response.read(512).decode("utf-8", errors="replace")
            return {
                "configured": settings.enable_qdrant,
                "endpoint": _safe_endpoint(endpoint),
                "reachable": 200 <= response.status < 300,
                "http_status": response.status,
                "response": _redact(body),
            }
    except (OSError, URLError, ValueError) as exc:
        return {
            "configured": settings.enable_qdrant,
            "endpoint": _safe_endpoint(endpoint),
            "reachable": False,
            "error": _redact(f"{type(exc).__name__}: {exc}"),
        }


def _neo4j_status() -> dict[str, Any]:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        return {
            "configured": settings.enable_neo4j,
            "endpoint": _safe_endpoint(settings.neo4j_uri),
            "reachable": False,
            "error": _redact(f"ImportError: {exc}"),
        }
    auth = None
    if settings.neo4j_user and settings.neo4j_password:
        auth = (settings.neo4j_user, settings.neo4j_password)
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=auth,
        connection_timeout=5.0,
    )
    try:
        driver.verify_connectivity()
        return {
            "configured": settings.enable_neo4j,
            "endpoint": _safe_endpoint(settings.neo4j_uri),
            "authentication_configured": auth is not None,
            "reachable": True,
        }
    except Exception as exc:  # driver exposes multiple version-specific exception types
        return {
            "configured": settings.enable_neo4j,
            "endpoint": _safe_endpoint(settings.neo4j_uri),
            "authentication_configured": auth is not None,
            "reachable": False,
            "error": _redact(f"{type(exc).__name__}: {exc}"),
        }
    finally:
        driver.close()


def _smoke_results(
    requested_models: list[str] | None,
    *,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    registry = ModelRegistryService()
    selected = list(requested_models or CORE_LOCAL_MODELS)
    if requested_models is None:
        selected.extend(
            str(item["model_id"])
            for item in registry.backends(include_runtime_health=False)
            if item.get("configured_status") == "active"
        )
    selected = list(dict.fromkeys(selected))
    configured = {
        str(item["model_id"]): str(item.get("configured_status", item.get("status")))
        for item in registry.backends(include_runtime_health=False)
    }
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="documind-ai-core-smoke-") as directory:
        for index, model_id in enumerate(selected):
            output_path = Path(directory) / f"result-{index}.json"
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "smoke_test_models.py"),
                        "--model",
                        model_id,
                        "--output",
                        str(output_path),
                        "--compact",
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_seconds,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            except subprocess.TimeoutExpired:
                results.append(
                    {
                        "model_id": model_id,
                        "configured_status": configured.get(model_id, "unknown"),
                        "status": "failed",
                        "health_status": "unhealthy",
                        "error": f"TimeoutExpired: smoke exceeded {timeout_seconds} seconds",
                        "output_schema": {
                            "status": "not_evaluated",
                            "valid": None,
                        },
                    }
                )
                continue
            if output_path.exists():
                try:
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    result = {
                        "model_id": model_id,
                        "configured_status": configured.get(model_id, "unknown"),
                        "status": "failed",
                        "health_status": "unhealthy",
                        "error": f"{type(exc).__name__}: invalid smoke result file",
                    }
                results.append(result)
                continue
            diagnostic = "\n".join(
                part.strip()
                for part in (completed.stdout, completed.stderr)
                if part.strip()
            )
            results.append(
                {
                    "model_id": model_id,
                    "configured_status": configured.get(model_id, "unknown"),
                    "status": "failed",
                    "health_status": "unhealthy",
                    "error": (
                        f"Smoke subprocess exited {completed.returncode} without a result: "
                        f"{_redact(diagnostic)}"
                    ),
                    "output_schema": {"status": "not_evaluated", "valid": None},
                }
            )
    return results


def _write(path_value: str, payload: dict[str, Any]) -> None:
    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="Run the live local core and every configured-active target adapter.",
    )
    parser.add_argument(
        "--smoke-model",
        action="append",
        help="Run only this model ID (repeatable); implies --run-smoke.",
    )
    parser.add_argument("--output", help="Atomically write the JSON audit to this path.")
    parser.add_argument(
        "--smoke-timeout",
        type=int,
        default=300,
        help="Maximum seconds for each isolated model probe (default: 300).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a configured-active dependency/service/model does not verify.",
    )
    arguments = parser.parse_args()

    memory_bytes = _memory_bytes()
    packages = {
        name: _package_status(distribution, module)
        for name, (distribution, module) in PACKAGE_PROBES.items()
    }
    registry = ModelRegistryService().registry()
    report: dict[str, Any] = {
        "generated_at": _now(),
        "system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "virtual_environment": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
            "memory_bytes": memory_bytes,
            "memory_gib": round(memory_bytes / (1024**3), 2) if memory_bytes else None,
        },
        "configuration": {
            "model_profile": settings.model_profile,
            "enable_heavy_models": settings.enable_heavy_models,
            "enable_paddleocr": settings.enable_paddleocr,
            "enable_paddleocr_structure": settings.enable_paddleocr_structure,
            "enable_bge_m3": settings.enable_bge_m3,
            "enable_qdrant": settings.enable_qdrant,
            "enable_neo4j": settings.enable_neo4j,
            "enable_qwen_local": settings.enable_qwen_local,
            "enable_external_llm": settings.enable_external_llm,
            "external_llm_credentials_configured": bool(
                settings.external_llm_base_url
                and settings.external_llm_api_key
                and settings.external_llm_model
            ),
        },
        "packages": packages,
        "project_checkpoints": _project_checkpoints(),
        "configured_checkpoints": {
            "bge_m3": _checkpoint_status(
                settings.bge_m3_checkpoint,
                allow_download=settings.bge_allow_download,
            ),
            "bge_reranker": _checkpoint_status(
                settings.bge_reranker_checkpoint,
                allow_download=settings.bge_allow_download,
            ),
            "local_llm": _checkpoint_status(
                settings.qwen_checkpoint,
                allow_download=settings.qwen_allow_download,
            ),
        },
        "docker": _docker_status(),
        "services": {
            "qdrant": _qdrant_status(),
            "neo4j": _neo4j_status(),
        },
        "registry": [
            {
                key: item.get(key)
                for key in (
                    "model_id",
                    "adapter_id",
                    "configured_status",
                    "status",
                    "health_status",
                    "last_smoke_test",
                    "package_versions",
                    "checkpoint",
                    "device",
                    "reason",
                    "blocker_category",
                    "requirements",
                )
            }
            for item in registry["backends"]
        ],
    }
    if arguments.run_smoke or arguments.smoke_model:
        smoke = _smoke_results(
            arguments.smoke_model,
            timeout_seconds=max(10, arguments.smoke_timeout),
        )
        report["smoke"] = {
            "total": len(smoke),
            "passed": sum(item["status"] == "passed" for item in smoke),
            "failed": sum(item["status"] == "failed" for item in smoke),
            "blocked": sum(item["status"] == "blocked" for item in smoke),
            "results": smoke,
        }

    failures: list[str] = []
    for item in report.get("smoke", {}).get("results", []):
        if item.get("configured_status") == "active" and item.get("status") != "passed":
            failures.append(f"model:{item.get('model_id')}")
    if settings.enable_qdrant and not report["services"]["qdrant"]["reachable"]:
        failures.append("service:qdrant")
    if settings.enable_neo4j and not report["services"]["neo4j"]["reachable"]:
        failures.append("service:neo4j")
    report["verification"] = {"passed": not failures, "failures": failures}

    if arguments.output:
        _write(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 1 if arguments.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
