"""Every module must import standalone, in a fresh interpreter.

A circular import between `app.services.model_registry` and
`app.services.vision.service` was introduced while sharing one constant, and
the suite did not catch it: pytest collects test modules in an order that
happens to import the service first, so the cycle never had to resolve from
the other direction. It would have surfaced in production the first time
anything imported the registry first.

`app.services.vision.__init__` instantiates VisionAIService on import, so any
import from inside that package drags the whole service in. That makes the
package a poor home for shared constants and a good place for this test.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

MODULES = [
    "app.core.answer_policy",
    "app.core.config",
    "app.core.exceptions",
    "app.core.middleware",
    "app.main",
    "app.services.model_registry",
    "app.services.vision.duplicate_detection",
    "app.services.vision.evaluation",
    "app.services.vision.generator_adapters",
    "app.services.vision.graph_builder",
    "app.services.vision.persistence",
    "app.services.vision.query_router",
    "app.services.vision.retrieval_adapters",
    "app.services.vision.service",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    """Each module is imported *first*, which is the direction a cycle breaks in."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{module} could not be imported on its own:\n{result.stderr[-1500:]}"
    )
    assert "circular import" not in result.stderr.lower(), result.stderr[-1500:]


def test_the_shared_abstention_constant_has_no_heavy_dependencies() -> None:
    """It is shared by the registry, the generators, the evaluator and the
    service. If importing it pulls in any of them, the cycle comes back."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.core.answer_policy; "
            "loaded = [m for m in sys.modules if m.startswith('app.services')]; "
            "print(loaded)",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    assert result.stdout.strip() == "[]", f"pulled in services: {result.stdout}"
