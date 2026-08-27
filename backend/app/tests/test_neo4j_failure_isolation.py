"""Neo4j must degrade immediately, not hang, when it is unreachable.

The Neo4j driver retries a failed transaction for `max_transaction_retry_time`
before surfacing the error — 30 seconds by default. Nothing in the code was
bounding that, so every graph write against an unreachable Neo4j slept for 30
seconds and *then* degraded. Measured on the suite: 156.5 of the 166.8 seconds
one four-test file took were spent inside `neo4j/_sync/work/session.py`
`_run_transaction`, sleeping between retries.

That is a production defect before it is a test-speed defect: with Neo4j down,
a user's upload would block for half a minute per document rather than report a
degraded graph. These tests pin both halves of the fix — a bounded driver, and
a circuit breaker so the second call does not pay the timeout again.
"""

from __future__ import annotations

import time

import pytest

from app.core.config import settings
from app.models.vision import VisionGraphProjection
from app.services.vision.persistence import VisionPersistenceCoordinator


class _DeadNeo4jDriver:
    """Stands in for a driver whose server is not answering."""

    def __init__(self) -> None:
        self.session_calls = 0

    def session(self, **_kwargs):
        self.session_calls += 1
        raise ConnectionRefusedError("Connection refused to bolt://neo4j:7687")

    def verify_connectivity(self):
        raise ConnectionRefusedError("Connection refused to bolt://neo4j:7687")


@pytest.fixture
def neo4j_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_neo4j", True)
    monkeypatch.setattr(settings, "neo4j_retry_backoff_seconds", 30.0)
    return settings


def _coordinator_with_dead_driver(
    monkeypatch,
) -> tuple[VisionPersistenceCoordinator, _DeadNeo4jDriver]:
    coordinator = VisionPersistenceCoordinator()
    driver = _DeadNeo4jDriver()
    monkeypatch.setattr(coordinator, "_neo4j_driver", lambda: driver)
    return coordinator, driver


def _graph() -> VisionGraphProjection:
    return VisionGraphProjection(nodes=[], edges=[])


def test_driver_is_constructed_with_bounded_retry_and_timeouts(monkeypatch):
    """An unbounded driver is the whole defect; assert the bounds are passed."""
    captured: dict[str, object] = {}

    class _FakeGraphDatabase:
        @staticmethod
        def driver(uri, **kwargs):
            captured["uri"] = uri
            captured.update(kwargs)
            return object()

    import sys
    import types

    module = types.ModuleType("neo4j")
    module.GraphDatabase = _FakeGraphDatabase
    monkeypatch.setitem(sys.modules, "neo4j", module)

    coordinator = VisionPersistenceCoordinator()
    coordinator._neo4j_driver()

    assert captured["max_transaction_retry_time"] == settings.neo4j_transaction_retry_seconds
    assert captured["connection_timeout"] == settings.neo4j_connection_timeout_seconds
    assert captured["connection_acquisition_timeout"] == settings.neo4j_connection_timeout_seconds
    # A bound that is not actually shorter than the driver default fixes nothing.
    assert settings.neo4j_transaction_retry_seconds < 30.0


def test_graph_write_against_dead_neo4j_degrades_instead_of_raising(neo4j_enabled, monkeypatch):
    coordinator, driver = _coordinator_with_dead_driver(monkeypatch)

    coordinator.upsert_graph("tenant-a", "doc-1", _graph())

    assert driver.session_calls == 1
    assert "neo4j" in coordinator.health()["errors"]
    assert coordinator.health()["status"] == "degraded"


def test_second_graph_write_short_circuits_instead_of_reconnecting(neo4j_enabled, monkeypatch):
    """The breaker exists so call two does not pay the timeout again."""
    coordinator, driver = _coordinator_with_dead_driver(monkeypatch)

    coordinator.upsert_graph("tenant-a", "doc-1", _graph())
    coordinator.upsert_graph("tenant-a", "doc-2", _graph())
    coordinator.upsert_graph("tenant-a", "doc-3", _graph())

    assert driver.session_calls == 1


def test_graph_query_short_circuits_while_the_breaker_is_open(neo4j_enabled, monkeypatch):
    coordinator, driver = _coordinator_with_dead_driver(monkeypatch)

    coordinator.upsert_graph("tenant-a", "doc-1", _graph())
    paths = coordinator.query_graph_paths("tenant-a", "which vendor issued this?", ["doc-1"])

    assert paths == []
    assert driver.session_calls == 1


def test_breaker_lets_one_call_through_once_the_backoff_elapses(neo4j_enabled, monkeypatch):
    """A permanently open breaker would hide a recovered Neo4j — also a defect."""
    coordinator, driver = _coordinator_with_dead_driver(monkeypatch)
    monkeypatch.setattr(settings, "neo4j_retry_backoff_seconds", 0.05)

    coordinator.upsert_graph("tenant-a", "doc-1", _graph())
    assert driver.session_calls == 1

    time.sleep(0.06)
    coordinator.upsert_graph("tenant-a", "doc-2", _graph())
    assert driver.session_calls == 2


def test_recovery_closes_the_breaker_and_clears_the_error(neo4j_enabled, monkeypatch):
    coordinator, _driver = _coordinator_with_dead_driver(monkeypatch)
    monkeypatch.setattr(settings, "neo4j_retry_backoff_seconds", 0.01)

    coordinator.upsert_graph("tenant-a", "doc-1", _graph())
    assert "neo4j" in coordinator.health()["errors"]

    class _LiveSession:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def execute_write(self, *_args, **_kwargs):
            return None

    class _LiveDriver:
        def session(self, **_kwargs):
            return _LiveSession()

    monkeypatch.setattr(coordinator, "_neo4j_driver", _LiveDriver)
    time.sleep(0.02)
    coordinator.upsert_graph("tenant-a", "doc-2", _graph())

    assert "neo4j" not in coordinator.health()["errors"]
    assert coordinator.health()["neo4j_circuit"]["state"] == "closed"


def test_health_discloses_the_open_circuit_rather_than_hiding_it(neo4j_enabled, monkeypatch):
    """Rule 2: a blocked capability must say so, not render as healthy."""
    coordinator, _driver = _coordinator_with_dead_driver(monkeypatch)

    coordinator.upsert_graph("tenant-a", "doc-1", _graph())
    health = coordinator.health()

    assert health["neo4j_circuit"]["state"] == "open"
    assert health["neo4j_circuit"]["retry_after_seconds"] == 30.0
    assert health["status"] == "degraded"


def test_degraded_neo4j_does_not_leak_credentials_into_the_error(neo4j_enabled, monkeypatch):
    coordinator = VisionPersistenceCoordinator()

    class _AuthLeakingDriver:
        def session(self, **_kwargs):
            raise RuntimeError("cannot connect to bolt://neo4j:hunter2@neo4j:7687 password=hunter2")

    monkeypatch.setattr(coordinator, "_neo4j_driver", _AuthLeakingDriver)
    coordinator.upsert_graph("tenant-a", "doc-1", _graph())

    recorded = coordinator.health()["errors"]["neo4j"]
    assert "hunter2" not in recorded
    assert "****" in recorded


def test_disabled_neo4j_never_opens_a_circuit(monkeypatch):
    monkeypatch.setattr(settings, "enable_neo4j", False)
    coordinator, driver = _coordinator_with_dead_driver(monkeypatch)

    coordinator.upsert_graph("tenant-a", "doc-1", _graph())

    assert driver.session_calls == 0
    assert coordinator.health()["neo4j_circuit"]["state"] == "closed"
