"""Deleting a document must not leave its content behind in the graph.

Verified against the live stack: after ``DELETE /documents/{id}`` the record was
gone from the API, its state file was removed and its vectors were dropped from
Qdrant -- but Neo4j still held 45 nodes and 87 relationships for it, including
the filename, the page structure and every extracted entity value. Deletion
cleaned MongoDB, Redis and MinIO and simply never touched the graph.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.services.vision.persistence import VisionPersistenceCoordinator


@pytest.fixture
def persistence(monkeypatch) -> VisionPersistenceCoordinator:
    store = VisionPersistenceCoordinator()
    monkeypatch.setattr(settings, "enable_neo4j", True)
    monkeypatch.setattr(settings, "enable_mongodb", False)
    monkeypatch.setattr(settings, "enable_redis", False)
    monkeypatch.setattr(settings, "enable_minio", False)
    return store


def _driver_double(store: VisionPersistenceCoordinator, monkeypatch) -> MagicMock:
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    monkeypatch.setattr(store, "_neo4j_driver", lambda: driver)
    monkeypatch.setattr(store, "_neo4j_circuit_open", lambda: False)
    return session


def test_deleting_a_document_deletes_its_graph(persistence, monkeypatch):
    session = _driver_double(persistence, monkeypatch)
    persistence.delete_document("vision-doc-abc123", "tenant-alpha")
    assert session.execute_write.called, "delete_document never touched Neo4j"
    args = session.execute_write.call_args[0]
    assert args[1] == "tenant-alpha"
    assert args[2] == "vision-doc-abc123"


def test_the_delete_query_is_parameterised_and_tenant_scoped():
    tx = MagicMock()
    VisionPersistenceCoordinator._delete_graph_tx(tx, "tenant-alpha", "vision-doc-abc123")
    cypher, kwargs = tx.run.call_args[0][0], tx.run.call_args[1]

    # Identifiers must be bound, never interpolated.
    assert "vision-doc-abc123" not in cypher
    assert "tenant-alpha" not in cypher
    assert kwargs == {"document_id": "vision-doc-abc123", "tenant_id": "tenant-alpha"}

    # Both scopes must be present, or one tenant could erase another's graph.
    assert "$document_id" in cypher and "$tenant_id" in cypher
    assert "DETACH DELETE" in cypher


def test_deletion_is_skipped_when_the_circuit_is_open(persistence, monkeypatch):
    session = _driver_double(persistence, monkeypatch)
    monkeypatch.setattr(persistence, "_neo4j_circuit_open", lambda: True)
    persistence.delete_document("vision-doc-abc123", "tenant-alpha")
    assert not session.execute_write.called


def test_deletion_is_skipped_when_neo4j_is_disabled(persistence, monkeypatch):
    session = _driver_double(persistence, monkeypatch)
    monkeypatch.setattr(settings, "enable_neo4j", False)
    persistence.delete_document("vision-doc-abc123", "tenant-alpha")
    assert not session.execute_write.called


def test_a_neo4j_failure_does_not_break_the_delete(persistence, monkeypatch):
    """The rest of the deletion must still succeed if the graph is unreachable."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    monkeypatch.setattr(persistence, "_neo4j_driver", lambda: driver)
    session.execute_write.side_effect = RuntimeError("neo4j down")

    assert persistence._neo4j_unavailable_since is None
    persistence.delete_document("vision-doc-abc123", "tenant-alpha")  # must not raise

    # The failure is recorded so the breaker backs off, exactly as other
    # Neo4j operations do -- it is not swallowed silently.
    assert persistence._neo4j_unavailable_since is not None
