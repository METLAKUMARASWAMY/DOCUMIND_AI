from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts/migrate_postgres.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("migrate_postgres_test_module", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeState:
    def __init__(self, *, tables: set[str] | None = None) -> None:
        self.tables = set(tables or set())
        self.tracked: dict[str, dict[str, Any]] = {}
        self.executed_sql: list[str] = []
        self.connections: list[FakeConnection] = []

    async def connect(self, database_url: str) -> FakeConnection:
        connection = FakeConnection(self, database_url)
        self.connections.append(connection)
        return connection


class FakeTransaction(AbstractAsyncContextManager):
    async def __aenter__(self) -> FakeTransaction:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class FakeConnection:
    def __init__(self, state: FakeState, database_url: str) -> None:
        self.state = state
        self.database_url = database_url
        self.closed = False

    async def fetchval(self, query: str, *_args: Any) -> Any:
        if query == "SELECT 1":
            return 1
        if query == "SHOW server_version":
            return "17.10"
        if query == "SELECT current_database()":
            return "documind"
        if "to_regclass" in query:
            return "documind_schema_migrations" in self.state.tables
        raise AssertionError(f"Unexpected fetchval query: {query}")

    async def fetch(self, query: str, *_args: Any) -> list[dict[str, Any]]:
        if "FROM pg_catalog.pg_tables" in query:
            return [{"tablename": name} for name in sorted(self.state.tables)]
        if "FROM documind_schema_migrations" in query:
            return [
                {
                    "migration_id": migration_id,
                    "checksum_sha256": record["checksum"],
                    "applied_at": record["applied_at"],
                    "execution_ms": record["execution_ms"],
                }
                for migration_id, record in sorted(self.state.tracked.items())
            ]
        raise AssertionError(f"Unexpected fetch query: {query}")

    async def execute(self, query: str, *args: Any) -> str:
        self.state.executed_sql.append(query)
        if "CREATE TABLE IF NOT EXISTS documind_schema_migrations" in query:
            self.state.tables.add("documind_schema_migrations")
        elif query.lstrip().startswith("INSERT INTO documind_schema_migrations"):
            self.state.tracked[str(args[0])] = {
                "checksum": str(args[1]),
                "applied_at": datetime(2026, 8, 6, tzinfo=UTC),
                "execution_ms": int(args[2]),
            }
        elif "CREATE EXTENSION IF NOT EXISTS" in query:
            self.state.tables.update(
                migration_table
                for migration in self._migrations
                if migration.migration_id == "001_initial_schema"
                for migration_table in migration.required_tables
            )
        elif "vision_document_completeness" in query:
            self.state.tables.update(
                migration_table
                for migration in self._migrations
                if migration.migration_id == "002_vision_remediation"
                for migration_table in migration.required_tables
            )
        elif "vision_field_candidate_resolutions" in query:
            self.state.tables.update(
                migration_table
                for migration in self._migrations
                if migration.migration_id == "003_core_intelligence_v4"
                for migration_table in migration.required_tables
            )
        return "OK"

    @property
    def _migrations(self):
        return MIGRATE_POSTGRES.MIGRATIONS

    def transaction(self) -> FakeTransaction:
        return FakeTransaction()

    async def close(self) -> None:
        self.closed = True


MIGRATE_POSTGRES = _load_script()


def test_postgres_url_normalisation_and_safe_config_never_return_credentials():
    raw = "postgresql+asyncpg://service-user:secret-value@db.internal:6432/documind?sslmode=require"

    normalised = MIGRATE_POSTGRES.normalise_database_url(raw)
    safe = MIGRATE_POSTGRES.safe_database_config(normalised)

    assert normalised.startswith("postgresql://")
    assert safe == {
        "scheme": "postgresql",
        "host": "db.internal",
        "port": 6432,
        "database": "documind",
        "credentials": "configured",
    }
    assert "secret-value" not in json.dumps(safe)
    assert "service-user" not in json.dumps(safe)


@pytest.mark.asyncio
async def test_postgres_default_mode_is_read_only_and_reports_pending_migrations():
    state = FakeState(tables={"tenants", "documents"})

    payload = await MIGRATE_POSTGRES.inspect_or_migrate(
        "postgresql+asyncpg://service-user:secret-value@127.0.0.1:5432/documind",
        connector=state.connect,
    )

    assert payload["status"] == "dry_run"
    assert payload["connection"]["select_1"] is True
    assert payload["server"] == {"version": "17.10", "database": "documind"}
    assert [row["outcome"] for row in payload["migrations"]] == [
        "pending",
        "pending",
        "pending",
    ]
    assert state.executed_sql == []
    assert state.connections[0].closed is True
    assert "secret-value" not in json.dumps(payload)
    assert "service-user" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_postgres_execute_is_idempotent_and_tracks_migration_checksums():
    state = FakeState()
    database_url = "postgresql://service-user:secret-value@127.0.0.1:5432/documind"

    first = await MIGRATE_POSTGRES.inspect_or_migrate(
        database_url,
        execute=True,
        connector=state.connect,
    )
    sql_count_after_first_run = len(state.executed_sql)
    second = await MIGRATE_POSTGRES.inspect_or_migrate(
        database_url,
        execute=True,
        connector=state.connect,
    )

    assert first["status"] == "executed"
    assert [row["outcome"] for row in first["migrations"]] == [
        "applied",
        "applied",
        "applied",
    ]
    assert first["tables"]["missing_required"] == []
    assert [row["outcome"] for row in second["migrations"]] == [
        "already_applied",
        "already_applied",
        "already_applied",
    ]
    second_run_sql = state.executed_sql[sql_count_after_first_run:]
    assert not any("CREATE EXTENSION IF NOT EXISTS" in query for query in second_run_sql)
    assert len(state.tracked) == 3
    assert all(connection.closed for connection in state.connections)


@pytest.mark.asyncio
async def test_postgres_execute_blocks_changed_applied_migration():
    state = FakeState(tables={"documind_schema_migrations"})
    state.tracked["001_initial_schema"] = {
        "checksum": "not-the-repository-checksum",
        "applied_at": datetime(2026, 8, 6, tzinfo=UTC),
        "execution_ms": 12,
    }

    payload = await MIGRATE_POSTGRES.inspect_or_migrate(
        "postgresql://127.0.0.1:5432/documind",
        execute=True,
        connector=state.connect,
    )

    assert payload["status"] == "blocked"
    assert payload["migrations"][0]["outcome"] == "checksum_mismatch"
    assert "create a new migration" in payload["error"]
    assert not any("CREATE EXTENSION IF NOT EXISTS" in query for query in state.executed_sql)
