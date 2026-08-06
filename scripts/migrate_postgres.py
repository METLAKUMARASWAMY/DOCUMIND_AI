#!/usr/bin/env python3
"""Inspect or apply DocuMind PostgreSQL migrations safely.

The command is read-only unless ``--execute`` is supplied. It deliberately
loads ``backend/.env`` and never includes the database URL, username, password,
or driver exception text in its JSON output.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings  # noqa: E402

TRACKING_TABLE = "documind_schema_migrations"
LOCK_NAME = "documind-schema-migrations-v1"


@dataclass(frozen=True)
class Migration:
    migration_id: str
    path: Path
    required_tables: tuple[str, ...]

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


MIGRATIONS = (
    Migration(
        "001_initial_schema",
        PROJECT_ROOT / "infra/docker/postgres-init.sql",
        (
            "tenants",
            "documents",
            "document_versions",
            "document_pages",
            "document_chunks",
            "evidence",
            "extracted_fields",
            "reports",
            "audit_logs",
            "graph_nodes",
            "graph_edges",
        ),
    ),
    Migration(
        "002_vision_remediation",
        PROJECT_ROOT / "infra/migrations/002_vision_remediation.sql",
        (
            "vision_document_completeness",
            "vision_processing_warnings",
            "vision_risk_indicators",
            "vision_model_runtime_status",
            "vision_approval_overrides",
            "vision_fraud_confirmations",
            "finance_narratives",
        ),
    ),
    Migration(
        "003_core_intelligence_v4",
        PROJECT_ROOT / "infra/migrations/003_core_intelligence_v4.sql",
        (
            "vision_field_candidate_resolutions",
            "vision_processing_timings",
            "vision_conversations",
            "vision_retrieval_traces",
        ),
    ),
)


class AsyncTransaction(Protocol):
    async def __aenter__(self) -> Any: ...

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...


class AsyncConnection(Protocol):
    async def execute(self, query: str, *args: Any) -> str: ...

    async def fetch(self, query: str, *args: Any) -> list[Any]: ...

    async def fetchval(self, query: str, *args: Any) -> Any: ...

    def transaction(self) -> AsyncTransaction: ...

    async def close(self) -> None: ...


Connector = Callable[[str], Awaitable[AsyncConnection]]


def load_database_url() -> str | None:
    """Load the backend environment without mutating process cwd."""

    backend_env = BACKEND_ROOT / ".env"
    configured = Settings(_env_file=backend_env if backend_env.exists() else None)
    return configured.database_url


def normalise_database_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"postgresql", "postgresql+asyncpg"}:
        raise ValueError("DATABASE_URL must use postgresql or postgresql+asyncpg")
    if not parts.hostname or not parts.path.strip("/"):
        raise ValueError("DATABASE_URL must include a host and database")
    return urlunsplit(("postgresql", parts.netloc, parts.path, parts.query, parts.fragment))


def safe_database_config(value: str) -> dict[str, Any]:
    """Return connection metadata that cannot disclose URL credentials."""

    parts = urlsplit(value)
    return {
        "scheme": parts.scheme,
        "host": parts.hostname or "",
        "port": parts.port or 5432,
        "database": parts.path.strip("/").split("/", maxsplit=1)[0],
        "credentials": "configured" if parts.username or parts.password else "not_in_url",
    }


async def _asyncpg_connect(database_url: str) -> AsyncConnection:
    import asyncpg

    return await asyncpg.connect(
        dsn=database_url,
        timeout=5,
        command_timeout=120,
    )


def _migration_sql(migration: Migration) -> str:
    sql = migration.path.read_text(encoding="utf-8")
    lines = sql.splitlines()
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    last = next(
        (index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()),
        None,
    )
    if (
        first is not None
        and last is not None
        and lines[first].strip().upper() == "BEGIN;"
        and lines[last].strip().upper() == "COMMIT;"
    ):
        lines.pop(last)
        lines.pop(first)
    return "\n".join(lines).strip()


async def _table_names(connection: AsyncConnection) -> list[str]:
    rows = await connection.fetch(
        """
        SELECT tablename
        FROM pg_catalog.pg_tables
        WHERE schemaname = current_schema()
        ORDER BY tablename
        """
    )
    return [str(row["tablename"]) for row in rows]


async def _tracked_migrations(connection: AsyncConnection) -> dict[str, dict[str, Any]]:
    exists = await connection.fetchval(
        "SELECT to_regclass(format('%I.%I', current_schema(), "
        "'documind_schema_migrations')) IS NOT NULL"
    )
    if not exists:
        return {}
    rows = await connection.fetch(
        f"""
        SELECT migration_id, checksum_sha256, applied_at, execution_ms
        FROM {TRACKING_TABLE}
        ORDER BY migration_id
        """
    )
    return {
        str(row["migration_id"]): {
            "checksum": str(row["checksum_sha256"]),
            "applied_at": _json_datetime(row["applied_at"]),
            "execution_ms": int(row["execution_ms"]),
        }
        for row in rows
    }


def _json_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


async def inspect_or_migrate(
    database_url: str,
    *,
    execute: bool = False,
    connector: Connector | None = None,
) -> dict[str, Any]:
    """Connect, inspect the schema, and optionally apply pending migrations."""

    normalised_url = normalise_database_url(database_url)
    safe_config = safe_database_config(normalised_url)
    connect = connector or _asyncpg_connect
    connection = await connect(normalised_url)
    migration_rows: list[dict[str, Any]] = []
    lock_acquired = False
    try:
        select_one = await connection.fetchval("SELECT 1")
        if select_one != 1:
            raise RuntimeError("PostgreSQL SELECT 1 returned an unexpected result")
        version = str(await connection.fetchval("SHOW server_version"))
        database = str(await connection.fetchval("SELECT current_database()"))
        tables_before = await _table_names(connection)
        tracked = await _tracked_migrations(connection)

        for migration in MIGRATIONS:
            record = tracked.get(migration.migration_id)
            if record and record["checksum"] != migration.checksum:
                outcome = "checksum_mismatch"
            elif record:
                outcome = "already_applied"
            else:
                outcome = "pending"
            migration_rows.append(
                {
                    "migration_id": migration.migration_id,
                    "file": str(migration.path.relative_to(PROJECT_ROOT)),
                    "checksum_sha256": migration.checksum,
                    "outcome": outcome,
                    "applied_at": record["applied_at"] if record else None,
                    "execution_ms": record["execution_ms"] if record else None,
                    "required_tables_present": all(
                        table in tables_before for table in migration.required_tables
                    ),
                }
            )

        mismatches = [
            row for row in migration_rows if row["outcome"] == "checksum_mismatch"
        ]
        if execute and mismatches:
            return _result(
                status="blocked",
                mode="execute",
                safe_config=safe_config,
                version=version,
                database=database,
                tables=tables_before,
                migrations=migration_rows,
                error=(
                    "A tracked migration checksum changed; create a new migration instead of "
                    "modifying applied SQL."
                ),
            )

        if execute:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TRACKING_TABLE} (
                  migration_id TEXT PRIMARY KEY,
                  checksum_sha256 TEXT NOT NULL,
                  applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  execution_ms BIGINT NOT NULL CHECK (execution_ms >= 0)
                )
                """
            )
            await connection.execute("SELECT pg_advisory_lock(hashtext($1))", LOCK_NAME)
            lock_acquired = True
            tracked = await _tracked_migrations(connection)
            late_mismatches = [
                migration
                for migration in MIGRATIONS
                if migration.migration_id in tracked
                and tracked[migration.migration_id]["checksum"] != migration.checksum
            ]
            if late_mismatches:
                mismatch_ids = {migration.migration_id for migration in late_mismatches}
                for row in migration_rows:
                    if row["migration_id"] in mismatch_ids:
                        row["outcome"] = "checksum_mismatch"
                return _result(
                    status="blocked",
                    mode="execute",
                    safe_config=safe_config,
                    version=version,
                    database=database,
                    tables=tables_before,
                    migrations=migration_rows,
                    error=(
                        "A tracked migration checksum changed while waiting for the migration "
                        "lock; create a new migration instead of modifying applied SQL."
                    ),
                )
            for migration, row in zip(MIGRATIONS, migration_rows, strict=True):
                record = tracked.get(migration.migration_id)
                if record:
                    row.update(
                        outcome="already_applied",
                        applied_at=record["applied_at"],
                        execution_ms=record["execution_ms"],
                    )
                    continue
                started = time.perf_counter()
                async with connection.transaction():
                    await connection.execute(_migration_sql(migration))
                    execution_ms = max(0, round((time.perf_counter() - started) * 1000))
                    await connection.execute(
                        f"""
                        INSERT INTO {TRACKING_TABLE}
                          (migration_id, checksum_sha256, execution_ms)
                        VALUES ($1, $2, $3)
                        """,
                        migration.migration_id,
                        migration.checksum,
                        execution_ms,
                    )
                row.update(
                    outcome="applied",
                    applied_at=None,
                    execution_ms=execution_ms,
                )
            tracked_after = await _tracked_migrations(connection)
            for row in migration_rows:
                record = tracked_after.get(str(row["migration_id"]))
                if record:
                    row.update(
                        applied_at=record["applied_at"],
                        execution_ms=record["execution_ms"],
                    )

        tables_after = await _table_names(connection)
        required_tables = sorted(
            {table for migration in MIGRATIONS for table in migration.required_tables}
        )
        for migration, row in zip(MIGRATIONS, migration_rows, strict=True):
            row["required_tables_present"] = all(
                table in tables_after for table in migration.required_tables
            )
        missing = [table for table in required_tables if table not in tables_after]
        status = "executed" if execute else "dry_run"
        if missing and execute:
            status = "incomplete"
        return _result(
            status=status,
            mode="execute" if execute else "dry_run",
            safe_config=safe_config,
            version=version,
            database=database,
            tables=tables_after,
            migrations=migration_rows,
            missing_required=missing,
        )
    finally:
        if lock_acquired:
            await connection.execute("SELECT pg_advisory_unlock(hashtext($1))", LOCK_NAME)
        await connection.close()


def _result(
    *,
    status: str,
    mode: str,
    safe_config: dict[str, Any],
    version: str,
    database: str,
    tables: list[str],
    migrations: list[dict[str, Any]],
    missing_required: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    required = sorted({table for migration in MIGRATIONS for table in migration.required_tables})
    result: dict[str, Any] = {
        "status": status,
        "mode": mode,
        "connection": {
            "authenticated": True,
            "select_1": True,
            "safe_config": safe_config,
        },
        "server": {"version": version, "database": database},
        "tables": {
            "existing_count": len(tables),
            "existing": tables,
            "required": required,
            "missing_required": missing_required
            if missing_required is not None
            else [table for table in required if table not in tables],
        },
        "migrations": migrations,
    }
    if error:
        result["error"] = error
    return result


def _safe_failure(database_url: str | None, exc: Exception, *, execute: bool) -> dict[str, Any]:
    safe_config: dict[str, Any] = {"configured": bool(database_url)}
    if database_url:
        try:
            safe_config = safe_database_config(normalise_database_url(database_url))
        except (TypeError, ValueError):
            safe_config = {"configured": True, "scheme": "invalid"}
    return {
        "status": "error",
        "mode": "execute" if execute else "dry_run",
        "connection": {
            "authenticated": False,
            "select_1": False,
            "safe_config": safe_config,
        },
        "error": (
            "PostgreSQL operation failed "
            f"({type(exc).__name__}); credentials and driver messages are intentionally redacted."
        ),
        "migrations": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect PostgreSQL migrations (read-only unless --execute is supplied)."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply pending repository migrations and record their checksums.",
    )
    arguments = parser.parse_args()
    database_url = load_database_url()
    if not database_url:
        payload = {
            "status": "blocked",
            "mode": "execute" if arguments.execute else "dry_run",
            "connection": {
                "authenticated": False,
                "select_1": False,
                "safe_config": {"configured": False},
            },
            "error": "DATABASE_URL is not configured in backend/.env or the process environment.",
            "migrations": [],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    try:
        payload = asyncio.run(inspect_or_migrate(database_url, execute=arguments.execute))
    except Exception as exc:
        payload = _safe_failure(database_url, exc, execute=arguments.execute)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] in {"dry_run", "executed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
