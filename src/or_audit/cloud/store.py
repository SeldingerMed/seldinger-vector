"""SQLite persistence for Vector Cloud job records."""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from or_audit.errors import TaskContractError

from .models import JobRecord, JobRequest, JobStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    request_json TEXT NOT NULL,
    provider_id TEXT NOT NULL DEFAULT '',
    artifact_path TEXT NOT NULL DEFAULT '',
    result_head TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    machine_name TEXT NOT NULL DEFAULT '',
    started_at TEXT,
    completed_at TEXT,
    provider_cost_micros INTEGER NOT NULL DEFAULT 0,
    runtime_seconds INTEGER NOT NULL DEFAULT 0,
    callback_token_hash TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs(created_at DESC);
"""


class JobStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            additions = {
                "callback_token_hash": "TEXT NOT NULL DEFAULT ''",
                "machine_name": "TEXT NOT NULL DEFAULT ''",
                "started_at": "TEXT",
                "completed_at": "TEXT",
                "provider_cost_micros": "INTEGER NOT NULL DEFAULT 0",
                "runtime_seconds": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, declaration in additions.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create(self, request: JobRequest) -> JobRecord:
        record = JobRecord.new(request)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO jobs
                   (id, created_at, updated_at, status, request_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    record.id,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.status.value,
                    request.model_dump_json(),
                ),
            )
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _record(row) if row is not None else None

    def list(self, *, limit: int = 100) -> tuple[JobRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return tuple(_record(row) for row in rows)

    def set_callback_token(self, job_id: str, token: str) -> None:
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET callback_token_hash = ?
                   WHERE id = ? AND status = ? AND callback_token_hash = ''""",
                (digest, job_id, JobStatus.QUEUED.value),
            )
            if cursor.rowcount != 1:
                raise TaskContractError(f"job {job_id!r} is not queued for callback setup")

    def verify_callback_token(self, job_id: str, token: str) -> bool:
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT callback_token_hash FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return (
            row is not None
            and bool(row["callback_token_hash"])
            and hmac.compare_digest(row["callback_token_hash"], digest)
        )

    def clear_callback_token(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE jobs SET callback_token_hash = '' WHERE id = ?", (job_id,))

    def transition(
        self,
        job_id: str,
        *,
        expected: tuple[JobStatus, ...],
        status: JobStatus,
        provider_id: str | None = None,
        artifact_path: str | None = None,
        result_head: str | None = None,
        error: str | None = None,
        machine_name: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        provider_cost_micros: int | None = None,
        runtime_seconds: int | None = None,
        callback_token: str | None = None,
        clear_callback: bool = False,
    ) -> JobRecord:
        if not expected:
            raise ValueError("transition requires at least one expected status")
        values: dict[str, object] = {
            "updated_at": datetime.now(UTC).isoformat(),
            "status": status.value,
        }
        if provider_id is not None:
            values["provider_id"] = provider_id
        if artifact_path is not None:
            values["artifact_path"] = artifact_path
        if result_head is not None:
            values["result_head"] = result_head
        if error is not None:
            values["error"] = error
        if machine_name is not None:
            values["machine_name"] = machine_name
        if started_at is not None:
            values["started_at"] = started_at.isoformat()
        if completed_at is not None:
            values["completed_at"] = completed_at.isoformat()
        if provider_cost_micros is not None:
            values["provider_cost_micros"] = provider_cost_micros
        if runtime_seconds is not None:
            values["runtime_seconds"] = runtime_seconds
        conditions = ["id = ?", f"status IN ({','.join('?' for _ in expected)})"]
        condition_values = [job_id, *(item.value for item in expected)]
        if callback_token is not None or clear_callback:
            values["callback_token_hash"] = ""
        if callback_token is not None:
            conditions.append("callback_token_hash = ?")
            condition_values.append(hashlib.sha256(callback_token.encode()).hexdigest())
        assignments = ", ".join(f"{field} = ?" for field in values)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {assignments} WHERE {' AND '.join(conditions)}",
                (*values.values(), *condition_values),
            )
            if cursor.rowcount != 1:
                raise TaskContractError(f"job {job_id!r} state changed before transition")
        record = self.get(job_id)
        if record is None:  # pragma: no cover - transition retains the primary key
            raise TaskContractError(f"unknown cloud job {job_id!r}")
        return record


def _record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        status=JobStatus(row["status"]),
        request=JobRequest.model_validate_json(row["request_json"]),
        provider_id=row["provider_id"],
        artifact_path=row["artifact_path"],
        result_head=row["result_head"],
        error=row["error"],
        machine_name=row["machine_name"],
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        provider_cost_micros=row["provider_cost_micros"],
        runtime_seconds=row["runtime_seconds"],
    )
