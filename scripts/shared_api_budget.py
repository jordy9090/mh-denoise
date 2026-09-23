#!/usr/bin/env python3
"""Process-safe request and USD budget for offline paid-data workers.

Every attempted paid request reserves its configured worst-case token cost in a
shared SQLite ledger before dispatch. Successful responses settle to reported
usage; failed/ambiguous dispatches retain the reservation conservatively.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetConfig:
    max_requests: int
    max_usd: float
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float

    def validate(self) -> None:
        if self.max_requests < 1 or self.max_usd <= 0:
            raise ValueError("Request and USD caps must be positive")
        if self.input_usd_per_million_tokens < 0 or self.output_usd_per_million_tokens < 0:
            raise ValueError("Token prices cannot be negative")


class SharedApiBudget:
    def __init__(self, ledger: str | Path, config: BudgetConfig) -> None:
        config.validate()
        self.ledger = Path(ledger).resolve()
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.ledger, timeout=30, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        expected = json.dumps(asdict(self.config), sort_keys=True, separators=(",", ":"))
        for attempt in range(10):
            try:
                with self._connect() as connection:
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), config_json TEXT NOT NULL)")
                    connection.execute(
                        """CREATE TABLE IF NOT EXISTS calls (
                        request_id TEXT PRIMARY KEY,
                        worker_id TEXT NOT NULL,
                        purpose TEXT NOT NULL,
                        status TEXT NOT NULL,
                        max_input_tokens INTEGER NOT NULL,
                        max_output_tokens INTEGER NOT NULL,
                        reserved_usd REAL NOT NULL,
                        actual_input_tokens INTEGER,
                        actual_output_tokens INTEGER,
                        actual_usd REAL,
                        failure_reason TEXT,
                        created_at REAL NOT NULL,
                        settled_at REAL
                        )"""
                    )
                    row = connection.execute("SELECT config_json FROM settings WHERE id=1").fetchone()
                    if row is None:
                        connection.execute("INSERT INTO settings(id, config_json) VALUES(1, ?)", (expected,))
                    elif row[0] != expected:
                        raise ValueError("Existing shared budget ledger has different immutable settings")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 9:
                    raise
                time.sleep(0.02 * (attempt + 1))

    def estimate_usd(self, input_tokens: int, output_tokens: int) -> float:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        return (
            input_tokens * self.config.input_usd_per_million_tokens
            + output_tokens * self.config.output_usd_per_million_tokens
        ) / 1_000_000.0

    def reserve(
        self,
        *,
        max_input_tokens: int,
        max_output_tokens: int,
        worker_id: str,
        purpose: str,
        request_id: str | None = None,
    ) -> str:
        request_id = request_id or uuid.uuid4().hex
        reserved = self.estimate_usd(max_input_tokens, max_output_tokens)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            calls, committed = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status IN ('settled','settled_overrun') THEN actual_usd ELSE reserved_usd END), 0.0) FROM calls"
            ).fetchone()
            if calls + 1 > self.config.max_requests:
                connection.execute("ROLLBACK")
                raise BudgetExceeded(f"request cap exceeded: {calls}+1 > {self.config.max_requests}")
            if float(committed) + reserved > self.config.max_usd + 1e-12:
                connection.execute("ROLLBACK")
                raise BudgetExceeded(
                    f"USD cap exceeded: {float(committed):.8f}+{reserved:.8f} > {self.config.max_usd:.8f}"
                )
            connection.execute(
                "INSERT INTO calls VALUES (?, ?, ?, 'reserved', ?, ?, ?, NULL, NULL, NULL, NULL, ?, NULL)",
                (request_id, worker_id, purpose, max_input_tokens, max_output_tokens, reserved, time.time()),
            )
            connection.execute("COMMIT")
        return request_id

    def settle(self, request_id: str, *, actual_input_tokens: int, actual_output_tokens: int) -> None:
        actual = self.estimate_usd(actual_input_tokens, actual_output_tokens)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, max_input_tokens, max_output_tokens FROM calls WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None or row[0] != "reserved":
                connection.execute("ROLLBACK")
                raise RuntimeError(f"Unknown or non-reserved request: {request_id}")
            status = (
                "settled_overrun"
                if actual_input_tokens > row[1] or actual_output_tokens > row[2]
                else "settled"
            )
            connection.execute(
                "UPDATE calls SET status=?, actual_input_tokens=?, actual_output_tokens=?, actual_usd=?, settled_at=? WHERE request_id=?",
                (status, actual_input_tokens, actual_output_tokens, actual, time.time(), request_id),
            )
            connection.execute("COMMIT")

    def fail(self, request_id: str, reason: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM calls WHERE request_id=?", (request_id,)).fetchone()
            if row is None or row[0] != "reserved":
                connection.execute("ROLLBACK")
                raise RuntimeError(f"Unknown or non-reserved request: {request_id}")
            connection.execute(
                "UPDATE calls SET status='failed_reserved', failure_reason=?, settled_at=? WHERE request_id=?",
                (reason[:1000], time.time(), request_id),
            )
            connection.execute("COMMIT")

    def status(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*), COALESCE(SUM(reserved_usd),0), COALESCE(SUM(actual_usd),0) FROM calls GROUP BY status"
            ).fetchall()
            committed = connection.execute(
                "SELECT COALESCE(SUM(CASE WHEN status IN ('settled','settled_overrun') THEN actual_usd ELSE reserved_usd END), 0.0) FROM calls"
            ).fetchone()[0]
        return {
            "ledger": str(self.ledger),
            "config": asdict(self.config),
            "calls": sum(row[1] for row in rows),
            "committed_usd": float(committed),
            "by_status": {
                row[0]: {"calls": row[1], "reserved_usd": row[2], "actual_usd": row[3]}
                for row in rows
            },
        }
