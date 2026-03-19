"""
Monitoring Memory — dedicated SQLite database for usage tracking.

Completely separate from the user-facing knowledge.db.
Stores per-call logs with timestamps and latency for time-series analysis.
"""

import os
import sqlite3
import threading
import time

from arqitect.config.loader import get_memory_dir

_DB_PATH = os.path.join(get_memory_dir(), "monitoring.db")


def _ensure_db(conn: sqlite3.Connection):
    """Create monitoring tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS call_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            subject_type TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            success INTEGER NOT NULL,
            latency_ms REAL DEFAULT 0,
            error_message TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_call_log_type_name
            ON call_log(subject_type, subject_name);
        CREATE INDEX IF NOT EXISTS idx_call_log_timestamp
            ON call_log(timestamp);
    """)
    conn.commit()


class MonitoringMemory:
    """Isolated database for usage tracking — never touches knowledge.db."""

    def __init__(self):
        self.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        _ensure_db(self.conn)

    def record_call(self, subject_type: str, subject_name: str,
                    success: bool, latency_ms: float = 0,
                    error_message: str = ""):
        """Log a single nerve or tool call.

        Args:
            subject_type: 'nerve' or 'tool'.
            subject_name: Name of the nerve or tool.
            success: Whether the call succeeded.
            latency_ms: Call duration in milliseconds.
            error_message: Error details on failure (truncated to 500 chars).
        """
        with self._lock:
            self.conn.execute(
                "INSERT INTO call_log (timestamp, subject_type, subject_name, "
                "success, latency_ms, error_message) VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), subject_type, subject_name,
                 int(success), latency_ms, error_message[:500]),
            )
            self.conn.commit()

    def get_usage_report(self, since: float = 0) -> dict:
        """Build a usage report aggregated from call_log.

        Args:
            since: Unix timestamp. Only include calls after this time.
                   Defaults to 0 (all time).

        Returns:
            Dict with 'nerves', 'tools', and 'mcps' keys, each a list of
            dicts with name, total, successes, failures, error_rate, avg_latency_ms.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT subject_type, subject_name, "
                "COUNT(*) as total, "
                "SUM(success) as successes, "
                "SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures, "
                "AVG(latency_ms) as avg_latency_ms, "
                "MAX(timestamp) as last_called_at "
                "FROM call_log WHERE timestamp > ? "
                "GROUP BY subject_type, subject_name "
                "ORDER BY total DESC",
                (since,),
            ).fetchall()

        nerves = []
        tools = []
        mcps = []
        for r in rows:
            total = r["total"]
            failures = r["failures"] or 0
            entry = {
                "name": r["subject_name"],
                "total": total,
                "successes": r["successes"] or 0,
                "failures": failures,
                "error_rate": round(failures / total, 4) if total else 0,
                "avg_latency_ms": round(r["avg_latency_ms"] or 0, 1),
                "last_called_at": r["last_called_at"],
            }
            if r["subject_type"] == "nerve":
                nerves.append(entry)
            elif r["subject_type"] == "mcp":
                mcps.append(entry)
            else:
                tools.append(entry)

        return {"nerves": nerves, "tools": tools, "mcps": mcps}

    def get_error_details(self, subject_type: str, subject_name: str,
                          limit: int = 10) -> list[dict]:
        """Get recent error details for a specific nerve or tool.

        Args:
            subject_type: 'nerve' or 'tool'.
            subject_name: Name of the nerve or tool.
            limit: Max number of errors to return.

        Returns:
            List of dicts with timestamp, error_message, latency_ms.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT timestamp, error_message, latency_ms FROM call_log "
                "WHERE subject_type = ? AND subject_name = ? AND success = 0 "
                "ORDER BY timestamp DESC LIMIT ?",
                (subject_type, subject_name, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def prune(self, older_than_days: int = 30):
        """Delete call logs older than the given number of days.

        Args:
            older_than_days: Remove entries older than this many days.
        """
        cutoff = time.time() - (older_than_days * 86400)
        with self._lock:
            self.conn.execute(
                "DELETE FROM call_log WHERE timestamp < ?", (cutoff,),
            )
            self.conn.commit()
