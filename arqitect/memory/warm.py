"""
Warm Memory — SQLite episodic memory.
Stores task execution episodes for recall and pattern matching.
Uses matching.py for keyword scoring (no vector DB needed).
"""

import json
import os
import sqlite3
import time

from arqitect.config.loader import get_memory_dir

_DB_PATH = os.path.join(get_memory_dir(), "episodes.db")
MAX_EPISODES = 500


def _ensure_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            task TEXT NOT NULL,
            nerve TEXT,
            tool TEXT,
            args TEXT,
            result_summary TEXT,
            success INTEGER DEFAULT 1,
            tokens INTEGER DEFAULT 0,
            user_id TEXT DEFAULT ''
        )
    """)
    conn.commit()


class WarmMemory:
    def __init__(self):
        self.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        _ensure_db(self.conn)
        # Migrate: add user_id column if missing
        try:
            self.conn.execute("SELECT user_id FROM episodes LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE episodes ADD COLUMN user_id TEXT DEFAULT ''")
            self.conn.commit()

    def record(self, episode: dict):
        """Insert an episode and prune old ones."""
        self.conn.execute(
            "INSERT INTO episodes (timestamp, task, nerve, tool, args, result_summary, success, tokens, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode.get("timestamp", time.time()),
                episode.get("task", ""),
                episode.get("nerve"),
                episode.get("tool"),
                json.dumps(episode.get("args", {})),
                episode.get("result_summary", ""),
                1 if episode.get("success", True) else 0,
                episode.get("tokens", 0),
                episode.get("user_id", ""),
            ),
        )
        self.conn.commit()
        self._prune()

    def recall(self, query: str, limit: int = 5, user_id: str = "") -> list[dict]:
        """Find relevant past episodes using keyword scoring with recency boost."""
        from arqitect.matching import match_score

        if user_id:
            rows = self.conn.execute(
                "SELECT * FROM episodes WHERE user_id=? ORDER BY timestamp DESC LIMIT 100",
                (user_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()

        now = time.time()
        scored = []
        for row in rows:
            text = f"{row['task']} {row['nerve'] or ''} {row['tool'] or ''}"
            base = match_score(query, row["tool"] or "", text)
            # Recency boost: episodes in last hour get +2, last day +1
            age_hours = (now - row["timestamp"]) / 3600
            if age_hours < 1:
                base += 2.0
            elif age_hours < 24:
                base += 1.0
            if base > 0:
                scored.append((dict(row), base))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:limit]]

    def _prune(self):
        """Keep only the most recent MAX_EPISODES."""
        count = self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        if count > MAX_EPISODES:
            self.conn.execute(
                "DELETE FROM episodes WHERE id IN "
                "(SELECT id FROM episodes ORDER BY timestamp ASC LIMIT ?)",
                (count - MAX_EPISODES,),
            )
            self.conn.commit()
