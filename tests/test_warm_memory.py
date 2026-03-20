"""Tests for arqitect.memory.warm — WarmMemory with SQLite episodic backend."""

import time
from unittest.mock import patch

import pytest


@pytest.fixture
def warm(tmp_path):
    """Create a WarmMemory backed by a temporary SQLite database."""
    db_path = str(tmp_path / "episodes.db")
    with patch("arqitect.memory.warm._DB_PATH", db_path):
        from arqitect.memory.warm import WarmMemory
        return WarmMemory()


def _make_episode(task: str, nerve: str = "test_nerve", tool: str = "test_tool",
                  success: bool = True, user_id: str = "",
                  timestamp: float | None = None) -> dict:
    """Build an episode dict with sensible defaults."""
    return {
        "timestamp": timestamp if timestamp is not None else time.time(),
        "task": task,
        "nerve": nerve,
        "tool": tool,
        "args": {},
        "result_summary": f"result for {task}",
        "success": success,
        "tokens": 100,
        "user_id": user_id,
    }


# ── Record & Recall ──────────────────────────────────────────────────────────


class TestRecordAndRecall:
    def test_record_and_recall(self, warm):
        warm.record(_make_episode("translate hello to French", tool="translate"))
        results = warm.recall("translate")
        assert len(results) >= 1
        assert results[0]["task"] == "translate hello to French"

    def test_recall_empty_database(self, warm):
        assert warm.recall("anything") == []

    def test_recall_with_limit(self, warm):
        for i in range(10):
            warm.record(_make_episode(f"task {i}", tool="widget"))
        results = warm.recall("widget", limit=3)
        assert len(results) <= 3

    def test_recall_keyword_matching(self, warm):
        warm.record(_make_episode("send email to bob", tool="email_sender"))
        warm.record(_make_episode("calculate tax", tool="tax_calc"))
        results = warm.recall("email")
        tasks = [r["task"] for r in results]
        assert "send email to bob" in tasks

    def test_recall_no_match_returns_empty(self, warm):
        """Query with zero keyword overlap returns no results."""
        results = warm.recall("xyzzyplugh")
        assert results == []


# ── User Scoping ─────────────────────────────────────────────────────────────


class TestUserScoping:
    def test_recall_scoped_to_user(self, warm):
        warm.record(_make_episode("user1 task", tool="widget", user_id="u1"))
        warm.record(_make_episode("user2 task", tool="widget", user_id="u2"))
        results = warm.recall("widget", user_id="u1")
        user_ids = {r["user_id"] for r in results}
        assert user_ids == {"u1"}

    def test_recall_global_includes_all(self, warm):
        warm.record(_make_episode("task a", tool="widget", user_id="u1"))
        warm.record(_make_episode("task b", tool="widget", user_id="u2"))
        results = warm.recall("widget", user_id="")
        assert len(results) == 2


# ── Recency Boost ────────────────────────────────────────────────────────────


class TestRecencyBoost:
    def test_recent_episode_scored_higher(self, warm):
        now = time.time()
        # Old episode: 2 days ago
        warm.record(_make_episode(
            "old weather check", tool="weather",
            timestamp=now - 2 * 86400,
        ))
        # Recent episode: just now
        warm.record(_make_episode(
            "new weather check", tool="weather",
            timestamp=now,
        ))
        results = warm.recall("weather check")
        assert len(results) >= 2
        assert results[0]["task"] == "new weather check"

    def test_within_hour_boost(self, warm):
        now = time.time()
        warm.record(_make_episode(
            "recent task", tool="widget",
            timestamp=now - 30 * 60,  # 30 min ago
        ))
        warm.record(_make_episode(
            "day old task", tool="widget",
            timestamp=now - 12 * 3600,  # 12 hours ago
        ))
        results = warm.recall("widget task")
        assert results[0]["task"] == "recent task"


# ── Pruning ──────────────────────────────────────────────────────────────────


class TestPruning:
    def test_prune_keeps_max_episodes(self, warm):
        from arqitect.memory.warm import MAX_EPISODES
        now = time.time()
        for i in range(MAX_EPISODES + 50):
            warm.record(_make_episode(
                f"task {i}", tool="widget",
                timestamp=now + i,
            ))
        count = warm.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert count == MAX_EPISODES

    def test_prune_removes_oldest(self, warm):
        from arqitect.memory.warm import MAX_EPISODES
        now = time.time()
        # Record MAX_EPISODES + 5, so the first 5 get pruned
        for i in range(MAX_EPISODES + 5):
            warm.record(_make_episode(
                f"task {i}", tool="widget",
                timestamp=now + i,
            ))
        oldest = warm.conn.execute(
            "SELECT task FROM episodes ORDER BY timestamp ASC LIMIT 1"
        ).fetchone()
        # Tasks 0-4 should have been pruned; earliest remaining is task 5
        assert oldest["task"] == "task 5"


# ── Episode Defaults ─────────────────────────────────────────────────────────


class TestEpisodeDefaults:
    def test_missing_fields_use_defaults(self, warm):
        warm.record({"task": "minimal"})
        row = warm.conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["task"] == "minimal"
        assert row["success"] == 1
        assert row["tokens"] == 0
        assert row["user_id"] == ""

    def test_failed_episode(self, warm):
        warm.record(_make_episode("broken", success=False))
        row = warm.conn.execute(
            "SELECT success FROM episodes ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["success"] == 0
