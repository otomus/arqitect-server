"""Tests for arqitect.memory.warm — WarmMemory with SQLite episodic backend."""

import time

import pytest
import time_machine
from dirty_equals import IsInstance, IsPositiveInt
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from arqitect.memory.warm import WarmMemory, MAX_EPISODES


@pytest.fixture
def warm(tmp_memory_dir):
    """Create a WarmMemory backed by the temp directory from conftest."""
    return WarmMemory()


def _make_episode(
    task: str,
    nerve: str = "test_nerve",
    tool: str = "test_tool",
    success: bool = True,
    user_id: str = "",
    timestamp: float | None = None,
) -> dict:
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


# -- Record & Recall ----------------------------------------------------------


@pytest.mark.timeout(10)
class TestRecordAndRecall:
    """Recording episodes and recalling them by keyword."""

    def test_record_and_recall(self, warm):
        """A recorded episode is found when querying by its tool name."""
        warm.record(_make_episode("translate hello to French", tool="translate"))
        results = warm.recall("translate")
        assert len(results) >= 1
        assert results[0]["task"] == "translate hello to French"

    def test_recall_empty_database(self, warm):
        """An empty database returns no results for any query."""
        assert warm.recall("anything") == []

    def test_recall_with_limit(self, warm):
        """The limit parameter caps the number of returned results."""
        for i in range(10):
            warm.record(_make_episode(f"task {i}", tool="widget"))
        results = warm.recall("widget", limit=3)
        assert len(results) <= 3

    def test_recall_keyword_matching(self, warm):
        """Recall matches against task text and tool name."""
        warm.record(_make_episode("send email to bob", tool="email_sender"))
        warm.record(_make_episode("calculate tax", tool="tax_calc"))
        results = warm.recall("email")
        tasks = [r["task"] for r in results]
        assert "send email to bob" in tasks

    def test_recall_no_match_returns_empty(self, warm):
        """Query with zero keyword overlap returns no results."""
        results = warm.recall("xyzzyplugh")
        assert results == []

    def test_recalled_episode_has_expected_fields(self, warm):
        """Each recalled episode dict contains all stored fields."""
        warm.record(_make_episode("check fields", tool="checker"))
        results = warm.recall("checker")
        assert len(results) >= 1
        ep = results[0]
        assert ep["task"] == "check fields"
        assert ep["tool"] == "checker"
        assert ep["nerve"] == "test_nerve"
        assert ep["tokens"] == IsPositiveInt
        assert ep["timestamp"] == IsInstance(float)


# -- User Scoping -------------------------------------------------------------


@pytest.mark.timeout(10)
class TestUserScoping:
    """Episodes can be scoped to a specific user."""

    def test_recall_scoped_to_user(self, warm):
        """When user_id is given, only that user's episodes are returned."""
        warm.record(_make_episode("user1 task", tool="widget", user_id="u1"))
        warm.record(_make_episode("user2 task", tool="widget", user_id="u2"))
        results = warm.recall("widget", user_id="u1")
        user_ids = {r["user_id"] for r in results}
        assert user_ids == {"u1"}

    def test_recall_global_includes_all(self, warm):
        """With empty user_id, all users' episodes are returned."""
        warm.record(_make_episode("task a", tool="widget", user_id="u1"))
        warm.record(_make_episode("task b", tool="widget", user_id="u2"))
        results = warm.recall("widget", user_id="")
        assert len(results) == 2


# -- Recency Boost ------------------------------------------------------------


@pytest.mark.timeout(10)
class TestRecencyBoost:
    """Recent episodes receive a higher score in recall results."""

    @time_machine.travel("2026-03-20 12:00:00", tick=False)
    def test_recent_episode_scored_higher(self, warm):
        """An episode recorded now outranks one from 2 days ago."""
        now = time.time()
        warm.record(_make_episode(
            "old weather check", tool="weather",
            timestamp=now - 2 * 86400,
        ))
        warm.record(_make_episode(
            "new weather check", tool="weather",
            timestamp=now,
        ))
        results = warm.recall("weather check")
        assert len(results) >= 2
        assert results[0]["task"] == "new weather check"

    @time_machine.travel("2026-03-20 12:00:00", tick=False)
    def test_within_hour_boost(self, warm):
        """An episode from 30 minutes ago outranks one from 12 hours ago."""
        now = time.time()
        warm.record(_make_episode(
            "recent task", tool="widget",
            timestamp=now - 30 * 60,
        ))
        warm.record(_make_episode(
            "day old task", tool="widget",
            timestamp=now - 12 * 3600,
        ))
        results = warm.recall("widget task")
        assert results[0]["task"] == "recent task"

    @time_machine.travel("2026-03-20 12:00:00", tick=False)
    def test_hour_boundary_boost_values(self, warm):
        """Episodes < 1h get +2 boost, 1-24h get +1, >24h get +0.

        Verifies the ordering is consistent with the documented boost tiers.
        """
        now = time.time()
        # >24h old -- no boost
        warm.record(_make_episode(
            "ancient", tool="samename",
            timestamp=now - 48 * 3600,
        ))
        # 6h old -- +1 boost
        warm.record(_make_episode(
            "medium", tool="samename",
            timestamp=now - 6 * 3600,
        ))
        # 10 min old -- +2 boost
        warm.record(_make_episode(
            "fresh", tool="samename",
            timestamp=now - 10 * 60,
        ))
        results = warm.recall("samename")
        tasks = [r["task"] for r in results]
        assert tasks.index("fresh") < tasks.index("medium") < tasks.index("ancient")


# -- Pruning -------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestPruning:
    """Old episodes are pruned when the count exceeds MAX_EPISODES."""

    def test_prune_keeps_max_episodes(self, warm):
        """After inserting more than MAX_EPISODES, the DB has exactly MAX_EPISODES rows."""
        now = time.time()
        for i in range(MAX_EPISODES + 50):
            warm.record(_make_episode(
                f"task {i}", tool="widget",
                timestamp=now + i,
            ))
        count = warm.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert count == MAX_EPISODES

    def test_prune_removes_oldest(self, warm):
        """Pruning removes the oldest episodes by timestamp."""
        now = time.time()
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


# -- Episode Defaults ----------------------------------------------------------


@pytest.mark.timeout(10)
class TestEpisodeDefaults:
    """Missing fields in the episode dict fall back to sane defaults."""

    def test_missing_fields_use_defaults(self, warm):
        """A minimal episode dict gets default values for all optional fields."""
        warm.record({"task": "minimal"})
        row = warm.conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["task"] == "minimal"
        assert row["success"] == 1
        assert row["tokens"] == 0
        assert row["user_id"] == ""

    def test_failed_episode(self, warm):
        """success=False is stored as 0 in the database."""
        warm.record(_make_episode("broken", success=False))
        row = warm.conn.execute(
            "SELECT success FROM episodes ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["success"] == 0


# -- Property-based tests (Hypothesis) ----------------------------------------


@pytest.mark.timeout(30)
class TestPropertyBased:
    """Property-based tests using Hypothesis for fuzz-style coverage."""

    @given(
        task=st.text(min_size=1, max_size=200),
        nerve=st.text(min_size=1, max_size=50),
        tool=st.text(min_size=1, max_size=50),
        success=st.booleans(),
        tokens=st.integers(min_value=0, max_value=100_000),
    )
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_record_roundtrip_with_random_data(
        self, tmp_memory_dir, task, nerve, tool, success, tokens,
    ):
        """Any valid episode can be recorded and retrieved from the DB."""
        warm = WarmMemory()
        episode = {
            "task": task,
            "nerve": nerve,
            "tool": tool,
            "success": success,
            "tokens": tokens,
        }
        warm.record(episode)
        row = warm.conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["task"] == task
        assert row["nerve"] == nerve
        assert row["tool"] == tool
        assert row["success"] == (1 if success else 0)
        assert row["tokens"] == tokens
        warm.conn.close()

    @given(query=st.text(min_size=0, max_size=100))
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_recall_never_crashes_on_random_query(self, tmp_memory_dir, query):
        """Recall with arbitrary query strings never raises an exception."""
        warm = WarmMemory()
        warm.record(_make_episode("seed task", tool="seed"))
        results = warm.recall(query)
        assert isinstance(results, list)
        warm.conn.close()

    @given(nerve=st.from_regex(r"[a-z_]{1,30}", fullmatch=True))
    @settings(
        max_examples=20,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_recall_with_random_nerve_names(self, tmp_memory_dir, nerve):
        """Recording and recalling with random nerve names works without error."""
        warm = WarmMemory()
        warm.record(_make_episode(f"do {nerve}", nerve=nerve, tool=nerve))
        results = warm.recall(nerve)
        assert isinstance(results, list)
        warm.conn.close()
