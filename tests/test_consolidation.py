"""Tests for nerve consolidation — winner selection, merging, and adapter migration.

Covers:
- Community nerves always win over fabricated ones
- Qualification score is the primary tiebreaker for fabricated nerves
- Tool and adapter migration during merges
- Qualified fabricated nerves are protected from fabricated winners
- Qualified fabricated nerves are absorbed by community winners
"""

import json
import os
from unittest.mock import patch

import pytest

from tests.conftest import make_nerve_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_consolidate_deps(mem):
    """Patch module-level dependencies for all consolidation tests."""
    with patch("arqitect.brain.consolidate.mem", mem), \
         patch("arqitect.brain.consolidate.publish_event"), \
         patch("arqitect.brain.consolidate.publish_nerve_status"), \
         patch("arqitect.brain.consolidate._llm_judge_same_purpose", return_value=True), \
         patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
         patch("arqitect.brain.synthesis.mem", mem), \
         patch("arqitect.brain.synthesis.publish_event"), \
         patch("arqitect.brain.synthesis.publish_nerve_status"):
        yield


# ---------------------------------------------------------------------------
# pick_winner
# ---------------------------------------------------------------------------

class TestPickWinner:
    """Winner selection in a cluster of similar nerves."""

    def test_community_nerve_wins_over_fabricated(self, mem):
        """A community nerve always wins regardless of score or tools."""
        cluster = [
            ("joke_nerve", "Generate jokes"),
            ("humor_nerve", "Funny stuff"),
        ]
        community_nerves = frozenset({"joke_nerve"})

        mem.cold.register_nerve("joke_nerve", "Generate jokes")
        mem.cold.register_nerve("humor_nerve", "Funny stuff")
        mem.cold.record_qualification("nerve", "humor_nerve", True, 0.99, 3, 10, 10)
        mem.cold.record_qualification("nerve", "joke_nerve", True, 0.50, 1, 5, 3)

        from arqitect.brain.consolidate import pick_winner
        winner, losers = pick_winner(cluster, community_nerves)

        assert winner == "joke_nerve"
        assert losers == ["humor_nerve"]

    def test_highest_score_wins_among_fabricated(self, mem):
        """Among fabricated nerves, highest qualification score wins."""
        cluster = [
            ("weather_nerve", "Weather data"),
            ("forecast_nerve", "Forecasts"),
        ]
        community_nerves = frozenset()

        mem.cold.register_nerve("weather_nerve", "Weather data")
        mem.cold.register_nerve("forecast_nerve", "Forecasts")
        mem.cold.record_qualification("nerve", "weather_nerve", True, 0.60, 2, 10, 6)
        mem.cold.record_qualification("nerve", "forecast_nerve", True, 0.85, 3, 10, 9)

        from arqitect.brain.consolidate import pick_winner
        winner, losers = pick_winner(cluster, community_nerves)

        assert winner == "forecast_nerve"
        assert losers == ["weather_nerve"]

    def test_tools_count_breaks_score_tie(self, mem):
        """When scores are equal, more tools wins."""
        cluster = [
            ("nerve_a", "Domain A"),
            ("nerve_b", "Domain B"),
        ]
        community_nerves = frozenset()

        mem.cold.register_nerve("nerve_a", "Domain A")
        mem.cold.register_nerve("nerve_b", "Domain B")
        mem.cold.record_qualification("nerve", "nerve_a", True, 0.80, 2, 10, 8)
        mem.cold.record_qualification("nerve", "nerve_b", True, 0.80, 2, 10, 8)
        mem.cold.add_nerve_tool("nerve_b", "tool_1")
        mem.cold.add_nerve_tool("nerve_b", "tool_2")

        from arqitect.brain.consolidate import pick_winner
        winner, losers = pick_winner(cluster, community_nerves)

        assert winner == "nerve_b"


# ---------------------------------------------------------------------------
# merge_nerve — tool and adapter migration
# ---------------------------------------------------------------------------

class TestMergeNerve:
    """Tool and adapter migration during merges."""

    def test_tools_migrate_to_winner(self, mem, nerves_dir):
        """Loser's tools are added to winner without duplicates."""
        mem.cold.register_nerve("winner", "Winner nerve")
        mem.cold.register_nerve("loser", "Loser nerve")
        mem.cold.add_nerve_tool("winner", "shared_tool")
        mem.cold.add_nerve_tool("loser", "shared_tool")
        mem.cold.add_nerve_tool("loser", "unique_tool")
        make_nerve_file(nerves_dir, "winner")
        make_nerve_file(nerves_dir, "loser")

        from arqitect.brain.consolidate import merge_nerve
        merge_nerve("winner", "loser", "Loser nerve")

        winner_tools = mem.cold.get_nerve_tools("winner")
        assert "shared_tool" in winner_tools
        assert "unique_tool" in winner_tools
        assert not os.path.isdir(os.path.join(nerves_dir, "loser"))

    def test_cache_files_migrate_without_overwrite(self, mem, nerves_dir, tmp_path):
        """Loser's community cache files are copied to winner without overwriting.

        Per-size-class context.json/meta.json files migrate for size classes
        the winner doesn't already have.
        """
        cache_dir = tmp_path / ".community" / "cache" / "nerves"

        # Winner has medium/context.json
        winner_cache = cache_dir / "winner" / "medium"
        winner_cache.mkdir(parents=True)
        (winner_cache / "context.json").write_text(json.dumps({"system_prompt": "winner"}))

        # Loser has medium/llama-8b/ and small/
        loser_medium_model = cache_dir / "loser" / "medium" / "llama-8b"
        loser_medium_model.mkdir(parents=True)
        (loser_medium_model / "context.json").write_text(json.dumps({"system_prompt": "loser model"}))
        loser_small = cache_dir / "loser" / "small"
        loser_small.mkdir(parents=True)
        (loser_small / "context.json").write_text(json.dumps({"system_prompt": "loser small"}))

        mem.cold.register_nerve("winner", "Winner")
        mem.cold.register_nerve("loser", "Loser")
        make_nerve_file(nerves_dir, "winner")
        make_nerve_file(nerves_dir, "loser")

        from arqitect.brain.consolidate import merge_nerve
        with patch("arqitect.brain.community._cache_dir", return_value=str(tmp_path / ".community" / "cache")):
            merge_nerve("winner", "loser", "Loser")

        # Winner's medium/context.json untouched
        assert json.loads((winner_cache / "context.json").read_text())["system_prompt"] == "winner"
        # Loser's model-specific dir migrated into winner
        migrated_model = cache_dir / "winner" / "medium" / "llama-8b" / "context.json"
        assert migrated_model.exists()
        assert json.loads(migrated_model.read_text())["system_prompt"] == "loser model"
        # Loser's small/ dir migrated entirely
        migrated_small = cache_dir / "winner" / "small" / "context.json"
        assert migrated_small.exists()
        assert json.loads(migrated_small.read_text())["system_prompt"] == "loser small"


# ---------------------------------------------------------------------------
# consolidate_nerves — community absorption
# ---------------------------------------------------------------------------

class TestConsolidateWithCommunity:
    """Community nerves absorb overlapping fabricated nerves."""

    def test_community_winner_absorbs_qualified_fabricated(
        self, mem, nerves_dir, tmp_path,
    ):
        """A qualified fabricated nerve is still merged into a community winner."""
        # Set up community manifest
        cache_dir = tmp_path / ".community" / "cache"
        cache_dir.mkdir(parents=True)
        manifest = {
            "nerves": {"joke_nerve": {"description": "Jokes", "role": "creative", "tools": []}},
            "tools": {}, "mcps": {}, "adapters": {},
        }
        manifest_path = str(cache_dir / "manifest.json")
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

        mem.cold.register_nerve("joke_nerve", "Generate funny jokes and humor")
        mem.cold.register_nerve("humor_nerve", "Generate funny jokes and humor content")
        mem.cold.record_qualification("nerve", "humor_nerve", True, 0.99, 5, 20, 20)
        mem.cold.add_nerve_tool("humor_nerve", "pun_generator")
        make_nerve_file(nerves_dir, "joke_nerve")
        make_nerve_file(nerves_dir, "humor_nerve")

        from arqitect.brain.consolidate import consolidate_nerves

        fake_clusters = [[
            ("joke_nerve", "Generate funny jokes and humor"),
            ("humor_nerve", "Generate funny jokes and humor content"),
        ]]
        with patch("arqitect.brain.community._manifest_path", return_value=manifest_path), \
             patch("arqitect.brain.consolidate.find_nerve_clusters", return_value=fake_clusters):
            result = consolidate_nerves()

        assert result["merged"] == 1
        joke_tools = mem.cold.get_nerve_tools("joke_nerve")
        assert "pun_generator" in joke_tools
        assert not os.path.isdir(os.path.join(nerves_dir, "humor_nerve"))

    def test_fabricated_winner_skips_qualified_loser(self, mem, nerves_dir):
        """A fabricated winner cannot absorb a qualified fabricated loser."""
        mem.cold.register_nerve("weather_nerve", "Weather data")
        mem.cold.register_nerve("forecast_nerve", "Weather forecasts")
        mem.cold.record_qualification("nerve", "weather_nerve", True, 0.90, 3, 10, 9)
        mem.cold.record_qualification("nerve", "forecast_nerve", True, 0.96, 5, 20, 19)
        make_nerve_file(nerves_dir, "weather_nerve")
        make_nerve_file(nerves_dir, "forecast_nerve")

        from arqitect.brain.consolidate import consolidate_nerves

        with patch("arqitect.brain.community._load_cached_manifest", return_value=None):
            fake_clusters = [[
                ("forecast_nerve", "Weather forecasts"),
                ("weather_nerve", "Weather data"),
            ]]
            with patch("arqitect.brain.consolidate.find_nerve_clusters", return_value=fake_clusters):
                result = consolidate_nerves()

        # forecast_nerve wins by score but weather_nerve is qualified — skip merge
        assert result["merged"] == 0
        assert os.path.isdir(os.path.join(nerves_dir, "weather_nerve"))


# ---------------------------------------------------------------------------
# Minimum test coverage gate
# ---------------------------------------------------------------------------

class TestMinimumCoverage:
    """Scores from partial test runs must not be trusted."""

    def test_sufficient_coverage(self):
        """80%+ coverage passes the gate."""
        from arqitect.brain.consolidate import _has_sufficient_coverage
        assert _has_sufficient_coverage(80, 100) is True
        assert _has_sufficient_coverage(100, 100) is True
        assert _has_sufficient_coverage(10, 10) is True

    def test_insufficient_coverage(self):
        """Below 80% coverage fails the gate."""
        from arqitect.brain.consolidate import _has_sufficient_coverage
        assert _has_sufficient_coverage(1, 100) is False
        assert _has_sufficient_coverage(79, 100) is False
        assert _has_sufficient_coverage(0, 10) is False

    def test_empty_test_bank(self):
        """No tests means no coverage."""
        from arqitect.brain.consolidate import _has_sufficient_coverage
        assert _has_sufficient_coverage(0, 0) is False
        assert _has_sufficient_coverage(5, 0) is False

    def test_save_final_score_skips_low_coverage(self, mem):
        """_save_final_score does not record when coverage is too low."""
        from arqitect.brain.consolidate import _save_final_score, _ImprovementState

        mem.cold.register_nerve("test_nerve", "Test")
        state = _ImprovementState("test_nerve", "Test", 0.0)
        # Simulate 1 test out of 100 scoring perfectly
        state.update(1.0, [{"score": 1.0, "passed": True}])

        _save_final_score("test_nerve", state, 0.0, 0.95, 5, total_tests=100)

        qual = mem.cold.get_qualification("nerve", "test_nerve")
        assert qual is None  # Score should NOT have been saved

    def test_save_final_score_records_with_full_coverage(self, mem):
        """_save_final_score records when coverage is sufficient."""
        from arqitect.brain.consolidate import _save_final_score, _ImprovementState

        mem.cold.register_nerve("test_nerve", "Test")
        state = _ImprovementState("test_nerve", "Test", 0.0)
        results = [{"score": 0.9, "passed": True}] * 90
        state.update(0.9, results)

        _save_final_score("test_nerve", state, 0.0, 0.95, 5, total_tests=100)

        qual = mem.cold.get_qualification("nerve", "test_nerve")
        assert qual is not None
        assert qual["score"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# last_invoked_at tracking
# ---------------------------------------------------------------------------

class TestLastInvokedAt:
    """Invocation timestamp tracking for usage-based priority."""

    def test_starts_as_none(self, mem):
        """New nerves have no last_invoked_at."""
        mem.cold.register_nerve("fresh_nerve", "Fresh")
        assert mem.cold.get_last_invoked_at("fresh_nerve") is None

    def test_set_on_invocation(self, mem):
        """Recording an invocation sets last_invoked_at."""
        mem.cold.register_nerve("used_nerve", "Used")
        mem.cold.record_nerve_invocation("used_nerve", success=True)
        assert mem.cold.get_last_invoked_at("used_nerve") is not None

    def test_updates_on_each_invocation(self, mem):
        """Each invocation updates the timestamp."""
        mem.cold.register_nerve("busy_nerve", "Busy")
        mem.cold.record_nerve_invocation("busy_nerve", success=True)
        first = mem.cold.get_last_invoked_at("busy_nerve")
        mem.cold.record_nerve_invocation("busy_nerve", success=False)
        second = mem.cold.get_last_invoked_at("busy_nerve")
        assert second >= first


# ---------------------------------------------------------------------------
# Work queue priority
# ---------------------------------------------------------------------------

class TestWorkQueuePriority:
    """Recently-used nerves are tuned before dormant ones."""

    def test_recently_used_nerves_come_first(self, mem, nerves_dir):
        """Nerves with last_invoked_at sort before those without."""
        mem.cold.register_nerve("dormant_nerve", "Dormant")
        mem.cold.register_nerve("active_nerve", "Active")
        mem.cold.record_nerve_invocation("active_nerve", success=True)
        make_nerve_file(nerves_dir, "dormant_nerve")
        make_nerve_file(nerves_dir, "active_nerve")

        from arqitect.brain.consolidate import _build_work_queue
        with patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir):
            queue = _build_work_queue()

        names = [item["name"] for item in queue]
        assert "active_nerve" in names
        assert "dormant_nerve" in names
        assert names.index("active_nerve") < names.index("dormant_nerve")
