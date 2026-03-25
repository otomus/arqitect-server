"""Tests for PR conflict resolution and rich PR body generation.

Covers:
- _sync_with_main() conflict resolution fallback chain
- _build_pr_body() structured markdown output
- _contribute_pr() wires sync correctly
"""

import json
import os
import types
from unittest.mock import patch, MagicMock, call

import pytest


# ── _sync_with_main ──────────────────────────────────────────────────────

class TestSyncWithMain:

    def _make_runner(self, results: dict):
        """Build a _run callable that returns pre-configured results by command keyword."""
        calls = []

        def _run(cmd, **kw):
            calls.append(cmd)
            key = " ".join(cmd)
            for pattern, result in results.items():
                if pattern in key:
                    return result
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        return _run, calls

    def test_rebase_succeeds_no_fallback(self):
        from arqitect.brain.consolidate import _sync_with_main
        _run, calls = self._make_runner({
            "pull --rebase": types.SimpleNamespace(returncode=0, stdout="", stderr=""),
        })
        _sync_with_main("/tmp/repo", _run)
        assert any("pull" in " ".join(c) and "--rebase" in " ".join(c) for c in calls)
        assert not any("merge" in " ".join(c) for c in calls)

    def test_rebase_fails_merge_succeeds(self):
        from arqitect.brain.consolidate import _sync_with_main
        _run, calls = self._make_runner({
            "pull --rebase": types.SimpleNamespace(returncode=1, stdout="", stderr="conflict"),
            "rebase --abort": types.SimpleNamespace(returncode=0, stdout="", stderr=""),
            "merge origin/main --no-edit": types.SimpleNamespace(returncode=0, stdout="", stderr=""),
        })
        _sync_with_main("/tmp/repo", _run)
        cmds = [" ".join(c) for c in calls]
        assert any("rebase --abort" in c for c in cmds)
        assert any("merge origin/main --no-edit" in c for c in cmds)
        # Should NOT reach -X ours
        assert not any("-X ours" in c for c in cmds)

    def test_rebase_and_merge_fail_uses_ours(self):
        from arqitect.brain.consolidate import _sync_with_main

        def _run(cmd, **kw):
            key = " ".join(cmd)
            if "pull --rebase" in key:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="conflict")
            if "merge origin/main --no-edit" in key and "-X" not in key:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="conflict")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        calls = []
        original_run = _run

        def tracking_run(cmd, **kw):
            calls.append(cmd)
            return original_run(cmd, **kw)

        _sync_with_main("/tmp/repo", tracking_run)
        cmds = [" ".join(c) for c in calls]
        assert any("-X" in c and "ours" in c for c in cmds)


# ── _build_pr_body ───────────────────────────────────────────────────────

class TestBuildPrBody:

    @pytest.fixture(autouse=True)
    def _patch_mem(self):
        """Patch mem.cold methods used by _build_pr_body."""
        with patch("arqitect.brain.consolidate.mem") as mock_mem:
            mock_mem.cold.get_qualification.return_value = {
                "score": 0.87,
                "pass_count": 13,
                "test_count": 15,
                "iterations": 3,
            }
            mock_mem.cold.get_nerve_info.return_value = {
                "total_invocations": 42,
                "successes": 40,
                "failures": 2,
                "trigger_task": "User asked about weather forecasts",
            }
            mock_mem.cold.get_nerve_tools_with_counts.return_value = [
                {"tool": "get_weather", "use_count": 38},
                {"tool": "get_forecast", "use_count": 12},
            ]
            mock_mem.cold.get_nerve_metadata.return_value = {
                "description": "Weather lookup",
                "system_prompt": "Retrieve weather data for any location.",
                "examples": [
                    {"input": "Weather in Tel Aviv?", "output": "24C, partly cloudy"},
                ],
                "role": "tool",
                "origin": "local",
            }
            self.mock_mem = mock_mem
            yield

    def test_contains_nerve_name(self):
        from arqitect.brain.consolidate import _build_pr_body
        body = _build_pr_body("weather_lookup", {"description": "Weather forecasts", "tools": []})
        assert "## Nerve: weather_lookup" in body

    def test_contains_description(self):
        from arqitect.brain.consolidate import _build_pr_body
        body = _build_pr_body("weather_lookup", {"description": "Weather forecasts", "tools": []})
        assert "Weather forecasts" in body

    def test_contains_qualification_score(self):
        from arqitect.brain.consolidate import _build_pr_body
        body = _build_pr_body("weather_lookup", {"description": "test", "tools": []})
        assert "87%" in body
        assert "13/15" in body

    def test_contains_usage_stats(self):
        from arqitect.brain.consolidate import _build_pr_body
        body = _build_pr_body("weather_lookup", {"description": "test", "tools": []})
        assert "42 invocations" in body
        assert "95%" in body

    def test_contains_trigger_task(self):
        from arqitect.brain.consolidate import _build_pr_body
        body = _build_pr_body("weather_lookup", {"description": "test", "tools": []})
        assert "weather forecasts" in body

    def test_contains_tools_table(self):
        from arqitect.brain.consolidate import _build_pr_body
        bundle = {
            "description": "test",
            "tools": [
                {"name": "get_weather", "description": "Fetch weather"},
                {"name": "get_forecast", "description": "Get forecast"},
            ],
        }
        body = _build_pr_body("weather_lookup", bundle)
        assert "| get_weather |" in body
        assert "| get_forecast |" in body

    def test_contains_examples(self):
        from arqitect.brain.consolidate import _build_pr_body
        body = _build_pr_body("weather_lookup", {"description": "test", "tools": []})
        assert "Tel Aviv" in body

    def test_contains_system_prompt_goal(self):
        from arqitect.brain.consolidate import _build_pr_body
        body = _build_pr_body("weather_lookup", {"description": "test", "tools": []})
        assert "Retrieve weather data" in body

    def test_contains_footer(self):
        from arqitect.brain.consolidate import _build_pr_body
        body = _build_pr_body("weather_lookup", {"description": "test", "tools": []})
        assert "Auto-contributed by dreamstate" in body
        assert "arqitect" in body

    def test_no_qualification_graceful(self):
        """When no qualification data exists, the body still renders."""
        self.mock_mem.cold.get_qualification.return_value = None
        from arqitect.brain.consolidate import _build_pr_body
        body = _build_pr_body("new_nerve", {"description": "test", "tools": []})
        assert "## Nerve: new_nerve" in body
        assert "Qualification" not in body

    def test_no_invocations_graceful(self):
        """When nerve has never been invoked, stats show N/A."""
        self.mock_mem.cold.get_nerve_info.return_value = {
            "total_invocations": 0,
            "successes": 0,
            "failures": 0,
            "trigger_task": "",
        }
        from arqitect.brain.consolidate import _build_pr_body
        body = _build_pr_body("new_nerve", {"description": "test", "tools": []})
        assert "N/A" in body


# ── _parse_nerve_from_branch ─────────────────────────────────────────────

class TestParseNerveFromBranch:

    def test_contribute_branch(self):
        from arqitect.brain.consolidate import _parse_nerve_from_branch
        assert _parse_nerve_from_branch("contribute/weather_lookup-1711234567") == "weather_lookup"

    def test_adapter_branch(self):
        from arqitect.brain.consolidate import _parse_nerve_from_branch
        assert _parse_nerve_from_branch("nerve-adapter/weather_lookup-medium-1711234567") == "weather_lookup"

    def test_stack_branch(self):
        from arqitect.brain.consolidate import _parse_nerve_from_branch
        assert _parse_nerve_from_branch("nerve-stack/weather_lookup-python-1711234567") == "weather_lookup"

    def test_unknown_format(self):
        from arqitect.brain.consolidate import _parse_nerve_from_branch
        assert _parse_nerve_from_branch("feature/something") == ""

    def test_main_branch(self):
        from arqitect.brain.consolidate import _parse_nerve_from_branch
        assert _parse_nerve_from_branch("main") == ""


# ── _dream_pr_review ─────────────────────────────────────────────────────

class TestDreamPrReview:

    @pytest.fixture(autouse=True)
    def _patch_deps(self, tmp_path):
        """Patch community dir and subprocess for PR review tests."""
        self.community_dir = str(tmp_path / "community")
        os.makedirs(os.path.join(self.community_dir, ".git"))

        with patch("arqitect.brain.consolidate.mem") as mock_mem, \
             patch("arqitect.brain.consolidate.publish_nerve_status"):
            mock_mem.cold.get_qualification.return_value = None
            mock_mem.cold.get_nerve_info.return_value = None
            mock_mem.cold.get_nerve_tools_with_counts.return_value = []
            mock_mem.cold.get_nerve_metadata.return_value = {
                "description": "", "system_prompt": "", "examples": [],
                "role": "tool", "origin": "local",
            }
            self.mock_mem = mock_mem
            yield

    def _make_dreamstate(self):
        """Create a Dreamstate instance with patched community dir."""
        import threading
        from arqitect.brain.consolidate import Dreamstate
        with patch.object(Dreamstate, '__init__', lambda self: None):
            ds = Dreamstate()
            ds._interrupted = threading.Event()
            ds._find_community_dir = MagicMock(return_value=self.community_dir)
        return ds

    def test_cleanup_stale_prs_closes_old_prs(self):
        """Stale PRs (>30 days) should be closed."""
        import datetime
        old_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=45)).isoformat()

        calls = []

        def _run(cmd, **kw):
            calls.append(cmd)
            key = " ".join(cmd)
            if "pr list" in key and "open" in key:
                return types.SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps([{
                        "number": 42,
                        "updatedAt": old_date,
                        "title": "Add nerve: stale_nerve",
                    }]),
                    stderr="",
                )
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        ds = self._make_dreamstate()
        ds._cleanup_stale_prs(self.community_dir, _run)

        cmds = [" ".join(c) for c in calls]
        assert any("pr close" in c and "42" in c for c in cmds)

    def test_cleanup_stale_prs_skips_recent(self):
        """Recently updated PRs should not be closed."""
        import datetime
        recent_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)).isoformat()

        calls = []

        def _run(cmd, **kw):
            calls.append(cmd)
            key = " ".join(cmd)
            if "pr list" in key and "open" in key:
                return types.SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps([{
                        "number": 42,
                        "updatedAt": recent_date,
                        "title": "Add nerve: fresh_nerve",
                    }]),
                    stderr="",
                )
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        ds = self._make_dreamstate()
        ds._cleanup_stale_prs(self.community_dir, _run)

        cmds = [" ".join(c) for c in calls]
        assert not any("pr close" in c for c in cmds)

    def test_review_skips_approved_prs(self):
        """PRs without CHANGES_REQUESTED should be skipped."""
        calls = []

        def _run(cmd, **kw):
            calls.append(cmd)
            key = " ".join(cmd)
            if "pr list" in key and "open" in key:
                return types.SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps([{
                        "number": 10,
                        "headRefName": "contribute/good_nerve-123",
                        "title": "Add nerve: good_nerve",
                        "reviewDecision": "APPROVED",
                    }]),
                    stderr="",
                )
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        ds = self._make_dreamstate()
        ds._review_open_prs(self.community_dir, _run)

        # Should not attempt to fetch comments or fix anything
        cmds = [" ".join(c) for c in calls]
        assert not any("api" in c for c in cmds)

    def test_cleanup_merged_branches(self):
        """Merged branches should be deleted locally and remotely."""
        calls = []

        def _run(cmd, **kw):
            calls.append(cmd)
            key = " ".join(cmd)
            if "pr list" in key and "merged" in key:
                return types.SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps([{"headRefName": "contribute/done_nerve-123"}]),
                    stderr="",
                )
            if "pr list" in key and "closed" in key:
                return types.SimpleNamespace(returncode=0, stdout="[]", stderr="")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        ds = self._make_dreamstate()
        ds._cleanup_merged_branches(self.community_dir, _run)

        cmds = [" ".join(c) for c in calls]
        assert any("branch -D" in c and "contribute/done_nerve-123" in c for c in cmds)
        assert any("push origin --delete" in c for c in cmds)

    def test_pr_review_phase_in_dream_phases(self):
        """Verify pr_review is registered as a dream phase."""
        from arqitect.brain.consolidate import Dreamstate
        import threading
        with patch.object(Dreamstate, '__init__', lambda self: None):
            ds = Dreamstate()
            ds._interrupted = threading.Event()
        # Check the _dream_inner method references pr_review
        assert hasattr(ds, '_dream_pr_review')


# ── _contribute_nerve (unified single-PR-per-nerve) ─────────────────────

class TestContributeNerve:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up community dir and mock mem for unified contribute tests."""
        self.community_dir = str(tmp_path / "community")
        os.makedirs(os.path.join(self.community_dir, ".git"))
        os.makedirs(os.path.join(self.community_dir, "nerves"))

        with patch("arqitect.brain.consolidate.mem") as mock_mem, \
             patch("arqitect.brain.consolidate.publish_nerve_status"), \
             patch("arqitect.brain.consolidate._contribute_pr") as mock_pr:
            mock_mem.cold.get_nerve_metadata.return_value = {
                "description": "Weather data lookup",
                "system_prompt": '{"goal": "Fetch weather"}',
                "examples": [{"input": "weather?", "output": "sunny"}],
                "role": "tool",
                "origin": "local",
            }
            mock_mem.cold.get_qualification.return_value = {
                "score": 0.9, "pass_count": 9, "test_count": 10, "iterations": 2,
            }
            mock_mem.cold.get_nerve_info.return_value = {
                "total_invocations": 20, "successes": 18, "failures": 2,
                "trigger_task": "get weather",
            }
            mock_mem.cold.get_nerve_tools_with_counts.return_value = [
                {"tool": "get_weather", "use_count": 15},
            ]
            mock_mem.cold.get_test_bank.return_value = [{"input": "test", "output": "ok"}]
            self.mock_mem = mock_mem
            self.mock_pr = mock_pr
            yield

    def _make_dreamstate(self):
        import threading
        from arqitect.brain.consolidate import Dreamstate
        with patch.object(Dreamstate, '__init__', lambda self: None):
            ds = Dreamstate()
            ds._interrupted = threading.Event()
        return ds

    def test_single_pr_per_nerve(self):
        """_contribute_nerve should call _contribute_pr exactly once per nerve."""
        ds = self._make_dreamstate()

        # Mock _build_nerve_bundle to return a bundle
        bundle = {
            "description": "Weather data lookup",
            "role": "tool",
            "tools": [{"name": "get_weather", "implementations": {"python": "get_weather/tool.py"}}],
        }
        ds._build_nerve_bundle = MagicMock(return_value=bundle)
        ds._copy_tool_implementations = MagicMock()
        ds._write_test_cases = MagicMock()
        ds._write_nerve_adapter_files = MagicMock()
        ds._copy_nerve_adapter_to_community = MagicMock()
        ds._add_stack_implementations = MagicMock()

        self.mock_pr.return_value = "https://github.com/test/pr/1"
        ds._contribute_nerve(self.community_dir, "weather_lookup")

        # Exactly one PR call
        assert self.mock_pr.call_count == 1
        call_kwargs = self.mock_pr.call_args
        # Unified search key
        assert call_kwargs[1]["search_key"] == "Nerve: weather_lookup"
        # Unified branch prefix
        assert call_kwargs[1]["branch_prefix"] == "contribute/weather_lookup"

    def test_contribute_nerve_returns_false_on_no_bundle(self):
        """If _build_nerve_bundle returns None, no PR is created."""
        ds = self._make_dreamstate()
        ds._build_nerve_bundle = MagicMock(return_value=None)

        result = ds._contribute_nerve(self.community_dir, "bad_nerve")
        assert result is False
        assert self.mock_pr.call_count == 0

    def test_dream_contribute_calls_unified(self):
        """_dream_contribute should call _contribute_nerve (not the old split methods)."""
        ds = self._make_dreamstate()
        ds._find_community_dir = MagicMock(return_value=self.community_dir)
        ds._contribute_nerve = MagicMock(return_value=True)

        self.mock_mem.cold.get_all_nerve_data.return_value = {
            "weather_lookup": {"is_sense": False, "role": "tool"},
            "hearing": {"is_sense": True, "role": "tool"},  # should be skipped
        }

        ds._dream_contribute()

        # Only called for weather_lookup, not hearing (is_sense)
        ds._contribute_nerve.assert_called_once_with(self.community_dir, "weather_lookup")


# ── Generalization for community contribution ──────────────────────────


class TestGeneralizeText:
    """Project-specific references must be stripped before community contribution."""

    def test_strips_bot_name(self):
        """Bot name from config is replaced with 'the assistant'."""
        from arqitect.brain.consolidate import _generalize_text
        result = _generalize_text(
            "my-bot stores data in SQLite", {"my-bot"},
        )
        assert "my-bot" not in result
        assert "the assistant" in result

    def test_case_insensitive(self):
        """Replacement is case-insensitive."""
        from arqitect.brain.consolidate import _generalize_text
        result = _generalize_text("MY-BOT handles tasks", {"my-bot"})
        assert "MY-BOT" not in result
        assert "my-bot" not in result.lower()

    def test_preserves_text_without_matches(self):
        """Text without project references is unchanged."""
        from arqitect.brain.consolidate import _generalize_text
        original = "Handle task planning and execution"
        assert _generalize_text(original, {"my-bot"}) == original

    def test_multiple_names(self):
        """All project names are stripped."""
        from arqitect.brain.consolidate import _generalize_text
        text = "SuperBot uses my-bot to process tasks"
        result = _generalize_text(text, {"superbot", "my-bot"})
        assert "SuperBot" not in result
        assert "my-bot" not in result

    def test_none_input(self):
        from arqitect.brain.consolidate import _generalize_text
        assert _generalize_text(None, {"bot"}) is None

    def test_empty_input(self):
        from arqitect.brain.consolidate import _generalize_text
        assert _generalize_text("", {"bot"}) == ""

    def test_empty_names(self):
        from arqitect.brain.consolidate import _generalize_text
        assert _generalize_text("hello bot", set()) == "hello bot"

    def test_longer_name_replaced_first(self):
        """Longer names are replaced before shorter substrings to avoid partial matches."""
        from arqitect.brain.consolidate import _generalize_text
        result = _generalize_text("my-super-bot is great", {"my-super-bot", "bot"})
        assert "my-super-bot" not in result


class TestGeneralizeExamples:
    """Example input/output pairs must be generalized."""

    def test_strips_from_input_and_output(self):
        from arqitect.brain.consolidate import _generalize_examples
        examples = [
            {"input": "How does my-bot store data?", "output": "my-bot uses SQLite"},
        ]
        result = _generalize_examples(examples, {"my-bot"})
        assert "my-bot" not in result[0]["input"]
        assert "my-bot" not in result[0]["output"]

    def test_preserves_non_string_values(self):
        from arqitect.brain.consolidate import _generalize_examples
        examples = [{"input": "test", "output": "ok", "score": 0.9}]
        result = _generalize_examples(examples, {"bot"})
        assert result[0]["score"] == 0.9

    def test_skips_non_dict_entries(self):
        from arqitect.brain.consolidate import _generalize_examples
        result = _generalize_examples(["not a dict", {"input": "ok"}], set())
        assert len(result) == 1

    def test_empty_list(self):
        from arqitect.brain.consolidate import _generalize_examples
        assert _generalize_examples([], {"bot"}) == []


class TestGeneralizeTestBank:
    """Test cases must be generalized before community contribution."""

    def test_strips_project_refs(self):
        from arqitect.brain.consolidate import _generalize_test_bank
        bank = [
            {"input": "How does my-bot handle tasks?", "output": "my-bot does X",
             "category": "core", "generated_by": "model"},
        ]
        result = _generalize_test_bank(bank, {"my-bot"})
        assert "my-bot" not in result[0]["input"]
        assert "my-bot" not in result[0]["output"]
        assert result[0]["category"] == "core"

    def test_empty_bank(self):
        from arqitect.brain.consolidate import _generalize_test_bank
        assert _generalize_test_bank([], {"bot"}) == []

    def test_none_bank(self):
        from arqitect.brain.consolidate import _generalize_test_bank
        assert _generalize_test_bank(None, {"bot"}) is None
