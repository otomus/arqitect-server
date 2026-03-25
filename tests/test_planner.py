"""Tests for arqitect.brain.planner — on-the-fly nerve chain composition."""

import json
from unittest.mock import patch

import pytest

from arqitect.brain.planner import (
    _candidates_to_steps,
    _filter_candidates,
    _find_gaps,
    _sort_into_sequence,
    _sort_pairwise,
    _sort_single_pass,
    _validate_steps,
    compose_chain,
)
from arqitect.types import Action
from tests.conftest import FakeLLM


# ── Helpers ──────────────────────────────────────────────────────────────────


def _catalog() -> dict[str, str]:
    """Build a small nerve catalog for testing."""
    return {
        "touch": "file and shell operations",
        "sight": "image analysis and screenshots",
        "hearing": "audio input and output",
        "communication": "formatting and tone adjustment",
        "awareness": "identity and capabilities",
        "code_analyzer": "static analysis of source code",
        "test_runner": "runs test suites and reports results",
    }


def _ranked(names: list[str]) -> list[tuple[str, float]]:
    """Build ranked candidates list from nerve names (descending scores)."""
    return [(name, 10.0 - i) for i, name in enumerate(names)]


# ── TestFilterCandidates ─────────────────────────────────────────────────────


@pytest.mark.timeout(10)
class TestFilterCandidates:
    """Tests for _filter_candidates — hybrid keyword+embedding matching."""

    def test_returns_ranked_pairs(self):
        """match_nerves result is passed through and capped."""
        mock_ranked = _ranked(["touch", "code_analyzer", "sight"])
        with patch("arqitect.brain.planner.match_nerves", return_value=mock_ranked):
            result = _filter_candidates("analyze code", _catalog())

        assert len(result) == 3
        assert result[0][0] == "touch"

    def test_caps_at_max_candidates(self):
        """Result is capped to MAX_CANDIDATES."""
        many = _ranked([f"nerve_{i}" for i in range(20)])
        with patch("arqitect.brain.planner.match_nerves", return_value=many):
            result = _filter_candidates("task", _catalog())

        assert len(result) <= 15

    def test_empty_catalog_returns_empty(self):
        with patch("arqitect.brain.planner.match_nerves", return_value=[]):
            result = _filter_candidates("task", {})

        assert result == []

    def test_no_matches_returns_empty(self):
        with patch("arqitect.brain.planner.match_nerves", return_value=[]):
            result = _filter_candidates("completely unrelated", _catalog())

        assert result == []


# ── TestSortIntoSequence ─────────────────────────────────────────────────────


@pytest.mark.timeout(10)
class TestSortIntoSequence:
    """Tests for _sort_into_sequence — LLM-based ordering."""

    def test_single_pass_for_medium(self):
        """Medium models use single-pass strategy."""
        llm_response = json.dumps([
            {"nerve": "code_analyzer", "args": "analyze the code"},
            {"nerve": "touch", "args": "write results"},
        ])
        fake = FakeLLM([("Available nerves", llm_response)])

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value=[
                 {"nerve": "code_analyzer", "args": "analyze the code"},
                 {"nerve": "touch", "args": "write results"},
             ]):
            result = _sort_into_sequence(
                "analyze and fix code",
                _ranked(["code_analyzer", "touch", "sight"]),
                _catalog(),
                project_facts=None,
                size_class="medium",
            )

        assert len(result) == 2
        assert result[0]["nerve"] == "code_analyzer"
        assert result[1]["nerve"] == "touch"

    def test_pairwise_for_small(self):
        """Small models use pairwise strategy with cap."""
        llm_response = json.dumps([
            {"nerve": "touch", "args": "run command"},
        ])
        fake = FakeLLM([("Nerves:", llm_response)])

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value=[
                 {"nerve": "touch", "args": "run command"},
             ]):
            result = _sort_into_sequence(
                "run a shell command",
                _ranked(["touch", "sight"]),
                _catalog(),
                project_facts=None,
                size_class="small",
            )

        assert len(result) == 1
        assert result[0]["nerve"] == "touch"

    def test_tinylm_uses_pairwise(self):
        """tinylm also routes to pairwise."""
        candidates = _ranked(["touch"])
        catalog = _catalog()

        with patch("arqitect.brain.planner._sort_pairwise", return_value=[{"nerve": "touch", "args": "x"}]) as mock_pw:
            _sort_into_sequence("task", candidates, catalog, None, "tinylm")

        mock_pw.assert_called_once()

    def test_none_size_class_uses_single_pass(self):
        """Default (None) uses single-pass."""
        candidates = _ranked(["touch"])
        catalog = _catalog()

        with patch("arqitect.brain.planner._sort_single_pass", return_value=[{"nerve": "touch", "args": "x"}]) as mock_sp:
            _sort_into_sequence("task", candidates, catalog, None, None)

        mock_sp.assert_called_once()

    def test_project_facts_included_in_prompt(self):
        """Project facts are passed through to the LLM prompt."""
        fake = FakeLLM([("Available nerves", '[{"nerve": "touch", "args": "build"}]')])

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value=[
                 {"nerve": "touch", "args": "build"},
             ]):
            _sort_single_pass(
                "build project",
                _ranked(["touch"]),
                _catalog(),
                project_facts={"language": "python", "framework": "flask"},
            )

        assert len(fake.prompts_containing("language=python")) == 1


# ── TestSortSinglePass ───────────────────────────────────────────────────────


@pytest.mark.timeout(10)
class TestSortSinglePass:
    """Tests for _sort_single_pass edge cases."""

    def test_llm_returns_dict_with_steps_key(self):
        """Handle LLM returning {steps: [...]} instead of [...]."""
        fake = FakeLLM([("Available nerves", '{"steps": [{"nerve": "touch", "args": "go"}]}')])

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value={
                 "steps": [{"nerve": "touch", "args": "go"}],
             }):
            result = _sort_single_pass("task", _ranked(["touch"]), _catalog(), None)

        assert len(result) == 1
        assert result[0]["nerve"] == "touch"

    def test_llm_returns_garbage(self):
        """LLM returning non-JSON results in empty chain."""
        fake = FakeLLM([("Available nerves", "I don't understand")])

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value=None):
            result = _sort_single_pass("task", _ranked(["touch"]), _catalog(), None)

        assert result == []

    def test_llm_returns_empty_array(self):
        """Empty array from LLM yields empty chain."""
        fake = FakeLLM([("Available nerves", "[]")])

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value=[]):
            result = _sort_single_pass("task", _ranked(["touch"]), _catalog(), None)

        assert result == []


# ── TestSortPairwise ─────────────────────────────────────────────────────────


@pytest.mark.timeout(10)
class TestSortPairwise:
    """Tests for _sort_pairwise — small model sorting."""

    def test_single_candidate_no_llm(self):
        """Single candidate returns directly without LLM call."""
        candidates = _ranked(["touch"])
        result = _sort_pairwise("task", candidates, _catalog(), None)

        assert len(result) == 1
        assert result[0]["nerve"] == "touch"

    def test_empty_candidates(self):
        result = _sort_pairwise("task", [], _catalog(), None)
        assert result == []

    def test_fallback_on_llm_garbage(self):
        """When LLM returns garbage, falls back to candidate order."""
        fake = FakeLLM([("Nerves:", "nope")])

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value=None):
            result = _sort_pairwise(
                "task",
                _ranked(["touch", "sight"]),
                _catalog(),
                None,
            )

        assert len(result) == 2
        assert result[0]["nerve"] == "touch"
        assert result[1]["nerve"] == "sight"


# ── TestFindGaps ─────────────────────────────────────────────────────────────


@pytest.mark.timeout(10)
class TestFindGaps:
    """Tests for _find_gaps — LLM completeness check."""

    def test_complete_chain_returns_empty(self):
        """When LLM says complete, no gaps returned."""
        fake = FakeLLM([("Is this chain complete", '{"complete": true}')])
        chain = [{"nerve": "touch", "args": "build"}]

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value={"complete": True}):
            result = _find_gaps("build project", chain, _catalog(), "medium")

        assert result == []

    def test_gaps_returned(self):
        """When LLM identifies gaps, they are returned as steps."""
        gap_response = {
            "complete": False,
            "gaps": [
                {"nerve": "linter", "description": "lint the code", "args": "run linting"},
            ],
        }
        fake = FakeLLM([("Is this chain complete", json.dumps(gap_response))])
        chain = [{"nerve": "touch", "args": "build"}]

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value=gap_response):
            result = _find_gaps("build and lint", chain, _catalog(), "medium")

        assert len(result) == 1
        assert result[0]["nerve"] == "linter"

    def test_skipped_for_tinylm(self):
        """tinylm models skip gap analysis entirely."""
        chain = [{"nerve": "touch", "args": "build"}]
        result = _find_gaps("task", chain, _catalog(), "tinylm")
        assert result == []

    def test_llm_returns_garbage(self):
        """Garbage LLM output returns no gaps."""
        fake = FakeLLM([("Is this chain complete", "yes it looks good")])
        chain = [{"nerve": "touch", "args": "build"}]

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value=None):
            result = _find_gaps("task", chain, _catalog(), "medium")

        assert result == []

    def test_malformed_gaps_filtered(self):
        """Gaps without 'nerve' key are filtered out."""
        gap_response = {
            "complete": False,
            "gaps": [
                {"description": "no nerve key"},
                {"nerve": "valid_gap", "args": "do something"},
            ],
        }
        fake = FakeLLM([("Is this chain complete", json.dumps(gap_response))])
        chain = [{"nerve": "touch", "args": "build"}]

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value=gap_response):
            result = _find_gaps("task", chain, _catalog(), "large")

        assert len(result) == 1
        assert result[0]["nerve"] == "valid_gap"

    def test_none_size_class_still_runs(self):
        """None size class does not skip gap analysis."""
        fake = FakeLLM([("Is this chain complete", '{"complete": true}')])
        chain = [{"nerve": "touch", "args": "x"}]

        with patch("arqitect.brain.planner.llm_generate", side_effect=fake), \
             patch("arqitect.brain.planner.extract_json", return_value={"complete": True}):
            result = _find_gaps("task", chain, _catalog(), None)

        assert result == []
        assert fake.call_count == 1


# ── TestComposeChain ─────────────────────────────────────────────────────────


@pytest.mark.timeout(10)
class TestComposeChain:
    """Tests for compose_chain — full 3-phase pipeline integration."""

    def test_full_pipeline(self):
        """Happy path: filter → sort → gaps → chain decision."""
        candidates = _ranked(["touch", "code_analyzer"])
        sorted_steps = [
            {"nerve": "code_analyzer", "args": "analyze"},
            {"nerve": "touch", "args": "apply fix"},
        ]

        with patch("arqitect.brain.planner._filter_candidates", return_value=candidates), \
             patch("arqitect.brain.planner._sort_into_sequence", return_value=sorted_steps), \
             patch("arqitect.brain.planner._find_gaps", return_value=[]):
            result = compose_chain("fix bugs", _catalog(), size_class="medium")

        assert result is not None
        assert result["action"] == Action.CHAIN_NERVES
        assert result["goal"] == "fix bugs"
        assert len(result["steps"]) == 2
        assert result["steps"][0]["nerve"] == "code_analyzer"

    def test_gaps_appended(self):
        """Gap nerves are appended to the chain."""
        sorted_steps = [{"nerve": "touch", "args": "build"}]
        gap_steps = [{"nerve": "linter", "args": "lint", "description": "lint code"}]

        with patch("arqitect.brain.planner._filter_candidates", return_value=_ranked(["touch"])), \
             patch("arqitect.brain.planner._sort_into_sequence", return_value=sorted_steps), \
             patch("arqitect.brain.planner._find_gaps", return_value=gap_steps):
            result = compose_chain("build and lint", _catalog())

        assert result is not None
        assert len(result["steps"]) == 2
        assert result["steps"][1]["nerve"] == "linter"

    def test_returns_none_for_empty_task(self):
        result = compose_chain("", _catalog())
        assert result is None

    def test_returns_none_for_empty_catalog(self):
        result = compose_chain("do something", {})
        assert result is None

    def test_returns_none_when_no_candidates(self):
        with patch("arqitect.brain.planner._filter_candidates", return_value=[]):
            result = compose_chain("impossible task", _catalog())

        assert result is None

    def test_returns_none_when_sort_fails(self):
        with patch("arqitect.brain.planner._filter_candidates", return_value=_ranked(["touch"])), \
             patch("arqitect.brain.planner._sort_into_sequence", return_value=[]):
            result = compose_chain("task", _catalog())

        assert result is None

    def test_project_facts_passed_through(self):
        """project_facts reach sort and gap phases."""
        facts = {"language": "rust", "framework": "actix"}
        sorted_steps = [{"nerve": "touch", "args": "cargo build"}]

        with patch("arqitect.brain.planner._filter_candidates", return_value=_ranked(["touch"])), \
             patch("arqitect.brain.planner._sort_into_sequence", return_value=sorted_steps) as mock_sort, \
             patch("arqitect.brain.planner._find_gaps", return_value=[]):
            compose_chain("build", _catalog(), project_facts=facts, size_class="large")

        _, kwargs = mock_sort.call_args
        # project_facts is the 4th positional arg
        call_args = mock_sort.call_args[0]
        assert call_args[3] == facts

    def test_single_nerve_result(self):
        """A chain with just one nerve is valid."""
        with patch("arqitect.brain.planner._filter_candidates", return_value=_ranked(["touch"])), \
             patch("arqitect.brain.planner._sort_into_sequence", return_value=[
                 {"nerve": "touch", "args": "list files"},
             ]), \
             patch("arqitect.brain.planner._find_gaps", return_value=[]):
            result = compose_chain("list files", _catalog())

        assert result is not None
        assert len(result["steps"]) == 1


# ── TestValidateSteps ────────────────────────────────────────────────────────


@pytest.mark.timeout(10)
class TestValidateSteps:
    """Tests for _validate_steps — input normalization."""

    def test_filters_non_dict_entries(self):
        result = _validate_steps(["string", 42, None, {"nerve": "touch", "args": "ok"}])
        assert len(result) == 1

    def test_filters_entries_without_nerve(self):
        result = _validate_steps([{"args": "no nerve"}, {"nerve": "", "args": "empty"}])
        assert len(result) == 0

    def test_uses_args_template_fallback(self):
        """Falls back to args_template if args is missing."""
        result = _validate_steps([{"nerve": "touch", "args_template": "echo hi"}])
        assert result[0]["args"] == "echo hi"

    def test_preserves_description(self):
        result = _validate_steps([{"nerve": "touch", "args": "go", "description": "do stuff"}])
        assert result[0]["description"] == "do stuff"

    def test_empty_list(self):
        assert _validate_steps([]) == []

    def test_deeply_nested_garbage(self):
        """Completely wrong structure is filtered out."""
        result = _validate_steps([[[]], {"nested": {"nerve": "touch"}}, True])
        assert result == []


# ── TestCandidatesToSteps ────────────────────────────────────────────────────


@pytest.mark.timeout(10)
class TestCandidatesToSteps:
    """Tests for _candidates_to_steps — fallback conversion."""

    def test_preserves_order(self):
        candidates = _ranked(["sight", "touch", "hearing"])
        result = _candidates_to_steps(candidates, _catalog(), "my task")

        assert [s["nerve"] for s in result] == ["sight", "touch", "hearing"]

    def test_uses_task_as_args(self):
        candidates = _ranked(["touch"])
        result = _candidates_to_steps(candidates, _catalog(), "build project")

        assert result[0]["args"] == "build project"

    def test_uses_catalog_description(self):
        candidates = _ranked(["touch"])
        result = _candidates_to_steps(candidates, _catalog(), "task")

        assert result[0]["description"] == "file and shell operations"

    def test_empty_candidates(self):
        assert _candidates_to_steps([], _catalog(), "task") == []
