"""Tests for arqitect.critic.qualify_nerve and arqitect.critic.qualify_tool.

Contract-based tests: mock only LLM calls, Redis, subprocess, HTTP requests.
Real logic for JSON extraction, rule filtering, description drift, etc.
"""

import json
import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest
from dirty_equals import IsInstance, IsFloat, IsStr
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Patch module-level globals before importing (Redis, dirs resolve at import)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_module_globals(tmp_path, test_redis):
    """Patch module-level globals that fire at import time.

    Uses the shared test_redis fixture from conftest instead of creating
    a standalone fakeredis instance.
    """
    nerves = str(tmp_path / "nerves")
    sandbox = str(tmp_path / "sandbox")
    tools = str(tmp_path / "mcp_tools")
    os.makedirs(nerves, exist_ok=True)
    os.makedirs(sandbox, exist_ok=True)
    os.makedirs(tools, exist_ok=True)

    with patch("arqitect.critic.qualify_nerve.NERVES_DIR", nerves), \
         patch("arqitect.critic.qualify_nerve.SANDBOX_DIR", sandbox), \
         patch("arqitect.critic.qualify_nerve._r", test_redis), \
         patch("arqitect.critic.qualify_nerve._MCP_TOOLS_DIR", tools), \
         patch("arqitect.critic.qualify_tool.MCP_TOOLS_DIR", tools), \
         patch("arqitect.critic.qualify_tool.MCP_URL", "http://localhost:9999"):
        yield {
            "nerves_dir": nerves,
            "sandbox_dir": sandbox,
            "tools_dir": tools,
            "redis": test_redis,
        }


# ---------------------------------------------------------------------------
# qualify_nerve._extract_json
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestExtractJsonNerve:
    """Contract: _extract_json must recover JSON from noisy LLM output."""

    def test_plain_json_object(self):
        from arqitect.critic.qualify_nerve import _extract_json
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_plain_json_array(self):
        from arqitect.critic.qualify_nerve import _extract_json
        assert _extract_json('[1, 2, 3]') == [1, 2, 3]

    def test_code_fenced_json(self):
        from arqitect.critic.qualify_nerve import _extract_json
        raw = '```json\n{"x": 42}\n```'
        assert _extract_json(raw) == {"x": 42}

    def test_noisy_prefix(self):
        from arqitect.critic.qualify_nerve import _extract_json
        raw = 'Here is the result:\n{"key": "val"}'
        assert _extract_json(raw) == {"key": "val"}

    def test_nested_braces(self):
        from arqitect.critic.qualify_nerve import _extract_json
        raw = 'prefix {"outer": {"inner": "val"}} suffix'
        assert _extract_json(raw) == {"outer": {"inner": "val"}}

    def test_returns_none_for_no_json(self):
        from arqitect.critic.qualify_nerve import _extract_json
        assert _extract_json("no json here") is None

    def test_empty_string(self):
        from arqitect.critic.qualify_nerve import _extract_json
        assert _extract_json("") is None

    def test_string_with_escaped_quotes(self):
        from arqitect.critic.qualify_nerve import _extract_json
        raw = '{"msg": "he said \\"hello\\""}'
        result = _extract_json(raw)
        assert result is not None
        assert result["msg"] == 'he said "hello"'

    def test_array_with_objects(self):
        from arqitect.critic.qualify_nerve import _extract_json
        raw = 'blah [{"a": 1}, {"b": 2}] blah'
        result = _extract_json(raw)
        assert isinstance(result, list)
        assert len(result) == 2

    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
        values=st.one_of(st.integers(), st.text(min_size=1, max_size=50)),
        min_size=1,
        max_size=5,
    ))
    @settings(max_examples=30)
    def test_roundtrips_valid_json_objects(self, data):
        """Any valid JSON object wrapped in noise should be recoverable."""
        from arqitect.critic.qualify_nerve import _extract_json
        raw = f"Here is the result: {json.dumps(data)} end"
        result = _extract_json(raw)
        assert result == data

    @given(st.lists(st.integers(), min_size=1, max_size=10))
    @settings(max_examples=20)
    def test_roundtrips_valid_json_arrays(self, data):
        """Any valid JSON array wrapped in noise should be recoverable."""
        from arqitect.critic.qualify_nerve import _extract_json
        raw = f"prefix {json.dumps(data)} suffix"
        result = _extract_json(raw)
        assert result == data


# ---------------------------------------------------------------------------
# qualify_nerve._parse_test_case_response
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestParseTestCaseResponse:
    """Contract: parse LLM output into test case list regardless of wrapping format."""

    def test_bare_array(self):
        from arqitect.critic.qualify_nerve import _parse_test_case_response
        raw = json.dumps([{"input": "hi", "output": "hello"}])
        result = _parse_test_case_response(raw)
        assert result is not None
        assert len(result) == 1

    def test_wrapped_object_test_cases(self):
        from arqitect.critic.qualify_nerve import _parse_test_case_response
        raw = json.dumps({"test_cases": [{"input": "a", "output": "b"}]})
        result = _parse_test_case_response(raw)
        assert result is not None
        assert result[0]["input"] == "a"

    def test_wrapped_object_tests_key(self):
        from arqitect.critic.qualify_nerve import _parse_test_case_response
        raw = json.dumps({"tests": [{"input": "x"}]})
        result = _parse_test_case_response(raw)
        assert result is not None

    def test_code_fenced(self):
        from arqitect.critic.qualify_nerve import _parse_test_case_response
        raw = '```json\n[{"input": "q", "output": "a"}]\n```'
        result = _parse_test_case_response(raw)
        assert result is not None

    def test_unparseable_returns_none(self):
        from arqitect.critic.qualify_nerve import _parse_test_case_response
        assert _parse_test_case_response("totally not json") is None

    def test_plain_object_without_list_key_returns_none(self):
        from arqitect.critic.qualify_nerve import _parse_test_case_response
        raw = json.dumps({"something": "else"})
        assert _parse_test_case_response(raw) is None

    @given(st.sampled_from(["test_cases", "tests", "cases", "data"]))
    @settings(max_examples=4)
    def test_all_wrapper_keys_accepted(self, key):
        """All documented wrapper keys must extract the inner list."""
        from arqitect.critic.qualify_nerve import _parse_test_case_response
        payload = {key: [{"input": "x", "output": "y"}]}
        result = _parse_test_case_response(json.dumps(payload))
        assert result is not None
        assert result[0]["input"] == "x"


# ---------------------------------------------------------------------------
# qualify_nerve._deterministic_check
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestDeterministicCheck:
    """Contract: fast checks return a score; None means 'defer to LLM'."""

    def test_empty_output_scores_zero(self):
        from arqitect.critic.qualify_nerve import _deterministic_check
        assert _deterministic_check("", "some input") == 0.0
        assert _deterministic_check(" ", "input") == 0.0

    def test_error_output_scores_low(self):
        from arqitect.critic.qualify_nerve import _deterministic_check
        assert _deterministic_check("Error: something went wrong", "query") == 0.1

    def test_traceback_scores_low(self):
        from arqitect.critic.qualify_nerve import _deterministic_check
        assert _deterministic_check("Traceback (most recent call last):\n...", "q") == 0.1

    def test_echoed_input_scores_low(self):
        from arqitect.critic.qualify_nerve import _deterministic_check
        assert _deterministic_check("what is the weather", "What is the weather") == 0.1

    def test_normal_output_defers_to_llm(self):
        from arqitect.critic.qualify_nerve import _deterministic_check
        assert _deterministic_check("The weather in NYC is 72F and sunny", "what is the weather") is None

    def test_timed_out_scores_low(self):
        from arqitect.critic.qualify_nerve import _deterministic_check
        assert _deterministic_check("timed out waiting for response", "q") == 0.1

    @given(st.sampled_from(["Error:", "Traceback", "timed out", "not found", "failed to"]))
    @settings(max_examples=5)
    def test_all_error_markers_detected(self, marker):
        """Every documented error marker must produce a 0.1 score."""
        from arqitect.critic.qualify_nerve import _deterministic_check
        assert _deterministic_check(f"prefix {marker} suffix", "query") == 0.1


# ---------------------------------------------------------------------------
# qualify_nerve._is_junk_rule
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestIsJunkRule:
    """Contract: metric-leak and vague-filler rules are rejected."""

    def test_metric_leak_detected(self):
        from arqitect.critic.qualify_nerve import _is_junk_rule
        assert _is_junk_rule("Improve embedding similarity for better results") is True

    def test_vague_filler_detected(self):
        from arqitect.critic.qualify_nerve import _is_junk_rule
        assert _is_junk_rule("Ensure outputs are contextually relevant") is True

    def test_specific_rule_passes(self):
        from arqitect.critic.qualify_nerve import _is_junk_rule
        assert _is_junk_rule("Always include the unit in conversion results") is False

    def test_cosine_similarity_rejected(self):
        from arqitect.critic.qualify_nerve import _is_junk_rule
        assert _is_junk_rule("Optimize cosine similarity between output and expected") is True

    @given(st.sampled_from([
        "embedding similarity", "embedding score", "cosine similarity",
        "scoring", "low score", "high score", "improve score",
    ]))
    @settings(max_examples=7)
    def test_all_metric_leak_terms_rejected(self, term):
        """Every metric-leak term must be flagged as junk."""
        from arqitect.critic.qualify_nerve import _is_junk_rule
        assert _is_junk_rule(f"Use {term} to improve results") is True


# ---------------------------------------------------------------------------
# qualify_nerve._is_duplicate_rule
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestIsDuplicateRule:
    """Contract: overlapping rules are flagged; novel rules pass."""

    def test_duplicate_detected(self):
        from arqitect.critic.qualify_nerve import _is_duplicate_rule
        existing = "Rule: Always include the unit in conversion results"
        new = "Always include units in your conversion results"
        assert _is_duplicate_rule(new, existing) is True

    def test_novel_rule_passes(self):
        from arqitect.critic.qualify_nerve import _is_duplicate_rule
        existing = "Rule: Always include the unit in conversion results"
        new = "Format dates as ISO 8601 timestamps"
        assert _is_duplicate_rule(new, existing) is False

    def test_very_short_rule_is_duplicate(self):
        from arqitect.critic.qualify_nerve import _is_duplicate_rule
        assert _is_duplicate_rule("be good", "Rule: something else") is True


# ---------------------------------------------------------------------------
# qualify_nerve._is_junk_description
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestIsJunkDescription:
    """Contract: reconciler-jargon descriptions are rejected."""

    def test_junk_description_detected(self):
        from arqitect.critic.qualify_nerve import _is_junk_description
        assert _is_junk_description("Refined rule to improve scoring cases") is True

    def test_clean_description_passes(self):
        from arqitect.critic.qualify_nerve import _is_junk_description
        assert _is_junk_description("Solve math problems: arithmetic, algebra") is False

    @given(st.sampled_from([
        "embedding similarity", "embedding score", "refine",
        "scoring cases", "improve score", "low scoring",
        "high embedding", "cosine",
    ]))
    @settings(max_examples=8)
    def test_all_junk_terms_detected(self, term):
        """Every junk term must flag the description."""
        from arqitect.critic.qualify_nerve import _is_junk_description
        assert _is_junk_description(f"A description with {term} inside") is True


# ---------------------------------------------------------------------------
# qualify_nerve._is_description_drift
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestIsDescriptionDrift:
    """Contract: <20% word overlap with original means drift."""

    def test_drift_detected(self):
        from arqitect.critic.qualify_nerve import _is_description_drift
        orig = "Weather forecasting for US cities"
        new = "Database administration and SQL queries"
        assert _is_description_drift(orig, new) is True

    def test_no_drift_for_related(self):
        from arqitect.critic.qualify_nerve import _is_description_drift
        orig = "Weather forecasting for US cities"
        new = "Weather lookup and forecasting for cities in the US"
        assert _is_description_drift(orig, new) is False

    def test_empty_strings(self):
        from arqitect.critic.qualify_nerve import _is_description_drift
        assert _is_description_drift("", "anything") is False
        assert _is_description_drift("something", "") is False

    def test_identical_descriptions_no_drift(self):
        """Identical descriptions must never drift."""
        from arqitect.critic.qualify_nerve import _is_description_drift
        desc = "Real-time weather forecasting for US cities"
        assert _is_description_drift(desc, desc) is False


# ---------------------------------------------------------------------------
# qualify_nerve._extract_tool_errors
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestExtractToolErrors:
    """Contract: structured tool errors extracted from stderr patterns."""

    def test_pattern1_tool_call_failed(self):
        from arqitect.critic.qualify_nerve import _extract_tool_errors
        failures = [{
            "raw_stderr": "[NERVE:weather] Tool call failed (weather_api): connection timeout",
            "raw_stdout": "",
            "input": "weather in NYC",
        }]
        errors = _extract_tool_errors(failures)
        assert len(errors) == 1
        assert errors[0]["tool"] == "weather_api"

    def test_pattern1b_mcp_call_error(self):
        from arqitect.critic.qualify_nerve import _extract_tool_errors
        failures = [{
            "raw_stderr": "[NERVE:x] Tool call failed: MCP call error (dateparser): bad format",
            "raw_stdout": "",
            "input": "parse date",
        }]
        errors = _extract_tool_errors(failures)
        assert len(errors) == 1
        assert errors[0]["tool"] == "dateparser"

    def test_deduplicates_errors(self):
        from arqitect.critic.qualify_nerve import _extract_tool_errors
        failures = [
            {"raw_stderr": "[NERVE:x] Tool call failed (api): timeout", "raw_stdout": "", "input": "a"},
            {"raw_stderr": "[NERVE:x] Tool call failed (api): timeout", "raw_stdout": "", "input": "b"},
        ]
        errors = _extract_tool_errors(failures)
        assert len(errors) == 1

    def test_caps_at_five(self):
        from arqitect.critic.qualify_nerve import _extract_tool_errors
        failures = [
            {"raw_stderr": f"[NERVE:x] Tool call failed (tool{i}): err{i}", "raw_stdout": "", "input": f"q{i}"}
            for i in range(10)
        ]
        errors = _extract_tool_errors(failures)
        assert len(errors) <= 5

    def test_empty_failures(self):
        from arqitect.critic.qualify_nerve import _extract_tool_errors
        assert _extract_tool_errors([]) == []


# ---------------------------------------------------------------------------
# qualify_nerve._apply_tool_fix / _rollback_tool_fix / _cleanup_bak_files
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestToolFixLifecycle:
    """Contract: tool fixes are applied atomically with backup/rollback."""

    def test_apply_tool_fix_success(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import _apply_tool_fix
        tools_dir = _patch_module_globals["tools_dir"]
        tool_path = os.path.join(tools_dir, "calc.py")
        with open(tool_path, "w") as f:
            f.write("def run(): return 'old'\n")

        fixed_code = "def run():\n    return 'fixed'\n"
        with patch("arqitect.mcp.server.CORE_TOOLS", frozenset()):
            result = _apply_tool_fix("calc", fixed_code, ["calc"])
        assert result is True
        with open(tool_path) as f:
            assert "fixed" in f.read()
        # Backup must exist for rollback
        assert os.path.isfile(tool_path + ".bak")

    def test_apply_rejects_unknown_tool(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import _apply_tool_fix
        result = _apply_tool_fix("calc", "def run(): pass", ["other_tool"])
        assert result is False

    def test_apply_rejects_core_tool(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import _apply_tool_fix
        tools_dir = _patch_module_globals["tools_dir"]
        with open(os.path.join(tools_dir, "image_generator.py"), "w") as f:
            f.write("def run(): pass\n")
        with patch("arqitect.mcp.server.CORE_TOOLS", frozenset({"image_generator"})):
            result = _apply_tool_fix("image_generator", "def run(): pass", ["image_generator"])
        assert result is False

    def test_apply_rejects_syntax_error(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import _apply_tool_fix
        tools_dir = _patch_module_globals["tools_dir"]
        with open(os.path.join(tools_dir, "calc.py"), "w") as f:
            f.write("def run(): pass\n")
        result = _apply_tool_fix("calc", "def run(:\n", ["calc"])
        assert result is False

    def test_apply_rejects_missing_run(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import _apply_tool_fix
        tools_dir = _patch_module_globals["tools_dir"]
        with open(os.path.join(tools_dir, "calc.py"), "w") as f:
            f.write("def run(): pass\n")
        result = _apply_tool_fix("calc", "def compute(): pass\n", ["calc"])
        assert result is False

    def test_apply_rejects_nonexistent_file(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import _apply_tool_fix
        result = _apply_tool_fix("nonexistent", "def run(): pass\n", ["nonexistent"])
        assert result is False

    def test_rollback_restores_backup(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import _rollback_tool_fix
        tools_dir = _patch_module_globals["tools_dir"]
        tool_path = os.path.join(tools_dir, "calc.py")
        with open(tool_path, "w") as f:
            f.write("fixed version")
        with open(tool_path + ".bak", "w") as f:
            f.write("original version")
        result = _rollback_tool_fix("calc")
        assert result is True
        with open(tool_path) as f:
            assert f.read() == "original version"
        assert not os.path.isfile(tool_path + ".bak")

    def test_rollback_no_backup_returns_false(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import _rollback_tool_fix
        assert _rollback_tool_fix("nonexistent") is False

    def test_cleanup_bak_files(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import _cleanup_bak_files
        tools_dir = _patch_module_globals["tools_dir"]
        bak_path = os.path.join(tools_dir, "calc.py.bak")
        with open(bak_path, "w") as f:
            f.write("old")
        _cleanup_bak_files(["calc"])
        assert not os.path.isfile(bak_path)


# ---------------------------------------------------------------------------
# qualify_nerve.evaluate_nerve_output
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestEvaluateNerveOutput:
    """Contract: layered evaluation -- deterministic checks first, then LLM."""

    def test_empty_output_fails_deterministically(self):
        from arqitect.critic.qualify_nerve import evaluate_nerve_output
        test_case = {"input": "hello", "output": "greeting"}
        nerve_output = {"raw_stdout": "", "raw_stderr": "", "exit_code": 0, "timed_out": False}
        result = evaluate_nerve_output(test_case, nerve_output)
        assert result["passed"] is False
        assert result["score"] == 0.0

    def test_error_output_fails_deterministically(self):
        from arqitect.critic.qualify_nerve import evaluate_nerve_output
        test_case = {"input": "hello", "output": "greeting"}
        nerve_output = {"raw_stdout": "Error: module not found", "raw_stderr": "", "exit_code": 1, "timed_out": False}
        result = evaluate_nerve_output(test_case, nerve_output)
        assert result["passed"] is False
        assert result["score"] == 0.1

    def test_echoed_input_fails(self):
        from arqitect.critic.qualify_nerve import evaluate_nerve_output
        test_case = {"input": "what is the weather", "output": "weather info"}
        nerve_output = {"raw_stdout": "what is the weather", "raw_stderr": "", "exit_code": 0, "timed_out": False}
        result = evaluate_nerve_output(test_case, nerve_output)
        assert result["passed"] is False
        assert "echo" in result["reasoning"]

    def test_good_output_defers_to_llm(self):
        from arqitect.critic.qualify_nerve import evaluate_nerve_output
        test_case = {"input": "hello", "output": "greeting"}
        nerve_output = {"raw_stdout": "Hello! How can I help you today?", "raw_stderr": "", "exit_code": 0, "timed_out": False}
        llm_response = json.dumps({"passed": True, "score": 0.9, "reasoning": "good", "issue": ""})
        with patch("arqitect.critic.qualify_nerve._llm", return_value=llm_response):
            result = evaluate_nerve_output(test_case, nerve_output)
        assert result["passed"] is True
        assert result["score"] == 0.9

    def test_llm_returns_unparseable(self):
        from arqitect.critic.qualify_nerve import evaluate_nerve_output
        test_case = {"input": "hello", "output": "greeting"}
        nerve_output = {"raw_stdout": "some valid output here", "raw_stderr": "", "exit_code": 0, "timed_out": False}
        with patch("arqitect.critic.qualify_nerve._llm", return_value="not json at all"):
            result = evaluate_nerve_output(test_case, nerve_output)
        assert result["passed"] is False
        assert "parse" in result["reasoning"].lower()

    def test_result_always_has_required_keys(self):
        """Contract: every result dict has passed, score, reasoning, issue."""
        from arqitect.critic.qualify_nerve import evaluate_nerve_output
        test_case = {"input": "hello", "output": "greeting"}
        nerve_output = {"raw_stdout": "", "raw_stderr": "", "exit_code": 0, "timed_out": False}
        result = evaluate_nerve_output(test_case, nerve_output)
        assert result == {
            "passed": IsInstance(bool),
            "score": IsFloat,
            "reasoning": IsStr,
            "issue": IsStr,
        }


# ---------------------------------------------------------------------------
# qualify_nerve.generate_test_cases
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGenerateTestCases:
    """Contract: brain-size gated; returns tagged test cases or empty list."""

    def test_small_brain_returns_empty(self):
        """Small brains are gated from generating test cases."""
        from arqitect.critic.qualify_nerve import generate_test_cases
        with patch("arqitect.brain.adapters.get_model_size_class", return_value="small"):
            result = generate_test_cases("weather", "get weather")
        assert result == []

    def test_medium_brain_generates_cases(self):
        """Medium brain produces test cases from LLM output."""
        from arqitect.critic.qualify_nerve import generate_test_cases
        cases = [{"input": "NYC weather", "output": "sunny", "category": "core"}]
        with patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value={"test_cases_per_batch": 4, "training_max_length": 1024}), \
             patch("arqitect.critic.qualify_nerve._get_batch_size", return_value=4), \
             patch("arqitect.critic.qualify_nerve._llm", return_value=json.dumps(cases)), \
             patch("arqitect.brain.adapters.get_raw_model_name", return_value="test-model"):
            result = generate_test_cases("weather", "get weather")
        assert len(result) == 1
        assert result[0]["generated_by"] == "test-model"

    def test_unparseable_llm_output_returns_empty(self):
        """Unparseable LLM output returns empty list."""
        from arqitect.critic.qualify_nerve import generate_test_cases
        with patch("arqitect.brain.adapters.get_model_size_class", return_value="large"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value={"test_cases_per_batch": 4, "training_max_length": 1024}), \
             patch("arqitect.critic.qualify_nerve._get_batch_size", return_value=4), \
             patch("arqitect.critic.qualify_nerve._llm", return_value="gibberish"):
            result = generate_test_cases("weather", "get weather")
        assert result == []

    @given(st.sampled_from(["tiny", "tinylm", "small"]))
    @settings(max_examples=3)
    def test_non_medium_large_brains_gated(self, size):
        """Only medium and large brains may generate test cases."""
        from arqitect.critic.qualify_nerve import generate_test_cases
        with patch("arqitect.brain.adapters.get_model_size_class", return_value=size):
            result = generate_test_cases("weather", "get weather")
        assert result == []


# ---------------------------------------------------------------------------
# qualify_nerve.run_nerve_with_input
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestRunNerveWithInput:
    """Contract: run a nerve subprocess and return structured result dict."""

    def test_nerve_not_found(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import run_nerve_with_input
        mem_mgr = MagicMock()
        mem_mgr.get_env_for_nerve.return_value = {}
        result = run_nerve_with_input("nonexistent", "hello", mem_mgr)
        assert result["exit_code"] == -1
        assert "not found" in result["raw_stderr"]

    def test_successful_run(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import run_nerve_with_input
        nerves_dir = _patch_module_globals["nerves_dir"]
        nerve_dir = os.path.join(nerves_dir, "test_nerve")
        os.makedirs(nerve_dir)
        with open(os.path.join(nerve_dir, "nerve.py"), "w") as f:
            f.write("import sys, json; print(json.dumps({'response': 'ok'}))\n")

        mem_mgr = MagicMock()
        mem_mgr.get_env_for_nerve.return_value = {}

        mock_result = MagicMock()
        mock_result.stdout = '{"response": "ok"}'
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("arqitect.critic.qualify_nerve.subprocess.run", return_value=mock_result):
            result = run_nerve_with_input("test_nerve", "hello", mem_mgr)
        assert result["exit_code"] == 0
        assert result["parsed"] == {"response": "ok"}
        assert result["timed_out"] is False

    def test_timeout(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import run_nerve_with_input
        nerves_dir = _patch_module_globals["nerves_dir"]
        nerve_dir = os.path.join(nerves_dir, "slow")
        os.makedirs(nerve_dir)
        with open(os.path.join(nerve_dir, "nerve.py"), "w") as f:
            f.write("import time; time.sleep(999)\n")

        mem_mgr = MagicMock()
        mem_mgr.get_env_for_nerve.return_value = {}

        with patch("arqitect.critic.qualify_nerve.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="python", timeout=60)):
            result = run_nerve_with_input("slow", "hello", mem_mgr)
        assert result["timed_out"] is True
        assert result["exit_code"] == -1

    def test_exception_during_run(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import run_nerve_with_input
        nerves_dir = _patch_module_globals["nerves_dir"]
        nerve_dir = os.path.join(nerves_dir, "bad")
        os.makedirs(nerve_dir)
        with open(os.path.join(nerve_dir, "nerve.py"), "w") as f:
            f.write("pass\n")

        mem_mgr = MagicMock()
        mem_mgr.get_env_for_nerve.return_value = {}

        with patch("arqitect.critic.qualify_nerve.subprocess.run",
                   side_effect=OSError("permission denied")):
            result = run_nerve_with_input("bad", "hello", mem_mgr)
        assert result["exit_code"] == -1
        assert "permission denied" in result["raw_stderr"]

    def test_result_shape_on_success(self, _patch_module_globals):
        """Contract: successful result always has all required keys."""
        from arqitect.critic.qualify_nerve import run_nerve_with_input
        nerves_dir = _patch_module_globals["nerves_dir"]
        nerve_dir = os.path.join(nerves_dir, "shape_test")
        os.makedirs(nerve_dir)
        with open(os.path.join(nerve_dir, "nerve.py"), "w") as f:
            f.write("print('hello')\n")

        mem_mgr = MagicMock()
        mem_mgr.get_env_for_nerve.return_value = {}

        mock_result = MagicMock()
        mock_result.stdout = "hello"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("arqitect.critic.qualify_nerve.subprocess.run", return_value=mock_result):
            result = run_nerve_with_input("shape_test", "hi", mem_mgr)

        assert result == {
            "raw_stdout": IsStr,
            "raw_stderr": IsStr,
            "parsed": None,  # "hello" is not valid JSON
            "exit_code": IsInstance(int),
            "timed_out": IsInstance(bool),
        }


# ---------------------------------------------------------------------------
# qualify_nerve._consolidate_prompt
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestConsolidatePrompt:
    """Contract: rewrite prompt or fall back to original on error."""

    def test_successful_consolidation(self):
        from arqitect.critic.qualify_nerve import _consolidate_prompt
        consolidated = "You are a weather assistant. Always include units."
        with patch("arqitect.critic.qualify_nerve._llm", return_value=consolidated):
            result = _consolidate_prompt("weather", "weather lookup", "old prompt with many rules")
        assert result == consolidated

    def test_consolidation_failure_returns_original(self):
        from arqitect.critic.qualify_nerve import _consolidate_prompt
        with patch("arqitect.critic.qualify_nerve._llm", side_effect=RuntimeError("llm down")):
            result = _consolidate_prompt("weather", "weather", "original prompt")
        assert result == "original prompt"


# ---------------------------------------------------------------------------
# qualify_tool._extract_json
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestExtractJsonTool:
    """Contract: same JSON extraction semantics as qualify_nerve variant."""

    def test_plain_json(self):
        from arqitect.critic.qualify_tool import _extract_json
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_code_fenced(self):
        from arqitect.critic.qualify_tool import _extract_json
        assert _extract_json('```\n[1,2]\n```') == [1, 2]

    def test_noisy_output(self):
        from arqitect.critic.qualify_tool import _extract_json
        result = _extract_json('Here: {"k": "v"} done')
        assert result == {"k": "v"}

    def test_returns_none_for_garbage(self):
        from arqitect.critic.qualify_tool import _extract_json
        assert _extract_json("no json") is None


# ---------------------------------------------------------------------------
# qualify_tool.generate_tool_tests
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGenerateToolTests:
    """Contract: produce test cases from LLM or empty list on failure."""

    def test_returns_parsed_tests(self):
        from arqitect.critic.qualify_tool import generate_tool_tests
        cases = [{"args": {"q": "hi"}, "expected_behavior": "greet", "category": "happy_path"}]
        with patch("arqitect.critic.qualify_tool._llm", return_value=json.dumps(cases)):
            result = generate_tool_tests("greet", "greet user", "q: str")
        assert len(result) == 1
        assert result[0]["category"] == "happy_path"

    def test_returns_empty_on_parse_failure(self):
        from arqitect.critic.qualify_tool import generate_tool_tests
        with patch("arqitect.critic.qualify_tool._llm", return_value="not json"):
            result = generate_tool_tests("t", "d", "p")
        assert result == []


# ---------------------------------------------------------------------------
# qualify_tool.call_tool
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestCallTool:
    """Contract: HTTP call to MCP returns structured success/failure dict."""

    def test_successful_call(self):
        from arqitect.critic.qualify_tool import call_tool
        import responses as responses_lib
        with responses_lib.RequestsMock() as rsps:
            rsps.add(
                responses_lib.POST,
                "http://localhost:9999/call/my_tool",
                json={"result": "42"},
                status=200,
            )
            result = call_tool("my_tool", {"x": 1})
        assert result["success"] is True
        assert result["result"] == "42"

    def test_error_in_response_body(self):
        from arqitect.critic.qualify_tool import call_tool
        import responses as responses_lib
        with responses_lib.RequestsMock() as rsps:
            rsps.add(
                responses_lib.POST,
                "http://localhost:9999/call/my_tool",
                json={"error": "bad input"},
                status=200,
            )
            result = call_tool("my_tool", {"x": 1})
        assert result["success"] is False
        assert result["error"] == "bad input"

    def test_http_error(self):
        from arqitect.critic.qualify_tool import call_tool
        import responses as responses_lib
        with responses_lib.RequestsMock() as rsps:
            rsps.add(
                responses_lib.POST,
                "http://localhost:9999/call/my_tool",
                json={"error": "server error"},
                status=500,
            )
            result = call_tool("my_tool", {"x": 1})
        assert result["success"] is False

    def test_connection_error(self):
        from arqitect.critic.qualify_tool import call_tool
        import responses as responses_lib
        with responses_lib.RequestsMock() as rsps:
            rsps.add(
                responses_lib.POST,
                "http://localhost:9999/call/my_tool",
                body=ConnectionError("refused"),
            )
            result = call_tool("my_tool", {"x": 1})
        assert result["success"] is False
        assert result["latency_ms"] >= 0

    def test_non_dict_args_wrapped(self):
        from arqitect.critic.qualify_tool import call_tool
        import responses as responses_lib
        with responses_lib.RequestsMock() as rsps:
            rsps.add(
                responses_lib.POST,
                "http://localhost:9999/call/t",
                json={"result": "ok"},
                status=200,
            )
            result = call_tool("t", "raw string")
        assert result["success"] is True

    def test_result_always_has_required_keys(self):
        """Contract: result dict always has success, result, error, latency_ms."""
        from arqitect.critic.qualify_tool import call_tool
        import responses as responses_lib
        with responses_lib.RequestsMock() as rsps:
            rsps.add(
                responses_lib.POST,
                "http://localhost:9999/call/t",
                json={"result": "ok"},
                status=200,
            )
            result = call_tool("t", {"q": "x"})
        assert result == {
            "success": IsInstance(bool),
            "result": IsStr | None,
            "error": IsStr | None,
            "latency_ms": IsInstance(int),
        }


# ---------------------------------------------------------------------------
# qualify_tool.evaluate_tool_result
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestEvaluateToolResult:
    """Contract: evaluate tool call result via LLM, with fallback on parse error."""

    def test_successful_evaluation(self):
        from arqitect.critic.qualify_tool import evaluate_tool_result
        tc = {"args": {"q": "hi"}, "expected_behavior": "greet", "category": "happy_path"}
        cr = {"success": True, "result": "Hello!", "error": None, "latency_ms": 50}
        llm_resp = json.dumps({"passed": True, "score": 0.9, "reasoning": "good"})
        with patch("arqitect.critic.qualify_tool._llm", return_value=llm_resp):
            result = evaluate_tool_result(tc, cr)
        assert result["passed"] is True
        assert result["score"] == 0.9

    def test_unparseable_evaluation(self):
        from arqitect.critic.qualify_tool import evaluate_tool_result
        tc = {"args": {}, "expected_behavior": "x", "category": "happy_path"}
        cr = {"success": True, "result": "ok", "error": None, "latency_ms": 10}
        with patch("arqitect.critic.qualify_tool._llm", return_value="garbage"):
            result = evaluate_tool_result(tc, cr)
        assert result["passed"] is False
        assert "parse" in result["reasoning"].lower()


# ---------------------------------------------------------------------------
# qualify_tool.quarantine_tool
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestQuarantineTool:
    """Contract: remove tool file or return empty string if missing."""

    def test_removes_existing_tool(self, _patch_module_globals):
        from arqitect.critic.qualify_tool import quarantine_tool
        tools_dir = _patch_module_globals["tools_dir"]
        tool_path = os.path.join(tools_dir, "bad_tool.py")
        with open(tool_path, "w") as f:
            f.write("def run(): pass\n")
        result = quarantine_tool("bad_tool")
        assert result == tool_path
        assert not os.path.exists(tool_path)

    def test_nonexistent_tool_returns_empty(self, _patch_module_globals):
        from arqitect.critic.qualify_tool import quarantine_tool
        result = quarantine_tool("nonexistent")
        assert result == ""


# ---------------------------------------------------------------------------
# qualify_tool.qualify_tool (main entry)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestQualifyTool:
    """Contract: orchestrate test generation, calling, and evaluation."""

    def test_no_tests_returns_unqualified(self):
        from arqitect.critic.qualify_tool import qualify_tool
        with patch("arqitect.critic.qualify_tool.generate_tool_tests", return_value=[]):
            result = qualify_tool("t", "desc", "params")
        assert result["qualified"] is False
        assert result["score"] == 0.0

    def test_passing_score_qualifies(self):
        from arqitect.critic.qualify_tool import qualify_tool
        tests = [
            {"args": {"q": "hi"}, "expected_behavior": "greet", "category": "happy_path"},
            {"args": {"q": ""}, "expected_behavior": "handle empty", "category": "bad_input"},
        ]
        call_result = {"success": True, "result": "Hello!", "error": None, "latency_ms": 50}
        eval_result = {"passed": True, "score": 0.9, "reasoning": "good"}

        with patch("arqitect.critic.qualify_tool.generate_tool_tests", return_value=tests), \
             patch("arqitect.critic.qualify_tool.call_tool", return_value=call_result), \
             patch("arqitect.critic.qualify_tool.evaluate_tool_result", return_value=eval_result):
            result = qualify_tool("t", "desc", "params")
        assert result["qualified"] is True
        assert result["score"] >= 0.6

    def test_failing_score_unqualifies(self):
        from arqitect.critic.qualify_tool import qualify_tool
        tests = [{"args": {"q": "hi"}, "expected_behavior": "greet", "category": "happy_path"}]
        call_result = {"success": False, "result": None, "error": "crash", "latency_ms": 50}
        eval_result = {"passed": False, "score": 0.1, "reasoning": "crashed"}

        with patch("arqitect.critic.qualify_tool.generate_tool_tests", return_value=tests), \
             patch("arqitect.critic.qualify_tool.call_tool", return_value=call_result), \
             patch("arqitect.critic.qualify_tool.evaluate_tool_result", return_value=eval_result):
            result = qualify_tool("t", "desc", "params")
        assert result["qualified"] is False

    def test_string_args_wrapped(self):
        """When test case args is a string, it gets wrapped in {query: ...}."""
        from arqitect.critic.qualify_tool import qualify_tool
        tests = [{"args": "raw_string", "expected_behavior": "x", "category": "happy_path"}]
        call_result = {"success": True, "result": "ok", "error": None, "latency_ms": 10}
        eval_result = {"passed": True, "score": 0.8, "reasoning": "ok"}

        with patch("arqitect.critic.qualify_tool.generate_tool_tests", return_value=tests), \
             patch("arqitect.critic.qualify_tool.call_tool", return_value=call_result) as mock_call, \
             patch("arqitect.critic.qualify_tool.evaluate_tool_result", return_value=eval_result):
            qualify_tool("t", "desc", "params")
        called_args = mock_call.call_args[0][1]
        assert isinstance(called_args, dict)
        assert "query" in called_args

    def test_list_args_wrapped(self):
        """When test case args is a list, first element used."""
        from arqitect.critic.qualify_tool import qualify_tool
        tests = [{"args": ["first", "second"], "expected_behavior": "x", "category": "happy_path"}]
        call_result = {"success": True, "result": "ok", "error": None, "latency_ms": 10}
        eval_result = {"passed": True, "score": 0.8, "reasoning": "ok"}

        with patch("arqitect.critic.qualify_tool.generate_tool_tests", return_value=tests), \
             patch("arqitect.critic.qualify_tool.call_tool", return_value=call_result) as mock_call, \
             patch("arqitect.critic.qualify_tool.evaluate_tool_result", return_value=eval_result):
            qualify_tool("t", "desc", "params")
        called_args = mock_call.call_args[0][1]
        assert called_args == {"query": "first"}


# ---------------------------------------------------------------------------
# qualify_nerve._get_max_system_tokens
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGetMaxSystemTokens:
    """Contract: read from yaml or fall back to default."""

    def test_returns_default_on_error(self):
        from arqitect.critic.qualify_nerve import _get_max_system_tokens, _DEFAULT_MAX_SYSTEM_TOKENS
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _get_max_system_tokens("tool")
        assert result == _DEFAULT_MAX_SYSTEM_TOKENS

    def test_returns_value_from_yaml(self):
        from arqitect.critic.qualify_nerve import _get_max_system_tokens
        mock_yaml = {"medium": {"max_system_tokens": 4096}}
        with patch("builtins.open", MagicMock()), \
             patch("yaml.safe_load", return_value=mock_yaml), \
             patch("arqitect.brain.adapters.get_active_variant", return_value="medium"):
            result = _get_max_system_tokens("tool")
        assert result == 4096


# ---------------------------------------------------------------------------
# qualify_nerve._publish_progress
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestPublishProgress:
    """Contract: publish to Redis channel; never raise on errors."""

    def test_publishes_to_redis(self, _patch_module_globals):
        from arqitect.critic.qualify_nerve import _publish_progress
        redis_client = _patch_module_globals["redis"]
        pubsub = redis_client.pubsub()
        pubsub.subscribe("nerve:qualification")
        # Consume the subscription confirmation message
        pubsub.get_message()

        _publish_progress("test_nerve", 0.85, True, ["tool1"], 2, 3)

        msg = pubsub.get_message()
        assert msg is not None
        data = json.loads(msg["data"])
        assert data["nerves"][0]["name"] == "test_nerve"
        assert data["nerves"][0]["qualified"] is True

    def test_silences_redis_errors(self, _patch_module_globals):
        """Never raises even when Redis is broken."""
        from arqitect.critic.qualify_nerve import _publish_progress
        with patch.object(_patch_module_globals["redis"], "publish", side_effect=ConnectionError):
            # Must not raise
            _publish_progress("n", 0.5, None, [], 1, 3)
