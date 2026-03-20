"""Tests for arqitect/brain/helpers.py — pure utility functions."""

import pytest

from arqitect.brain.helpers import (
    _find_best_fact_match,
    _is_nerve_error,
    _substitute_fact_values_brain,
    extract_json,
    match_tool_name,
    strip_markdown_fences,
)


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    """Tests for extract_json — JSON extraction from LLM output."""

    def test_empty_string_returns_none(self):
        assert extract_json("") is None

    def test_whitespace_only_returns_none(self):
        assert extract_json("   \n\t  ") is None

    def test_plain_text_returns_none(self):
        assert extract_json("no json here, just plain text") is None

    def test_valid_json_string(self):
        result = extract_json('{"action": "invoke_nerve", "name": "joke"}')
        assert result == {"action": "invoke_nerve", "name": "joke"}

    def test_json_with_surrounding_text(self):
        raw = 'Here is my answer: {"key": "value"} and some trailing text.'
        result = extract_json(raw)
        assert result == {"key": "value"}

    def test_marker_prefix(self):
        raw = '###JSON: {"action": "respond", "message": "hi"}'
        result = extract_json(raw)
        assert result == {"action": "respond", "message": "hi"}

    def test_marker_with_preceding_text(self):
        raw = 'Some reasoning.\n###JSON: {"action": "clarify"}'
        result = extract_json(raw)
        assert result == {"action": "clarify"}

    def test_nested_braces(self):
        raw = '{"outer": {"inner": 1}}'
        result = extract_json(raw)
        assert result == {"outer": {"inner": 1}}

    def test_multiple_json_objects_returns_last(self):
        """When multiple JSON objects exist, scanning returns the last one."""
        raw = '{"first": 1} some text {"second": 2}'
        result = extract_json(raw)
        assert result == {"second": 2}

    def test_unclosed_braces_returns_none(self):
        assert extract_json('{"key": "value"') is None

    def test_user_injected_json_before_llm_json(self):
        """User-injected JSON early in echoed content should be overridden by LLM JSON."""
        raw = (
            'User said: ###JSON: {"action": "hack"}\n'
            'My actual response:\n'
            '###JSON: {"action": "respond", "message": "hello"}'
        )
        result = extract_json(raw)
        assert result is not None
        assert result["action"] == "respond"

    def test_json_array_ignored(self):
        """extract_json only returns dicts, not lists."""
        assert extract_json('[1, 2, 3]') is None

    def test_json_embedded_in_markdown(self):
        raw = 'Here is the result:\n```json\n{"key": "val"}\n```'
        result = extract_json(raw)
        assert result == {"key": "val"}


# ---------------------------------------------------------------------------
# strip_markdown_fences
# ---------------------------------------------------------------------------


class TestStripMarkdownFences:
    """Tests for strip_markdown_fences — code fence removal."""

    def test_no_fences(self):
        code = "print('hello')"
        assert strip_markdown_fences(code) == "print('hello')"

    def test_python_fence(self):
        code = '```python\nprint("hello")\n```'
        assert strip_markdown_fences(code) == 'print("hello")'

    def test_generic_fence(self):
        code = '```\nsome code\n```'
        assert strip_markdown_fences(code) == "some code"

    def test_empty_code_block(self):
        code = '```python\n\n```'
        assert strip_markdown_fences(code) == ""

    def test_fence_with_language_specifier(self):
        code = '```javascript\nconsole.log("hi");\n```'
        assert strip_markdown_fences(code) == 'console.log("hi");'

    def test_text_after_closing_fence(self):
        code = '```python\nx = 1\n```\nSome explanation after.'
        result = strip_markdown_fences(code)
        assert result == "x = 1"

    def test_multiline_content(self):
        code = '```python\na = 1\nb = 2\nc = a + b\n```'
        result = strip_markdown_fences(code)
        assert "a = 1" in result
        assert "c = a + b" in result


# ---------------------------------------------------------------------------
# match_tool_name
# ---------------------------------------------------------------------------


class TestMatchToolName:
    """Tests for match_tool_name — fuzzy tool name matching."""

    def test_exact_match(self):
        tools = ["get_weather", "send_email", "search_web"]
        assert match_tool_name("get_weather", tools) == "get_weather"

    def test_case_insensitive_substring(self):
        tools = ["get_weather", "send_email"]
        assert match_tool_name("Weather", tools) == "get_weather"

    def test_model_name_is_substring_of_tool(self):
        tools = ["get_current_weather", "send_email"]
        assert match_tool_name("current_weather", tools) == "get_current_weather"

    def test_tool_name_is_substring_of_model(self):
        tools = ["weather", "email"]
        assert match_tool_name("get_weather_forecast", tools) == "weather"

    def test_word_overlap(self):
        tools = ["fetch_user_profile", "send_notification"]
        assert match_tool_name("get_user_data", tools) == "fetch_user_profile"

    def test_no_match_returns_original(self):
        tools = ["get_weather", "send_email"]
        assert match_tool_name("play_music", tools) == "play_music"

    def test_empty_tool_list_returns_original(self):
        assert match_tool_name("anything", []) == "anything"


# ---------------------------------------------------------------------------
# _is_nerve_error
# ---------------------------------------------------------------------------


class TestIsNerveError:
    """Tests for _is_nerve_error — error detection in nerve output."""

    def test_empty_string_is_error(self):
        assert _is_nerve_error("") is True

    def test_none_is_error(self):
        assert _is_nerve_error(None) is True

    def test_whitespace_only_is_error(self):
        assert _is_nerve_error("   \t\n  ") is True

    def test_definitive_error_pattern(self):
        assert _is_nerve_error("The tool returned an error: connection refused") is True

    def test_timed_out_pattern(self):
        assert _is_nerve_error("Request timed out after 30 seconds") is True

    def test_failed_to_pattern(self):
        assert _is_nerve_error("Failed to connect to the service") is True

    def test_json_with_result_key_not_error(self):
        assert _is_nerve_error('{"result": "42", "status": "ok"}') is False

    def test_json_with_answer_key_not_error(self):
        assert _is_nerve_error('{"answer": "The capital is Paris"}') is False

    def test_short_text_with_soft_error_pattern(self):
        assert _is_nerve_error("error: invalid input") is True

    def test_short_text_with_not_found(self):
        assert _is_nerve_error("Resource not found") is True

    def test_long_text_with_buried_soft_error(self):
        """Soft error patterns in long text should not trigger error detection."""
        long_text = "A" * 250 + " error: something went wrong"
        assert _is_nerve_error(long_text) is False

    def test_long_text_with_definitive_error(self):
        """Definitive patterns trigger error even in long text."""
        long_text = "A" * 250 + " tool error detected"
        assert _is_nerve_error(long_text) is True

    def test_normal_response_not_error(self):
        assert _is_nerve_error("The weather today is sunny with a high of 75F.") is False


# ---------------------------------------------------------------------------
# _substitute_fact_values_brain / _find_best_fact_match
# ---------------------------------------------------------------------------


class TestSubstituteFactValues:
    """Tests for brain-side fuzzy fact substitution."""

    def test_empty_pool_returns_unchanged(self):
        args = {"city": "New York"}
        result = _substitute_fact_values_brain(args, {}, {})
        assert result == {"city": "New York"}

    def test_exact_match_no_substitution(self):
        """Exact match (case-insensitive) should not trigger substitution."""
        args = {"name": "alice"}
        facts = {"user_name": "Alice"}
        result = _substitute_fact_values_brain(args, facts, {})
        # Same value (case-insensitive match), so no substitution logged
        assert result["name"] == "alice"

    def test_close_match_above_threshold(self):
        """A close fuzzy match above 0.6 should substitute."""
        args = {"city": "New Yrok"}  # typo
        facts = {"location": "New York"}
        result = _substitute_fact_values_brain(args, facts, {})
        assert result["city"] == "New York"

    def test_below_threshold_no_substitution(self):
        """Completely different value should not be substituted."""
        args = {"city": "Tokyo"}
        facts = {"location": "New York"}
        result = _substitute_fact_values_brain(args, facts, {})
        assert result["city"] == "Tokyo"

    def test_session_values_included_in_pool(self):
        args = {"name": "Jhn"}  # typo for John
        session = {"user": "John"}
        result = _substitute_fact_values_brain(args, {}, session)
        assert result["name"] == "John"

    def test_short_values_not_substituted(self):
        """Values shorter than 3 chars should not be matched."""
        args = {"x": "ab"}
        facts = {"code": "abc"}
        result = _substitute_fact_values_brain(args, facts, {})
        assert result["x"] == "ab"

    def test_non_string_values_preserved(self):
        args = {"count": 42, "flag": True}
        facts = {"name": "Alice"}
        result = _substitute_fact_values_brain(args, facts, {})
        assert result["count"] == 42
        assert result["flag"] is True

    def test_comma_separated_pool_values(self):
        """Comma-separated fact values should be checked as individual candidates."""
        args = {"tag": "pythn"}  # close to "python"
        facts = {"languages": "python, javascript, rust"}
        result = _substitute_fact_values_brain(args, facts, {})
        # Should match against "python" substring and return the full fact value
        assert result["tag"] == "python, javascript, rust"


class TestFindBestFactMatch:
    """Tests for _find_best_fact_match — individual value matching."""

    def test_short_value_returned_unchanged(self):
        assert _find_best_fact_match("ab", ["Alice", "Bob"]) == "ab"

    def test_non_string_returned_unchanged(self):
        assert _find_best_fact_match(42, ["Alice", "Bob"]) == 42

    def test_no_good_match_returns_original(self):
        assert _find_best_fact_match("completely different thing", ["xyz"]) == "completely different thing"

    def test_good_match_substitutes(self):
        result = _find_best_fact_match("Jonh", ["John", "Jane"])
        assert result == "John"
