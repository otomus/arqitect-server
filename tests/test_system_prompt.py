"""Tests for system_prompt unwrapping and plain-text conversion.

Ensures that system_prompt values written to context.json are always
plain text strings — never JSON objects, never double-wrapped.

Reference format (from community nerves like academic_research_nerve):
  "system_prompt": "Find and summarize academic papers on a topic. ..."
"""

import json
import os
from unittest.mock import patch

import pytest

from arqitect.brain.consolidate import (
    _system_prompt_to_plain_text,
    _adapt_prompt_for_size,
)
from arqitect.memory.cold import _unwrap_system_prompt


def _is_plain_text(value: str) -> bool:
    """Return True if value is plain text (no JSON object on the first line)."""
    first_line = value.strip().split("\n")[0]
    try:
        obj = json.loads(first_line)
        return not isinstance(obj, dict)
    except (json.JSONDecodeError, TypeError):
        return True


# ---------------------------------------------------------------------------
# _unwrap_system_prompt (storage guard)
# ---------------------------------------------------------------------------

class TestUnwrapSystemPrompt:

    @pytest.mark.parametrize("raw, expected", [
        # Known JSON wrapper shapes from real LLM output
        ('{"system_prompt": "You are an expert."}', "You are an expert."),
        ('{"name": "x", "description": "Brief intros."}', "Brief intros."),
        ('{"goal": "Summarize papers."}', "Summarize papers."),
        # Plain text passes through
        ("You are an expert.", "You are an expert."),
        # JSON + trailing rules can't be unwrapped (json.loads fails)
        ('{"name": "x", "description": "E."}\nRule: specific', '{"name": "x", "description": "E."}\nRule: specific'),
    ])
    def test_known_shapes(self, raw, expected):
        assert _unwrap_system_prompt(raw) == expected

    # ── Edge cases ────────────────────────────────────────────────────────
    def test_none(self):
        assert _unwrap_system_prompt(None) is None

    def test_empty_string(self):
        assert _unwrap_system_prompt("") == ""

    def test_whitespace_only(self):
        assert _unwrap_system_prompt("   \n\t  ") == "   \n\t  "

    def test_integer_input(self):
        """Wrong type — should not crash."""
        assert _unwrap_system_prompt(42) == 42

    def test_json_array(self):
        assert _unwrap_system_prompt('[1, 2, 3]') == '[1, 2, 3]'

    def test_json_string_literal(self):
        assert _unwrap_system_prompt('"just a string"') == '"just a string"'

    def test_nested_json_in_description(self):
        """Description value itself contains braces — should still extract it."""
        raw = '{"description": "Parse {user_input} and return {result}"}'
        assert _unwrap_system_prompt(raw) == "Parse {user_input} and return {result}"

    def test_empty_description(self):
        """Empty description string — unwrap returns it as-is."""
        raw = '{"name": "x", "description": ""}'
        assert _unwrap_system_prompt(raw) == ""

    def test_system_prompt_value_is_not_string(self):
        """system_prompt key exists but value is not a string — pass through."""
        raw = '{"system_prompt": 42}'
        assert _unwrap_system_prompt(raw) == raw

    def test_unicode_content(self):
        raw = '{"system_prompt": "Résumé des données météo. 日本語テスト"}'
        assert _unwrap_system_prompt(raw) == "Résumé des données météo. 日本語テスト"

    def test_escaped_quotes_in_value(self):
        raw = '{"system_prompt": "Say \\"hello\\" to the user."}'
        assert _unwrap_system_prompt(raw) == 'Say "hello" to the user.'

    def test_very_long_string(self):
        long_text = "word " * 10000
        raw = json.dumps({"system_prompt": long_text})
        assert _unwrap_system_prompt(raw) == long_text


# ---------------------------------------------------------------------------
# _system_prompt_to_plain_text (output pipeline)
# ---------------------------------------------------------------------------

class TestSystemPromptToPlainText:

    @pytest.mark.parametrize("raw", [
        '{"system_prompt": "You are an expert."}',
        '{"name": "x", "description": "Brief intros."}',
        '{"goal": "Summarize", "output": "JSON", "boundary": "No medical"}',
        '{"system_prompt": "Expert."}\nRule: be specific',
        '{"name": "x", "description": "Intros."}\nRule: use tools\nRule: be brief',
        "Plain text prompt.",
    ])
    def test_output_is_always_plain_text(self, raw):
        """The one invariant: output must never contain a JSON object."""
        result = _system_prompt_to_plain_text(raw)
        assert _is_plain_text(result), f"JSON object in output for input: {raw!r}\nGot: {result!r}"

    def test_preserves_trailing_rules(self):
        raw = '{"description": "Expert."}\nRule: be specific\nRule: no jargon'
        result = _system_prompt_to_plain_text(raw)
        assert "Rule: be specific" in result
        assert "Rule: no jargon" in result

    def test_goal_output_boundary_flattened(self):
        raw = '{"goal": "Summarize", "output": "JSON", "boundary": "No medical"}'
        result = _system_prompt_to_plain_text(raw, tool_names=["search"])
        assert "Summarize" in result
        assert "Output: JSON" in result
        assert "Boundary: No medical" in result
        assert "Available tools: search" in result

    # ── Edge cases ────────────────────────────────────────────────────────
    def test_empty_string(self):
        assert _system_prompt_to_plain_text("") == ""

    def test_whitespace_only(self):
        assert _system_prompt_to_plain_text("   ") == "   "

    def test_truncated_json(self):
        """Partial JSON — should pass through as plain text."""
        raw = '{"system_prompt": "trunc'
        result = _system_prompt_to_plain_text(raw)
        assert result == raw

    def test_json_with_extra_fields(self):
        """Unknown extra keys alongside description — still extracts description."""
        raw = '{"name": "x", "description": "Expert.", "version": "1.0", "extra": true}'
        result = _system_prompt_to_plain_text(raw)
        assert result == "Expert."

    def test_deeply_nested_json(self):
        """Nested objects should not crash the brace scanner."""
        raw = '{"system_prompt": {"inner": {"deep": "value"}}}'
        result = _system_prompt_to_plain_text(raw)
        # system_prompt value is a dict not a string — should not unwrap
        assert isinstance(result, str)

    def test_multiple_json_objects(self):
        """Two JSON objects concatenated — only first should be parsed."""
        raw = '{"description": "First."}\n{"description": "Second."}'
        result = _system_prompt_to_plain_text(raw)
        assert "First." in result

    def test_braces_in_plain_text(self):
        """Plain text containing braces should not be treated as JSON."""
        raw = "Format: {name} - {value}. Use this template."
        result = _system_prompt_to_plain_text(raw)
        assert result == raw

    def test_newlines_and_tabs_in_json_value(self):
        raw = '{"system_prompt": "Line 1.\\nLine 2.\\tTabbed."}'
        result = _system_prompt_to_plain_text(raw)
        assert "Line 1.\nLine 2.\tTabbed." == result

    def test_empty_json_object(self):
        result = _system_prompt_to_plain_text("{}")
        assert result == "{}"  # No recognized keys — pass through


# ---------------------------------------------------------------------------
# _adapt_prompt_for_size (size-class integration)
# ---------------------------------------------------------------------------

class TestAdaptPromptForSize:

    @pytest.mark.parametrize("size_class", ["large", "medium", "small", "tinylm"])
    @pytest.mark.parametrize("raw", [
        '{"system_prompt": "Expert in science."}',
        '{"name": "x", "description": "Intros."}\nRule: use tools',
        "Plain text prompt.",
        "",
    ])
    def test_output_is_always_plain_text(self, size_class, raw):
        result = _adapt_prompt_for_size(raw, size_class, tool_names=["search"])
        assert _is_plain_text(result), (
            f"JSON in output for size={size_class}, input={raw!r}\nGot: {result!r}"
        )


# ---------------------------------------------------------------------------
# _write_nerve_adapter_files (integration — worst-case input)
# ---------------------------------------------------------------------------

class TestWriteNerveAdapterFiles:
    """One integration test with the worst-case input from real data:
    JSON wrapper + trailing rules (the exact pattern from PR #68)."""

    def test_all_context_json_files_have_plain_text_system_prompt(self, tmp_path):
        import threading
        from arqitect.brain.consolidate import Dreamstate

        nerve_dir = str(tmp_path / "nerves" / "test_nerve")
        os.makedirs(nerve_dir, exist_ok=True)

        with patch.object(Dreamstate, '__init__', lambda self: None):
            ds = Dreamstate()
            ds._interrupted = threading.Event()

        # Worst case: JSON wrapper + accumulated rules (from actual PR #68)
        broken_prompt = (
            '{"name": "introduction_nerve", "description": "Specialized expert in introductions."}'
            '\nRule: Always use search tools.'
            '\nRule: Handle special characters gracefully.'
        )

        mock_meta = {
            "description": "Test nerve",
            "system_prompt": broken_prompt,
            "examples": [{"input": "hi", "output": "hello"}],
            "role": "tool",
            "origin": "local",
        }

        with patch("arqitect.brain.consolidate.mem") as mock_mem, \
             patch("arqitect.brain.consolidate.NERVES_DIR", str(tmp_path / "nerves")), \
             patch("arqitect.brain.adapters.SIZE_CLASSES", ["large", "medium", "small", "tinylm"]), \
             patch("arqitect.brain.adapters.get_active_variant", return_value="medium"), \
             patch("arqitect.brain.adapters.get_model_name_for_role", return_value="test-model"), \
             patch("arqitect.brain.adapters.get_temperature", return_value=0.7), \
             patch("arqitect.brain.adapters.build_meta_json", return_value={"model": "test"}):
            mock_mem.cold.get_nerve_metadata.return_value = mock_meta
            mock_mem.cold.get_nerve_tools_with_counts.return_value = []
            mock_mem.cold.get_qualification.return_value = {"score": 0.9}
            ds._write_nerve_adapter_files("test_nerve", "tool", nerve_dir)

        context_files_found = 0
        for root, _dirs, files in os.walk(nerve_dir):
            for f in files:
                if f != "context.json":
                    continue
                context_files_found += 1
                path = os.path.join(root, f)
                with open(path) as fh:
                    ctx = json.load(fh)
                sp = ctx["system_prompt"]
                rel = os.path.relpath(path, nerve_dir)
                assert isinstance(sp, str), f"{rel}: system_prompt is not a string"
                assert _is_plain_text(sp), f"{rel}: system_prompt contains JSON object:\n{sp[:200]}"
                assert "Specialized expert in introductions" in sp, f"{rel}: lost the prompt content"

        assert context_files_found >= 4, f"Expected at least 4 context.json files, found {context_files_found}"
