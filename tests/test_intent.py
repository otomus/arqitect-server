"""Tests for arqitect.brain.intent — intent classification."""

from unittest.mock import patch

import pytest

from arqitect.types import IntentType


class TestClassifyIntent:
    """Tests for classify_intent."""

    def _call(self, llm_return, extract_return):
        """Invoke classify_intent with mocked LLM and JSON extractor.

        Args:
            llm_return: Raw string the LLM mock returns.
            extract_return: Parsed dict that extract_json returns (or None).

        Returns:
            The dict returned by classify_intent.
        """
        with patch("arqitect.brain.intent.llm_generate", return_value=llm_return), \
             patch("arqitect.brain.intent.extract_json", return_value=extract_return):
            from arqitect.brain.intent import classify_intent
            return classify_intent("some user message")

    def test_valid_workflow_json(self):
        """LLM returns valid workflow intent with category."""
        raw = '{"type": "workflow", "category": "development"}'
        parsed = {"type": "workflow", "category": "development"}

        result = self._call(raw, parsed)

        assert result["type"] == IntentType.WORKFLOW
        assert result["category"] == "development"

    def test_valid_direct_json(self):
        """LLM returns valid direct intent."""
        raw = '{"type": "direct"}'
        parsed = {"type": "direct"}

        result = self._call(raw, parsed)

        assert result["type"] == IntentType.DIRECT

    def test_non_json_text_falls_back_to_direct(self):
        """LLM returns plain text — extract_json returns None."""
        result = self._call("I think this is a greeting", None)

        assert result == {"type": IntentType.DIRECT}

    def test_invalid_type_field_falls_back_to_direct(self):
        """LLM returns JSON with an unrecognized type value."""
        parsed = {"type": "unknown_type"}

        result = self._call('{"type": "unknown_type"}', parsed)

        assert result == {"type": IntentType.DIRECT}

    def test_empty_string_falls_back_to_direct(self):
        """LLM returns empty string — extract_json returns None."""
        result = self._call("", None)

        assert result == {"type": IntentType.DIRECT}

    def test_workflow_with_category_preserved(self):
        """Category field is preserved in the returned dict."""
        parsed = {"type": "workflow", "category": "debugging"}

        result = self._call('{"type": "workflow", "category": "debugging"}', parsed)

        assert result["type"] == IntentType.WORKFLOW
        assert result["category"] == "debugging"

    def test_extract_json_returns_empty_dict(self):
        """extract_json returns {} (no 'type' key) — falls back to direct."""
        result = self._call("{}", {})

        assert result == {"type": IntentType.DIRECT}

    def test_llm_called_with_correct_args(self):
        """Verify llm_generate receives the model and system prompt."""
        with patch("arqitect.brain.intent.llm_generate", return_value="{}") as mock_llm, \
             patch("arqitect.brain.intent.extract_json", return_value=None), \
             patch("arqitect.brain.intent.BRAIN_MODEL", "test-model"):
            from arqitect.brain.intent import classify_intent
            classify_intent("build me a web app")

        mock_llm.assert_called_once()
        args, kwargs = mock_llm.call_args
        assert args[0] == "test-model"
        assert "build me a web app" in args[1]
        assert "system" in kwargs or len(args) >= 3
