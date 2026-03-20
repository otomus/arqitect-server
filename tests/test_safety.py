"""Tests for arqitect.brain.safety — LLM-based input/output screening."""

import json
from unittest.mock import patch

import pytest

from arqitect.brain.safety import (
    _CLASSIFY_MAX_CHARS,
    _CLASSIFY_SYSTEM,
    _FALLBACK_REFUSAL,
    _CATEGORY_LABELS,
    _classify,
    _generate_refusal,
    check_input,
    check_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_classify_safe(*_args, **_kwargs):
    """Simulate LLM returning safe classification."""
    return json.dumps({"safe": True})


def _mock_classify_unsafe_sexual(*_args, **_kwargs):
    """Simulate LLM returning unsafe/sexual classification."""
    return json.dumps({"safe": False, "category": "sexual"})


def _mock_classify_unsafe_hate(*_args, **_kwargs):
    return json.dumps({"safe": False, "category": "hate"})


def _mock_classify_unsafe_harmful(*_args, **_kwargs):
    return json.dumps({"safe": False, "category": "harmful"})


def _mock_classify_unsafe_sensitive(*_args, **_kwargs):
    return json.dumps({"safe": False, "category": "sensitive_data"})


def _mock_classify_unsafe_nsfw_url(*_args, **_kwargs):
    return json.dumps({"safe": False, "category": "nsfw_url"})


def _mock_refusal(*_args, **_kwargs):
    return "I can't do that, sorry."


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

class TestClassify:
    """Tests for the internal LLM classification helper."""

    @patch("arqitect.brain.safety.generate_for_role", return_value='{"safe": true}')
    def test_returns_safe_dict_for_safe_content(self, mock_gen):
        result = _classify("hello world")
        assert result == {"safe": True}

    @patch("arqitect.brain.safety.generate_for_role", return_value='{"safe": false, "category": "hate"}')
    def test_returns_unsafe_dict_with_category(self, mock_gen):
        result = _classify("some hateful text")
        assert result["safe"] is False
        assert result["category"] == "hate"

    @patch("arqitect.brain.safety.generate_for_role", return_value="Error: model unavailable")
    def test_defaults_to_safe_on_error_prefix(self, mock_gen):
        result = _classify("anything")
        assert result == {"safe": True}

    @patch("arqitect.brain.safety.generate_for_role", return_value="not json at all")
    def test_defaults_to_safe_on_invalid_json(self, mock_gen):
        result = _classify("anything")
        assert result == {"safe": True}

    @patch("arqitect.brain.safety.generate_for_role", return_value='{"unrelated": 42}')
    def test_defaults_to_safe_when_safe_key_missing(self, mock_gen):
        result = _classify("anything")
        assert result == {"safe": True}

    @patch("arqitect.brain.safety.generate_for_role", return_value='{"safe": "yes"}')
    def test_defaults_to_safe_when_safe_is_not_bool(self, mock_gen):
        result = _classify("anything")
        assert result == {"safe": True}

    @patch("arqitect.brain.safety.generate_for_role", side_effect=RuntimeError("boom"))
    def test_defaults_to_safe_on_exception(self, mock_gen):
        result = _classify("anything")
        assert result == {"safe": True}

    @patch("arqitect.brain.safety.generate_for_role", return_value='{"safe": true}')
    def test_truncates_long_text(self, mock_gen):
        long_text = "x" * 10000
        _classify(long_text)
        call_args = mock_gen.call_args
        prompt_text = call_args[0][1]
        # The text portion in the prompt should be truncated
        assert len(prompt_text) <= _CLASSIFY_MAX_CHARS + 200  # prompt template overhead


# ---------------------------------------------------------------------------
# _generate_refusal
# ---------------------------------------------------------------------------

class TestGenerateRefusal:
    """Tests for refusal message generation."""

    @patch("arqitect.brain.safety.generate_for_role", return_value="Sorry, I can't help with that.")
    def test_returns_llm_refusal(self, mock_gen):
        result = _generate_refusal("hate", "some hateful message")
        assert result == "Sorry, I can't help with that."

    @patch("arqitect.brain.safety.generate_for_role", return_value="Error: failed")
    def test_returns_fallback_on_error_prefix(self, mock_gen):
        result = _generate_refusal("hate", "some message")
        assert result == _FALLBACK_REFUSAL

    @patch("arqitect.brain.safety.generate_for_role", return_value="")
    def test_returns_fallback_on_empty_response(self, mock_gen):
        result = _generate_refusal("hate", "some message")
        assert result == _FALLBACK_REFUSAL

    @patch("arqitect.brain.safety.generate_for_role", return_value="   ")
    def test_returns_fallback_on_whitespace_only(self, mock_gen):
        result = _generate_refusal("hate", "some message")
        assert result == _FALLBACK_REFUSAL

    @patch("arqitect.brain.safety.generate_for_role", side_effect=Exception("timeout"))
    def test_returns_fallback_on_exception(self, mock_gen):
        result = _generate_refusal("hate", "some message")
        assert result == _FALLBACK_REFUSAL

    @patch("arqitect.brain.safety.generate_for_role", return_value="Nope.")
    def test_uses_category_label_in_prompt(self, mock_gen):
        _generate_refusal("sexual", "explicit content here")
        call_args = mock_gen.call_args
        prompt = call_args[0][1]
        assert _CATEGORY_LABELS["sexual"] in prompt

    @patch("arqitect.brain.safety.generate_for_role", return_value="Nope.")
    def test_unknown_category_passes_raw_string(self, mock_gen):
        _generate_refusal("new_category", "some text")
        call_args = mock_gen.call_args
        prompt = call_args[0][1]
        assert "new_category" in prompt

    @patch("arqitect.brain.safety.generate_for_role", return_value="Nope.")
    def test_snippet_truncated_to_120_chars(self, mock_gen):
        long_text = "a" * 300
        _generate_refusal("hate", long_text)
        call_args = mock_gen.call_args
        prompt = call_args[0][1]
        # The snippet should not contain the full 300 chars
        assert "a" * 121 not in prompt


# ---------------------------------------------------------------------------
# check_input
# ---------------------------------------------------------------------------

class TestCheckInput:
    """Tests for the public input screening API."""

    def test_empty_string_is_safe(self):
        is_safe, reason = check_input("")
        assert is_safe is True
        assert reason == ""

    def test_none_is_safe(self):
        is_safe, reason = check_input(None)
        assert is_safe is True
        assert reason == ""

    def test_whitespace_only_is_safe(self):
        is_safe, reason = check_input("   \n\t  ")
        assert is_safe is True
        assert reason == ""

    @patch("arqitect.brain.safety._classify", return_value={"safe": True})
    def test_safe_message_passes(self, mock_cls):
        is_safe, reason = check_input("Tell me a joke")
        assert is_safe is True
        assert reason == ""

    @patch("arqitect.brain.safety._generate_refusal", return_value="Blocked.")
    @patch("arqitect.brain.safety._classify", return_value={"safe": False, "category": "hate"})
    def test_unsafe_message_blocked_with_refusal(self, mock_cls, mock_ref):
        is_safe, reason = check_input("hateful content")
        assert is_safe is False
        assert reason == "Blocked."

    @patch("arqitect.brain.safety._generate_refusal", return_value="No.")
    @patch("arqitect.brain.safety._classify", return_value={"safe": False})
    def test_unsafe_without_category_uses_unknown(self, mock_cls, mock_ref):
        is_safe, reason = check_input("bad stuff")
        assert is_safe is False
        mock_ref.assert_called_once()
        # category arg should be "unknown"
        assert mock_ref.call_args[0][0] == "unknown"


# ---------------------------------------------------------------------------
# check_output
# ---------------------------------------------------------------------------

class TestCheckOutput:
    """Tests for the public output screening API."""

    def test_empty_response_is_safe(self):
        is_safe, reason = check_output("")
        assert is_safe is True
        assert reason == ""

    def test_none_response_is_safe(self):
        is_safe, reason = check_output(None)
        assert is_safe is True
        assert reason == ""

    @patch("arqitect.brain.safety._classify", return_value={"safe": True})
    def test_safe_response_passes(self, mock_cls):
        is_safe, reason = check_output("Here is a joke for you.")
        assert is_safe is True
        assert reason == ""

    @patch("arqitect.brain.safety._generate_refusal", return_value="Blocked output.")
    @patch("arqitect.brain.safety._classify", return_value={"safe": False, "category": "sensitive_data"})
    def test_unsafe_response_blocked(self, mock_cls, mock_ref):
        is_safe, reason = check_output("Here is your API key: sk-12345")
        assert is_safe is False
        assert reason == "Blocked output."

    @patch("arqitect.brain.safety._classify", return_value={"safe": True})
    def test_no_media_urls_passes(self, mock_cls):
        is_safe, reason = check_output("safe text", media_urls=None)
        assert is_safe is True

    @patch("arqitect.brain.safety._classify", return_value={"safe": True})
    def test_empty_media_urls_passes(self, mock_cls):
        is_safe, reason = check_output("safe text", media_urls=[])
        assert is_safe is True

    @patch("arqitect.brain.safety._classify")
    def test_unsafe_media_url_blocks_response(self, mock_cls):
        # First call for response text: safe; second call for URLs: unsafe
        mock_cls.side_effect = [
            {"safe": True},
            {"safe": False, "category": "nsfw_url"},
        ]
        with patch("arqitect.brain.safety._generate_refusal", return_value="NSFW blocked."):
            is_safe, reason = check_output("see image", media_urls=["https://nsfw.example.com/img.jpg"])
        assert is_safe is False
        assert reason == "NSFW blocked."

    @patch("arqitect.brain.safety._classify", return_value={"safe": True})
    def test_media_urls_with_none_entries_filtered(self, mock_cls):
        is_safe, reason = check_output("text", media_urls=[None, "", None])
        assert is_safe is True
        # _classify called once for text, urls_text is empty so no second call
        assert mock_cls.call_count == 1

    @patch("arqitect.brain.safety._classify")
    def test_safe_media_urls_pass(self, mock_cls):
        mock_cls.side_effect = [
            {"safe": True},  # response text
            {"safe": True},  # media urls
        ]
        is_safe, reason = check_output("text", media_urls=["https://example.com/photo.jpg"])
        assert is_safe is True
        assert mock_cls.call_count == 2


# ---------------------------------------------------------------------------
# Category labels
# ---------------------------------------------------------------------------

class TestCategoryLabels:
    """Verify all expected categories have labels."""

    def test_all_categories_have_labels(self):
        expected = {"sexual", "hate", "harmful", "sensitive_data", "nsfw_url"}
        assert set(_CATEGORY_LABELS.keys()) == expected

    def test_labels_are_non_empty_strings(self):
        for key, label in _CATEGORY_LABELS.items():
            assert isinstance(label, str)
            assert len(label) > 0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify safety module constants."""

    def test_classify_max_chars_is_positive(self):
        assert _CLASSIFY_MAX_CHARS > 0

    def test_fallback_refusal_is_non_empty(self):
        assert len(_FALLBACK_REFUSAL) > 0

    def test_classify_system_mentions_json(self):
        assert "JSON" in _CLASSIFY_SYSTEM
