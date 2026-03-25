"""Tests for arqitect.brain.safety — LLM-based input/output screening."""

import json
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
from tests.conftest import FakeLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_TARGET = "arqitect.brain.safety.generate_for_role"


def _make_safety_llm(responses: list) -> FakeLLM:
    """Build a FakeLLM pre-loaded with safety classification responses.

    Args:
        responses: List of (substring, response_text) or
                   (substring, response_text, reuse) tuples.

    Returns:
        FakeLLM instance. Use _patch_generate() to install it; that
        wrapper handles the extra kwargs generate_for_role passes.
    """
    return FakeLLM(responses)


def _patch_generate(fake: FakeLLM):
    """Patch generate_for_role at its import site inside the safety module.

    Uses new_callable to replace generate_for_role with a thin wrapper
    that forwards only (role, prompt, system) to FakeLLM, dropping the
    extra keyword args (max_tokens, temperature, json_mode) that
    generate_for_role accepts but FakeLLM does not.

    Args:
        fake: A FakeLLM (created via _make_safety_llm).

    Returns:
        A unittest.mock.patch context manager.
    """

    def _wrapper(role, prompt, system="", **_kwargs):
        return fake(role, prompt, system)

    return patch(_MOCK_TARGET, side_effect=_wrapper)


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestClassify:
    """Tests for the internal LLM classification helper."""

    def test_returns_safe_dict_for_safe_content(self):
        fake = _make_safety_llm([("safety filter", '{"safe": true}', True)])
        with _patch_generate(fake):
            result = _classify("hello world")
        assert result == {"safe": True}

    def test_returns_unsafe_dict_with_category(self):
        fake = _make_safety_llm([
            ("safety filter", '{"safe": false, "category": "hate"}', True),
        ])
        with _patch_generate(fake):
            result = _classify("some hateful text")
        assert result["safe"] is False
        assert result["category"] == "hate"

    def test_defaults_to_safe_on_error_prefix(self):
        fake = _make_safety_llm([
            ("safety filter", "Error: model unavailable", True),
        ])
        with _patch_generate(fake):
            result = _classify("anything")
        assert result == {"safe": True}

    def test_defaults_to_safe_on_invalid_json(self):
        fake = _make_safety_llm([
            ("safety filter", "not json at all", True),
        ])
        with _patch_generate(fake):
            result = _classify("anything")
        assert result == {"safe": True}

    def test_defaults_to_safe_when_safe_key_missing(self):
        fake = _make_safety_llm([
            ("safety filter", '{"unrelated": 42}', True),
        ])
        with _patch_generate(fake):
            result = _classify("anything")
        assert result == {"safe": True}

    def test_defaults_to_safe_when_safe_is_not_bool(self):
        fake = _make_safety_llm([
            ("safety filter", '{"safe": "yes"}', True),
        ])
        with _patch_generate(fake):
            result = _classify("anything")
        assert result == {"safe": True}

    def test_defaults_to_safe_on_exception(self):
        with patch(_MOCK_TARGET, side_effect=RuntimeError("boom")):
            result = _classify("anything")
        assert result == {"safe": True}

    def test_truncates_long_text(self):
        fake = _make_safety_llm([("safety filter", '{"safe": true}', True)])
        with _patch_generate(fake):
            long_text = "x" * 10_000
            _classify(long_text)
        # The prompt sent to FakeLLM should contain truncated text
        assert len(fake.calls) == 1
        prompt_text = fake.calls[0]["prompt"]
        assert len(prompt_text) <= _CLASSIFY_MAX_CHARS + 200  # template overhead

    @given(text=st.text(min_size=1, max_size=500))
    @settings(max_examples=30, deadline=5000)
    def test_never_raises_on_arbitrary_input(self, text):
        """_classify must never propagate exceptions to the caller."""
        fake = _make_safety_llm([("safety filter", '{"safe": true}', True)])
        with _patch_generate(fake):
            result = _classify(text)
        assert isinstance(result, dict)
        assert "safe" in result


# ---------------------------------------------------------------------------
# _generate_refusal
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGenerateRefusal:
    """Tests for refusal message generation."""

    def test_returns_llm_refusal(self):
        fake = _make_safety_llm([
            ("blocked", "Sorry, I can't help with that.", True),
        ])
        with _patch_generate(fake):
            result = _generate_refusal("hate", "some hateful message")
        assert result == "Sorry, I can't help with that."

    def test_returns_fallback_on_error_prefix(self):
        fake = _make_safety_llm([
            ("blocked", "Error: failed", True),
        ])
        with _patch_generate(fake):
            result = _generate_refusal("hate", "some message")
        assert result == _FALLBACK_REFUSAL

    def test_returns_fallback_on_empty_response(self):
        fake = _make_safety_llm([("blocked", "", True)])
        with _patch_generate(fake):
            result = _generate_refusal("hate", "some message")
        assert result == _FALLBACK_REFUSAL

    def test_returns_fallback_on_whitespace_only(self):
        fake = _make_safety_llm([("blocked", "   ", True)])
        with _patch_generate(fake):
            result = _generate_refusal("hate", "some message")
        assert result == _FALLBACK_REFUSAL

    def test_returns_fallback_on_exception(self):
        with patch(_MOCK_TARGET, side_effect=Exception("timeout")):
            result = _generate_refusal("hate", "some message")
        assert result == _FALLBACK_REFUSAL

    def test_uses_category_label_in_prompt(self):
        fake = _make_safety_llm([("blocked", "Nope.", True)])
        with _patch_generate(fake):
            _generate_refusal("sexual", "explicit content here")
        assert len(fake.calls) == 1
        prompt = fake.calls[0]["prompt"]
        assert _CATEGORY_LABELS["sexual"] in prompt

    def test_unknown_category_passes_raw_string(self):
        fake = _make_safety_llm([("blocked", "Nope.", True)])
        with _patch_generate(fake):
            _generate_refusal("new_category", "some text")
        prompt = fake.calls[0]["prompt"]
        assert "new_category" in prompt

    def test_snippet_truncated_to_120_chars(self):
        fake = _make_safety_llm([("blocked", "Nope.", True)])
        with _patch_generate(fake):
            long_text = "a" * 300
            _generate_refusal("hate", long_text)
        prompt = fake.calls[0]["prompt"]
        # The snippet should not contain the full 300 chars
        assert "a" * 121 not in prompt

    @given(
        category=st.sampled_from(list(_CATEGORY_LABELS.keys()) + ["unknown_cat"]),
        text=st.text(min_size=1, max_size=300),
    )
    @settings(max_examples=20, deadline=5000)
    def test_never_raises_on_arbitrary_category_and_text(self, category, text):
        """_generate_refusal must always return a string, never raise."""
        fake = _make_safety_llm([("blocked", "Nope.", True)])
        with _patch_generate(fake):
            result = _generate_refusal(category, text)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# check_input
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
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

    def test_safe_message_passes(self):
        fake = _make_safety_llm([
            ("safety filter", '{"safe": true}', True),
        ])
        with _patch_generate(fake):
            is_safe, reason = check_input("Tell me a joke")
        assert is_safe is True
        assert reason == ""

    def test_unsafe_message_blocked_with_refusal(self):
        fake = _make_safety_llm([
            ("safety filter", '{"safe": false, "category": "hate"}'),
            ("blocked", "Blocked."),
        ])
        with _patch_generate(fake):
            is_safe, reason = check_input("hateful content")
        assert is_safe is False
        assert reason == "Blocked."

    def test_unsafe_without_category_uses_unknown(self):
        fake = _make_safety_llm([
            ("safety filter", '{"safe": false}'),
            ("blocked", "No."),
        ])
        with _patch_generate(fake):
            is_safe, reason = check_input("bad stuff")
        assert is_safe is False
        # Verify the refusal call received "unknown" as category —
        # check that the prompt contains the raw "unknown" string
        refusal_call = fake.calls[1]
        assert "unknown" in refusal_call["prompt"]


# ---------------------------------------------------------------------------
# check_output
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
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

    def test_safe_response_passes(self):
        fake = _make_safety_llm([
            ("safety filter", '{"safe": true}', True),
        ])
        with _patch_generate(fake):
            is_safe, reason = check_output("Here is a joke for you.")
        assert is_safe is True
        assert reason == ""

    def test_unsafe_response_blocked(self):
        fake = _make_safety_llm([
            ("safety filter", '{"safe": false, "category": "sensitive_data"}'),
            ("blocked", "Blocked output."),
        ])
        with _patch_generate(fake):
            is_safe, reason = check_output("Here is your API key: sk-12345")
        assert is_safe is False
        assert reason == "Blocked output."

    def test_no_media_urls_passes(self):
        fake = _make_safety_llm([
            ("safety filter", '{"safe": true}', True),
        ])
        with _patch_generate(fake):
            is_safe, reason = check_output("safe text", media_urls=None)
        assert is_safe is True

    def test_empty_media_urls_passes(self):
        fake = _make_safety_llm([
            ("safety filter", '{"safe": true}', True),
        ])
        with _patch_generate(fake):
            is_safe, reason = check_output("safe text", media_urls=[])
        assert is_safe is True

    def test_unsafe_media_url_blocks_response(self):
        fake = _make_safety_llm([
            # First call: response text is safe
            ("safety filter", '{"safe": true}'),
            # Second call: media URL is unsafe
            ("safety filter", '{"safe": false, "category": "nsfw_url"}'),
            ("blocked", "NSFW blocked."),
        ])
        with _patch_generate(fake):
            is_safe, reason = check_output(
                "see image",
                media_urls=["https://nsfw.example.com/img.jpg"],
            )
        assert is_safe is False
        assert reason == "NSFW blocked."

    def test_media_urls_with_none_entries_filtered(self):
        fake = _make_safety_llm([
            ("safety filter", '{"safe": true}', True),
        ])
        with _patch_generate(fake):
            is_safe, reason = check_output("text", media_urls=[None, "", None])
        assert is_safe is True
        # _classify called once for text; urls_text is empty so no second call
        assert fake.call_count == 1

    def test_safe_media_urls_pass(self):
        fake = _make_safety_llm([
            # Response text: safe
            ("safety filter", '{"safe": true}'),
            # Media URLs: safe
            ("safety filter", '{"safe": true}'),
        ])
        with _patch_generate(fake):
            is_safe, reason = check_output(
                "text",
                media_urls=["https://example.com/photo.jpg"],
            )
        assert is_safe is True
        assert fake.call_count == 2


# ---------------------------------------------------------------------------
# Category labels
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
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

@pytest.mark.timeout(10)
class TestConstants:
    """Verify safety module constants."""

    def test_classify_max_chars_is_positive(self):
        assert _CLASSIFY_MAX_CHARS > 0

    def test_fallback_refusal_is_non_empty(self):
        assert len(_FALLBACK_REFUSAL) > 0

    def test_classify_system_mentions_json(self):
        assert "JSON" in _CLASSIFY_SYSTEM
