"""Tests for the unified communication nerve response pipeline.

Covers rewrite_response(), _decide_format(), task-context in _try_llm_rewrite(),
and personality signal recording in publish_response().
"""

import json
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# _decide_format
# ---------------------------------------------------------------------------

class TestDecideFormat:
    """Format decision based on personality traits and message content."""

    def _make_traits(self, wit: float = 0.5, swagger: float = 0.3) -> dict:
        return {"trait_weights": {"wit": wit, "swagger": swagger}}

    @pytest.mark.timeout(5)
    def test_explicit_format_hint_takes_precedence(self):
        """When format_hint is a known format, _decide_format returns it directly."""
        from arqitect.senses.communication.nerve import _decide_format
        result = _decide_format(self._make_traits(), "hello", format_hint="gif")
        assert result == "gif"

    @pytest.mark.timeout(5)
    def test_allow_media_false_forces_text(self):
        """When allow_media=False, always returns 'text' regardless of traits."""
        from arqitect.senses.communication.nerve import _decide_format
        traits = self._make_traits(wit=0.95, swagger=0.95)
        with patch("random.random", return_value=0.001):
            result = _decide_format(traits, "hello", allow_media=False)
        assert result == "text"

    @pytest.mark.timeout(5)
    def test_structured_data_always_text(self):
        """Structured data (JSON, code) always gets plain text format."""
        from arqitect.senses.communication.nerve import _decide_format
        traits = self._make_traits(wit=0.95, swagger=0.95)
        with patch("random.random", return_value=0.001):
            assert _decide_format(traits, '{"key": "value"}') == "text"
            assert _decide_format(traits, '```python\nprint("hi")\n```') == "text"

    @pytest.mark.timeout(5)
    def test_gif_selected_when_roll_below_gif_chance(self):
        """High wit+swagger with low roll selects gif format."""
        from arqitect.senses.communication.nerve import _decide_format
        traits = self._make_traits(wit=0.95, swagger=0.95)
        # gif_chance = (0.95 + 0.95 - 1.0) * 0.12 = 0.108
        with patch("random.random", return_value=0.05):
            result = _decide_format(traits, "tell me a joke")
        assert result == "gif"

    @pytest.mark.timeout(5)
    def test_emoji_selected_when_roll_in_emoji_range(self):
        """Roll above gif_chance but below gif+emoji selects emoji."""
        from arqitect.senses.communication.nerve import _decide_format
        traits = self._make_traits(wit=0.8, swagger=0.3)
        # gif_chance = (0.8 + 0.3 - 1.0) * 0.12 = 0.012
        # emoji_chance = (0.8 - 0.3) * 0.15 = 0.075
        # total = 0.087
        with patch("random.random", return_value=0.05):
            result = _decide_format(traits, "hello there")
        assert result == "emoji"

    @pytest.mark.timeout(5)
    def test_text_selected_for_default_traits(self):
        """Default trait weights produce text for most rolls."""
        from arqitect.senses.communication.nerve import _decide_format
        traits = self._make_traits(wit=0.5, swagger=0.3)
        with patch("random.random", return_value=0.5):
            result = _decide_format(traits, "hello")
        assert result == "text"


# ---------------------------------------------------------------------------
# rewrite_response
# ---------------------------------------------------------------------------

class TestRewriteResponse:
    """Single entry point for all response formatting."""

    @pytest.mark.timeout(5)
    def test_empty_message_passthrough(self):
        """Empty message returns immediately without LLM call."""
        from arqitect.senses.communication.nerve import rewrite_response
        result = rewrite_response("")
        assert result["response"] == ""
        assert result["format"] == "text"

    @pytest.mark.timeout(5)
    def test_structured_data_passes_through_untouched(self):
        """JSON/code messages are not personality-altered."""
        from arqitect.senses.communication.nerve import rewrite_response
        msg = '{"status": "ok", "count": 42}'
        with patch("arqitect.senses.communication.nerve._try_llm_rewrite", return_value=msg) as mock_rw:
            result = rewrite_response(msg, task="get status")
        assert result["format"] == "text"
        # LLM rewrite is still called (it handles structured data internally)
        mock_rw.assert_called_once()

    @pytest.mark.timeout(5)
    def test_task_context_included_in_rewrite(self):
        """Task is forwarded to the LLM rewrite for context."""
        from arqitect.senses.communication.nerve import rewrite_response
        with patch("arqitect.senses.communication.nerve._try_llm_rewrite",
                   return_value="Rewritten") as mock_rw:
            result = rewrite_response("raw output", task="what's the weather?")
        mock_rw.assert_called_once()
        assert mock_rw.call_args.kwargs.get("task") == "what's the weather?"

    @pytest.mark.timeout(5)
    def test_gif_format_returns_media_fields(self):
        """When format is gif, result includes gif_url and gif_query."""
        from arqitect.senses.communication.nerve import rewrite_response
        with patch("arqitect.senses.communication.nerve._decide_format", return_value="gif"), \
             patch("arqitect.senses.communication.nerve._load_personality_traits",
                   return_value={"trait_weights": {"wit": 0.9, "swagger": 0.9}}), \
             patch("arqitect.senses.communication.nerve.format_gif",
                   return_value={"format": "gif", "response": "Here!", "gif_url": "http://example.com/gif.gif", "gif_query": "funny"}), \
             patch("arqitect.senses.communication.nerve._try_llm_rewrite", return_value="Here!"):
            result = rewrite_response("nerve output", task="show me something funny")
        assert result["format"] == "gif"
        assert result["gif_url"] == "http://example.com/gif.gif"

    @pytest.mark.timeout(5)
    def test_emoji_format_returns_enhanced_text(self):
        """When format is emoji, result uses format_emoji output."""
        from arqitect.senses.communication.nerve import rewrite_response
        with patch("arqitect.senses.communication.nerve._decide_format", return_value="emoji"), \
             patch("arqitect.senses.communication.nerve._load_personality_traits",
                   return_value={"trait_weights": {"wit": 0.8, "swagger": 0.3}}), \
             patch("arqitect.senses.communication.nerve.format_emoji",
                   return_value={"format": "emoji", "response": "Hello! :wave:"}):
            result = rewrite_response("Hello", task="greet me")
        assert result["format"] == "emoji"


# ---------------------------------------------------------------------------
# _try_llm_rewrite task context
# ---------------------------------------------------------------------------

class TestTryLlmRewriteTaskContext:
    """Verify task is included in the LLM prompt when provided."""

    @pytest.mark.timeout(5)
    def test_task_included_in_user_prompt(self):
        """When task is provided, the user prompt includes it."""
        from arqitect.senses.communication.nerve import _try_llm_rewrite
        with patch("arqitect.inference.router.generate_for_role",
                   return_value="Rewritten message") as mock_gen:
            result = _try_llm_rewrite("raw output", "neutral", task="what time is it?")
        assert result == "Rewritten message"
        user_prompt = mock_gen.call_args[0][1]
        assert "what time is it?" in user_prompt

    @pytest.mark.timeout(5)
    def test_no_task_omits_prefix(self):
        """When task is empty, the user prompt has no task prefix."""
        from arqitect.senses.communication.nerve import _try_llm_rewrite
        with patch("arqitect.inference.router.generate_for_role",
                   return_value="Rewritten") as mock_gen:
            _try_llm_rewrite("raw output", "neutral", task="")
        user_prompt = mock_gen.call_args[0][1]
        assert "The user asked:" not in user_prompt


# ---------------------------------------------------------------------------
# Signal recording in publish_response
# ---------------------------------------------------------------------------

class TestSignalRecording:
    """Verify publish_response calls record_signal for personality evolution."""

    @pytest.mark.timeout(5)
    def test_signal_recorded_on_publish(self):
        """publish_response must call record_signal with correct fields."""
        with patch("arqitect.brain.events.mem") as mock_mem, \
             patch("arqitect.brain.safety.generate_for_role", return_value='{"safe": true}'), \
             patch("arqitect.brain.events.build_envelope", return_value={"message": "hi"}), \
             patch("arqitect.brain.events.publish_event"), \
             patch("arqitect.brain.events._is_voice_origin", return_value=False), \
             patch("arqitect.brain.events._validate_response", return_value=None), \
             patch("arqitect.brain.events.merge_nerve_result_into_envelope"), \
             patch("arqitect.brain.events.get_task_origin", return_value={"source": "", "chat_id": "", "user_id": ""}), \
             patch("arqitect.brain.personality.record_signal") as mock_signal:
            from arqitect.brain.events import publish_response
            publish_response("hello world", task="greet me", tone="neutral")

        mock_signal.assert_called_once()
        signal = mock_signal.call_args[0][1]
        assert signal["task_snippet"] == "greet me"
        assert signal["tone_used"] == "neutral"
        assert signal["response_length"] == len("hello world")
        assert "timestamp" in signal

    @pytest.mark.timeout(5)
    def test_signal_failure_does_not_block_response(self):
        """If record_signal raises, publish_response still completes."""
        with patch("arqitect.brain.events.mem") as mock_mem, \
             patch("arqitect.brain.safety.generate_for_role", return_value='{"safe": true}'), \
             patch("arqitect.brain.events.build_envelope", return_value={"message": "hi"}), \
             patch("arqitect.brain.events.publish_event") as mock_publish, \
             patch("arqitect.brain.events._is_voice_origin", return_value=False), \
             patch("arqitect.brain.events._validate_response", return_value=None), \
             patch("arqitect.brain.events.merge_nerve_result_into_envelope"), \
             patch("arqitect.brain.events.get_task_origin", return_value={"source": "", "chat_id": "", "user_id": ""}), \
             patch("arqitect.brain.personality.record_signal", side_effect=RuntimeError("db error")):
            from arqitect.brain.events import publish_response
            # Should not raise
            publish_response("hello world", task="test")

        # Verify the response was still published despite signal failure
        mock_publish.assert_called()
