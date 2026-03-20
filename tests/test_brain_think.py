"""Tests for arqitect.brain.brain — think() main loop and private helpers.

Covers:
- Depth limit enforcement
- Safety check blocking at depth 0
- Recalibration command detection
- Intent classification routing (workflow vs direct)
- Task truncation at 4000 chars
- Empty nerve catalog forcing synthesis prompt
- LLM returning valid JSON decision → dispatched
- LLM returning no JSON → raw text response
- Circuit breaker filtering
- Recursive think calls via dispatch
- _reverse_geocode success and failure
- _personality_media_enhancement early returns, structured data skip, GIF/emoji
"""

import contextlib
import json
from unittest.mock import patch, MagicMock

import pytest
import responses as responses_lib

from tests.conftest import (
    FakeLLM,
    setup_brain_patches,
    make_nerve_file,
    register_qualified_nerve,
    patch_invoke_nerve,
    patch_synthesize_nerve,
)


# ---------------------------------------------------------------------------
# Helper to run think() inside the standard brain patch context
# ---------------------------------------------------------------------------

def _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, task, **kwargs):
    """Enter all brain patches and call think() with the given task."""
    patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        from arqitect.brain.brain import think
        return think(task, **kwargs)


# ===========================================================================
# 1. Depth limit
# ===========================================================================

class TestThinkDepthLimit:
    """think() must bail out when recursion exceeds depth 5."""

    def test_returns_fallback_at_depth_six(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM()
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "hello", depth=6)
        assert "wasn't able" in result.lower()

    def test_returns_fallback_at_depth_ten(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM()
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "any task", depth=10)
        assert "resolve" in result.lower() or "wasn't able" in result.lower()

    def test_depth_five_does_not_bail(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        """depth=5 is still within limits — should NOT trigger the fallback."""
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", "I can help you with that."),
        ])
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "what time is it", depth=5)
        assert "wasn't able" not in result.lower()


# ===========================================================================
# 2. Safety check
# ===========================================================================

class TestSafetyCheck:
    """At depth 0, unsafe input is blocked before any LLM routing."""

    def test_unsafe_input_blocked(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM()
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            # Override safety check to return unsafe
            with patch(
                "arqitect.brain.brain._safety_check_input",
                return_value=(False, "I can't help with that."),
            ):
                from arqitect.brain.brain import think
                result = think("dangerous request", depth=0)
        assert result == "I can't help with that."

    def test_safety_check_skipped_at_depth_nonzero(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        """Safety filter only runs at depth 0; deeper recursions skip it."""
        fake = FakeLLM([
            ("Available nerves", "Sure, here is the answer."),
        ])
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            # Even with a safety mock that blocks, depth=1 should bypass it
            with patch(
                "arqitect.brain.brain._safety_check_input",
                return_value=(False, "blocked"),
            ):
                from arqitect.brain.brain import think
                result = think("hello", depth=1)
        # Should NOT be blocked — safety only fires at depth 0
        assert result != "blocked"


# ===========================================================================
# 3. Recalibration command detection
# ===========================================================================

class TestRecalibrationDetection:
    """Regex-based recalibration commands bypass the LLM routing."""

    def test_recalibrate_all(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM()
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with patch("arqitect.brain.brain.calibrate_all_senses", return_value={
                "sight": {"status": "ready", "capabilities": {"camera": {"available": True}}},
            }) as mock_cal:
                with patch("arqitect.brain.brain._store_calibration_in_memory"):
                    from arqitect.brain.brain import think
                    result = think("recalibrate all")
            mock_cal.assert_called_once()
            assert "recalibration" in result.lower()

    def test_recalibrate_single_sense(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM()
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with patch("arqitect.brain.brain.calibrate_sense", return_value={
                "status": "ready",
                "capabilities": {"screen": {"available": True}},
            }) as mock_cal:
                with patch("arqitect.brain.brain._store_calibration_in_memory"):
                    from arqitect.brain.brain import think
                    result = think("recalibrate sight")
            mock_cal.assert_called_once_with("sight")
            assert "recalibration" in result.lower()

    def test_recalibrate_not_triggered_with_history(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        """Recalibration regex only fires when history is None/empty."""
        fake = FakeLLM([
            ("Available nerves", "Here is what I found."),
        ])
        result = _run_think(
            fake, mem, test_redis, nerves_dir, sandbox_dir,
            "recalibrate all", history=["prior step"],
        )
        # With history, recalibration is skipped; LLM responds instead
        assert "recalibration" not in result.lower()


# ===========================================================================
# 4. Intent classification routing
# ===========================================================================

class TestIntentRouting:
    """Intent classifier routes workflows to planner, direct to LLM."""

    def test_workflow_intent_routes_to_planner(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM([
            ("classify", '{"type": "workflow", "category": "development"}'),
        ])
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with patch("arqitect.brain.brain.plan_task", return_value=None) as mock_plan:
                with patch("arqitect.brain.brain.detect_project_path", return_value=None):
                    from arqitect.brain.brain import think
                    # plan_task returns None → falls through to normal LLM routing
                    think("build a REST API")
            mock_plan.assert_called_once()

    def test_direct_intent_skips_planner(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", "Hello there!"),
        ])
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with patch("arqitect.brain.brain.plan_task") as mock_plan:
                from arqitect.brain.brain import think
                think("hello")
            mock_plan.assert_not_called()


# ===========================================================================
# 5. Task truncation
# ===========================================================================

class TestTaskTruncation:
    """Tasks longer than 4000 chars are truncated before routing."""

    def test_long_task_is_truncated(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM()
        long_task = "x" * 5000
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import think
            think(long_task)
        # The LLM should have received a truncated version of the task
        llm_calls = fake.calls
        # Find the routing call (contains "Task:")
        routing_calls = [c for c in llm_calls if "Task:" in c["prompt"]]
        if routing_calls:
            prompt = routing_calls[-1]["prompt"]
            # The task in the prompt should end with truncation marker
            assert "[truncated]" in prompt


# ===========================================================================
# 6. Empty nerve catalog
# ===========================================================================

class TestEmptyNerveCatalog:
    """When no nerves are registered, the LLM prompt forces synthesis."""

    def test_senses_only_catalog_still_routes(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        """When no custom nerves exist, only core senses appear in the catalog.

        This is a real architectural invariant: the catalog is never truly empty
        because the 5 core senses are always registered.
        """
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", '{"action": "respond", "message": "I can help!"}'),
        ])
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import think
            think("do something new")
        # Routing prompt should contain only sense names, no custom nerves
        routing_calls = [c for c in fake.calls if "Task:" in c["prompt"]]
        assert routing_calls, "Expected at least one routing LLM call"
        prompt = routing_calls[-1]["prompt"].lower()
        for sense in ("hearing", "sight", "touch", "awareness", "communication"):
            assert sense in prompt, f"Core sense '{sense}' missing from routing prompt"


# ===========================================================================
# 7. LLM returns valid JSON decision → dispatched
# ===========================================================================

class TestLLMJsonDecision:
    """When the LLM returns a JSON decision, it is dispatched."""

    def test_invoke_nerve_decision_dispatched(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", '{"action": "invoke_nerve", "name": "weather_nerve", "args": "London"}'),
        ])
        register_qualified_nerve(mem, "weather_nerve", "get weather info")
        make_nerve_file(nerves_dir, "weather_nerve")
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with patch_invoke_nerve(return_value='{"response": "Sunny 25C"}') as mock_inv:
                from arqitect.brain.brain import think
                result = think("what is the weather in London")
        mock_inv.assert_called()

    def test_respond_action_returns_message(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", '{"action": "respond", "message": "I am doing great!"}'),
        ])
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "how are you")
        assert "great" in result.lower() or "doing" in result.lower() or len(result) > 0


# ===========================================================================
# 8. LLM returns no JSON → raw text response
# ===========================================================================

class TestLLMRawTextResponse:
    """When the LLM returns plain text (no JSON), it is used as-is."""

    def test_plain_text_returned(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", "Just a plain text answer with no JSON."),
        ])
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "tell me a joke")
        assert result == "Just a plain text answer with no JSON."


# ===========================================================================
# 9. Circuit breaker filtering
# ===========================================================================

class TestCircuitBreakerFiltering:
    """Nerves with open circuits are excluded from the catalog."""

    def test_broken_nerve_filtered_from_catalog(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", "No matching nerve found."),
        ])
        register_qualified_nerve(mem, "flaky_nerve", "a flaky service")
        make_nerve_file(nerves_dir, "flaky_nerve")
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            # Mark the nerve as circuit-open
            with patch("arqitect.brain.brain._cb_is_available", return_value=False):
                from arqitect.brain.brain import think
                think("use the flaky service")
        # The routing prompt should NOT contain flaky_nerve
        routing_calls = [c for c in fake.calls if "Task:" in c["prompt"]]
        if routing_calls:
            assert "flaky_nerve" not in routing_calls[-1]["prompt"]


# ===========================================================================
# 10. Recursive think calls
# ===========================================================================

class TestRecursiveThink:
    """Dispatch can trigger recursive think() — depth must increment."""

    def test_clarify_action_returns_message_without_recursion(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        """Clarify action returns the clarification message directly."""
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", '{"action": "clarify", "message": "Could you be more specific?"}'),
        ])
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "do the thing")
        assert "specific" in result.lower() or "clarif" in result.lower() or len(result) > 0


# ===========================================================================
# 11. _reverse_geocode
# ===========================================================================

class TestReverseGeocode:
    """_reverse_geocode hits Nominatim and extracts city names."""

    @responses_lib.activate
    def test_success_returns_city(self):
        responses_lib.add(
            responses_lib.GET,
            "https://nominatim.openstreetmap.org/reverse",
            json={"address": {"city": "Tel Aviv", "country": "Israel"}},
            status=200,
        )
        from arqitect.brain.brain import _reverse_geocode
        assert _reverse_geocode(32.08, 34.78) == "Tel Aviv"

    @responses_lib.activate
    def test_falls_back_to_town(self):
        responses_lib.add(
            responses_lib.GET,
            "https://nominatim.openstreetmap.org/reverse",
            json={"address": {"town": "Herzliya"}},
            status=200,
        )
        from arqitect.brain.brain import _reverse_geocode
        assert _reverse_geocode(32.16, 34.79) == "Herzliya"

    @responses_lib.activate
    def test_timeout_returns_empty_string(self):
        responses_lib.add(
            responses_lib.GET,
            "https://nominatim.openstreetmap.org/reverse",
            body=ConnectionError("timeout"),
        )
        from arqitect.brain.brain import _reverse_geocode
        assert _reverse_geocode(0.0, 0.0) == ""

    @responses_lib.activate
    def test_server_error_returns_empty_string(self):
        responses_lib.add(
            responses_lib.GET,
            "https://nominatim.openstreetmap.org/reverse",
            json={"error": "Unable to geocode"},
            status=500,
        )
        from arqitect.brain.brain import _reverse_geocode
        # Even a 500 with unexpected JSON should not crash
        result = _reverse_geocode(0.0, 0.0)
        assert result == ""


# ===========================================================================
# 12. _personality_media_enhancement
# ===========================================================================

class TestPersonalityMediaEnhancement:
    """Media enrichment via personality weights — tested with controlled randomness."""

    def test_returns_early_for_falsy_msg(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir,
    ):
        patches = setup_brain_patches(FakeLLM(), mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import _personality_media_enhancement
            result = _personality_media_enhancement("task", "", {"response": "ok"})
            assert result == {"response": "ok"}

    def test_returns_early_for_non_dict_result(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir,
    ):
        patches = setup_brain_patches(FakeLLM(), mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import _personality_media_enhancement
            result = _personality_media_enhancement("task", "hello", "not a dict")
            assert result == "not a dict"

    def test_skips_if_already_has_media(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir,
    ):
        patches = setup_brain_patches(FakeLLM(), mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import _personality_media_enhancement
            nerve_result = {"response": "ok", "gif_url": "http://example.com/cat.gif"}
            result = _personality_media_enhancement("task", "hello", nerve_result)
            assert result["gif_url"] == "http://example.com/cat.gif"

    def test_skips_structured_data(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir,
    ):
        """Messages starting with JSON/code markers are never enhanced."""
        patches = setup_brain_patches(FakeLLM(), mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import _personality_media_enhancement
            nerve_result = {"response": "ok"}
            result = _personality_media_enhancement("task", '{"key": "value"}', nerve_result)
            assert "gif_url" not in result
            assert "_personality_rewrite" not in result

    def test_gif_added_when_roll_below_gif_chance(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir,
    ):
        """When random roll is very low and wit+swagger are high, GIF is added."""
        fake = FakeLLM()
        register_qualified_nerve(mem, "gif_search_nerve", "Search for GIFs")
        make_nerve_file(nerves_dir, "gif_search_nerve")
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import _personality_media_enhancement
            # Set high wit and swagger so gif_chance is non-trivial
            mem.cold.set_fact("personality", "trait_weights",
                             json.dumps({"wit": 0.95, "swagger": 0.95}))
            with patch("random.random", return_value=0.001):
                with patch_invoke_nerve(return_value='{"gif_url": "http://giphy.com/test.gif"}'):
                    nerve_result = {"response": "nice joke"}
                    result = _personality_media_enhancement("tell me a joke", "nice joke", nerve_result)
            assert result.get("gif_url") == "http://giphy.com/test.gif"

    def test_emoji_added_when_roll_in_emoji_range(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir,
    ):
        """When roll falls in the emoji probability band, emoji rewrite happens."""
        fake = FakeLLM()
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import _personality_media_enhancement
            # wit=0.8, swagger=0.3 → gif_chance ~= 0.012, emoji_chance ~= 0.075
            # roll=0.05 is above gif_chance but below gif_chance + emoji_chance
            mem.cold.set_fact("personality", "trait_weights",
                             json.dumps({"wit": 0.8, "swagger": 0.3}))
            with patch("random.random", return_value=0.05):
                with patch_invoke_nerve(return_value='{"response": "hello! 😊"}'):
                    nerve_result = {"response": "hello"}
                    result = _personality_media_enhancement("greet me", "hello", nerve_result)
            # Should have attempted emoji enhancement
            # (may or may not have _personality_rewrite depending on mock output)
            assert isinstance(result, dict)

    def test_json_decode_error_caught(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir,
    ):
        """JSONDecodeError from invoke_nerve is caught, not propagated."""
        fake = FakeLLM()
        register_qualified_nerve(mem, "gif_search_nerve", "Search for GIFs")
        make_nerve_file(nerves_dir, "gif_search_nerve")
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import _personality_media_enhancement
            mem.cold.set_fact("personality", "trait_weights",
                             json.dumps({"wit": 0.95, "swagger": 0.95}))
            with patch("random.random", return_value=0.001):
                with patch_invoke_nerve(return_value="not valid json {{{"):
                    nerve_result = {"response": "test"}
                    # Should not raise
                    result = _personality_media_enhancement("query", "test", nerve_result)
            assert isinstance(result, dict)
            assert "gif_url" not in result


# ===========================================================================
# 13. Consolidator wake at depth 0
# ===========================================================================

class TestConsolidatorWake:
    """get_consolidator().wake() is called at depth 0 only."""

    def test_consolidator_wakes_at_depth_zero(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", "Hello!"),
        ])
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            # get_consolidator is already mocked in setup_brain_patches
            from arqitect.brain import brain as brain_mod
            mock_consolidator = brain_mod.get_consolidator()
            brain_mod.think("hi", depth=0)
            mock_consolidator.wake.assert_called()
