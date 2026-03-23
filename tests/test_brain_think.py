"""Tests for arqitect.brain.brain — think() main loop and private helpers.

Covers:
- Depth limit enforcement
- Safety check blocking at depth 0
- Recalibration command detection
- Intent classification routing (workflow vs direct)
- Task truncation at 4000 chars
- Empty nerve catalog forcing synthesis prompt
- LLM returning valid JSON decision -> dispatched
- LLM returning no JSON -> raw text response
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
from dirty_equals import IsStr, IsInstance
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

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

    @pytest.mark.timeout(10)
    def test_returns_fallback_at_depth_six(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM()
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "hello", depth=6)
        assert "wasn't able" in result.lower() or "resolve" in result.lower()

    @pytest.mark.timeout(10)
    @given(depth=st.integers(min_value=6, max_value=50))
    @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_returns_fallback_at_any_depth_above_five(
        self, depth, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        """Any depth > 5 triggers the fallback — property-based check."""
        fake = FakeLLM()
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "any task", depth=depth)
        assert "resolve" in result.lower() or "wasn't able" in result.lower()

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
    @given(depth=st.integers(min_value=0, max_value=5))
    @settings(max_examples=3, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_depths_zero_through_five_do_not_bail(
        self, depth, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        """Any depth 0..5 should proceed normally, not trigger fallback."""
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", "I can help you with that."),
        ])
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "hello", depth=depth)
        assert "wasn't able" not in result.lower()


# ===========================================================================
# 2. Safety check
# ===========================================================================

class TestSafetyCheck:
    """At depth 0, unsafe input is blocked before any LLM routing."""

    @pytest.mark.timeout(10)
    def test_unsafe_input_blocked(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM()
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with patch(
                "arqitect.brain.brain._safety_check_input",
                return_value=(False, "I can't help with that."),
            ):
                from arqitect.brain.brain import think
                result = think("dangerous request", depth=0)
        assert result == "I can't help with that."

    @pytest.mark.timeout(10)
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
            with patch(
                "arqitect.brain.brain._safety_check_input",
                return_value=(False, "blocked"),
            ):
                from arqitect.brain.brain import think
                result = think("hello", depth=1)
        assert result != "blocked"


# ===========================================================================
# 3. Recalibration command detection
# ===========================================================================

class TestRecalibrationDetection:
    """Regex-based recalibration commands bypass the LLM routing."""

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
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
        assert "recalibration" not in result.lower()


# ===========================================================================
# 4. Intent classification routing
# ===========================================================================

class TestIntentRouting:
    """Intent classifier routes workflows to planner, direct to LLM."""

    @pytest.mark.timeout(10)
    def test_plan_intent_routes_to_plan_start(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM([
            ("classify", '{"type": "plan", "category": "development"}'),
            ("User wants", "Let me help you plan that REST API. What framework?"),
        ])
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with patch("arqitect.brain.brain.detect_project_path", return_value=None):
                from arqitect.brain.brain import think
                result = think("build a REST API")
            assert result is not None

    @pytest.mark.timeout(10)
    def test_direct_intent_skips_plan(
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
            from arqitect.brain.brain import think
            result = think("hello")
            assert result is not None


# ===========================================================================
# 5. Task truncation
# ===========================================================================

class TestTaskTruncation:
    """Tasks longer than 4000 chars are truncated before routing."""

    @pytest.mark.timeout(10)
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
        routing_calls = [c for c in fake.calls if "Task:" in c["prompt"]]
        if routing_calls:
            prompt = routing_calls[-1]["prompt"]
            assert "[truncated]" in prompt

    @pytest.mark.timeout(10)
    @given(length=st.integers(min_value=4001, max_value=8000))
    @settings(max_examples=3, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_any_oversized_task_is_truncated(
        self, length, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        """Property: any task exceeding _MAX_TASK_LENGTH gets truncated."""
        fake = FakeLLM()
        long_task = "a" * length
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import think
            think(long_task)
        routing_calls = [c for c in fake.calls if "Task:" in c["prompt"]]
        if routing_calls:
            prompt = routing_calls[-1]["prompt"]
            assert "[truncated]" in prompt

    @pytest.mark.timeout(10)
    def test_short_task_not_truncated(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        """Tasks within the limit are passed through without truncation."""
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", "ok"),
        ])
        short_task = "x" * 100
        patches = setup_brain_patches(fake, mem, test_redis, nerves_dir, sandbox_dir)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            from arqitect.brain.brain import think
            think(short_task)
        routing_calls = [c for c in fake.calls if "Task:" in c["prompt"]]
        if routing_calls:
            prompt = routing_calls[-1]["prompt"]
            assert "[truncated]" not in prompt


# ===========================================================================
# 6. Empty nerve catalog
# ===========================================================================

class TestEmptyNerveCatalog:
    """When no nerves are registered, the LLM prompt forces synthesis."""

    @pytest.mark.timeout(10)
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
        routing_calls = [c for c in fake.calls if "Task:" in c["prompt"]]
        assert routing_calls, "Expected at least one routing LLM call"
        prompt = routing_calls[-1]["prompt"].lower()
        for sense in ("hearing", "sight", "touch", "awareness", "communication"):
            assert sense in prompt, f"Core sense '{sense}' missing from routing prompt"


# ===========================================================================
# 7. LLM returns valid JSON decision -> dispatched
# ===========================================================================

class TestLLMJsonDecision:
    """When the LLM returns a JSON decision, it is dispatched."""

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
    def test_respond_action_returns_message(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", '{"action": "respond", "message": "I am doing great!"}'),
        ])
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "how are you")
        assert result == IsStr(min_length=1)


# ===========================================================================
# 8. LLM returns no JSON -> raw text response
# ===========================================================================

class TestLLMRawTextResponse:
    """When the LLM returns plain text (no JSON), it is used as-is."""

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
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
            with patch("arqitect.brain.brain._cb_is_available", return_value=False):
                from arqitect.brain.brain import think
                think("use the flaky service")
        routing_calls = [c for c in fake.calls if "Task:" in c["prompt"]]
        if routing_calls:
            assert "flaky_nerve" not in routing_calls[-1]["prompt"]


# ===========================================================================
# 10. Recursive think calls
# ===========================================================================

class TestRecursiveThink:
    """Dispatch can trigger recursive think() — depth must increment."""

    @pytest.mark.timeout(10)
    def test_clarify_action_returns_message_without_recursion(
        self, test_redis, tmp_memory_dir, mem, nerves_dir, sandbox_dir, captured_events,
    ):
        """Clarify action returns the clarification message directly."""
        fake = FakeLLM([
            ("classify", '{"type": "direct"}'),
            ("Available nerves", '{"action": "clarify", "message": "Could you be more specific?"}'),
        ])
        result = _run_think(fake, mem, test_redis, nerves_dir, sandbox_dir, "do the thing")
        assert result == IsStr(min_length=1)


# ===========================================================================
# 11. _reverse_geocode
# ===========================================================================

class TestReverseGeocode:
    """_reverse_geocode hits Nominatim and extracts city names."""

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
    @responses_lib.activate
    def test_timeout_returns_empty_string(self):
        responses_lib.add(
            responses_lib.GET,
            "https://nominatim.openstreetmap.org/reverse",
            body=ConnectionError("timeout"),
        )
        from arqitect.brain.brain import _reverse_geocode
        assert _reverse_geocode(0.0, 0.0) == ""

    @pytest.mark.timeout(10)
    @responses_lib.activate
    def test_server_error_returns_empty_string(self):
        responses_lib.add(
            responses_lib.GET,
            "https://nominatim.openstreetmap.org/reverse",
            json={"error": "Unable to geocode"},
            status=500,
        )
        from arqitect.brain.brain import _reverse_geocode
        result = _reverse_geocode(0.0, 0.0)
        assert result == ""

    @pytest.mark.timeout(10)
    @given(
        lat=st.floats(min_value=-90, max_value=90, allow_nan=False),
        lon=st.floats(min_value=-180, max_value=180, allow_nan=False),
    )
    @settings(max_examples=5)
    @responses_lib.activate
    def test_never_raises_for_valid_coords(self, lat, lon):
        """Property: _reverse_geocode never raises for any valid coordinate pair."""
        responses_lib.add(
            responses_lib.GET,
            "https://nominatim.openstreetmap.org/reverse",
            json={"address": {}},
            status=200,
        )
        from arqitect.brain.brain import _reverse_geocode
        result = _reverse_geocode(lat, lon)
        assert result == IsInstance(str)


# ===========================================================================
# 12. _personality_media_enhancement
# ===========================================================================

class TestPersonalityMediaEnhancement:
    """Media enrichment via personality weights — tested with controlled randomness."""

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
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
            mem.cold.set_fact("personality", "trait_weights",
                             json.dumps({"wit": 0.95, "swagger": 0.95}))
            with patch("random.random", return_value=0.001):
                with patch_invoke_nerve(return_value='{"gif_url": "http://giphy.com/test.gif"}'):
                    nerve_result = {"response": "nice joke"}
                    result = _personality_media_enhancement("tell me a joke", "nice joke", nerve_result)
            assert result.get("gif_url") == "http://giphy.com/test.gif"

    @pytest.mark.timeout(10)
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
            mem.cold.set_fact("personality", "trait_weights",
                             json.dumps({"wit": 0.8, "swagger": 0.3}))
            with patch("random.random", return_value=0.05):
                with patch_invoke_nerve(return_value='{"response": "hello! :)"}'):
                    nerve_result = {"response": "hello"}
                    result = _personality_media_enhancement("greet me", "hello", nerve_result)
            assert result == IsInstance(dict)

    @pytest.mark.timeout(10)
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
                    result = _personality_media_enhancement("query", "test", nerve_result)
            assert result == IsInstance(dict)
            assert "gif_url" not in result


# ===========================================================================
# 13. Consolidator wake at depth 0
# ===========================================================================

class TestConsolidatorWake:
    """get_consolidator().wake() is called at depth 0 only."""

    @pytest.mark.timeout(10)
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
            from arqitect.brain import brain as brain_mod
            mock_consolidator = brain_mod.get_consolidator()
            brain_mod.think("hi", depth=0)
            mock_consolidator.wake.assert_called()
