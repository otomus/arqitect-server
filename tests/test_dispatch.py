"""Tests for brain action dispatch — normalization, redirect, and routing.

TDD: these tests define the contract for arqitect.brain.dispatch.
Each test targets a specific flow or edge case in the routing pipeline.
All test data is produced by typed factories — no bare dict literals.
"""

import json
from dataclasses import asdict
from unittest.mock import patch, MagicMock

import pytest

from arqitect.brain.types import Action, Sense
from arqitect.brain.dispatch import (
    InvokeDecision,
    SynthesizeDecision,
    ChainDecision,
    ChainStep,
    ClarifyDecision,
    FeedbackDecision,
    UpdateContextDecision,
    RespondDecision,
    UseSenseDecision,
    parse_decision,
)
from tests.factories import (
    InvokeDecisionFactory,
    SynthesizeDecisionFactory,
    ChainDecisionFactory,
    ChainStepFactory,
    ClarifyDecisionFactory,
    FeedbackDecisionFactory,
    UpdateContextDecisionFactory,
    RespondDecisionFactory,
    UseSenseDecisionFactory,
    DispatchContextFactory,
    as_dict,
)


# ---------------------------------------------------------------------------
# 0. parse_decision — typed parsing of raw LLM dicts
# ---------------------------------------------------------------------------

class TestParseDecision:
    """parse_decision must convert raw dicts into typed Decision dataclasses."""

    def test_invoke_decision(self):
        d = InvokeDecisionFactory.build()
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert isinstance(parsed, InvokeDecision)
        assert parsed.name == d.name
        assert parsed.args == d.args

    def test_synthesize_decision(self):
        d = SynthesizeDecisionFactory.build()
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert isinstance(parsed, SynthesizeDecision)
        assert parsed.name == d.name
        assert parsed.description == d.description

    def test_chain_decision(self):
        steps = [ChainStepFactory.build(nerve="joke_nerve"), ChainStepFactory.build(nerve="translate_nerve")]
        d = ChainDecisionFactory.build(steps=steps)
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert isinstance(parsed, ChainDecision)
        assert len(parsed.steps) == 2
        assert parsed.steps[0].nerve == "joke_nerve"

    def test_clarify_decision(self):
        d = ClarifyDecisionFactory.build()
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert isinstance(parsed, ClarifyDecision)
        assert parsed.message == d.message
        assert parsed.suggestions == d.suggestions

    def test_feedback_decision(self):
        d = FeedbackDecisionFactory.build(sentiment="negative", message="That was wrong")
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert isinstance(parsed, FeedbackDecision)
        assert parsed.sentiment == "negative"
        assert parsed.message == "That was wrong"

    def test_update_context_decision(self):
        d = UpdateContextDecisionFactory.build(context={"city": "Tel Aviv"})
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert isinstance(parsed, UpdateContextDecision)
        assert parsed.context["city"] == "Tel Aviv"

    def test_respond_decision(self):
        d = RespondDecisionFactory.build()
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert isinstance(parsed, RespondDecision)
        assert parsed.message == d.message

    def test_use_sense_decision(self):
        d = UseSenseDecisionFactory.build(sense="sight", args="describe image")
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert isinstance(parsed, UseSenseDecision)
        assert parsed.sense == "sight"

    def test_missing_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            parse_decision({})

    def test_unknown_action_returns_invoke_fallback(self):
        parsed = parse_decision({"action": "fly_to_moon", "name": "x"})
        assert isinstance(parsed, InvokeDecision)
        assert parsed.action == "fly_to_moon"

    def test_dict_args_coerced_to_string(self):
        raw = {"action": "invoke_nerve", "name": "test", "args": {"key": "val"}}
        parsed = parse_decision(raw)
        assert isinstance(parsed.args, str)
        assert "key" in parsed.args


# ---------------------------------------------------------------------------
# 1. normalize_action — fix LLM output before dispatch
# ---------------------------------------------------------------------------

class TestNormalizeAction:
    """LLM output is messy. normalize_action must clean it up."""

    def test_valid_action_passes_through(self):
        from arqitect.brain.dispatch import normalize_action
        d = InvokeDecisionFactory.build()
        action, result = normalize_action(as_dict(d), nerve_catalog={"joke_nerve": "jokes"})
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "joke_nerve"

    def test_typo_gets_fuzzy_matched(self):
        """LLM typo like 'invok_nerve' should be corrected."""
        from arqitect.brain.dispatch import normalize_action
        d = InvokeDecisionFactory.build(action="invok_nerve", name="weather_nerve")
        action, _ = normalize_action(as_dict(d), nerve_catalog={})
        assert action == Action.INVOKE_NERVE

    def test_use_sense_maps_to_invoke(self):
        """use_sense is not a real action — should become invoke_nerve."""
        from arqitect.brain.dispatch import normalize_action
        d = UseSenseDecisionFactory.build(sense="touch", args="list files")
        action, result = normalize_action(as_dict(d), nerve_catalog={})
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "touch"

    def test_nerve_name_as_action(self):
        """LLM returns action='touch' instead of invoke_nerve('touch')."""
        from arqitect.brain.dispatch import normalize_action
        catalog = {"touch": "file ops", "awareness": "identity"}
        d = InvokeDecisionFactory.build(action="touch", args="list files")
        action, result = normalize_action(as_dict(d), nerve_catalog=catalog)
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "touch"

    def test_sense_name_as_action(self):
        """LLM returns action='awareness' — a core sense name."""
        from arqitect.brain.dispatch import normalize_action
        catalog = {"awareness": "identity"}
        d = InvokeDecisionFactory.build(action="awareness", args="who are you?")
        action, result = normalize_action(as_dict(d), nerve_catalog=catalog)
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "awareness"

    def test_unknown_action_stays_unknown(self):
        """Truly unknown action that doesn't fuzzy-match stays as-is."""
        from arqitect.brain.dispatch import normalize_action
        d = InvokeDecisionFactory.build(action="do_magic", args="abracadabra")
        action, _ = normalize_action(as_dict(d), nerve_catalog={})
        assert action == "do_magic"

    def test_use_sense_with_dict_args_coerced_to_string(self):
        """use_sense args may be a dict — must be coerced to JSON string."""
        from arqitect.brain.dispatch import normalize_action
        raw = {"action": "use_sense", "sense": "touch", "args": {"mode": "list", "path": "/home"}}
        action, result = normalize_action(raw, nerve_catalog={})
        assert action == Action.INVOKE_NERVE
        assert isinstance(result["args"], str)

    def test_use_sense_without_sense_field_uses_name(self):
        """use_sense with 'name' instead of 'sense' field."""
        from arqitect.brain.dispatch import normalize_action
        d = UseSenseDecisionFactory.build(sense="", name="hearing", args="play music")
        action, result = normalize_action(as_dict(d), nerve_catalog={})
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "hearing"


# ---------------------------------------------------------------------------
# 2. resolve_synthesize_redirect — stop LLM from re-synthesizing existing nerves
# ---------------------------------------------------------------------------

class TestResolveSynthesizeRedirect:
    """When the LLM asks to synthesize a nerve that already exists,
    redirect to invoke instead of looping."""

    def test_existing_nerve_redirects_to_invoke(self):
        """Exact name match — must redirect to invoke_nerve."""
        from arqitect.brain.dispatch import resolve_synthesize_redirect
        d = SynthesizeDecisionFactory.build(name="joke_nerve", description="tells jokes")
        available = ["joke_nerve", "awareness", "communication"]
        action, result = resolve_synthesize_redirect(
            Action.SYNTHESIZE_NERVE, as_dict(d), available, {}, "tell me a joke"
        )
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "joke_nerve"

    def test_fuzzy_match_redirects_to_invoke(self):
        """Name differs but description matches an existing nerve."""
        from arqitect.brain.dispatch import resolve_synthesize_redirect
        catalog = {"greeting_nerve": "generates friendly greetings based on user preferences"}
        d = SynthesizeDecisionFactory.build(
            name="hello_nerve",
            description="generates friendly greetings based on preferences",
        )
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[("greeting_nerve", 4.0)]):
            action, result = resolve_synthesize_redirect(
                Action.SYNTHESIZE_NERVE, as_dict(d), ["greeting_nerve"], catalog, "hello"
            )
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "greeting_nerve"

    def test_truly_new_nerve_stays_synthesize(self):
        """No match at all — must stay as synthesize_nerve."""
        from arqitect.brain.dispatch import resolve_synthesize_redirect
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[]):
            action, result = resolve_synthesize_redirect(
                Action.SYNTHESIZE_NERVE, as_dict(d), ["joke_nerve"], {}, "how far is mars"
            )
        assert action == Action.SYNTHESIZE_NERVE

    def test_non_synthesize_action_passes_through(self):
        """Only synthesize actions should be checked — others pass through."""
        from arqitect.brain.dispatch import resolve_synthesize_redirect
        d = InvokeDecisionFactory.build()
        action, result = resolve_synthesize_redirect(
            Action.INVOKE_NERVE, as_dict(d), ["joke_nerve"], {}, "tell me a joke"
        )
        assert action == Action.INVOKE_NERVE

    def test_fuzzy_match_below_threshold_stays_synthesize(self):
        """Fuzzy match exists but score too low — must synthesize."""
        from arqitect.brain.dispatch import resolve_synthesize_redirect
        catalog = {"weather_nerve": "weather data"}
        d = SynthesizeDecisionFactory.build(name="news_nerve", description="fetches news articles")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[("weather_nerve", 1.5)]):
            action, _ = resolve_synthesize_redirect(
                Action.SYNTHESIZE_NERVE, as_dict(d), ["weather_nerve"], catalog, "latest news"
            )
        assert action == Action.SYNTHESIZE_NERVE

    def test_fuzzy_match_insufficient_margin_stays_synthesize(self):
        """Two nerves score similarly — ambiguous, must synthesize."""
        from arqitect.brain.dispatch import resolve_synthesize_redirect
        catalog = {"weather_nerve": "weather", "news_nerve": "news"}
        d = SynthesizeDecisionFactory.build(name="info_nerve", description="general info")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[("weather_nerve", 3.0), ("news_nerve", 2.8)]):
            action, _ = resolve_synthesize_redirect(
                Action.SYNTHESIZE_NERVE, as_dict(d), ["weather_nerve", "news_nerve"], catalog, "info"
            )
        assert action == Action.SYNTHESIZE_NERVE

    def test_missing_description_stays_synthesize(self):
        """No description — can't fuzzy match, stays as synthesize."""
        from arqitect.brain.dispatch import resolve_synthesize_redirect
        d = SynthesizeDecisionFactory.build(name="mystery_nerve", description="")
        action, _ = resolve_synthesize_redirect(
            Action.SYNTHESIZE_NERVE, as_dict(d), [], {}, "something"
        )
        assert action == Action.SYNTHESIZE_NERVE

    def test_case_insensitive_name_match(self):
        """Name normalization: 'Joke_Nerve' should match 'joke_nerve'."""
        from arqitect.brain.dispatch import resolve_synthesize_redirect
        d = SynthesizeDecisionFactory.build(name="Joke_Nerve", description="tells jokes")
        action, result = resolve_synthesize_redirect(
            Action.SYNTHESIZE_NERVE, as_dict(d), ["joke_nerve"], {}, "tell me a joke"
        )
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "joke_nerve"


# ---------------------------------------------------------------------------
# 3. dispatch_action — full routing integration
# ---------------------------------------------------------------------------

class TestDispatchAction:
    """dispatch_action must route to the correct handler after normalization
    and redirect. These tests verify the full pipeline.
    LLM is always mocked — we test how the rest of the code handles
    different mocked responses."""

    def _make_ctx(self, decision_obj, task="hello", nerve_catalog=None, **kwargs):
        """Build a DispatchContext from a typed decision factory object.

        Args:
            decision_obj: A typed decision dataclass (or raw dict).
            task: User's input task.
            nerve_catalog: {nerve_name: description} mapping.
            **kwargs: Forwarded to DispatchContextFactory.build().
        """
        raw = as_dict(decision_obj) if hasattr(decision_obj, '__dataclass_fields__') else decision_obj
        catalog = nerve_catalog or {}
        return DispatchContextFactory.build(
            task=task,
            decision=raw,
            nerve_catalog=catalog,
            available=list(catalog.keys()),
            **kwargs,
        )

    def test_synthesize_existing_nerve_reaches_invoke(self):
        """THE BUG: synthesize for existing nerve must reach invoke, not loop."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"greeting_nerve": "friendly greetings", "awareness": "identity"}
        d = SynthesizeDecisionFactory.build(name="greeting_nerve", description="friendly greetings")
        ctx = self._make_ctx(d, task="hi", nerve_catalog=catalog)
        with patch("arqitect.brain.dispatch.invoke_nerve", return_value='{"response": "Hello!"}') as mock_invoke, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.brain.dispatch.r") as mock_r, \
             patch("arqitect.brain.dispatch.llm_generate", return_value="Hello there!"), \
             patch("arqitect.brain.dispatch._resolve_adapter", return_value={"system_prompt": "be nice"}):
            mock_mem.cold.get_user_role.return_value = "user"
            mock_mem.cold.get_nerve_metadata.return_value = {"role": "creative"}
            mock_mem.cold.list_nerves.return_value = list(catalog.keys())
            mock_r.hget.return_value = None
            result = dispatch_action(ctx)
            mock_invoke.assert_called_once()
            assert result is not None

    def test_clarify_returns_message(self):
        from arqitect.brain.dispatch import dispatch_action
        d = ClarifyDecisionFactory.build(message="What do you mean?", suggestions=["option A", "option B"])
        ctx = self._make_ctx(d)
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.mem"):
            result = dispatch_action(ctx)
        assert "What do you mean?" in result
        assert "option A" in result

    def test_feedback_positive_records_episode(self):
        from arqitect.brain.dispatch import dispatch_action
        d = FeedbackDecisionFactory.build(sentiment="positive", message="Great!")
        ctx = self._make_ctx(d)
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch._cb_success"), \
             patch("arqitect.brain.dispatch._cb_failure"):
            mock_mem.warm.recall.return_value = [{"task": "prev", "nerve": "joke_nerve", "tool": ""}]
            result = dispatch_action(ctx)
        assert result == "Great!"

    def test_feedback_negative_records_failure(self):
        """Negative feedback must record a failed episode and trigger circuit breaker."""
        from arqitect.brain.dispatch import dispatch_action
        d = FeedbackDecisionFactory.build(sentiment="negative", message="That was wrong")
        ctx = self._make_ctx(d)
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch._cb_success") as mock_success, \
             patch("arqitect.brain.dispatch._cb_failure") as mock_failure:
            mock_mem.warm.recall.return_value = [{"task": "prev", "nerve": "joke_nerve", "tool": ""}]
            result = dispatch_action(ctx)
        assert result == "That was wrong"
        mock_failure.assert_called_once_with("joke_nerve")
        mock_success.assert_not_called()

    def test_update_context_stores_facts(self):
        from arqitect.brain.dispatch import dispatch_action
        d = UpdateContextDecisionFactory.build(
            context={"city": "Tel Aviv"},
            message="Noted!",
        )
        ctx = self._make_ctx(d, user_id="user1")
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            result = dispatch_action(ctx)
        assert result == "Noted!"
        mock_mem.hot.update_session.assert_called_once()
        mock_mem.cold.set_user_fact.assert_called_once_with("user1", "city", "Tel Aviv", confidence=1.0)

    def test_update_context_location_normalized_to_city(self):
        """LLM says 'location' but our session key is 'city'."""
        from arqitect.brain.dispatch import dispatch_action
        d = UpdateContextDecisionFactory.build(
            context={"location": "Haifa"},
            message="Got it!",
        )
        ctx = self._make_ctx(d, user_id="user1")
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            result = dispatch_action(ctx)
        mock_mem.cold.set_user_fact.assert_called_once_with("user1", "city", "Haifa", confidence=1.0)

    def test_unknown_action_re_thinks(self):
        from arqitect.brain.dispatch import dispatch_action
        d = InvokeDecisionFactory.build(action="fly_to_moon")
        ctx = self._make_ctx(d)
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.mem"):
            result = dispatch_action(ctx)
        ctx.think_fn.assert_called_once()

    def test_respond_redirects_to_awareness(self):
        from arqitect.brain.dispatch import dispatch_action
        d = RespondDecisionFactory.build(message="I am sentient")
        ctx = self._make_ctx(d)
        with patch("arqitect.brain.dispatch.mem"):
            result = dispatch_action(ctx)
        ctx.think_fn.assert_called_once()
        call_args = ctx.think_fn.call_args
        assert "awareness" in str(call_args).lower()

    def test_chain_single_step_converts_to_invoke(self):
        """Single-step chain should be treated as a direct invoke."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"joke_nerve": "tells jokes"}
        step = ChainStepFactory.build(nerve="joke_nerve", args="tell joke")
        d = ChainDecisionFactory.build(steps=[step], goal="tell a joke")
        ctx = self._make_ctx(d, nerve_catalog=catalog, task="tell me a joke")
        with patch("arqitect.brain.dispatch.invoke_nerve", return_value='{"response": "Why did the chicken..."}') as mock_invoke, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.brain.dispatch.r") as mock_r, \
             patch("arqitect.brain.dispatch.llm_generate", return_value="Why did the chicken cross the road?"), \
             patch("arqitect.brain.dispatch._resolve_adapter", return_value={"system_prompt": "be nice"}):
            mock_mem.cold.get_user_role.return_value = "user"
            mock_mem.cold.get_nerve_metadata.return_value = {"role": "creative"}
            mock_mem.cold.list_nerves.return_value = list(catalog.keys())
            mock_r.hget.return_value = None
            result = dispatch_action(ctx)
        mock_invoke.assert_called_once()

    def test_synthesize_truly_new_nerve_synthesizes(self):
        """New nerve that doesn't exist anywhere should be synthesized."""
        from arqitect.brain.dispatch import dispatch_action
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        ctx = self._make_ctx(d, nerve_catalog={"awareness": "identity"}, user_id="user123")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[]), \
             patch("arqitect.brain.dispatch.synthesize_nerve", return_value=("astronomy_nerve", "/nerves/astronomy_nerve/nerve.py")), \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            mock_mem.cold.get_user_role.return_value = "user"
            result = dispatch_action(ctx)
        ctx.think_fn.assert_called_once()

    def test_depth_not_incremented_on_redirect(self):
        """Synthesize→invoke redirect must NOT increase depth (it's not a re-think)."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"joke_nerve": "tells jokes"}
        d = SynthesizeDecisionFactory.build(name="joke_nerve", description="tells jokes")
        ctx = self._make_ctx(d, nerve_catalog=catalog, depth=4)
        with patch("arqitect.brain.dispatch.invoke_nerve", return_value='{"response": "ha"}') as mock_invoke, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.brain.dispatch.r") as mock_r, \
             patch("arqitect.brain.dispatch.llm_generate", return_value="ha!"), \
             patch("arqitect.brain.dispatch._resolve_adapter", return_value={"system_prompt": "ok"}):
            mock_mem.cold.get_user_role.return_value = "user"
            mock_mem.cold.get_nerve_metadata.return_value = {"role": "tool"}
            mock_mem.cold.list_nerves.return_value = list(catalog.keys())
            mock_r.hget.return_value = None
            result = dispatch_action(ctx)
        mock_invoke.assert_called_once()
        ctx.think_fn.assert_not_called()

    def test_invoke_nerve_error_returns_graceful_message(self):
        """When nerve returns an error, dispatch must return a graceful message."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"joke_nerve": "tells jokes"}
        d = InvokeDecisionFactory.build(name="joke_nerve", args="tell me a joke")
        ctx = self._make_ctx(d, nerve_catalog=catalog, task="tell me a joke")
        with patch("arqitect.brain.dispatch.invoke_nerve", return_value='{"response": "Error: connection refused"}'), \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.brain.dispatch.r") as mock_r, \
             patch("arqitect.brain.dispatch._is_nerve_error", return_value=True), \
             patch("arqitect.brain.dispatch._graceful_failure_message", return_value="Sorry, something went wrong."), \
             patch("arqitect.brain.dispatch._cb_success"), \
             patch("arqitect.brain.dispatch._cb_failure") as mock_fail:
            mock_mem.cold.get_user_role.return_value = "user"
            mock_mem.cold.get_nerve_metadata.return_value = {"role": "tool"}
            mock_r.hget.return_value = None
            result = dispatch_action(ctx)
        assert result == "Sorry, something went wrong."
        mock_fail.assert_called()

    def test_invoke_needs_data_re_thinks(self):
        """When nerve needs more data, dispatch must re-think to find it."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"weather_nerve": "weather data"}
        d = InvokeDecisionFactory.build(name="weather_nerve", args="weather forecast")
        ctx = self._make_ctx(d, nerve_catalog=catalog, task="weather forecast")
        needs_data_response = json.dumps({"status": "needs_data", "needs": "city name", "tool": "weather_api"})
        with patch("arqitect.brain.dispatch.invoke_nerve", return_value=needs_data_response), \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.brain.dispatch.r") as mock_r, \
             patch("arqitect.brain.dispatch._cb_failure"):
            mock_mem.cold.get_user_role.return_value = "user"
            mock_mem.cold.get_nerve_metadata.return_value = {"role": "tool"}
            mock_r.hget.return_value = None
            result = dispatch_action(ctx)
        ctx.think_fn.assert_called_once()
        call_args_str = str(ctx.think_fn.call_args)
        assert "city name" in call_args_str

    def test_invoke_wrong_nerve_re_thinks_without_reinvoking(self):
        """When nerve says wrong_nerve, dispatch must NOT re-invoke it."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"joke_nerve": "tells jokes"}
        d = InvokeDecisionFactory.build(name="joke_nerve", args="translate this")
        ctx = self._make_ctx(d, nerve_catalog=catalog, task="translate this")
        wrong_response = json.dumps({"status": "wrong_nerve", "reason": "not a translation nerve"})
        with patch("arqitect.brain.dispatch.invoke_nerve", return_value=wrong_response), \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.brain.dispatch.r") as mock_r, \
             patch("arqitect.brain.dispatch._cb_failure"):
            mock_mem.cold.get_user_role.return_value = "user"
            mock_mem.cold.get_nerve_metadata.return_value = {"role": "tool"}
            mock_r.hget.return_value = None
            result = dispatch_action(ctx)
        ctx.think_fn.assert_called_once()
        call_args_str = str(ctx.think_fn.call_args)
        assert "Do NOT re-invoke" in call_args_str

    def test_invoke_nonexistent_nerve_triggers_synthesize_hint(self):
        """Invoking a nerve that doesn't exist should hint to synthesize."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"awareness": "identity"}
        d = InvokeDecisionFactory.build(name="missing_nerve", args="do something")
        ctx = self._make_ctx(d, nerve_catalog=catalog, task="do something")
        with patch("arqitect.brain.dispatch.mem") as mock_mem:
            mock_mem.cold.list_nerves.return_value = ["awareness"]
            result = dispatch_action(ctx)
        ctx.think_fn.assert_called_once()
        call_args_str = str(ctx.think_fn.call_args)
        assert "does not exist" in call_args_str
        assert "Synthesize" in call_args_str

    def test_invoke_permission_denied(self):
        """Restricted nerve returns restriction message without invoking."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"touch": "file ops"}
        d = InvokeDecisionFactory.build(name="touch", args="rm -rf /")
        ctx = self._make_ctx(d, nerve_catalog=catalog, task="delete everything")
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=False), \
             patch("arqitect.brain.dispatch.invoke_nerve") as mock_invoke:
            mock_mem.cold.get_user_role.return_value = "anon"
            mock_mem.cold.get_nerve_metadata.return_value = {"role": "code"}
            result = dispatch_action(ctx)
        mock_invoke.assert_not_called()
        assert result is not None


# ---------------------------------------------------------------------------
# 4. Synthesis permission enforcement — anon cannot fabricate nerves
# ---------------------------------------------------------------------------

class TestSynthesisPermissionEnforcement:
    """Anon users can use existing nerves but cannot synthesize new ones.
    Registered users (role >= 'user') can synthesize freely."""

    def _make_ctx(self, decision_obj, task="hello", nerve_catalog=None, **kwargs):
        raw = as_dict(decision_obj) if hasattr(decision_obj, '__dataclass_fields__') else decision_obj
        catalog = nerve_catalog or {}
        return DispatchContextFactory.build(
            task=task,
            decision=raw,
            nerve_catalog=catalog,
            available=list(catalog.keys()),
            **kwargs,
        )

    def test_anon_synthesize_blocked(self):
        """Anon user attempting to synthesize a new nerve must be denied."""
        from arqitect.brain.dispatch import dispatch_action
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        ctx = self._make_ctx(d, nerve_catalog={"awareness": "identity"}, user_id="")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[]), \
             patch("arqitect.brain.dispatch.synthesize_nerve") as mock_synth, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response") as mock_pub, \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            mock_mem.cold.get_user_role.return_value = "anon"
            result = dispatch_action(ctx)
        mock_synth.assert_not_called()
        ctx.think_fn.assert_not_called()
        assert result is not None
        # Response should tell the user to send their email
        assert "email" in result.lower()

    def test_anon_synthesize_blocked_signals_request_identity(self):
        """Synthesis blocked for anon must publish with request_identity=True."""
        from arqitect.brain.dispatch import dispatch_action
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        ctx = self._make_ctx(d, nerve_catalog={"awareness": "identity"}, user_id="")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[]), \
             patch("arqitect.brain.dispatch.synthesize_nerve") as mock_synth, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response") as mock_pub, \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            mock_mem.cold.get_user_role.return_value = "anon"
            dispatch_action(ctx)
        mock_pub.assert_called_once()
        _, kwargs = mock_pub.call_args
        assert kwargs.get("request_identity") is True

    def test_registered_user_synthesize_no_request_identity(self):
        """Synthesis allowed for identified user must NOT signal request_identity."""
        from arqitect.brain.dispatch import dispatch_action
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        ctx = self._make_ctx(d, nerve_catalog={"awareness": "identity"}, user_id="user123")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[]), \
             patch("arqitect.brain.dispatch.synthesize_nerve", return_value=("astronomy_nerve", "/path")) as mock_synth, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response") as mock_pub, \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            mock_mem.cold.get_user_role.return_value = "user"
            dispatch_action(ctx)
        # publish_response should NOT have been called with request_identity
        for call in mock_pub.call_args_list:
            kwargs = call[1] if call[1] else {}
            assert not kwargs.get("request_identity")

    def test_registered_user_synthesize_allowed(self):
        """Registered user (role='user') can synthesize new nerves."""
        from arqitect.brain.dispatch import dispatch_action
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        ctx = self._make_ctx(d, nerve_catalog={"awareness": "identity"}, user_id="user123")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[]), \
             patch("arqitect.brain.dispatch.synthesize_nerve", return_value=("astronomy_nerve", "/nerves/astronomy_nerve/nerve.py")) as mock_synth, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            mock_mem.cold.get_user_role.return_value = "user"
            result = dispatch_action(ctx)
        mock_synth.assert_called_once()
        ctx.think_fn.assert_called_once()

    def test_anon_invoke_existing_nerve_still_works(self):
        """Anon users must still be able to invoke existing non-restricted nerves."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"joke_nerve": "tells jokes"}
        d = InvokeDecisionFactory.build(name="joke_nerve", args="tell me a joke")
        ctx = self._make_ctx(d, nerve_catalog=catalog, task="tell me a joke", user_id="")
        with patch("arqitect.brain.dispatch.invoke_nerve", return_value='{"response": "Why did the chicken..."}') as mock_invoke, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.brain.dispatch.r") as mock_r, \
             patch("arqitect.brain.dispatch.llm_generate", return_value="Why did the chicken cross the road?"), \
             patch("arqitect.brain.dispatch._resolve_adapter", return_value={"system_prompt": "be nice"}):
            mock_mem.cold.get_user_role.return_value = "anon"
            mock_mem.cold.get_nerve_metadata.return_value = {"role": "tool"}
            mock_mem.cold.list_nerves.return_value = list(catalog.keys())
            mock_r.hget.return_value = None
            result = dispatch_action(ctx)
        mock_invoke.assert_called_once()

    def test_anon_chain_blocked_at_synthesis_step(self):
        """Chain that requires synthesizing a missing nerve must block for anon."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"joke_nerve": "tells jokes"}
        # Chain has two steps: one existing nerve, one that needs synthesis
        steps = [
            ChainStepFactory.build(nerve="joke_nerve", args="tell joke"),
            ChainStepFactory.build(nerve="missing_nerve", args="do something"),
        ]
        d = ChainDecisionFactory.build(steps=steps, goal="joke then magic")
        ctx = self._make_ctx(d, nerve_catalog=catalog, task="joke then magic", user_id="")
        with patch("arqitect.brain.dispatch.invoke_nerve", return_value='{"response": "ha"}'), \
             patch("arqitect.brain.dispatch.synthesize_nerve") as mock_synth, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response") as mock_pub, \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.brain.dispatch.r") as mock_r, \
             patch("arqitect.brain.dispatch.llm_generate", return_value="ha!"), \
             patch("arqitect.brain.dispatch._resolve_adapter", return_value={"system_prompt": "ok"}):
            mock_mem.cold.get_user_role.return_value = "anon"
            mock_mem.cold.get_nerve_metadata.return_value = {"role": "tool"}
            mock_mem.cold.list_nerves.return_value = list(catalog.keys())
            mock_r.hget.return_value = None
            result = dispatch_action(ctx)
        mock_synth.assert_not_called()
        assert result is not None

    def test_registered_user_chain_with_synthesis_works(self):
        """Registered user's chain can synthesize missing nerves."""
        from arqitect.brain.dispatch import dispatch_action
        catalog = {"joke_nerve": "tells jokes"}
        steps = [
            ChainStepFactory.build(nerve="joke_nerve", args="tell joke"),
            ChainStepFactory.build(nerve="missing_nerve", args="do something"),
        ]
        d = ChainDecisionFactory.build(steps=steps, goal="joke then magic")
        ctx = self._make_ctx(d, nerve_catalog=catalog, task="joke then magic", user_id="user123")
        with patch("arqitect.brain.dispatch.invoke_nerve", return_value='{"response": "ha"}'), \
             patch("arqitect.brain.dispatch.synthesize_nerve", return_value=("missing_nerve", "/nerves/missing_nerve/nerve.py")) as mock_synth, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.brain.dispatch.r") as mock_r, \
             patch("arqitect.brain.dispatch.llm_generate", return_value="result"), \
             patch("arqitect.brain.dispatch._resolve_adapter", return_value={"system_prompt": "ok"}):
            mock_mem.cold.get_user_role.return_value = "user"
            mock_mem.cold.get_nerve_metadata.return_value = {"role": "tool"}
            mock_mem.cold.list_nerves.return_value = list(catalog.keys())
            mock_r.hget.return_value = None
            result = dispatch_action(ctx)
        mock_synth.assert_called_once()
