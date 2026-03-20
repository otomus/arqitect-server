"""Tests for brain action dispatch — normalization, redirect, and routing.

TDD: these tests define the contract for arqitect.brain.dispatch.
Each test targets a specific flow or edge case in the routing pipeline.
All test data is produced by typed factories — no bare dict literals.

Uses hypothesis for property-based fuzz testing of parse_decision and
normalize_action. Uses dirty_equals for expressive assertions.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from dirty_equals import IsInstance, IsStr, Contains
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from arqitect.types import Action, Sense
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
    normalize_action,
    resolve_synthesize_redirect,
    dispatch_action,
    _coerce_args,
    _parse_nerve_output,
    FUZZY_MATCH_THRESHOLD,
    FUZZY_MATCH_MARGIN,
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
# Hypothesis strategies for decision dicts
# ---------------------------------------------------------------------------

_KNOWN_ACTIONS = [
    Action.INVOKE_NERVE, Action.SYNTHESIZE_NERVE, Action.CHAIN_NERVES,
    Action.UPDATE_CONTEXT, Action.RESPOND, Action.CLARIFY, Action.FEEDBACK,
    Action.USE_SENSE,
]

_action_strategy = st.sampled_from(_KNOWN_ACTIONS)
_name_strategy = st.from_regex(r"[a-z][a-z0-9_]{2,20}", fullmatch=True)
_args_strategy = st.text(min_size=0, max_size=100)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(decision_obj, task="hello", nerve_catalog=None, **kwargs):
    """Build a DispatchContext from a typed decision factory object.

    Args:
        decision_obj: A typed decision dataclass (or raw dict).
        task: User's input task.
        nerve_catalog: {nerve_name: description} mapping.
        **kwargs: Forwarded to DispatchContextFactory.build().

    Returns:
        A DispatchContext ready for dispatch_action.
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


# ---------------------------------------------------------------------------
# 0. _coerce_args — pure function, great hypothesis target
# ---------------------------------------------------------------------------

class TestCoerceArgs:
    """_coerce_args must convert any args value to a string."""

    @pytest.mark.timeout(10)
    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.text(max_size=50), max_size=5))
    @settings(max_examples=50)
    def test_dict_args_become_valid_json_string(self, d):
        """Any dict args must produce a parseable JSON string."""
        result = _coerce_args(d)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == d

    @pytest.mark.timeout(10)
    @given(st.text(max_size=200))
    @settings(max_examples=50)
    def test_string_args_pass_through(self, s):
        """String args must pass through unchanged."""
        assert _coerce_args(s) == (s if s else "")

    @pytest.mark.timeout(10)
    def test_none_becomes_empty_string(self):
        assert _coerce_args(None) == ""

    @pytest.mark.timeout(10)
    def test_empty_dict_becomes_json(self):
        assert _coerce_args({}) == "{}"


# ---------------------------------------------------------------------------
# 0b. _parse_nerve_output — pure JSON parsing
# ---------------------------------------------------------------------------

class TestParseNerveOutput:
    """_parse_nerve_output must extract JSON from noisy nerve stdout."""

    @pytest.mark.timeout(10)
    def test_clean_json(self):
        result = _parse_nerve_output('{"response": "hello"}')
        assert result == {"response": "hello"}

    @pytest.mark.timeout(10)
    def test_json_buried_in_logs(self):
        output = "Loading model...\nWarm-up done\n{\"response\": \"ok\"}\n"
        result = _parse_nerve_output(output)
        assert result["response"] == "ok"

    @pytest.mark.timeout(10)
    def test_non_json_returns_empty_dict(self):
        assert _parse_nerve_output("just plain text") == {}

    @pytest.mark.timeout(10)
    @given(st.dictionaries(st.text(min_size=1, max_size=10).filter(lambda s: '"' not in s and '\\' not in s),
                           st.text(max_size=30).filter(lambda s: '"' not in s and '\\' not in s),
                           min_size=1, max_size=3))
    @settings(max_examples=30)
    def test_valid_json_always_roundtrips(self, d):
        """Any valid JSON dict must be parsed back correctly."""
        raw = json.dumps(d)
        result = _parse_nerve_output(raw)
        assert result == d


# ---------------------------------------------------------------------------
# 1. parse_decision — typed parsing of raw LLM dicts
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestParseDecision:
    """parse_decision must convert raw dicts into typed Decision dataclasses."""

    def test_invoke_decision(self):
        d = InvokeDecisionFactory.build()
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert parsed == IsInstance(InvokeDecision)
        assert parsed.name == d.name
        assert parsed.args == d.args

    def test_synthesize_decision(self):
        d = SynthesizeDecisionFactory.build()
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert parsed == IsInstance(SynthesizeDecision)
        assert parsed.name == d.name
        assert parsed.description == d.description

    def test_chain_decision(self):
        steps = [ChainStepFactory.build(nerve="joke_nerve"), ChainStepFactory.build(nerve="translate_nerve")]
        d = ChainDecisionFactory.build(steps=steps)
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert parsed == IsInstance(ChainDecision)
        assert len(parsed.steps) == 2
        assert parsed.steps[0].nerve == "joke_nerve"

    def test_clarify_decision(self):
        d = ClarifyDecisionFactory.build()
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert parsed == IsInstance(ClarifyDecision)
        assert parsed.message == d.message
        assert parsed.suggestions == d.suggestions

    def test_feedback_decision(self):
        d = FeedbackDecisionFactory.build(sentiment="negative", message="That was wrong")
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert parsed == IsInstance(FeedbackDecision)
        assert parsed.sentiment == "negative"
        assert parsed.message == "That was wrong"

    def test_update_context_decision(self):
        d = UpdateContextDecisionFactory.build(context={"city": "Tel Aviv"})
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert parsed == IsInstance(UpdateContextDecision)
        assert parsed.context["city"] == "Tel Aviv"

    def test_respond_decision(self):
        d = RespondDecisionFactory.build()
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert parsed == IsInstance(RespondDecision)
        assert parsed.message == d.message

    def test_use_sense_decision(self):
        d = UseSenseDecisionFactory.build(sense="sight", args="describe image")
        raw = as_dict(d)
        parsed = parse_decision(raw)
        assert parsed == IsInstance(UseSenseDecision)
        assert parsed.sense == "sight"

    def test_missing_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            parse_decision({})

    def test_unknown_action_returns_invoke_fallback(self):
        parsed = parse_decision({"action": "fly_to_moon", "name": "x"})
        assert parsed == IsInstance(InvokeDecision)
        assert parsed.action == "fly_to_moon"

    def test_dict_args_coerced_to_string(self):
        raw = {"action": "invoke_nerve", "name": "test", "args": {"key": "val"}}
        parsed = parse_decision(raw)
        assert parsed.args == IsStr()
        assert "key" in parsed.args

    @given(
        action=_action_strategy,
        name=_name_strategy,
    )
    @settings(max_examples=50)
    def test_any_known_action_parses_without_error(self, action, name):
        """Any known action string must parse without raising."""
        raw = {"action": action, "name": name, "args": "test"}
        parsed = parse_decision(raw)
        assert parsed is not None
        assert hasattr(parsed, "action")

    @given(action=st.text(min_size=1, max_size=30).filter(lambda a: a not in _KNOWN_ACTIONS))
    @settings(max_examples=30)
    def test_unknown_action_falls_back_to_invoke(self, action):
        """Any unknown action must produce an InvokeDecision fallback."""
        raw = {"action": action, "name": "x"}
        parsed = parse_decision(raw)
        assert parsed == IsInstance(InvokeDecision)
        assert parsed.action == action


# ---------------------------------------------------------------------------
# 2. normalize_action — fix LLM output before dispatch
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestNormalizeAction:
    """LLM output is messy. normalize_action must clean it up."""

    def test_valid_action_passes_through(self):
        d = InvokeDecisionFactory.build()
        action, result = normalize_action(as_dict(d), nerve_catalog={"joke_nerve": "jokes"})
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "joke_nerve"

    def test_typo_gets_fuzzy_matched(self):
        """LLM typo like 'invok_nerve' should be corrected."""
        d = InvokeDecisionFactory.build(action="invok_nerve", name="weather_nerve")
        action, _ = normalize_action(as_dict(d), nerve_catalog={})
        assert action == Action.INVOKE_NERVE

    def test_use_sense_maps_to_invoke(self):
        """use_sense is not a real action — should become invoke_nerve."""
        d = UseSenseDecisionFactory.build(sense="touch", args="list files")
        action, result = normalize_action(as_dict(d), nerve_catalog={})
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "touch"

    def test_nerve_name_as_action(self):
        """LLM returns action='touch' instead of invoke_nerve('touch')."""
        catalog = {"touch": "file ops", "awareness": "identity"}
        d = InvokeDecisionFactory.build(action="touch", args="list files")
        action, result = normalize_action(as_dict(d), nerve_catalog=catalog)
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "touch"

    def test_sense_name_as_action(self):
        """LLM returns action='awareness' — a core sense name."""
        catalog = {"awareness": "identity"}
        d = InvokeDecisionFactory.build(action="awareness", args="who are you?")
        action, result = normalize_action(as_dict(d), nerve_catalog=catalog)
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "awareness"

    def test_unknown_action_stays_unknown(self):
        """Truly unknown action that doesn't fuzzy-match stays as-is."""
        d = InvokeDecisionFactory.build(action="do_magic", args="abracadabra")
        action, _ = normalize_action(as_dict(d), nerve_catalog={})
        assert action == "do_magic"

    def test_use_sense_with_dict_args_coerced_to_string(self):
        """use_sense args may be a dict — must be coerced to JSON string."""
        raw = {"action": "use_sense", "sense": "touch", "args": {"mode": "list", "path": "/home"}}
        action, result = normalize_action(raw, nerve_catalog={})
        assert action == Action.INVOKE_NERVE
        assert result["args"] == IsStr()

    def test_use_sense_without_sense_field_uses_name(self):
        """use_sense with 'name' instead of 'sense' field."""
        d = UseSenseDecisionFactory.build(sense="", name="hearing", args="play music")
        action, result = normalize_action(as_dict(d), nerve_catalog={})
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "hearing"

    @given(
        nerve_name=_name_strategy,
    )
    @settings(max_examples=30)
    def test_nerve_in_catalog_always_becomes_invoke(self, nerve_name):
        """Any action matching a catalog key must resolve to invoke_nerve."""
        catalog = {nerve_name: "some description"}
        raw = {"action": nerve_name, "args": "test"}
        action, result = normalize_action(raw, nerve_catalog=catalog)
        assert action == Action.INVOKE_NERVE
        assert result["name"] == nerve_name


# ---------------------------------------------------------------------------
# 3. resolve_synthesize_redirect — stop LLM from re-synthesizing existing nerves
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestResolveSynthesizeRedirect:
    """When the LLM asks to synthesize a nerve that already exists,
    redirect to invoke instead of looping."""

    def test_existing_nerve_redirects_to_invoke(self):
        """Exact name match — must redirect to invoke_nerve."""
        d = SynthesizeDecisionFactory.build(name="joke_nerve", description="tells jokes")
        available = ["joke_nerve", "awareness", "communication"]
        action, result = resolve_synthesize_redirect(
            Action.SYNTHESIZE_NERVE, as_dict(d), available, {}, "tell me a joke"
        )
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "joke_nerve"

    def test_fuzzy_match_redirects_to_invoke(self):
        """Name differs but description matches an existing nerve."""
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
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[]):
            action, result = resolve_synthesize_redirect(
                Action.SYNTHESIZE_NERVE, as_dict(d), ["joke_nerve"], {}, "how far is mars"
            )
        assert action == Action.SYNTHESIZE_NERVE

    def test_non_synthesize_action_passes_through(self):
        """Only synthesize actions should be checked — others pass through."""
        d = InvokeDecisionFactory.build()
        action, result = resolve_synthesize_redirect(
            Action.INVOKE_NERVE, as_dict(d), ["joke_nerve"], {}, "tell me a joke"
        )
        assert action == Action.INVOKE_NERVE

    def test_fuzzy_match_below_threshold_stays_synthesize(self):
        """Fuzzy match exists but score too low — must synthesize."""
        catalog = {"weather_nerve": "weather data"}
        d = SynthesizeDecisionFactory.build(name="news_nerve", description="fetches news articles")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[("weather_nerve", 1.5)]):
            action, _ = resolve_synthesize_redirect(
                Action.SYNTHESIZE_NERVE, as_dict(d), ["weather_nerve"], catalog, "latest news"
            )
        assert action == Action.SYNTHESIZE_NERVE

    def test_fuzzy_match_insufficient_margin_stays_synthesize(self):
        """Two nerves score similarly — ambiguous, must synthesize."""
        catalog = {"weather_nerve": "weather", "news_nerve": "news"}
        d = SynthesizeDecisionFactory.build(name="info_nerve", description="general info")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[("weather_nerve", 3.0), ("news_nerve", 2.8)]):
            action, _ = resolve_synthesize_redirect(
                Action.SYNTHESIZE_NERVE, as_dict(d), ["weather_nerve", "news_nerve"], catalog, "info"
            )
        assert action == Action.SYNTHESIZE_NERVE

    def test_missing_description_stays_synthesize(self):
        """No description — can't fuzzy match, stays as synthesize."""
        d = SynthesizeDecisionFactory.build(name="mystery_nerve", description="")
        action, _ = resolve_synthesize_redirect(
            Action.SYNTHESIZE_NERVE, as_dict(d), [], {}, "something"
        )
        assert action == Action.SYNTHESIZE_NERVE

    def test_case_insensitive_name_match(self):
        """Name normalization: 'Joke_Nerve' should match 'joke_nerve'."""
        d = SynthesizeDecisionFactory.build(name="Joke_Nerve", description="tells jokes")
        action, result = resolve_synthesize_redirect(
            Action.SYNTHESIZE_NERVE, as_dict(d), ["joke_nerve"], {}, "tell me a joke"
        )
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "joke_nerve"

    @given(nerve_name=_name_strategy)
    @settings(max_examples=30)
    def test_exact_match_always_redirects(self, nerve_name):
        """Any exact name match (case-insensitive) must always redirect."""
        raw = {"action": Action.SYNTHESIZE_NERVE, "name": nerve_name, "description": "whatever"}
        action, result = resolve_synthesize_redirect(
            Action.SYNTHESIZE_NERVE, raw, [nerve_name], {}, "task"
        )
        assert action == Action.INVOKE_NERVE
        assert result["name"] == nerve_name

    @given(
        best_score=st.floats(min_value=FUZZY_MATCH_THRESHOLD, max_value=10.0),
        margin=st.floats(min_value=FUZZY_MATCH_MARGIN, max_value=5.0),
    )
    @settings(max_examples=30)
    def test_high_score_with_margin_always_redirects(self, best_score, margin):
        """Score above threshold with sufficient margin must always redirect."""
        second_score = best_score - margin
        with patch("arqitect.brain.dispatch.match_nerves",
                    return_value=[("matched_nerve", best_score), ("other", second_score)]):
            action, result = resolve_synthesize_redirect(
                Action.SYNTHESIZE_NERVE,
                {"action": Action.SYNTHESIZE_NERVE, "name": "new_nerve", "description": "something"},
                ["matched_nerve", "other"],
                {"matched_nerve": "desc", "other": "desc"},
                "task",
            )
        assert action == Action.INVOKE_NERVE
        assert result["name"] == "matched_nerve"


# ---------------------------------------------------------------------------
# 4. dispatch_action — full routing integration
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestDispatchAction:
    """dispatch_action must route to the correct handler after normalization
    and redirect. These tests verify the full pipeline.
    LLM is always mocked — we test how the rest of the code handles
    different mocked responses."""

    def test_synthesize_existing_nerve_reaches_invoke(self):
        """THE BUG: synthesize for existing nerve must reach invoke, not loop."""
        catalog = {"greeting_nerve": "friendly greetings", "awareness": "identity"}
        d = SynthesizeDecisionFactory.build(name="greeting_nerve", description="friendly greetings")
        ctx = _make_ctx(d, task="hi", nerve_catalog=catalog)
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
        d = ClarifyDecisionFactory.build(message="What do you mean?", suggestions=["option A", "option B"])
        ctx = _make_ctx(d)
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.mem"):
            result = dispatch_action(ctx)
        assert result == Contains("What do you mean?")
        assert result == Contains("option A")

    def test_feedback_positive_records_episode(self):
        d = FeedbackDecisionFactory.build(sentiment="positive", message="Great!")
        ctx = _make_ctx(d)
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem, \
             patch("arqitect.brain.dispatch._cb_success"), \
             patch("arqitect.brain.dispatch._cb_failure"):
            mock_mem.warm.recall.return_value = [{"task": "prev", "nerve": "joke_nerve", "tool": ""}]
            result = dispatch_action(ctx)
        assert result == "Great!"

    def test_feedback_negative_records_failure(self):
        """Negative feedback must record a failed episode and trigger circuit breaker."""
        d = FeedbackDecisionFactory.build(sentiment="negative", message="That was wrong")
        ctx = _make_ctx(d)
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
        d = UpdateContextDecisionFactory.build(
            context={"city": "Tel Aviv"},
            message="Noted!",
        )
        ctx = _make_ctx(d, user_id="user1")
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            result = dispatch_action(ctx)
        assert result == "Noted!"
        mock_mem.hot.update_session.assert_called_once()
        mock_mem.cold.set_user_fact.assert_called_once_with("user1", "city", "Tel Aviv", confidence=1.0)

    def test_update_context_location_normalized_to_city(self):
        """LLM says 'location' but our session key is 'city'."""
        d = UpdateContextDecisionFactory.build(
            context={"location": "Haifa"},
            message="Got it!",
        )
        ctx = _make_ctx(d, user_id="user1")
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            result = dispatch_action(ctx)
        mock_mem.cold.set_user_fact.assert_called_once_with("user1", "city", "Haifa", confidence=1.0)

    def test_unknown_action_re_thinks(self):
        d = InvokeDecisionFactory.build(action="fly_to_moon")
        ctx = _make_ctx(d)
        with patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.mem"):
            result = dispatch_action(ctx)
        ctx.think_fn.assert_called_once()

    def test_respond_redirects_to_awareness(self):
        d = RespondDecisionFactory.build(message="I am sentient")
        ctx = _make_ctx(d)
        with patch("arqitect.brain.dispatch.mem"):
            result = dispatch_action(ctx)
        ctx.think_fn.assert_called_once()
        call_args = ctx.think_fn.call_args
        assert "awareness" in str(call_args).lower()

    def test_chain_single_step_converts_to_invoke(self):
        """Single-step chain should be treated as a direct invoke."""
        catalog = {"joke_nerve": "tells jokes"}
        step = ChainStepFactory.build(nerve="joke_nerve", args="tell joke")
        d = ChainDecisionFactory.build(steps=[step], goal="tell a joke")
        ctx = _make_ctx(d, nerve_catalog=catalog, task="tell me a joke")
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
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        ctx = _make_ctx(d, nerve_catalog={"awareness": "identity"}, user_id="user123")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[]), \
             patch("arqitect.brain.dispatch.synthesize_nerve", return_value=("astronomy_nerve", "/nerves/astronomy_nerve/nerve.py")), \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            mock_mem.cold.get_user_role.return_value = "user"
            result = dispatch_action(ctx)
        ctx.think_fn.assert_called_once()

    def test_depth_not_incremented_on_redirect(self):
        """Synthesize->invoke redirect must NOT increase depth (it's not a re-think)."""
        catalog = {"joke_nerve": "tells jokes"}
        d = SynthesizeDecisionFactory.build(name="joke_nerve", description="tells jokes")
        ctx = _make_ctx(d, nerve_catalog=catalog, depth=4)
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
        catalog = {"joke_nerve": "tells jokes"}
        d = InvokeDecisionFactory.build(name="joke_nerve", args="tell me a joke")
        ctx = _make_ctx(d, nerve_catalog=catalog, task="tell me a joke")
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
        catalog = {"weather_nerve": "weather data"}
        d = InvokeDecisionFactory.build(name="weather_nerve", args="weather forecast")
        ctx = _make_ctx(d, nerve_catalog=catalog, task="weather forecast")
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
        catalog = {"joke_nerve": "tells jokes"}
        d = InvokeDecisionFactory.build(name="joke_nerve", args="translate this")
        ctx = _make_ctx(d, nerve_catalog=catalog, task="translate this")
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
        catalog = {"awareness": "identity"}
        d = InvokeDecisionFactory.build(name="missing_nerve", args="do something")
        ctx = _make_ctx(d, nerve_catalog=catalog, task="do something")
        with patch("arqitect.brain.dispatch.mem") as mock_mem:
            mock_mem.cold.list_nerves.return_value = ["awareness"]
            result = dispatch_action(ctx)
        ctx.think_fn.assert_called_once()
        call_args_str = str(ctx.think_fn.call_args)
        assert "does not exist" in call_args_str
        assert "Synthesize" in call_args_str

    def test_invoke_permission_denied(self):
        """Restricted nerve returns restriction message without invoking."""
        catalog = {"touch": "file ops"}
        d = InvokeDecisionFactory.build(name="touch", args="rm -rf /")
        ctx = _make_ctx(d, nerve_catalog=catalog, task="delete everything")
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
# 5. Synthesis permission enforcement — anon cannot fabricate nerves
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestSynthesisPermissionEnforcement:
    """Anon users can use existing nerves but cannot synthesize new ones.
    Registered users (role >= 'user') can synthesize freely."""

    def test_anon_synthesize_blocked(self):
        """Anon user attempting to synthesize a new nerve must be denied."""
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        ctx = _make_ctx(d, nerve_catalog={"awareness": "identity"}, user_id="")
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
        assert "email" in result.lower()

    def test_anon_synthesize_blocked_signals_request_identity(self):
        """Synthesis blocked for anon must publish with request_identity=True."""
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        ctx = _make_ctx(d, nerve_catalog={"awareness": "identity"}, user_id="")
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
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        ctx = _make_ctx(d, nerve_catalog={"awareness": "identity"}, user_id="user123")
        with patch("arqitect.brain.dispatch.match_nerves", return_value=[]), \
             patch("arqitect.brain.dispatch.synthesize_nerve", return_value=("astronomy_nerve", "/path")) as mock_synth, \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response") as mock_pub, \
             patch("arqitect.brain.dispatch.mem") as mock_mem:
            mock_mem.cold.get_user_role.return_value = "user"
            dispatch_action(ctx)
        for call in mock_pub.call_args_list:
            kwargs = call[1] if call[1] else {}
            assert not kwargs.get("request_identity")

    def test_registered_user_synthesize_allowed(self):
        """Registered user (role='user') can synthesize new nerves."""
        d = SynthesizeDecisionFactory.build(name="astronomy_nerve", description="answers space questions")
        ctx = _make_ctx(d, nerve_catalog={"awareness": "identity"}, user_id="user123")
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
        catalog = {"joke_nerve": "tells jokes"}
        d = InvokeDecisionFactory.build(name="joke_nerve", args="tell me a joke")
        ctx = _make_ctx(d, nerve_catalog=catalog, task="tell me a joke", user_id="")
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
        catalog = {"joke_nerve": "tells jokes"}
        steps = [
            ChainStepFactory.build(nerve="joke_nerve", args="tell joke"),
            ChainStepFactory.build(nerve="missing_nerve", args="do something"),
        ]
        d = ChainDecisionFactory.build(steps=steps, goal="joke then magic")
        ctx = _make_ctx(d, nerve_catalog=catalog, task="joke then magic", user_id="")
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
        catalog = {"joke_nerve": "tells jokes"}
        steps = [
            ChainStepFactory.build(nerve="joke_nerve", args="tell joke"),
            ChainStepFactory.build(nerve="missing_nerve", args="do something"),
        ]
        d = ChainDecisionFactory.build(steps=steps, goal="joke then magic")
        ctx = _make_ctx(d, nerve_catalog=catalog, task="joke then magic", user_id="user123")
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
