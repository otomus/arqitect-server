"""Tests for personality evolution -- signals, observation, evolution, anchors, history.

Covers:
- Signal collection and retrieval
- Observation with sufficient and insufficient signals
- Evolution with confidence thresholds
- Anchor validation (never list, bounds, allowed values)
- Trait clamping (drift limits, absolute bounds)
- History recording and retrieval
- Rollback to previous state
- Admin controls (set trait, add/remove anchor, pause/resume, reset)
- Evolution skipped when paused
- Cold memory personality tables (signals, history)
"""

import json
import time
from unittest.mock import patch

import pytest
from dirty_equals import IsInstance, IsPositive
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from arqitect.brain.personality import (
    MIN_SIGNALS_FOR_OBSERVATION,
    MIN_EVOLUTION_CONFIDENCE,
    MAX_DRIFT_PER_CYCLE,
    TRAIT_MIN,
    TRAIT_MAX,
    add_anchor,
    clamp_trait,
    evolve_personality,
    get_history,
    is_evolution_enabled,
    load_anchors,
    load_current_traits,
    observe_personality,
    pause_evolution,
    record_signal,
    remove_anchor,
    reset_to_seed,
    resume_evolution,
    rollback,
    set_anchor_bounds,
    set_trait,
    validate_against_anchors,
)


# -- Shared seed fixture -----------------------------------------------------

SEED = {
    "name": "TestBot",
    "trait_weights": {
        "wit": 0.5,
        "warmth": 0.7,
        "formality": 0.3,
        "verbosity": 0.3,
    },
    "voice": "warm, knowledgeable",
}


@pytest.fixture(autouse=True)
def _patch_personality_deps(mem):
    """Patch module-level mem for all personality tests."""
    with patch("arqitect.brain.consolidate.mem", mem), \
         patch("arqitect.brain.consolidate.publish_event"), \
         patch("arqitect.brain.consolidate.publish_nerve_status"):
        yield


# -- Helpers ------------------------------------------------------------------

def _make_signals(count: int) -> list[dict]:
    """Generate N dummy interaction signals."""
    return [
        {
            "timestamp": time.time() + i,
            "user_responded": True,
            "user_tone_shift": "neutral",
            "topic_domain": "task",
            "personality_traits_used": ["warmth"],
        }
        for i in range(count)
    ]


def _seed_signals(cold, count: int) -> None:
    """Insert N signals into cold memory."""
    for signal in _make_signals(count):
        record_signal(cold, signal)


def _make_observation_result() -> dict:
    """Return a mock observation report."""
    return {
        "trait_scores": {"wit": 0.6, "warmth": 0.9, "formality": 0.5},
        "insights": [
            "Users engage more when warmth is high",
            "Humor attempts in technical contexts get ignored",
        ],
        "explicit_feedback_summary": "No explicit feedback",
        "recommendation": "increase warmth, reduce humor in technical contexts",
    }


def _make_evolution_result(confidence: float = 0.8) -> dict:
    """Return a mock evolution proposal."""
    return {
        "changes": [
            {"trait": "warmth", "old": 0.7, "new": 0.8, "reason": "users engage more with warmer responses"},
            {"trait": "wit", "old": 0.5, "new": 0.4, "reason": "humor underperforms in technical contexts"},
        ],
        "unchanged": ["formality", "verbosity"],
        "confidence": confidence,
    }


# -- Signal Collection -------------------------------------------------------

class TestSignalCollection:
    """Interaction signals are stored and retrieved correctly."""

    @pytest.mark.timeout(10)
    def test_record_and_retrieve_signal(self, mem):
        """A single recorded signal is retrievable from cold memory."""
        signal = {"user_responded": True, "topic_domain": "casual"}
        record_signal(mem.cold, signal)

        signals = mem.cold.get_personality_signals()
        assert len(signals) == 1
        assert signals[0]["user_responded"] is True
        assert signals[0]["topic_domain"] == "casual"

    @pytest.mark.timeout(10)
    def test_signals_accumulate_in_order(self, mem):
        """Multiple signals accumulate in chronological order."""
        for i in range(5):
            record_signal(mem.cold, {"index": i})

        signals = mem.cold.get_personality_signals()
        assert len(signals) == 5
        assert [s["index"] for s in signals] == [0, 1, 2, 3, 4]

    @pytest.mark.timeout(10)
    def test_flush_clears_all_signals(self, mem):
        """Flushing removes all signals from cold memory."""
        _seed_signals(mem.cold, 5)
        assert len(mem.cold.get_personality_signals()) == 5

        mem.cold.flush_personality_signals()
        assert len(mem.cold.get_personality_signals()) == 0

    @pytest.mark.timeout(10)
    def test_flush_is_idempotent(self, mem):
        """Flushing an empty signal store does not error."""
        mem.cold.flush_personality_signals()
        assert len(mem.cold.get_personality_signals()) == 0

    @pytest.mark.timeout(10)
    @given(count=st.integers(min_value=1, max_value=25))
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_signal_count_matches_insertions(self, mem, count):
        """The number of retrievable signals equals the number recorded."""
        # Flush before each hypothesis example to avoid accumulation
        mem.cold.flush_personality_signals()

        for i in range(count):
            record_signal(mem.cold, {"index": i})

        signals = mem.cold.get_personality_signals()
        assert len(signals) == count


# -- Observation --------------------------------------------------------------

class TestObservation:
    """Personality observation analyzes signals and produces a report."""

    @pytest.mark.timeout(10)
    def test_skips_with_insufficient_signals(self, mem):
        """Returns None when too few signals have accumulated."""
        _seed_signals(mem.cold, MIN_SIGNALS_FOR_OBSERVATION - 1)

        result = observe_personality(mem.cold, lambda *a, **kw: "{}", SEED)
        assert result is None

    @pytest.mark.timeout(10)
    def test_runs_with_sufficient_signals(self, mem):
        """Returns an observation report when enough signals exist."""
        _seed_signals(mem.cold, MIN_SIGNALS_FOR_OBSERVATION)

        fake_report = _make_observation_result()

        def fake_generate(model, prompt, system="", max_tokens=512):
            assert model == "brain"
            assert "trait weights" in prompt.lower() or "trait_weights" in prompt
            return json.dumps(fake_report)

        result = observe_personality(mem.cold, fake_generate, SEED)
        assert result is not None
        assert "trait_scores" in result
        assert "recommendation" in result

    @pytest.mark.timeout(10)
    def test_returns_none_on_unparseable_llm_output(self, mem):
        """Returns None when LLM produces invalid JSON."""
        _seed_signals(mem.cold, MIN_SIGNALS_FOR_OBSERVATION)

        result = observe_personality(
            mem.cold, lambda *a, **kw: "not json at all", SEED,
        )
        assert result is None

    @pytest.mark.timeout(10)
    def test_handles_code_fenced_json(self, mem):
        """Parses JSON wrapped in markdown code fences."""
        _seed_signals(mem.cold, MIN_SIGNALS_FOR_OBSERVATION)

        report = _make_observation_result()
        fenced = f"```json\n{json.dumps(report)}\n```"

        result = observe_personality(
            mem.cold, lambda *a, **kw: fenced, SEED,
        )
        assert result is not None
        assert result["recommendation"] == report["recommendation"]

    @pytest.mark.timeout(10)
    @given(signal_count=st.integers(min_value=0, max_value=MIN_SIGNALS_FOR_OBSERVATION - 1))
    @settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_observation_always_none_below_threshold(self, mem, signal_count):
        """Observation returns None for any count below the minimum."""
        mem.cold.flush_personality_signals()
        _seed_signals(mem.cold, signal_count)

        result = observe_personality(mem.cold, lambda *a, **kw: "{}", SEED)
        assert result is None


# -- Evolution ----------------------------------------------------------------

class TestEvolution:
    """Personality evolution applies changes within constraints."""

    @pytest.mark.timeout(10)
    def test_applies_changes_above_confidence_threshold(self, mem):
        """Changes are applied when LLM confidence exceeds the minimum."""
        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=0.8)

        applied = evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        assert len(applied) == 2
        trait_names = {c["trait"] for c in applied}
        assert "warmth" in trait_names
        assert "wit" in trait_names

    @pytest.mark.timeout(10)
    def test_skips_below_confidence_threshold(self, mem):
        """No changes applied when confidence is below the minimum."""
        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=MIN_EVOLUTION_CONFIDENCE - 0.1)

        applied = evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        assert len(applied) == 0

    @pytest.mark.timeout(10)
    def test_records_history_on_successful_evolution(self, mem):
        """A history entry is created when traits change."""
        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=0.8)

        evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        history = mem.cold.get_personality_history()
        assert len(history) == 1
        assert history[0]["confidence"] == 0.8
        assert len(history[0]["changes"]) == 2

    @pytest.mark.timeout(10)
    def test_flushes_signals_after_evolution(self, mem):
        """Processed signals are deleted after successful evolution."""
        _seed_signals(mem.cold, 15)
        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=0.8)

        evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        assert len(mem.cold.get_personality_signals()) == 0

    @pytest.mark.timeout(10)
    def test_does_not_flush_signals_on_no_changes(self, mem):
        """Signals are preserved when evolution produces no changes."""
        _seed_signals(mem.cold, 15)
        observation = _make_observation_result()
        proposal = {"changes": [], "unchanged": ["warmth"], "confidence": 0.9}

        evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        assert len(mem.cold.get_personality_signals()) == 15

    @pytest.mark.timeout(10)
    def test_updates_trait_weights_in_cold_memory(self, mem):
        """Evolved traits are persisted to cold memory facts."""
        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=0.8)

        evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        raw = mem.cold.get_fact("personality", "trait_weights")
        traits = json.loads(raw)
        assert traits["warmth"] == 0.8
        assert traits["wit"] == 0.4

    @pytest.mark.timeout(10)
    def test_skipped_when_evolution_paused(self, mem):
        """No changes when admin has paused evolution."""
        pause_evolution(mem.cold)
        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=0.9)

        applied = evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        assert len(applied) == 0

    @pytest.mark.timeout(10)
    def test_handles_unparseable_llm_output(self, mem):
        """Returns empty list when LLM output is not valid JSON."""
        observation = _make_observation_result()

        applied = evolve_personality(
            mem.cold,
            lambda *a, **kw: "not json",
            observation, SEED,
        )

        assert applied == []

    @pytest.mark.timeout(10)
    @given(confidence=st.floats(min_value=0.0, max_value=MIN_EVOLUTION_CONFIDENCE - 0.01))
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_evolution_rejected_for_any_low_confidence(self, mem, confidence):
        """Evolution produces no changes for any confidence below threshold."""
        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=confidence)

        applied = evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        assert applied == []


# -- Anchor Validation --------------------------------------------------------

class TestAnchorValidation:
    """Anchors constrain what personality changes are allowed."""

    @pytest.mark.timeout(10)
    def test_never_list_blocks_matching_traits(self):
        """Traits matching the 'never' list are filtered out."""
        changes = [
            {"trait": "sarcastic_tone", "old": 0.3, "new": 0.5, "reason": "test"},
            {"trait": "warmth", "old": 0.7, "new": 0.8, "reason": "test"},
        ]
        anchors = {"never": ["sarcastic"], "always": [], "bounds": {}}

        validated = validate_against_anchors(changes, anchors)

        assert len(validated) == 1
        assert validated[0]["trait"] == "warmth"

    @pytest.mark.timeout(10)
    def test_bounds_min_blocks_below_minimum(self):
        """Values below the min bound are blocked."""
        changes = [
            {"trait": "formality", "old": 0.5, "new": 0.3, "reason": "test"},
        ]
        anchors = {"never": [], "always": [], "bounds": {"formality": {"min": 0.4}}}

        validated = validate_against_anchors(changes, anchors)
        assert len(validated) == 0

    @pytest.mark.timeout(10)
    def test_bounds_max_blocks_above_maximum(self):
        """Values above the max bound are blocked."""
        changes = [
            {"trait": "formality", "old": 0.7, "new": 0.95, "reason": "test"},
        ]
        anchors = {"never": [], "always": [], "bounds": {"formality": {"max": 0.9}}}

        validated = validate_against_anchors(changes, anchors)
        assert len(validated) == 0

    @pytest.mark.timeout(10)
    def test_bounds_allowed_blocks_disallowed_values(self):
        """Values not in the allowed list are blocked."""
        changes = [
            {"trait": "humor_frequency", "old": "occasional", "new": "always", "reason": "test"},
        ]
        anchors = {
            "never": [], "always": [],
            "bounds": {"humor_frequency": {"allowed": ["never", "rare", "occasional"]}},
        }

        validated = validate_against_anchors(changes, anchors)
        assert len(validated) == 0

    @pytest.mark.timeout(10)
    def test_allowed_values_pass_through(self):
        """Values within the allowed list pass validation."""
        changes = [
            {"trait": "humor_frequency", "old": "rare", "new": "occasional", "reason": "test"},
        ]
        anchors = {
            "never": [], "always": [],
            "bounds": {"humor_frequency": {"allowed": ["never", "rare", "occasional"]}},
        }

        validated = validate_against_anchors(changes, anchors)
        assert len(validated) == 1

    @pytest.mark.timeout(10)
    def test_within_bounds_passes(self):
        """Values within min/max bounds pass validation."""
        changes = [
            {"trait": "formality", "old": 0.5, "new": 0.6, "reason": "test"},
        ]
        anchors = {"never": [], "always": [], "bounds": {"formality": {"min": 0.4, "max": 0.9}}}

        validated = validate_against_anchors(changes, anchors)
        assert len(validated) == 1

    @pytest.mark.timeout(10)
    def test_empty_anchors_passes_all(self):
        """No anchors means all changes pass."""
        changes = [
            {"trait": "warmth", "old": 0.5, "new": 0.9, "reason": "test"},
            {"trait": "wit", "old": 0.5, "new": 0.1, "reason": "test"},
        ]
        anchors = {"never": [], "always": [], "bounds": {}}

        validated = validate_against_anchors(changes, anchors)
        assert len(validated) == 2

    @pytest.mark.timeout(10)
    def test_never_list_case_insensitive(self):
        """Never list matching is case-insensitive."""
        changes = [
            {"trait": "Sarcastic_Humor", "old": 0.3, "new": 0.5, "reason": "test"},
        ]
        anchors = {"never": ["sarcastic"], "always": [], "bounds": {}}

        validated = validate_against_anchors(changes, anchors)
        assert len(validated) == 0

    @pytest.mark.timeout(10)
    @given(
        new_val=st.floats(min_value=0.0, max_value=0.39, allow_nan=False),
    )
    @settings(max_examples=10, deadline=None)
    def test_bounds_min_blocks_any_value_below(self, new_val):
        """Any value below the min bound is rejected."""
        changes = [{"trait": "formality", "old": 0.5, "new": new_val, "reason": "test"}]
        anchors = {"never": [], "always": [], "bounds": {"formality": {"min": 0.4}}}

        validated = validate_against_anchors(changes, anchors)
        assert len(validated) == 0

    @pytest.mark.timeout(10)
    @given(
        new_val=st.floats(min_value=0.91, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=10, deadline=None)
    def test_bounds_max_blocks_any_value_above(self, new_val):
        """Any value above the max bound is rejected."""
        changes = [{"trait": "formality", "old": 0.5, "new": new_val, "reason": "test"}]
        anchors = {"never": [], "always": [], "bounds": {"formality": {"max": 0.9}}}

        validated = validate_against_anchors(changes, anchors)
        assert len(validated) == 0


# -- Trait Clamping -----------------------------------------------------------

class TestTraitClamping:
    """Trait values are clamped within drift and absolute bounds."""

    @pytest.mark.timeout(10)
    def test_clamps_within_max_drift(self):
        """Change is limited to MAX_DRIFT_PER_CYCLE from old value."""
        result = clamp_trait(0.9, 0.5)
        assert result == 0.5 + MAX_DRIFT_PER_CYCLE

    @pytest.mark.timeout(10)
    def test_clamps_negative_drift(self):
        """Negative drift is also limited."""
        result = clamp_trait(0.1, 0.5)
        assert result == 0.5 - MAX_DRIFT_PER_CYCLE

    @pytest.mark.timeout(10)
    def test_respects_absolute_minimum(self):
        """Value cannot go below TRAIT_MIN."""
        result = clamp_trait(0.0, TRAIT_MIN)
        assert result == TRAIT_MIN

    @pytest.mark.timeout(10)
    def test_respects_absolute_maximum(self):
        """Value cannot exceed TRAIT_MAX."""
        result = clamp_trait(1.0, TRAIT_MAX)
        assert result == TRAIT_MAX

    @pytest.mark.timeout(10)
    def test_no_change_within_drift(self):
        """Small changes within drift pass through."""
        result = clamp_trait(0.55, 0.5)
        assert result == 0.55

    @pytest.mark.timeout(10)
    def test_rounds_to_two_decimals(self):
        """Result is rounded to 2 decimal places."""
        result = clamp_trait(0.5333, 0.5)
        assert result == round(0.5333, 2)

    @pytest.mark.timeout(10)
    @given(
        value=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        old_value=st.floats(min_value=TRAIT_MIN, max_value=TRAIT_MAX, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_clamp_always_within_absolute_bounds(self, value, old_value):
        """Clamped result is always between TRAIT_MIN and TRAIT_MAX."""
        result = clamp_trait(value, old_value)
        assert TRAIT_MIN <= result <= TRAIT_MAX

    @pytest.mark.timeout(10)
    @given(
        value=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        old_value=st.floats(min_value=TRAIT_MIN, max_value=TRAIT_MAX, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_clamp_drift_never_exceeds_max(self, value, old_value):
        """The drift from old_value never exceeds MAX_DRIFT_PER_CYCLE."""
        result = clamp_trait(value, old_value)
        # Account for float rounding: result is rounded to 2 decimals
        assert abs(result - old_value) <= MAX_DRIFT_PER_CYCLE + 0.005

    @pytest.mark.timeout(10)
    @given(
        value=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        old_value=st.floats(min_value=TRAIT_MIN, max_value=TRAIT_MAX, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_clamp_result_always_two_decimals(self, value, old_value):
        """Clamped result is always rounded to at most 2 decimal places."""
        result = clamp_trait(value, old_value)
        assert result == round(result, 2)


# -- History & Rollback -------------------------------------------------------

class TestHistory:
    """Personality history tracks all evolution events."""

    @pytest.mark.timeout(10)
    def test_empty_history_initially(self, mem):
        """No history entries exist before any evolution."""
        assert get_history(mem.cold) == []

    @pytest.mark.timeout(10)
    def test_history_records_after_evolution(self, mem):
        """History entries are created by successful evolution."""
        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=0.8)

        evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        history = get_history(mem.cold)
        assert len(history) == 1
        assert history[0]["observation_summary"] == observation["recommendation"]

    @pytest.mark.timeout(10)
    def test_history_limit(self, mem):
        """History retrieval respects the limit parameter."""
        for i in range(5):
            mem.cold.append_personality_history({
                "timestamp": time.time() + i,
                "old_traits": {"warmth": 0.5 + i * 0.02},
                "new_traits": {"warmth": 0.52 + i * 0.02},
                "changes": [{"trait": "warmth", "old": 0.5 + i * 0.02, "new": 0.52 + i * 0.02}],
                "observation_summary": f"change {i}",
                "confidence": 0.8,
            })

        limited = get_history(mem.cold, limit=3)
        assert len(limited) == 3

    @pytest.mark.timeout(10)
    def test_rollback_reverts_traits(self, mem):
        """Rollback restores the previous trait state."""
        mem.cold.set_fact("personality", "trait_weights", json.dumps({"warmth": 0.8}))
        mem.cold.append_personality_history({
            "timestamp": time.time(),
            "old_traits": {"warmth": 0.7},
            "new_traits": {"warmth": 0.8},
            "changes": [{"trait": "warmth", "old": 0.7, "new": 0.8}],
            "observation_summary": "test",
            "confidence": 0.8,
        })

        result = rollback(mem.cold, SEED, steps=1)

        assert result["new_traits"]["warmth"] == 0.7
        current = json.loads(mem.cold.get_fact("personality", "trait_weights"))
        assert current["warmth"] == 0.7

    @pytest.mark.timeout(10)
    def test_rollback_records_history_entry(self, mem):
        """Rollback itself is recorded in history."""
        mem.cold.append_personality_history({
            "timestamp": time.time(),
            "old_traits": {"warmth": 0.7},
            "new_traits": {"warmth": 0.8},
            "changes": [],
            "observation_summary": "test",
            "confidence": 0.8,
        })

        rollback(mem.cold, SEED, steps=1)

        history = get_history(mem.cold)
        assert len(history) == 2
        last = history[-1]
        assert last["changes"][0]["type"] == "rollback"

    @pytest.mark.timeout(10)
    def test_rollback_fails_with_insufficient_history(self, mem):
        """Rollback raises ValueError when there aren't enough entries."""
        with pytest.raises(ValueError, match="Cannot rollback"):
            rollback(mem.cold, SEED, steps=1)

    @pytest.mark.timeout(10)
    def test_rollback_result_has_expected_shape(self, mem):
        """Rollback returns a dict with old_traits and new_traits keys."""
        mem.cold.set_fact("personality", "trait_weights", json.dumps({"warmth": 0.8}))
        mem.cold.append_personality_history({
            "timestamp": time.time(),
            "old_traits": {"warmth": 0.7},
            "new_traits": {"warmth": 0.8},
            "changes": [],
            "observation_summary": "test",
            "confidence": 0.8,
        })

        result = rollback(mem.cold, SEED, steps=1)

        assert result == {
            "old_traits": IsInstance(dict),
            "new_traits": IsInstance(dict),
        }


# -- Admin Controls -----------------------------------------------------------

class TestAdminControls:
    """Admin personality management functions."""

    @pytest.mark.timeout(10)
    def test_set_trait_immediate(self, mem):
        """Admin can set a trait value immediately."""
        set_trait(mem.cold, "warmth", 0.9, SEED)

        traits = load_current_traits(mem.cold, SEED)
        assert traits["warmth"] == 0.9

    @pytest.mark.timeout(10)
    def test_set_trait_records_history(self, mem):
        """Trait override is recorded in history."""
        set_trait(mem.cold, "warmth", 0.9, SEED)

        history = get_history(mem.cold)
        assert len(history) == 1
        assert history[0]["changes"][0]["reason"] == "admin override"

    @pytest.mark.timeout(10)
    def test_add_anchor_never(self, mem):
        """Adding to the never list persists in cold memory."""
        add_anchor(mem.cold, "never", "sarcastic")

        anchors = load_anchors(mem.cold)
        assert "sarcastic" in anchors["never"]

    @pytest.mark.timeout(10)
    def test_add_anchor_idempotent(self, mem):
        """Adding the same anchor twice does not duplicate."""
        add_anchor(mem.cold, "never", "sarcastic")
        add_anchor(mem.cold, "never", "sarcastic")

        anchors = load_anchors(mem.cold)
        assert anchors["never"].count("sarcastic") == 1

    @pytest.mark.timeout(10)
    def test_remove_anchor(self, mem):
        """Removing an anchor removes it from the list."""
        add_anchor(mem.cold, "never", "sarcastic")
        remove_anchor(mem.cold, "never", "sarcastic")

        anchors = load_anchors(mem.cold)
        assert "sarcastic" not in anchors["never"]

    @pytest.mark.timeout(10)
    def test_set_anchor_bounds(self, mem):
        """Admin can set min/max bounds for a trait."""
        set_anchor_bounds(mem.cold, "formality", {"min": 0.4, "max": 0.9})

        anchors = load_anchors(mem.cold)
        assert anchors["bounds"]["formality"]["min"] == 0.4
        assert anchors["bounds"]["formality"]["max"] == 0.9

    @pytest.mark.timeout(10)
    def test_pause_and_resume(self, mem):
        """Evolution can be paused and resumed."""
        assert is_evolution_enabled(mem.cold) is True

        pause_evolution(mem.cold)
        assert is_evolution_enabled(mem.cold) is False

        resume_evolution(mem.cold)
        assert is_evolution_enabled(mem.cold) is True

    @pytest.mark.timeout(10)
    def test_reset_to_seed(self, mem):
        """Reset restores the seed trait weights."""
        mem.cold.set_fact("personality", "trait_weights", json.dumps({"warmth": 0.9, "wit": 0.1}))

        reset_to_seed(mem.cold, SEED)

        traits = load_current_traits(mem.cold, SEED)
        assert traits == SEED["trait_weights"]

    @pytest.mark.timeout(10)
    def test_reset_records_history(self, mem):
        """Reset is recorded in history."""
        reset_to_seed(mem.cold, SEED)

        history = get_history(mem.cold)
        assert len(history) == 1
        assert history[0]["changes"][0]["type"] == "reset"

    @pytest.mark.timeout(10)
    def test_set_trait_history_entry_has_confidence(self, mem):
        """Admin override history entries always have confidence 1.0."""
        set_trait(mem.cold, "warmth", 0.9, SEED)

        history = get_history(mem.cold)
        assert history[0]["confidence"] == 1.0


# -- Trait Loading ------------------------------------------------------------

class TestTraitLoading:
    """Current traits are loaded from cold memory with seed fallback."""

    @pytest.mark.timeout(10)
    def test_falls_back_to_seed(self, mem):
        """Returns seed weights when cold memory has no personality facts."""
        traits = load_current_traits(mem.cold, SEED)
        assert traits == SEED["trait_weights"]

    @pytest.mark.timeout(10)
    def test_loads_from_cold_memory(self, mem):
        """Returns cold memory weights when they exist."""
        stored = {"warmth": 0.9, "wit": 0.3}
        mem.cold.set_fact("personality", "trait_weights", json.dumps(stored))

        traits = load_current_traits(mem.cold, SEED)
        assert traits == stored

    @pytest.mark.timeout(10)
    def test_falls_back_on_corrupt_json(self, mem):
        """Returns seed weights when cold memory has corrupt JSON."""
        mem.cold.set_fact("personality", "trait_weights", "not json")

        traits = load_current_traits(mem.cold, SEED)
        assert traits == SEED["trait_weights"]

    @pytest.mark.timeout(10)
    def test_supports_traits_key_in_seed(self, mem):
        """Seed can use 'traits' key instead of 'trait_weights'."""
        alt_seed = {"traits": {"warmth": 0.6, "wit": 0.4}}
        traits = load_current_traits(mem.cold, alt_seed)
        assert traits == alt_seed["traits"]


# -- Anchor Loading -----------------------------------------------------------

class TestAnchorLoading:
    """Anchors are loaded from cold memory with safe defaults."""

    @pytest.mark.timeout(10)
    def test_default_anchors(self, mem):
        """Returns empty anchors when none are configured."""
        anchors = load_anchors(mem.cold)
        assert anchors == {"never": [], "always": [], "bounds": {}}

    @pytest.mark.timeout(10)
    def test_loads_from_cold_memory(self, mem):
        """Returns stored anchors from cold memory."""
        stored = {"never": ["rude"], "always": ["polite"], "bounds": {}}
        mem.cold.set_fact("personality", "anchors", json.dumps(stored))

        anchors = load_anchors(mem.cold)
        assert anchors == stored


# -- Cold Memory Tables -------------------------------------------------------

class TestColdMemoryPersonality:
    """Personality-specific cold memory operations."""

    @pytest.mark.timeout(10)
    def test_signal_round_trip(self, mem):
        """Signal data survives JSON serialization through cold memory."""
        original = {
            "user_responded": True,
            "user_tone_shift": "warmer",
            "explicit_feedback": "be more casual",
            "nested": {"key": [1, 2, 3]},
        }
        mem.cold.append_personality_signal(original)

        signals = mem.cold.get_personality_signals()
        assert len(signals) == 1
        assert signals[0] == original

    @pytest.mark.timeout(10)
    def test_history_round_trip(self, mem):
        """History entries survive JSON serialization through cold memory."""
        original = {
            "timestamp": time.time(),
            "old_traits": {"warmth": 0.7},
            "new_traits": {"warmth": 0.8},
            "changes": [{"trait": "warmth", "old": 0.7, "new": 0.8}],
            "observation_summary": "test observation",
            "confidence": 0.85,
        }
        mem.cold.append_personality_history(original)

        history = mem.cold.get_personality_history()
        assert len(history) == 1
        entry = history[0]
        assert entry["old_traits"] == original["old_traits"]
        assert entry["new_traits"] == original["new_traits"]
        assert entry["changes"] == original["changes"]
        assert entry["confidence"] == original["confidence"]

    @pytest.mark.timeout(10)
    def test_history_ordering(self, mem):
        """History entries are returned in chronological order."""
        for i in range(3):
            mem.cold.append_personality_history({
                "timestamp": 1000.0 + i,
                "old_traits": {}, "new_traits": {},
                "changes": [], "observation_summary": f"entry {i}",
                "confidence": 0.5,
            })

        history = mem.cold.get_personality_history()
        summaries = [h["observation_summary"] for h in history]
        assert summaries == ["entry 0", "entry 1", "entry 2"]


# -- Integration: Dream Phase ------------------------------------------------

class TestDreamPhaseIntegration:
    """End-to-end personality dream phase with observation + evolution."""

    @pytest.mark.timeout(10)
    def test_full_cycle_observe_then_evolve(self, mem):
        """Full dream cycle: seed signals, observe, evolve, verify."""
        _seed_signals(mem.cold, 15)

        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=0.8)

        call_count = {"observe": 0, "evolve": 0}

        def fake_generate(model, prompt, system="", max_tokens=512):
            if "analyze" in system.lower() or "analyst" in system.lower():
                call_count["observe"] += 1
                return json.dumps(observation)
            if "tuner" in system.lower():
                call_count["evolve"] += 1
                return json.dumps(proposal)
            return "{}"

        obs_result = observe_personality(mem.cold, fake_generate, SEED)
        assert obs_result is not None

        applied = evolve_personality(mem.cold, fake_generate, obs_result, SEED)
        assert len(applied) == 2

        assert call_count["observe"] == 1
        assert call_count["evolve"] == 1

        traits = load_current_traits(mem.cold, SEED)
        assert traits["warmth"] == 0.8
        assert traits["wit"] == 0.4

        history = get_history(mem.cold)
        assert len(history) == 1

        assert len(mem.cold.get_personality_signals()) == 0

    @pytest.mark.timeout(10)
    def test_evolution_blocked_by_anchors_records_no_history(self, mem):
        """When all changes are blocked by anchors, no history is recorded."""
        _seed_signals(mem.cold, 15)
        add_anchor(mem.cold, "never", "warmth")
        add_anchor(mem.cold, "never", "wit")

        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=0.9)

        applied = evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        assert len(applied) == 0
        assert len(get_history(mem.cold)) == 0

    @pytest.mark.timeout(10)
    def test_partial_anchor_blocking(self, mem):
        """Only the blocked changes are filtered; valid ones still apply."""
        _seed_signals(mem.cold, 15)
        set_anchor_bounds(mem.cold, "warmth", {"max": 0.7})

        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=0.8)

        applied = evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        applied_traits = {c["trait"] for c in applied}
        assert "warmth" not in applied_traits
        assert "wit" in applied_traits

    @pytest.mark.timeout(10)
    def test_history_entry_shape_after_evolution(self, mem):
        """History entries from evolution have the expected structure."""
        observation = _make_observation_result()
        proposal = _make_evolution_result(confidence=0.8)

        evolve_personality(
            mem.cold,
            lambda *a, **kw: json.dumps(proposal),
            observation, SEED,
        )

        history = get_history(mem.cold)
        assert len(history) == 1
        entry = history[0]
        assert entry["timestamp"] == IsInstance(float) & IsPositive
        assert entry["old_traits"] == IsInstance(dict)
        assert entry["new_traits"] == IsInstance(dict)
        assert entry["changes"] == IsInstance(list)
        assert entry["confidence"] == IsInstance(float) & IsPositive
        assert entry["observation_summary"] == IsInstance(str)
