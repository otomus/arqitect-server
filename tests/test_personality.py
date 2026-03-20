"""Tests for personality.py — pure function logic not covered by test_personality_evolution.py.

Focuses on:
- _parse_llm_json edge cases
- clamp_trait boundary arithmetic
- _is_blocked_by_never_list matching
- _is_blocked_by_bounds with all bound types
- _apply_validated_changes detail behavior
- load_current_traits with various seed formats
- load_anchors with corrupt data
- DEFAULT_SEED_WEIGHTS fallback
- record_signal delegation
- format_profile_for_prompt edge handling

test_personality_evolution.py already covers the full observe/evolve/rollback/admin pipeline.
This file targets the lower-level helpers and edge cases.
"""

import json
from unittest.mock import patch, MagicMock, call

import pytest
from dirty_equals import IsInstance
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from arqitect.brain.personality import (
    DEFAULT_SEED_WEIGHTS,
    MAX_DRIFT_PER_CYCLE,
    MIN_EVOLUTION_CONFIDENCE,
    TRAIT_MAX,
    TRAIT_MIN,
    _apply_validated_changes,
    _is_blocked_by_bounds,
    _is_blocked_by_never_list,
    _parse_llm_json,
    clamp_trait,
    load_anchors,
    load_current_traits,
    record_signal,
    validate_against_anchors,
)


# ── _parse_llm_json ──────────────────────────────────────────────────────────

class TestParseLLMJson:
    """Parse JSON from LLM output with various formats."""

    @pytest.mark.timeout(10)
    def test_parses_plain_json(self):
        """Parses a plain JSON object."""
        result = _parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    @pytest.mark.timeout(10)
    def test_parses_code_fenced_json(self):
        """Parses JSON wrapped in markdown code fences."""
        result = _parse_llm_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    @pytest.mark.timeout(10)
    def test_parses_code_fenced_without_language(self):
        """Parses JSON wrapped in code fences without language hint."""
        result = _parse_llm_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    @pytest.mark.timeout(10)
    def test_returns_none_for_empty_string(self):
        """Returns None for empty input."""
        assert _parse_llm_json("") is None

    @pytest.mark.timeout(10)
    def test_returns_none_for_none(self):
        """Returns None for None input."""
        assert _parse_llm_json(None) is None

    @pytest.mark.timeout(10)
    def test_returns_none_for_invalid_json(self):
        """Returns None for text that is not valid JSON."""
        assert _parse_llm_json("This is not JSON") is None

    @pytest.mark.timeout(10)
    def test_returns_none_for_partial_json(self):
        """Returns None for truncated JSON."""
        assert _parse_llm_json('{"key": "val') is None

    @pytest.mark.timeout(10)
    def test_handles_whitespace_around_json(self):
        """Handles whitespace around JSON content."""
        result = _parse_llm_json('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    @pytest.mark.timeout(10)
    def test_handles_code_fence_with_only_two_lines(self):
        """Handles malformed code fence with only two lines."""
        result = _parse_llm_json('```\n```')
        assert result is None

    @pytest.mark.timeout(10)
    def test_parses_nested_json(self):
        """Parses nested JSON structures."""
        nested = {"outer": {"inner": [1, 2, 3]}, "flag": True}
        result = _parse_llm_json(json.dumps(nested))
        assert result == nested

    @pytest.mark.timeout(10)
    @given(data=st.fixed_dictionaries({
        "key": st.text(min_size=1, max_size=50),
        "value": st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=50)),
    }))
    @settings(max_examples=30)
    def test_roundtrips_arbitrary_json(self, data):
        """Any valid JSON dict survives a parse roundtrip."""
        raw = json.dumps(data)
        result = _parse_llm_json(raw)
        assert result == data

    @pytest.mark.timeout(10)
    @given(data=st.fixed_dictionaries({
        "key": st.text(min_size=1, max_size=50),
    }))
    @settings(max_examples=20)
    def test_roundtrips_code_fenced_json(self, data):
        """Any valid JSON dict roundtrips through code fences."""
        raw = f"```json\n{json.dumps(data)}\n```"
        result = _parse_llm_json(raw)
        assert result == data


# ── _is_blocked_by_never_list ────────────────────────────────────────────────

class TestIsBlockedByNeverList:
    """Never-list matching for trait names."""

    @pytest.mark.timeout(10)
    def test_exact_match(self):
        """Exact match blocks the trait."""
        assert _is_blocked_by_never_list("sarcastic", ["sarcastic"]) is True

    @pytest.mark.timeout(10)
    def test_substring_match(self):
        """Substring match blocks the trait."""
        assert _is_blocked_by_never_list("sarcastic_humor", ["sarcastic"]) is True

    @pytest.mark.timeout(10)
    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        assert _is_blocked_by_never_list("SARCASTIC", ["sarcastic"]) is True

    @pytest.mark.timeout(10)
    def test_no_match(self):
        """Non-matching trait passes."""
        assert _is_blocked_by_never_list("warmth", ["sarcastic"]) is False

    @pytest.mark.timeout(10)
    def test_empty_never_list(self):
        """Empty never list blocks nothing."""
        assert _is_blocked_by_never_list("anything", []) is False

    @pytest.mark.timeout(10)
    def test_empty_trait(self):
        """Empty trait string is not blocked by non-empty list."""
        assert _is_blocked_by_never_list("", ["sarcastic"]) is False

    @pytest.mark.timeout(10)
    def test_multiple_never_entries(self):
        """Blocked if any entry matches."""
        assert _is_blocked_by_never_list("rude_tone", ["sarcastic", "rude"]) is True

    @pytest.mark.timeout(10)
    def test_never_entry_is_substring_of_trait(self):
        """Never entry 'wit' blocks 'twitter_style'."""
        assert _is_blocked_by_never_list("twitter_style", ["wit"]) is True

    @pytest.mark.timeout(10)
    @given(
        trait=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L",))),
    )
    @settings(max_examples=30)
    def test_trait_always_blocked_by_itself(self, trait):
        """Any non-empty trait is always blocked by its own name in the never list."""
        assert _is_blocked_by_never_list(trait, [trait]) is True


# ── _is_blocked_by_bounds ────────────────────────────────────────────────────

class TestIsBlockedByBounds:
    """Bounds checking for trait values."""

    @pytest.mark.timeout(10)
    def test_no_bounds_for_trait(self):
        """Trait without bounds is not blocked."""
        assert _is_blocked_by_bounds("warmth", 0.9, {}) is False

    @pytest.mark.timeout(10)
    def test_below_min_blocked(self):
        """Value below min is blocked."""
        assert _is_blocked_by_bounds("warmth", 0.2, {"warmth": {"min": 0.3}}) is True

    @pytest.mark.timeout(10)
    def test_above_max_blocked(self):
        """Value above max is blocked."""
        assert _is_blocked_by_bounds("warmth", 0.95, {"warmth": {"max": 0.9}}) is True

    @pytest.mark.timeout(10)
    def test_within_bounds_passes(self):
        """Value within min/max passes."""
        assert _is_blocked_by_bounds("warmth", 0.5, {"warmth": {"min": 0.3, "max": 0.9}}) is False

    @pytest.mark.timeout(10)
    def test_not_in_allowed_list_blocked(self):
        """Value not in allowed list is blocked."""
        assert _is_blocked_by_bounds("style", "aggressive", {"style": {"allowed": ["calm", "neutral"]}}) is True

    @pytest.mark.timeout(10)
    def test_in_allowed_list_passes(self):
        """Value in allowed list passes."""
        assert _is_blocked_by_bounds("style", "calm", {"style": {"allowed": ["calm", "neutral"]}}) is False

    @pytest.mark.timeout(10)
    def test_non_numeric_value_with_min(self):
        """Non-numeric value is not blocked by min/max (no comparison)."""
        assert _is_blocked_by_bounds("style", "hello", {"style": {"min": 0.3}}) is False

    @pytest.mark.timeout(10)
    def test_exact_min_value_passes(self):
        """Value exactly at min is not blocked."""
        assert _is_blocked_by_bounds("warmth", 0.3, {"warmth": {"min": 0.3}}) is False

    @pytest.mark.timeout(10)
    def test_exact_max_value_passes(self):
        """Value exactly at max is not blocked."""
        assert _is_blocked_by_bounds("warmth", 0.9, {"warmth": {"max": 0.9}}) is False

    @pytest.mark.timeout(10)
    @given(data=st.data())
    @settings(max_examples=50)
    def test_within_bounds_never_blocked(self, data):
        """A value within [min, max] is never blocked."""
        bound_min = data.draw(st.floats(min_value=0.0, max_value=0.9, allow_nan=False))
        bound_max = data.draw(st.floats(min_value=bound_min, max_value=1.0, allow_nan=False))
        value = data.draw(st.floats(min_value=bound_min, max_value=bound_max, allow_nan=False))
        bounds = {"t": {"min": bound_min, "max": bound_max}}
        assert _is_blocked_by_bounds("t", value, bounds) is False


# ── clamp_trait edge cases ───────────────────────────────────────────────────

class TestClampTraitEdges:
    """Additional edge cases for trait clamping."""

    @pytest.mark.timeout(10)
    def test_identical_values(self):
        """No change when proposed equals current."""
        assert clamp_trait(0.5, 0.5) == 0.5

    @pytest.mark.timeout(10)
    def test_clamp_at_absolute_minimum(self):
        """Cannot go below TRAIT_MIN even with drift allowance."""
        result = clamp_trait(0.0, 0.15)
        assert result >= TRAIT_MIN

    @pytest.mark.timeout(10)
    def test_clamp_at_absolute_maximum(self):
        """Cannot go above TRAIT_MAX even with drift allowance."""
        result = clamp_trait(1.0, 0.85)
        assert result <= TRAIT_MAX

    @pytest.mark.timeout(10)
    def test_large_positive_jump_clamped(self):
        """Large positive jump is clamped to max drift."""
        result = clamp_trait(0.9, 0.3)
        assert result == round(0.3 + MAX_DRIFT_PER_CYCLE, 2)

    @pytest.mark.timeout(10)
    def test_large_negative_jump_clamped(self):
        """Large negative jump is clamped to max drift."""
        result = clamp_trait(0.1, 0.8)
        assert result == round(0.8 - MAX_DRIFT_PER_CYCLE, 2)

    @pytest.mark.timeout(10)
    def test_result_always_rounded(self):
        """Result is always rounded to 2 decimal places."""
        result = clamp_trait(0.333333, 0.3)
        assert result == round(result, 2)

    @pytest.mark.timeout(10)
    def test_negative_proposed_value(self):
        """Negative proposed value is clamped to TRAIT_MIN."""
        result = clamp_trait(-1.0, TRAIT_MIN)
        assert result == TRAIT_MIN

    @pytest.mark.timeout(10)
    def test_value_above_one(self):
        """Value above 1.0 is clamped to TRAIT_MAX."""
        result = clamp_trait(5.0, TRAIT_MAX)
        assert result == TRAIT_MAX

    @pytest.mark.timeout(10)
    @given(
        value=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        old_value=st.floats(min_value=TRAIT_MIN, max_value=TRAIT_MAX, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_result_always_within_absolute_bounds(self, value, old_value):
        """Clamped result is always within [TRAIT_MIN, TRAIT_MAX]."""
        result = clamp_trait(value, old_value)
        assert TRAIT_MIN <= result <= TRAIT_MAX

    @pytest.mark.timeout(10)
    @given(
        value=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        old_value=st.floats(min_value=TRAIT_MIN, max_value=TRAIT_MAX, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_drift_never_exceeds_max(self, value, old_value):
        """Absolute drift from old_value never exceeds MAX_DRIFT_PER_CYCLE."""
        result = clamp_trait(value, old_value)
        # Allow small floating-point tolerance
        assert abs(result - old_value) <= MAX_DRIFT_PER_CYCLE + 1e-9

    @pytest.mark.timeout(10)
    @given(
        value=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        old_value=st.floats(min_value=TRAIT_MIN, max_value=TRAIT_MAX, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_result_always_rounded_to_two_decimals(self, value, old_value):
        """Result is always rounded to exactly 2 decimal places."""
        result = clamp_trait(value, old_value)
        assert result == round(result, 2)


# ── _apply_validated_changes ─────────────────────────────────────────────────

class TestApplyValidatedChanges:
    """Detailed behavior of applying validated trait changes."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, mem):
        """Patch module-level dependencies for personality tests."""
        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.consolidate.publish_event"), \
             patch("arqitect.brain.consolidate.publish_nerve_status"):
            yield

    @pytest.mark.timeout(10)
    def test_skips_changes_below_threshold(self, mem):
        """Changes smaller than 0.01 are skipped."""
        current = {"warmth": 0.500}
        seed = {"trait_weights": {"warmth": 0.5}}
        # 0.504 clamps to 0.5 (rounded), so abs(0.5 - 0.5) < 0.01
        changes = [{"trait": "warmth", "new": 0.504, "reason": "tiny change"}]

        result = _apply_validated_changes(
            mem.cold, current, changes, seed, {}, 0.8,
        )
        assert result == []

    @pytest.mark.timeout(10)
    def test_skips_changes_without_trait_name(self, mem):
        """Changes missing trait name are skipped."""
        current = {"warmth": 0.5}
        seed = {"trait_weights": {"warmth": 0.5}}
        changes = [{"trait": "", "new": 0.6, "reason": "no trait"}]

        result = _apply_validated_changes(
            mem.cold, current, changes, seed, {}, 0.8,
        )
        assert result == []

    @pytest.mark.timeout(10)
    def test_skips_changes_with_non_numeric_new_value(self, mem):
        """Changes with non-numeric new value are skipped."""
        current = {"warmth": 0.5}
        seed = {"trait_weights": {"warmth": 0.5}}
        changes = [{"trait": "warmth", "new": "high", "reason": "text value"}]

        result = _apply_validated_changes(
            mem.cold, current, changes, seed, {}, 0.8,
        )
        assert result == []

    @pytest.mark.timeout(10)
    def test_persists_changes_to_cold_memory(self, mem):
        """Applied changes are persisted to cold memory."""
        current = {"warmth": 0.5}
        seed = {"trait_weights": {"warmth": 0.5}}
        changes = [{"trait": "warmth", "new": 0.6, "reason": "test"}]

        _apply_validated_changes(
            mem.cold, current, changes, seed, {"recommendation": "more warmth"}, 0.8,
        )

        stored = json.loads(mem.cold.get_fact("personality", "trait_weights"))
        assert stored["warmth"] == 0.6

    @pytest.mark.timeout(10)
    def test_records_history_entry(self, mem):
        """Applied changes create a history entry."""
        current = {"warmth": 0.5}
        seed = {"trait_weights": {"warmth": 0.5}}
        changes = [{"trait": "warmth", "new": 0.6, "reason": "test"}]

        _apply_validated_changes(
            mem.cold, current, changes, seed, {"recommendation": "more warmth"}, 0.8,
        )

        history = mem.cold.get_personality_history()
        assert len(history) == 1
        assert history[0]["confidence"] == 0.8

    @pytest.mark.timeout(10)
    def test_history_entry_structure(self, mem):
        """History entry contains all expected fields."""
        current = {"warmth": 0.5}
        seed = {"trait_weights": {"warmth": 0.5}}
        changes = [{"trait": "warmth", "new": 0.6, "reason": "test"}]

        _apply_validated_changes(
            mem.cold, current, changes, seed, {"recommendation": "be warmer"}, 0.75,
        )

        history = mem.cold.get_personality_history()
        entry = history[0]
        assert entry["old_traits"] == {"warmth": 0.5}
        assert entry["new_traits"]["warmth"] == 0.6
        assert entry["confidence"] == 0.75
        assert entry["observation_summary"] == "be warmer"
        assert entry["timestamp"] == IsInstance(float)

    @pytest.mark.timeout(10)
    def test_flushes_signals(self, mem):
        """Signals are flushed after successful application."""
        mem.cold.append_personality_signal({"test": True})
        current = {"warmth": 0.5}
        seed = {"trait_weights": {"warmth": 0.5}}
        changes = [{"trait": "warmth", "new": 0.6, "reason": "test"}]

        _apply_validated_changes(
            mem.cold, current, changes, seed, {}, 0.8,
        )

        assert len(mem.cold.get_personality_signals()) == 0

    @pytest.mark.timeout(10)
    def test_uses_seed_weights_fallback_for_unknown_trait(self, mem):
        """Uses 0.5 default for traits not in current or seed."""
        current = {}
        seed = {"trait_weights": {}}
        changes = [{"trait": "new_trait", "new": 0.6, "reason": "test"}]

        result = _apply_validated_changes(
            mem.cold, current, changes, seed, {}, 0.8,
        )
        assert len(result) == 1
        assert result[0]["old"] == 0.5  # default fallback

    @pytest.mark.timeout(10)
    def test_multiple_changes_applied_independently(self, mem):
        """Multiple valid changes each apply with independent clamping."""
        current = {"warmth": 0.5, "wit": 0.4}
        seed = {"trait_weights": {"warmth": 0.5, "wit": 0.4}}
        changes = [
            {"trait": "warmth", "new": 0.6, "reason": "warmer"},
            {"trait": "wit", "new": 0.5, "reason": "wittier"},
        ]

        result = _apply_validated_changes(
            mem.cold, current, changes, seed, {}, 0.9,
        )
        assert len(result) == 2
        traits = {c["trait"] for c in result}
        assert traits == {"warmth", "wit"}


# ── load_current_traits edge cases ───────────────────────────────────────────

class TestLoadCurrentTraitsEdges:
    """Edge cases for loading traits from cold memory."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, mem):
        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.consolidate.publish_event"), \
             patch("arqitect.brain.consolidate.publish_nerve_status"):
            yield

    @pytest.mark.timeout(10)
    def test_empty_seed_uses_defaults(self, mem):
        """Empty seed falls back to DEFAULT_SEED_WEIGHTS."""
        traits = load_current_traits(mem.cold, {})
        assert traits == DEFAULT_SEED_WEIGHTS

    @pytest.mark.timeout(10)
    def test_seed_with_traits_key(self, mem):
        """Seed using 'traits' key instead of 'trait_weights'."""
        seed = {"traits": {"custom": 0.4}}
        traits = load_current_traits(mem.cold, seed)
        assert traits == {"custom": 0.4}

    @pytest.mark.timeout(10)
    def test_stored_traits_override_seed(self, mem):
        """Cold memory traits override seed."""
        stored = {"a": 0.1, "b": 0.2}
        mem.cold.set_fact("personality", "trait_weights", json.dumps(stored))
        seed = {"trait_weights": {"a": 0.9, "b": 0.9}}
        traits = load_current_traits(mem.cold, seed)
        assert traits == stored

    @pytest.mark.timeout(10)
    def test_returns_copy_not_reference(self, mem):
        """Returns a new dict, not a reference to seed internals."""
        seed = {"trait_weights": {"warmth": 0.5}}
        traits = load_current_traits(mem.cold, seed)
        traits["warmth"] = 0.9
        # Original seed should be unmodified
        assert seed["trait_weights"]["warmth"] == 0.5

    @pytest.mark.timeout(10)
    def test_corrupt_stored_traits_falls_back_to_seed(self, mem):
        """Corrupt JSON in cold memory falls back to seed weights."""
        mem.cold.set_fact("personality", "trait_weights", "not-json!!!")
        seed = {"trait_weights": {"warmth": 0.7}}
        traits = load_current_traits(mem.cold, seed)
        assert traits == {"warmth": 0.7}


# ── load_anchors edge cases ──────────────────────────────────────────────────

class TestLoadAnchorsEdges:
    """Edge cases for loading anchors from cold memory."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, mem):
        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.consolidate.publish_event"), \
             patch("arqitect.brain.consolidate.publish_nerve_status"):
            yield

    @pytest.mark.timeout(10)
    def test_corrupt_json_returns_defaults(self, mem):
        """Corrupt JSON in cold memory returns default anchors."""
        mem.cold.set_fact("personality", "anchors", "not valid json")
        anchors = load_anchors(mem.cold)
        assert anchors == {"never": [], "always": [], "bounds": {}}

    @pytest.mark.timeout(10)
    def test_none_value_returns_defaults(self, mem):
        """None value in cold memory returns default anchors."""
        anchors = load_anchors(mem.cold)
        assert anchors == {"never": [], "always": [], "bounds": {}}

    @pytest.mark.timeout(10)
    def test_valid_anchors_returned(self, mem):
        """Valid stored anchors are returned as-is."""
        stored = {"never": ["rude"], "always": ["polite"], "bounds": {"warmth": {"min": 0.3}}}
        mem.cold.set_fact("personality", "anchors", json.dumps(stored))
        anchors = load_anchors(mem.cold)
        assert anchors == stored


# ── record_signal ────────────────────────────────────────────────────────────

class TestRecordSignal:
    """Signal recording delegation to cold memory."""

    @pytest.mark.timeout(10)
    def test_delegates_to_cold_memory(self, mem):
        """record_signal appends to cold memory personality signals."""
        signal = {"user_responded": True, "topic": "test"}
        record_signal(mem.cold, signal)
        signals = mem.cold.get_personality_signals()
        assert len(signals) == 1
        assert signals[0]["user_responded"] is True
        assert signals[0]["topic"] == "test"

    @pytest.mark.timeout(10)
    def test_multiple_signals_accumulate(self, mem):
        """Multiple record_signal calls accumulate signals."""
        record_signal(mem.cold, {"a": 1})
        record_signal(mem.cold, {"b": 2})
        record_signal(mem.cold, {"c": 3})
        signals = mem.cold.get_personality_signals()
        assert len(signals) == 3


# ── validate_against_anchors edge cases ──────────────────────────────────────

class TestValidateAgainstAnchorsEdges:
    """Edge cases for anchor validation."""

    @pytest.mark.timeout(10)
    def test_missing_trait_key(self):
        """Change without trait key is passed through."""
        changes = [{"new": 0.5, "reason": "test"}]
        anchors = {"never": [], "always": [], "bounds": {}}
        result = validate_against_anchors(changes, anchors)
        assert len(result) == 1

    @pytest.mark.timeout(10)
    def test_empty_changes_list(self):
        """Empty changes list returns empty."""
        anchors = {"never": ["x"], "always": [], "bounds": {"y": {"min": 0.5}}}
        result = validate_against_anchors([], anchors)
        assert result == []

    @pytest.mark.timeout(10)
    def test_mixed_blocked_and_passed(self):
        """Mix of blocked and passed changes."""
        changes = [
            {"trait": "sarcasm", "new": 0.5, "reason": "t"},
            {"trait": "warmth", "new": 0.6, "reason": "t"},
            {"trait": "wit", "new": 0.1, "reason": "t"},
        ]
        anchors = {
            "never": ["sarcasm"],
            "always": [],
            "bounds": {"wit": {"min": 0.3}},
        }
        result = validate_against_anchors(changes, anchors)
        assert len(result) == 1
        assert result[0]["trait"] == "warmth"

    @pytest.mark.timeout(10)
    def test_all_changes_blocked(self):
        """All changes blocked returns empty list."""
        changes = [
            {"trait": "sarcasm", "new": 0.5, "reason": "t"},
            {"trait": "rude", "new": 0.3, "reason": "t"},
        ]
        anchors = {"never": ["sarcasm", "rude"], "always": [], "bounds": {}}
        result = validate_against_anchors(changes, anchors)
        assert result == []

    @pytest.mark.timeout(10)
    def test_no_anchors_passes_everything(self):
        """No never/bounds anchors let all changes through."""
        changes = [
            {"trait": "warmth", "new": 0.6, "reason": "t"},
            {"trait": "wit", "new": 0.8, "reason": "t"},
        ]
        anchors = {"never": [], "always": [], "bounds": {}}
        result = validate_against_anchors(changes, anchors)
        assert len(result) == 2

    @pytest.mark.timeout(10)
    @given(
        trait=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
        value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_no_anchors_never_filters(self, trait, value):
        """With empty anchors, any change always passes validation."""
        changes = [{"trait": trait, "new": value, "reason": "hypothesis"}]
        anchors = {"never": [], "always": [], "bounds": {}}
        result = validate_against_anchors(changes, anchors)
        assert len(result) == 1
