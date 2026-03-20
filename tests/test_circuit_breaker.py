"""Tests for circuit breaker state transitions.

Covers:
- Closed state (normal operation)
- Open state (blocking after threshold failures)
- Half-open state (cooldown recovery probe)
- Reset on success
- State tracking across multiple nerves
"""

import json
import time
from unittest.mock import patch

import pytest
import time_machine
from dirty_equals import IsPositive
from hypothesis import given, settings, strategies as st, HealthCheck

from arqitect.brain.circuit_breaker import (
    record_success, record_failure, is_available, get_all_states,
    CB_KEY, FAILURE_THRESHOLD, COOLDOWN_SECONDS,
)


@pytest.mark.timeout(10)
class TestCircuitBreakerClosed:
    """Closed state -- normal operation."""

    def test_new_nerve_is_available(self, test_redis):
        """A nerve with no history should be available (closed circuit)."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            assert is_available("fresh_nerve") is True

    def test_single_failure_stays_closed(self, test_redis):
        """One failure should not open the circuit."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            record_failure("flaky_nerve")
            assert is_available("flaky_nerve") is True

    def test_two_failures_stays_closed(self, test_redis):
        """Two failures (below threshold of 3) should keep circuit closed."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            record_failure("flaky_nerve")
            record_failure("flaky_nerve")
            assert is_available("flaky_nerve") is True

    @given(
        failure_count=st.integers(min_value=0, max_value=FAILURE_THRESHOLD - 1),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_below_threshold_stays_closed(self, test_redis, failure_count):
        """Any failure count below threshold keeps the circuit closed."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            test_redis.flushall()
            for _ in range(failure_count):
                record_failure("prop_nerve")
            assert is_available("prop_nerve") is True


@pytest.mark.timeout(10)
class TestCircuitBreakerOpen:
    """Open state -- blocking after threshold failures."""

    def test_three_failures_opens_circuit(self, test_redis):
        """FAILURE_THRESHOLD consecutive failures should open the circuit."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            for _ in range(FAILURE_THRESHOLD):
                record_failure("bad_nerve")
            assert is_available("bad_nerve") is False

    def test_open_circuit_blocks_routing(self, test_redis):
        """An open circuit should return False for is_available."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            for _ in range(FAILURE_THRESHOLD):
                record_failure("bad_nerve")
            assert is_available("bad_nerve") is False

    @given(
        extra_failures=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_excess_failures_keep_circuit_open(self, test_redis, extra_failures):
        """Failures beyond threshold still keep the circuit open."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            test_redis.flushall()
            for _ in range(FAILURE_THRESHOLD + extra_failures):
                record_failure("overloaded_nerve")
            assert is_available("overloaded_nerve") is False


@pytest.mark.timeout(10)
class TestCircuitBreakerHalfOpen:
    """Half-open state -- cooldown recovery."""

    def test_cooldown_transitions_to_half_open(self, test_redis):
        """After COOLDOWN_SECONDS, an open circuit should become half-open."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            for _ in range(FAILURE_THRESHOLD):
                record_failure("recovering_nerve")

            # Simulate time passing beyond cooldown
            state = json.loads(test_redis.hget(CB_KEY, "recovering_nerve"))
            state["opened_at"] = time.time() - COOLDOWN_SECONDS - 1
            test_redis.hset(CB_KEY, "recovering_nerve", json.dumps(state))

            # Should now be available (half-open allows a probe)
            assert is_available("recovering_nerve") is True

    @time_machine.travel("2026-03-20 12:00:00", tick=False)
    def test_cooldown_not_yet_elapsed_stays_open(self, test_redis):
        """Before COOLDOWN_SECONDS elapse, the circuit stays open."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            for _ in range(FAILURE_THRESHOLD):
                record_failure("stuck_nerve")

            # opened_at is now(), and we don't tick, so cooldown hasn't passed
            assert is_available("stuck_nerve") is False


@pytest.mark.timeout(10)
class TestCircuitBreakerReset:
    """Reset behavior on success."""

    def test_success_resets_after_failures(self, test_redis):
        """A success should reset the failure count and close the circuit."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            record_failure("flaky_nerve")
            record_failure("flaky_nerve")
            record_success("flaky_nerve")

            assert is_available("flaky_nerve") is True
            state = json.loads(test_redis.hget(CB_KEY, "flaky_nerve"))
            assert state["failures"] == 0
            assert state["state"] == "closed"

    def test_success_closes_open_circuit(self, test_redis):
        """A success on a half-open circuit should close it."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            for _ in range(FAILURE_THRESHOLD):
                record_failure("recovering_nerve")

            # Simulate half-open
            state = json.loads(test_redis.hget(CB_KEY, "recovering_nerve"))
            state["opened_at"] = time.time() - COOLDOWN_SECONDS - 1
            test_redis.hset(CB_KEY, "recovering_nerve", json.dumps(state))

            # Probe succeeds
            record_success("recovering_nerve")
            assert is_available("recovering_nerve") is True
            state = json.loads(test_redis.hget(CB_KEY, "recovering_nerve"))
            assert state["state"] == "closed"


@pytest.mark.timeout(10)
class TestGetAllStates:
    """State tracking across multiple nerves."""

    def test_returns_all_tracked_nerves(self, test_redis):
        """get_all_states should return states for all nerves with history."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            record_failure("nerve_a")
            record_success("nerve_b")

            states = get_all_states()
            assert "nerve_a" in states
            assert "nerve_b" in states

    def test_states_contain_expected_fields(self, test_redis):
        """Each state entry should have at least 'state' and 'failures' fields."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            record_failure("field_nerve")
            states = get_all_states()
            state = states["field_nerve"]
            assert "state" in state
            assert "failures" in state
            assert state["failures"] == IsPositive
