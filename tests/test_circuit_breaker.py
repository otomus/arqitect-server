"""P3.4 — Circuit breaker state transitions."""

import json
import time
from unittest.mock import patch

import pytest

from arqitect.brain.circuit_breaker import (
    record_success, record_failure, is_available, get_all_states,
    CB_KEY, FAILURE_THRESHOLD, COOLDOWN_SECONDS,
)


class TestCircuitBreakerClosed:
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


class TestCircuitBreakerOpen:
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
            # Should be blocked
            assert is_available("bad_nerve") is False


class TestCircuitBreakerHalfOpen:
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


class TestCircuitBreakerReset:
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


class TestGetAllStates:
    def test_returns_all_tracked_nerves(self, test_redis):
        """get_all_states should return states for all nerves with history."""
        with patch("arqitect.brain.circuit_breaker.r", test_redis):
            record_failure("nerve_a")
            record_success("nerve_b")

            states = get_all_states()
            assert "nerve_a" in states
            assert "nerve_b" in states
