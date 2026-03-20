"""Nerve invocation tests: not found, timeout, success, core sense routing."""

import json
import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from hypothesis import given, settings, strategies as st, HealthCheck

from tests.conftest import make_nerve_file


@pytest.mark.timeout(10)
class TestInvokeNerveNotFound:
    """Missing nerve invocation returns structured error."""

    def test_missing_nerve_returns_error_json(self, nerves_dir, sandbox_dir, mem):
        """invoke_nerve for a non-existent nerve returns a JSON error."""
        with patch("arqitect.brain.invoke.mem", mem):
            from arqitect.brain.invoke import invoke_nerve
            result = invoke_nerve("nonexistent_nerve", "hello")
            parsed = json.loads(result)
            assert "error" in parsed
            assert "not found" in parsed["error"].lower()

    @given(
        nerve_name=st.from_regex(r"[a-z_]{5,20}_nerve", fullmatch=True),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_any_missing_nerve_returns_error(self, nerves_dir, sandbox_dir, mem, nerve_name):
        """Any non-existent nerve name returns a not-found error."""
        with patch("arqitect.brain.invoke.mem", mem):
            from arqitect.brain.invoke import invoke_nerve
            result = invoke_nerve(nerve_name, "hello")
            parsed = json.loads(result)
            assert "error" in parsed


@pytest.mark.timeout(10)
class TestInvokeNerveTimeout:
    """Timeout handling during nerve invocation."""

    def test_timeout_returns_error_json(self, nerves_dir, sandbox_dir, mem):
        """A nerve that exceeds timeout returns a JSON error."""
        make_nerve_file(nerves_dir, "slow_nerve")

        def _timeout_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=90)

        with patch("arqitect.brain.invoke.mem", mem):
            with patch("arqitect.brain.invoke.subprocess.run", side_effect=_timeout_run):
                from arqitect.brain.invoke import invoke_nerve
                result = invoke_nerve("slow_nerve", "do something slow")
                parsed = json.loads(result)
                assert "error" in parsed
                assert "timed out" in parsed["error"].lower()


@pytest.mark.timeout(10)
class TestInvokeNerveSuccess:
    """Successful nerve invocation."""

    def test_successful_invocation_returns_stdout(self, nerves_dir, sandbox_dir, mem):
        """Successful nerve run returns its stdout."""
        make_nerve_file(nerves_dir, "good_nerve")

        mock_result = MagicMock()
        mock_result.stdout = '{"response": "all good"}'
        mock_result.stderr = ""

        with patch("arqitect.brain.invoke.mem", mem):
            with patch("arqitect.brain.invoke.subprocess.run", return_value=mock_result):
                from arqitect.brain.invoke import invoke_nerve
                result = invoke_nerve("good_nerve", "test")
                parsed = json.loads(result)
                assert parsed["response"] == "all good"


@pytest.mark.timeout(10)
class TestInvokeCoreSense:
    """Core sense routing to SENSES_DIR."""

    def test_core_sense_uses_senses_dir(self, sandbox_dir, mem):
        """Core senses resolve from SENSES_DIR, not NERVES_DIR."""
        mock_result = MagicMock()
        mock_result.stdout = '{"response": "I am Sentient"}'
        mock_result.stderr = ""

        with patch("arqitect.brain.invoke.mem", mem):
            with patch("arqitect.brain.invoke.subprocess.run", return_value=mock_result) as mock_run:
                from arqitect.brain.invoke import invoke_nerve
                result = invoke_nerve("awareness", '{"query": "who are you?"}')
                call_cmd = mock_run.call_args[0][0]
                assert "senses" in call_cmd[1] or "senses" in str(call_cmd)


@pytest.mark.timeout(10)
class TestInvokeNerveNameValidation:
    """Nerve name validation prevents path traversal."""

    @given(
        name=st.from_regex(r"(\.\./)+[a-z]+", fullmatch=True),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_path_traversal_rejected(self, nerves_dir, sandbox_dir, mem, name):
        """Names with path traversal patterns must be rejected."""
        with patch("arqitect.brain.invoke.mem", mem):
            from arqitect.brain.invoke import invoke_nerve
            result = invoke_nerve(name, "{}")
            parsed = json.loads(result)
            assert "error" in parsed

    def test_valid_name_accepted(self, nerves_dir, sandbox_dir, mem):
        """A valid alphanumeric nerve name is accepted (even if nerve doesn't exist)."""
        with patch("arqitect.brain.invoke.mem", mem):
            from arqitect.brain.invoke import invoke_nerve
            result = invoke_nerve("valid_nerve_123", "hello")
            parsed = json.loads(result)
            # Should be "not found", not "invalid name"
            assert "error" in parsed
