"""P3 — Nerve invocation tests: not found, timeout, wrong_nerve bounce."""

import json
import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import make_nerve_file


class TestInvokeNerveNotFound:
    def test_missing_nerve_returns_error_json(self, nerves_dir, sandbox_dir, mem):
        """invoke_nerve for a non-existent nerve returns a JSON error."""
        with patch("arqitect.brain.invoke.mem", mem):
            from arqitect.brain.invoke import invoke_nerve
            result = invoke_nerve("nonexistent_nerve", "hello")
            parsed = json.loads(result)
            assert "error" in parsed
            assert "not found" in parsed["error"].lower()


class TestInvokeNerveTimeout:
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


class TestInvokeNerveSuccess:
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


class TestInvokeCoreSense:
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
