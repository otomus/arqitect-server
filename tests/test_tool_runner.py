"""Tests for arqitect.mcp.tool_runner — JSON-RPC stdio bridge for Python tools."""

import io
import json
import os
import sys
from unittest.mock import patch

import pytest

from arqitect.mcp.tool_runner import (
    _load_tool_module,
    _serve_stdio,
    _write_response,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_file(tmp_path, filename, code):
    """Write a Python tool file and return its path."""
    path = tmp_path / filename
    path.write_text(code)
    return str(path)


def _capture_stdout(func, *args, **kwargs):
    """Run func with captured stdout, return list of parsed JSON lines."""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        func(*args, **kwargs)
    buf.seek(0)
    lines = [l.strip() for l in buf.readlines() if l.strip()]
    return [json.loads(l) for l in lines]


# ---------------------------------------------------------------------------
# _load_tool_module
# ---------------------------------------------------------------------------

class TestLoadToolModule:
    def test_load_named_function(self, tmp_path):
        """Loads function matching filename."""
        code = "def greet(name='world'):\n    return f'Hello {name}'\n"
        path = _make_tool_file(tmp_path, "greet.py", code)
        name, func = _load_tool_module(path)
        assert name == "greet"
        assert func(name="Alice") == "Hello Alice"

    def test_load_run_fallback(self, tmp_path):
        """Falls back to run() when named function is missing."""
        code = "def run(x=1):\n    return x * 2\n"
        path = _make_tool_file(tmp_path, "doubler.py", code)
        name, func = _load_tool_module(path)
        assert name == "doubler"
        assert func(x=5) == 10

    def test_load_prefers_named_over_run(self, tmp_path):
        """Named function takes precedence over run()."""
        code = (
            "def my_tool(q=''):\n    return 'named'\n"
            "def run(q=''):\n    return 'fallback'\n"
        )
        path = _make_tool_file(tmp_path, "my_tool.py", code)
        _, func = _load_tool_module(path)
        assert func() == "named"

    def test_load_no_callable_raises(self, tmp_path):
        """Raises ValueError when neither named nor run() found."""
        code = "x = 42\n"
        path = _make_tool_file(tmp_path, "no_func.py", code)
        with pytest.raises(ValueError, match="No callable"):
            _load_tool_module(path)

    def test_load_named_not_callable_falls_to_run(self, tmp_path):
        """If named attribute exists but is not callable, falls back to run()."""
        code = "calculator = 'not a function'\ndef run(): return 'ok'\n"
        path = _make_tool_file(tmp_path, "calculator.py", code)
        _, func = _load_tool_module(path)
        assert func() == "ok"

    def test_load_strips_extension(self, tmp_path):
        """Tool name is derived from filename without .py extension."""
        code = "def run(): return 1\n"
        path = _make_tool_file(tmp_path, "weather_tool.py", code)
        name, _ = _load_tool_module(path)
        assert name == "weather_tool"

    def test_load_syntax_error_raises(self, tmp_path):
        """Syntax errors in the tool file propagate."""
        code = "def broken(\n"
        path = _make_tool_file(tmp_path, "broken.py", code)
        with pytest.raises(SyntaxError):
            _load_tool_module(path)

    def test_load_import_error_raises(self, tmp_path):
        """Import errors in the tool file propagate."""
        code = "import nonexistent_module_xyz_123\ndef run(): pass\n"
        path = _make_tool_file(tmp_path, "bad_import.py", code)
        with pytest.raises(ModuleNotFoundError):
            _load_tool_module(path)


# ---------------------------------------------------------------------------
# _write_response
# ---------------------------------------------------------------------------

class TestWriteResponse:
    def test_writes_result(self):
        """Writes a result response."""
        lines = _capture_stdout(_write_response, "req1", result="hello")
        assert lines[0] == {"id": "req1", "result": "hello"}

    def test_writes_error(self):
        """Writes an error response."""
        lines = _capture_stdout(_write_response, "req2", error="boom")
        assert lines[0] == {"id": "req2", "error": "boom"}

    def test_writes_none_id(self):
        """Handles None request id."""
        lines = _capture_stdout(_write_response, None, result="ok")
        assert lines[0]["id"] is None

    def test_result_takes_precedence_when_no_error(self):
        """When error is None, result key is used."""
        lines = _capture_stdout(_write_response, "r", result="val", error=None)
        assert "result" in lines[0]
        assert "error" not in lines[0]


# ---------------------------------------------------------------------------
# _serve_stdio
# ---------------------------------------------------------------------------

class TestServeStdio:
    def test_ready_signal(self):
        """First output is the ready signal."""
        stdin = io.StringIO("")  # empty, loop exits immediately
        stdout = io.StringIO()
        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            _serve_stdio("test_tool", lambda: "ok")
        stdout.seek(0)
        first = json.loads(stdout.readline())
        assert first == {"ready": True}

    def test_successful_call(self):
        """Processes a valid JSON-RPC call and returns result."""
        req = json.dumps({"id": "1", "method": "call", "params": {"x": 10}}) + "\n"
        stdin = io.StringIO(req)
        stdout = io.StringIO()

        def tool_func(x=0):
            return x * 2

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            _serve_stdio("doubler", tool_func)

        stdout.seek(0)
        lines = [json.loads(l) for l in stdout.readlines() if l.strip()]
        # First line is ready, second is result
        assert lines[0] == {"ready": True}
        assert lines[1] == {"id": "1", "result": "20"}

    def test_invalid_json_returns_error(self):
        """Invalid JSON input returns parse error."""
        stdin = io.StringIO("not json\n")
        stdout = io.StringIO()

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            _serve_stdio("t", lambda: None)

        stdout.seek(0)
        lines = [json.loads(l) for l in stdout.readlines() if l.strip()]
        assert any("Invalid JSON" in l.get("error", "") for l in lines)

    def test_type_error_returns_param_error(self):
        """Wrong params produce a parameter error."""
        req = json.dumps({"id": "2", "params": {"wrong_param": 1}}) + "\n"
        stdin = io.StringIO(req)
        stdout = io.StringIO()

        def strict_func(required_param):
            return required_param

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            _serve_stdio("strict", strict_func)

        stdout.seek(0)
        lines = [json.loads(l) for l in stdout.readlines() if l.strip()]
        error_lines = [l for l in lines if "error" in l]
        assert any("Parameter error" in l["error"] for l in error_lines)

    def test_tool_exception_returns_tool_error(self):
        """Runtime exception in tool returns tool error."""
        req = json.dumps({"id": "3", "params": {}}) + "\n"
        stdin = io.StringIO(req)
        stdout = io.StringIO()

        def failing_func():
            raise ValueError("division by zero")

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            _serve_stdio("fail", failing_func)

        stdout.seek(0)
        lines = [json.loads(l) for l in stdout.readlines() if l.strip()]
        error_lines = [l for l in lines if "error" in l]
        assert any("Tool error" in l["error"] for l in error_lines)

    def test_none_result_returns_empty_string(self):
        """When tool returns None, result is empty string."""
        req = json.dumps({"id": "4", "params": {}}) + "\n"
        stdin = io.StringIO(req)
        stdout = io.StringIO()

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            _serve_stdio("noop", lambda: None)

        stdout.seek(0)
        lines = [json.loads(l) for l in stdout.readlines() if l.strip()]
        result_line = [l for l in lines if "result" in l][0]
        assert result_line["result"] == ""

    def test_blank_lines_skipped(self):
        """Blank lines in stdin are ignored."""
        lines_in = "\n\n" + json.dumps({"id": "5", "params": {}}) + "\n\n"
        stdin = io.StringIO(lines_in)
        stdout = io.StringIO()

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            _serve_stdio("t", lambda: "ok")

        stdout.seek(0)
        lines = [json.loads(l) for l in stdout.readlines() if l.strip()]
        result_lines = [l for l in lines if "result" in l]
        assert len(result_lines) == 1

    def test_missing_params_defaults_to_empty(self):
        """Missing 'params' key defaults to empty dict."""
        req = json.dumps({"id": "6"}) + "\n"
        stdin = io.StringIO(req)
        stdout = io.StringIO()

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            _serve_stdio("t", lambda: "no args")

        stdout.seek(0)
        lines = [json.loads(l) for l in stdout.readlines() if l.strip()]
        result_lines = [l for l in lines if "result" in l]
        assert result_lines[0]["result"] == "no args"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_no_args_exits(self):
        """Exits with code 1 when no tool file arg provided."""
        with patch("sys.argv", ["tool_runner"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_load_failure_writes_ready_false(self, tmp_path):
        """When tool module fails to load, writes ready=false and exits."""
        bad_path = str(tmp_path / "nonexistent.py")
        stdout = io.StringIO()
        with patch("sys.argv", ["tool_runner", bad_path]), \
             patch("sys.stdout", stdout):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
        stdout.seek(0)
        msg = json.loads(stdout.readline())
        assert msg["ready"] is False
        assert "error" in msg

    def test_successful_load_enters_loop(self, tmp_path):
        """Successful load enters the stdio loop (ready signal emitted)."""
        code = "def my_tool(): return 'hi'\n"
        path = _make_tool_file(tmp_path, "my_tool.py", code)
        stdin = io.StringIO("")  # empty stdin, loop exits immediately
        stdout = io.StringIO()
        with patch("sys.argv", ["tool_runner", path]), \
             patch("sys.stdin", stdin), \
             patch("sys.stdout", stdout):
            main()
        stdout.seek(0)
        ready = json.loads(stdout.readline())
        assert ready == {"ready": True}
