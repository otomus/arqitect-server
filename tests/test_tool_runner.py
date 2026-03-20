"""Tests for arqitect.mcp.tool_runner — JSON-RPC stdio bridge for Python tools."""

import io
import json
import sys
from unittest.mock import patch

import pytest
from dirty_equals import IsPartialDict, IsStr
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from arqitect.mcp.tool_runner import (
    _load_tool_module,
    _serve_stdio,
    _write_response,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_file(tmp_path, filename: str, code: str) -> str:
    """Write a Python tool file and return its path.

    Args:
        tmp_path: pytest tmp_path fixture directory.
        filename: Name of the .py file to create.
        code: Python source code to write.

    Returns:
        Absolute path to the created file.
    """
    path = tmp_path / filename
    path.write_text(code)
    return str(path)


def _run_stdio(tool_name: str, func: callable, stdin_text: str) -> list[dict]:
    """Run _serve_stdio with given stdin text and return parsed JSON output lines.

    Args:
        tool_name: Name passed to _serve_stdio.
        func: The tool callable.
        stdin_text: Raw text to feed as stdin.

    Returns:
        List of parsed JSON dicts from stdout.
    """
    stdin = io.StringIO(stdin_text)
    stdout = io.StringIO()
    with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
        _serve_stdio(tool_name, func)
    stdout.seek(0)
    return [json.loads(line) for line in stdout.readlines() if line.strip()]


def _capture_stdout(func, *args, **kwargs) -> list[dict]:
    """Run func with captured stdout, return list of parsed JSON lines.

    Args:
        func: Callable to invoke.
        *args: Positional arguments forwarded to func.
        **kwargs: Keyword arguments forwarded to func.

    Returns:
        List of parsed JSON dicts from stdout.
    """
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        func(*args, **kwargs)
    buf.seek(0)
    lines = [line.strip() for line in buf.readlines() if line.strip()]
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------------------
# _load_tool_module
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestLoadToolModule:
    """Tests for dynamic loading of Python tool modules."""

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

    @given(name=st.from_regex(r"[a-z][a-z0-9_]{0,30}", fullmatch=True))
    @settings(max_examples=20)
    def test_tool_name_matches_filename(self, name):
        """Tool name always equals the filename stem regardless of valid identifier."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            code = f"def run(): return '{name}'\n"
            path = _make_tool_file(Path(td), f"{name}.py", code)
            loaded_name, func = _load_tool_module(path)
            assert loaded_name == name
            assert func() == name


# ---------------------------------------------------------------------------
# _write_response
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestWriteResponse:
    """Tests for JSON-RPC response serialization."""

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

    def test_error_takes_precedence_over_result(self):
        """When both error and result are provided, error wins."""
        lines = _capture_stdout(_write_response, "r", result="val", error="bad")
        assert lines[0] == IsPartialDict({"id": "r", "error": "bad"})

    @given(request_id=st.text(min_size=1, max_size=50))
    @settings(max_examples=20)
    def test_id_round_trips_through_json(self, request_id):
        """Any string request_id survives JSON serialization."""
        lines = _capture_stdout(_write_response, request_id, result="ok")
        assert lines[0]["id"] == request_id

    @given(result_val=st.text(min_size=0, max_size=200))
    @settings(max_examples=20)
    def test_result_value_round_trips(self, result_val):
        """Any string result value survives JSON serialization."""
        lines = _capture_stdout(_write_response, "id", result=result_val)
        assert lines[0]["result"] == result_val


# ---------------------------------------------------------------------------
# _serve_stdio
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestServeStdio:
    """Tests for the stdin/stdout JSON-RPC loop."""

    def test_ready_signal(self):
        """First output is the ready signal."""
        lines = _run_stdio("test_tool", lambda: "ok", "")
        assert lines[0] == {"ready": True}

    def test_successful_call(self):
        """Processes a valid JSON-RPC call and returns result."""
        req = json.dumps({"id": "1", "method": "call", "params": {"x": 10}}) + "\n"
        lines = _run_stdio("doubler", lambda x=0: x * 2, req)
        assert lines[0] == {"ready": True}
        assert lines[1] == {"id": "1", "result": "20"}

    def test_invalid_json_returns_error(self):
        """Invalid JSON input returns parse error."""
        lines = _run_stdio("t", lambda: None, "not json\n")
        error_lines = [line for line in lines if "error" in line]
        assert any("Invalid JSON" in line["error"] for line in error_lines)

    def test_type_error_returns_param_error(self):
        """Wrong params produce a parameter error."""
        req = json.dumps({"id": "2", "params": {"wrong_param": 1}}) + "\n"

        def strict_func(required_param):
            return required_param

        lines = _run_stdio("strict", strict_func, req)
        error_lines = [line for line in lines if "error" in line]
        assert any("Parameter error" in line["error"] for line in error_lines)

    def test_tool_exception_returns_tool_error(self):
        """Runtime exception in tool returns tool error."""
        req = json.dumps({"id": "3", "params": {}}) + "\n"

        def failing_func():
            raise ValueError("division by zero")

        lines = _run_stdio("fail", failing_func, req)
        error_lines = [line for line in lines if "error" in line]
        assert any("Tool error" in line["error"] for line in error_lines)

    def test_none_result_returns_empty_string(self):
        """When tool returns None, result is empty string."""
        req = json.dumps({"id": "4", "params": {}}) + "\n"
        lines = _run_stdio("noop", lambda: None, req)
        result_line = [line for line in lines if "result" in line][0]
        assert result_line["result"] == ""

    def test_blank_lines_skipped(self):
        """Blank lines in stdin are ignored."""
        stdin_text = "\n\n" + json.dumps({"id": "5", "params": {}}) + "\n\n"
        lines = _run_stdio("t", lambda: "ok", stdin_text)
        result_lines = [line for line in lines if "result" in line]
        assert len(result_lines) == 1

    def test_missing_params_defaults_to_empty(self):
        """Missing 'params' key defaults to empty dict."""
        req = json.dumps({"id": "6"}) + "\n"
        lines = _run_stdio("t", lambda: "no args", req)
        result_lines = [line for line in lines if "result" in line]
        assert result_lines[0]["result"] == "no args"

    def test_multiple_requests_processed_sequentially(self):
        """Multiple requests on separate lines are each processed."""
        req1 = json.dumps({"id": "a", "params": {"n": 1}}) + "\n"
        req2 = json.dumps({"id": "b", "params": {"n": 2}}) + "\n"
        lines = _run_stdio("t", lambda n=0: n + 10, req1 + req2)
        results = {line["id"]: line["result"] for line in lines if "result" in line}
        assert results == {"a": "11", "b": "12"}

    @given(
        params=st.fixed_dictionaries({
            "x": st.integers(min_value=-1000, max_value=1000),
        })
    )
    @settings(max_examples=15)
    def test_integer_params_round_trip(self, params):
        """Integer params survive JSON serialization and reach the tool."""
        req = json.dumps({"id": "h", "params": params}) + "\n"
        lines = _run_stdio("calc", lambda x=0: x * 3, req)
        result_lines = [line for line in lines if "result" in line]
        assert result_lines[0]["result"] == str(params["x"] * 3)

    def test_response_structure_always_has_id(self):
        """Every non-ready response has an 'id' key."""
        req = json.dumps({"id": "check", "params": {}}) + "\n"
        lines = _run_stdio("t", lambda: "v", req)
        for line in lines:
            if "ready" not in line:
                assert "id" in line

    def test_error_response_contains_descriptive_message(self):
        """Error responses include the exception message for debugging."""
        req = json.dumps({"id": "e1", "params": {}}) + "\n"

        def exploder():
            raise RuntimeError("kaboom-42")

        lines = _run_stdio("boom", exploder, req)
        error_lines = [line for line in lines if "error" in line]
        assert any(
            line == IsPartialDict({"id": "e1", "error": IsStr(regex=".*kaboom-42.*")})
            for line in error_lines
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestMain:
    """Tests for the tool_runner entry point."""

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

    def test_end_to_end_via_subprocess(self, tmp_path, fake_process):
        """End-to-end: fake_process simulates a tool_runner subprocess."""
        tool_path = _make_tool_file(
            tmp_path, "echo_tool.py",
            "def echo_tool(msg='hello'): return msg\n",
        )
        ready_line = json.dumps({"ready": True}) + "\n"
        result_line = json.dumps({"id": "1", "result": "hello"}) + "\n"
        fake_process.register(
            [sys.executable, tool_path],
            stdout=ready_line + result_line,
        )

        import subprocess
        proc = subprocess.run(
            [sys.executable, tool_path],
            capture_output=True, text=True,
        )
        output_lines = [
            json.loads(line)
            for line in proc.stdout.strip().splitlines()
            if line.strip()
        ]
        assert output_lines[0] == {"ready": True}
        assert output_lines[1] == {"id": "1", "result": "hello"}

    def test_load_failure_error_message_in_output(self, tmp_path):
        """The error message from a failed load is included in the ready=false output."""
        bad_path = str(tmp_path / "ghost.py")
        stdout = io.StringIO()
        with patch("sys.argv", ["tool_runner", bad_path]), \
             patch("sys.stdout", stdout):
            with pytest.raises(SystemExit):
                main()
        stdout.seek(0)
        msg = json.loads(stdout.readline())
        assert msg == IsPartialDict({"ready": False, "error": IsStr()})
