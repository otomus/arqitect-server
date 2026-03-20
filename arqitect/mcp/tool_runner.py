"""Tool Runner — bridges bare Python tools to the stdio JSON-RPC protocol.

Wraps mcp_tools/*.py files that define a named function (or run() fallback)
so they can be called as subprocess-based tools via stdin/stdout JSON-RPC.

Usage:
    .venv/bin/python -m arqitect.mcp.tool_runner /path/to/mcp_tools/weather_tool.py
"""

import importlib.util
import json
import sys


def _load_tool_module(filepath: str) -> tuple[str, callable]:
    """Load a Python tool module and return (tool_name, callable).

    Finds the primary callable: either a named function matching the filename,
    or a run() function.

    Args:
        filepath: Absolute path to the .py tool file.

    Returns:
        Tuple of (tool_name, callable_function).

    Raises:
        ValueError: If no callable tool function is found in the module.
    """
    import os

    tool_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(tool_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Prefer named function matching filename, fall back to run()
    func = getattr(mod, tool_name, None)
    if func is None or not callable(func):
        func = getattr(mod, "run", None)
    if func is None or not callable(func):
        raise ValueError(f"No callable '{tool_name}()' or 'run()' found in {filepath}")

    return tool_name, func


def _serve_stdio(tool_name: str, func: callable) -> None:
    """Enter the stdin/stdout JSON-RPC loop.

    Reads one JSON object per line from stdin, calls the tool function,
    and writes one JSON object per line to stdout.

    Protocol:
        -> {"id": "abc", "method": "call", "params": {"query": "NYC"}}
        <- {"id": "abc", "result": "72F, sunny"}
        <- {"id": "abc", "error": "Missing required param: query"}

    Args:
        tool_name: Name of the tool (for error messages).
        func: The callable tool function.
    """
    # Signal readiness
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            _write_response(None, error=f"Invalid JSON: {e}")
            continue

        request_id = request.get("id")
        params = request.get("params", {})

        try:
            result = func(**params)
            _write_response(request_id, result=str(result) if result is not None else "")
        except TypeError as e:
            _write_response(request_id, error=f"Parameter error: {e}")
        except Exception as e:
            _write_response(request_id, error=f"Tool error: {e}")


def _write_response(request_id: str | None, result: str | None = None,
                    error: str | None = None) -> None:
    """Write a JSON-RPC response line to stdout.

    Args:
        request_id: The request ID to echo back.
        result: Successful result string.
        error: Error message string.
    """
    response = {"id": request_id}
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def main() -> None:
    """Entry point for tool_runner subprocess execution."""
    if len(sys.argv) < 2:
        print("Usage: python -m arqitect.mcp.tool_runner <tool_file.py>", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]
    try:
        tool_name, func = _load_tool_module(filepath)
    except Exception as e:
        # Write error as ready=false so the manager knows startup failed
        sys.stdout.write(json.dumps({"ready": False, "error": str(e)}) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    _serve_stdio(tool_name, func)


if __name__ == "__main__":
    main()
