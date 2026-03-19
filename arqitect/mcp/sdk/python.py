"""SDK template for Python tool contributors.

Copy this file as your tool's entry point (run.py) and implement the handle() function.

Protocol: reads JSON-RPC from stdin, writes JSON-RPC to stdout.

Usage in tool.json:
    {"runtime": "python", "entry": "run.py"}
"""

import json
import sys


def handle(params: dict) -> str:
    """Implement your tool logic here.

    Args:
        params: The parameters passed by the caller.

    Returns:
        Result string (use json.dumps for structured data).
    """
    raise NotImplementedError("Replace this with your tool logic")


def main() -> None:
    """Stdio JSON-RPC loop — do not modify."""
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        try:
            result = handle(req.get("params", {}))
            resp = {"id": req.get("id"), "result": str(result)}
        except Exception as e:
            resp = {"id": req.get("id"), "error": str(e)}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
