"""
Tool Qualification — tests fabricated MCP tools before they go live.
Uses LLM critic to validate tool behavior.
"""

import json
import os
import sys
import shutil

import requests

from arqitect.config.loader import get_mcp_tools_dir, get_mcp_url

BRAIN_MODEL = "brain"
MCP_URL = get_mcp_url()
MCP_TOOLS_DIR = get_mcp_tools_dir()


def _log(msg: str):
    print(msg, file=sys.stderr)


def _llm(prompt: str, system: str = "") -> str:
    """Call the brain model for critic reasoning via in-process inference."""
    try:
        from arqitect.inference.router import generate_for_role
        return generate_for_role("brain", prompt, system=system)
    except Exception as e:
        return f"Error: {e}"


def _extract_json(raw: str) -> dict | list | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for open_ch, close_ch in [("[", "]"), ("{", "}")]:
        start = text.find(open_ch)
        if start < 0:
            continue
        depth_sq, depth_cu = 0, 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "[": depth_sq += 1
            elif c == "]": depth_sq -= 1
            elif c == "{": depth_cu += 1
            elif c == "}": depth_cu -= 1
            if depth_sq == 0 and depth_cu == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    return None


def generate_tool_tests(tool_name: str, description: str, params: str) -> list[dict]:
    """Generate 4-6 test cases for a tool."""
    prompt = (
        f"Generate test cases for an MCP tool called '{tool_name}'.\n"
        f"Description: {description}\n"
        f"Parameters: {params}\n\n"
        "Generate 4-6 test cases as a JSON array. Each has:\n"
        '  {"args": {{...}}, "expected_behavior": "what should happen", "category": "happy_path|edge_case|bad_input"}\n\n'
        "Categories:\n"
        "- happy_path (2-3): typical valid inputs\n"
        "- edge_case (1-2): unusual but valid inputs\n"
        "- bad_input (1): invalid/empty input that should be handled gracefully\n\n"
        "Return ONLY the JSON array."
    )
    raw = _llm(prompt)
    result = _extract_json(raw)
    if isinstance(result, list):
        return result
    return []


def call_tool(tool_name: str, args: dict) -> dict:
    """Call a tool via MCP and return structured result."""
    if not isinstance(args, dict):
        args = {"query": str(args)}
    import time as _time
    start = _time.monotonic()
    try:
        resp = requests.post(
            f"{MCP_URL}/call/{tool_name}",
            json=args,
            timeout=30,
        )
        latency = int((_time.monotonic() - start) * 1000)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            return {"success": False, "result": None, "error": data["error"], "latency_ms": latency}
        result = data.get("result", data) if isinstance(data, dict) else data
        return {"success": True, "result": result, "error": None, "latency_ms": latency}
    except Exception as e:
        latency = int((_time.monotonic() - start) * 1000)
        return {"success": False, "result": None, "error": str(e), "latency_ms": latency}


def evaluate_tool_result(test_case: dict, call_result: dict) -> dict:
    """Evaluate whether a tool call result is acceptable."""
    prompt = (
        f"Evaluate this tool call result.\n\n"
        f"Args: {json.dumps(test_case.get('args', {}))}\n"
        f"Expected: {test_case.get('expected_behavior', '')}\n"
        f"Category: {test_case.get('category', '')}\n\n"
        f"Success: {call_result.get('success', False)}\n"
        f"Result: {str(call_result.get('result', ''))[:500]}\n"
        f"Error: {call_result.get('error', '')}\n"
        f"Latency: {call_result.get('latency_ms', 0)}ms\n\n"
        '- Return JSON: {"passed": true/false, "score": 0.0-1.0, "reasoning": "why"}\n'
        "- For bad_input: graceful error handling is a pass\n"
        "- For happy_path: must return a meaningful, non-error result\n"
        "Return ONLY the JSON object."
    )
    raw = _llm(prompt)
    result = _extract_json(raw)
    if isinstance(result, dict):
        return {
            "passed": bool(result.get("passed", False)),
            "score": float(result.get("score", 0.0)),
            "reasoning": result.get("reasoning", ""),
        }
    return {"passed": False, "score": 0.0, "reasoning": "Failed to parse evaluation"}


def quarantine_tool(tool_name: str) -> str:
    """Remove a failed tool file."""
    src = os.path.join(MCP_TOOLS_DIR, f"{tool_name}.py")
    if os.path.exists(src):
        os.remove(src)
        _log(f"[QUALIFY-TOOL] Removed failed tool: {tool_name}")
        return src
    return ""


def qualify_tool(tool_name: str, description: str, params: str) -> dict:
    """Main tool qualification: generate tests, call tool, evaluate.

    Threshold is 0.6 (lower than nerves due to API flakiness).
    Returns {qualified, score, test_results}.
    """
    THRESHOLD = 0.6

    tests = generate_tool_tests(tool_name, description, params)
    if not tests:
        _log(f"[QUALIFY-TOOL] No test cases generated for '{tool_name}'")
        return {"qualified": False, "score": 0.0, "test_results": []}

    results = []
    total_score = 0.0

    for tc in tests:
        args = tc.get("args", {})
        if isinstance(args, str):
            args = {"query": args}
        elif isinstance(args, list):
            args = {"query": str(args[0]) if args else ""}
        elif not isinstance(args, dict):
            args = {"query": str(args)}

        call_result = call_tool(tool_name, args)
        evaluation = evaluate_tool_result(tc, call_result)
        evaluation["args"] = args
        evaluation["category"] = tc.get("category", "")
        results.append(evaluation)
        total_score += evaluation["score"]

    if not results:
        return {"qualified": False, "score": 0.0, "test_results": []}

    avg_score = total_score / len(results)
    passed_count = sum(1 for r in results if r["passed"])
    qualified = avg_score >= THRESHOLD

    _log(f"[QUALIFY-TOOL] Tool '{tool_name}': score={avg_score:.2f}, passed={passed_count}/{len(results)}, qualified={qualified}")

    return {
        "qualified": qualified,
        "score": avg_score,
        "test_results": results,
    }
