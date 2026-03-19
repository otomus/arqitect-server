"""
Synapse MCP Server — FastMCP-based tool server with isolated process pool.

Tools run as isolated subprocesses via the ToolManager, communicating over
stdin/stdout JSON-RPC. Supports Python, Node, Go, Rust, binary, and Docker runtimes.

Provides both:
  - Standard MCP protocol (via FastMCP, for external MCP clients)
  - HTTP endpoints (for internal nerve_runtime communication):
      GET  /tools          — list all available tools (local + external)
      POST /call/<tool>    — call a tool with JSON body as arguments
      GET  /health         — health check
"""

import atexit
import json
import os
import sys

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from arqitect.config.loader import get_mcp_tools_dir
from arqitect.mcp.external_manager import ExternalMCPManager
from arqitect.mcp.tool_manager import ToolManager

sys.stdout.reconfigure(line_buffering=True)

# Monitoring database for tool usage stats (separate from user data)
_monitoring = None

# Redis for publishing tool call events to dashboard
_redis = None


def _ensure_monitoring():
    """Lazy-init the monitoring database. Returns the MonitoringMemory instance."""
    global _monitoring
    if _monitoring is None:
        from arqitect.memory.monitoring import MonitoringMemory
        _monitoring = MonitoringMemory()
    return _monitoring


def _record_tool_usage(tool_name: str, success: bool, latency_ms: float = 0,
                       error_message: str = ""):
    """Persist tool call stats in monitoring database. Best-effort, never blocks."""
    try:
        _ensure_monitoring().record_call("tool", tool_name, success,
                                         latency_ms=latency_ms,
                                         error_message=error_message)
    except Exception:
        pass


def _record_mcp_usage(server_name: str, tool_name: str, success: bool,
                      latency_ms: float = 0, error_message: str = ""):
    """Persist external MCP call stats in monitoring database. Best-effort, never blocks."""
    try:
        _ensure_monitoring().record_call("mcp", f"{server_name}/{tool_name}", success,
                                         latency_ms=latency_ms,
                                         error_message=error_message)
    except Exception:
        pass


def _publish_tool_event(tool_name: str, args: dict, result=None, error=None, elapsed=0):
    """Publish tool call events to Redis for dashboard visibility."""
    global _redis
    try:
        if _redis is None:
            import redis
            from arqitect.config.loader import get_redis_host_port
            _host, _port = get_redis_host_port()
            _redis = redis.Redis(host=_host, port=_port, decode_responses=True)
        event = {"tool": tool_name, "args": {k: str(v)[:100] for k, v in args.items()}, "elapsed": round(elapsed, 2)}
        if error:
            event["error"] = str(error)[:200]
        elif result is not None:
            event["result_preview"] = str(result)[:200]
        _redis.publish("mcp:tool_call", json.dumps(event))
    except Exception:
        pass  # Best-effort


TOOLS_DIR = get_mcp_tools_dir()
os.makedirs(TOOLS_DIR, exist_ok=True)

# External MCP server manager
ext_mcp = ExternalMCPManager()
atexit.register(ext_mcp.shutdown_all)

# Tool process pool manager
tool_mgr = ToolManager()
atexit.register(tool_mgr.shutdown)

# Core tools — never overwritten by fabrication
CORE_TOOLS = frozenset({"image_generator"})

# ── FastMCP Server ─────────────────────────────────────────────────────────

mcp = FastMCP("Synapse Tools")


# ── Legacy HTTP Endpoints (for nerve_runtime compatibility) ────────────────

@mcp.custom_route("/tools", methods=["GET"])
async def list_tools_handler(request: Request):
    """List all available tools (managed process pool + external)."""
    tool_mgr.scan()

    # Start with ToolManager's registry (subprocess-based tools)
    tools_list = tool_mgr.list_tools()

    # Merge external MCP server tools
    ext_tools = ext_mcp.list_all_tools()
    for name, info in ext_tools.items():
        if name not in tools_list:
            tools_list[name] = {
                "description": info["description"],
                "params": info["params"],
                "source": "external",
                "server": info["server"],
            }
    return JSONResponse({"tools": tools_list})


@mcp.custom_route("/call/{tool_name}", methods=["POST"])
async def call_tool_handler(request: Request):
    """Call a tool — routes through ToolManager process pool or external MCP."""
    import asyncio
    import time as _time

    tool_name = request.path_params["tool_name"]
    tool_mgr.scan()  # Hot-reload in case brain just fabricated a new tool

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Route 1: ToolManager process pool (preferred — isolated subprocess)
    meta = tool_mgr.get_meta(tool_name)
    if meta:
        # Remap args if the model used wrong parameter names
        expected_params = list(meta.params.keys()) if isinstance(meta.params, dict) else meta.params
        if body and expected_params and set(body.keys()) != set(expected_params):
            remapped = {k: v for k, v in body.items() if k in expected_params}
            unmatched_vals = [v for k, v in body.items() if k not in expected_params]
            remaining_params = [p for p in expected_params if p not in remapped]
            for param, val in zip(remaining_params, unmatched_vals):
                remapped[param] = val
            body = remapped

        try:
            t0 = _time.time()
            result = await asyncio.to_thread(tool_mgr.call, tool_name, body)
            elapsed = _time.time() - t0
            _publish_tool_event(tool_name, body, result=result, elapsed=elapsed)
            _record_tool_usage(tool_name, success=True, latency_ms=elapsed * 1000)
            return JSONResponse({"result": result})
        except Exception as e:
            elapsed = _time.time() - t0
            _publish_tool_event(tool_name, body, error=e)
            _record_tool_usage(tool_name, success=False, latency_ms=elapsed * 1000,
                               error_message=str(e))
            return JSONResponse({"error": f"Tool error: {e}"}, status_code=500)

    # Route 2: External MCP servers
    server_name = ext_mcp.get_server_for_tool(tool_name)
    if server_name:
        try:
            t0 = _time.time()
            result = await asyncio.to_thread(ext_mcp.call_tool, server_name, tool_name, body)
            elapsed = _time.time() - t0
            _publish_tool_event(tool_name, body, result=result, elapsed=elapsed)
            _record_mcp_usage(server_name, tool_name, success=True, latency_ms=elapsed * 1000)
            return JSONResponse({"result": result})
        except Exception as e:
            elapsed = _time.time() - t0
            _publish_tool_event(tool_name, body, error=e)
            _record_mcp_usage(server_name, tool_name, success=False,
                              latency_ms=elapsed * 1000, error_message=str(e))
            return JSONResponse({"error": f"External tool error: {e}"}, status_code=500)

    _publish_tool_event(tool_name, body, error="Unknown tool")
    return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=404)


@mcp.custom_route("/install", methods=["POST"])
async def install_handler(request: Request):
    """Install an external MCP server."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    source = body.get("source", "npm")
    package = body.get("package", "")
    repo_url = body.get("repo_url", "")
    name = body.get("name")

    if source == "npm" and package:
        result = ext_mcp.install_from_npm(package, server_name=name)
    elif source == "github" and repo_url:
        result = ext_mcp.install_from_github(repo_url, server_name=name)
    else:
        return JSONResponse({"error": "Provide 'package' (npm) or 'repo_url' (github)"}, status_code=400)

    return JSONResponse({"result": result})


@mcp.custom_route("/health", methods=["GET"])
async def health_handler(request: Request):
    """Health check — lists all tools from all sources."""
    tool_mgr.scan()
    managed_tools = list(tool_mgr.list_tools().keys())
    ext_tools = list(ext_mcp.list_all_tools().keys())
    all_tools = managed_tools + ext_tools
    return JSONResponse({
        "status": "ok",
        "tools": all_tools,
        "managed_tools": managed_tools,
        "external_tools": ext_tools,
    })


# ── Main ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("[MCP] Scanning tools...")
    tool_mgr.scan()

    n_managed = len(tool_mgr.list_tools())
    print(f"[MCP] {n_managed} managed tool(s)")

    ext_n = len(ext_mcp.list_all_tools())
    if ext_n:
        print(f"[MCP] {ext_n} external tool(s) registered (spawned on demand)")

    # Start health check loop
    tool_mgr.start_health_loop()

    from arqitect.config.loader import get_mcp_port, get_config
    _mcp_host = get_config("ports.mcp_host", "127.0.0.1")
    _mcp_port = get_mcp_port()
    print(f"[MCP] Starting on http://{_mcp_host}:{_mcp_port}")
    mcp.run(transport="http", host=_mcp_host, port=_mcp_port, log_level="warning")
