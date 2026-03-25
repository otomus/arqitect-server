# Tools (MCP)

Tools are language-agnostic processes that speak JSON-RPC over stdio. Each tool runs in its own isolated subprocess. A crash in one tool never takes down another tool or the MCP server.

## What MCP Tools Are

MCP (Model Context Protocol) tools are structured integrations — the muscles and limbs of the nervous system. Unlike nerves (which are AI agents with LLMs), tools are deterministic functions: given the same input, they produce the same output.

Tools can be written in any language: Python, Node.js, Go, Rust, or run as Docker containers. The MCP server doesn't care what's inside — it only speaks JSON-RPC.

## The MCP Server

The MCP server (`arqitect/mcp/server.py`) is built on FastMCP and exposes both standard MCP protocol and HTTP endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/tools` | GET | List all available tools (managed + external) |
| `/call/{tool_name}` | POST | Call a tool with JSON body as arguments |
| `/install` | POST | Install an external MCP server (npm or GitHub) |
| `/health` | GET | Health check — lists all tools from all sources |

The server hot-reloads on every request: `tool_mgr.scan()` runs before listing or calling, so a tool fabricated by the brain is immediately available.

## Tool Discovery

Tools live in `mcp_tools/` as directories. Each tool directory contains:

- `tool.json` — the manifest (name, description, params, runtime, entry, timeout, version)
- Entry point file — `run.py`, `index.js`, `main.go`, or whatever the manifest declares

On startup (and on every `/tools` or `/call` request), the `ToolManager` scans `mcp_tools/` for directories containing a `tool.json` manifest. Each valid manifest is parsed into a `ToolMeta` object:

```json
{
  "name": "weather_tool",
  "description": "Fetch current weather for a location",
  "params": {"location": {"type": "string", "description": "City name"}},
  "runtime": "python",
  "entry": "run.py",
  "timeout": 30,
  "version": "1.0.0"
}
```

### Supported Runtimes

| Runtime | Command | Notes |
|---|---|---|
| `python` | Uses tool-local `.venv/bin/python` if available, else system Python | Dependencies installed per-tool |
| `node` | `node {entry}` | |
| `go` / `rust` / `binary` | Direct execution of compiled binary | |
| `docker` | `docker run -i --rm --network=none arqitect-tool-{name}` | Network isolated by default |

## Tool Runner

The tool runner (`arqitect/mcp/tool_runner.py`) bridges bare Python tool files to the JSON-RPC protocol. It's the subprocess entry point for Python tools.

### JSON-RPC Loop

```
Tool starts -> writes {"ready": true} to stdout
Server reads ready signal

For each request on stdin:
  -> {"id": "abc", "method": "call", "params": {"query": "NYC"}}
  <- {"id": "abc", "result": "72F, sunny"}
     or
  <- {"id": "abc", "error": "Missing required param: query"}
```

### Module Loading

The tool runner loads a Python file and finds the callable:

1. Look for a function named after the file (e.g., `weather_tool()` in `weather_tool.py`)
2. Fall back to `run()`
3. Raise `ValueError` if neither exists

The function is called with `**params` from the JSON-RPC request.

## Writing a Tool

A minimal Python tool:

```python
# mcp_tools/my_tool/run.py
def run(query: str) -> str:
    """Process a query and return a result."""
    return f"Result for: {query}"
```

And its manifest:

```json
{
  "name": "my_tool",
  "description": "Does something useful",
  "params": {"query": {"type": "string", "description": "The input query"}},
  "runtime": "python",
  "entry": "run.py"
}
```

### Conventions

- The function signature defines the tool's parameters. Parameter names must match what's declared in `tool.json`.
- Return a string. The MCP server converts non-string returns with `str()`.
- Handle errors gracefully — unhandled exceptions are caught and returned as `{"error": "..."}`.
- Keep tools focused. One tool, one job.
- Use only standard library + `requests` for fabricated tools (the brain's LLM generates code with this constraint).

## Tool Fabrication

The brain can generate new tools on the fly when a nerve needs a capability that doesn't exist yet.

### How It Works

1. The nerve's planner returns an `acquire` or `fabricate` action
2. `fabricate_mcp_tool(name, description, params)` in `arqitect/brain/synthesis.py` generates the code
3. The LLM writes a Python `run()` function with a docstring
4. The code is syntax-checked with `compile()` — up to 2 retries on syntax errors
5. The file is written to `mcp_tools/{name}.py`
6. The MCP server picks it up on the next `scan()`

### Validation

Before a fabricated tool is written to disk, `tool_validator.py` scans the generated code for quality issues:

- **Placeholder credentials** — `YOUR_API_KEY`, `DUMMY_*`, `INSERT_*_HERE`, `example.com`, and hardcoded `sk-...` API keys are rejected
- **Credential dependency detection** — `get_credential()` calls are scanned to build a schema of which services and keys the tool requires at runtime. The brain uses this to request credentials from the user before the tool runs.

If validation fails, the fabrication is retried. Tools that pass validation and use `get_credential()` automatically trigger the [credentials flow](/guide/bridge#credentials-flow) the first time they run.

### Constraints

- Fabricated tools must use only standard library + `requests`
- Maximum 30 lines
- Core MCP tools (`image_generator`) are never overwritten
- The generated code must contain a `run()` function with the correct signature

## Process Pool

The `ToolManager` maintains an LRU process pool (default max 50 processes, ~1.75GB memory budget).

### Lifecycle

1. **Spawn** — when a tool is called for the first time, a subprocess starts. The runner sends `{"ready": true}` on stdout. If the ready signal doesn't arrive within 30 seconds, the process is killed.
2. **Warm** — subsequent calls reuse the warm process. Requests are serialized per-process via a threading lock.
3. **Evict** — when the pool is full, the least recently used process receives SIGTERM. The next call to that tool spawns a fresh process.
4. **Health check** — a background thread runs every 30 seconds, removing dead processes.
5. **Pre-warm** — on startup, the top N most-used tools (by nerve association count from cold memory) are spawned ahead of time.

Cold starts are fast: 200-500ms for Python, ~10ms for Go. Negligible against LLM inference time.

### Crash Recovery

When a tool process dies mid-call (broken pipe), the manager:
1. Removes it from the pool
2. Spawns a fresh process
3. Retries the call once

If the retry also fails, a `RuntimeError` is raised.

## Usage Tracking

Every tool call is tracked in two places:

- **Monitoring database** (`MonitoringMemory`) — records tool name, success/failure, latency, and error messages. Best-effort, never blocks the call.
- **Redis events** — publishes to `mcp:tool_call` channel with tool name, args preview (truncated to 100 chars), elapsed time, and result preview or error. Used by the dashboard for real-time visibility.

Both are best-effort. A monitoring failure never causes a tool call to fail.

## External MCP Servers

The MCP server can connect to external MCP servers in addition to locally managed tools.

### Installation

External servers can be installed from npm or GitHub via the `/install` endpoint:

```json
{"source": "npm", "package": "@modelcontextprotocol/server-weather"}
```

```json
{"source": "github", "repo_url": "https://github.com/user/mcp-server"}
```

### Routing

When a tool call comes in, the MCP server checks:

1. **ToolManager registry** first (local subprocess-based tools)
2. **External MCP servers** as fallback

External tools appear in the `/tools` listing with `"source": "external"` and their server name.

### Lifecycle

External MCP servers are managed by `ExternalMCPManager`. All external servers are shut down on process exit via `atexit`.

## Argument Remapping

Small LLMs often use wrong parameter names when calling tools. The MCP server handles this:

- If the model's arg keys don't match the tool's expected params, values are remapped positionally
- For single-parameter tools, the first provided value (or user input) is mapped to the expected param name

::: tip Related
- [Nerves](/guide/nerves) — how nerves discover and call tools
- [Architecture Overview](/architecture/overview) — tools as muscles and limbs
:::
