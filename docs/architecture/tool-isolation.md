# Tool Isolation

Tools are muscles. They do the physical work — call APIs, read files, query databases. But muscles don't share blood supply, and tools don't share process space.

Every MCP tool runs in its own isolated subprocess. If a tool crashes, nothing else dies.

<ToolIsolationDiagram />

## Why Isolation

Three reasons. All non-negotiable.

**Crash safety.** A tool that segfaults, hangs, or leaks memory gets killed. The nerve that called it gets an error response. The brain re-thinks. Nothing else is affected.

**Language-agnostic.** Tools are Python functions today, but the MCP protocol doesn't care. JSON-RPC over stdio means any language that can read stdin and write stdout can be a tool. The subprocess boundary is the contract.

**Security.** Tools run with minimal permissions. They can't access the brain's memory, other tools' state, or the nerve's context. The sandbox is the isolation — not a policy document, but a process boundary.

## The MCP Protocol

Tools speak [JSON-RPC 2.0](https://www.jsonrpc.org/specification) over stdio. That's it. No HTTP, no sockets, no shared memory.

The flow:

1. Nerve decides it needs a tool
2. Brain sends a JSON-RPC request to the MCP server's stdin
3. MCP server routes the request to the right tool handler
4. Tool handler spawns an isolated subprocess
5. Subprocess executes, writes JSON-RPC response to stdout
6. MCP server reads the response, routes it back to the nerve

Request and response are both typed. The MCP server validates schemas on both ends.

## Discovery and Registration

Tools are discovered at startup. The MCP server scans the tools directory, loads each module, and registers every function decorated with `@mcp.tool()`. Each tool declares:

- **Name** — unique identifier, used in JSON-RPC method calls
- **Description** — what the tool does, used by the brain for matching
- **Input schema** — JSON Schema for parameters
- **Output schema** — JSON Schema for return value

The brain's catalog knows every registered tool. When synthesizing a nerve, it pre-seeds the nerve with tools that match its domain — a weather nerve gets the weather tool, not the database tool.

## Subprocess Lifecycle

When a tool is invoked:

1. The MCP server forks a new subprocess
2. The tool function runs in that subprocess with its own memory space
3. Stdout is captured as the JSON-RPC response
4. The subprocess exits — clean termination, no lingering state
5. If the subprocess hangs, a timeout kills it and returns an error

No connection pooling. No warm processes. Every invocation is fresh. This is intentional — simplicity over performance. Tools are I/O-bound anyway.

## Tool Fabrication

The brain can create new tools at runtime. When a nerve needs a capability that doesn't exist:

1. The brain generates Python code via the LLM
2. Syntax is validated before writing to disk
3. The tool is registered with the MCP server
4. Retries on generation errors (up to 3 attempts)

Core tools are protected — fabrication can't overwrite built-in tools. Fabricated tools go through the same isolation path as any other tool.

## Usage Tracking

Every tool invocation is tracked:

- Call count
- Success/failure rate
- Latency (p50, p95)
- Which nerves use which tools

This data flows two ways:

1. **Monitoring** — metrics are recorded for dashboard visibility
2. **Redis** — events are published for real-time subscribers

Both paths are best-effort. If Redis is down or monitoring fails, the tool call still succeeds. Observability never blocks execution.

The brain uses usage stats during dream state to identify underperforming tools and wire better alternatives into nerves.

---

For implementation details — writing tools, the `@mcp.tool()` decorator, testing tools locally — see the [Tools Guide](/guide/tools).
