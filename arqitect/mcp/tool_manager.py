"""Tool Manager — LRU process pool for isolated tool execution.

Manages the lifecycle of tool processes: spawning, evicting, calling, health checking,
and pre-warming. Each tool runs as an isolated subprocess communicating via stdin/stdout
JSON-RPC, regardless of its language runtime.

The pool size is bounded (default 50) to keep memory usage under ~1.75GB.
Cold starts (200-500ms Python, 10ms Go) are negligible against LLM inference time.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from arqitect.config.loader import get_mcp_tools_dir
from arqitect.mcp.env_builder import build_env, env_ready

logger = logging.getLogger(__name__)

DEFAULT_MAX_POOL = 50
READY_TIMEOUT = 30
CALL_TIMEOUT = 30
HEALTH_CHECK_INTERVAL = 30


@dataclass
class ToolMeta:
    """Parsed tool.json metadata for a registered tool.

    Attributes:
        name: Tool name.
        description: Human-readable description.
        params: Parameter schema dict (name -> {type, description}).
        runtime: Runtime type (python, node, go, rust, binary, docker).
        entry: Entry point filename.
        timeout: Per-call timeout in seconds.
        version: Semantic version string.
        tool_dir: Absolute path to the tool directory.
    """
    name: str
    description: str
    params: dict
    runtime: str
    entry: str
    timeout: int
    version: str
    tool_dir: str


@dataclass
class ToolProcess:
    """A warm tool process in the pool.

    Attributes:
        name: Tool name.
        process: The subprocess.Popen instance.
        lock: Per-process lock for serializing calls.
    """
    name: str
    process: subprocess.Popen
    lock: threading.Lock = field(default_factory=threading.Lock)


class ToolManager:
    """LRU process pool managing all tool execution.

    Scans mcp_tools/ for tool.json manifests, maintains a bounded pool of
    warm processes, and handles spawning, eviction, health checks, and
    pre-warming.

    Args:
        max_pool: Maximum number of warm processes. Defaults to DEFAULT_MAX_POOL.
    """

    def __init__(self, max_pool: int = DEFAULT_MAX_POOL):
        self._pool: dict[str, ToolProcess] = {}
        self._lru: OrderedDict[str, float] = OrderedDict()
        self._registry: dict[str, ToolMeta] = {}
        self._max_pool = max_pool
        self._lock = threading.Lock()
        self._health_thread: threading.Thread | None = None
        self._shutdown = threading.Event()

    def scan(self) -> None:
        """Scan mcp_tools/ directory and populate the tool registry.

        Discovers directory-based tools with tool.json manifests.
        """
        tools_dir = str(get_mcp_tools_dir())
        if not os.path.isdir(tools_dir):
            return

        discovered = {}

        for entry in os.listdir(tools_dir):
            full_path = os.path.join(tools_dir, entry)

            # Directory-based tool with tool.json
            if os.path.isdir(full_path):
                meta = self._scan_tool_dir(full_path)
                if meta:
                    discovered[meta.name] = meta

        with self._lock:
            self._registry = discovered

        logger.info("[TOOL_MANAGER] Scanned %d tool(s)", len(discovered))

    def _scan_tool_dir(self, tool_dir: str) -> ToolMeta | None:
        """Parse a tool directory's tool.json into ToolMeta.

        Args:
            tool_dir: Path to the tool directory.

        Returns:
            ToolMeta if valid tool.json found, None otherwise.
        """
        manifest_path = os.path.join(tool_dir, "tool.json")
        if not os.path.isfile(manifest_path):
            return None

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        name = manifest.get("name", os.path.basename(tool_dir))
        return ToolMeta(
            name=name,
            description=manifest.get("description", f"Tool: {name}"),
            params=manifest.get("params", {}),
            runtime=manifest.get("runtime", "python"),
            entry=manifest.get("entry", "run.py"),
            timeout=manifest.get("timeout", CALL_TIMEOUT),
            version=manifest.get("version", "0.0.0"),
            tool_dir=tool_dir,
        )

    def call(self, tool_name: str, params: dict) -> str:
        """Call a tool by name with the given parameters.

        Ensures the tool process is warm (spawning if needed), sends a JSON-RPC
        request via stdin, and reads the response from stdout.

        Args:
            tool_name: Name of the tool to call.
            params: Parameter dict to pass to the tool.

        Returns:
            Result string from the tool.

        Raises:
            ValueError: If the tool is not registered.
            TimeoutError: If the tool does not respond within its timeout.
            RuntimeError: If the tool returns an error.
        """
        meta = self._registry.get(tool_name)
        if not meta:
            raise ValueError(f"Unknown tool: {tool_name}")

        tp = self._ensure_warm(tool_name)

        request_id = str(uuid.uuid4())[:8]
        request = {"id": request_id, "method": "call", "params": params}
        request_line = json.dumps(request) + "\n"

        with tp.lock:
            try:
                tp.process.stdin.write(request_line)
                tp.process.stdin.flush()

                # Read response with timeout
                response_line = self._read_with_timeout(tp, meta.timeout)
                response = json.loads(response_line)

                # Update LRU
                with self._lock:
                    self._lru.move_to_end(tool_name)
                    self._lru[tool_name] = time.time()

                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result", "")

            except (BrokenPipeError, OSError):
                # Process died — remove from pool and retry once
                self._remove_from_pool(tool_name)
                return self._retry_call(tool_name, params)

    def _retry_call(self, tool_name: str, params: dict) -> str:
        """Retry a failed call once by respawning the process.

        Args:
            tool_name: Name of the tool.
            params: Parameters for the call.

        Returns:
            Result string from the retried call.

        Raises:
            RuntimeError: If the retry also fails.
        """
        try:
            return self.call(tool_name, params)
        except Exception as e:
            raise RuntimeError(f"Tool '{tool_name}' failed after retry: {e}") from e

    def _read_with_timeout(self, tp: ToolProcess, timeout: int) -> str:
        """Read a line from a tool process stdout with a timeout.

        Args:
            tp: The warm tool process.
            timeout: Timeout in seconds.

        Returns:
            The response line string.

        Raises:
            TimeoutError: If no response within timeout.
        """
        result = [None]
        error = [None]

        def _read():
            try:
                result[0] = tp.process.stdout.readline()
            except Exception as e:
                error[0] = e

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(timeout=timeout)

        if reader.is_alive():
            raise TimeoutError(f"Tool '{tp.name}' timed out after {timeout}s")
        if error[0]:
            raise error[0]
        if not result[0]:
            raise RuntimeError(f"Tool '{tp.name}' returned empty response (process may have died)")

        return result[0].strip()

    def _ensure_warm(self, tool_name: str) -> ToolProcess:
        """Ensure a tool process is warm in the pool, spawning if needed.

        Evicts the LRU process if the pool is full.

        Args:
            tool_name: Name of the tool to warm up.

        Returns:
            The warm ToolProcess.
        """
        with self._lock:
            if tool_name in self._pool:
                tp = self._pool[tool_name]
                if tp.process.poll() is None:
                    return tp
                # Process died — clean up
                del self._pool[tool_name]
                self._lru.pop(tool_name, None)

            # Evict LRU if pool is full
            if len(self._pool) >= self._max_pool:
                self._evict_lru()

        tp = self._spawn(tool_name)

        with self._lock:
            self._pool[tool_name] = tp
            self._lru[tool_name] = time.time()

        return tp

    def _spawn(self, tool_name: str) -> ToolProcess:
        """Start a tool process based on its runtime configuration.

        Builds the appropriate command for the tool's runtime, starts the
        subprocess, and waits for the ready signal.

        Args:
            tool_name: Name of the tool to spawn.

        Returns:
            The spawned ToolProcess.

        Raises:
            RuntimeError: If the process fails to start or send ready signal.
        """
        meta = self._registry.get(tool_name)
        if not meta:
            raise RuntimeError(f"Tool '{tool_name}' not in registry")

        # Build environment on-demand if not ready
        if not env_ready(meta.tool_dir):
            logger.info("[TOOL_MANAGER] Building env for %s on first call", tool_name)
            build_env(meta.tool_dir)

        cmd = self._build_command(meta)
        logger.info("[TOOL_MANAGER] Spawning %s: %s", tool_name, " ".join(cmd))

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=meta.tool_dir,
            text=True,
            bufsize=1,
        )

        # Wait for ready signal
        try:
            ready_line = ""
            deadline = time.time() + READY_TIMEOUT
            while time.time() < deadline:
                if process.poll() is not None:
                    stderr = process.stderr.read()
                    raise RuntimeError(
                        f"Process exited during startup (code={process.returncode}): {stderr[:500]}"
                    )
                # Non-blocking readline with short timeout
                reader_result = [None]

                def _read_ready():
                    reader_result[0] = process.stdout.readline()

                t = threading.Thread(target=_read_ready, daemon=True)
                t.start()
                t.join(timeout=min(1.0, deadline - time.time()))

                if reader_result[0]:
                    ready_line = reader_result[0].strip()
                    break

            if not ready_line:
                process.kill()
                raise RuntimeError(f"Tool '{tool_name}' did not send ready signal within {READY_TIMEOUT}s")

            ready = json.loads(ready_line)
            if not ready.get("ready"):
                process.kill()
                raise RuntimeError(
                    f"Tool '{tool_name}' startup failed: {ready.get('error', 'unknown')}"
                )

        except json.JSONDecodeError:
            process.kill()
            raise RuntimeError(f"Tool '{tool_name}' sent invalid ready signal: {ready_line}")

        return ToolProcess(name=tool_name, process=process)

    def _build_command(self, meta: ToolMeta) -> list[str]:
        """Build the subprocess command for a tool based on its runtime.

        Args:
            meta: Tool metadata.

        Returns:
            Command list suitable for subprocess.Popen.
        """
        runtime = meta.runtime
        entry = meta.entry

        if runtime == "python":
            venv_python = os.path.join(meta.tool_dir, ".venv", "bin", "python")
            python = venv_python if os.path.isfile(venv_python) else sys.executable
            return [python, entry]

        if runtime == "node":
            return ["node", entry]

        if runtime in ("go", "rust", "binary"):
            run_path = os.path.join(meta.tool_dir, entry if entry != "run.py" else "run")
            return [run_path]

        if runtime == "docker":
            name = meta.name
            cmd = ["docker", "run", "-i", "--rm"]
            # Network isolation by default unless tool declares network access
            cmd.extend(["--network=none"])
            cmd.append(f"arqitect-tool-{name}")
            return cmd

        # Fallback: try to run the entry directly
        return [os.path.join(meta.tool_dir, entry)]

    def _evict_lru(self) -> None:
        """Evict the least recently used process from the pool.

        Sends SIGTERM and removes the process from pool and LRU tracking.
        Must be called while holding self._lock.
        """
        if not self._lru:
            return

        oldest_name = next(iter(self._lru))
        tp = self._pool.pop(oldest_name, None)
        self._lru.pop(oldest_name, None)

        if tp and tp.process.poll() is None:
            try:
                tp.process.terminate()
                tp.process.wait(timeout=5)
            except Exception:
                tp.process.kill()

        logger.info("[TOOL_MANAGER] Evicted LRU: %s", oldest_name)

    def _remove_from_pool(self, tool_name: str) -> None:
        """Remove a tool from the pool (e.g., after it crashes).

        Args:
            tool_name: Name of the tool to remove.
        """
        with self._lock:
            tp = self._pool.pop(tool_name, None)
            self._lru.pop(tool_name, None)
        if tp and tp.process.poll() is None:
            try:
                tp.process.terminate()
            except Exception:
                pass

    def pre_warm(self, count: int = 10) -> None:
        """Pre-warm the top N most-used tools from cold memory.

        Reads use_count from cold memory and spawns the top tools.

        Args:
            count: Number of tools to pre-warm.
        """
        try:
            from arqitect.brain.config import mem
            catalog = mem.cold.list_nerves()
            # Build a use-count ranking from cold memory
            tool_scores = {}
            for nerve in catalog:
                tools = mem.cold.get_nerve_tools(nerve)
                for tool_name in tools:
                    if tool_name in self._registry:
                        tool_scores[tool_name] = tool_scores.get(tool_name, 0) + 1

            top_tools = sorted(tool_scores, key=tool_scores.get, reverse=True)[:count]
            for tool_name in top_tools:
                try:
                    self._ensure_warm(tool_name)
                    logger.info("[TOOL_MANAGER] Pre-warmed: %s", tool_name)
                except Exception as e:
                    logger.warning("[TOOL_MANAGER] Failed to pre-warm %s: %s", tool_name, e)
        except Exception as e:
            logger.warning("[TOOL_MANAGER] Pre-warm skipped: %s", e)

    def restart_tool(self, name: str) -> None:
        """Stop a warm tool process and re-scan its metadata.

        The next call to this tool will spawn a fresh process.

        Args:
            name: Tool name to restart.
        """
        self._remove_from_pool(name)

        # Re-scan the single tool
        tools_dir = str(get_mcp_tools_dir())
        tool_dir = os.path.join(tools_dir, name)

        if os.path.isdir(tool_dir):
            meta = self._scan_tool_dir(tool_dir)
            if meta:
                with self._lock:
                    self._registry[meta.name] = meta

        logger.info("[TOOL_MANAGER] Restarted tool: %s", name)

    def restart_all(self) -> None:
        """Drain the entire pool and re-scan all tools."""
        with self._lock:
            for tp in self._pool.values():
                if tp.process.poll() is None:
                    try:
                        tp.process.terminate()
                        tp.process.wait(timeout=5)
                    except Exception:
                        tp.process.kill()
            self._pool.clear()
            self._lru.clear()

        self.scan()
        logger.info("[TOOL_MANAGER] Restarted all tools")

    def health_check(self) -> None:
        """Check all warm processes and remove dead ones."""
        with self._lock:
            dead = [
                name for name, tp in self._pool.items()
                if tp.process.poll() is not None
            ]

        for name in dead:
            self._remove_from_pool(name)
            logger.warning("[TOOL_MANAGER] Removed dead process: %s", name)

    def start_health_loop(self) -> None:
        """Start the background health check thread."""
        if self._health_thread and self._health_thread.is_alive():
            return

        def _loop():
            while not self._shutdown.is_set():
                self._shutdown.wait(timeout=HEALTH_CHECK_INTERVAL)
                if not self._shutdown.is_set():
                    self.health_check()

        self._health_thread = threading.Thread(target=_loop, daemon=True, name="tool-health")
        self._health_thread.start()

    def shutdown(self) -> None:
        """Shut down all tool processes and stop the health check loop."""
        self._shutdown.set()
        with self._lock:
            for tp in self._pool.values():
                if tp.process.poll() is None:
                    try:
                        tp.process.terminate()
                        tp.process.wait(timeout=3)
                    except Exception:
                        tp.process.kill()
            self._pool.clear()
            self._lru.clear()

    def list_tools(self) -> dict[str, dict]:
        """Return the tool registry as a dict suitable for API responses.

        Returns:
            Dict mapping tool name to {description, params}.
        """
        return {
            name: {
                "description": meta.description,
                "params": list(meta.params.keys()) if isinstance(meta.params, dict) else meta.params,
            }
            for name, meta in self._registry.items()
        }

    def get_meta(self, tool_name: str) -> ToolMeta | None:
        """Get metadata for a tool.

        Args:
            tool_name: Tool name.

        Returns:
            ToolMeta or None if not registered.
        """
        return self._registry.get(tool_name)
