"""
External MCP Server Manager — manages lifecycle of external MCP servers.
Handles install (npm/GitHub), spawn (stdio transport), JSON-RPC communication,
and persistent registry tracking.
"""

import json
import os
import subprocess
import sys
import threading
import time

from arqitect.config.loader import get_project_root

REGISTRY_PATH = os.path.join(str(get_project_root()), "mcp_registry.json")


class ExternalMCPManager:
    """Manages external MCP servers: install, spawn, communicate, shutdown."""

    def __init__(self):
        self._processes = {}  # server_name -> subprocess.Popen
        self._initialized = set()  # server names that completed MCP handshake
        self._locks = {}  # server_name -> threading.Lock (serializes stdio access)
        self._registry = self._load_registry()

    # ── Registry Persistence ─────────────────────────────────────────────

    def _load_registry(self) -> dict:
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "r") as f:
                return json.load(f)
        # No local registry — seed from community manifest
        registry = self._seed_from_community()
        return registry

    def _seed_from_community(self) -> dict:
        """Fetch the MCP server catalog from arqitect-community and seed the local registry."""
        registry = {"servers": {}}
        manifest = None
        try:
            from create_arqitect.registry.manifest import get_manifest
            manifest = get_manifest()
        except ImportError:
            # create_arqitect not installed (e.g. scaffolded server) — fetch directly
            try:
                import requests
                url = "https://raw.githubusercontent.com/otomus/sentient-community/main/manifest.json"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                manifest = resp.json()
            except Exception as e:
                print(f"[EXT-MCP] Could not fetch community manifest: {e}")
        if not manifest:
            return registry

        # Manifest uses "mcps" key for MCP servers
        mcp_servers = manifest.get("mcps", {}) or manifest.get("mcp_servers", {})
        for name, info in mcp_servers.items():
            # Only seed npm-based servers (no hardcoded paths)
            if info.get("source") != "npm":
                continue
            registry["servers"][name] = {
                "source": "npm",
                "package": info["package"],
                "command": info.get("command", ["npx", "-y", info["package"]]),
                "auth_type": info.get("auth_type", "none"),
                "tools": info.get("tools", []),
                "capabilities": info.get("capabilities", []),
            }
        if registry["servers"]:
            # Persist so we don't re-fetch every startup
            with open(REGISTRY_PATH, "w") as f:
                json.dump(registry, f, indent=2)
            print(f"[EXT-MCP] Seeded {len(registry['servers'])} servers from community registry")
        return registry

    def _save_registry(self):
        with open(REGISTRY_PATH, "w") as f:
            json.dump(self._registry, f, indent=2)

    # ── Installation ─────────────────────────────────────────────────────

    def install_from_npm(self, package_name: str, server_name: str | None = None) -> str:
        """Install an MCP server from npm. Uses npx to run it (no global install needed)."""
        name = server_name or package_name.split("/")[-1]

        if name in self._registry["servers"]:
            return f"Server '{name}' already installed"

        # Verify npx is available
        try:
            subprocess.run(["npx", "--version"], capture_output=True, timeout=10, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "Error: npx not found. Install Node.js to use npm MCP servers."

        # Record in registry — npx handles on-demand installation
        self._registry["servers"][name] = {
            "source": "npm",
            "package": package_name,
            "command": ["npx", "-y", package_name],
            "installed_at": time.strftime("%Y-%m-%d"),
            "tools": [],
        }
        self._save_registry()

        # Try to discover tools by spawning briefly
        tools = self.list_tools(name)
        if tools:
            self._registry["servers"][name]["tools"] = [t["name"] for t in tools]
            self._save_registry()
            self.shutdown(name)

        return f"Installed MCP server '{name}' from npm ({package_name})"

    def install_from_github(self, repo_url: str, server_name: str | None = None) -> str:
        """Clone and install an MCP server from GitHub."""
        name = server_name or repo_url.rstrip("/").split("/")[-1].removesuffix(".git")

        if name in self._registry["servers"]:
            return f"Server '{name}' already installed"

        install_dir = os.path.join(str(get_project_root()), "mcp_repos", name)
        os.makedirs(install_dir, exist_ok=True)

        try:
            # Clone
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, install_dir],
                capture_output=True, text=True, timeout=60, check=True,
            )

            # Detect project type and install
            command = None
            if os.path.exists(os.path.join(install_dir, "package.json")):
                subprocess.run(
                    ["npm", "install"], cwd=install_dir,
                    capture_output=True, timeout=120, check=True,
                )
                # Check for common entry points
                pkg = json.load(open(os.path.join(install_dir, "package.json")))
                bin_entry = pkg.get("bin")
                if isinstance(bin_entry, str):
                    command = ["node", os.path.join(install_dir, bin_entry)]
                elif isinstance(bin_entry, dict):
                    first = next(iter(bin_entry.values()))
                    command = ["node", os.path.join(install_dir, first)]
                else:
                    main = pkg.get("main", "index.js")
                    command = ["node", os.path.join(install_dir, main)]

            elif os.path.exists(os.path.join(install_dir, "pyproject.toml")) or \
                 os.path.exists(os.path.join(install_dir, "setup.py")):
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", install_dir],
                    capture_output=True, timeout=120, check=True,
                )
                command = [sys.executable, "-m", name.replace("-", "_")]

            if not command:
                command = ["node", os.path.join(install_dir, "index.js")]

            self._registry["servers"][name] = {
                "source": "github",
                "repo": repo_url,
                "path": install_dir,
                "command": command,
                "installed_at": time.strftime("%Y-%m-%d"),
                "tools": [],
            }
            self._save_registry()

            # Discover tools
            tools = self.list_tools(name)
            if tools:
                self._registry["servers"][name]["tools"] = [t["name"] for t in tools]
                self._save_registry()
                self.shutdown(name)

            return f"Installed MCP server '{name}' from GitHub ({repo_url})"

        except Exception as e:
            return f"Error installing from GitHub: {e}"

    # ── Server Lifecycle ─────────────────────────────────────────────────

    def spawn_server(self, server_name: str) -> bool:
        """Start an external MCP server subprocess (stdio transport)."""
        if server_name in self._processes:
            proc = self._processes[server_name]
            if proc.poll() is None:
                return True  # Already running
            # Process died — clean up stale state
            self._processes.pop(server_name, None)
            self._initialized.discard(server_name)

        server = self._registry["servers"].get(server_name)
        if not server:
            print(f"[EXT-MCP] Server '{server_name}' not in registry")
            return False

        command = server["command"]
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._processes[server_name] = proc
            time.sleep(1)  # Give server time to initialize
            if proc.poll() is not None:
                stderr = proc.stderr.read()
                print(f"[EXT-MCP] Server '{server_name}' exited immediately: {stderr}")
                self._processes.pop(server_name, None)
                return False
            print(f"[EXT-MCP] Spawned server '{server_name}' (PID {proc.pid})")
            return True
        except Exception as e:
            print(f"[EXT-MCP] Failed to spawn '{server_name}': {e}")
            return False

    def _server_lock(self, server_name: str) -> threading.RLock:
        """Get or create a per-server lock for serializing stdio access."""
        if server_name not in self._locks:
            self._locks[server_name] = threading.RLock()
        return self._locks[server_name]

    def _ensure_connection(self, server_name: str) -> bool:
        """Ensure a server is spawned and has completed the MCP initialize handshake."""
        if not self.spawn_server(server_name):
            return False
        if server_name not in self._initialized:
            init_resp = self._send_jsonrpc(server_name, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "synapse", "version": "1.0"},
            }, _skip_ensure=True)
            if init_resp and "result" in init_resp:
                self._send_jsonrpc(server_name, "notifications/initialized", _skip_ensure=True)
                self._initialized.add(server_name)
            else:
                print(f"[EXT-MCP] Failed to initialize '{server_name}'")
                return False
        return True

    def _send_jsonrpc(self, server_name: str, method: str, params: dict | None = None,
                      timeout: float = 15, _skip_ensure: bool = False) -> dict | None:
        """Send a JSON-RPC request over stdio to a running MCP server."""
        with self._server_lock(server_name):
            if _skip_ensure:
                # Called from _ensure_connection itself — just check process is alive
                if server_name not in self._processes or self._processes[server_name].poll() is not None:
                    return None
            elif not self._ensure_connection(server_name):
                return None

            proc = self._processes[server_name]
            request = {
                "jsonrpc": "2.0",
                "id": int(time.time() * 1000),
                "method": method,
            }
            if params:
                request["params"] = params

            try:
                message = json.dumps(request)
                # MCP stdio uses Content-Length header framing
                header = f"Content-Length: {len(message)}\r\n\r\n"
                proc.stdin.write(header + message)
                proc.stdin.flush()

                # Read response with Content-Length framing and timeout
                import selectors
                sel = selectors.DefaultSelector()
                sel.register(proc.stdout, selectors.EVENT_READ)

                content_length = 0
                while True:
                    ready = sel.select(timeout=timeout)
                    if not ready:
                        print(f"[EXT-MCP] Timeout ({timeout}s) waiting for '{server_name}' ({method})")
                        sel.close()
                        return None
                    line = proc.stdout.readline()
                    if not line:
                        sel.close()
                        return None
                    line = line.strip()
                    if line.startswith("Content-Length:"):
                        try:
                            content_length = int(line.split(":")[1].strip())
                        except (ValueError, IndexError):
                            print(f"[EXT-MCP] Malformed Content-Length header: {line}")
                            sel.close()
                            return None
                    elif line == "":
                        # Empty line after headers — read body
                        if content_length <= 0:
                            sel.close()
                            return None
                        body = proc.stdout.read(content_length)
                        sel.close()
                        return json.loads(body)
            except BrokenPipeError:
                print(f"[EXT-MCP] Broken pipe to '{server_name}' — killing and will respawn on next call")
                stale = self._processes.pop(server_name, None)
                self._initialized.discard(server_name)
                if stale:
                    try:
                        stale.kill()
                        stale.wait(timeout=3)
                    except Exception:
                        pass
                return None
            except Exception as e:
                print(f"[EXT-MCP] JSON-RPC error ({server_name}.{method}): {e}")
                return None

    def list_tools(self, server_name: str) -> list[dict]:
        """List tools from an external MCP server via JSON-RPC."""
        resp = self._send_jsonrpc(server_name, "tools/list")
        if resp and "result" in resp:
            return resp["result"].get("tools", [])
        return []

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """Call a tool on an external MCP server via JSON-RPC."""
        resp = self._send_jsonrpc(server_name, "tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        if resp and "result" in resp:
            content = resp["result"].get("content", [])
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(texts) if texts else json.dumps(resp["result"])
        if resp and "error" in resp:
            return f"Tool error: {resp['error'].get('message', resp['error'])}"
        return "Error: no response from external MCP server"

    def list_all_tools(self) -> dict:
        """List all tools across all installed external MCP servers.
        Returns {tool_name: {"server": server_name, "description": str, "params": list}}
        """
        all_tools = {}
        for name, server in self._registry["servers"].items():
            # Use cached tools from registry first
            for tool_name in server.get("tools", []):
                all_tools[tool_name] = {
                    "server": name,
                    "description": f"Tool from external MCP server '{name}'",
                    "params": [],
                    "source": "external",
                }
        return all_tools

    def get_server_for_tool(self, tool_name: str) -> str | None:
        """Find which external server provides a given tool."""
        for name, server in self._registry["servers"].items():
            if tool_name in server.get("tools", []):
                return name
        return None

    # ── Shutdown ─────────────────────────────────────────────────────────

    def shutdown(self, server_name: str):
        """Gracefully stop a running external MCP server."""
        proc = self._processes.pop(server_name, None)
        self._initialized.discard(server_name)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"[EXT-MCP] Stopped server '{server_name}'")

    def shutdown_all(self):
        """Stop all running external MCP servers."""
        for name in list(self._processes.keys()):
            self.shutdown(name)

    def get_registry(self) -> dict:
        """Return the current registry data."""
        return self._registry
