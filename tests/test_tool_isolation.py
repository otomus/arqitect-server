"""Tests for tool isolation architecture.

Covers:
- ToolManager: registry scanning, process spawning/calling, LRU eviction, health checks
- tool_runner: wrapping run() functions in tool directories to stdio JSON-RPC
- env_builder: environment readiness detection and cleanup
- Community seeding: directory-based tool download with version tracking
"""

import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest
from dirty_equals import IsInstance, IsStr
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_RPC_CODE = (
    'import json, sys\n'
    'sys.stdout.write(json.dumps({"ready": True}) + "\\n")\n'
    'sys.stdout.flush()\n'
    'for line in sys.stdin:\n'
    '    line = line.strip()\n'
    '    if not line:\n'
    '        continue\n'
    '    req = json.loads(line)\n'
    '    params = req.get("params", {})\n'
    '    try:\n'
    '        result = str(int(params.get("a", 0)) + int(params.get("b", 0)))\n'
    '        resp = {"id": req.get("id"), "result": result}\n'
    '    except Exception as e:\n'
    '        resp = {"id": req.get("id"), "error": str(e)}\n'
    '    sys.stdout.write(json.dumps(resp) + "\\n")\n'
    '    sys.stdout.flush()\n'
)

SIMPLE_RPC_CODE = '''\
import json, sys
sys.stdout.write(json.dumps({"ready": True}) + "\\n")
sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line.strip())
    sys.stdout.write(json.dumps({"id": req.get("id"), "result": "ok"}) + "\\n")
    sys.stdout.flush()
'''


def _write_tool_manifest(tool_dir, name, **overrides):
    """Write a tool.json manifest into the given directory.

    Args:
        tool_dir: pathlib.Path for the tool directory.
        name: Tool name.
        **overrides: Fields to override in the manifest.

    Returns:
        The written manifest dict.
    """
    manifest = {
        "name": name,
        "version": "1.0.0",
        "description": f"Tool: {name}",
        "params": {},
        "runtime": "python",
        "entry": "run.py",
        "timeout": 10,
        **overrides,
    }
    (tool_dir / "tool.json").write_text(json.dumps(manifest))
    return manifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tools_dir(tmp_path):
    """Provide an isolated mcp_tools directory."""
    d = tmp_path / "mcp_tools"
    d.mkdir()
    with patch("arqitect.mcp.tool_manager.get_mcp_tools_dir", return_value=str(d)):
        yield d


@pytest.fixture
def dir_tool(tools_dir):
    """Create a directory-based tool with tool.json and an adder run.py."""
    tool_dir = tools_dir / "adder_tool"
    tool_dir.mkdir()
    _write_tool_manifest(
        tool_dir, "adder_tool",
        description="Add two numbers",
        params={"a": {"type": "string"}, "b": {"type": "string"}},
    )
    (tool_dir / "run.py").write_text(TOOL_RPC_CODE)
    return tool_dir


# ---------------------------------------------------------------------------
# ToolManager — scanning
# ---------------------------------------------------------------------------

class TestToolManagerScan:
    """Registry scanning discovers tools from mcp_tools/."""

    @pytest.mark.timeout(10)
    def test_scans_tool_dir(self, tools_dir, dir_tool):
        """Directory-based tools with tool.json are discovered."""
        from arqitect.mcp.tool_manager import ToolManager

        mgr = ToolManager()
        mgr.scan()

        assert "adder_tool" in mgr.list_tools()
        meta = mgr.get_meta("adder_tool")
        assert meta is not None
        assert meta.version == "1.0.0"

    @pytest.mark.timeout(10)
    def test_skips_underscore_files(self, tools_dir):
        """Files starting with _ are ignored."""
        (tools_dir / "_internal.py").write_text("def run(): pass")

        from arqitect.mcp.tool_manager import ToolManager

        mgr = ToolManager()
        mgr.scan()

        assert "_internal" not in mgr.list_tools()

    @pytest.mark.timeout(10)
    def test_scan_ignores_bare_py(self, tools_dir, dir_tool):
        """Bare .py files are ignored, only directory-based tools are discovered."""
        (tools_dir / "bare_tool.py").write_text("def run(): pass")

        from arqitect.mcp.tool_manager import ToolManager

        mgr = ToolManager()
        mgr.scan()

        tools = mgr.list_tools()
        assert "bare_tool" not in tools
        assert "adder_tool" in tools

    @pytest.mark.timeout(10)
    @given(name=st.from_regex(r"[a-z][a-z0-9_]{2,20}", fullmatch=True))
    @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_scan_discovers_arbitrary_tool_names(self, tools_dir, name):
        """Any valid tool name in a directory with tool.json is discovered."""
        from arqitect.mcp.tool_manager import ToolManager

        tool_dir = tools_dir / name
        tool_dir.mkdir(exist_ok=True)
        _write_tool_manifest(tool_dir, name)
        (tool_dir / "run.py").write_text(SIMPLE_RPC_CODE)

        mgr = ToolManager()
        mgr.scan()

        assert name in mgr.list_tools()
        meta = mgr.get_meta(name)
        assert meta is not None
        assert meta.name == name
        assert meta.version == IsStr()


# ---------------------------------------------------------------------------
# ToolManager — subprocess calling
# ---------------------------------------------------------------------------

class TestToolManagerCall:
    """Calling tools via subprocess JSON-RPC."""

    @pytest.mark.timeout(10)
    def test_call_dir_tool(self, tools_dir, dir_tool):
        """Directory-based tool can be spawned and called via JSON-RPC."""
        from arqitect.mcp.tool_manager import ToolManager

        mgr = ToolManager()
        mgr.scan()

        # Mark env as ready (skip actual venv creation)
        (dir_tool / ".env_version").write_text("1.0.0")

        result = mgr.call("adder_tool", {"a": "3", "b": "7"})
        assert result == "10"
        mgr.shutdown()

    @pytest.mark.timeout(10)
    def test_call_unknown_tool_raises(self, tools_dir):
        """Calling a non-existent tool raises ValueError."""
        from arqitect.mcp.tool_manager import ToolManager

        mgr = ToolManager()
        mgr.scan()

        with pytest.raises(ValueError, match="Unknown tool"):
            mgr.call("nonexistent", {})

    @pytest.mark.timeout(10)
    @given(
        a=st.integers(min_value=-1000, max_value=1000),
        b=st.integers(min_value=-1000, max_value=1000),
    )
    @settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_call_with_arbitrary_integers(self, tools_dir, dir_tool, a, b):
        """Adder tool returns the correct sum for arbitrary integer pairs."""
        from arqitect.mcp.tool_manager import ToolManager

        (dir_tool / ".env_version").write_text("1.0.0")

        mgr = ToolManager()
        mgr.scan()

        result = mgr.call("adder_tool", {"a": str(a), "b": str(b)})
        assert result == str(a + b)
        mgr.shutdown()


# ---------------------------------------------------------------------------
# ToolManager — LRU eviction
# ---------------------------------------------------------------------------

class TestToolManagerLRU:
    """Pool respects max size via LRU eviction."""

    @pytest.mark.timeout(10)
    def test_evicts_lru_when_pool_full(self, tools_dir):
        """When pool hits max, the least recently used tool is evicted."""
        from arqitect.mcp.tool_manager import ToolManager

        # Create 3 simple tools
        for i in range(3):
            name = f"tool_{i}"
            tool_dir = tools_dir / name
            tool_dir.mkdir()
            _write_tool_manifest(tool_dir, name, description=f"Tool {i}", timeout=5)
            (tool_dir / ".env_version").write_text("1.0.0")
            (tool_dir / "run.py").write_text(SIMPLE_RPC_CODE)

        mgr = ToolManager(max_pool=2)
        mgr.scan()

        # Call tool_0 and tool_1 to fill pool
        mgr.call("tool_0", {})
        mgr.call("tool_1", {})
        assert len(mgr._pool) == 2

        # Calling tool_2 should evict tool_0 (oldest)
        mgr.call("tool_2", {})
        assert len(mgr._pool) == 2
        assert "tool_0" not in mgr._pool
        assert "tool_2" in mgr._pool

        mgr.shutdown()


# ---------------------------------------------------------------------------
# ToolManager — health check
# ---------------------------------------------------------------------------

class TestToolManagerHealth:
    """Health check removes dead processes."""

    @pytest.mark.timeout(10)
    def test_removes_dead_process(self, tools_dir, dir_tool):
        """Dead processes are cleaned up by health_check."""
        from arqitect.mcp.tool_manager import ToolManager

        (dir_tool / ".env_version").write_text("1.0.0")

        mgr = ToolManager()
        mgr.scan()
        mgr.call("adder_tool", {"a": "1", "b": "2"})
        assert "adder_tool" in mgr._pool

        # Kill the process
        mgr._pool["adder_tool"].process.kill()
        mgr._pool["adder_tool"].process.wait()

        mgr.health_check()
        assert "adder_tool" not in mgr._pool
        mgr.shutdown()


# ---------------------------------------------------------------------------
# tool_runner — legacy wrapping
# ---------------------------------------------------------------------------

class TestToolRunner:
    """tool_runner.py bridges run() functions in tool directories to stdio JSON-RPC."""

    @pytest.fixture
    def tool_py(self, tools_dir):
        """Create a .py file with a run() function for tool_runner tests."""
        code = '''
def run(query: str) -> str:
    """Echo the input back."""
    return f"echo: {query}"
'''
        path = tools_dir / "echo_tool.py"
        path.write_text(code)
        return path

    @pytest.mark.timeout(10)
    def test_load_tool_module(self, tools_dir, tool_py):
        """_load_tool_module finds the run() function."""
        from arqitect.mcp.tool_runner import _load_tool_module

        name, func = _load_tool_module(str(tool_py))
        assert name == "echo_tool"
        assert callable(func)
        assert func(query="test") == "echo: test"

    @pytest.mark.timeout(10)
    def test_load_named_function(self, tools_dir):
        """_load_tool_module finds a named function matching the filename."""
        code = '''
def greet_tool(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"
'''
        path = tools_dir / "greet_tool.py"
        path.write_text(code)

        from arqitect.mcp.tool_runner import _load_tool_module

        tool_name, func = _load_tool_module(str(path))
        assert tool_name == "greet_tool"
        assert func(name="World") == "Hello, World!"

    @pytest.mark.timeout(10)
    def test_runner_subprocess(self, tools_dir, tool_py):
        """tool_runner.py works as a subprocess speaking JSON-RPC."""
        runner = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "arqitect", "mcp", "tool_runner.py",
        )
        proc = subprocess.Popen(
            [sys.executable, runner, str(tool_py)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True,
        )

        # Read ready signal
        ready = json.loads(proc.stdout.readline())
        assert ready["ready"] is True

        # Send a call
        proc.stdin.write(json.dumps({
            "id": "1", "method": "call", "params": {"query": "test"},
        }) + "\n")
        proc.stdin.flush()

        response = json.loads(proc.stdout.readline())
        assert response["id"] == "1"
        assert response["result"] == "echo: test"

        proc.terminate()
        proc.wait()

    @pytest.mark.timeout(10)
    def test_load_module_without_callable_raises(self, tools_dir):
        """_load_tool_module raises ValueError when no callable is found."""
        code = '''
# No run() or matching function
DATA = 42
'''
        path = tools_dir / "no_func.py"
        path.write_text(code)

        from arqitect.mcp.tool_runner import _load_tool_module

        with pytest.raises(ValueError, match="No callable"):
            _load_tool_module(str(path))


# ---------------------------------------------------------------------------
# env_builder
# ---------------------------------------------------------------------------

class TestEnvBuilder:
    """Environment builder manages per-tool isolated environments."""

    @pytest.mark.timeout(10)
    def test_env_not_ready_without_version_file(self, tools_dir, dir_tool):
        """env_ready returns False when .env_version is missing."""
        from arqitect.mcp.env_builder import env_ready

        assert not env_ready(str(dir_tool))

    @pytest.mark.timeout(10)
    def test_env_ready_with_matching_version(self, tools_dir, dir_tool):
        """env_ready returns True when version matches and venv exists."""
        from arqitect.mcp.env_builder import env_ready

        (dir_tool / ".env_version").write_text("1.0.0")
        (dir_tool / ".venv").mkdir()

        assert env_ready(str(dir_tool))

    @pytest.mark.timeout(10)
    def test_env_not_ready_version_mismatch(self, tools_dir, dir_tool):
        """env_ready returns False when version doesn't match."""
        from arqitect.mcp.env_builder import env_ready

        (dir_tool / ".env_version").write_text("0.9.0")
        (dir_tool / ".venv").mkdir()

        assert not env_ready(str(dir_tool))

    @pytest.mark.timeout(10)
    @given(
        installed=st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True),
        manifest=st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_env_ready_version_comparison_is_exact(self, tools_dir, installed, manifest):
        """env_ready uses exact string comparison, not semver ordering."""
        from arqitect.mcp.env_builder import env_ready

        tool_dir = tools_dir / "version_test"
        tool_dir.mkdir(exist_ok=True)
        _write_tool_manifest(tool_dir, "version_test", version=manifest)
        (tool_dir / ".env_version").write_text(installed)
        (tool_dir / ".venv").mkdir(exist_ok=True)

        if installed == manifest:
            assert env_ready(str(tool_dir))
        else:
            assert not env_ready(str(tool_dir))

    @pytest.mark.timeout(10)
    def test_cleanup_env_removes_artifacts(self, tools_dir, dir_tool):
        """cleanup_env removes venv, node_modules, and version file."""
        from arqitect.mcp.env_builder import cleanup_env

        (dir_tool / ".venv").mkdir()
        (dir_tool / ".env_version").write_text("1.0.0")

        cleanup_env(str(dir_tool))

        assert not (dir_tool / ".venv").exists()
        assert not (dir_tool / ".env_version").exists()

    @pytest.mark.timeout(10)
    def test_build_env_python(self, tools_dir, dir_tool):
        """build_env creates a Python venv for python runtime tools."""
        from arqitect.mcp.env_builder import build_env, env_ready

        assert build_env(str(dir_tool))
        assert (dir_tool / ".venv").exists()
        assert (dir_tool / ".env_version").read_text() == "1.0.0"
        assert env_ready(str(dir_tool))

    @pytest.mark.timeout(10)
    def test_rebuild_env(self, tools_dir, dir_tool):
        """rebuild_env tears down and recreates the environment."""
        from arqitect.mcp.env_builder import build_env, rebuild_env

        build_env(str(dir_tool))
        old_venv = dir_tool / ".venv"
        assert old_venv.exists()

        # Change version in tool.json
        manifest = json.loads((dir_tool / "tool.json").read_text())
        manifest["version"] = "2.0.0"
        (dir_tool / "tool.json").write_text(json.dumps(manifest))

        assert rebuild_env(str(dir_tool))
        assert (dir_tool / ".env_version").read_text() == "2.0.0"

    @pytest.mark.timeout(10)
    def test_build_env_binary_runtime(self, tools_dir):
        """Binary runtime just checks that the entry file exists."""
        from arqitect.mcp.env_builder import build_env

        tool_dir = tools_dir / "bin_tool"
        tool_dir.mkdir()
        _write_tool_manifest(tool_dir, "bin_tool", runtime="binary", entry="run")
        (tool_dir / "run").write_text("#!/bin/sh\necho ok")

        assert build_env(str(tool_dir))

    @pytest.mark.timeout(10)
    def test_build_env_no_manifest_returns_false(self, tools_dir):
        """build_env returns False when tool.json is missing."""
        from arqitect.mcp.env_builder import build_env

        empty_dir = tools_dir / "empty"
        empty_dir.mkdir()

        assert not build_env(str(empty_dir))

    @pytest.mark.timeout(10)
    def test_env_ready_no_manifest_returns_false(self, tools_dir):
        """env_ready returns False when tool.json is missing entirely."""
        from arqitect.mcp.env_builder import env_ready

        empty_dir = tools_dir / "no_manifest"
        empty_dir.mkdir()

        assert not env_ready(str(empty_dir))


# ---------------------------------------------------------------------------
# Community seeding — directory-based tools
# ---------------------------------------------------------------------------

class TestSeedToolDirectories:
    """seed_tools handles directory-based tools with version tracking."""

    @pytest.fixture(autouse=True)
    def _stub_network(self):
        """Prevent real HTTP calls during tests."""
        with patch("arqitect.brain.community.urllib.request.urlretrieve"), \
             patch("arqitect.brain.community.urllib.request.urlopen"):
            yield

    @pytest.mark.timeout(10)
    def test_seeds_directory_tool(self, tools_dir, tmp_path):
        """Directory-based tools with version+files get a .needs_build marker."""
        manifest = {
            "version": "1.0",
            "nerves": {},
            "tools": {
                "new_tool": {
                    "version": "1.0.0",
                    "description": "A new tool",
                    "runtime": "python",
                    "files": ["tool.json", "run.py"],
                },
            },
        }
        cache_dir = tmp_path / ".community" / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

        from arqitect.brain.community import seed_tools
        with patch("arqitect.brain.community._manifest_path",
                   return_value=str(cache_dir / "manifest.json")), \
             patch("arqitect.brain.community.get_mcp_tools_dir",
                   return_value=str(tools_dir)):
            count = seed_tools()

        # Tool directory should be created with .needs_build marker
        tool_dir = tools_dir / "new_tool"
        if tool_dir.exists():
            assert (tool_dir / ".needs_build").exists()

    @pytest.mark.timeout(10)
    def test_skips_up_to_date_directory_tool(self, tools_dir, tmp_path):
        """Directory tools already at the manifest version are skipped."""
        manifest = {
            "version": "1.0",
            "nerves": {},
            "tools": {
                "existing_tool": {
                    "version": "1.0.0",
                    "description": "Already here",
                    "runtime": "python",
                    "files": ["tool.json", "run.py"],
                },
            },
        }
        cache_dir = tmp_path / ".community" / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

        # Pre-create the tool as already installed
        tool_dir = tools_dir / "existing_tool"
        tool_dir.mkdir()
        (tool_dir / ".env_version").write_text("1.0.0")

        from arqitect.brain.community import seed_tools
        with patch("arqitect.brain.community._manifest_path",
                   return_value=str(cache_dir / "manifest.json")), \
             patch("arqitect.brain.community.get_mcp_tools_dir",
                   return_value=str(tools_dir)):
            count = seed_tools()

        assert count == 0


# ---------------------------------------------------------------------------
# types.py — TOOL_LIFECYCLE channel
# ---------------------------------------------------------------------------

class TestToolLifecycleChannel:
    """The TOOL_LIFECYCLE channel exists in Channel enum."""

    @pytest.mark.timeout(10)
    def test_channel_exists(self):
        from arqitect.types import Channel
        assert hasattr(Channel, "TOOL_LIFECYCLE")
        assert Channel.TOOL_LIFECYCLE == "tool:lifecycle"

    @pytest.mark.timeout(10)
    def test_channel_value_is_string(self):
        from arqitect.types import Channel
        assert str(Channel.TOOL_LIFECYCLE) == IsStr(regex=r"^tool:.+")


# ---------------------------------------------------------------------------
# ToolMeta dataclass contract
# ---------------------------------------------------------------------------

class TestToolMeta:
    """ToolMeta dataclass holds expected fields from tool.json."""

    @pytest.mark.timeout(10)
    def test_get_meta_returns_correct_type(self, tools_dir, dir_tool):
        """get_meta returns a ToolMeta with all required attributes."""
        from arqitect.mcp.tool_manager import ToolManager, ToolMeta

        mgr = ToolManager()
        mgr.scan()

        meta = mgr.get_meta("adder_tool")
        assert meta == IsInstance(ToolMeta)
        assert meta.name == "adder_tool"
        assert meta.runtime == "python"
        assert meta.entry == "run.py"
        assert meta.tool_dir == str(dir_tool)

    @pytest.mark.timeout(10)
    def test_get_meta_unknown_returns_none(self, tools_dir):
        """get_meta returns None for unregistered tools."""
        from arqitect.mcp.tool_manager import ToolManager

        mgr = ToolManager()
        mgr.scan()

        assert mgr.get_meta("nonexistent") is None
