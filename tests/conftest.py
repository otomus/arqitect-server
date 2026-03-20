"""Shared test fixtures using proper testing tools.

- fakeredis: in-memory Redis (no server needed)
- pyfakefs: fake filesystem for nerve files/sandbox
- responses: mock HTTP for MCP server
- pytest-subprocess: fake subprocess for nerve invocation
- FakeLLM: custom scripted LLM replacement (only truly custom mock)
- Brain patch helpers: shared across brain routing, chain, and security tests
"""

import json
import os
import sqlite3
from unittest.mock import patch, MagicMock

import fakeredis
import pytest
import responses as responses_lib


# ---------------------------------------------------------------------------
# FakeLLM — the only custom mock (no standard tool for LLM inference)
# ---------------------------------------------------------------------------

class FakeLLM:
    """Scripted replacement for llm_generate.

    Usage:
        fake = FakeLLM([
            ("Available nerves", '{"action": "invoke_nerve", "name": "weather_nerve"}'),
            ("classify this", '{"type": "direct"}'),
        ])

    Each (substring, response) pair matched against prompt+system in order.
    Consumed after use unless reuse=True. Unmatched calls return a
    pass-through for communication rewrites or empty JSON otherwise.
    """

    def __init__(self, responses: list | None = None):
        self._responses: list[dict] = []
        self.calls: list[dict] = []
        if responses:
            for item in responses:
                if len(item) == 3:
                    substr, resp, reuse = item
                else:
                    substr, resp = item
                    reuse = False
                self._responses.append({
                    "substr": substr, "response": resp, "reuse": reuse, "used": False,
                })

    def add(self, substr: str, response: str, reuse: bool = False):
        self._responses.append({
            "substr": substr, "response": response, "reuse": reuse, "used": False,
        })

    def __call__(self, model: str, prompt: str, system: str = "") -> str:
        call = {"model": model, "prompt": prompt, "system": system}
        self.calls.append(call)
        for entry in self._responses:
            if entry["used"] and not entry["reuse"]:
                continue
            if entry["substr"] in prompt or entry["substr"] in system:
                entry["used"] = True
                return entry["response"]
        # Intent classification — default to "direct"
        if "classify" in system.lower() or "classify" in prompt.lower():
            return '{"type": "direct"}'
        # Communication rewrite pass-through — return the data portion as-is
        if "rewrite" in system.lower() or "personality" in system.lower():
            if "Data:" in prompt:
                return prompt.split("Data:")[-1].strip()
            return prompt[:200]
        # Sense arg translation — pass through as-is
        if "Sense:" in prompt and "extract" in system.lower():
            return '{}'
        return "{}"

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def prompts_containing(self, substr: str) -> list[dict]:
        return [c for c in self.calls if substr in c["prompt"] or substr in c["system"]]


# ---------------------------------------------------------------------------
# fakeredis — in-memory Redis, no server needed
# ---------------------------------------------------------------------------

@pytest.fixture
def test_redis():
    """Provide an in-memory fake Redis instance."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=True)
    yield client
    client.flushall()


# ---------------------------------------------------------------------------
# SQLite — real SQLite in pytest tmp_path (no fake needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Redirect cold/warm memory SQLite DBs to temp directory.

    Patches the module-level _DB_PATH in cold.py and warm.py since
    they resolve at import time from get_memory_dir().
    """
    cold_path = str(tmp_path / "knowledge.db")
    warm_path = str(tmp_path / "episodes.db")
    with patch("arqitect.memory.cold._DB_PATH", cold_path):
        with patch("arqitect.memory.warm._DB_PATH", warm_path):
            yield tmp_path


# ---------------------------------------------------------------------------
# Memory manager — combines fakeredis + temp SQLite
# ---------------------------------------------------------------------------

@pytest.fixture
def mem(test_redis, tmp_memory_dir):
    """Provide a fresh MemoryManager with isolated storage."""
    from arqitect.memory import MemoryManager
    manager = MemoryManager(test_redis)
    yield manager


# ---------------------------------------------------------------------------
# pyfakefs — fake filesystem for nerve files and sandbox
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_fs(fs):
    """Provide pyfakefs with real modules passed through.

    pyfakefs intercepts all os/io calls. We pass through:
    - sqlite3 (uses C-level file access)
    - arqitect package (needs real imports)
    """
    # Allow real access to the arqitect package and sqlite
    fs.add_real_directory(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        read_only=True,
    )
    yield fs


@pytest.fixture
def nerves_dir(tmp_path):
    """Provide an isolated nerves directory."""
    d = str(tmp_path / "nerves")
    os.makedirs(d, exist_ok=True)
    with patch("arqitect.brain.config.NERVES_DIR", d):
        with patch("arqitect.brain.invoke.NERVES_DIR", d):
            with patch("arqitect.brain.synthesis.NERVES_DIR", d):
                with patch("arqitect.brain.catalog.NERVES_DIR", d):
                    yield d


@pytest.fixture
def sandbox_dir(tmp_path):
    """Provide an isolated sandbox directory."""
    d = str(tmp_path / "sandbox")
    os.makedirs(d, exist_ok=True)
    with patch("arqitect.brain.config.SANDBOX_DIR", d):
        with patch("arqitect.brain.invoke.SANDBOX_DIR", d):
            yield d


# ---------------------------------------------------------------------------
# pytest-subprocess — fake nerve subprocess invocation
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_process(fp):
    """Provide pytest-subprocess's fake process fixture.

    Usage:
        fp.register(["python", fp.any()], stdout='{"response": "hello"}')

    Or register specific nerve outputs:
        fp.register([sys.executable, fp.any(endswith="weather_nerve/nerve.py"), fp.any()],
                    stdout='{"response": "sunny 25C"}')
    """
    return fp


# ---------------------------------------------------------------------------
# responses — mock HTTP for MCP server
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mcp():
    """Mock MCP tool listing and calling via the responses library.

    Usage:
        mock_mcp.tools = {"weather_tool": {"description": "weather"}}
    """

    class MCPMock:
        def __init__(self):
            self.tools = {}

        def _get_tools(self, request):
            return (200, {}, json.dumps({"tools": self.tools}))

        def _call_tool(self, request):
            return (200, {}, json.dumps({"result": "mock tool result"}))

    mcp = MCPMock()
    with responses_lib.RequestsMock() as rsps:
        rsps.add_callback(
            responses_lib.GET,
            "http://127.0.0.1:8100/tools",
            callback=mcp._get_tools,
            content_type="application/json",
        )
        rsps.add_callback(
            responses_lib.POST,
            url=responses_lib.matchers.url_params_matcher({}, allow_blank=True),
            callback=mcp._call_tool,
            match=[responses_lib.matchers.request_header_matcher({})],
            content_type="application/json",
        )
        # Simpler: catch all POST to 8100
        rsps.add(
            responses_lib.POST,
            url="http://127.0.0.1:8100/call/",
            json={"result": "mock"},
            match=[],
        )
        yield mcp


# ---------------------------------------------------------------------------
# Event capture — lightweight spy on publish_event
# ---------------------------------------------------------------------------

@pytest.fixture
def captured_events():
    """Capture all publish_event calls without Redis pub/sub."""
    events = []

    def _capture(channel, data):
        events.append({"channel": channel, "data": data})

    with patch("arqitect.brain.brain.publish_event", side_effect=_capture):
        with patch("arqitect.brain.brain.publish_response"):
            with patch("arqitect.brain.brain.publish_memory_state"):
                with patch("arqitect.brain.brain.publish_nerve_status"):
                    yield events


# ---------------------------------------------------------------------------
# Flow recorder — structured trace capture for flow testing
# ---------------------------------------------------------------------------

@pytest.fixture
def flow_recorder():
    """Provide a FlowRecorder that intercepts all publish_event calls.

    Patches publish_event at ALL import sites so no events are missed.
    Also patches publish_response / publish_memory_state / publish_nerve_status
    to prevent Redis errors.

    Yields the recorder. After the test, call ``recorder.trace()`` for
    the full structured trace.

    Note: Do NOT combine with ``captured_events`` — they conflict.
    """
    from arqitect.tracing import FlowRecorder, PUBLISH_EVENT_SITES

    recorder = FlowRecorder()
    patches = [patch(site, side_effect=recorder.intercept) for site in PUBLISH_EVENT_SITES]
    # Also silence publish_response / publish_memory_state / publish_nerve_status
    silence_patches = [
        patch("arqitect.brain.brain.publish_response"),
        patch("arqitect.brain.brain.publish_memory_state"),
        patch("arqitect.brain.brain.publish_nerve_status"),
        patch("arqitect.brain.dispatch.publish_response"),
        patch("arqitect.brain.dispatch.publish_memory_state"),
        patch("arqitect.brain.consolidate.publish_nerve_status"),
    ]

    for p in patches + silence_patches:
        p.start()

    yield recorder

    for p in patches + silence_patches:
        p.stop()


# ---------------------------------------------------------------------------
# Shared brain test helpers — used by routing, chain, and security tests
# ---------------------------------------------------------------------------

def make_mem(redis_client):
    """Create a MemoryManager backed by the given Redis client.

    Args:
        redis_client: A fakeredis or real Redis client.

    Returns:
        A fresh MemoryManager instance.
    """
    from arqitect.memory import MemoryManager
    return MemoryManager(redis_client)


def register_qualified_nerve(mem, name, description="stub nerve", role="tool"):
    """Register a nerve in cold memory with passing qualification.

    Args:
        mem: MemoryManager instance.
        name: Nerve name to register.
        description: Human-readable description.
        role: Nerve role (tool, creative, code).
    """
    if role != "tool":
        mem.cold.register_nerve_rich(name, description, role=role)
    else:
        mem.cold.register_nerve(name, description)
    mem.cold.record_qualification(
        "nerve", name, qualified=True, score=0.8,
        iterations=3, test_count=15, pass_count=12,
    )


def make_nerve_file(nerves_dir, name, code=None):
    """Create a minimal nerve.py so invoke_nerve finds it.

    Args:
        nerves_dir: Base directory for nerves.
        name: Nerve name (directory name).
        code: Optional Python source. Defaults to a stub that returns ok.

    Returns:
        Path to the created nerve.py file.
    """
    if code is None:
        code = "import sys, json; print(json.dumps({'response': 'ok'}))\n"
    d = os.path.join(nerves_dir, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "nerve.py")
    with open(path, "w") as f:
        f.write(code)
    return path


def setup_brain_patches(fake_llm, mem, test_redis, nerves_dir, sandbox_dir):
    """Return minimal patches for brain.think() tests.

    Mocked: LLM (no model), Redis pub/sub events (no listeners), consolidator (heavy).
    Real: memory, matching, catalog, filesystem, intent (uses mocked LLM).

    Args:
        fake_llm: FakeLLM instance (or any callable matching llm_generate signature).
        mem: MemoryManager instance.
        test_redis: fakeredis client.
        nerves_dir: Path to isolated nerves directory.
        sandbox_dir: Path to isolated sandbox directory.

    Returns:
        List of unittest.mock.patch objects (not yet started).
    """
    return [
        patch("arqitect.brain.brain.mem", mem),
        patch("arqitect.brain.brain.r", test_redis),
        patch("arqitect.brain.dispatch.mem", mem),
        patch("arqitect.brain.dispatch.r", test_redis),
        patch("arqitect.brain.catalog.mem", mem),
        patch("arqitect.brain.brain.NERVES_DIR", nerves_dir),
        patch("arqitect.brain.brain.SANDBOX_DIR", sandbox_dir),
        # Patch llm_generate at every import site
        patch("arqitect.brain.brain.llm_generate", side_effect=fake_llm),
        patch("arqitect.brain.dispatch.llm_generate", side_effect=fake_llm),
        patch("arqitect.brain.helpers.llm_generate", side_effect=fake_llm),
        patch("arqitect.brain.intent.llm_generate", side_effect=fake_llm),
        patch("arqitect.matching._get_nerve_embedding", return_value=None),
        patch("arqitect.brain.synthesis.classify_nerve_role", return_value="tool"),
        patch(
            "arqitect.brain.synthesis.threading.Thread",
            type("_NoOp", (), {
                "__init__": lambda *a, **kw: None,
                "start": lambda self: None,
            }),
        ),
        # Events — patched at both brain and dispatch import sites
        patch("arqitect.brain.brain.publish_event"),
        patch("arqitect.brain.brain.publish_response"),
        patch("arqitect.brain.brain.publish_memory_state"),
        patch("arqitect.brain.brain.publish_nerve_status"),
        patch("arqitect.brain.dispatch.publish_event"),
        patch("arqitect.brain.dispatch.publish_response"),
        patch("arqitect.brain.dispatch.publish_memory_state"),
        patch("arqitect.brain.brain.get_consolidator", return_value=MagicMock()),
        # Permissions — bypassed at both import sites
        patch("arqitect.brain.permissions.can_use_nerve", return_value=True),
        patch("arqitect.brain.dispatch.can_use_nerve", return_value=True),
    ]


def patch_invoke_nerve(return_value=None, side_effect=None):
    """Patch invoke_nerve at both brain.py and dispatch.py import sites.

    Uses a single shared mock so assertions work regardless of which module
    calls invoke_nerve.

    Args:
        return_value: Fixed return value for the mock.
        side_effect: Side effect function or exception.

    Returns:
        Context manager that yields the shared mock.
    """
    import contextlib

    @contextlib.contextmanager
    def _cm():
        kwargs = {}
        if side_effect is not None:
            kwargs["side_effect"] = side_effect
        elif return_value is not None:
            kwargs["return_value"] = return_value
        shared_mock = MagicMock(**kwargs)

        with patch("arqitect.brain.brain.invoke_nerve", shared_mock), \
             patch("arqitect.brain.dispatch.invoke_nerve", shared_mock):
            yield shared_mock

    return _cm()


def patch_synthesize_nerve(return_value=None, side_effect=None):
    """Patch synthesize_nerve at both brain.py and dispatch.py import sites.

    Uses a single shared mock so assertions work regardless of which module
    calls synthesize_nerve.

    Args:
        return_value: Fixed return value for the mock.
        side_effect: Side effect function or exception.

    Returns:
        Context manager that yields the shared mock.
    """
    import contextlib

    @contextlib.contextmanager
    def _cm():
        kwargs = {}
        if side_effect is not None:
            kwargs["side_effect"] = side_effect
        elif return_value is not None:
            kwargs["return_value"] = return_value
        shared_mock = MagicMock(**kwargs)

        with patch("arqitect.brain.brain.synthesize_nerve", shared_mock), \
             patch("arqitect.brain.dispatch.synthesize_nerve", shared_mock):
            yield shared_mock

    return _cm()


def setup_brain_patches_no_perms(fake_llm, mem, test_redis, nerves_dir, sandbox_dir):
    """Brain patches with real permissions (not bypassed) + safety bypass.

    Used by security tests that need real permission enforcement.

    Args:
        Same as setup_brain_patches.

    Returns:
        List of unittest.mock.patch objects (not yet started).
    """
    return [
        patch("arqitect.brain.brain.mem", mem),
        patch("arqitect.brain.brain.r", test_redis),
        patch("arqitect.brain.dispatch.mem", mem),
        patch("arqitect.brain.dispatch.r", test_redis),
        patch("arqitect.brain.catalog.mem", mem),
        patch("arqitect.brain.brain.NERVES_DIR", nerves_dir),
        patch("arqitect.brain.brain.SANDBOX_DIR", sandbox_dir),
        patch("arqitect.brain.brain.llm_generate", side_effect=fake_llm),
        patch("arqitect.brain.dispatch.llm_generate", side_effect=fake_llm),
        patch("arqitect.brain.helpers.llm_generate", side_effect=fake_llm),
        patch("arqitect.brain.intent.llm_generate", side_effect=fake_llm),
        patch("arqitect.matching._get_nerve_embedding", return_value=None),
        patch("arqitect.brain.synthesis.classify_nerve_role", return_value="tool"),
        patch(
            "arqitect.brain.synthesis.threading.Thread",
            type("_NoOp", (), {
                "__init__": lambda *a, **kw: None,
                "start": lambda self: None,
            }),
        ),
        patch("arqitect.brain.brain.publish_event"),
        patch("arqitect.brain.brain.publish_response"),
        patch("arqitect.brain.brain.publish_memory_state"),
        patch("arqitect.brain.brain.publish_nerve_status"),
        patch("arqitect.brain.dispatch.publish_event"),
        patch("arqitect.brain.dispatch.publish_response"),
        patch("arqitect.brain.dispatch.publish_memory_state"),
        patch("arqitect.brain.brain.get_consolidator", return_value=MagicMock()),
        # Safety bypass — tested separately in TestSafetyFilter
        patch("arqitect.brain.brain._safety_check_input", return_value=(True, "")),
        # NOTE: permissions NOT patched — real enforcement
    ]
