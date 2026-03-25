"""Flow scenario tests — run stress cases through think() with FlowRecorder.

Picks representative cases from stress_cases_1000, routes them through the
real think() pipeline with a scripted FakeLLM, and captures structured traces.
Each test asserts on the flow shape: which events fired, which nerves were
invoked/synthesized, and whether the pipeline completed without errors.

These tests are NOT about correctness of LLM output — they verify that the
brain's orchestration (routing, dispatch, memory, events) holds together
end-to-end for each task category.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from arqitect.tracing import FlowRecorder, PUBLISH_EVENT_SITES
from tests.conftest import (
    FakeLLM,
    make_nerve_file,
    register_qualified_nerve,
)
from tests.stress_cases_1000 import CASES


# ---------------------------------------------------------------------------
# Tag → LLM routing configuration
# ---------------------------------------------------------------------------

# Maps stress case tags to LLM responses the FakeLLM should produce.
# Each entry: (prompt_substr, json_response, reuse).
# "Available nerves" matches the routing prompt built by think().

_RESPOND_ACTION = json.dumps({"action": "respond", "message": "Hello there!"})
_INVOKE_WEATHER = json.dumps({"action": "invoke_nerve", "name": "weather_nerve"})
_INVOKE_MATH = json.dumps({"action": "invoke_nerve", "name": "math_nerve"})
_INVOKE_CODE = json.dumps({"action": "invoke_nerve", "name": "code_nerve"})
_INVOKE_WRITER = json.dumps({"action": "invoke_nerve", "name": "writer_nerve"})
_INVOKE_DATA = json.dumps({"action": "invoke_nerve", "name": "data_nerve"})
_SYNTHESIZE = json.dumps({
    "action": "synthesize_nerve",
    "name": "task_nerve",
    "description": "Handles the requested task",
    "spec": "A nerve that processes user tasks",
})

TAG_ROUTES: dict[str, str] = {
    "greeting": _RESPOND_ACTION,
    "identity": _RESPOND_ACTION,
    "knowledge": _RESPOND_ACTION,
    "fun": _RESPOND_ACTION,
    "weather": _INVOKE_WEATHER,
    "math": _INVOKE_MATH,
    "productivity": _RESPOND_ACTION,
    "writing": _INVOKE_WRITER,
    "code-python": _INVOKE_CODE,
    "code-js": _INVOKE_CODE,
    "code-sql": _INVOKE_CODE,
    "code-web": _INVOKE_CODE,
    "code-api": _INVOKE_CODE,
    "code-git": _INVOKE_CODE,
    "code-devops": _INVOKE_CODE,
    "data": _INVOKE_DATA,
    "ml": _INVOKE_DATA,
    "system": _RESPOND_ACTION,
    "multilang": _RESPOND_ACTION,
    "context": _RESPOND_ACTION,
}

# Nerves that TAG_ROUTES reference — must exist in cold memory + filesystem.
REQUIRED_NERVES = {
    "weather_nerve": "Fetch current weather for a location",
    "math_nerve": "Evaluate math expressions and solve equations",
    "code_nerve": "Write, debug, and explain code",
    "writer_nerve": "Write, edit, and improve text content",
    "data_nerve": "Analyze and transform data",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_cases_by_tag(tag: str, limit: int = 3) -> list[dict]:
    """Select up to ``limit`` cases for a given tag."""
    return [c for c in CASES if c["tag"] == tag][:limit]


def _build_fake_llm(tag: str) -> FakeLLM:
    """Build a FakeLLM scripted for a specific tag's routing decision."""
    route_response = TAG_ROUTES.get(tag, _RESPOND_ACTION)
    return FakeLLM([
        ("Available nerves", route_response, True),
        ("No nerves exist", route_response, True),
    ])


def _setup_brain(fake_llm, mem, test_redis, nerves_dir, sandbox_dir):
    """Return patch list for brain.think() — uses flow_recorder for events."""
    return [
        patch("arqitect.brain.brain.mem", mem),
        patch("arqitect.brain.brain.r", test_redis),
        patch("arqitect.brain.dispatch.mem", mem),
        patch("arqitect.brain.dispatch.r", test_redis),
        patch("arqitect.brain.catalog.mem", mem),
        patch("arqitect.brain.brain.NERVES_DIR", nerves_dir),
        patch("arqitect.brain.brain.SANDBOX_DIR", sandbox_dir),
        patch("arqitect.brain.brain.llm_generate", side_effect=fake_llm),
        patch("arqitect.senses.communication.nerve.rewrite_response",
              side_effect=lambda message="", **kw: {"response": message, "format": "text", "tone": "neutral"}),
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
        patch("arqitect.brain.brain.get_consolidator", return_value=MagicMock()),
        patch("arqitect.brain.permissions.can_use_nerve", return_value=True),
        patch("arqitect.brain.dispatch.can_use_nerve", return_value=True),
        # Safety — real pipeline, FakeLLM handles classification
        patch("arqitect.brain.safety.generate_for_role",
              side_effect=fake_llm.generate_for_role),
    ]


def _register_all_nerves(mem, nerves_dir: str) -> None:
    """Register every nerve referenced by TAG_ROUTES."""
    for name, desc in REQUIRED_NERVES.items():
        register_qualified_nerve(mem, name, description=desc)
        make_nerve_file(nerves_dir, name)


def _run_case_with_trace(
    case: dict,
    fake_llm: FakeLLM,
    mem,
    test_redis,
    nerves_dir: str,
    sandbox_dir: str,
) -> tuple[str, "FlowTrace"]:
    """Run a single stress case through think() and return (response, trace).

    Args:
        case: Stress case dict with 'id', 'tag', 'msgs'.
        fake_llm: Scripted LLM for this case's tag.
        mem: MemoryManager instance.
        test_redis: fakeredis client.
        nerves_dir: Path to isolated nerves directory.
        sandbox_dir: Path to isolated sandbox directory.

    Returns:
        Tuple of (brain response string, FlowTrace).
    """
    recorder = FlowRecorder(flow_id=f"case-{case['id']}-{case['tag']}")

    # Event patches — route to flow recorder
    event_patches = [
        patch(site, side_effect=recorder.intercept) for site in PUBLISH_EVENT_SITES
    ]
    silence_patches = [
        patch("arqitect.brain.brain.publish_response"),
        patch("arqitect.brain.brain.publish_memory_state"),
        patch("arqitect.brain.brain.publish_nerve_status"),
        patch("arqitect.brain.dispatch.publish_response"),
        patch("arqitect.brain.dispatch.publish_memory_state"),
        patch("arqitect.brain.consolidate.publish_nerve_status"),
    ]
    brain_patches = _setup_brain(fake_llm, mem, test_redis, nerves_dir, sandbox_dir)

    all_patches = brain_patches + event_patches + silence_patches
    for p in all_patches:
        p.start()

    try:
        from arqitect.brain.brain import think
        task = case["msgs"][0]
        response = think(task)
        trace = recorder.trace()
        return response, trace
    finally:
        for p in all_patches:
            p.stop()


# ---------------------------------------------------------------------------
# Collect representative cases per tag (3 per tag)
# ---------------------------------------------------------------------------

_TAGS_TO_TEST = [
    "greeting", "identity", "knowledge", "fun", "weather",
    "math", "writing", "code-python", "code-js",
    "system", "multilang", "context",
]

_SCENARIO_CASES: list[tuple[str, dict]] = []
for _tag in _TAGS_TO_TEST:
    for _case in _pick_cases_by_tag(_tag, limit=3):
        _SCENARIO_CASES.append((_tag, _case))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.timeout(30)
class TestFlowScenarioRouting:
    """Verify think() routes each tag category correctly."""

    @pytest.fixture(autouse=True)
    def _setup(self, mem, test_redis, nerves_dir, sandbox_dir):
        """Register all required nerves before each test."""
        _register_all_nerves(mem, nerves_dir)
        self.mem = mem
        self.test_redis = test_redis
        self.nerves_dir = nerves_dir
        self.sandbox_dir = sandbox_dir

    @pytest.mark.parametrize(
        "tag,case",
        _SCENARIO_CASES,
        ids=[f"{tag}-{c['id']}" for tag, c in _SCENARIO_CASES],
    )
    def test_case_produces_valid_trace(self, tag, case):
        """Each case must produce a trace with at least a thinking event."""
        fake_llm = _build_fake_llm(tag)
        response, trace = _run_case_with_trace(
            case, fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        assert response is not None
        assert len(response) > 0
        assert trace.flow_id == f"case-{case['id']}-{tag}"

        # Every case must produce a brain:thought "thinking" event
        assert trace.has_event("brain:thought", data_contains={"stage": "thinking"})

    @pytest.mark.parametrize(
        "tag,case",
        [(t, c) for t, c in _SCENARIO_CASES if t in ("greeting", "identity", "fun", "system")],
        ids=[f"respond-{c['id']}" for t, c in _SCENARIO_CASES if t in ("greeting", "identity", "fun", "system")],
    )
    def test_respond_tags_produce_no_nerve_invocation(self, tag, case):
        """Tags that should get a direct response must not invoke nerves."""
        fake_llm = _build_fake_llm(tag)
        _, trace = _run_case_with_trace(
            case, fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        assert trace.nerves_invoked() == []

    @pytest.mark.parametrize(
        "tag,expected_nerve",
        [
            ("weather", "weather_nerve"),
            ("math", "math_nerve"),
            ("code-python", "code_nerve"),
            ("code-js", "code_nerve"),
            ("writing", "writer_nerve"),
        ],
    )
    def test_invoke_tags_route_to_correct_nerve(self, tag, expected_nerve):
        """Tags with invoke routing must dispatch to the right nerve."""
        cases = _pick_cases_by_tag(tag, limit=1)
        assert cases, f"No cases for tag {tag} — missing test data is a failure, not a skip"

        fake_llm = _build_fake_llm(tag)
        _, trace = _run_case_with_trace(
            cases[0], fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        invoked = trace.nerves_invoked()
        assert expected_nerve in invoked


@pytest.mark.timeout(30)
class TestFlowScenarioEvents:
    """Verify event flow structure for different action types."""

    @pytest.fixture(autouse=True)
    def _setup(self, mem, test_redis, nerves_dir, sandbox_dir):
        _register_all_nerves(mem, nerves_dir)
        self.mem = mem
        self.test_redis = test_redis
        self.nerves_dir = nerves_dir
        self.sandbox_dir = sandbox_dir

    def test_greeting_flow_events(self):
        """Greeting produces thinking event and stops — no dispatch events."""
        case = _pick_cases_by_tag("greeting", limit=1)[0]
        fake_llm = _build_fake_llm("greeting")
        _, trace = _run_case_with_trace(
            case, fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        stages = trace.dreamstate_stages()
        assert "thinking" in stages
        # No nerve:result events for greetings
        assert trace.nerves_resulted() == []

    def test_invoke_flow_produces_action_event(self):
        """Invoke routing emits a brain:action event with the nerve name."""
        case = _pick_cases_by_tag("weather", limit=1)[0]
        fake_llm = _build_fake_llm("weather")
        _, trace = _run_case_with_trace(
            case, fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        assert trace.has_event("brain:action")

    def test_multiple_cases_get_unique_flow_ids(self):
        """Each case run produces a trace with a unique flow_id."""
        cases = _pick_cases_by_tag("greeting", limit=3)
        flow_ids = set()
        for case in cases:
            fake_llm = _build_fake_llm("greeting")
            _, trace = _run_case_with_trace(
                case, fake_llm, self.mem, self.test_redis,
                self.nerves_dir, self.sandbox_dir,
            )
            flow_ids.add(trace.flow_id)

        assert len(flow_ids) == len(cases)

    def test_event_ordering_preserved(self):
        """Events within a trace must be ordered by insertion time."""
        case = _pick_cases_by_tag("weather", limit=1)[0]
        fake_llm = _build_fake_llm("weather")
        _, trace = _run_case_with_trace(
            case, fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        timestamps = [e.timestamp for e in trace.events]
        assert timestamps == sorted(timestamps)


@pytest.mark.timeout(30)
class TestFlowScenarioMemory:
    """Verify that think() records conversation in memory."""

    @pytest.fixture(autouse=True)
    def _setup(self, mem, test_redis, nerves_dir, sandbox_dir):
        _register_all_nerves(mem, nerves_dir)
        self.mem = mem
        self.test_redis = test_redis
        self.nerves_dir = nerves_dir
        self.sandbox_dir = sandbox_dir

    def test_user_message_stored_in_hot_memory(self):
        """think() must add the user message to hot memory."""
        case = _pick_cases_by_tag("greeting", limit=1)[0]
        fake_llm = _build_fake_llm("greeting")
        _run_case_with_trace(
            case, fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        conversation = self.mem.hot.get_conversation(limit=10)
        user_messages = [m for m in conversation if m.get("role") == "user"]
        assert len(user_messages) >= 1
        assert case["msgs"][0] in user_messages[0]["content"]

    def test_sequential_cases_build_conversation(self):
        """Running multiple cases accumulates conversation history."""
        cases = _pick_cases_by_tag("greeting", limit=2)
        for case in cases:
            fake_llm = _build_fake_llm("greeting")
            _run_case_with_trace(
                case, fake_llm, self.mem, self.test_redis,
                self.nerves_dir, self.sandbox_dir,
            )

        conversation = self.mem.hot.get_conversation(limit=20)
        user_messages = [m["content"] for m in conversation if m.get("role") == "user"]
        assert len(user_messages) >= 2


@pytest.mark.timeout(60)
class TestFlowScenarioSynthesis:
    """Verify synthesize_nerve flow when no nerves exist."""

    @pytest.fixture(autouse=True)
    def _setup(self, mem, test_redis, nerves_dir, sandbox_dir):
        # Deliberately do NOT register nerves — force synthesis path
        self.mem = mem
        self.test_redis = test_redis
        self.nerves_dir = nerves_dir
        self.sandbox_dir = sandbox_dir

    def test_empty_catalog_triggers_synthesis_route(self):
        """When no nerves exist, the LLM context says 'No nerves exist yet'."""
        case = {"id": 9999, "tag": "synth", "msgs": ["build me a timer"]}
        fake_llm = FakeLLM([
            ("No nerves exist", _SYNTHESIZE, True),
        ])

        _, trace = _run_case_with_trace(
            case, fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        # Should have at least the thinking stage
        assert trace.has_event("brain:thought", data_contains={"stage": "thinking"})

    def test_synthesis_llm_receives_catalog_without_custom_nerves(self):
        """With no custom nerves registered, catalog has only core senses."""
        case = {"id": 9998, "tag": "synth", "msgs": ["create a calculator"]}
        fake_llm = FakeLLM([
            ("Available nerves", _SYNTHESIZE, True),
        ])

        _run_case_with_trace(
            case, fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        # LLM should see "Available nerves" (core senses) but no custom ones
        matching = fake_llm.prompts_containing("Available nerves")
        assert len(matching) >= 1
        # The routing prompt (not system prompt) should only list core senses
        routing_call = matching[0]
        prompt_text = routing_call["prompt"]
        for name in REQUIRED_NERVES:
            assert name not in prompt_text, f"{name} should not be in catalog prompt"


@pytest.mark.timeout(30)
class TestFlowScenarioSafety:
    """Verify safety filter fires before routing."""

    @pytest.fixture(autouse=True)
    def _setup(self, mem, test_redis, nerves_dir, sandbox_dir):
        _register_all_nerves(mem, nerves_dir)
        self.mem = mem
        self.test_redis = test_redis
        self.nerves_dir = nerves_dir
        self.sandbox_dir = sandbox_dir

    def test_safe_input_reaches_thinking_stage(self):
        """Normal input passes safety and reaches the thinking stage."""
        case = _pick_cases_by_tag("greeting", limit=1)[0]
        fake_llm = _build_fake_llm("greeting")
        _, trace = _run_case_with_trace(
            case, fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        assert trace.has_event("brain:thought", data_contains={"stage": "thinking"})

    def test_blocked_input_produces_safety_event(self):
        """If safety blocks input, a safety_block event fires."""
        fake_llm = FakeLLM([
            ("safety filter", '{"safe": false, "category": "harmful"}', True),
            ("refusal message", "Blocked for safety.", True),
        ])

        recorder = FlowRecorder(flow_id="safety-block")
        event_patches = [
            patch(site, side_effect=recorder.intercept) for site in PUBLISH_EVENT_SITES
        ]
        silence_patches = [
            patch("arqitect.brain.brain.publish_response"),
            patch("arqitect.brain.brain.publish_memory_state"),
            patch("arqitect.brain.brain.publish_nerve_status"),
            patch("arqitect.brain.dispatch.publish_response"),
            patch("arqitect.brain.dispatch.publish_memory_state"),
            patch("arqitect.brain.consolidate.publish_nerve_status"),
        ]
        brain_patches = _setup_brain(
            fake_llm, self.mem, self.test_redis,
            self.nerves_dir, self.sandbox_dir,
        )

        all_patches = brain_patches + event_patches + silence_patches
        for p in all_patches:
            p.start()

        try:
            from arqitect.brain.brain import think
            response = think("harmless greeting")
            trace = recorder.trace()

            assert response == "Blocked for safety."
            assert trace.has_event("brain:thought", data_contains={"stage": "safety_block"})
            # No thinking event — blocked before routing
            assert not trace.has_event("brain:thought", data_contains={"stage": "thinking"})
        finally:
            for p in all_patches:
                p.stop()
