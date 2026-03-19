"""P1 — Name propagation tests for synthesize_nerve.

Covers: catch-all rename, sense-collision rename, tool-as-nerve rename,
and verifying the caller receives the actual (renamed) name.
"""

import json
import os
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nerve_dir(nerves_dir, name):
    """Create a minimal nerve directory so it appears to exist."""
    d = os.path.join(nerves_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "nerve.py"), "w") as f:
        f.write("# stub\n")


def _stub_synthesis_deps():
    """Return a set of patches that stub out heavy synthesis dependencies."""

    class _NoOpThread:
        """Prevents background qualification from running during tests."""
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass

    return [
        patch("arqitect.brain.synthesis.llm_generate", return_value='{"system_prompt": "stub", "examples": []}'),
        patch("arqitect.brain.synthesis.classify_nerve_role", return_value="tool"),
        patch("arqitect.brain.synthesis.list_mcp_tools_with_info", return_value={}),
        patch("arqitect.brain.synthesis.publish_nerve_status"),
        patch("arqitect.brain.synthesis.publish_event"),
        patch("arqitect.brain.synthesis.threading.Thread", _NoOpThread),
    ]


# ---------------------------------------------------------------------------
# 1.1 Catch-all name rejected and renamed
# ---------------------------------------------------------------------------

class TestCatchAllRename:
    def test_general_knowledge_nerve_gets_renamed(self, nerves_dir, mem):
        """A catch-all name like 'general_knowledge_nerve' must be renamed."""
        patches = _stub_synthesis_deps()
        with patch("arqitect.brain.synthesis.mem", mem):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.synthesis import synthesize_nerve
                actual_name, path = synthesize_nerve(
                    "general_knowledge_nerve",
                    "answers trivia and general knowledge questions",
                    trigger_task="what is the capital of France?",
                )
                assert actual_name != "general_knowledge_nerve", \
                    f"Catch-all name should have been renamed, got '{actual_name}'"
                assert os.path.isfile(path), f"Nerve file not created at {path}"
                assert actual_name in path, \
                    f"Path should contain renamed name '{actual_name}', got {path}"
            finally:
                for p in patches:
                    p.stop()

    @pytest.mark.parametrize("catchall_name", [
        "knowledge_nerve", "general_nerve", "utility_nerve",
        "misc_nerve", "everything_nerve", "default_nerve", "info_nerve",
    ])
    def test_all_catchall_names_rejected(self, nerves_dir, mem, catchall_name):
        """Every name in the catch-all list gets renamed."""
        patches = _stub_synthesis_deps()
        with patch("arqitect.brain.synthesis.mem", mem):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.synthesis import synthesize_nerve
                actual_name, _ = synthesize_nerve(
                    catchall_name, "does stuff", trigger_task="do something",
                )
                assert actual_name != catchall_name
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 1.2 Sense-collision rename
# ---------------------------------------------------------------------------

class TestSenseCollisionRename:
    @pytest.mark.parametrize("collision_name", [
        "awareness", "awareness_nerve",
        "touch", "touch_nerve",
        "sight", "sight_nerve",
        "hearing", "hearing_nerve",
        "communication", "communication_nerve",
    ])
    def test_sense_name_collision_gets_renamed(self, nerves_dir, mem, collision_name):
        """A nerve name that collides with a core sense must be renamed."""
        patches = _stub_synthesis_deps()
        with patch("arqitect.brain.synthesis.mem", mem):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.synthesis import synthesize_nerve
                actual_name, path = synthesize_nerve(
                    collision_name,
                    "provides information about the system",
                    trigger_task="who are you?",
                )
                assert actual_name != collision_name, \
                    f"Sense-collision name '{collision_name}' should have been renamed"
                assert actual_name != collision_name.removesuffix("_nerve"), \
                    f"Renamed to bare sense name '{actual_name}'"
            finally:
                for p in patches:
                    p.stop()

    def test_returned_name_matches_filesystem(self, nerves_dir, mem):
        """The returned actual_name matches the directory created on disk."""
        patches = _stub_synthesis_deps()
        with patch("arqitect.brain.synthesis.mem", mem):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.synthesis import synthesize_nerve
                actual_name, path = synthesize_nerve(
                    "touch_nerve", "file operations", trigger_task="list files",
                )
                expected_dir = os.path.join(nerves_dir, actual_name)
                assert os.path.isdir(expected_dir), \
                    f"Directory for '{actual_name}' not found at {expected_dir}"
                assert os.path.isfile(os.path.join(expected_dir, "nerve.py"))
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 1.3 Tool-as-nerve rename
# ---------------------------------------------------------------------------

class TestToolAsNerveRename:
    def test_nerve_named_after_mcp_tool_gets_renamed(self, nerves_dir, mem):
        """If a nerve name matches an MCP tool, it gets renamed and the tool is pre-seeded."""
        mcp_tools = {
            "weather_tool": {"description": "get weather data", "params": {}},
        }
        patches = _stub_synthesis_deps()
        # Override the MCP tools mock to return our tools
        patches[2] = patch("arqitect.brain.synthesis.list_mcp_tools_with_info", return_value=mcp_tools)
        with patch("arqitect.brain.synthesis.mem", mem):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.synthesis import synthesize_nerve
                actual_name, _ = synthesize_nerve(
                    "weather_tool", "get weather", trigger_task="weather in Paris",
                )
                assert actual_name != "weather_tool", \
                    "Nerve should not keep the MCP tool name"
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 1.4 Return value contract — always (name, path)
# ---------------------------------------------------------------------------

class TestReturnContract:
    def test_returns_tuple_of_name_and_path(self, nerves_dir, mem):
        """synthesize_nerve must return (actual_name, nerve_path)."""
        patches = _stub_synthesis_deps()
        with patch("arqitect.brain.synthesis.mem", mem):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.synthesis import synthesize_nerve
                result = synthesize_nerve(
                    "recipe_nerve", "cooking recipes", trigger_task="how to make pasta",
                )
                assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
                assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
                name, path = result
                assert isinstance(name, str)
                assert isinstance(path, str)
                assert name in path
            finally:
                for p in patches:
                    p.stop()

    def test_normal_name_passes_through_unchanged(self, nerves_dir, mem):
        """A valid, non-colliding name is returned unchanged."""
        patches = _stub_synthesis_deps()
        with patch("arqitect.brain.synthesis.mem", mem):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.synthesis import synthesize_nerve
                actual_name, _ = synthesize_nerve(
                    "recipe_nerve", "cooking recipes", trigger_task="make pasta",
                )
                assert actual_name == "recipe_nerve"
            finally:
                for p in patches:
                    p.stop()
