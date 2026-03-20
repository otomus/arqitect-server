"""Contract tests for synthesize_nerve name propagation.

Covers: catch-all rename, sense-collision rename, tool-as-nerve rename,
return-value contract, and property-based nerve naming validation.
"""

import json
import os
import re
from unittest.mock import patch

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from dirty_equals import IsInstance, IsStr

from tests.conftest import FakeLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NoOpThread:
    """Prevents background qualification from running during tests."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


def _synthesis_patches(fake_llm):
    """Return patches that isolate synthesize_nerve from heavy dependencies.

    Uses FakeLLM instead of raw MagicMock for llm_generate, keeping the
    other stubs minimal.
    """
    return [
        patch("arqitect.brain.synthesis.llm_generate", side_effect=fake_llm),
        patch("arqitect.brain.synthesis.classify_nerve_role", return_value="tool"),
        patch("arqitect.brain.synthesis.list_mcp_tools_with_info", return_value={}),
        patch("arqitect.brain.synthesis.publish_nerve_status"),
        patch("arqitect.brain.synthesis.publish_event"),
        patch("arqitect.brain.synthesis.threading.Thread", _NoOpThread),
        patch("arqitect.brain.synthesis.find_community_bundle", return_value=None),
    ]


@pytest.fixture
def fake_llm():
    """Provide a FakeLLM pre-loaded with synthesis-friendly responses."""
    return FakeLLM([
        # Metadata generation — matched by "designing a nerve"
        ("designing a nerve", '{"system_prompt": "stub prompt", "examples": []}', True),
        # Description generalization — matched by "being created"
        ("being created", "Handles domain-specific operations for the requested area", True),
        # Specific prompt regeneration fallback
        ("Write a system prompt", "You are a domain-specific nerve.", True),
    ])


def _run_synthesize(nerves_dir, mem, fake_llm, name, description, trigger_task="do something",
                    mcp_tools_override=None):
    """Run synthesize_nerve with all heavy deps patched.

    Returns:
        (actual_name, path) tuple from synthesize_nerve.
    """
    patches = _synthesis_patches(fake_llm)
    if mcp_tools_override is not None:
        # Replace the list_mcp_tools_with_info patch
        patches[2] = patch(
            "arqitect.brain.synthesis.list_mcp_tools_with_info",
            return_value=mcp_tools_override,
        )

    with patch("arqitect.brain.synthesis.mem", mem):
        for p in patches:
            p.start()
        try:
            from arqitect.brain.synthesis import synthesize_nerve
            return synthesize_nerve(name, description, trigger_task=trigger_task)
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# 1.1 Catch-all name rejected and renamed
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestCatchAllRename:
    """Catch-all names like 'general_knowledge_nerve' must be renamed to
    something domain-specific so they don't swallow all tasks."""

    def test_general_knowledge_nerve_gets_renamed(self, nerves_dir, mem, fake_llm):
        """A catch-all name like 'general_knowledge_nerve' must be renamed."""
        actual_name, path = _run_synthesize(
            nerves_dir, mem, fake_llm,
            "general_knowledge_nerve",
            "answers trivia and general knowledge questions",
            trigger_task="what is the capital of France?",
        )
        assert actual_name != "general_knowledge_nerve", \
            f"Catch-all name should have been renamed, got '{actual_name}'"
        assert os.path.isfile(path), f"Nerve file not created at {path}"
        assert actual_name in path, \
            f"Path should contain renamed name '{actual_name}', got {path}"

    @pytest.mark.parametrize("catchall_name", [
        "knowledge_nerve", "general_nerve", "utility_nerve",
        "misc_nerve", "everything_nerve", "default_nerve", "info_nerve",
    ])
    def test_all_catchall_names_rejected(self, nerves_dir, mem, fake_llm, catchall_name):
        """Every name in the catch-all list gets renamed."""
        actual_name, _ = _run_synthesize(
            nerves_dir, mem, fake_llm,
            catchall_name, "does stuff", trigger_task="do something",
        )
        assert actual_name != catchall_name


# ---------------------------------------------------------------------------
# 1.2 Sense-collision rename
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestSenseCollisionRename:
    """Nerve names that collide with core senses (sight, hearing, touch,
    awareness, communication) must be renamed to avoid shadowing."""

    @pytest.mark.parametrize("collision_name", [
        "awareness", "awareness_nerve",
        "touch", "touch_nerve",
        "sight", "sight_nerve",
        "hearing", "hearing_nerve",
        "communication", "communication_nerve",
    ])
    def test_sense_name_collision_gets_renamed(self, nerves_dir, mem, fake_llm, collision_name):
        """A nerve name that collides with a core sense must be renamed."""
        actual_name, path = _run_synthesize(
            nerves_dir, mem, fake_llm,
            collision_name,
            "provides information about the system",
            trigger_task="who are you?",
        )
        assert actual_name != collision_name, \
            f"Sense-collision name '{collision_name}' should have been renamed"
        assert actual_name != collision_name.removesuffix("_nerve"), \
            f"Renamed to bare sense name '{actual_name}'"

    def test_returned_name_matches_filesystem(self, nerves_dir, mem, fake_llm):
        """The returned actual_name matches the directory created on disk."""
        actual_name, path = _run_synthesize(
            nerves_dir, mem, fake_llm,
            "touch_nerve", "file operations", trigger_task="list files",
        )
        expected_dir = os.path.join(nerves_dir, actual_name)
        assert os.path.isdir(expected_dir), \
            f"Directory for '{actual_name}' not found at {expected_dir}"
        assert os.path.isfile(os.path.join(expected_dir, "nerve.py"))


# ---------------------------------------------------------------------------
# 1.3 Tool-as-nerve rename
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestToolAsNerveRename:
    """If a nerve name matches an MCP tool, it gets renamed and the tool
    is pre-seeded into the nerve."""

    def test_nerve_named_after_mcp_tool_gets_renamed(self, nerves_dir, mem, fake_llm):
        """If a nerve name matches an MCP tool, it gets renamed and the tool is pre-seeded."""
        mcp_tools = {
            "weather_tool": {"description": "get weather data", "params": {}},
        }
        actual_name, _ = _run_synthesize(
            nerves_dir, mem, fake_llm,
            "weather_tool", "get weather", trigger_task="weather in Paris",
            mcp_tools_override=mcp_tools,
        )
        assert actual_name != "weather_tool", \
            "Nerve should not keep the MCP tool name"


# ---------------------------------------------------------------------------
# 1.4 Return value contract — always (name, path)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestReturnContract:
    """synthesize_nerve must always return a (name, path) tuple where both
    elements are strings and name appears in path."""

    def test_returns_tuple_of_name_and_path(self, nerves_dir, mem, fake_llm):
        """synthesize_nerve must return (actual_name, nerve_path)."""
        result = _run_synthesize(
            nerves_dir, mem, fake_llm,
            "recipe_nerve", "cooking recipes", trigger_task="how to make pasta",
        )
        assert result == (IsInstance(str), IsInstance(str))
        name, path = result
        assert name == IsStr(regex=r"^[a-z0-9_]+$")
        assert name in path

    def test_normal_name_passes_through_unchanged(self, nerves_dir, mem, fake_llm):
        """A valid, non-colliding name is returned unchanged."""
        actual_name, _ = _run_synthesize(
            nerves_dir, mem, fake_llm,
            "recipe_nerve", "cooking recipes", trigger_task="make pasta",
        )
        assert actual_name == "recipe_nerve"


# ---------------------------------------------------------------------------
# 2. Property-based tests — nerve naming invariants
# ---------------------------------------------------------------------------

# Strategy: generate names that are valid Python identifiers, lowercase, with underscores
_valid_nerve_name = st.from_regex(r"[a-z][a-z0-9_]{2,30}_nerve", fullmatch=True)

# Strategy: generate catch-all names from the known set
_catchall_names = st.sampled_from([
    "general_knowledge_nerve", "knowledge_nerve", "general_nerve",
    "utility_nerve", "misc_nerve", "everything_nerve", "general_purpose_nerve",
    "catchall_nerve", "default_nerve", "info_nerve",
])

# Strategy: sense names (bare and with _nerve suffix)
_sense_names = st.sampled_from([
    "sight", "hearing", "touch", "awareness", "communication",
    "sight_nerve", "hearing_nerve", "touch_nerve", "awareness_nerve", "communication_nerve",
])


@pytest.mark.timeout(10)
class TestNerveNamingProperties:
    """Property-based tests verifying naming invariants hold across
    a wide range of inputs."""

    @given(name=_catchall_names)
    @settings(max_examples=20)
    def test_catchall_names_always_rejected(self, name):
        """No catch-all name ever survives _apply_name_guards."""
        from arqitect.brain.synthesis import _apply_name_guards
        fixed_name, _ = _apply_name_guards(name, "does stuff", None, {})
        assert fixed_name != name, f"Catch-all name '{name}' was not rejected"

    @given(name=_sense_names)
    @settings(max_examples=20)
    def test_sense_names_always_rejected(self, name):
        """No sense-collision name ever survives _apply_name_guards."""
        from arqitect.brain.synthesis import _apply_name_guards
        fixed_name, _ = _apply_name_guards(name, "provides system info", None, {})
        bare = name.removesuffix("_nerve")
        assert fixed_name != name, f"Sense name '{name}' was not rejected"
        assert fixed_name != bare, f"Renamed to bare sense name '{bare}'"

    @given(name=_valid_nerve_name)
    @settings(max_examples=50)
    def test_non_reserved_names_pass_through(self, name):
        """Names that are not catch-all or sense-collision pass through unchanged."""
        from arqitect.brain.synthesis import _apply_name_guards, _CATCHALL_NAMES
        from arqitect.brain.config import CORE_SENSES

        sense_names = {s for s in CORE_SENSES} | {f"{s}_nerve" for s in CORE_SENSES}
        assume(name not in _CATCHALL_NAMES)
        assume(name not in sense_names)

        fixed_name, _ = _apply_name_guards(name, "does stuff", None, {})
        assert fixed_name == name, f"Non-reserved name '{name}' was unexpectedly changed to '{fixed_name}'"

    @given(name=_valid_nerve_name)
    @settings(max_examples=30)
    def test_derived_name_is_always_valid_identifier(self, name):
        """_derive_nerve_name always returns a valid Python-style identifier."""
        from arqitect.brain.synthesis import _derive_nerve_name
        derived = _derive_nerve_name("retrieves weather lookups", name)
        assert re.match(r"^[a-z0-9_]+$", derived), \
            f"Derived name '{derived}' contains invalid characters"
        assert len(derived) > 0, "Derived name must not be empty"

    @given(description=st.text(min_size=1, max_size=100).filter(lambda s: any(c.isalpha() for c in s)))
    @settings(max_examples=30)
    def test_derive_nerve_name_produces_valid_names(self, description):
        """_derive_nerve_name produces valid snake_case identifiers for any description."""
        from arqitect.brain.synthesis import _derive_nerve_name
        derived = _derive_nerve_name(description, "fallback_nerve")
        assert re.match(r"^[a-z0-9_]+$", derived), \
            f"Derived name '{derived}' is not a valid snake_case identifier"


# ---------------------------------------------------------------------------
# 3. Internal helper contracts
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestIsGenericPrompt:
    """_is_generic_prompt should detect prompts that are too vague."""

    def test_empty_prompt_is_generic(self):
        from arqitect.brain.synthesis import _is_generic_prompt
        assert _is_generic_prompt("") is True

    def test_specific_prompt_is_not_generic(self):
        from arqitect.brain.synthesis import _is_generic_prompt
        result = _is_generic_prompt(
            "You are a weather specialist. Provide temperature in Celsius "
            "and humidity percentages. Do not give travel advice."
        )
        assert result is False

    def test_prompt_with_multiple_generic_phrases_is_generic(self):
        from arqitect.brain.synthesis import _is_generic_prompt
        result = _is_generic_prompt(
            "You should provide helpful responses and assist the user "
            "with their questions in a helpful manner."
        )
        assert result is True


@pytest.mark.timeout(10)
class TestDeriveNerveName:
    """_derive_nerve_name should extract a meaningful domain name from a description."""

    def test_extracts_first_meaningful_word(self):
        from arqitect.brain.synthesis import _derive_nerve_name
        # "performs" is a stopword, so the first non-stopword "weather" is used
        result = _derive_nerve_name("performs weather lookups", "fallback_nerve")
        assert result == "weather_nerve"

    def test_skips_stopwords(self):
        from arqitect.brain.synthesis import _derive_nerve_name
        # "the" and "a" are stopwords, "calculator" should be picked
        result = _derive_nerve_name("the a calculator for math", "fallback_nerve")
        assert result == "calculator_nerve"

    def test_falls_back_when_all_stopwords(self):
        from arqitect.brain.synthesis import _derive_nerve_name
        result = _derive_nerve_name("the a an", "my_fallback")
        assert result == "my_fallback"

    def test_sanitizes_special_characters(self):
        from arqitect.brain.synthesis import _derive_nerve_name
        result = _derive_nerve_name("weather-data! provider", "fallback")
        assert re.match(r"^[a-z0-9_]+$", result), \
            f"Name '{result}' contains invalid characters"
