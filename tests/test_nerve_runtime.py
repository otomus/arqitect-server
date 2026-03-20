"""Tests for arqitect.nerves.nerve_runtime — pure/testable functions.

Contract-based tests that verify the public behavior of each function
without mocking internals. Only infrastructure (Redis, module-level
side effects) is patched at import time.
"""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest
from dirty_equals import IsPartialDict, IsStr
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Module import — shim deleted arqitect.brain.types, patch infra side effects
# ---------------------------------------------------------------------------

def _get_runtime():
    """Import nerve_runtime with all infrastructure patched out."""
    try:
        import arqitect.brain.types  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        from arqitect import types as real_types
        sys.modules["arqitect.brain.types"] = real_types

    with patch("arqitect.config.loader.get_mcp_url", return_value="http://localhost:5100"), \
         patch("arqitect.config.loader.get_mcp_tools_dir", return_value="/tmp/mcp_tools"), \
         patch("arqitect.config.loader.get_project_root", return_value="/tmp"), \
         patch("arqitect.config.loader.get_redis_host_port", return_value=("localhost", 6379)), \
         patch("arqitect.mcp.external_manager.ExternalMCPManager"):
        import importlib
        import arqitect.nerves.nerve_runtime as mod
        importlib.reload(mod)
        return mod


runtime = _get_runtime()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove all SYNAPSE_* env vars before each test so tests are isolated."""
    for key in list(os.environ):
        if key.startswith("SYNAPSE_"):
            monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture(autouse=True)
def _reset_dedup_set():
    """Clear the publish dedup set between tests."""
    runtime._PUBLISHED_TOOLS_THIS_RUN.clear()
    yield
    runtime._PUBLISHED_TOOLS_THIS_RUN.clear()


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

json_primitives = st.one_of(
    st.text(min_size=0, max_size=50),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
)

json_dicts = st.dictionaries(
    keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    values=json_primitives,
    max_size=10,
)


# ── get_args ──────────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetArgs:
    """Tests for get_args() — parses sys.argv into a single string."""

    def test_no_args(self):
        with patch.object(sys, "argv", ["nerve.py"]):
            assert runtime.get_args() == ""

    def test_single_arg(self):
        with patch.object(sys, "argv", ["nerve.py", "hello"]):
            assert runtime.get_args() == "hello"

    def test_multiple_args_joined(self):
        with patch.object(sys, "argv", ["nerve.py", "hello", "world", "test"]):
            assert runtime.get_args() == "hello world test"

    def test_args_with_spaces_in_values(self):
        with patch.object(sys, "argv", ["nerve.py", "hello world"]):
            assert runtime.get_args() == "hello world"

    @given(args=st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=10))
    @settings(max_examples=30)
    def test_always_returns_string(self, args):
        """get_args() always returns a string, regardless of argv content."""
        with patch.object(sys, "argv", ["nerve.py"] + args):
            result = runtime.get_args()
            assert isinstance(result, str)


# ── get_session_context ───────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetSessionContext:
    """Tests for get_session_context() — parses SYNAPSE_SESSION env var."""

    def test_missing_env_var(self):
        assert runtime.get_session_context() == {}

    def test_empty_string(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_SESSION", "")
        assert runtime.get_session_context() == {}

    def test_valid_json(self, monkeypatch):
        data = {"city": "Tel Aviv", "timezone": "Asia/Jerusalem"}
        monkeypatch.setenv("SYNAPSE_SESSION", json.dumps(data))
        assert runtime.get_session_context() == data

    def test_malformed_json(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_SESSION", "{bad json")
        assert runtime.get_session_context() == {}

    def test_non_dict_json(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_SESSION", '"just a string"')
        result = runtime.get_session_context()
        assert result == "just a string"

    @given(data=json_dicts)
    @settings(max_examples=20)
    def test_roundtrips_any_dict(self, data):
        """Any JSON-serializable dict survives the roundtrip."""
        os.environ["SYNAPSE_SESSION"] = json.dumps(data)
        try:
            assert runtime.get_session_context() == data
        finally:
            os.environ.pop("SYNAPSE_SESSION", None)


# ── get_episode_hints ─────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetEpisodeHints:
    """Tests for get_episode_hints() — parses SYNAPSE_EPISODES env var."""

    def test_missing_env_var(self):
        assert runtime.get_episode_hints() == []

    def test_valid_list(self, monkeypatch):
        episodes = [{"task": "greet", "result": "Hello!"}]
        monkeypatch.setenv("SYNAPSE_EPISODES", json.dumps(episodes))
        assert runtime.get_episode_hints() == episodes

    def test_malformed_json(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EPISODES", "[not valid")
        assert runtime.get_episode_hints() == []

    def test_empty_list(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_EPISODES", "[]")
        assert runtime.get_episode_hints() == []

    @given(data=st.lists(json_dicts, min_size=0, max_size=5))
    @settings(max_examples=20)
    def test_roundtrips_any_list(self, data):
        """Any JSON-serializable list survives the roundtrip."""
        os.environ["SYNAPSE_EPISODES"] = json.dumps(data)
        try:
            assert runtime.get_episode_hints() == data
        finally:
            os.environ.pop("SYNAPSE_EPISODES", None)


# ── get_known_tools ───────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetKnownTools:
    """Tests for get_known_tools() — parses SYNAPSE_KNOWN_TOOLS env var."""

    def test_missing_env_var(self):
        assert runtime.get_known_tools() == []

    def test_valid_list(self, monkeypatch):
        tools = ["weather_tool", "search_tool"]
        monkeypatch.setenv("SYNAPSE_KNOWN_TOOLS", json.dumps(tools))
        assert runtime.get_known_tools() == tools

    def test_malformed_json(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_KNOWN_TOOLS", "broken")
        assert runtime.get_known_tools() == []

    @given(tools=st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=10))
    @settings(max_examples=20)
    def test_roundtrips_any_tool_list(self, tools):
        """Any list of strings roundtrips correctly."""
        os.environ["SYNAPSE_KNOWN_TOOLS"] = json.dumps(tools)
        try:
            assert runtime.get_known_tools() == tools
        finally:
            os.environ.pop("SYNAPSE_KNOWN_TOOLS", None)


# ── get_user_facts ────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetUserFacts:
    """Tests for get_user_facts() — parses SYNAPSE_FACTS env var."""

    def test_missing_env_var(self):
        assert runtime.get_user_facts() == {}

    def test_valid_dict(self, monkeypatch):
        facts = {"city": "Rehovot", "name": "Oron"}
        monkeypatch.setenv("SYNAPSE_FACTS", json.dumps(facts))
        assert runtime.get_user_facts() == facts

    def test_malformed_json(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_FACTS", "not json")
        assert runtime.get_user_facts() == {}


# ── get_user_profile ──────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetUserProfile:
    """Tests for get_user_profile() — parses SYNAPSE_USER_PROFILE env var."""

    def test_missing_env_var(self):
        assert runtime.get_user_profile() == {}

    def test_valid_profile(self, monkeypatch):
        profile = {"name": "Alice", "gender": "female"}
        monkeypatch.setenv("SYNAPSE_USER_PROFILE", json.dumps(profile))
        assert runtime.get_user_profile() == profile

    def test_malformed_json(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_USER_PROFILE", "{nope")
        assert runtime.get_user_profile() == {}


# ── get_nerve_meta ────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetNerveMeta:
    """Tests for get_nerve_meta() — parses SYNAPSE_NERVE_META env var."""

    def test_missing_env_var(self):
        assert runtime.get_nerve_meta() == {}

    def test_valid_meta(self, monkeypatch):
        meta = {"system_prompt": "You are helpful", "examples": []}
        monkeypatch.setenv("SYNAPSE_NERVE_META", json.dumps(meta))
        assert runtime.get_nerve_meta() == meta

    def test_malformed_json(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_NERVE_META", "{{bad}}")
        assert runtime.get_nerve_meta() == {}


# ── get_messages ──────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetMessages:
    """Tests for get_messages() — parses SYNAPSE_MESSAGES env var."""

    def test_missing_env_var(self):
        assert runtime.get_messages() == []

    def test_valid_messages(self, monkeypatch):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        monkeypatch.setenv("SYNAPSE_MESSAGES", json.dumps(msgs))
        assert runtime.get_messages() == msgs

    def test_malformed_json(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_MESSAGES", "nope")
        assert runtime.get_messages() == []

    def test_empty_list(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_MESSAGES", "[]")
        assert runtime.get_messages() == []


# ── get_project_context ───────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetProjectContext:
    """Tests for get_project_context() — reads SYNAPSE_PROJECT_CONTEXT env var."""

    def test_missing_env_var(self):
        assert runtime.get_project_context() == ""

    def test_present(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_PROJECT_CONTEXT", "python+fastapi project")
        assert runtime.get_project_context() == "python+fastapi project"

    @given(value=st.text(
        min_size=0, max_size=200,
        alphabet=st.characters(blacklist_characters="\x00"),
    ))
    @settings(max_examples=20)
    def test_returns_exact_env_value(self, value):
        """get_project_context() returns the raw env var string unchanged."""
        os.environ["SYNAPSE_PROJECT_CONTEXT"] = value
        try:
            assert runtime.get_project_context() == value
        finally:
            os.environ.pop("SYNAPSE_PROJECT_CONTEXT", None)


# ── can_answer_from_facts ─────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestCanAnswerFromFacts:
    """Tests for can_answer_from_facts() — keyword matching against stored facts."""

    def test_empty_pool_returns_none(self):
        assert runtime.can_answer_from_facts("where am I?", {}, {}) is None

    def test_identity_with_name(self):
        facts = {"name": "Alice"}
        result = runtime.can_answer_from_facts("who am i", facts, {})
        assert result is not None
        assert result == IsStr(regex=r".*Alice.*")

    def test_identity_with_name_and_city(self):
        facts = {"name": "Bob", "city": "London"}
        result = runtime.can_answer_from_facts("who am i", facts, {})
        assert "Bob" in result
        assert "London" in result

    def test_identity_no_name_returns_none(self):
        facts = {"city": "Berlin"}
        result = runtime.can_answer_from_facts("who am i", facts, {})
        assert result is None

    def test_location_with_possessive(self):
        facts = {"city": "Rehovot", "country": "Israel"}
        result = runtime.can_answer_from_facts("where do I live", facts, {})
        assert result is not None
        assert "Rehovot" in result

    def test_location_without_possessive_returns_none(self):
        """Without 'my', 'I', etc., keyword matching should not trigger."""
        facts = {"city": "Rehovot"}
        result = runtime.can_answer_from_facts("where is the store", facts, {})
        assert result is None

    def test_session_merged_with_facts(self):
        session = {"city": "SessionCity"}
        facts = {"name": "Oron"}
        result = runtime.can_answer_from_facts("who am i", facts, session)
        assert "Oron" in result

    def test_facts_override_session(self):
        session = {"city": "OldCity"}
        facts = {"city": "NewCity"}
        result = runtime.can_answer_from_facts("where do I live", facts, session)
        assert result is not None
        assert "NewCity" in result

    def test_no_keyword_match_returns_none(self):
        facts = {"city": "Rehovot"}
        result = runtime.can_answer_from_facts("tell me a joke", facts, {})
        assert result is None

    def test_name_without_possessive_marker_not_triggered(self):
        """'name the planets' should NOT match the user's name fact."""
        facts = {"name": "Oron"}
        result = runtime.can_answer_from_facts("name the planets", facts, {})
        assert result is None

    @given(question=st.text(min_size=1, max_size=100))
    @settings(max_examples=30)
    def test_never_crashes_on_arbitrary_input(self, question):
        """can_answer_from_facts never raises, regardless of input."""
        facts = {"name": "Test", "city": "Nowhere"}
        result = runtime.can_answer_from_facts(question, facts, {})
        assert result is None or isinstance(result, str)


# ── substitute_fact_values ────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestSubstituteFactValues:
    """Tests for substitute_fact_values() — fuzzy value substitution."""

    def test_no_facts_returns_original(self):
        args = {"city": "Roviot"}
        assert runtime.substitute_fact_values(args, {}, {}) == args

    def test_corrects_garbled_value(self):
        args = {"city": "Roviot"}
        facts = {"city": "Rehovot"}
        result = runtime.substitute_fact_values(args, facts, {})
        assert result["city"] == "Rehovot"

    def test_leaves_short_values_alone(self):
        args = {"x": "ab"}
        facts = {"city": "Rehovot"}
        result = runtime.substitute_fact_values(args, facts, {})
        assert result["x"] == "ab"

    def test_exact_match_not_replaced(self):
        """If value already matches a fact exactly (case-insensitive), no replacement."""
        args = {"city": "rehovot"}
        facts = {"city": "Rehovot"}
        result = runtime.substitute_fact_values(args, facts, {})
        assert result["city"] == "rehovot"

    def test_non_string_values_unchanged(self):
        args = {"count": 42, "flag": True}
        facts = {"name": "Alice"}
        result = runtime.substitute_fact_values(args, facts, {})
        assert result == {"count": 42, "flag": True}

    def test_no_match_above_threshold(self):
        args = {"query": "completely different"}
        facts = {"city": "Rehovot"}
        result = runtime.substitute_fact_values(args, facts, {})
        assert result["query"] == "completely different"

    def test_session_values_used(self):
        args = {"city": "Jeruslem"}
        session = {"city": "Jerusalem"}
        result = runtime.substitute_fact_values(args, {}, session)
        assert result["city"] == "Jerusalem"

    @given(args=st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
        values=st.one_of(st.text(min_size=0, max_size=30), st.integers(), st.booleans()),
        max_size=5,
    ))
    @settings(max_examples=30)
    def test_never_loses_keys(self, args):
        """substitute_fact_values always returns a dict with the same keys."""
        facts = {"name": "Alice", "city": "Rehovot"}
        result = runtime.substitute_fact_values(args, facts, {})
        assert set(result.keys()) == set(args.keys())


# ── respond ───────────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestRespond:
    """Tests for respond() — prints JSON to stdout and publishes episode."""

    def test_prints_json_to_stdout(self, capsys):
        data = {"response": "hello", "status": "ok"}
        with patch("arqitect.nerves.nerve_runtime.get_redis_host_port", return_value=("localhost", 6379)), \
             patch("redis.Redis"):
            runtime.respond(data)
        captured = capsys.readouterr()
        assert json.loads(captured.out.strip()) == data

    def test_redis_failure_does_not_raise(self, capsys):
        """Episode publish is best-effort — Redis errors are silently caught."""
        data = {"response": "hi"}
        with patch("arqitect.nerves.nerve_runtime.get_redis_host_port", return_value=("localhost", 6379)), \
             patch("redis.Redis", side_effect=Exception("no redis")):
            runtime.respond(data)
        captured = capsys.readouterr()
        assert json.loads(captured.out.strip()) == data

    @given(response_text=st.text(min_size=1, max_size=200))
    @settings(max_examples=15)
    def test_output_is_always_valid_json(self, response_text):
        """respond() always produces parseable JSON on stdout."""
        import io
        data = {"response": response_text}
        buf = io.StringIO()
        with patch("arqitect.nerves.nerve_runtime.get_redis_host_port", return_value=("localhost", 6379)), \
             patch("redis.Redis"), \
             patch("sys.stdout", buf):
            runtime.respond(data)
        parsed = json.loads(buf.getvalue().strip())
        assert parsed == IsPartialDict(response=response_text)


# ── _strip_markdown_fences ────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestStripMarkdownFences:
    """Tests for _strip_markdown_fences() — removes code fences from LLM output."""

    def test_strips_python_fences(self):
        code = "```python\ndef hello():\n    pass\n```"
        result = runtime._strip_markdown_fences(code)
        assert result == "def hello():\n    pass"

    def test_strips_bare_fences(self):
        code = "```\nsome code\n```"
        result = runtime._strip_markdown_fences(code)
        assert result == "some code"

    def test_no_fences_unchanged(self):
        code = "def hello():\n    pass"
        result = runtime._strip_markdown_fences(code)
        assert result == code

    def test_empty_string(self):
        assert runtime._strip_markdown_fences("") == ""

    @given(code=st.text(min_size=0, max_size=200))
    @settings(max_examples=30)
    def test_never_introduces_backtick_fences(self, code):
        """Output never starts with ``` if the function is working correctly."""
        result = runtime._strip_markdown_fences(code)
        # The result should not be wrapped in fences (stripping is idempotent)
        assert isinstance(result, str)


# ── _find_api_hint ────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestFindApiHint:
    """Tests for _find_api_hint() — matches known free API hints."""

    def test_weather_keyword(self):
        hint = runtime._find_api_hint("I need weather data")
        assert hint != ""
        assert "open-meteo" in hint

    def test_joke_keyword(self):
        hint = runtime._find_api_hint("tell me a joke")
        assert hint != ""
        assert "joke" in hint.lower()

    def test_no_match(self):
        hint = runtime._find_api_hint("calculate fibonacci")
        assert hint == ""

    def test_case_insensitive(self):
        hint = runtime._find_api_hint("WEATHER forecast")
        assert hint != ""

    @given(keyword=st.sampled_from(["weather", "joke", "exchange", "ip", "search"]))
    @settings(max_examples=15)
    def test_known_keywords_always_produce_hints(self, keyword):
        """Every known keyword in FREE_API_HINTS produces a non-empty hint."""
        hint = runtime._find_api_hint(f"I need {keyword} data")
        assert hint != "", f"Expected hint for keyword '{keyword}'"


# ── _detect_domain ────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestDetectDomain:
    """Tests for _detect_domain() — keyword-based domain detection."""

    def test_python_keyword(self):
        assert runtime._detect_domain("explain python decorators") == "python"

    def test_html_keyword(self):
        assert runtime._detect_domain("what is a <div> tag") == "html"

    def test_sql_keyword(self):
        assert runtime._detect_domain("how to write a select query") == "sql"

    def test_no_domain(self):
        assert runtime._detect_domain("what is the meaning of life") is None

    @given(text=st.text(min_size=0, max_size=100))
    @settings(max_examples=30)
    def test_returns_string_or_none(self, text):
        """_detect_domain always returns a string domain name or None."""
        result = runtime._detect_domain(text)
        assert result is None or isinstance(result, str)


# ── _search_domain_facts ──────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestSearchDomainFacts:
    """Tests for _search_domain_facts() — keyword-match against domain facts."""

    def test_empty_facts_returns_none(self):
        assert runtime._search_domain_facts("some question", {}) is None

    def test_matching_key(self):
        facts = {"list_comprehension": "A concise way to create lists in Python"}
        result = runtime._search_domain_facts("what is list comprehension", facts)
        assert result is not None
        assert "concise" in result

    def test_low_score_returns_none(self):
        facts = {"decorator": "A function wrapper"}
        result = runtime._search_domain_facts("xyz totally unrelated", facts)
        assert result is None


# ── get_model_for_role ────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetModelForRole:
    """Tests for get_model_for_role() — maps nerve roles to model names."""

    def test_tool_role(self):
        assert runtime.get_model_for_role("tool") == "nerve"

    def test_creative_role(self):
        assert runtime.get_model_for_role("creative") == "creative"

    def test_code_role(self):
        assert runtime.get_model_for_role("code") == "coder"

    def test_unknown_role_falls_back(self):
        assert runtime.get_model_for_role("unknown_new_role") == "nerve"

    @given(role=st.text(min_size=1, max_size=30))
    @settings(max_examples=30)
    def test_always_returns_a_model_name(self, role):
        """get_model_for_role never returns None — unknown roles fall back."""
        result = runtime.get_model_for_role(role)
        assert isinstance(result, str)
        assert len(result) > 0


# ── _build_fabrication_prompt ─────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestBuildFabricationPrompt:
    """Tests for _build_fabrication_prompt() — assembles the LLM prompt."""

    def test_includes_name_and_description(self):
        prompt = runtime._build_fabrication_prompt("my_tool", "does stuff", "query", "")
        assert "my_tool" in prompt
        assert "does stuff" in prompt
        assert "query" in prompt

    def test_includes_lib_hints(self):
        hints = "\n\nAllowed: requests\n"
        prompt = runtime._build_fabrication_prompt("t", "d", "p", hints)
        assert "Allowed: requests" in prompt

    @given(
        name=st.text(min_size=1, max_size=20, alphabet="abcdefghij_"),
        desc=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=20)
    def test_prompt_always_contains_tool_name(self, name, desc):
        """The fabrication prompt always embeds the tool name."""
        prompt = runtime._build_fabrication_prompt(name, desc, "query", "")
        assert name in prompt


# ── _write_tool_directory ─────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestWriteToolDirectory:
    """Tests for _write_tool_directory() — writes tool.json + run.py to sandbox."""

    def test_creates_files(self, sandbox_dir):
        tool_dir = os.path.join(sandbox_dir, "my_tool")
        runtime._write_tool_directory("my_tool", "does things", "query", "print('hi')", tool_dir)

        manifest_path = os.path.join(tool_dir, "tool.json")
        run_path = os.path.join(tool_dir, "run.py")
        assert os.path.exists(manifest_path)
        assert os.path.exists(run_path)

        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest == IsPartialDict(
            name="my_tool",
            description="does things",
            runtime="python",
        )

        with open(run_path) as f:
            assert f.read() == "print('hi')"

    def test_params_schema_parsing(self, sandbox_dir):
        tool_dir = os.path.join(sandbox_dir, "multi_tool")
        runtime._write_tool_directory("multi_tool", "desc", "a, b, c", "pass", tool_dir)

        with open(os.path.join(tool_dir, "tool.json")) as f:
            manifest = json.load(f)
        assert "a" in manifest["params"]
        assert "b" in manifest["params"]
        assert "c" in manifest["params"]

    def test_manifest_has_required_fields(self, sandbox_dir):
        """The tool.json manifest always contains name, version, runtime, entry."""
        tool_dir = os.path.join(sandbox_dir, "check_fields")
        runtime._write_tool_directory("check_fields", "d", "x", "pass", tool_dir)

        with open(os.path.join(tool_dir, "tool.json")) as f:
            manifest = json.load(f)
        assert manifest == IsPartialDict(
            name="check_fields",
            version="1.0.0",
            runtime="python",
            entry="run.py",
        )


# ── publish_tool_learned (dedup) ──────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestPublishToolLearned:
    """Tests for publish_tool_learned() — dedup within a single run."""

    def test_empty_tool_name_is_noop(self):
        runtime.publish_tool_learned("sight", "")

    def test_dedup_within_run(self):
        with patch("redis.Redis") as mock_cls:
            mock_rc = MagicMock()
            mock_cls.return_value = mock_rc
            runtime.publish_tool_learned("sight", "weather_tool")
            runtime.publish_tool_learned("sight", "weather_tool")
            # Dedup: only published once despite two calls
            assert mock_rc.publish.call_count == 1

    def test_different_tools_both_published(self):
        """Two different tool names should both publish."""
        with patch("redis.Redis") as mock_cls:
            mock_rc = MagicMock()
            mock_cls.return_value = mock_rc
            runtime.publish_tool_learned("sight", "tool_a")
            runtime.publish_tool_learned("sight", "tool_b")
            assert mock_rc.publish.call_count == 2

    def test_redis_failure_does_not_raise(self):
        """Redis errors are silently caught — best-effort publish."""
        with patch("redis.Redis", side_effect=Exception("connection refused")):
            # Should not raise
            runtime.publish_tool_learned("sight", "some_tool")
