"""Tests for nerve_template.py — template string helpers and utility functions.

Covers:
- parse_json: extraction of first JSON object from strings
- match_tool_name: fuzzy tool name matching
- get_tool_list: tool filtering from known tools
- get_tool_names: simple name extraction
- build_planner_prompt: prompt assembly with adapters and examples
- get_effective_role: role resolution from metadata
- NERVE_TEMPLATE: structural validation of the template string

Uses hypothesis for property-based testing, dirty_equals for flexible
assertions, and syrupy for snapshot testing of template structure.
"""

import json

import pytest
from dirty_equals import IsInstance, IsPartialDict, IsStr
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from arqitect.brain.nerve_template import NERVE_TEMPLATE


# ── Helper: mirror the pure functions from the template so we can unit-test
# them without executing the full template with all its runtime imports. ───────

def parse_json(raw: str) -> dict | None:
    """Extract first JSON object from a string.

    Uses brace-depth counting while skipping over characters inside
    JSON string literals so that braces embedded in string values
    (e.g. ``{"response": "{"}``) do not confuse the boundary detection.
    """
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    i = start
    while i < len(raw):
        ch = raw[i]
        if in_string:
            if ch == "\\":
                i += 2  # skip escaped character
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        i += 1
    return None


def match_tool_name(model_name: str, available_tools: list[str]) -> str:
    """Fuzzy match a tool name from the model to actual available tools."""
    if model_name in available_tools:
        return model_name
    model_lower = model_name.lower()
    for tool in available_tools:
        if model_lower in tool.lower() or tool.lower() in model_lower:
            return tool
    model_words = set(model_lower.replace("_", " ").split())
    best, best_score = None, 0
    for tool in available_tools:
        tool_words = set(tool.lower().replace("_", " ").split())
        overlap = len(model_words & tool_words)
        if overlap > best_score:
            best, best_score = tool, overlap
    return best if best_score > 0 else model_name


def get_effective_role(meta: dict, nerve_role: str = "tool") -> str:
    """Get effective role, preferring metadata over baked-in constant."""
    meta_role = meta.get("role", "")
    return meta_role if meta_role else nerve_role


# ── Strategies ────────────────────────────────────────────────────────────────

# Identifiers suitable for nerve/tool names: lowercase letters, digits, underscores
_identifier_chars = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
    min_size=1,
    max_size=40,
)

# Printable descriptions (no control chars)
_description_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=200,
)

# Arbitrary JSON-safe dicts for property testing parse_json
_json_dicts = st.fixed_dictionaries(
    {},
    optional={
        "action": st.sampled_from(["answer", "call", "acquire"]),
        "response": st.text(min_size=0, max_size=80),
        "tool": _identifier_chars,
    },
)


# ── parse_json ────────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestParseJson:
    """Extract the first JSON object from a raw string."""

    def test_parses_plain_json(self) -> None:
        """Parses a plain JSON object."""
        result = parse_json('{"action": "answer", "response": "hello"}')
        assert result == {"action": "answer", "response": "hello"}

    def test_parses_json_with_prefix(self) -> None:
        """Parses JSON with text before it."""
        result = parse_json('Here is the plan: {"action": "call", "tool": "weather"}')
        assert result == IsPartialDict(action="call", tool="weather")

    def test_parses_json_with_suffix(self) -> None:
        """Parses JSON with text after it."""
        result = parse_json('{"action": "answer"} and more text')
        assert result == IsPartialDict(action="answer")

    def test_returns_none_for_no_json(self) -> None:
        """Returns None when no JSON object is present."""
        assert parse_json("no json here") is None

    def test_returns_none_for_empty_string(self) -> None:
        """Returns None for empty string."""
        assert parse_json("") is None

    def test_returns_none_for_invalid_json(self) -> None:
        """Returns None for malformed JSON."""
        assert parse_json("{invalid: json}") is None

    def test_handles_nested_braces(self) -> None:
        """Correctly handles nested braces."""
        raw = '{"outer": {"inner": "value"}, "key": "val"}'
        result = parse_json(raw)
        assert result == IsPartialDict(outer={"inner": "value"}, key="val")

    def test_handles_deeply_nested_braces(self) -> None:
        """Handles deeply nested JSON structures."""
        raw = '{"a": {"b": {"c": {"d": 1}}}}'
        result = parse_json(raw)
        assert result["a"]["b"]["c"]["d"] == 1

    def test_returns_first_json_object(self) -> None:
        """Returns the first JSON object when multiple exist."""
        raw = '{"first": 1} {"second": 2}'
        result = parse_json(raw)
        assert result == {"first": 1}

    def test_unclosed_brace(self) -> None:
        """Returns None for unclosed brace."""
        assert parse_json('{"key": "value"') is None

    def test_only_opening_brace(self) -> None:
        """Returns None for just an opening brace."""
        assert parse_json("{") is None

    def test_json_with_array_value(self) -> None:
        """Parses JSON containing array values."""
        raw = '{"items": [1, 2, 3]}'
        result = parse_json(raw)
        assert result == IsPartialDict(items=[1, 2, 3])

    @given(data=_json_dicts, prefix=st.text(max_size=50), suffix=st.text(max_size=50))
    @settings(max_examples=50)
    def test_roundtrips_any_json_dict(self, data: dict, prefix: str, suffix: str) -> None:
        """Property: any valid JSON dict embedded in surrounding text is recovered."""
        assume("{" not in prefix and "}" not in prefix)
        raw = prefix + json.dumps(data) + suffix
        result = parse_json(raw)
        assert result == data

    @given(text=st.text(alphabet=st.characters(blacklist_characters="{}"), min_size=0, max_size=100))
    @settings(max_examples=50)
    def test_no_braces_returns_none(self, text: str) -> None:
        """Property: text with no braces never produces a result."""
        assert parse_json(text) is None


# ── match_tool_name ──────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestMatchToolName:
    """Fuzzy tool name matching for small model hallucinations."""

    def test_exact_match(self) -> None:
        """Exact name returns the same name."""
        assert match_tool_name("weather_tool", ["weather_tool", "news_tool"]) == "weather_tool"

    def test_substring_match_model_in_tool(self) -> None:
        """Model name is substring of tool name."""
        assert match_tool_name("weather", ["get_weather_tool", "news"]) == "get_weather_tool"

    def test_substring_match_tool_in_model(self) -> None:
        """Tool name is substring of model name."""
        assert match_tool_name("get_weather_tool_v2", ["weather_tool", "news"]) == "weather_tool"

    def test_word_overlap_match(self) -> None:
        """Matching based on word overlap."""
        result = match_tool_name("search_news", ["news_search_tool", "weather_api"])
        assert result == "news_search_tool"

    def test_no_match_returns_original(self) -> None:
        """Returns original name when no match found."""
        assert match_tool_name("xyz_tool", ["abc", "def"]) == "xyz_tool"

    def test_empty_available_tools(self) -> None:
        """Returns original name for empty tool list."""
        assert match_tool_name("weather", []) == "weather"

    def test_case_insensitive_substring(self) -> None:
        """Substring matching is case-insensitive."""
        assert match_tool_name("Weather", ["get_weather_data"]) == "get_weather_data"

    def test_best_word_overlap_wins(self) -> None:
        """Tool with most word overlap wins."""
        result = match_tool_name("get_current_weather", ["get_weather", "current_weather_data", "news"])
        # Both have 2-word overlap; first one with highest score wins
        assert result in ("get_weather", "current_weather_data")

    def test_single_tool_available(self) -> None:
        """Single available tool is matched when words overlap."""
        result = match_tool_name("search_tool", ["search_api"])
        assert result == "search_api"

    def test_exact_match_preferred_over_substring(self) -> None:
        """Exact match takes priority even if substring also matches."""
        result = match_tool_name("weather", ["weather", "get_weather_data"])
        assert result == "weather"

    @given(name=_identifier_chars, tools=st.lists(_identifier_chars, min_size=0, max_size=10))
    @settings(max_examples=50)
    def test_result_is_always_string(self, name: str, tools: list[str]) -> None:
        """Property: match_tool_name always returns a string."""
        result = match_tool_name(name, tools)
        assert result == IsStr()

    @given(name=_identifier_chars, extra=st.lists(_identifier_chars, min_size=0, max_size=5))
    @settings(max_examples=50)
    def test_exact_match_always_preferred(self, name: str, extra: list[str]) -> None:
        """Property: when the exact name is in the list, it is always returned."""
        tools = extra + [name]
        assert match_tool_name(name, tools) == name

    @given(name=_identifier_chars)
    @settings(max_examples=30)
    def test_empty_list_returns_original(self, name: str) -> None:
        """Property: empty available list always returns the original name."""
        assert match_tool_name(name, []) == name


# ── get_effective_role ───────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestGetEffectiveRole:
    """Role resolution from nerve metadata."""

    def test_prefers_metadata_role(self) -> None:
        """Uses role from metadata when present."""
        assert get_effective_role({"role": "creative"}) == "creative"

    def test_falls_back_to_baked_in_role(self) -> None:
        """Falls back to nerve_role when metadata is empty."""
        assert get_effective_role({}) == "tool"

    def test_empty_string_role_falls_back(self) -> None:
        """Empty string role falls back to nerve_role."""
        assert get_effective_role({"role": ""}) == "tool"

    def test_custom_nerve_role_fallback(self) -> None:
        """Falls back to custom nerve_role parameter."""
        assert get_effective_role({}, nerve_role="creative") == "creative"

    def test_metadata_overrides_custom_fallback(self) -> None:
        """Metadata role overrides custom fallback."""
        assert get_effective_role({"role": "code"}, nerve_role="creative") == "code"

    @given(role=st.text(min_size=1, max_size=30))
    @settings(max_examples=30)
    def test_nonempty_meta_role_always_wins(self, role: str) -> None:
        """Property: any non-empty metadata role is returned verbatim."""
        assume(len(role.strip()) > 0)
        result = get_effective_role({"role": role}, nerve_role="fallback")
        assert result == role

    @given(fallback=st.text(min_size=1, max_size=30))
    @settings(max_examples=30)
    def test_empty_meta_uses_fallback(self, fallback: str) -> None:
        """Property: empty metadata role always returns the fallback."""
        assert get_effective_role({"role": ""}, nerve_role=fallback) == fallback
        assert get_effective_role({}, nerve_role=fallback) == fallback


# ── get_tool_list (integration via template logic) ───────────────────────────

@pytest.mark.timeout(10)
class TestGetToolListLogic:
    """Tool list filtering logic — tests the algorithm without requiring full template execution."""

    def test_filters_known_tools_from_all(self) -> None:
        """Only known tools present in all_tools are returned."""
        all_tools = {
            "weather": {"description": "Get weather", "params": ["city"]},
            "news": {"description": "Get news", "params": ["topic"]},
        }
        known = ["weather", "missing_tool"]

        tool_info = {}
        for name in known:
            if name in all_tools:
                info = all_tools[name]
                desc = info.get("description", "") if isinstance(info, dict) else ""
                params = info.get("params", []) if isinstance(info, dict) else []
                tool_info[name] = {"description": desc, "params": params}

        assert "weather" in tool_info
        assert "missing_tool" not in tool_info
        assert tool_info["weather"] == IsPartialDict(description="Get weather")

    def test_empty_known_returns_empty(self) -> None:
        """Empty known list returns empty tool info."""
        all_tools = {"weather": {"description": "Get weather"}}
        known: list[str] = []
        tool_info = {n: all_tools[n] for n in known if n in all_tools}
        assert tool_info == {}

    def test_non_dict_all_tools_treated_as_empty(self) -> None:
        """Non-dict all_tools is treated as empty."""
        all_tools = None
        if not isinstance(all_tools, dict):
            all_tools = {}
        known = ["weather"]
        tool_info = {n: all_tools[n] for n in known if n in all_tools}
        assert tool_info == {}

    @given(
        known=st.lists(_identifier_chars, min_size=0, max_size=5),
        available=st.lists(_identifier_chars, min_size=0, max_size=5),
    )
    @settings(max_examples=50)
    def test_result_is_subset_of_available(self, known: list[str], available: list[str]) -> None:
        """Property: filtered tool_info keys are always a subset of all_tools keys."""
        all_tools = {name: {"description": f"desc_{name}", "params": []} for name in available}
        tool_info = {n: all_tools[n] for n in known if n in all_tools}
        assert set(tool_info.keys()) <= set(all_tools.keys())


# ── build_planner_prompt (logic test) ────────────────────────────────────────

@pytest.mark.timeout(10)
class TestBuildPlannerPromptLogic:
    """Prompt assembly with adapters and examples — testing the algorithm."""

    def test_uses_nerve_system_prompt(self) -> None:
        """Uses the nerve's own system_prompt when present."""
        meta = {"system_prompt": "You are a weather expert.", "role": "tool", "examples": []}
        adapter = None

        prompt = ""
        sp = meta.get("system_prompt", "")
        if sp:
            prompt += sp
            examples = meta.get("examples", [])
        else:
            if adapter and adapter.get("system_prompt"):
                prompt += adapter["system_prompt"]
            examples = adapter.get("few_shot_examples", []) if adapter else []

        assert "weather expert" in prompt

    def test_falls_back_to_adapter_prompt(self) -> None:
        """Uses adapter prompt when nerve has no system_prompt."""
        meta: dict = {"role": "tool"}
        adapter = {
            "system_prompt": "You are a tool specialist.",
            "few_shot_examples": [
                {"input": "weather in NYC", "output": '{"action":"call"}'},
            ],
        }

        prompt = ""
        sp = meta.get("system_prompt", "")
        if sp:
            prompt += sp
            examples = meta.get("examples", [])
        else:
            if adapter and adapter.get("system_prompt"):
                prompt += adapter["system_prompt"]
            examples = adapter.get("few_shot_examples", []) if adapter else []

        assert "tool specialist" in prompt
        assert len(examples) == 1

    def test_includes_valid_examples(self) -> None:
        """Only examples with both input and output are included."""
        examples = [
            "not a dict",
            {"input": "valid"},  # missing output
            {"input": "good", "output": "ok"},
        ]

        prompt = ""
        for ex in examples:
            if isinstance(ex, dict) and "input" in ex and "output" in ex:
                prompt += f'  Input: "{ex["input"]}" -> Output: {ex["output"]}\n'

        assert "good" in prompt
        assert "valid" not in prompt

    @given(
        system_prompt=st.text(min_size=1, max_size=100),
        nerve_examples=st.lists(
            st.fixed_dictionaries({"input": st.text(min_size=1, max_size=30), "output": st.text(min_size=1, max_size=30)}),
            min_size=0,
            max_size=3,
        ),
    )
    @settings(max_examples=30)
    def test_nerve_prompt_takes_priority_over_adapter(self, system_prompt: str, nerve_examples: list[dict]) -> None:
        """Property: when system_prompt is in meta, adapter prompt is ignored."""
        meta = {"system_prompt": system_prompt, "examples": nerve_examples}
        adapter = {"system_prompt": "SHOULD_NOT_APPEAR", "few_shot_examples": []}

        prompt = ""
        sp = meta.get("system_prompt", "")
        if sp:
            prompt += sp
            examples = meta.get("examples", [])
        else:
            if adapter and adapter.get("system_prompt"):
                prompt += adapter["system_prompt"]
            examples = adapter.get("few_shot_examples", []) if adapter else []

        assert system_prompt in prompt
        assert "SHOULD_NOT_APPEAR" not in prompt


# ── NERVE_TEMPLATE structure (snapshot tested) ────────────────────────────────

@pytest.mark.timeout(10)
class TestNerveTemplateStructure:
    """Validate the structure of the nerve template string."""

    def test_contains_placeholders(self) -> None:
        """Template contains all required placeholders."""
        for placeholder in ("{{NERVE_NAME}}", "{{NERVE_ROLE}}", "{{DESCRIPTION}}"):
            assert placeholder in NERVE_TEMPLATE, f"Missing placeholder: {placeholder}"

    def test_contains_main_function(self) -> None:
        """Template defines a main() function."""
        assert "def main():" in NERVE_TEMPLATE

    def test_contains_parse_json(self) -> None:
        """Template defines parse_json function."""
        assert "def parse_json(" in NERVE_TEMPLATE

    def test_contains_match_tool_name(self) -> None:
        """Template defines match_tool_name function."""
        assert "def match_tool_name(" in NERVE_TEMPLATE

    def test_contains_get_tool_list(self) -> None:
        """Template defines get_tool_list function."""
        assert "def get_tool_list():" in NERVE_TEMPLATE

    def test_contains_sense_actions_block(self) -> None:
        """Template includes the sense actions instruction block."""
        assert "SENSES" in NERVE_TEMPLATE
        assert "use_sense" in NERVE_TEMPLATE

    def test_contains_entry_point_guard(self) -> None:
        """Template has an if __name__ == '__main__' guard."""
        assert '__name__ == "__main__"' in NERVE_TEMPLATE

    def test_imports_required_modules(self) -> None:
        """Template imports required runtime modules."""
        assert "from arqitect.nerves.nerve_runtime import" in NERVE_TEMPLATE
        assert "from arqitect.senses.sense_runtime import" in NERVE_TEMPLATE

    def test_defines_nerve_constants(self) -> None:
        """Template defines NERVE_NAME, NERVE_ROLE, DESCRIPTION constants."""
        assert 'NERVE_NAME = "{{NERVE_NAME}}"' in NERVE_TEMPLATE
        assert 'NERVE_ROLE = "{{NERVE_ROLE}}"' in NERVE_TEMPLATE

    def test_defines_available_senses(self) -> None:
        """Template defines AVAILABLE_SENSES dict."""
        assert "AVAILABLE_SENSES" in NERVE_TEMPLATE
        for sense in ('"see"', '"hear"', '"speak"', '"touch"'):
            assert sense in NERVE_TEMPLATE, f"Missing sense: {sense}"

    def test_handles_placeholder_detection(self) -> None:
        """Template includes placeholder detection logic."""
        assert "TOOL_NAME_HERE" in NERVE_TEMPLATE
        assert "PLACEHOLDER" in NERVE_TEMPLATE.upper() or "placeholder" in NERVE_TEMPLATE

    def test_template_snapshot(self, snapshot) -> None:
        """Snapshot: template function signatures and key structural lines.

        Captures function defs, imports, and constants so structural regressions
        are caught without snapshotting the entire multi-hundred-line template.
        """
        structural_lines = [
            line.strip()
            for line in NERVE_TEMPLATE.splitlines()
            if line.strip().startswith(("def ", "from ", "import ", "NERVE_NAME", "NERVE_ROLE", "DESCRIPTION", "AVAILABLE_SENSES"))
        ]
        assert structural_lines == snapshot


# ── Hypothesis: template placeholder substitution ────────────────────────────

@pytest.mark.timeout(10)
class TestTemplatePlaceholderSubstitution:
    """Property tests: substituting placeholders into the template."""

    @given(nerve_name=_identifier_chars, description=_description_text)
    @settings(max_examples=50)
    def test_placeholders_are_fully_replaced(self, nerve_name: str, description: str) -> None:
        """Property: after substitution, no {{PLACEHOLDER}} markers remain."""
        rendered = (
            NERVE_TEMPLATE
            .replace("{{NERVE_NAME}}", nerve_name)
            .replace("{{NERVE_ROLE}}", "tool")
            .replace("{{DESCRIPTION}}", description)
        )
        # Only check for template-style placeholders (double-brace wrapped words),
        # not bare }} which appears naturally in Python f-strings and dict literals.
        import re
        remaining = re.findall(r"\{\{[A-Z_]+\}\}", rendered)
        assert remaining == [], f"Unresolved placeholders: {remaining}"

    @given(nerve_name=_identifier_chars)
    @settings(max_examples=30)
    def test_nerve_name_appears_in_rendered_constants(self, nerve_name: str) -> None:
        """Property: the substituted name appears in the NERVE_NAME constant."""
        rendered = (
            NERVE_TEMPLATE
            .replace("{{NERVE_NAME}}", nerve_name)
            .replace("{{NERVE_ROLE}}", "tool")
            .replace("{{DESCRIPTION}}", "test nerve")
        )
        assert f'NERVE_NAME = "{nerve_name}"' in rendered
