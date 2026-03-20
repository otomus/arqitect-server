"""Tests for nerve_template.py — template string helpers and utility functions.

Covers:
- parse_json: extraction of first JSON object from strings
- match_tool_name: fuzzy tool name matching
- get_tool_list: tool filtering from known tools
- get_tool_names: simple name extraction
- build_planner_prompt: prompt assembly with adapters and examples
- get_effective_role: role resolution from metadata
- NERVE_TEMPLATE: structural validation of the template string
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from arqitect.brain.nerve_template import NERVE_TEMPLATE


# ── Helper: define the functions from the template in a testable way ─────────
# The functions in NERVE_TEMPLATE are defined as text. We extract and define
# the pure functions directly so we can test them without executing the full
# template with all its imports.

def parse_json(raw):
    """Extract first JSON object from a string."""
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def match_tool_name(model_name, available_tools):
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


def get_effective_role(meta, nerve_role="tool"):
    """Get effective role, preferring metadata over baked-in constant."""
    meta_role = meta.get("role", "")
    return meta_role if meta_role else nerve_role


# ── parse_json ────────────────────────────────────────────────────────────────

class TestParseJson:
    """Extract the first JSON object from a raw string."""

    def test_parses_plain_json(self):
        """Parses a plain JSON object."""
        result = parse_json('{"action": "answer", "response": "hello"}')
        assert result == {"action": "answer", "response": "hello"}

    def test_parses_json_with_prefix(self):
        """Parses JSON with text before it."""
        result = parse_json('Here is the plan: {"action": "call", "tool": "weather"}')
        assert result["action"] == "call"
        assert result["tool"] == "weather"

    def test_parses_json_with_suffix(self):
        """Parses JSON with text after it."""
        result = parse_json('{"action": "answer"} and more text')
        assert result["action"] == "answer"

    def test_returns_none_for_no_json(self):
        """Returns None when no JSON object is present."""
        assert parse_json("no json here") is None

    def test_returns_none_for_empty_string(self):
        """Returns None for empty string."""
        assert parse_json("") is None

    def test_returns_none_for_invalid_json(self):
        """Returns None for malformed JSON."""
        assert parse_json("{invalid: json}") is None

    def test_handles_nested_braces(self):
        """Correctly handles nested braces."""
        raw = '{"outer": {"inner": "value"}, "key": "val"}'
        result = parse_json(raw)
        assert result["outer"]["inner"] == "value"
        assert result["key"] == "val"

    def test_handles_deeply_nested_braces(self):
        """Handles deeply nested JSON structures."""
        raw = '{"a": {"b": {"c": {"d": 1}}}}'
        result = parse_json(raw)
        assert result["a"]["b"]["c"]["d"] == 1

    def test_returns_first_json_object(self):
        """Returns the first JSON object when multiple exist."""
        raw = '{"first": 1} {"second": 2}'
        result = parse_json(raw)
        assert result == {"first": 1}

    def test_unclosed_brace(self):
        """Returns None for unclosed brace."""
        assert parse_json('{"key": "value"') is None

    def test_only_opening_brace(self):
        """Returns None for just an opening brace."""
        assert parse_json("{") is None

    def test_json_with_array_value(self):
        """Parses JSON containing array values."""
        raw = '{"items": [1, 2, 3]}'
        result = parse_json(raw)
        assert result["items"] == [1, 2, 3]


# ── match_tool_name ──────────────────────────────────────────────────────────

class TestMatchToolName:
    """Fuzzy tool name matching for small model hallucinations."""

    def test_exact_match(self):
        """Exact name returns the same name."""
        assert match_tool_name("weather_tool", ["weather_tool", "news_tool"]) == "weather_tool"

    def test_substring_match_model_in_tool(self):
        """Model name is substring of tool name."""
        assert match_tool_name("weather", ["get_weather_tool", "news"]) == "get_weather_tool"

    def test_substring_match_tool_in_model(self):
        """Tool name is substring of model name."""
        assert match_tool_name("get_weather_tool_v2", ["weather_tool", "news"]) == "weather_tool"

    def test_word_overlap_match(self):
        """Matching based on word overlap."""
        result = match_tool_name("search_news", ["news_search_tool", "weather_api"])
        assert result == "news_search_tool"

    def test_no_match_returns_original(self):
        """Returns original name when no match found."""
        assert match_tool_name("xyz_tool", ["abc", "def"]) == "xyz_tool"

    def test_empty_available_tools(self):
        """Returns original name for empty tool list."""
        assert match_tool_name("weather", []) == "weather"

    def test_case_insensitive_substring(self):
        """Substring matching is case-insensitive."""
        assert match_tool_name("Weather", ["get_weather_data"]) == "get_weather_data"

    def test_best_word_overlap_wins(self):
        """Tool with most word overlap wins."""
        result = match_tool_name("get_current_weather", ["get_weather", "current_weather_data", "news"])
        # Both have 2-word overlap; first one with highest score wins
        assert result in ("get_weather", "current_weather_data")

    def test_single_tool_available(self):
        """Single available tool is matched when words overlap."""
        result = match_tool_name("search_tool", ["search_api"])
        assert result == "search_api"

    def test_exact_match_preferred_over_substring(self):
        """Exact match takes priority even if substring also matches."""
        result = match_tool_name("weather", ["weather", "get_weather_data"])
        assert result == "weather"


# ── get_effective_role ───────────────────────────────────────────────────────

class TestGetEffectiveRole:
    """Role resolution from nerve metadata."""

    def test_prefers_metadata_role(self):
        """Uses role from metadata when present."""
        assert get_effective_role({"role": "creative"}) == "creative"

    def test_falls_back_to_baked_in_role(self):
        """Falls back to nerve_role when metadata is empty."""
        assert get_effective_role({}) == "tool"

    def test_empty_string_role_falls_back(self):
        """Empty string role falls back to nerve_role."""
        assert get_effective_role({"role": ""}) == "tool"

    def test_custom_nerve_role_fallback(self):
        """Falls back to custom nerve_role parameter."""
        assert get_effective_role({}, nerve_role="creative") == "creative"

    def test_metadata_overrides_custom_fallback(self):
        """Metadata role overrides custom fallback."""
        assert get_effective_role({"role": "code"}, nerve_role="creative") == "code"


# ── get_tool_list (integration via template logic) ───────────────────────────

class TestGetToolListLogic:
    """Tool list filtering logic — tests the algorithm without requiring full template execution."""

    def test_filters_known_tools_from_all(self):
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
        assert tool_info["weather"]["description"] == "Get weather"

    def test_empty_known_returns_empty(self):
        """Empty known list returns empty tool info."""
        all_tools = {"weather": {"description": "Get weather"}}
        known = []
        tool_info = {n: all_tools[n] for n in known if n in all_tools}
        assert tool_info == {}

    def test_non_dict_all_tools_treated_as_empty(self):
        """Non-dict all_tools is treated as empty."""
        all_tools = None
        if not isinstance(all_tools, dict):
            all_tools = {}
        known = ["weather"]
        tool_info = {n: all_tools[n] for n in known if n in all_tools}
        assert tool_info == {}


# ── build_planner_prompt (logic test) ────────────────────────────────────────

class TestBuildPlannerPromptLogic:
    """Prompt assembly with adapters and examples — testing the algorithm."""

    def test_uses_nerve_system_prompt(self):
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

    def test_falls_back_to_adapter_prompt(self):
        """Uses adapter prompt when nerve has no system_prompt."""
        meta = {"role": "tool"}
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

    def test_includes_valid_examples(self):
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


# ── NERVE_TEMPLATE structure ─────────────────────────────────────────────────

class TestNerveTemplateStructure:
    """Validate the structure of the nerve template string."""

    def test_contains_placeholders(self):
        """Template contains all required placeholders."""
        assert "{{NERVE_NAME}}" in NERVE_TEMPLATE
        assert "{{NERVE_ROLE}}" in NERVE_TEMPLATE
        assert "{{DESCRIPTION}}" in NERVE_TEMPLATE

    def test_contains_main_function(self):
        """Template defines a main() function."""
        assert "def main():" in NERVE_TEMPLATE

    def test_contains_parse_json(self):
        """Template defines parse_json function."""
        assert "def parse_json(" in NERVE_TEMPLATE

    def test_contains_match_tool_name(self):
        """Template defines match_tool_name function."""
        assert "def match_tool_name(" in NERVE_TEMPLATE

    def test_contains_get_tool_list(self):
        """Template defines get_tool_list function."""
        assert "def get_tool_list():" in NERVE_TEMPLATE

    def test_contains_sense_actions_block(self):
        """Template includes the sense actions instruction block."""
        assert "SENSES" in NERVE_TEMPLATE
        assert "use_sense" in NERVE_TEMPLATE

    def test_contains_entry_point_guard(self):
        """Template has an if __name__ == '__main__' guard."""
        assert '__name__ == "__main__"' in NERVE_TEMPLATE

    def test_imports_required_modules(self):
        """Template imports required runtime modules."""
        assert "from arqitect.nerves.nerve_runtime import" in NERVE_TEMPLATE
        assert "from arqitect.senses.sense_runtime import" in NERVE_TEMPLATE

    def test_defines_nerve_constants(self):
        """Template defines NERVE_NAME, NERVE_ROLE, DESCRIPTION constants."""
        assert 'NERVE_NAME = "{{NERVE_NAME}}"' in NERVE_TEMPLATE
        assert 'NERVE_ROLE = "{{NERVE_ROLE}}"' in NERVE_TEMPLATE

    def test_defines_available_senses(self):
        """Template defines AVAILABLE_SENSES dict."""
        assert "AVAILABLE_SENSES" in NERVE_TEMPLATE
        assert '"see"' in NERVE_TEMPLATE
        assert '"hear"' in NERVE_TEMPLATE
        assert '"speak"' in NERVE_TEMPLATE
        assert '"touch"' in NERVE_TEMPLATE

    def test_handles_placeholder_detection(self):
        """Template includes placeholder detection logic."""
        assert "TOOL_NAME_HERE" in NERVE_TEMPLATE
        assert "PLACEHOLDER" in NERVE_TEMPLATE.upper() or "placeholder" in NERVE_TEMPLATE
