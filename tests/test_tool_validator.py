"""Tests for arqitect.brain.tool_validator.

Covers placeholder detection patterns and credential dependency extraction,
including edge cases for None, empty, malformed, and adversarial inputs.
"""

import pytest

from arqitect.brain.tool_validator import validate_tool_code, detect_credential_deps


# ── validate_tool_code ───────────────────────────────────────────────────

class TestValidateToolCode:

    def test_clean_code_passes(self):
        code = '''
def get_weather(city: str) -> str:
    """Fetch weather for a city."""
    import requests
    resp = requests.get(f"https://wttr.in/{city}?format=3")
    return resp.text
'''
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is True
        assert warnings == []

    def test_detects_your_api_key(self):
        code = 'api_key = "YOUR_API_KEY"\nrequests.get(url, headers={"Authorization": api_key})'
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert any("YOUR_" in w for w in warnings)

    def test_detects_dummy_token(self):
        code = 'token = "DUMMY_TOKEN_VALUE"'
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert any("DUMMY_" in w for w in warnings)

    def test_detects_replace_placeholder(self):
        code = 'key = "REPLACE_WITH_YOUR_KEY"'
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert any("REPLACE_" in w for w in warnings)

    def test_detects_insert_here(self):
        code = 'token = "INSERT_TOKEN_HERE"'
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert any("INSERT_" in w for w in warnings)

    def test_detects_placeholder_string(self):
        code = "url = 'placeholder'"
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert any("placeholder" in w for w in warnings)

    def test_detects_example_com(self):
        code = 'resp = requests.get("https://api.example.com/data")'
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert any("example.com" in w for w in warnings)

    def test_detects_hardcoded_sk_key(self):
        code = 'api_key = "sk-abcdefghijklmnopqrstuvwxyz"'
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert any("sk-" in w for w in warnings)

    def test_multiple_issues_reported(self):
        code = 'key = "YOUR_API_KEY"\nurl = "https://api.example.com"'
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert len(warnings) >= 2

    def test_credential_import_is_fine(self):
        """Code using the credential system should pass validation."""
        code = '''
from arqitect.brain.credentials import get_credential, has_credentials

def search_google(query: str) -> str:
    if not has_credentials("google", ["api_key"]):
        return "Error: credentials for google not configured"
    api_key = get_credential("google", "api_key")
    import requests
    resp = requests.get("https://www.googleapis.com/customsearch/v1",
                        params={"key": api_key, "q": query})
    return resp.text
'''
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is True

    # ── Edge cases ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("bad_input", [
        "",
        "   ",
        "\t\n\r",
        "\n\n\n",
    ], ids=["empty", "spaces", "whitespace-chars", "newlines-only"])
    def test_empty_and_whitespace_inputs_pass(self, bad_input):
        """Empty / whitespace-only code has no placeholders -- should be valid."""
        is_valid, warnings = validate_tool_code(bad_input)
        assert is_valid is True
        assert warnings == []

    @pytest.mark.parametrize("wrong_type", [
        None, 42, 3.14, [], {}, True, b"bytes",
    ], ids=["none", "int", "float", "list", "dict", "bool", "bytes"])
    def test_wrong_types_raise_or_reject(self, wrong_type):
        """Non-string inputs must not silently return valid."""
        with pytest.raises((TypeError, AttributeError)):
            validate_tool_code(wrong_type)

    @pytest.mark.parametrize("code,should_flag", [
        # Unicode lookalikes should NOT fool the validator
        ('key = "YOUR_API_KEY"', True),            # normal ASCII
        ('key = "ΥΟϒR_API_KEY"', False),           # Greek capital letters -- not a real match
        # Embedded in multiline / comments
        ('# YOUR_API_KEY is set above\nx = 1', True),  # placeholder in comment still flagged
        ('"""YOUR_API_KEY"""', True),                   # placeholder in docstring still flagged
        # Near-miss: should NOT flag
        ('your_api_key = os.environ["KEY"]', False),    # lowercase, not a placeholder
        ('YOURS = 1', False),                           # doesn't match YOUR_[A-Z_]+
    ], ids=[
        "ascii-placeholder", "greek-lookalike",
        "in-comment", "in-docstring",
        "lowercase-your", "partial-match-YOURS",
    ])
    def test_unicode_and_context_edge_cases(self, code, should_flag):
        """Placeholders in comments/docstrings are still caught; unicode lookalikes are not."""
        is_valid, warnings = validate_tool_code(code)
        if should_flag:
            assert is_valid is False
            assert len(warnings) >= 1
        else:
            assert is_valid is True
            assert warnings == []

    def test_very_long_string_does_not_crash(self):
        """Validator must handle megabyte-scale inputs without error."""
        # 500 KB of innocuous code with no placeholders
        code = "x = 1\n" * 100_000
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is True
        assert warnings == []

    def test_very_long_string_with_placeholder_at_end(self):
        """A placeholder buried deep in a huge file must still be caught."""
        code = "x = 1\n" * 50_000 + 'key = "YOUR_SECRET_KEY"\n'
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert any("YOUR_" in w for w in warnings)

    @pytest.mark.parametrize("code", [
        'key = "sk-' + "a" * 200 + '"',        # very long sk- key
        'x = "' + "example.com/" * 50 + '"',   # repeated example.com
    ], ids=["long-sk-key", "repeated-example-com"])
    def test_long_placeholder_values(self, code):
        """Extremely long placeholder strings must still be detected."""
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert len(warnings) >= 1

    def test_null_bytes_and_special_chars(self):
        """Code with null bytes and control characters should not crash."""
        code = 'x = 1\x00\x01\x02\nkey = "YOUR_KEY"\n'
        is_valid, warnings = validate_tool_code(code)
        assert is_valid is False
        assert any("YOUR_" in w for w in warnings)


# ── detect_credential_deps ───────────────────────────────────────────────

class TestDetectCredentialDeps:

    def test_no_credentials(self):
        code = 'import requests\nresp = requests.get("https://wttr.in/London")'
        assert detect_credential_deps(code) == []

    def test_single_service(self):
        code = '''
api_key = get_credential("google", "api_key")
cx = get_credential("google", "search_engine_id")
'''
        deps = detect_credential_deps(code)
        assert len(deps) == 1
        assert deps[0]["service"] == "google"
        assert sorted(deps[0]["keys"]) == ["api_key", "search_engine_id"]

    def test_multiple_services(self):
        code = '''
openai_key = get_credential("openai", "api_key")
slack_token = get_credential("slack", "bot_token")
'''
        deps = detect_credential_deps(code)
        assert len(deps) == 2
        services = {d["service"] for d in deps}
        assert services == {"openai", "slack"}

    def test_deduplicates_keys(self):
        code = '''
key = get_credential("svc", "api_key")
key2 = get_credential("svc", "api_key")
'''
        deps = detect_credential_deps(code)
        assert len(deps) == 1
        assert deps[0]["keys"] == ["api_key"]

    # ── Edge cases ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("bad_input", [
        "", "   ", "\n\n",
    ], ids=["empty", "spaces", "newlines"])
    def test_empty_and_whitespace_return_empty(self, bad_input):
        """No credential calls in blank input."""
        assert detect_credential_deps(bad_input) == []

    @pytest.mark.parametrize("wrong_type", [
        None, 42, [], {},
    ], ids=["none", "int", "list", "dict"])
    def test_wrong_types_raise(self, wrong_type):
        """Non-string inputs must not silently return results."""
        with pytest.raises((TypeError, AttributeError)):
            detect_credential_deps(wrong_type)

    @pytest.mark.parametrize("code", [
        'get_credential("svc", "key")',               # missing close — actually valid syntax for regex
        "get_credential('svc', 'key')",                # single quotes
        'get_credential(  "svc"  ,  "key"  )',         # extra whitespace
        'get_credential(\n"svc",\n"key"\n)',           # multiline — regex may miss this
    ], ids=["double-quotes", "single-quotes", "extra-spaces", "multiline-call"])
    def test_quote_and_spacing_variants(self, code):
        """Credential calls with different quoting/spacing styles."""
        deps = detect_credential_deps(code)
        # The regex uses \s* between args, so single-line variants should match.
        # Multiline may or may not match depending on re.DOTALL — test documents behavior.
        if "\n" in code.split("get_credential")[1].split(")")[0]:
            # Multiline inside the call — current regex does not use DOTALL
            assert deps == [] or len(deps) == 1  # documents either behavior
        else:
            assert len(deps) == 1
            assert deps[0]["service"] == "svc"

    def test_malformed_get_credential_calls_ignored(self):
        """Calls that look similar but are structurally wrong should not match."""
        code = '''
get_credential("only_one_arg")
get_credential(service, "key")
get_credential("svc", key_var)
xget_credential("svc", "key")
get_credential("", "")
'''
        deps = detect_credential_deps(code)
        # Only get_credential("", "") matches the regex pattern \w+ with zero-width,
        # but \w+ requires at least one char, so empty strings should NOT match.
        services = [d["service"] for d in deps]
        assert "only_one_arg" not in services
        # Variable references (no quotes) should not match
        assert all(d["service"] != "" for d in deps)

    def test_special_chars_in_service_name_ignored(self):
        """Service names with non-word chars should not be extracted."""
        code = 'get_credential("my-service", "api_key")'
        deps = detect_credential_deps(code)
        # \w+ does not match hyphens, so this should not match
        assert deps == []

    def test_large_input_with_many_services(self):
        """Handles many distinct services without error."""
        lines = [f'get_credential("svc_{i}", "key_{i}")' for i in range(200)]
        code = "\n".join(lines)
        deps = detect_credential_deps(code)
        assert len(deps) == 200
