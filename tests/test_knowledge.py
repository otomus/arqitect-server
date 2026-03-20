"""Tests for domain_indexer and project_profiler — deterministic project scanning and domain detection.

Covers:
- Domain detection from keywords
- is_indexed checks against SQLite
- URL fetching with HTML stripping
- LLM fact extraction and JSON parse
- Fact storage in SQLite with upsert
- index_domain pipeline branches
- Project profiling: package.json, pyproject.toml, Cargo.toml, go.mod
- Structure scanning (directories, extensions, entry points)
- Convention detection (linters, Docker, CI, monorepo)
- Profile formatting for prompt injection
- store_profile and get_stored_profile round-trips
"""

import json
import os
import sqlite3
import tempfile
from unittest.mock import patch

import pytest
import responses
from dirty_equals import IsInstance, IsNonNegative
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from arqitect.knowledge.domain_indexer import (
    AUTHORITY_URLS,
    DOMAIN_MAP,
    detect_domain,
    is_indexed,
    _fetch_url,
    _extract_facts_with_llm,
    _store_facts,
    index_domain,
)
from arqitect.knowledge.project_profiler import (
    _read_json,
    _read_toml,
    _detect_from_package_json,
    _detect_from_pyproject,
    _detect_from_cargo,
    _detect_from_go_mod,
    _scan_structure,
    _detect_conventions,
    profile_project,
    format_profile_for_prompt,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _create_facts_table(db_path: str) -> None:
    """Create the facts table in a SQLite database at db_path."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            UNIQUE(category, key)
        )
    """)
    conn.commit()
    conn.close()


def _insert_fact(db_path: str, category: str, key: str, value: str) -> None:
    """Insert a single fact row into the facts table."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO facts (category, key, value) VALUES (?, ?, ?)",
        (category, key, value),
    )
    conn.commit()
    conn.close()


def _query_facts(db_path: str, category: str) -> list[tuple]:
    """Return all (key, value) pairs for a given category."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT key, value FROM facts WHERE category=?", (category,)
    ).fetchall()
    conn.close()
    return rows


# ── Domain Detection ─────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestDetectDomain:
    """Keyword-based domain detection from user input."""

    def test_detects_html_keyword(self):
        """HTML keywords are detected correctly."""
        assert detect_domain("How do I use <div> elements?") == "html"

    def test_detects_python_keyword(self):
        """Python keyword detected."""
        assert detect_domain("write a python script") == "python"

    def test_detects_css_keyword(self):
        """CSS keyword detected."""
        assert detect_domain("How does flexbox work?") == "css"

    def test_detects_javascript_keyword(self):
        """JavaScript keyword detected."""
        assert detect_domain("explain async await in JavaScript") == "javascript"

    def test_detects_sql_keyword(self):
        """SQL keyword detected."""
        assert detect_domain("how to write a select query") == "sql"

    def test_detects_git_keyword(self):
        """Git keyword detected."""
        assert detect_domain("how to rebase a branch") == "git"

    def test_returns_none_for_unrecognized_input(self):
        """Returns None when no domain keywords match."""
        assert detect_domain("what is the meaning of life") is None

    def test_case_insensitive(self):
        """Detection is case-insensitive."""
        assert detect_domain("PYTHON is great") == "python"

    def test_empty_input(self):
        """Empty string returns None."""
        assert detect_domain("") is None

    @given(keyword=st.sampled_from(list(DOMAIN_MAP.keys())))
    @settings(max_examples=30)
    def test_every_keyword_maps_to_its_domain(self, keyword: str):
        """Every keyword in DOMAIN_MAP triggers its declared domain."""
        expected_domain = DOMAIN_MAP[keyword]
        result = detect_domain(f"tell me about {keyword} please")
        assert result == expected_domain

    @given(text=st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll")), min_size=1, max_size=20))
    @settings(max_examples=30)
    def test_returns_string_or_none(self, text: str):
        """detect_domain always returns a str or None — never raises."""
        result = detect_domain(text)
        assert result is None or isinstance(result, str)


# ── is_indexed ────────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestIsIndexed:
    """Check whether domain facts already exist in SQLite."""

    def test_returns_false_when_db_missing(self, tmp_path):
        """Returns False when the database file does not exist."""
        with patch("arqitect.knowledge.domain_indexer._DB_PATH", str(tmp_path / "nonexistent.db")):
            assert is_indexed("python") is False

    def test_returns_false_for_empty_db(self, tmp_path):
        """Returns False when the database exists but has no matching facts."""
        db_path = str(tmp_path / "knowledge.db")
        _create_facts_table(db_path)

        with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path):
            assert is_indexed("python") is False

    def test_returns_true_when_facts_exist(self, tmp_path):
        """Returns True when matching domain facts exist."""
        db_path = str(tmp_path / "knowledge.db")
        _create_facts_table(db_path)
        _insert_fact(db_path, "domain:python", "list_comprehension", "Compact syntax for lists")

        with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path):
            assert is_indexed("python") is True

    def test_returns_false_on_table_missing(self, tmp_path):
        """Returns False when the facts table does not exist."""
        db_path = str(tmp_path / "knowledge.db")
        sqlite3.connect(db_path).close()

        with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path):
            assert is_indexed("python") is False


# ── _fetch_url ────────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestFetchUrl:
    """URL fetching with HTML stripping."""

    @responses.activate
    def test_strips_html_tags(self):
        """HTML tags are stripped from the response."""
        responses.add(
            responses.GET, "https://example.com",
            body="<html><body><p>Hello World</p></body></html>",
            status=200,
        )
        result = _fetch_url("https://example.com")
        assert "Hello World" in result
        assert "<p>" not in result

    @responses.activate
    def test_strips_script_and_style_tags(self):
        """Script and style tags and their content are removed."""
        responses.add(
            responses.GET, "https://example.com",
            body="<html><script>alert('x')</script><style>.x{}</style><p>Clean</p></html>",
            status=200,
        )
        result = _fetch_url("https://example.com")
        assert "alert" not in result
        assert "Clean" in result

    @responses.activate
    def test_truncates_to_5000_chars(self):
        """Result is truncated to 5000 characters."""
        long_body = "<html><body>" + "a" * 10000 + "</body></html>"
        responses.add(responses.GET, "https://example.com", body=long_body, status=200)
        result = _fetch_url("https://example.com")
        assert len(result) <= 5000

    @responses.activate
    def test_returns_error_on_http_failure(self):
        """Returns error string on HTTP failure."""
        responses.add(responses.GET, "https://example.com", status=500)
        result = _fetch_url("https://example.com")
        assert result.startswith("Error:")

    @responses.activate
    def test_returns_error_on_network_error(self):
        """Returns error string on connection error."""
        responses.add(
            responses.GET, "https://example.com",
            body=ConnectionError("refused"),
        )
        result = _fetch_url("https://example.com")
        assert result.startswith("Error:")


# ── _extract_facts_with_llm ──────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestExtractFactsWithLLM:
    """LLM-based fact extraction from source text."""

    def test_parses_valid_json_array(self):
        """Extracts facts from a well-formed JSON array response."""
        facts_json = json.dumps([
            {"key": "div_purpose", "value": "Generic container element"},
            {"key": "span_usage", "value": "Inline container for text"},
        ])
        with patch("arqitect.inference.router.generate_for_role", return_value=facts_json):
            facts = _extract_facts_with_llm("html", "some html reference text")
        assert len(facts) == 2
        assert facts[0]["key"] == "div_purpose"

    def test_handles_code_fenced_json(self):
        """Extracts facts from markdown code-fenced JSON."""
        facts_json = json.dumps([{"key": "test", "value": "works"}])
        fenced = f"```json\n{facts_json}\n```"
        with patch("arqitect.inference.router.generate_for_role", return_value=fenced):
            facts = _extract_facts_with_llm("html", "text")
        assert len(facts) == 1

    def test_filters_invalid_entries(self):
        """Entries missing key or value are filtered out."""
        facts_json = json.dumps([
            {"key": "valid", "value": "yes"},
            {"key": "", "value": "no key"},
            {"value": "no key field"},
            {"key": "missing_value"},
        ])
        with patch("arqitect.inference.router.generate_for_role", return_value=facts_json):
            facts = _extract_facts_with_llm("html", "text")
        assert len(facts) == 1
        assert facts[0]["key"] == "valid"

    def test_returns_empty_on_invalid_json(self):
        """Returns empty list when LLM returns non-JSON."""
        with patch("arqitect.inference.router.generate_for_role", return_value="not json"):
            facts = _extract_facts_with_llm("html", "text")
        assert facts == []

    def test_returns_empty_on_exception(self):
        """Returns empty list when generate_for_role raises."""
        with patch("arqitect.inference.router.generate_for_role", side_effect=Exception("boom")):
            facts = _extract_facts_with_llm("html", "text")
        assert facts == []


# ── _store_facts ──────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestStoreFacts:
    """Fact storage in SQLite with upsert semantics."""

    def test_creates_table_and_stores_facts(self, tmp_path):
        """Facts are stored in a newly created table."""
        db_path = str(tmp_path / "knowledge.db")
        with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path):
            _store_facts("python", [
                {"key": "List Comprehension", "value": "Compact syntax"},
                {"key": "decorator", "value": "Function wrapper"},
            ])

        rows = _query_facts(db_path, "domain:python")
        assert len(rows) == 2
        keys = {r[0] for r in rows}
        assert "list_comprehension" in keys
        assert "decorator" in keys

    def test_upsert_updates_existing_facts(self, tmp_path):
        """Storing a fact with an existing key updates the value."""
        db_path = str(tmp_path / "knowledge.db")
        with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path):
            _store_facts("python", [{"key": "test", "value": "original"}])
            _store_facts("python", [{"key": "test", "value": "updated"}])

        rows = _query_facts(db_path, "domain:python")
        assert len(rows) == 1
        assert rows[0][1] == "updated"

    def test_normalizes_keys(self, tmp_path):
        """Keys are lowercased and spaces replaced with underscores."""
        db_path = str(tmp_path / "knowledge.db")
        with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path):
            _store_facts("python", [{"key": " My Key Name ", "value": "test"}])

        rows = _query_facts(db_path, "domain:python")
        assert rows[0][0] == "my_key_name"

    @given(
        key=st.text(
            alphabet=st.characters(whitelist_categories=("L", "Nd", "Zs")),
            min_size=1, max_size=30,
        ),
        value=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_store_and_retrieve_roundtrip(self, tmp_path, key, value):
        """Any key/value pair survives a store-then-query round-trip."""
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "knowledge.db")
            with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path):
                _store_facts("roundtrip", [{"key": key, "value": value}])

            rows = _query_facts(db_path, "domain:roundtrip")
            assert len(rows) >= 1
            stored_value = rows[0][1]
            assert stored_value == value.strip()


# ── index_domain ──────────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestIndexDomain:
    """Full indexing pipeline for a domain."""

    def test_skips_already_indexed_domain(self, tmp_path, capsys):
        """Skips indexing when domain already has facts."""
        db_path = str(tmp_path / "knowledge.db")
        _create_facts_table(db_path)
        _insert_fact(db_path, "domain:python", "k", "v")

        with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path):
            index_domain("python")

        captured = capsys.readouterr()
        assert "already indexed" in captured.out

    @responses.activate
    def test_fetches_from_authority_url(self, tmp_path):
        """Uses authority URL for known domains."""
        db_path = str(tmp_path / "knowledge.db")
        url = AUTHORITY_URLS["python"]
        responses.add(responses.GET, url, body="<p>Python docs</p>", status=200)

        with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path), \
             patch("arqitect.knowledge.domain_indexer._extract_facts_with_llm",
                   return_value=[{"key": "test_fact", "value": "test value"}]):
            index_domain("python")

        rows = _query_facts(db_path, "domain:python")
        assert len(rows) == 1

    @responses.activate
    def test_falls_back_to_duckduckgo_for_unknown_domain(self, tmp_path):
        """Uses DuckDuckGo for domains without authority URLs."""
        db_path = str(tmp_path / "knowledge.db")
        responses.add(
            responses.GET, "https://api.duckduckgo.com/",
            json={"AbstractText": "Kotlin is a programming language", "RelatedTopics": []},
            status=200,
        )
        with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path), \
             patch("arqitect.knowledge.domain_indexer._extract_facts_with_llm",
                   return_value=[{"key": "kotlin", "value": "JVM language"}]):
            index_domain("kotlin")

        rows = _query_facts(db_path, "domain:kotlin")
        assert len(rows) == 1

    def test_returns_early_when_no_facts_extracted(self, tmp_path, capsys):
        """Returns early when LLM extracts no facts."""
        db_path = str(tmp_path / "knowledge.db")
        with patch("arqitect.knowledge.domain_indexer._DB_PATH", db_path), \
             patch("arqitect.knowledge.domain_indexer._fetch_url", return_value="some text"), \
             patch("arqitect.knowledge.domain_indexer._extract_facts_with_llm", return_value=[]):
            index_domain("python")

        captured = capsys.readouterr()
        assert "No facts extracted" in captured.out


# ── Project Profiler: _read_json / _read_toml ────────────────────────────────

@pytest.mark.timeout(10)
class TestReadHelpers:
    """JSON and TOML file reading helpers."""

    def test_read_json_valid(self, tmp_path):
        """Reads a valid JSON file."""
        p = str(tmp_path / "test.json")
        with open(p, "w") as f:
            json.dump({"key": "value"}, f)
        assert _read_json(p) == {"key": "value"}

    def test_read_json_invalid(self, tmp_path):
        """Returns empty dict for invalid JSON."""
        p = str(tmp_path / "test.json")
        with open(p, "w") as f:
            f.write("not json")
        assert _read_json(p) == {}

    def test_read_json_missing_file(self):
        """Returns empty dict for missing file."""
        assert _read_json("/nonexistent/path.json") == {}

    def test_read_toml_missing_file(self):
        """Returns empty dict for missing TOML file."""
        assert _read_toml("/nonexistent/path.toml") == {}

    @given(data=st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L",))),
        values=st.text(min_size=0, max_size=50),
        min_size=0, max_size=5,
    ))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_read_json_roundtrip(self, tmp_path, data):
        """Any JSON-serializable dict survives a write-then-read round-trip."""
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "roundtrip.json")
            with open(p, "w") as f:
                json.dump(data, f)
            assert _read_json(p) == data


# ── Project Profiler: _detect_from_package_json ──────────────────────────────

@pytest.mark.timeout(10)
class TestDetectFromPackageJson:
    """Extract project facts from package.json."""

    def test_detects_react_typescript_project(self, tmp_path):
        """Detects React + TypeScript from dependencies."""
        pj = {
            "dependencies": {"react": "^18.0.0", "react-dom": "^18.0.0"},
            "devDependencies": {"typescript": "^5.0.0", "vite": "^5.0.0"},
            "scripts": {"dev": "vite", "build": "tsc && vite build"},
        }
        with open(tmp_path / "package.json", "w") as f:
            json.dump(pj, f)

        facts = _detect_from_package_json(str(tmp_path))
        assert facts["language"] == "typescript"
        assert facts["framework"] == "react"
        assert facts["bundler"] == "vite"

    def test_detects_next_js(self, tmp_path):
        """Next.js takes priority over React."""
        pj = {"dependencies": {"next": "^14", "react": "^18"}, "devDependencies": {}}
        with open(tmp_path / "package.json", "w") as f:
            json.dump(pj, f)

        facts = _detect_from_package_json(str(tmp_path))
        assert facts["framework"] == "next.js"

    def test_detects_tailwind_styling(self, tmp_path):
        """Detects Tailwind CSS from devDependencies."""
        pj = {"dependencies": {}, "devDependencies": {"tailwindcss": "^3.0"}}
        with open(tmp_path / "package.json", "w") as f:
            json.dump(pj, f)

        facts = _detect_from_package_json(str(tmp_path))
        assert facts["styling"] == "tailwind"

    def test_detects_jest_test_framework(self, tmp_path):
        """Detects Jest test framework."""
        pj = {"dependencies": {}, "devDependencies": {"jest": "^29"}}
        with open(tmp_path / "package.json", "w") as f:
            json.dump(pj, f)

        facts = _detect_from_package_json(str(tmp_path))
        assert facts["test_framework"] == "jest"

    def test_detects_pnpm_package_manager(self, tmp_path):
        """Detects pnpm from lock file."""
        with open(tmp_path / "package.json", "w") as f:
            json.dump({"dependencies": {}, "devDependencies": {}}, f)
        (tmp_path / "pnpm-lock.yaml").write_text("")

        facts = _detect_from_package_json(str(tmp_path))
        assert facts["pkg_manager"] == "pnpm"

    def test_detects_zustand_state_management(self, tmp_path):
        """Detects Zustand state management."""
        pj = {"dependencies": {"zustand": "^4"}, "devDependencies": {}}
        with open(tmp_path / "package.json", "w") as f:
            json.dump(pj, f)

        facts = _detect_from_package_json(str(tmp_path))
        assert facts["state_mgmt"] == "zustand"

    def test_returns_empty_for_missing_package_json(self, tmp_path):
        """Returns empty dict when package.json does not exist."""
        facts = _detect_from_package_json(str(tmp_path))
        assert facts == {}

    def test_includes_scripts(self, tmp_path):
        """Script names are included in facts."""
        pj = {"dependencies": {}, "devDependencies": {}, "scripts": {"build": "tsc", "test": "jest"}}
        with open(tmp_path / "package.json", "w") as f:
            json.dump(pj, f)

        facts = _detect_from_package_json(str(tmp_path))
        assert "build" in facts["scripts"]
        assert "test" in facts["scripts"]

    def test_includes_entry_point(self, tmp_path):
        """Main entry point is included."""
        pj = {"dependencies": {}, "devDependencies": {}, "main": "index.js"}
        with open(tmp_path / "package.json", "w") as f:
            json.dump(pj, f)

        facts = _detect_from_package_json(str(tmp_path))
        assert facts["entry_point"] == "index.js"


# ── Project Profiler: _detect_from_pyproject ─────────────────────────────────

@pytest.mark.timeout(10)
class TestDetectFromPyproject:
    """Extract project facts from pyproject.toml or requirements.txt."""

    def test_detects_fastapi_framework(self, tmp_path):
        """Detects FastAPI from pyproject.toml dependencies."""
        toml_content = b'[project]\ndependencies = ["fastapi>=0.100"]\n[build-system]\nrequires = ["hatchling"]\n'
        with open(tmp_path / "pyproject.toml", "wb") as f:
            f.write(toml_content)

        facts = _detect_from_pyproject(str(tmp_path))
        assert facts["language"] == "python"
        assert facts["framework"] == "fastapi"
        assert facts["build_tool"] == "hatch"

    def test_detects_pytest_from_tool_section(self, tmp_path):
        """Detects pytest from [tool.pytest] in pyproject.toml."""
        toml_content = b'[project]\ndependencies = []\n[tool.pytest.ini_options]\naddopts = "-v"\n'
        with open(tmp_path / "pyproject.toml", "wb") as f:
            f.write(toml_content)

        facts = _detect_from_pyproject(str(tmp_path))
        assert facts["test_framework"] == "pytest"

    def test_detects_uv_package_manager(self, tmp_path):
        """Detects uv from uv.lock file."""
        (tmp_path / "pyproject.toml").write_bytes(b"[project]\ndependencies = []\n")
        (tmp_path / "uv.lock").write_text("")

        facts = _detect_from_pyproject(str(tmp_path))
        assert facts["pkg_manager"] == "uv"

    def test_detects_poetry_package_manager(self, tmp_path):
        """Detects poetry from poetry.lock file."""
        (tmp_path / "pyproject.toml").write_bytes(b"[project]\ndependencies = []\n")
        (tmp_path / "poetry.lock").write_text("")

        facts = _detect_from_pyproject(str(tmp_path))
        assert facts["pkg_manager"] == "poetry"

    def test_falls_back_to_requirements_txt(self, tmp_path):
        """Falls back to requirements.txt when pyproject.toml has no deps."""
        (tmp_path / "requirements.txt").write_text("flask>=2.0\nrequests\n")

        facts = _detect_from_pyproject(str(tmp_path))
        assert facts["language"] == "python"
        assert "flask" in facts["dependencies"]
        assert facts.get("framework") == "flask"

    def test_detects_venv(self, tmp_path):
        """Detects .venv directory."""
        (tmp_path / "pyproject.toml").write_bytes(b"[project]\ndependencies = []\n")
        os.makedirs(tmp_path / ".venv")

        facts = _detect_from_pyproject(str(tmp_path))
        assert facts["venv"] == ".venv"

    def test_no_pyproject_defaults_to_pip(self, tmp_path):
        """Defaults to pip when no lock files exist."""
        facts = _detect_from_pyproject(str(tmp_path))
        assert facts["pkg_manager"] == "pip"


# ── Project Profiler: _detect_from_cargo ─────────────────────────────────────

@pytest.mark.timeout(10)
class TestDetectFromCargo:
    """Extract project facts from Cargo.toml."""

    def test_detects_rust_with_actix(self, tmp_path):
        """Detects Rust + Actix from Cargo.toml."""
        toml_content = b'[dependencies]\nactix-web = "4"\nserde = "1"\n'
        with open(tmp_path / "Cargo.toml", "wb") as f:
            f.write(toml_content)

        facts = _detect_from_cargo(str(tmp_path))
        assert facts["language"] == "rust"
        assert facts["framework"] == "actix"
        assert "actix-web" in facts["dependencies"]

    def test_returns_empty_without_cargo_toml(self, tmp_path):
        """Returns empty dict when Cargo.toml is missing."""
        facts = _detect_from_cargo(str(tmp_path))
        assert facts == {}


# ── Project Profiler: _detect_from_go_mod ────────────────────────────────────

@pytest.mark.timeout(10)
class TestDetectFromGoMod:
    """Extract project facts from go.mod."""

    def test_detects_go_with_gin(self, tmp_path):
        """Detects Go + Gin from go.mod."""
        go_mod = (
            "module github.com/example/myapp\n\n"
            "go 1.21\n\n"
            "require (\n"
            "\tgithub.com/gin-gonic/gin v1.9.0\n"
            "\tgithub.com/stretchr/testify v1.8.0\n"
            ")\n"
        )
        (tmp_path / "go.mod").write_text(go_mod)

        facts = _detect_from_go_mod(str(tmp_path))
        assert facts["language"] == "go"
        assert facts["module"] == "github.com/example/myapp"
        assert facts["framework"] == "gin"

    def test_returns_empty_without_go_mod(self, tmp_path):
        """Returns empty dict when go.mod is missing."""
        facts = _detect_from_go_mod(str(tmp_path))
        assert facts == {}

    def test_detects_fiber_framework(self, tmp_path):
        """Detects Fiber framework from go.mod."""
        go_mod = (
            "module example.com/app\n\n"
            "require (\n"
            "\tgithub.com/gofiber/fiber/v2 v2.50.0\n"
            ")\n"
        )
        (tmp_path / "go.mod").write_text(go_mod)

        facts = _detect_from_go_mod(str(tmp_path))
        assert facts["framework"] == "fiber"


# ── Project Profiler: _scan_structure ────────────────────────────────────────

@pytest.mark.timeout(10)
class TestScanStructure:
    """Directory structure scanning."""

    def test_finds_top_directories(self, tmp_path):
        """Top-level directories are listed."""
        os.makedirs(tmp_path / "src")
        os.makedirs(tmp_path / "tests")
        (tmp_path / "main.py").write_text("pass")

        facts = _scan_structure(str(tmp_path))
        assert "src" in facts["top_dirs"]
        assert "tests" in facts["top_dirs"]

    def test_counts_file_extensions(self, tmp_path):
        """File extensions are counted correctly."""
        for i in range(3):
            (tmp_path / f"file{i}.py").write_text("pass")
        (tmp_path / "style.css").write_text("")

        facts = _scan_structure(str(tmp_path))
        assert ".py(" in facts["file_types"]

    def test_skips_noise_directories(self, tmp_path):
        """Directories like node_modules and .git are skipped."""
        os.makedirs(tmp_path / "node_modules" / "pkg")
        os.makedirs(tmp_path / ".git" / "objects")
        os.makedirs(tmp_path / "src")
        (tmp_path / "src" / "app.py").write_text("pass")

        facts = _scan_structure(str(tmp_path))
        top = facts.get("top_dirs", "")
        assert "node_modules" not in top
        assert ".git" not in top

    def test_detects_entry_candidates(self, tmp_path):
        """Common entry point files are detected."""
        (tmp_path / "main.py").write_text("pass")

        facts = _scan_structure(str(tmp_path))
        assert "main.py" in facts.get("entry_candidates", "")

    def test_respects_max_depth(self, tmp_path):
        """Does not scan below max_depth."""
        deep = tmp_path / "a" / "b" / "c" / "d"
        os.makedirs(deep)
        (deep / "deep.py").write_text("pass")

        facts = _scan_structure(str(tmp_path), max_depth=2)
        # deep.py should not appear in file_types since it's too deep
        file_types = facts.get("file_types", "")
        assert "deep.py" not in file_types

    def test_empty_directory(self, tmp_path):
        """Empty directory returns minimal facts."""
        facts = _scan_structure(str(tmp_path))
        assert "top_dirs" not in facts or facts["top_dirs"] == ""


# ── Project Profiler: _detect_conventions ────────────────────────────────────

@pytest.mark.timeout(10)
class TestDetectConventions:
    """Convention detection from config files."""

    def test_detects_eslint(self, tmp_path):
        """Detects ESLint from config file."""
        (tmp_path / ".eslintrc.json").write_text("{}")
        facts = _detect_conventions(str(tmp_path))
        assert "eslint" in facts["linter"]

    def test_detects_docker(self, tmp_path):
        """Detects Docker from Dockerfile."""
        (tmp_path / "Dockerfile").write_text("FROM python:3.11")
        facts = _detect_conventions(str(tmp_path))
        assert facts["containerized"] == "docker"

    def test_detects_github_actions_ci(self, tmp_path):
        """Detects GitHub Actions CI."""
        os.makedirs(tmp_path / ".github" / "workflows")
        facts = _detect_conventions(str(tmp_path))
        assert facts["ci"] == "github-actions"

    def test_detects_monorepo(self, tmp_path):
        """Detects monorepo from turbo.json."""
        (tmp_path / "turbo.json").write_text("{}")
        facts = _detect_conventions(str(tmp_path))
        assert facts["monorepo"] == "true"

    def test_detects_tsconfig(self, tmp_path):
        """Detects tsconfig.json."""
        (tmp_path / "tsconfig.json").write_text("{}")
        facts = _detect_conventions(str(tmp_path))
        assert facts["tsconfig"] == "tsconfig.json"

    def test_no_conventions_detected(self, tmp_path):
        """Returns empty dict when no convention files exist."""
        facts = _detect_conventions(str(tmp_path))
        assert facts == {}

    def test_deduplicates_linters(self, tmp_path):
        """Multiple ESLint configs count as one."""
        (tmp_path / ".eslintrc.json").write_text("{}")
        (tmp_path / "eslint.config.js").write_text("")
        facts = _detect_conventions(str(tmp_path))
        assert facts["linter"].count("eslint") == 1


# ── Project Profiler: profile_project ────────────────────────────────────────

@pytest.mark.timeout(10)
class TestProfileProject:
    """Full project profiling integration."""

    def test_profiles_python_project(self, tmp_path):
        """Profiles a Python project with pyproject.toml."""
        toml_content = b'[project]\ndependencies = ["fastapi"]\n[build-system]\nrequires = ["hatchling"]\n'
        with open(tmp_path / "pyproject.toml", "wb") as f:
            f.write(toml_content)
        os.makedirs(tmp_path / "src")

        facts = profile_project(str(tmp_path))
        assert facts["language"] == "python"
        assert facts["name"] == tmp_path.name
        assert facts["path"] == str(tmp_path)

    def test_returns_error_for_nonexistent_path(self):
        """Returns error dict for nonexistent path."""
        facts = profile_project("/nonexistent/path")
        assert "error" in facts

    def test_guesses_language_from_extensions(self, tmp_path):
        """Guesses language from file extensions when no config files exist."""
        for i in range(3):
            (tmp_path / f"file{i}.py").write_text("pass")

        facts = profile_project(str(tmp_path))
        assert facts.get("language") == "python"

    def test_first_detector_wins(self, tmp_path):
        """First matching config file detector wins."""
        with open(tmp_path / "package.json", "w") as f:
            json.dump({"dependencies": {"react": "^18"}, "devDependencies": {}}, f)
        (tmp_path / "pyproject.toml").write_bytes(b'[project]\ndependencies = []\n')

        facts = profile_project(str(tmp_path))
        assert facts["language"] == "javascript"

    def test_profile_always_includes_name_and_path(self, tmp_path):
        """Every successful profile contains name and path keys."""
        facts = profile_project(str(tmp_path))
        assert facts["name"] == IsInstance(str)
        assert facts["path"] == str(tmp_path)


# ── Project Profiler: format_profile_for_prompt ──────────────────────────────

@pytest.mark.timeout(10)
class TestFormatProfileForPrompt:
    """Formatting project facts for LLM prompt injection."""

    def test_formats_complete_profile(self):
        """All fields are included in formatted output."""
        facts = {
            "name": "myapp", "path": "/app",
            "language": "typescript", "framework": "react",
            "bundler": "vite", "pkg_manager": "pnpm",
            "test_framework": "vitest", "linter": "eslint",
            "ci": "github-actions", "top_dirs": "src,tests",
            "file_types": ".ts(10),.tsx(5)",
            "dependencies": "react,react-dom",
        }
        result = format_profile_for_prompt(facts)
        assert "myapp" in result
        assert "typescript + react + vite" in result
        assert "pnpm" in result
        assert "vitest" in result

    def test_returns_empty_for_empty_facts(self):
        """Returns empty string for empty or None facts."""
        assert format_profile_for_prompt({}) == ""
        assert format_profile_for_prompt(None) == ""

    def test_returns_empty_for_error_facts(self):
        """Returns empty string for error facts."""
        assert format_profile_for_prompt({"error": "not a dir"}) == ""

    def test_minimal_profile(self):
        """Handles facts with only name and path."""
        facts = {"name": "myapp", "path": "/app"}
        result = format_profile_for_prompt(facts)
        assert "myapp" in result
        assert "/app" in result

    @given(
        name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "Nd"))),
        path=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "Nd", "P"))),
    )
    @settings(max_examples=15)
    def test_format_never_raises(self, name, path):
        """format_profile_for_prompt never raises for valid name/path dicts."""
        result = format_profile_for_prompt({"name": name, "path": path})
        assert isinstance(result, str)
        assert len(result) == IsNonNegative
