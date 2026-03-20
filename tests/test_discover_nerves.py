"""Tests for discover_nerves -- single source of truth for nerve discovery.

discover_nerves() has one job: find ALL nerves from filesystem and return
them as {name: description}. No filtering, no qualification gating -- just
discovery.

Sources:
  1. Filesystem scan (NERVES_DIR) -- already filtered at seed time
  2. Community manifest -- for descriptions
  3. Core senses (SENSES_DIR) -- only enabled ones
"""

import json
import os
from unittest.mock import patch

import pytest
from dirty_equals import IsStr
from hypothesis import given, settings, strategies as st, HealthCheck

from arqitect.brain.catalog import discover_nerves

CORE_SENSES = ("sight", "hearing", "touch", "awareness", "communication")

# Module-level store -- each test writes its result here, autouse fixture checks it.
_last_result = {}


@pytest.fixture(autouse=True)
def assert_senses_in_every_result():
    """After every test, verify all 5 core senses are present in the discovery result."""
    _last_result.clear()
    yield
    if _last_result.get("data") is not None:
        result = _last_result["data"]
        for sense in CORE_SENSES:
            if _last_result.get(f"disabled_{sense}"):
                continue
            assert sense in result, \
                f"Core sense '{sense}' missing from discover_nerves() result: {list(result.keys())}"


def _make_nerve_file(nerves_dir, name):
    """Create a minimal nerve.py on disk."""
    d = os.path.join(nerves_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "nerve.py"), "w") as f:
        f.write("# stub\n")


# ---------------------------------------------------------------------------
# 1. Empty state -- only core senses
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestEmptyState:
    """Empty state should still return core senses."""

    def test_returns_core_senses_when_nothing_else_exists(self, mem, nerves_dir):
        """Even with no user nerves, all 5 core senses must be discovered."""
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result

    def test_returns_dict_of_name_to_description(self, mem, nerves_dir):
        """Return type must be {str: str}."""
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert isinstance(result, dict)
            for name, desc in result.items():
                assert isinstance(name, str)
                assert isinstance(desc, str)
                assert len(desc) > 0, f"Nerve '{name}' has empty description"


# ---------------------------------------------------------------------------
# 2. Filesystem discovery (primary source)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestFilesystemDiscovery:
    """Nerve discovery from the filesystem."""

    def test_nerve_on_disk_is_discovered(self, mem, nerves_dir):
        """A nerve directory with nerve.py on disk should be discovered."""
        _make_nerve_file(nerves_dir, "weather_nerve")
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert "weather_nerve" in result

    def test_multiple_nerves_on_disk(self, mem, nerves_dir):
        """All nerves on disk appear, not just the first."""
        _make_nerve_file(nerves_dir, "weather_nerve")
        _make_nerve_file(nerves_dir, "joke_nerve")
        _make_nerve_file(nerves_dir, "recipe_nerve")
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert "weather_nerve" in result
            assert "joke_nerve" in result
            assert "recipe_nerve" in result

    def test_directory_without_nerve_py_is_ignored(self, mem, nerves_dir):
        """A directory without nerve.py is not a nerve."""
        os.makedirs(os.path.join(nerves_dir, "not_a_nerve"))
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert "not_a_nerve" not in result

    def test_hidden_directories_are_ignored(self, mem, nerves_dir):
        """Dotfiles/directories should be skipped."""
        hidden = os.path.join(nerves_dir, ".hidden_nerve")
        os.makedirs(hidden)
        with open(os.path.join(hidden, "nerve.py"), "w") as f:
            f.write("# stub\n")
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert ".hidden_nerve" not in result

    def test_filesystem_nerve_gets_registered_in_cold(self, mem, nerves_dir):
        """Filesystem-only nerves should be auto-registered in cold memory."""
        _make_nerve_file(nerves_dir, "disk_nerve")
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            all_nerves = mem.cold.list_nerves()
            assert "disk_nerve" in all_nerves

    @given(
        nerve_name=st.from_regex(r"[a-z][a-z0-9_]{2,20}_nerve", fullmatch=True),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_any_valid_nerve_name_is_discovered(self, mem, nerves_dir, nerve_name):
        """Any valid nerve name with nerve.py on disk gets discovered."""
        _make_nerve_file(nerves_dir, nerve_name)
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert nerve_name in result


# ---------------------------------------------------------------------------
# 3. Manifest description enrichment
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestManifestDescriptions:
    """Manifest-driven description enrichment."""

    def test_description_from_manifest(self, mem, nerves_dir, tmp_path):
        """Nerves on disk get descriptions from the manifest when available."""
        _make_nerve_file(nerves_dir, "weather_nerve")

        cache_dir = tmp_path / ".community" / "cache"
        cache_dir.mkdir(parents=True)
        manifest = {
            "nerves": {
                "weather_nerve": {"description": "Fetches weather data"},
            },
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

        with patch("arqitect.brain.catalog.mem", mem), \
             patch("arqitect.brain.community._manifest_path",
                   return_value=str(cache_dir / "manifest.json")):
            result = discover_nerves()
            _last_result["data"] = result
            assert result["weather_nerve"] == "Fetches weather data"

    def test_fallback_to_name_without_manifest(self, mem, nerves_dir):
        """Without a manifest, the nerve name is used as description."""
        _make_nerve_file(nerves_dir, "orphan_nerve")
        with patch("arqitect.brain.catalog.mem", mem), \
             patch("arqitect.brain.community._load_cached_manifest",
                   return_value=None):
            result = discover_nerves()
            _last_result["data"] = result
            assert "orphan_nerve" in result


# ---------------------------------------------------------------------------
# 4. Senses discovery
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestSensesDiscovery:
    """Core sense discovery and registration."""

    def test_senses_have_meaningful_descriptions(self, mem, nerves_dir):
        """Core senses should have real descriptions, not just their name."""
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            for sense in CORE_SENSES:
                assert result[sense] != sense, \
                    f"Sense '{sense}' has placeholder description: '{result[sense]}'"
                assert result[sense] == IsStr(min_length=1)

    def test_senses_are_marked_as_senses_in_cold(self, mem, nerves_dir):
        """After discovery, core senses should be registered as senses in cold memory."""
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            for sense in CORE_SENSES:
                assert mem.cold.is_sense(sense), \
                    f"'{sense}' should be marked as a sense in cold memory"

    def test_disabled_sense_excluded(self, mem, nerves_dir):
        """A sense disabled in config is excluded from discovery."""
        with patch("arqitect.brain.catalog.mem", mem), \
             patch("arqitect.brain.catalog.get_config") as mock_cfg:

            def _cfg(path, default=None):
                if path == "senses.sight.enabled":
                    return False
                return default if default is not None else True

            mock_cfg.side_effect = _cfg

            result = discover_nerves()
            _last_result["data"] = result
            _last_result["disabled_sight"] = True
            assert "sight" not in result


# ---------------------------------------------------------------------------
# 5. Deduplication
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestDeduplication:
    """Deduplication of filesystem and manifest entries."""

    def test_nerve_on_disk_with_manifest_desc_appears_once(self, mem, nerves_dir, tmp_path):
        """A nerve present on disk should appear once with manifest description."""
        _make_nerve_file(nerves_dir, "weather_nerve")

        cache_dir = tmp_path / ".community" / "cache"
        cache_dir.mkdir(parents=True)
        manifest = {
            "nerves": {
                "weather_nerve": {"description": "fetches weather data"},
            },
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))

        with patch("arqitect.brain.catalog.mem", mem), \
             patch("arqitect.brain.community._manifest_path",
                   return_value=str(cache_dir / "manifest.json")):
            result = discover_nerves()
            _last_result["data"] = result
            assert result["weather_nerve"] == "fetches weather data"
