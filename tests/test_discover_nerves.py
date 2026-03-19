"""Tests for discover_nerves — single source of truth for nerve discovery.

discover_nerves() has one job: find ALL nerves from every source and return
them as {name: description}. No filtering, no qualification gating — just
discovery.

Sources:
  1. Cold memory registry (SQLite)
  2. Filesystem scan (NERVES_DIR)
  3. Core senses (SENSES_DIR)
"""

import os
from unittest.mock import patch

import pytest

from arqitect.brain.catalog import discover_nerves

CORE_SENSES = ("sight", "hearing", "touch", "awareness", "communication")

# Module-level store — each test writes its result here, autouse fixture checks it.
_last_result = {}


@pytest.fixture(autouse=True)
def assert_senses_in_every_result():
    """After every test, verify all 5 core senses are present in the discovery result."""
    _last_result.clear()
    yield
    if _last_result.get("data") is not None:
        result = _last_result["data"]
        for sense in CORE_SENSES:
            assert sense in result, \
                f"Core sense '{sense}' missing from discover_nerves() result: {list(result.keys())}"


# ---------------------------------------------------------------------------
# 1. Empty state — only core senses
# ---------------------------------------------------------------------------

class TestEmptyState:
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
# 2. Registry nerves
# ---------------------------------------------------------------------------

class TestRegistryDiscovery:
    def test_registered_nerve_is_discovered(self, mem, nerves_dir):
        """A nerve in cold memory registry should appear in results."""
        mem.cold.register_nerve("weather_nerve", "fetches weather data")
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert "weather_nerve" in result
            assert result["weather_nerve"] == "fetches weather data"

    def test_multiple_registered_nerves(self, mem, nerves_dir):
        """All registered nerves appear, not just the first."""
        mem.cold.register_nerve("weather_nerve", "fetches weather data")
        mem.cold.register_nerve("joke_nerve", "tells jokes")
        mem.cold.register_nerve("recipe_nerve", "cooking recipes")
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert "weather_nerve" in result
            assert "joke_nerve" in result
            assert "recipe_nerve" in result

    def test_unqualified_nerve_still_discovered(self, mem, nerves_dir):
        """discover_nerves does NOT filter by qualification — that's routing's job."""
        mem.cold.register_nerve("new_nerve", "just created, no qual yet")
        # Intentionally NOT recording any qualification
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert "new_nerve" in result, \
                "Unqualified nerves must be discovered — filtering is not discovery's job"


# ---------------------------------------------------------------------------
# 3. Filesystem discovery
# ---------------------------------------------------------------------------

class TestFilesystemDiscovery:
    def test_nerve_on_disk_not_in_registry_is_discovered(self, mem, nerves_dir):
        """A nerve directory with nerve.py on disk should be discovered even if not registered."""
        nerve_dir = os.path.join(nerves_dir, "orphan_nerve")
        os.makedirs(nerve_dir)
        with open(os.path.join(nerve_dir, "nerve.py"), "w") as f:
            f.write("# stub\n")
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert "orphan_nerve" in result

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
        """Filesystem-only nerves should be auto-registered in cold memory for future lookups."""
        nerve_dir = os.path.join(nerves_dir, "disk_nerve")
        os.makedirs(nerve_dir)
        with open(os.path.join(nerve_dir, "nerve.py"), "w") as f:
            f.write("# stub\n")
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            # Verify it was registered
            all_nerves = mem.cold.list_nerves()
            assert "disk_nerve" in all_nerves


# ---------------------------------------------------------------------------
# 4. Senses discovery
# ---------------------------------------------------------------------------

class TestSensesDiscovery:
    def test_senses_have_meaningful_descriptions(self, mem, nerves_dir):
        """Core senses should have real descriptions, not just their name."""
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            for sense in CORE_SENSES:
                # Description should be more than just the sense name
                assert result[sense] != sense, \
                    f"Sense '{sense}' has placeholder description: '{result[sense]}'"

    def test_senses_are_marked_as_senses_in_cold(self, mem, nerves_dir):
        """After discovery, core senses should be registered as senses in cold memory."""
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            for sense in CORE_SENSES:
                assert mem.cold.is_sense(sense), \
                    f"'{sense}' should be marked as a sense in cold memory"


# ---------------------------------------------------------------------------
# 5. Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_nerve_in_registry_and_on_disk_appears_once(self, mem, nerves_dir):
        """A nerve present in both registry and filesystem should not be duplicated."""
        mem.cold.register_nerve("weather_nerve", "fetches weather data")
        nerve_dir = os.path.join(nerves_dir, "weather_nerve")
        os.makedirs(nerve_dir)
        with open(os.path.join(nerve_dir, "nerve.py"), "w") as f:
            f.write("# stub\n")
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            # Should appear exactly once with the registry description (richer)
            assert result["weather_nerve"] == "fetches weather data"

    def test_registry_description_takes_precedence_over_filesystem(self, mem, nerves_dir):
        """If a nerve is in both registry and filesystem, registry description wins."""
        mem.cold.register_nerve("my_nerve", "detailed description from registry")
        nerve_dir = os.path.join(nerves_dir, "my_nerve")
        os.makedirs(nerve_dir)
        with open(os.path.join(nerve_dir, "nerve.py"), "w") as f:
            f.write("# stub\n")
        with patch("arqitect.brain.catalog.mem", mem):
            result = discover_nerves()
            _last_result["data"] = result
            assert result["my_nerve"] == "detailed description from registry"
