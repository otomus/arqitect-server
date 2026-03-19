"""Catalog tests: discover_nerves returns all nerves, senses always present."""

import os
from unittest.mock import patch

import pytest

from arqitect.types import Sense

CORE_SENSES = tuple(Sense)

# Module-level store — each test writes its catalog here, autouse fixture checks it.
_last_catalog = {}


@pytest.fixture(autouse=True)
def assert_senses_always_in_catalog():
    """After every test, verify all 5 core senses are present in the catalog."""
    _last_catalog.clear()
    yield
    if _last_catalog.get("data") is not None:
        catalog = _last_catalog["data"]
        for sense in CORE_SENSES:
            assert sense in catalog, \
                f"Core sense '{sense}' missing from catalog: {list(catalog.keys())}"
            assert catalog[sense], f"Sense '{sense}' has empty description"


def _make_nerve_file(nerves_dir, name):
    d = os.path.join(nerves_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "nerve.py"), "w") as f:
        f.write("# stub\n")


class TestDiscoverNervesIncludesAll:
    def test_registered_nerve_included(self, nerves_dir, mem):
        """Any registered nerve should appear in discover_nerves — no filtering."""
        mem.cold.register_nerve("good_nerve", "does good things")
        _make_nerve_file(nerves_dir, "good_nerve")

        with patch("arqitect.brain.catalog.mem", mem):
            from arqitect.brain.catalog import discover_nerves
            catalog = discover_nerves()
            _last_catalog["data"] = catalog
            assert "good_nerve" in catalog

    def test_unqualified_nerve_still_included(self, nerves_dir, mem):
        """discover_nerves does not filter — unqualified nerves appear too."""
        mem.cold.register_nerve("new_nerve", "just created")
        _make_nerve_file(nerves_dir, "new_nerve")

        with patch("arqitect.brain.catalog.mem", mem):
            from arqitect.brain.catalog import discover_nerves
            catalog = discover_nerves()
            _last_catalog["data"] = catalog
            assert "new_nerve" in catalog

    def test_zero_score_nerve_still_included(self, nerves_dir, mem):
        """Even a nerve that failed qualification is discovered."""
        mem.cold.register_nerve("failed_nerve", "always fails")
        mem.cold.record_qualification("nerve", "failed_nerve", qualified=False, score=0.0, iterations=5, test_count=15, pass_count=0)
        _make_nerve_file(nerves_dir, "failed_nerve")

        with patch("arqitect.brain.catalog.mem", mem):
            from arqitect.brain.catalog import discover_nerves
            catalog = discover_nerves()
            _last_catalog["data"] = catalog
            assert "failed_nerve" in catalog


class TestSensesAlwaysPresent:
    def test_all_five_senses_in_catalog(self, nerves_dir, mem):
        """All 5 core senses must appear regardless of qualification state."""
        with patch("arqitect.brain.catalog.mem", mem):
            from arqitect.brain.catalog import discover_nerves
            catalog = discover_nerves()
            _last_catalog["data"] = catalog

    def test_senses_have_descriptions(self, nerves_dir, mem):
        """Each sense should have a non-empty description."""
        with patch("arqitect.brain.catalog.mem", mem):
            from arqitect.brain.catalog import discover_nerves
            catalog = discover_nerves()
            _last_catalog["data"] = catalog


class TestFilesystemFallback:
    def test_nerve_on_disk_but_not_in_registry_gets_registered(self, nerves_dir, mem):
        """A nerve that exists on disk but not in cold memory should be auto-registered."""
        _make_nerve_file(nerves_dir, "orphan_nerve")

        with patch("arqitect.brain.catalog.mem", mem):
            from arqitect.brain.catalog import discover_nerves
            catalog = discover_nerves()
            _last_catalog["data"] = catalog
            all_nerves = mem.cold.list_nerves()
            assert "orphan_nerve" in all_nerves, \
                f"Orphan nerve should be auto-registered. Registry: {list(all_nerves.keys())}"
