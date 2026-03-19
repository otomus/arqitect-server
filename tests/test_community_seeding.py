"""Tests for community nerve and tool seeding.

Covers:
- seed_nerves bootstraps new nerves from manifest
- seed_nerves rewires tools for existing nerves
- seed_nerves skips nerves that already have nerve.py on disk
- _rewire_nerve_tools fills gaps without duplicating
- synthesize_nerve uses community bundle when available
"""

import json
import os
from unittest.mock import patch

import pytest

from tests.conftest import make_nerve_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _stub_network():
    """Prevent real HTTP calls during tests."""
    with patch("arqitect.brain.community.urllib.request.urlretrieve"), \
         patch("arqitect.brain.community.urllib.request.urlopen"):
        yield


def _write_manifest(tmp_path, nerves: dict) -> str:
    """Write a manifest to a temp cache dir and return the path."""
    cache_dir = tmp_path / ".community" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "1.0",
        "nerves": nerves,
        "tools": {},
        "mcps": {},
        "adapters": {},
    }
    path = cache_dir / "manifest.json"
    path.write_text(json.dumps(manifest))
    return str(path)


# ---------------------------------------------------------------------------
# _rewire_nerve_tools
# ---------------------------------------------------------------------------

class TestRewireNerveTools:
    """Tool wiring from manifest declarations."""

    def test_adds_missing_tools(self, mem):
        """Tools declared in manifest but missing from cold memory are added."""
        mem.cold.register_nerve("joke_nerve", "Jokes")
        mem.cold.add_nerve_tool("joke_nerve", "existing_tool")

        manifest_info = {"tools": ["existing_tool", "new_tool", "another_tool"]}

        from arqitect.brain.community import _rewire_nerve_tools
        _rewire_nerve_tools("joke_nerve", manifest_info, mem.cold)

        tools = mem.cold.get_nerve_tools("joke_nerve")
        assert "existing_tool" in tools
        assert "new_tool" in tools
        assert "another_tool" in tools

    def test_no_duplicates(self, mem):
        """Already-wired tools are not re-added."""
        mem.cold.register_nerve("joke_nerve", "Jokes")
        mem.cold.add_nerve_tool("joke_nerve", "tool_a")

        manifest_info = {"tools": ["tool_a"]}

        from arqitect.brain.community import _rewire_nerve_tools
        _rewire_nerve_tools("joke_nerve", manifest_info, mem.cold)

        tools = mem.cold.get_nerve_tools("joke_nerve")
        assert tools.count("tool_a") == 1

    def test_empty_tools_list_is_noop(self, mem):
        """No tools declared means no wiring happens."""
        mem.cold.register_nerve("joke_nerve", "Jokes")

        from arqitect.brain.community import _rewire_nerve_tools
        _rewire_nerve_tools("joke_nerve", {"tools": []}, mem.cold)
        _rewire_nerve_tools("joke_nerve", {}, mem.cold)

        assert mem.cold.get_nerve_tools("joke_nerve") == []


# ---------------------------------------------------------------------------
# seed_nerves
# ---------------------------------------------------------------------------

class TestSeedNerves:
    """Bootstrap community nerves at startup."""

    def test_creates_nerve_files_for_new_nerves(self, mem, nerves_dir, tmp_path):
        """Nerves in manifest that don't exist on disk get created."""
        manifest_path = _write_manifest(tmp_path, {
            "test_nerve": {
                "description": "A test nerve",
                "role": "tool",
                "tools": ["test_tool"],
            },
        })

        from arqitect.brain.community import seed_nerves
        with patch("arqitect.brain.community._manifest_path", return_value=manifest_path), \
             patch("arqitect.brain.community.sync_nerve_bundle", return_value=None), \
             patch("arqitect.brain.config.mem", mem):
            count = seed_nerves()

        assert count == 1
        nerve_path = os.path.join(nerves_dir, "test_nerve", "nerve.py")
        assert os.path.isfile(nerve_path)
        tools = mem.cold.get_nerve_tools("test_nerve")
        assert "test_tool" in tools

    def test_skips_existing_nerves_but_rewires_tools(self, mem, nerves_dir, tmp_path):
        """Existing nerves are not recreated but tools are rewired."""
        make_nerve_file(nerves_dir, "existing_nerve")
        mem.cold.register_nerve("existing_nerve", "Already here")

        manifest_path = _write_manifest(tmp_path, {
            "existing_nerve": {
                "description": "Already here",
                "role": "tool",
                "tools": ["new_community_tool"],
            },
        })

        from arqitect.brain.community import seed_nerves
        with patch("arqitect.brain.community._manifest_path", return_value=manifest_path), \
             patch("arqitect.brain.config.mem", mem):
            count = seed_nerves()

        assert count == 0
        tools = mem.cold.get_nerve_tools("existing_nerve")
        assert "new_community_tool" in tools

    def test_returns_zero_without_manifest(self):
        """No manifest means nothing to seed."""
        from arqitect.brain.community import seed_nerves
        with patch("arqitect.brain.community._load_cached_manifest", return_value=None):
            assert seed_nerves() == 0


# ---------------------------------------------------------------------------
# synthesize_nerve — community-first path
# ---------------------------------------------------------------------------

class TestSynthesizeCommunityFirst:
    """synthesize_nerve uses community bundle when available."""

    def _stub_patches(self):
        """Return patches that stub out heavy synthesis dependencies."""
        class _NoOpThread:
            def __init__(self, *a, **kw):
                pass
            def start(self):
                pass

        return [
            patch("arqitect.brain.synthesis.llm_generate",
                  return_value='{"system_prompt": "stub", "examples": []}'),
            patch("arqitect.brain.synthesis.classify_nerve_role", return_value="tool"),
            patch("arqitect.brain.synthesis.list_mcp_tools_with_info", return_value={}),
            patch("arqitect.brain.synthesis.publish_nerve_status"),
            patch("arqitect.brain.synthesis.publish_event"),
            patch("arqitect.brain.synthesis.threading.Thread", _NoOpThread),
        ]

    def test_community_bundle_uses_bundle_metadata(self, nerves_dir, mem, tmp_path):
        """When a community bundle exists, its metadata is used directly.

        context.json in the cache provides system_prompt and few_shot_examples.
        """
        bundle = {
            "description": "Community jokes",
            "role": "creative",
            "tools": [],
        }

        # Set up cached context.json so resolve_nerve_prompt can find it
        cache_dir = tmp_path / ".community" / "cache"
        nerve_cache = cache_dir / "nerves" / "joke_nerve" / "medium"
        nerve_cache.mkdir(parents=True)
        (nerve_cache / "context.json").write_text(json.dumps({
            "system_prompt": "Be funny",
            "few_shot_examples": [],
            "temperature": 0.7,
        }))

        patches = self._stub_patches()
        with patch("arqitect.brain.synthesis.mem", mem), \
             patch("arqitect.brain.synthesis.find_community_bundle", return_value=bundle), \
             patch("arqitect.brain.community._cache_dir", return_value=str(cache_dir)), \
             patch("arqitect.brain.adapters._cache_dir", return_value=str(cache_dir)), \
             patch("arqitect.brain.adapters.get_active_variant", return_value="medium"), \
             patch("arqitect.brain.adapters.get_model_name_for_role", return_value=None):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.synthesis import synthesize_nerve
                name, path = synthesize_nerve(
                    "joke_nerve", "tell jokes", trigger_task="tell me a joke",
                )
                assert name == "joke_nerve"
                assert os.path.isfile(path)
                meta = mem.cold.get_nerve_metadata("joke_nerve")
                assert meta["system_prompt"] == "Be funny"
                assert meta["role"] == "creative"
            finally:
                for p in patches:
                    p.stop()

    def test_falls_back_to_scratch_without_bundle(self, nerves_dir, mem):
        """Without a community bundle, synthesis proceeds from scratch."""
        patches = self._stub_patches()
        with patch("arqitect.brain.synthesis.mem", mem), \
             patch("arqitect.brain.synthesis.find_community_bundle", return_value=None):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.synthesis import synthesize_nerve
                name, path = synthesize_nerve(
                    "recipe_nerve", "cooking recipes",
                    trigger_task="how to make pasta",
                )
                assert name == "recipe_nerve"
                assert os.path.isfile(path)
            finally:
                for p in patches:
                    p.stop()
