"""Tests for community nerve and tool seeding.

Covers:
- seed_nerves bootstraps new nerves from manifest
- seed_nerves rewires tools for existing nerves
- seed_nerves skips nerves that already have nerve.py on disk
- _rewire_nerve_tools fills gaps without duplicating
- synthesize_nerve uses community bundle when available
- Environment filtering (iot, desktop, server)
"""

import json
import os
from unittest.mock import patch

import pytest

from hypothesis import given, settings, strategies as st, HealthCheck

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
# _matches_environment
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestMatchesEnvironment:
    """Unit tests for _matches_environment tag filtering."""

    def test_no_tags_matches_all(self):
        from arqitect.brain.community import _matches_environment
        assert _matches_environment([], "server")
        assert _matches_environment([], "iot")
        assert _matches_environment([], "desktop")

    def test_non_exclusive_tags_match_all(self):
        from arqitect.brain.community import _matches_environment
        assert _matches_environment(["utility", "general"], "server")
        assert _matches_environment(["utility", "general"], "iot")

    def test_iot_tag_only_matches_iot(self):
        from arqitect.brain.community import _matches_environment
        assert _matches_environment(["iot"], "iot")
        assert not _matches_environment(["iot"], "server")
        assert not _matches_environment(["iot"], "desktop")

    def test_desktop_tag_only_matches_desktop(self):
        from arqitect.brain.community import _matches_environment
        assert _matches_environment(["desktop"], "desktop")
        assert not _matches_environment(["desktop"], "server")
        assert not _matches_environment(["desktop"], "iot")

    def test_mixed_exclusive_and_regular_tags(self):
        from arqitect.brain.community import _matches_environment
        assert _matches_environment(["iot", "sensor", "utility"], "iot")
        assert not _matches_environment(["iot", "sensor", "utility"], "server")

    @given(
        non_exclusive_tags=st.lists(
            st.sampled_from(["utility", "general", "sensor", "api", "data"]),
            min_size=0,
            max_size=5,
        ),
        env=st.sampled_from(["server", "iot", "desktop"]),
    )
    @settings(max_examples=30)
    def test_non_exclusive_tags_always_match(self, non_exclusive_tags, env):
        """Tags that are not in ENV_EXCLUSIVE_TAGS match every environment."""
        from arqitect.brain.community import _matches_environment
        assert _matches_environment(non_exclusive_tags, env)


# ---------------------------------------------------------------------------
# _rewire_nerve_tools
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
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

    @given(
        tool_names=st.lists(
            st.from_regex(r"[a-z_]{3,15}", fullmatch=True),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_all_declared_tools_are_wired(self, mem, tool_names):
        """Every tool in the manifest ends up wired to the nerve."""
        nerve_name = "prop_nerve"
        mem.cold.register_nerve(nerve_name, "Property test nerve")

        from arqitect.brain.community import _rewire_nerve_tools
        _rewire_nerve_tools(nerve_name, {"tools": tool_names}, mem.cold)

        wired = mem.cold.get_nerve_tools(nerve_name)
        for tool in tool_names:
            assert tool in wired


# ---------------------------------------------------------------------------
# seed_nerves
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
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

    def test_filters_iot_nerves_in_server_environment(self, mem, nerves_dir, tmp_path):
        """IoT-tagged nerves should be skipped when environment is server."""
        manifest_path = _write_manifest(tmp_path, {
            "iot_sensor": {
                "description": "IoT sensor control",
                "role": "tool",
                "tools": [],
                "tags": ["iot"],
            },
            "universal_nerve": {
                "description": "Works everywhere",
                "role": "tool",
                "tools": [],
                "tags": ["utility"],
            },
        })

        from arqitect.brain.community import seed_nerves
        with patch("arqitect.brain.community._manifest_path", return_value=manifest_path), \
             patch("arqitect.brain.community.sync_nerve_bundle", return_value=None), \
             patch("arqitect.brain.config.mem", mem), \
             patch("arqitect.brain.community.get_config", return_value="server"):
            count = seed_nerves()

        assert count == 1
        assert os.path.isfile(os.path.join(nerves_dir, "universal_nerve", "nerve.py"))
        assert not os.path.exists(os.path.join(nerves_dir, "iot_sensor", "nerve.py"))

    def test_filters_desktop_nerves_in_iot_environment(self, mem, nerves_dir, tmp_path):
        """Desktop-tagged nerves should be skipped when environment is iot."""
        manifest_path = _write_manifest(tmp_path, {
            "screen_capture": {
                "description": "Desktop screen capture",
                "role": "tool",
                "tools": [],
                "tags": ["desktop"],
            },
            "iot_sensor": {
                "description": "IoT sensor",
                "role": "tool",
                "tools": [],
                "tags": ["iot"],
            },
        })

        from arqitect.brain.community import seed_nerves
        with patch("arqitect.brain.community._manifest_path", return_value=manifest_path), \
             patch("arqitect.brain.community.sync_nerve_bundle", return_value=None), \
             patch("arqitect.brain.config.mem", mem), \
             patch("arqitect.brain.community.get_config", return_value="iot"):
            count = seed_nerves()

        assert count == 1
        assert os.path.isfile(os.path.join(nerves_dir, "iot_sensor", "nerve.py"))
        assert not os.path.exists(os.path.join(nerves_dir, "screen_capture", "nerve.py"))

    def test_prunes_existing_nerves_that_no_longer_match(self, mem, nerves_dir, tmp_path):
        """Nerves on disk from a previous unfiltered seed get removed if they don't match."""
        make_nerve_file(nerves_dir, "iot_sensor")
        mem.cold.register_nerve("iot_sensor", "IoT sensor control")

        manifest_path = _write_manifest(tmp_path, {
            "iot_sensor": {
                "description": "IoT sensor control",
                "role": "tool",
                "tools": [],
                "tags": ["iot"],
            },
        })

        from arqitect.brain.community import seed_nerves
        with patch("arqitect.brain.community._manifest_path", return_value=manifest_path), \
             patch("arqitect.brain.community.sync_nerve_bundle", return_value=None), \
             patch("arqitect.brain.config.mem", mem), \
             patch("arqitect.brain.community.get_config", return_value="server"):
            count = seed_nerves()

        assert count == 0
        assert not os.path.exists(os.path.join(nerves_dir, "iot_sensor")), \
            "IoT nerve should be pruned from disk in server environment"
        assert "iot_sensor" not in mem.cold.list_nerves(), \
            "IoT nerve should be removed from cold memory"

    def test_universal_nerves_seeded_in_all_environments(self, mem, nerves_dir, tmp_path):
        """Nerves with no env-exclusive tags are seeded in every environment."""
        manifest_path = _write_manifest(tmp_path, {
            "utility_nerve": {
                "description": "General utility",
                "role": "tool",
                "tools": [],
                "tags": ["utility", "general"],
            },
        })

        from arqitect.brain.community import seed_nerves
        for env in ("server", "iot", "desktop"):
            nerve_path = os.path.join(nerves_dir, "utility_nerve")
            if os.path.exists(nerve_path):
                import shutil
                shutil.rmtree(nerve_path)

            with patch("arqitect.brain.community._manifest_path", return_value=manifest_path), \
                 patch("arqitect.brain.community.sync_nerve_bundle", return_value=None), \
                 patch("arqitect.brain.config.mem", mem), \
                 patch("arqitect.brain.community.get_config", return_value=env):
                count = seed_nerves()
            assert count == 1, f"Universal nerve should be seeded in {env} environment"


# ---------------------------------------------------------------------------
# synthesize_nerve — community-first path
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
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
        """When a community bundle exists, its metadata is used directly."""
        bundle = {
            "description": "Community jokes",
            "role": "creative",
            "tools": [],
        }

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
                assert meta["origin"] == "community"
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
