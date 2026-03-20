"""Tests for arqitect.inference.model_registry — registry building, proxy, resolution."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
from dirty_equals import IsInstance, IsNonNegative
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from arqitect.inference.model_registry import (
    _REGISTRY_ROLES,
    _RegistryProxy,
    _build_registry,
    resolve_registry_key,
    find_registry_entry_by_file,
    resolve_model_path,
    ROLE_TO_REGISTRY_KEY,
    CHAT_HANDLER_MOONDREAM,
    BACKEND_STABLE_DIFFUSION,
)
import arqitect.inference.model_registry as registry_mod


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestConstants:
    """Verify named constants are set correctly."""

    def test_chat_handler_moondream(self):
        assert CHAT_HANDLER_MOONDREAM == "moondream"

    def test_backend_stable_diffusion(self):
        assert BACKEND_STABLE_DIFFUSION == "stable_diffusion"

    def test_registry_roles_contains_core_roles(self):
        for role in ("brain", "nerve", "vision", "embedding"):
            assert role in _REGISTRY_ROLES

    def test_registry_roles_is_tuple_of_strings(self):
        assert _REGISTRY_ROLES == IsInstance(tuple)
        for role in _REGISTRY_ROLES:
            assert role == IsInstance(str)
            assert len(role) > 0

    def test_role_to_registry_key_values_are_valid_roles(self):
        """Every alias target must be a known registry role."""
        for target in ROLE_TO_REGISTRY_KEY.values():
            assert target in _REGISTRY_ROLES, (
                f"Alias target '{target}' is not a known registry role"
            )


# ---------------------------------------------------------------------------
# _build_registry
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestBuildRegistry:
    """Tests for building the registry from yaml config."""

    @patch("arqitect.config.loader.get_model_config")
    def test_includes_roles_with_file(self, mock_gmc):
        mock_gmc.side_effect = lambda role: (
            {"file": "brain.gguf", "source": "hf/brain"} if role == "brain"
            else None
        )
        reg = _build_registry()
        assert "brain" in reg
        assert reg["brain"]["file"] == "brain.gguf"

    @patch("arqitect.config.loader.get_model_config")
    def test_excludes_roles_without_file(self, mock_gmc):
        mock_gmc.return_value = {"source": "hf/x"}  # no "file" key
        reg = _build_registry()
        assert len(reg) == 0

    @patch("arqitect.config.loader.get_model_config")
    def test_excludes_none_configs(self, mock_gmc):
        mock_gmc.return_value = None
        reg = _build_registry()
        assert len(reg) == 0

    @patch("arqitect.config.loader.get_model_config")
    def test_all_roles_queried(self, mock_gmc):
        mock_gmc.return_value = None
        _build_registry()
        called_roles = {c.args[0] for c in mock_gmc.call_args_list}
        assert called_roles == set(_REGISTRY_ROLES)

    @patch("arqitect.config.loader.get_model_config")
    def test_registry_keys_are_subset_of_roles(self, mock_gmc):
        """Every key in the built registry must be a known role."""
        mock_gmc.side_effect = lambda role: {"file": f"{role}.gguf"}
        reg = _build_registry()
        assert set(reg.keys()) <= set(_REGISTRY_ROLES)

    @patch("arqitect.config.loader.get_model_config")
    def test_config_with_empty_file_string_excluded(self, mock_gmc):
        """A config with file='' (falsy) should be excluded."""
        mock_gmc.return_value = {"file": "", "source": "hf/x"}
        reg = _build_registry()
        assert len(reg) == 0


# ---------------------------------------------------------------------------
# _RegistryProxy
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestRegistryProxy:
    """Tests for the lazy dict proxy."""

    def _make_proxy(self, data: dict) -> _RegistryProxy:
        """Create a proxy pre-loaded with data (bypass lazy build)."""
        proxy = _RegistryProxy()
        dict.update(proxy, data)
        return proxy

    def test_getitem(self):
        proxy = self._make_proxy({"a": 1})
        assert proxy["a"] == 1

    def test_contains(self):
        proxy = self._make_proxy({"a": 1})
        assert "a" in proxy
        assert "b" not in proxy

    def test_get_with_default(self):
        proxy = self._make_proxy({"a": 1})
        assert proxy.get("a") == 1
        assert proxy.get("missing", 42) == 42

    def test_keys_values_items(self):
        proxy = self._make_proxy({"x": 10, "y": 20})
        assert set(proxy.keys()) == {"x", "y"}
        assert set(proxy.values()) == {10, 20}
        assert set(proxy.items()) == {("x", 10), ("y", 20)}

    def test_len(self):
        proxy = self._make_proxy({"a": 1, "b": 2})
        assert len(proxy) == 2

    def test_iter(self):
        proxy = self._make_proxy({"a": 1, "b": 2})
        assert set(proxy) == {"a", "b"}

    def test_repr(self):
        proxy = self._make_proxy({"k": "v"})
        assert "k" in repr(proxy)

    @patch.object(registry_mod, "_get_registry", return_value={"brain": {"file": "b.gguf"}})
    def test_lazy_build_on_first_access(self, mock_get):
        proxy = _RegistryProxy()
        _ = proxy["brain"]
        mock_get.assert_called_once()

    def test_getitem_missing_raises_keyerror(self):
        proxy = self._make_proxy({"a": 1})
        with pytest.raises(KeyError):
            _ = proxy["nonexistent"]

    @given(data=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.integers(),
        min_size=1,
        max_size=10,
    ))
    @settings(max_examples=30)
    def test_proxy_behaves_like_dict(self, data: dict):
        """A pre-loaded proxy contains all inserted keys with correct values."""
        proxy = self._make_proxy(data)
        for key in data:
            assert proxy[key] == data[key]
            assert key in proxy


# ---------------------------------------------------------------------------
# resolve_registry_key
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestResolveRegistryKey:
    """Tests for role alias resolution."""

    def test_known_alias_tool(self):
        assert resolve_registry_key("tool") == "nerve"

    def test_known_alias_code(self):
        assert resolve_registry_key("code") == "coder"

    def test_known_alias_awareness(self):
        assert resolve_registry_key("awareness") == "brain"

    def test_known_alias_scheduler(self):
        assert resolve_registry_key("scheduler") == "nerve"

    def test_known_alias_generative(self):
        assert resolve_registry_key("generative") == "creative"

    def test_unknown_role_returns_unchanged(self):
        assert resolve_registry_key("embedding") == "embedding"

    def test_all_aliases_covered(self):
        for alias, target in ROLE_TO_REGISTRY_KEY.items():
            assert resolve_registry_key(alias) == target

    @given(role=st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_resolve_always_returns_string(self, role: str):
        """resolve_registry_key must always return a non-empty string for non-empty input."""
        result = resolve_registry_key(role)
        assert result == IsInstance(str)
        assert len(result) > 0

    @given(role=st.text(min_size=1, max_size=50).filter(lambda r: r not in ROLE_TO_REGISTRY_KEY))
    @settings(max_examples=30)
    def test_non_alias_is_identity(self, role: str):
        """Any role not in the alias map must be returned unchanged."""
        assert resolve_registry_key(role) == role

    def test_resolve_is_idempotent_for_targets(self):
        """Resolving an already-resolved target must not change it further."""
        for target in ROLE_TO_REGISTRY_KEY.values():
            assert resolve_registry_key(target) == target


# ---------------------------------------------------------------------------
# find_registry_entry_by_file
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestFindRegistryEntryByFile:
    """Tests for filename-based registry lookup."""

    @patch.object(registry_mod, "MODEL_REGISTRY", {"brain": {"file": "brain.gguf"}, "nerve": {"file": "nerve.gguf"}})
    def test_finds_existing_file(self):
        entry = find_registry_entry_by_file("brain.gguf")
        assert entry is not None
        assert entry["file"] == "brain.gguf"

    @patch.object(registry_mod, "MODEL_REGISTRY", {"brain": {"file": "brain.gguf"}})
    def test_returns_none_for_missing_file(self):
        assert find_registry_entry_by_file("unknown.gguf") is None

    @patch.object(registry_mod, "MODEL_REGISTRY", {})
    def test_empty_registry(self):
        assert find_registry_entry_by_file("any.gguf") is None

    @given(filename=st.text(min_size=1, max_size=50))
    @settings(max_examples=20)
    @patch.object(registry_mod, "MODEL_REGISTRY", {})
    def test_empty_registry_always_returns_none(self, filename: str):
        """An empty registry must return None for any filename."""
        assert find_registry_entry_by_file(filename) is None

    @patch.object(
        registry_mod, "MODEL_REGISTRY",
        {"a": {"file": "x.gguf"}, "b": {"file": "y.gguf"}},
    )
    def test_returns_dict_with_file_field(self):
        """The returned entry must contain the 'file' key matching the query."""
        entry = find_registry_entry_by_file("y.gguf")
        assert entry is not None
        assert entry == IsInstance(dict)
        assert entry["file"] == "y.gguf"


# ---------------------------------------------------------------------------
# resolve_model_path
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestResolveModelPath:
    """Tests for model path resolution."""

    def test_absolute_path_exists(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        result = resolve_model_path(str(model_file), "/unused")
        assert result == str(model_file)

    def test_absolute_path_missing(self):
        result = resolve_model_path("/nonexistent/model.gguf", "/unused")
        assert result is None

    def test_relative_name_found_in_models_dir(self, tmp_path):
        model_file = tmp_path / "nerve.gguf"
        model_file.touch()
        result = resolve_model_path("nerve.gguf", str(tmp_path))
        assert result == str(model_file)

    def test_relative_name_not_found(self, tmp_path):
        result = resolve_model_path("missing.gguf", str(tmp_path))
        assert result is None

    @given(filename=st.from_regex(r"[a-z][a-z0-9_]{0,19}\.gguf", fullmatch=True))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_relative_name_missing_in_empty_dir(self, tmp_path, filename: str):
        """A relative name never resolves in an empty directory."""
        result = resolve_model_path(filename, str(tmp_path))
        assert result is None

    def test_resolve_prefers_absolute_over_relative(self, tmp_path):
        """When name is an absolute path that exists, models_dir is irrelevant."""
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        result = resolve_model_path(str(model_file), "/completely/wrong")
        assert result == str(model_file)
