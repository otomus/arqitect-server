"""Tests for arqitect/config/loader.py — project root, YAML loading, and config accessors."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from arqitect.config.loader import (
    _deep_merge,
    find_project_root,
    get_config,
    get_model_for_role,
    get_per_role_model,
    get_per_role_provider,
    get_project_root,
    get_redis_host_port,
    get_redis_url,
    get_secret,
    load_config,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear LRU caches before every test so results are independent."""
    load_config.cache_clear()
    get_project_root.cache_clear()
    yield
    load_config.cache_clear()
    get_project_root.cache_clear()


# ── _deep_merge ──────────────────────────────────────────────────────────


class TestDeepMerge:
    """Tests for recursive dict merging."""

    def test_nested_dicts_are_merged(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_override_replaces_leaf_value(self):
        base = {"a": 1}
        override = {"a": 99}
        assert _deep_merge(base, override) == {"a": 99}

    def test_new_keys_are_added(self):
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_non_dict_override_replaces_dict(self):
        """When override value is not a dict, it replaces entirely."""
        base = {"a": {"nested": True}}
        override = {"a": "flat"}
        assert _deep_merge(base, override) == {"a": "flat"}

    def test_dict_override_replaces_non_dict(self):
        """When base value is not a dict but override is, override wins."""
        base = {"a": "flat"}
        override = {"a": {"nested": True}}
        assert _deep_merge(base, override) == {"a": {"nested": True}}

    def test_original_dicts_are_not_mutated(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}


# ── find_project_root ────────────────────────────────────────────────────


class TestFindProjectRoot:
    """Tests for project root detection."""

    def test_env_var_overrides_everything(self, tmp_path):
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert find_project_root() == tmp_path

    def test_env_var_ignored_if_not_a_directory(self, tmp_path):
        fake = str(tmp_path / "does_not_exist")
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": fake}):
            # Should NOT return the fake path; falls through to other logic
            root = find_project_root()
            assert root != Path(fake)

    def test_arqitect_yaml_found_in_parent(self, tmp_path):
        (tmp_path / "arqitect.yaml").write_text("name: test\n")
        child = tmp_path / "sub" / "dir"
        child.mkdir(parents=True)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARQITECT_PROJECT_ROOT", None)
            with patch.object(Path, "cwd", return_value=child):
                assert find_project_root() == tmp_path

    def test_falls_back_to_cwd_when_nothing_found(self, tmp_path):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARQITECT_PROJECT_ROOT", None)
            with patch.object(Path, "cwd", return_value=tmp_path):
                assert find_project_root() == tmp_path


# ── load_config ──────────────────────────────────────────────────────────


class TestLoadConfig:
    """Tests for YAML loading and merging with defaults."""

    def test_loads_yaml_and_merges_with_defaults(self, tmp_path):
        yaml_content = "name: MyBot\ninference:\n  provider: openai\n"
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            config = load_config()
        assert config["name"] == "MyBot"
        assert config["inference"]["provider"] == "openai"
        # Defaults still present for keys not overridden
        assert "storage" in config

    def test_returns_defaults_when_no_yaml(self, tmp_path):
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            config = load_config()
        assert config["name"] == "Arqitect"
        assert config["inference"]["provider"] == "gguf"


# ── get_config ───────────────────────────────────────────────────────────


class TestGetConfig:
    """Tests for dot-path config access."""

    def test_nested_path(self, tmp_path):
        yaml_content = "a:\n  b:\n    c: 42\n"
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_config("a.b.c") == 42

    def test_missing_path_returns_default(self, tmp_path):
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_config("no.such.path", "fallback") == "fallback"

    def test_top_level_key(self, tmp_path):
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_config("name") == "Arqitect"


# ── get_secret ───────────────────────────────────────────────────────────


class TestGetSecret:
    """Tests for secrets accessor."""

    def test_secret_exists(self, tmp_path):
        yaml_content = "secrets:\n  jwt_secret: s3cret\n"
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_secret("jwt_secret") == "s3cret"

    def test_missing_secret_returns_default(self, tmp_path):
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_secret("nonexistent", "default_val") == "default_val"

    def test_missing_secret_returns_empty_string_by_default(self, tmp_path):
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_secret("nonexistent") == ""


# ── get_redis_host_port ──────────────────────────────────────────────────


class TestGetRedisHostPort:
    """Tests for Redis URL parsing."""

    def test_standard_url(self, tmp_path):
        yaml_content = "storage:\n  hot:\n    url: redis://myhost:6380\n"
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            host, port = get_redis_host_port()
        assert host == "myhost"
        assert port == 6380

    def test_url_without_port_defaults_to_6379(self, tmp_path):
        yaml_content = "storage:\n  hot:\n    url: redis://myhost\n"
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            host, port = get_redis_host_port()
        assert host == "myhost"
        assert port == 6379

    def test_url_with_trailing_slash(self, tmp_path):
        yaml_content = "storage:\n  hot:\n    url: redis://myhost:6380/\n"
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            host, port = get_redis_host_port()
        assert host == "myhost"
        assert port == 6380

    def test_url_with_db_number(self, tmp_path):
        yaml_content = "storage:\n  hot:\n    url: redis://myhost:6380/0\n"
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            host, port = get_redis_host_port()
        assert host == "myhost"
        assert port == 6380


# ── get_model_for_role ───────────────────────────────────────────────────


class TestGetModelForRole:
    """Tests for model filename resolution by role."""

    def test_dict_value_returns_file_key(self, tmp_path):
        yaml_content = (
            "inference:\n"
            "  models:\n"
            "    brain:\n"
            "      file: brain-model.gguf\n"
        )
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_model_for_role("brain") == "brain-model.gguf"

    def test_string_value_returned_directly(self, tmp_path):
        yaml_content = (
            "inference:\n"
            "  models:\n"
            "    brain: my-model.gguf\n"
        )
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_model_for_role("brain") == "my-model.gguf"

    def test_missing_role_falls_back_to_brain(self, tmp_path):
        yaml_content = (
            "inference:\n"
            "  models:\n"
            "    brain: fallback.gguf\n"
        )
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_model_for_role("nonexistent") == "fallback.gguf"


# ── get_per_role_provider / get_per_role_model ───────────────────────────


class TestPerRoleOverrides:
    """Tests for per-role provider and model overrides."""

    def test_per_role_provider_returns_override(self, tmp_path):
        yaml_content = (
            "inference:\n"
            "  roles:\n"
            "    creative:\n"
            "      provider: anthropic\n"
        )
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_per_role_provider("creative") == "anthropic"

    def test_per_role_provider_returns_none_when_unset(self, tmp_path):
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_per_role_provider("brain") is None

    def test_per_role_model_returns_override(self, tmp_path):
        yaml_content = (
            "inference:\n"
            "  roles:\n"
            "    coder:\n"
            "      model: gpt-4\n"
        )
        (tmp_path / "arqitect.yaml").write_text(yaml_content)
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_per_role_model("coder") == "gpt-4"

    def test_per_role_model_returns_none_when_unset(self, tmp_path):
        with patch.dict(os.environ, {"ARQITECT_PROJECT_ROOT": str(tmp_path)}):
            assert get_per_role_model("brain") is None
