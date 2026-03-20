"""Tests for arqitect.inference.router — per-role inference provider routing.

Contract-based tests verifying resolution order, API key validation,
provider caching, LoRA forwarding, and parameter defaults.
"""

import pytest
from dirty_equals import IsStr, IsInstance
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from unittest.mock import patch, MagicMock

from arqitect.inference.router import (
    _resolve_role_config,
    generate_for_role,
    get_role_provider,
    reset_cache,
    _validate_api_key,
    _config_key_to_kwarg,
)
from arqitect.inference.providers import PROVIDER_META, PROVIDER_REGISTRY


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

KNOWN_PROVIDERS = list(PROVIDER_REGISTRY.keys())
KNOWN_ROLES = ["brain", "nerve", "coder", "creative", "communication"]

cloud_providers = st.sampled_from(
    [name for name, meta in PROVIDER_META.items() if meta.get("auth_type") == "api_key"]
)
local_providers = st.sampled_from(
    [name for name, meta in PROVIDER_META.items() if meta.get("auth_type") != "api_key"]
)
valid_roles = st.sampled_from(KNOWN_ROLES)
valid_providers = st.sampled_from(KNOWN_PROVIDERS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_router_cache():
    """Reset provider cache before each test to avoid cross-contamination."""
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# _config_key_to_kwarg
# ---------------------------------------------------------------------------

class TestConfigKeyToKwarg:
    """Verify config key stripping maps to correct constructor kwargs."""

    def test_strips_openai_prefix(self):
        assert _config_key_to_kwarg("openai_base_url") == "base_url"

    def test_strips_azure_openai_prefix(self):
        assert _config_key_to_kwarg("azure_openai_endpoint") == "endpoint"

    def test_leaves_unknown_prefix_intact(self):
        assert _config_key_to_kwarg("some_other_key") == "some_other_key"

    @given(suffix=st.text(min_size=1, alphabet=st.characters(whitelist_categories=("Ll",))))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.timeout(10)
    def test_openai_prefix_always_stripped(self, suffix):
        """Any key starting with 'openai_' should have that prefix removed."""
        result = _config_key_to_kwarg(f"openai_{suffix}")
        assert result == suffix


# ---------------------------------------------------------------------------
# _resolve_role_config
# ---------------------------------------------------------------------------

class TestResolveRoleConfig:
    """Verify per-role-first, flat-second, raise-on-missing resolution."""

    @pytest.mark.timeout(10)
    def test_per_role_config_takes_priority(self):
        """When inference.roles.brain.provider is set, it wins over flat config."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value="anthropic"), \
             patch("arqitect.inference.router.get_per_role_model", return_value="claude-sonnet-4-20250514"), \
             patch("arqitect.inference.router.get_inference_provider", return_value="gguf"), \
             patch("arqitect.inference.router._get_flat_model", return_value="Qwen.gguf"):
            provider, model = _resolve_role_config("brain")
            assert provider == "anthropic"
            assert model == "claude-sonnet-4-20250514"

    @pytest.mark.timeout(10)
    def test_falls_back_to_flat_config(self):
        """When per-role is not set, uses flat inference.provider / models."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value=None), \
             patch("arqitect.inference.router.get_per_role_model", return_value=None), \
             patch("arqitect.inference.router.get_inference_provider", return_value="gguf"), \
             patch("arqitect.inference.router._get_flat_model", return_value="Qwen.gguf"):
            provider, model = _resolve_role_config("nerve")
            assert provider == "gguf"
            assert model == "Qwen.gguf"

    @pytest.mark.timeout(10)
    def test_raises_on_no_config(self):
        """Raises ValueError when neither per-role nor flat config exists."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value=None), \
             patch("arqitect.inference.router.get_per_role_model", return_value=None), \
             patch("arqitect.inference.router.get_inference_provider", return_value=""), \
             patch("arqitect.inference.router._get_flat_model", return_value=""):
            with pytest.raises(ValueError, match="No inference provider configured"):
                _resolve_role_config("brain")

    @pytest.mark.timeout(10)
    def test_unknown_provider_raises(self):
        """Raises ValueError for an unrecognised provider name."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value="does_not_exist"), \
             patch("arqitect.inference.router.get_per_role_model", return_value="x"):
            with pytest.raises(ValueError, match="Unknown provider"):
                _resolve_role_config("brain")

    @pytest.mark.timeout(10)
    def test_per_role_model_falls_back_to_flat_model(self):
        """Per-role provider set but model missing — falls back to flat model."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value="groq"), \
             patch("arqitect.inference.router.get_per_role_model", return_value=None), \
             patch("arqitect.inference.router._get_flat_model", return_value="llama-3.3-70b"):
            provider, model = _resolve_role_config("coder")
            assert provider == "groq"
            assert model == "llama-3.3-70b"

    @given(role=valid_roles, provider=valid_providers)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.timeout(10)
    def test_per_role_always_returns_known_provider(self, role, provider):
        """Any valid provider/role combination resolves without error."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value=provider), \
             patch("arqitect.inference.router.get_per_role_model", return_value="some-model"), \
             patch("arqitect.inference.router._get_flat_model", return_value="fallback"):
            result_provider, result_model = _resolve_role_config(role)
            assert result_provider == provider
            assert result_model == IsStr()

    @given(
        provider_name=st.text(min_size=1).filter(lambda s: s not in PROVIDER_REGISTRY),
        role=valid_roles,
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.timeout(10)
    def test_unknown_provider_always_raises(self, provider_name, role):
        """Any string not in PROVIDER_REGISTRY raises ValueError."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value=provider_name), \
             patch("arqitect.inference.router.get_per_role_model", return_value="model"):
            with pytest.raises(ValueError, match="Unknown provider"):
                _resolve_role_config(role)

    @given(role=valid_roles)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.timeout(10)
    def test_no_config_always_raises(self, role):
        """All roles raise ValueError when completely unconfigured."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value=None), \
             patch("arqitect.inference.router.get_per_role_model", return_value=None), \
             patch("arqitect.inference.router.get_inference_provider", return_value=""), \
             patch("arqitect.inference.router._get_flat_model", return_value=""):
            with pytest.raises(ValueError, match="No inference provider configured"):
                _resolve_role_config(role)


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------

class TestApiKeyValidation:
    """Cloud providers must have a non-empty API key."""

    @pytest.mark.timeout(10)
    def test_anthropic_requires_key(self):
        with patch("arqitect.inference.router.get_secret", return_value=""):
            with pytest.raises(ValueError, match="anthropic_api_key"):
                _validate_api_key("anthropic")

    @pytest.mark.timeout(10)
    def test_groq_requires_key(self):
        with patch("arqitect.inference.router.get_secret", return_value=""):
            with pytest.raises(ValueError, match="groq_api_key"):
                _validate_api_key("groq")

    @pytest.mark.timeout(10)
    def test_gguf_does_not_require_key(self):
        """Local providers should not raise."""
        _validate_api_key("gguf")

    @pytest.mark.timeout(10)
    def test_valid_key_passes(self):
        with patch("arqitect.inference.router.get_secret", return_value="sk-test-123"):
            _validate_api_key("anthropic")  # should not raise

    @given(provider=cloud_providers)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.timeout(10)
    def test_all_cloud_providers_reject_empty_key(self, provider):
        """Every cloud provider must raise when its API key is empty."""
        expected_secret_key = PROVIDER_META[provider]["secret_key"]
        with patch("arqitect.inference.router.get_secret", return_value=""):
            with pytest.raises(ValueError, match=expected_secret_key):
                _validate_api_key(provider)

    @given(provider=cloud_providers, key=st.text(min_size=1).filter(lambda s: s.strip()))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.timeout(10)
    def test_all_cloud_providers_accept_nonempty_key(self, provider, key):
        """Every cloud provider must pass when its API key is non-empty."""
        with patch("arqitect.inference.router.get_secret", return_value=key):
            _validate_api_key(provider)  # should not raise

    @given(provider=local_providers)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.timeout(10)
    def test_local_providers_never_require_key(self, provider):
        """Local providers never raise regardless of key state."""
        _validate_api_key(provider)  # should not raise


# ---------------------------------------------------------------------------
# generate_for_role
# ---------------------------------------------------------------------------

class TestGenerateForRole:
    """End-to-end generation through the router with mocked providers."""

    def _patch_resolve(self, provider_name: str = "gguf", model: str = "test.gguf"):
        return patch(
            "arqitect.inference.router._resolve_role_config",
            return_value=(provider_name, model),
        )

    @pytest.mark.timeout(10)
    def test_routes_to_correct_provider(self):
        """generate_for_role delegates to the resolved provider's generate()."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Hello from mock"
        mock_provider.supports_lora = False

        with self._patch_resolve("gguf", "test.gguf"), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            result = generate_for_role(
                "brain", "Hi", system="Be helpful",
                max_tokens=2048, temperature=0.7, json_mode=False,
            )
            assert result == "Hello from mock"
            mock_provider.generate.assert_called_once_with(
                model="test.gguf",
                prompt="Hi",
                system="Be helpful",
                max_tokens=2048,
                temperature=0.7,
                json_mode=False,
            )

    @pytest.mark.timeout(10)
    def test_defaults_from_community_adapter(self):
        """When caller omits max_tokens/temperature/json_mode, resolved from adapter."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "adapter defaults"
        mock_provider.supports_lora = False

        with self._patch_resolve("gguf", "test.gguf"), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider), \
             patch("arqitect.brain.adapters.get_max_tokens", return_value=256), \
             patch("arqitect.brain.adapters.get_temperature", return_value=0.3), \
             patch("arqitect.brain.adapters.get_json_mode", return_value=True):
            generate_for_role("nerve", "classify this")

            call_kwargs = mock_provider.generate.call_args[1]
            assert call_kwargs["max_tokens"] == 256
            assert call_kwargs["temperature"] == 0.3
            assert call_kwargs["json_mode"] is True

    @pytest.mark.timeout(10)
    def test_explicit_values_override_adapter(self):
        """Caller-supplied values take precedence over community adapter config."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "overridden"
        mock_provider.supports_lora = False

        with self._patch_resolve("gguf", "test.gguf"), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            generate_for_role(
                "nerve", "x",
                max_tokens=100, temperature=0.0, json_mode=False,
            )
            call_kwargs = mock_provider.generate.call_args[1]
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["temperature"] == 0.0
            assert call_kwargs["json_mode"] is False

    @pytest.mark.timeout(10)
    def test_lora_passed_to_supporting_provider(self):
        """LoRA path is forwarded when the provider supports it."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "lora result"
        mock_provider.supports_lora = True

        with self._patch_resolve("gguf", "test.gguf"), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            result = generate_for_role(
                "nerve", "x", lora_path="/path/to/adapter.gguf",
                max_tokens=512, temperature=0.5, json_mode=False,
            )
            assert result == "lora result"
            call_kwargs = mock_provider.generate.call_args[1]
            assert call_kwargs["lora_path"] == "/path/to/adapter.gguf"

    @pytest.mark.timeout(10)
    def test_lora_not_passed_to_cloud_provider(self):
        """LoRA path is silently ignored when provider doesn't support it."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "cloud result"
        mock_provider.supports_lora = False

        with self._patch_resolve("anthropic", "claude-sonnet-4-20250514"), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            result = generate_for_role(
                "nerve", "x", lora_path="/path/to/adapter.gguf",
                max_tokens=512, temperature=0.5, json_mode=False,
            )
            assert result == "cloud result"
            call_kwargs = mock_provider.generate.call_args[1]
            assert "lora_path" not in call_kwargs

    @pytest.mark.timeout(10)
    def test_json_mode_forwarded(self):
        mock_provider = MagicMock()
        mock_provider.generate.return_value = '{"key": "val"}'
        mock_provider.supports_lora = False

        with self._patch_resolve(), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            generate_for_role("nerve", "classify", json_mode=True,
                              max_tokens=256, temperature=0.3)
            call_kwargs = mock_provider.generate.call_args[1]
            assert call_kwargs["json_mode"] is True

    @given(role=valid_roles)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.timeout(10)
    def test_generate_always_returns_string(self, role):
        """generate_for_role always returns a string regardless of role."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "output"
        mock_provider.supports_lora = False

        with patch("arqitect.inference.router._resolve_role_config", return_value=("gguf", "m.gguf")), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            result = generate_for_role(
                role, "prompt", max_tokens=100, temperature=0.5, json_mode=False,
            )
            assert result == IsStr()

    @pytest.mark.timeout(10)
    def test_generate_passes_system_prompt(self):
        """System prompt is always forwarded to the provider."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "ok"
        mock_provider.supports_lora = False

        with self._patch_resolve(), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            generate_for_role(
                "brain", "hello", system="You are a robot.",
                max_tokens=100, temperature=0.5, json_mode=False,
            )
            call_kwargs = mock_provider.generate.call_args[1]
            assert call_kwargs["system"] == "You are a robot."

    @pytest.mark.timeout(10)
    def test_generate_empty_system_by_default(self):
        """When no system prompt is given, an empty string is passed."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "ok"
        mock_provider.supports_lora = False

        with self._patch_resolve(), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            generate_for_role(
                "brain", "hello",
                max_tokens=100, temperature=0.5, json_mode=False,
            )
            call_kwargs = mock_provider.generate.call_args[1]
            assert call_kwargs["system"] == ""


# ---------------------------------------------------------------------------
# Provider caching
# ---------------------------------------------------------------------------

class TestProviderCaching:
    """Providers should be instantiated once and shared across roles."""

    @pytest.mark.timeout(10)
    def test_same_provider_shared_across_roles(self):
        """Two roles using 'gguf' should get the same provider instance."""
        mock_provider = MagicMock()

        with patch("arqitect.inference.router._resolve_role_config") as mock_resolve, \
             patch("arqitect.inference.router._build_provider_kwargs", return_value={}), \
             patch("arqitect.inference.router.get_provider", return_value=mock_provider) as mock_get:
            mock_resolve.return_value = ("gguf", "model.gguf")

            p1 = get_role_provider("brain")
            p2 = get_role_provider("nerve")

            # get_provider should only be called once (cached after first)
            assert mock_get.call_count == 1
            assert p1 is p2

    @pytest.mark.timeout(10)
    def test_different_providers_cached_separately(self):
        """Two roles using different providers get separate instances."""
        mock_gguf = MagicMock(name="gguf_provider")
        mock_anthropic = MagicMock(name="anthropic_provider")

        call_count = 0

        def resolve_side_effect(role):
            if role == "brain":
                return ("gguf", "model.gguf")
            return ("anthropic", "claude-sonnet-4-20250514")

        def get_provider_side_effect(name, **kwargs):
            if name == "gguf":
                return mock_gguf
            return mock_anthropic

        with patch("arqitect.inference.router._resolve_role_config", side_effect=resolve_side_effect), \
             patch("arqitect.inference.router._build_provider_kwargs", return_value={}), \
             patch("arqitect.inference.router.get_provider", side_effect=get_provider_side_effect) as mock_get:

            p1 = get_role_provider("brain")
            p2 = get_role_provider("nerve")

            assert mock_get.call_count == 2
            assert p1 is not p2
            assert p1 is mock_gguf
            assert p2 is mock_anthropic

    @pytest.mark.timeout(10)
    def test_reset_cache_clears_providers(self):
        """After reset_cache(), providers are re-created on next access."""
        mock_provider = MagicMock()

        with patch("arqitect.inference.router._resolve_role_config", return_value=("gguf", "m.gguf")), \
             patch("arqitect.inference.router._build_provider_kwargs", return_value={}), \
             patch("arqitect.inference.router.get_provider", return_value=mock_provider) as mock_get:

            get_role_provider("brain")
            assert mock_get.call_count == 1

            reset_cache()

            get_role_provider("brain")
            assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Flat config resolution
# ---------------------------------------------------------------------------

class TestFlatConfig:
    """Flat config (inference.provider + inference.models.<role>) resolves correctly."""

    @pytest.mark.timeout(10)
    def test_flat_config_resolves(self):
        with patch("arqitect.inference.router.get_per_role_provider", return_value=None), \
             patch("arqitect.inference.router.get_per_role_model", return_value=None), \
             patch("arqitect.inference.router.get_inference_provider", return_value="gguf"), \
             patch("arqitect.inference.router._get_flat_model", return_value="Qwen2.5-Coder-7B.gguf"):
            provider, model = _resolve_role_config("brain")
            assert provider == "gguf"
            assert model == "Qwen2.5-Coder-7B.gguf"

    @given(role=valid_roles, model=st.text(min_size=1))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @pytest.mark.timeout(10)
    def test_flat_config_always_returns_tuple(self, role, model):
        """Flat config resolution always yields a (str, str) tuple."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value=None), \
             patch("arqitect.inference.router.get_per_role_model", return_value=None), \
             patch("arqitect.inference.router.get_inference_provider", return_value="gguf"), \
             patch("arqitect.inference.router._get_flat_model", return_value=model):
            provider, resolved_model = _resolve_role_config(role)
            assert provider == IsStr()
            assert resolved_model == IsStr()


# ---------------------------------------------------------------------------
# PROVIDER_META / PROVIDER_REGISTRY consistency
# ---------------------------------------------------------------------------

class TestProviderMetaConsistency:
    """Verify structural invariants of the provider metadata tables."""

    @pytest.mark.timeout(10)
    def test_all_registry_entries_have_meta(self):
        """Every provider in PROVIDER_REGISTRY should have a PROVIDER_META entry
        (except aliases like openai_compat)."""
        aliases = {"openai_compat"}
        for name in PROVIDER_REGISTRY:
            if name in aliases:
                continue
            assert name in PROVIDER_META, (
                f"Provider '{name}' is in PROVIDER_REGISTRY but missing from PROVIDER_META"
            )

    @pytest.mark.timeout(10)
    def test_cloud_providers_have_secret_keys(self):
        """Every cloud provider must declare a secret_key for API key validation."""
        for name, meta in PROVIDER_META.items():
            if meta.get("auth_type") == "api_key":
                assert meta.get("secret_key"), (
                    f"Cloud provider '{name}' has auth_type=api_key but no secret_key"
                )

    @pytest.mark.timeout(10)
    def test_local_providers_have_no_secret_key_requirement(self):
        """Local providers must not have auth_type=api_key."""
        for name, meta in PROVIDER_META.items():
            if meta.get("category") == "local":
                assert meta.get("auth_type") != "api_key", (
                    f"Local provider '{name}' should not require an API key"
                )
