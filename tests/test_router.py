"""Tests for arqitect.inference.router — per-role inference provider routing."""

import pytest
from unittest.mock import patch, MagicMock

from arqitect.inference.router import (
    _resolve_role_config,
    generate_for_role,
    get_role_provider,
    reset_cache,
    _validate_api_key,
)


@pytest.fixture(autouse=True)
def clean_router_cache():
    """Reset provider cache before each test to avoid cross-contamination."""
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# _resolve_role_config
# ---------------------------------------------------------------------------

class TestResolveRoleConfig:
    """Verify per-role-first, flat-second, raise-on-missing resolution."""

    def test_per_role_config_takes_priority(self):
        """When inference.roles.brain.provider is set, it wins over flat config."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value="anthropic"), \
             patch("arqitect.inference.router.get_per_role_model", return_value="claude-sonnet-4-20250514"), \
             patch("arqitect.inference.router.get_inference_provider", return_value="gguf"), \
             patch("arqitect.inference.router._get_flat_model", return_value="Qwen.gguf"):
            provider, model = _resolve_role_config("brain")
            assert provider == "anthropic"
            assert model == "claude-sonnet-4-20250514"

    def test_falls_back_to_flat_config(self):
        """When per-role is not set, uses flat inference.provider / models."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value=None), \
             patch("arqitect.inference.router.get_per_role_model", return_value=None), \
             patch("arqitect.inference.router.get_inference_provider", return_value="gguf"), \
             patch("arqitect.inference.router._get_flat_model", return_value="Qwen.gguf"):
            provider, model = _resolve_role_config("nerve")
            assert provider == "gguf"
            assert model == "Qwen.gguf"

    def test_raises_on_no_config(self):
        """Raises ValueError when neither per-role nor flat config exists."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value=None), \
             patch("arqitect.inference.router.get_per_role_model", return_value=None), \
             patch("arqitect.inference.router.get_inference_provider", return_value=""), \
             patch("arqitect.inference.router._get_flat_model", return_value=""):
            with pytest.raises(ValueError, match="No inference provider configured"):
                _resolve_role_config("brain")

    def test_unknown_provider_raises(self):
        """Raises ValueError for an unrecognised provider name."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value="does_not_exist"), \
             patch("arqitect.inference.router.get_per_role_model", return_value="x"):
            with pytest.raises(ValueError, match="Unknown provider"):
                _resolve_role_config("brain")

    def test_per_role_model_falls_back_to_flat_model(self):
        """Per-role provider set but model missing — falls back to flat model."""
        with patch("arqitect.inference.router.get_per_role_provider", return_value="groq"), \
             patch("arqitect.inference.router.get_per_role_model", return_value=None), \
             patch("arqitect.inference.router._get_flat_model", return_value="llama-3.3-70b"):
            provider, model = _resolve_role_config("coder")
            assert provider == "groq"
            assert model == "llama-3.3-70b"


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------

class TestApiKeyValidation:
    """Cloud providers must have a non-empty API key."""

    def test_anthropic_requires_key(self):
        with patch("arqitect.inference.router.get_secret", return_value=""):
            with pytest.raises(ValueError, match="anthropic_api_key"):
                _validate_api_key("anthropic")

    def test_groq_requires_key(self):
        with patch("arqitect.inference.router.get_secret", return_value=""):
            with pytest.raises(ValueError, match="groq_api_key"):
                _validate_api_key("groq")

    def test_gguf_does_not_require_key(self):
        """Local providers should not raise."""
        _validate_api_key("gguf")
        _validate_api_key("ollama")

    def test_valid_key_passes(self):
        with patch("arqitect.inference.router.get_secret", return_value="sk-test-123"):
            _validate_api_key("anthropic")  # should not raise


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

    def test_routes_to_correct_provider(self):
        """generate_for_role delegates to the resolved provider's generate()."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Hello from mock"
        mock_provider.supports_lora = False

        with self._patch_resolve("gguf", "test.gguf"), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            result = generate_for_role("brain", "Hi", system="Be helpful")
            assert result == "Hello from mock"
            mock_provider.generate.assert_called_once_with(
                model="test.gguf",
                prompt="Hi",
                system="Be helpful",
                max_tokens=2048,
                temperature=0.7,
                json_mode=False,
            )

    def test_lora_passed_to_supporting_provider(self):
        """LoRA path is forwarded when the provider supports it."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "lora result"
        mock_provider.supports_lora = True

        with self._patch_resolve("gguf", "test.gguf"), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            result = generate_for_role("nerve", "x", lora_path="/path/to/adapter.gguf")
            assert result == "lora result"
            call_kwargs = mock_provider.generate.call_args[1]
            assert call_kwargs["lora_path"] == "/path/to/adapter.gguf"

    def test_lora_not_passed_to_cloud_provider(self):
        """LoRA path is silently ignored when provider doesn't support it."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "cloud result"
        mock_provider.supports_lora = False

        with self._patch_resolve("anthropic", "claude-sonnet-4-20250514"), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            result = generate_for_role("nerve", "x", lora_path="/path/to/adapter.gguf")
            assert result == "cloud result"
            call_kwargs = mock_provider.generate.call_args[1]
            assert "lora_path" not in call_kwargs

    def test_json_mode_forwarded(self):
        mock_provider = MagicMock()
        mock_provider.generate.return_value = '{"key": "val"}'
        mock_provider.supports_lora = False

        with self._patch_resolve(), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):
            generate_for_role("nerve", "classify", json_mode=True)
            call_kwargs = mock_provider.generate.call_args[1]
            assert call_kwargs["json_mode"] is True


# ---------------------------------------------------------------------------
# Provider caching
# ---------------------------------------------------------------------------

class TestProviderCaching:
    """Providers should be instantiated once and shared across roles."""

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


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Old flat config (inference.provider: gguf, inference.models.brain: X) still works."""

    def test_flat_config_resolves(self):
        with patch("arqitect.inference.router.get_per_role_provider", return_value=None), \
             patch("arqitect.inference.router.get_per_role_model", return_value=None), \
             patch("arqitect.inference.router.get_inference_provider", return_value="gguf"), \
             patch("arqitect.inference.router._get_flat_model", return_value="Qwen2.5-Coder-7B.gguf"):
            provider, model = _resolve_role_config("brain")
            assert provider == "gguf"
            assert model == "Qwen2.5-Coder-7B.gguf"
