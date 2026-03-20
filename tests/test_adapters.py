"""Tests for arqitect.brain.adapters — model sizing, prompt resolution, and tuning config."""

import json
import os
from unittest.mock import patch

import pytest

from arqitect.brain.adapters import (
    _extract_param_billions,
    _model_slug,
    _params_to_size_class,
    get_conversation_window,
    get_json_mode,
    get_max_tokens,
    get_message_truncation,
    get_model_size_class,
    get_temperature,
    get_tuning_config,
    resolve_meta,
    resolve_prompt,
)


# ── _extract_param_billions ───────────────────────────────────────────────────


class TestExtractParamBillions:
    """Tests for regex-based parameter count extraction from model names."""

    def test_integer_params(self):
        assert _extract_param_billions("llama-7b.gguf") == 7.0

    def test_decimal_params(self):
        assert _extract_param_billions("phi-3.8b-mini.gguf") == 3.8

    def test_large_params(self):
        assert _extract_param_billions("llama-70b-chat.gguf") == 70.0

    def test_sub_one_params(self):
        assert _extract_param_billions("smollm-0.5b.gguf") == 0.5

    def test_no_params_returns_none(self):
        assert _extract_param_billions("model.gguf") is None

    def test_no_b_suffix_returns_none(self):
        assert _extract_param_billions("model-v2.gguf") is None

    def test_case_insensitive(self):
        assert _extract_param_billions("Llama-7B.gguf") == 7.0

    def test_plain_name_no_extension(self):
        assert _extract_param_billions("llama-13b") == 13.0


# ── _params_to_size_class ────────────────────────────────────────────────────


class TestParamsToSizeClass:
    """Tests for parameter count to size class boundary mapping."""

    def test_below_tinylm_boundary(self):
        assert _params_to_size_class(2.9) == "tinylm"

    def test_at_tinylm_boundary(self):
        assert _params_to_size_class(3.0) == "small"

    def test_below_small_boundary(self):
        assert _params_to_size_class(5.9) == "small"

    def test_at_small_boundary(self):
        assert _params_to_size_class(6.0) == "medium"

    def test_below_medium_boundary(self):
        assert _params_to_size_class(31.9) == "medium"

    def test_at_medium_boundary(self):
        assert _params_to_size_class(32.0) == "large"

    def test_well_above_large(self):
        assert _params_to_size_class(405.0) == "large"

    def test_tiny_model(self):
        assert _params_to_size_class(0.5) == "tinylm"


# ── _model_slug ───────────────────────────────────────────────────────────────


class TestModelSlug:
    """Tests for model filename to directory-safe slug conversion."""

    def test_gguf_extension_stripped(self):
        assert _model_slug("My-Model-7B.gguf") == "my-model-7b"

    def test_bin_extension_stripped(self):
        assert _model_slug("Model.bin") == "model"

    def test_safetensors_extension_stripped(self):
        assert _model_slug("weights.safetensors") == "weights"

    def test_lowercased(self):
        assert _model_slug("LLaMA-Chat") == "llama-chat"

    def test_spaces_replaced_with_hyphens(self):
        assert _model_slug("My Model 7B.gguf") == "my-model-7b"

    def test_no_extension(self):
        assert _model_slug("llama-7b") == "llama-7b"

    def test_preserves_numbers(self):
        assert _model_slug("vision-model-f16.gguf") == "vision-model-f16"


# ── get_model_size_class ──────────────────────────────────────────────────────


class TestGetModelSizeClass:
    """Tests for size class resolution from model file or cloud provider."""

    @patch("arqitect.brain.adapters._is_cloud_provider", return_value=False)
    @patch("arqitect.brain.adapters._get_model_file_for_role", return_value="llama-7b.gguf")
    def test_gguf_model_extracts_size(self, _mock_file, _mock_cloud):
        assert get_model_size_class("nerve") == "medium"

    @patch("arqitect.brain.adapters._is_cloud_provider", return_value=True)
    @patch("arqitect.brain.adapters._get_model_file_for_role", return_value=None)
    def test_cloud_provider_returns_large(self, _mock_file, _mock_cloud):
        assert get_model_size_class("brain") == "large"

    @patch("arqitect.brain.adapters._is_cloud_provider", return_value=False)
    @patch("arqitect.brain.adapters._get_model_file_for_role", return_value=None)
    def test_unknown_returns_none(self, _mock_file, _mock_cloud):
        assert get_model_size_class("brain") is None

    @patch("arqitect.brain.adapters._is_cloud_provider", return_value=False)
    @patch("arqitect.brain.adapters._get_model_file_for_role", return_value="model-no-params.gguf")
    def test_no_params_not_cloud_returns_none(self, _mock_file, _mock_cloud):
        assert get_model_size_class("nerve") is None


# ── resolve_prompt ────────────────────────────────────────────────────────────


class TestResolvePrompt:
    """Tests for prompt resolution with model-specific and size-class fallback."""

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value="llama-7b")
    @patch("arqitect.brain.adapters.get_model_size_class", return_value="medium")
    @patch("arqitect.brain.adapters._load_context")
    def test_model_specific_found(self, mock_load, _mock_size, _mock_name):
        model_ctx = {"temperature": 0.3, "system_prompt": "be precise"}
        mock_load.side_effect = lambda role, *parts: model_ctx if len(parts) == 2 else None

        result = resolve_prompt("nerve")
        assert result == model_ctx

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value="llama-7b")
    @patch("arqitect.brain.adapters.get_model_size_class", return_value="medium")
    @patch("arqitect.brain.adapters._load_context")
    def test_falls_back_to_size_class(self, mock_load, _mock_size, _mock_name):
        size_ctx = {"temperature": 0.5}
        # Model-specific returns None, size-class returns context
        mock_load.side_effect = lambda role, *parts: size_ctx if len(parts) == 1 else None

        result = resolve_prompt("nerve")
        assert result == size_ctx

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value=None)
    @patch("arqitect.brain.adapters.get_model_size_class", return_value=None)
    @patch("arqitect.brain.adapters.get_active_variant", return_value="small")
    @patch("arqitect.brain.adapters._load_context", return_value=None)
    def test_both_missing_returns_none(self, _mock_load, _mock_variant, _mock_size, _mock_name):
        assert resolve_prompt("nerve") is None


# ── resolve_meta ──────────────────────────────────────────────────────────────


class TestResolveMeta:
    """Tests for meta.json resolution with deep merge behavior."""

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value="llama-7b")
    @patch("arqitect.brain.adapters.get_active_variant", return_value="medium")
    @patch("arqitect.brain.adapters._load_meta")
    def test_deep_merge(self, mock_load_meta, _mock_variant, _mock_name):
        base = {"tuning": {"lora_rank": 8, "lora_lr": 1e-4}, "description": "base"}
        specific = {"tuning": {"lora_rank": 16}, "version": "2"}
        mock_load_meta.side_effect = lambda role, *parts: specific if len(parts) == 2 else base

        result = resolve_meta("nerve")

        assert result["tuning"]["lora_rank"] == 16  # overridden
        assert result["tuning"]["lora_lr"] == 1e-4  # kept from base
        assert result["description"] == "base"  # kept from base
        assert result["version"] == "2"  # added from specific

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value=None)
    @patch("arqitect.brain.adapters.get_active_variant", return_value="medium")
    @patch("arqitect.brain.adapters._load_meta")
    def test_no_model_slug_returns_base(self, mock_load_meta, _mock_variant, _mock_name):
        base = {"description": "base only"}
        mock_load_meta.return_value = base

        result = resolve_meta("nerve")
        assert result == base

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value="llama-7b")
    @patch("arqitect.brain.adapters.get_active_variant", return_value="medium")
    @patch("arqitect.brain.adapters._load_meta")
    def test_no_base_returns_specific(self, mock_load_meta, _mock_variant, _mock_name):
        specific = {"version": "model-only"}
        mock_load_meta.side_effect = lambda role, *parts: specific if len(parts) == 2 else None

        result = resolve_meta("nerve")
        assert result == specific

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value="llama-7b")
    @patch("arqitect.brain.adapters.get_active_variant", return_value="medium")
    @patch("arqitect.brain.adapters._load_meta")
    def test_both_none(self, mock_load_meta, _mock_variant, _mock_name):
        mock_load_meta.return_value = None

        assert resolve_meta("nerve") is None


# ── Convenience getters ───────────────────────────────────────────────────────


class TestConvenienceGetters:
    """Tests for get_temperature, get_max_tokens, and other convenience wrappers."""

    @patch("arqitect.brain.adapters.resolve_prompt")
    def test_get_temperature_from_context(self, mock_prompt):
        mock_prompt.return_value = {"temperature": 0.3}
        assert get_temperature("brain") == 0.3

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    def test_get_temperature_default(self, _mock_prompt):
        assert get_temperature("brain") == 0.7

    @patch("arqitect.brain.adapters.resolve_prompt")
    def test_get_temperature_missing_key_uses_default(self, mock_prompt):
        mock_prompt.return_value = {"system_prompt": "hi"}
        assert get_temperature("brain") == 0.7

    @patch("arqitect.brain.adapters.resolve_prompt")
    def test_get_max_tokens_from_context(self, mock_prompt):
        mock_prompt.return_value = {"max_tokens": 4096}
        assert get_max_tokens("nerve") == 4096

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    def test_get_max_tokens_default(self, _mock_prompt):
        assert get_max_tokens("nerve") == 2048

    @patch("arqitect.brain.adapters.resolve_prompt")
    def test_get_conversation_window_from_context(self, mock_prompt):
        mock_prompt.return_value = {"conversation_window": 5}
        assert get_conversation_window("brain") == 5

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    def test_get_conversation_window_default(self, _mock_prompt):
        assert get_conversation_window("brain") == 10

    @patch("arqitect.brain.adapters.resolve_prompt")
    def test_get_message_truncation_from_context(self, mock_prompt):
        mock_prompt.return_value = {"message_truncation": 500}
        assert get_message_truncation("nerve") == 500

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    def test_get_message_truncation_default(self, _mock_prompt):
        assert get_message_truncation("nerve") == 200

    @patch("arqitect.brain.adapters.resolve_meta")
    def test_get_json_mode_true(self, mock_meta):
        mock_meta.return_value = {"capabilities": {"json_mode": True}}
        assert get_json_mode("brain") is True

    @patch("arqitect.brain.adapters.resolve_meta", return_value=None)
    def test_get_json_mode_default(self, _mock_meta):
        assert get_json_mode("brain") is False


# ── get_tuning_config ─────────────────────────────────────────────────────────


class TestGetTuningConfig:
    """Tests for 4-layer tuning config resolution."""

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    @patch("arqitect.brain.adapters.resolve_meta", return_value=None)
    def test_returns_role_overrides_as_base(self, _mock_meta, _mock_prompt):
        from arqitect.brain.adapters import ROLE_TUNING_OVERRIDES
        from arqitect.types import NerveRole

        result = get_tuning_config(NerveRole.CODE)

        assert result["lora_target_modules"] == ROLE_TUNING_OVERRIDES[NerveRole.CODE]["lora_target_modules"]
        assert result["default_temperature"] == 0.2

    @patch("arqitect.brain.adapters.resolve_prompt")
    @patch("arqitect.brain.adapters.resolve_meta", return_value=None)
    def test_context_overlays_temperature(self, _mock_meta, mock_prompt):
        mock_prompt.return_value = {"temperature": 0.9, "max_tokens": 8192}

        result = get_tuning_config("brain")

        assert result["default_temperature"] == 0.9
        assert result["default_max_tokens"] == 8192

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    @patch("arqitect.brain.adapters.resolve_meta")
    def test_meta_overlays_tuning_fields(self, mock_meta, _mock_prompt):
        mock_meta.return_value = {
            "tuning": {"lora_rank": 32, "lora_epochs": 5},
        }

        result = get_tuning_config("brain")

        assert result["lora_rank"] == 32
        assert result["lora_epochs"] == 5

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    @patch("arqitect.brain.adapters.resolve_meta", return_value=None)
    @patch("arqitect.brain.adapters._load_json", return_value=None)
    def test_nerve_name_loads_nerve_meta(self, mock_load, _mock_meta, _mock_prompt):
        result = get_tuning_config("brain", nerve_name="joke_nerve")

        # Verify _load_json was called with the nerve meta path
        mock_load.assert_called_once()
        call_path = mock_load.call_args[0][0]
        assert "nerves" in call_path
        assert "joke_nerve" in call_path
        assert "meta.json" in call_path

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    @patch("arqitect.brain.adapters.resolve_meta", return_value=None)
    def test_unknown_role_returns_empty_base(self, _mock_meta, _mock_prompt):
        result = get_tuning_config("totally_unknown_role_xyz")
        assert isinstance(result, dict)
