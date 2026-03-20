"""Tests for arqitect.brain.adapters — model sizing, prompt resolution, and tuning config."""

from unittest.mock import patch

import pytest
from dirty_equals import IsDict, IsInstance, IsPartialDict
from hypothesis import given, strategies as st

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
    SIZE_CLASSES,
)


# ── _extract_param_billions ───────────────────────────────────────────────────


@pytest.mark.timeout(10)
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

    @given(
        param_count=st.floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False),
        prefix=st.from_regex(r"[a-z][a-z\-]{0,20}", fullmatch=True),
        suffix=st.sampled_from([".gguf", ".bin", ".safetensors", ""]),
    )
    def test_roundtrip_extraction(self, param_count, prefix, suffix):
        """Any model name with a valid '{N}b' token should extract that number.

        The prefix is restricted to letters and hyphens so it cannot contain
        a digit-b sequence that the regex would match first.
        """
        formatted = f"{param_count:.1f}"
        model_name = f"{prefix}-{formatted}b{suffix}"
        result = _extract_param_billions(model_name)
        assert result == pytest.approx(float(formatted), abs=1e-6)

    @given(name=st.from_regex(r"[a-z][a-z\-]{0,20}\.(gguf|bin)", fullmatch=True))
    def test_no_param_token_returns_none(self, name):
        """Model names without a '{N}b' pattern should return None."""
        assert _extract_param_billions(name) is None


# ── _params_to_size_class ────────────────────────────────────────────────────


@pytest.mark.timeout(10)
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

    @given(billions=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False))
    def test_always_returns_valid_size_class(self, billions):
        """Every positive param count must map to a valid size class."""
        result = _params_to_size_class(billions)
        assert result in SIZE_CLASSES

    @given(billions=st.floats(min_value=32.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    def test_large_models_always_large(self, billions):
        """Anything at or above the medium boundary is always 'large'."""
        assert _params_to_size_class(billions) == "large"

    @given(billions=st.floats(min_value=0.01, max_value=2.99, allow_nan=False, allow_infinity=False))
    def test_tiny_models_always_tinylm(self, billions):
        """Anything below 3B is always 'tinylm'."""
        assert _params_to_size_class(billions) == "tinylm"


# ── _model_slug ───────────────────────────────────────────────────────────────


@pytest.mark.timeout(10)
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

    @given(
        base=st.from_regex(r"[A-Za-z][A-Za-z0-9\- ]{0,30}", fullmatch=True),
        ext=st.sampled_from([".gguf", ".bin", ".safetensors", ""]),
    )
    def test_slug_is_always_lowercase_no_spaces(self, base, ext):
        """Slugs must be lowercase and never contain spaces."""
        slug = _model_slug(base + ext)
        assert slug == slug.lower()
        assert " " not in slug

    @given(
        base=st.from_regex(r"[A-Za-z][A-Za-z0-9\-]{0,20}", fullmatch=True),
        ext=st.sampled_from([".gguf", ".bin", ".safetensors"]),
    )
    def test_slug_never_contains_extension(self, base, ext):
        """Known extensions must be stripped from the slug."""
        slug = _model_slug(base + ext)
        assert not slug.endswith(ext)


# ── get_model_size_class ──────────────────────────────────────────────────────


@pytest.mark.timeout(10)
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


@pytest.mark.timeout(10)
class TestResolvePrompt:
    """Tests for prompt resolution with model-specific and size-class fallback."""

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value="llama-7b")
    @patch("arqitect.brain.adapters.get_model_size_class", return_value="medium")
    @patch("arqitect.brain.adapters._load_context")
    def test_model_specific_found(self, mock_load, _mock_size, _mock_name):
        model_ctx = {"temperature": 0.3, "system_prompt": "be precise"}
        mock_load.side_effect = lambda role, *parts: model_ctx if len(parts) == 2 else None

        result = resolve_prompt("nerve")
        assert result == IsDict(temperature=0.3, system_prompt=IsInstance(str))

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value="llama-7b")
    @patch("arqitect.brain.adapters.get_model_size_class", return_value="medium")
    @patch("arqitect.brain.adapters._load_context")
    def test_falls_back_to_size_class(self, mock_load, _mock_size, _mock_name):
        size_ctx = {"temperature": 0.5}
        # Model-specific returns None, size-class returns context
        mock_load.side_effect = lambda role, *parts: size_ctx if len(parts) == 1 else None

        result = resolve_prompt("nerve")
        assert result == IsDict(temperature=0.5)

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value=None)
    @patch("arqitect.brain.adapters.get_model_size_class", return_value=None)
    @patch("arqitect.brain.adapters.get_active_variant", return_value="small")
    @patch("arqitect.brain.adapters._load_context", return_value=None)
    def test_both_missing_returns_none(self, _mock_load, _mock_variant, _mock_size, _mock_name):
        assert resolve_prompt("nerve") is None


# ── resolve_meta ──────────────────────────────────────────────────────────────


@pytest.mark.timeout(10)
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

        assert result == IsDict(
            tuning=IsDict(lora_rank=16, lora_lr=1e-4),
            description="base",
            version="2",
        )

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value=None)
    @patch("arqitect.brain.adapters.get_active_variant", return_value="medium")
    @patch("arqitect.brain.adapters._load_meta")
    def test_no_model_slug_returns_base(self, mock_load_meta, _mock_variant, _mock_name):
        base = {"description": "base only"}
        mock_load_meta.return_value = base

        result = resolve_meta("nerve")
        assert result == IsDict(description="base only")

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value="llama-7b")
    @patch("arqitect.brain.adapters.get_active_variant", return_value="medium")
    @patch("arqitect.brain.adapters._load_meta")
    def test_no_base_returns_specific(self, mock_load_meta, _mock_variant, _mock_name):
        specific = {"version": "model-only"}
        mock_load_meta.side_effect = lambda role, *parts: specific if len(parts) == 2 else None

        result = resolve_meta("nerve")
        assert result == IsDict(version="model-only")

    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value="llama-7b")
    @patch("arqitect.brain.adapters.get_active_variant", return_value="medium")
    @patch("arqitect.brain.adapters._load_meta")
    def test_both_none(self, mock_load_meta, _mock_variant, _mock_name):
        mock_load_meta.return_value = None

        assert resolve_meta("nerve") is None

    @given(
        base_keys=st.dictionaries(
            keys=st.sampled_from(["description", "provider", "model", "has_lora"]),
            values=st.text(min_size=1, max_size=10),
            min_size=1,
            max_size=3,
        ),
        override_keys=st.dictionaries(
            keys=st.sampled_from(["description", "provider", "version"]),
            values=st.text(min_size=1, max_size=10),
            min_size=1,
            max_size=2,
        ),
    )
    @patch("arqitect.brain.adapters.get_model_name_for_role", return_value="llama-7b")
    @patch("arqitect.brain.adapters.get_active_variant", return_value="medium")
    @patch("arqitect.brain.adapters._load_meta")
    def test_merge_specific_overrides_base(self, mock_load_meta, _mock_variant, _mock_name, base_keys, override_keys):
        """Model-specific keys must override base keys; base-only keys survive."""
        mock_load_meta.side_effect = lambda role, *parts: override_keys if len(parts) == 2 else base_keys

        result = resolve_meta("nerve")

        # Every key from the override must appear with its override value
        for key, val in override_keys.items():
            assert result[key] == val
        # Base-only keys must survive the merge
        for key, val in base_keys.items():
            if key not in override_keys:
                assert result[key] == val


# ── Convenience getters ───────────────────────────────────────────────────────


@pytest.mark.timeout(10)
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


@pytest.mark.timeout(10)
class TestGetTuningConfig:
    """Tests for 4-layer tuning config resolution."""

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    @patch("arqitect.brain.adapters.resolve_meta", return_value=None)
    def test_returns_role_overrides_as_base(self, _mock_meta, _mock_prompt):
        from arqitect.brain.adapters import ROLE_TUNING_OVERRIDES
        from arqitect.types import NerveRole

        result = get_tuning_config(NerveRole.CODE)

        assert result == IsPartialDict(
            lora_target_modules=ROLE_TUNING_OVERRIDES[NerveRole.CODE]["lora_target_modules"],
            default_temperature=0.2,
        )

    @patch("arqitect.brain.adapters.resolve_prompt")
    @patch("arqitect.brain.adapters.resolve_meta", return_value=None)
    def test_context_overlays_temperature(self, _mock_meta, mock_prompt):
        mock_prompt.return_value = {"temperature": 0.9, "max_tokens": 8192}

        result = get_tuning_config("brain")

        assert result == IsPartialDict(default_temperature=0.9, default_max_tokens=8192)

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    @patch("arqitect.brain.adapters.resolve_meta")
    def test_meta_overlays_tuning_fields(self, mock_meta, _mock_prompt):
        mock_meta.return_value = {
            "tuning": {"lora_rank": 32, "lora_epochs": 5},
        }

        result = get_tuning_config("brain")

        assert result == IsPartialDict(lora_rank=32, lora_epochs=5)

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
        assert result == IsInstance(dict)

    @patch("arqitect.brain.adapters.resolve_prompt", return_value=None)
    @patch("arqitect.brain.adapters.resolve_meta", return_value=None)
    def test_tuning_config_always_returns_dict(self, _mock_meta, _mock_prompt):
        """Contract: get_tuning_config never returns None."""
        result = get_tuning_config("brain")
        assert isinstance(result, dict)
