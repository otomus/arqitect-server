"""Tests for inference provider passthrough — each provider correctly forwards
model, prompt, system, max_tokens, temperature, json_mode to its API client.

Every provider must pass community adapter config through to the underlying
API without hardcoding defaults. The router resolves adapter values before
calling provider.generate(), and the provider must forward them faithfully.
"""

import importlib
import threading
from unittest.mock import MagicMock, patch

import pytest
from dirty_equals import IsInstance, IsPositiveInt
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mock_anthropic_client() -> MagicMock:
    """Build a mock Anthropic client with a successful response."""
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="response")]
    )
    return client


def _mock_openai_client() -> MagicMock:
    """Build a mock OpenAI-compatible client with a successful response."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="response"))]
    )
    return client


def _mock_gguf_llm() -> MagicMock:
    """Build a mock llama_cpp Llama instance with a successful response."""
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": " response "}}]
    }
    return llm


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestAnthropicProvider:
    """Anthropic provider passes parameters to anthropic.Anthropic.messages.create."""

    def _make_provider(self):
        with patch("arqitect.inference.providers.anthropic.get_secret", return_value="sk-test"):
            from arqitect.inference.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider(api_key="sk-test")
        client = _mock_anthropic_client()
        provider._client = client
        return provider, client

    def test_forwards_all_params(self):
        """All generation params reach the Anthropic API client."""
        provider, client = self._make_provider()

        result = provider.generate(
            model="claude-sonnet-4-20250514",
            prompt="Hello",
            system="Be helpful",
            max_tokens=512,
            temperature=0.3,
        )

        assert result == "response"
        kwargs = client.messages.create.call_args[1]
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.3
        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert kwargs["system"] == "Be helpful"

    def test_system_omitted_when_empty(self):
        """Empty system prompt is not sent to the API."""
        provider, client = self._make_provider()

        provider.generate(model="claude-sonnet-4-20250514", prompt="Hi", system="")

        kwargs = client.messages.create.call_args[1]
        assert "system" not in kwargs

    def test_low_temperature_forwarded(self):
        """Temperature=0.0 (from community adapter) is forwarded, not replaced by default."""
        provider, client = self._make_provider()

        provider.generate(
            model="claude-sonnet-4-20250514", prompt="x",
            temperature=0.0, max_tokens=50,
        )

        kwargs = client.messages.create.call_args[1]
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_tokens"] == 50

    @given(temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False))
    @settings(max_examples=50)
    def test_temperature_clamping_with_random_floats(self, temperature):
        """Any valid temperature float is forwarded unchanged to the API."""
        provider, client = self._make_provider()

        provider.generate(
            model="claude-sonnet-4-20250514",
            prompt="test",
            temperature=temperature,
        )

        kwargs = client.messages.create.call_args[1]
        assert kwargs["temperature"] == temperature

    @given(max_tokens=st.integers(min_value=1, max_value=100_000))
    @settings(max_examples=50)
    def test_max_tokens_with_random_ints(self, max_tokens):
        """Any positive integer max_tokens is forwarded unchanged to the API."""
        provider, client = self._make_provider()

        provider.generate(
            model="claude-sonnet-4-20250514",
            prompt="test",
            max_tokens=max_tokens,
        )

        kwargs = client.messages.create.call_args[1]
        assert kwargs["max_tokens"] == max_tokens
        assert kwargs["max_tokens"] == IsPositiveInt

    @given(json_mode=st.booleans())
    @settings(max_examples=10)
    def test_json_mode_flag_handling(self, json_mode):
        """Anthropic provider ignores json_mode (not supported by API) without crashing."""
        provider, client = self._make_provider()

        # Anthropic's generate() signature accepts json_mode but does not use it
        result = provider.generate(
            model="claude-sonnet-4-20250514",
            prompt="test",
            json_mode=json_mode,
        )

        assert result == IsInstance(str)


# ---------------------------------------------------------------------------
# OpenAI-compatible (base class for 7+ providers)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestOpenAICompatProvider:
    """OpenAI-compat provider passes parameters to OpenAI chat.completions.create."""

    def _make_provider(self):
        with patch("arqitect.inference.providers.openai_compat.get_secret", return_value="sk-test"):
            from arqitect.inference.providers.openai_compat import OpenAICompatProvider
            provider = OpenAICompatProvider(api_key="sk-test", base_url="https://api.openai.com/v1")
        client = _mock_openai_client()
        provider._client = client
        return provider, client

    def test_forwards_all_params(self):
        """All generation params reach the OpenAI API client."""
        provider, client = self._make_provider()

        result = provider.generate(
            model="gpt-4o",
            prompt="Hello",
            system="Be helpful",
            max_tokens=512,
            temperature=0.3,
            json_mode=False,
        )

        assert result == "response"
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.3
        assert kwargs["messages"] == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        assert "response_format" not in kwargs

    def test_json_mode_sets_response_format(self):
        """json_mode=True from community adapter produces response_format."""
        provider, client = self._make_provider()

        provider.generate(model="gpt-4o", prompt="classify", json_mode=True)

        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_no_system_omits_system_message(self):
        """Empty system string produces only a user message."""
        provider, client = self._make_provider()

        provider.generate(model="gpt-4o", prompt="Hi", system="")

        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    @given(temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False))
    @settings(max_examples=50)
    def test_temperature_clamping_with_random_floats(self, temperature):
        """Any valid temperature float is forwarded unchanged to the API."""
        provider, client = self._make_provider()

        provider.generate(
            model="gpt-4o",
            prompt="test",
            temperature=temperature,
        )

        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["temperature"] == temperature

    @given(max_tokens=st.integers(min_value=1, max_value=100_000))
    @settings(max_examples=50)
    def test_max_tokens_with_random_ints(self, max_tokens):
        """Any positive integer max_tokens is forwarded unchanged to the API."""
        provider, client = self._make_provider()

        provider.generate(
            model="gpt-4o",
            prompt="test",
            max_tokens=max_tokens,
        )

        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["max_tokens"] == max_tokens
        assert kwargs["max_tokens"] == IsPositiveInt

    @given(json_mode=st.booleans())
    @settings(max_examples=10)
    def test_json_mode_flag_handling(self, json_mode):
        """json_mode=True sets response_format, False omits it entirely."""
        provider, client = self._make_provider()

        provider.generate(
            model="gpt-4o",
            prompt="test",
            json_mode=json_mode,
        )

        kwargs = client.chat.completions.create.call_args[1]
        if json_mode:
            assert kwargs["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in kwargs


# ---------------------------------------------------------------------------
# OpenAI-compat subclasses — all inherit generate() unchanged
# ---------------------------------------------------------------------------

_OPENAI_SUBCLASSES = [
    ("arqitect.inference.providers.groq", "GroqProvider", "groq_api_key"),
    ("arqitect.inference.providers.deepseek", "DeepSeekProvider", "deepseek_api_key"),
    ("arqitect.inference.providers.mistral", "MistralProvider", "mistral_api_key"),
    ("arqitect.inference.providers.openrouter", "OpenRouterProvider", "openrouter_api_key"),
    ("arqitect.inference.providers.google_gemini", "GoogleGeminiProvider", "google_ai_api_key"),
    ("arqitect.inference.providers.xai", "XAIProvider", "xai_api_key"),
    ("arqitect.inference.providers.together_ai", "TogetherAIProvider", "together_api_key"),
]


@pytest.mark.timeout(10)
@pytest.mark.parametrize("module_path,class_name,secret_key", _OPENAI_SUBCLASSES,
                         ids=[c[1] for c in _OPENAI_SUBCLASSES])
class TestOpenAICompatSubclasses:
    """Each OpenAI-compat subclass correctly passes params through to its client."""

    def _make_provider(self, module_path, class_name, secret_key):
        with patch(f"{module_path}.get_secret", return_value="sk-test"):
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            provider = cls(api_key="sk-test")

        client = _mock_openai_client()
        # Override the response text for subclass identification
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="subclass response"))]
        )
        provider._client = client
        return provider, client

    def test_forwards_params(self, module_path, class_name, secret_key):
        """Provider forwards model, prompt, system, max_tokens, temperature."""
        provider, client = self._make_provider(module_path, class_name, secret_key)

        result = provider.generate(
            model="test-model",
            prompt="Hello",
            system="Be helpful",
            max_tokens=256,
            temperature=0.3,
            json_mode=False,
        )

        assert result == "subclass response"
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["model"] == "test-model"
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.3
        assert kwargs["messages"] == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]

    def test_json_mode(self, module_path, class_name, secret_key):
        """json_mode=True produces response_format on the API call."""
        provider, client = self._make_provider(module_path, class_name, secret_key)

        provider.generate(model="test-model", prompt="x", json_mode=True)

        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# GGUF (local inference)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGGUFProvider:
    """GGUF provider passes parameters to llama_cpp Llama.create_chat_completion."""

    def _make_provider(self):
        from arqitect.inference.providers.gguf import GGUFProvider
        provider = GGUFProvider(models_dir="/tmp/fake")

        mock_llm = _mock_gguf_llm()
        provider._models["test-model"] = mock_llm
        provider._model_locks["test-model"] = threading.Lock()
        return provider, mock_llm

    def test_forwards_all_params(self):
        """All generation params reach llama_cpp's create_chat_completion."""
        provider, mock_llm = self._make_provider()

        result = provider.generate(
            model="test-model",
            prompt="Hello",
            system="Be helpful",
            max_tokens=256,
            temperature=0.3,
            json_mode=False,
        )

        assert result == "response"  # stripped
        kwargs = mock_llm.create_chat_completion.call_args[1]
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.3
        assert kwargs["messages"] == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        assert "response_format" not in kwargs

    def test_json_mode_sets_response_format(self):
        """json_mode=True produces response_format on the llama_cpp call."""
        provider, mock_llm = self._make_provider()

        provider.generate(model="test-model", prompt="classify", json_mode=True)

        kwargs = mock_llm.create_chat_completion.call_args[1]
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_no_system_omits_system_message(self):
        """Empty system string produces only a user message."""
        provider, mock_llm = self._make_provider()

        provider.generate(model="test-model", prompt="Hi", system="")

        kwargs = mock_llm.create_chat_completion.call_args[1]
        assert kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    def test_preload_delegates_to_ensure_loaded(self):
        """preload() is the public interface; it delegates to _ensure_loaded."""
        from arqitect.inference.providers.gguf import GGUFProvider
        provider = GGUFProvider(models_dir="/tmp/fake")
        provider._models["already-loaded"] = MagicMock()
        provider._model_locks["already-loaded"] = threading.Lock()

        # Should not raise — model already loaded
        provider.preload("already-loaded")
        assert "already-loaded" in provider.list_loaded()

    def test_list_loaded_returns_model_names(self):
        """list_loaded() returns all currently loaded model names."""
        provider, _ = self._make_provider()
        assert provider.list_loaded() == ["test-model"]

    def test_generate_strips_whitespace(self):
        """GGUF provider strips leading/trailing whitespace from responses."""
        provider, mock_llm = self._make_provider()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "  padded response  "}}]
        }

        result = provider.generate(model="test-model", prompt="test")

        assert result == "padded response"

    @given(temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False))
    @settings(max_examples=50)
    def test_temperature_clamping_with_random_floats(self, temperature):
        """Any valid temperature float is forwarded unchanged to llama_cpp."""
        provider, mock_llm = self._make_provider()

        provider.generate(
            model="test-model",
            prompt="test",
            temperature=temperature,
        )

        kwargs = mock_llm.create_chat_completion.call_args[1]
        assert kwargs["temperature"] == temperature

    @given(max_tokens=st.integers(min_value=1, max_value=100_000))
    @settings(max_examples=50)
    def test_max_tokens_with_random_ints(self, max_tokens):
        """Any positive integer max_tokens is forwarded unchanged to llama_cpp."""
        provider, mock_llm = self._make_provider()

        provider.generate(
            model="test-model",
            prompt="test",
            max_tokens=max_tokens,
        )

        kwargs = mock_llm.create_chat_completion.call_args[1]
        assert kwargs["max_tokens"] == max_tokens
        assert kwargs["max_tokens"] == IsPositiveInt

    @given(json_mode=st.booleans())
    @settings(max_examples=10)
    def test_json_mode_flag_handling(self, json_mode):
        """json_mode=True sets response_format, False omits it entirely."""
        provider, mock_llm = self._make_provider()

        provider.generate(
            model="test-model",
            prompt="test",
            json_mode=json_mode,
        )

        kwargs = mock_llm.create_chat_completion.call_args[1]
        if json_mode:
            assert kwargs["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in kwargs


# ---------------------------------------------------------------------------
# Router integration — adapter defaults flow through to any provider
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestAdapterDefaultsReachProvider:
    """Verify community adapter config flows: adapter -> router -> provider."""

    def test_adapter_temperature_reaches_provider(self):
        """Community adapter temperature=0.3 reaches the provider's generate()."""
        from arqitect.inference.router import generate_for_role, reset_cache

        reset_cache()
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "ok"
        mock_provider.supports_lora = False

        with patch("arqitect.inference.router._resolve_role_config", return_value=("anthropic", "claude-sonnet-4-20250514")), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider), \
             patch("arqitect.brain.adapters.get_temperature", return_value=0.3), \
             patch("arqitect.brain.adapters.get_max_tokens", return_value=256), \
             patch("arqitect.brain.adapters.get_json_mode", return_value=True):

            generate_for_role("brain", "route this")

        kwargs = mock_provider.generate.call_args[1]
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 256
        assert kwargs["json_mode"] is True
        reset_cache()

    def test_caller_overrides_adapter(self):
        """Explicit caller values override community adapter defaults."""
        from arqitect.inference.router import generate_for_role, reset_cache

        reset_cache()
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "ok"
        mock_provider.supports_lora = False

        with patch("arqitect.inference.router._resolve_role_config", return_value=("groq", "llama-3.3-70b")), \
             patch("arqitect.inference.router._get_or_create_provider", return_value=mock_provider):

            generate_for_role("nerve", "x", temperature=0.9, max_tokens=4096, json_mode=False)

        kwargs = mock_provider.generate.call_args[1]
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_tokens"] == 4096
        assert kwargs["json_mode"] is False
        reset_cache()


# ---------------------------------------------------------------------------
# Base class contract — _build_messages
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestBuildMessages:
    """InferenceProvider._build_messages follows the chat message contract."""

    def test_with_system(self):
        """System string produces system + user message pair."""
        from arqitect.inference.providers.base import InferenceProvider
        messages = InferenceProvider._build_messages("hello", "system prompt")
        assert messages == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ]

    def test_without_system(self):
        """Empty system string produces only a user message."""
        from arqitect.inference.providers.base import InferenceProvider
        messages = InferenceProvider._build_messages("hello", "")
        assert messages == [{"role": "user", "content": "hello"}]

    def test_without_system_default(self):
        """Omitting system entirely produces only a user message."""
        from arqitect.inference.providers.base import InferenceProvider
        messages = InferenceProvider._build_messages("hello")
        assert messages == [{"role": "user", "content": "hello"}]
