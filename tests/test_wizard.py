"""Tests for the arqitect init wizard.

Mocks all interactive I/O (input + file dialogs) so tests run headlessly.
Uses dirty_equals for flexible config assertions and parametrize for
repetitive patterns.
"""

import copy
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml
from dirty_equals import IsStr, IsDict, IsPartialDict, IsPositiveInt

from arqitect.cli.wizard import (
    _step_environment,
    _step_brain_model,
    _step_vision,
    _step_hearing,
    _step_personality,
    _step_connectors,
    _step_touch,
    _step_embeddings,
    _step_storage,
    _step_admin,
    _write_config,
    find_local_name,
    resolve_model_selection,
    run_wizard,
    STEPS,
)
from arqitect.config.defaults import DEFAULTS


@pytest.fixture()
def fresh_config() -> dict:
    """Return a deep copy of DEFAULTS for each test."""
    return copy.deepcopy(DEFAULTS)


# -- Helpers ----------------------------------------------------------------

class _InputSequence:
    """Feed a predefined sequence of answers to input() calls."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self._idx = 0

    def __call__(self, prompt: str = "") -> str:
        if self._idx >= len(self._answers):
            return ""
        val = self._answers[self._idx]
        self._idx += 1
        return val


# -- Model resolution (pure logic, no I/O) ---------------------------------

@pytest.mark.timeout(10)
class TestFindLocalName:
    """find_local_name resolves symlinks and direct matches in models_dir."""

    def test_direct_file_match(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        result = find_local_name(tmp_path, model_file.resolve())
        assert result == "model.gguf"

    def test_symlink_resolves_to_target(self, tmp_path: Path) -> None:
        blobs_dir = tmp_path / "blobs"
        blobs_dir.mkdir()
        blob = blobs_dir / "sha256-abc123"
        blob.touch()

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        link = models_dir / "Qwen2.5-Coder-7B.gguf"
        link.symlink_to(blob)

        result = find_local_name(models_dir, blob.resolve())
        assert result == "Qwen2.5-Coder-7B.gguf"

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "other.gguf").touch()

        unrelated = tmp_path / "elsewhere" / "blob"
        unrelated.parent.mkdir()
        unrelated.touch()

        result = find_local_name(models_dir, unrelated.resolve())
        assert result is None

    def test_nonexistent_dir_returns_none(self, tmp_path: Path) -> None:
        fake_dir = tmp_path / "does_not_exist"
        target = tmp_path / "file.gguf"
        target.touch()
        result = find_local_name(fake_dir, target.resolve())
        assert result is None

    def test_multiple_symlinks_returns_first_match(self, tmp_path: Path) -> None:
        blob = tmp_path / "blob"
        blob.touch()

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "link_a.gguf").symlink_to(blob)
        (models_dir / "link_b.gguf").symlink_to(blob)

        result = find_local_name(models_dir, blob.resolve())
        assert result in ("link_a.gguf", "link_b.gguf")


@pytest.mark.timeout(10)
class TestResolveModelSelection:
    """resolve_model_selection picks the right (model_name, models_dir) pair."""

    def test_symlink_in_models_dir(self, tmp_path: Path) -> None:
        """Picking a blob that has a symlink in models_dir returns the symlink name."""
        blobs_dir = tmp_path / "blobs"
        blobs_dir.mkdir()
        blob = blobs_dir / "sha256-abc123"
        blob.touch()

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "Qwen2.5-Coder-7B.gguf").symlink_to(blob)

        name, dir_ = resolve_model_selection(str(blob), str(models_dir))
        assert name == "Qwen2.5-Coder-7B.gguf"
        assert dir_ == str(models_dir)

    def test_direct_file_in_models_dir(self, tmp_path: Path) -> None:
        """Picking a file already in models_dir returns its name."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model = models_dir / "brain.gguf"
        model.touch()

        name, dir_ = resolve_model_selection(str(model), str(models_dir))
        assert name == "brain.gguf"
        assert dir_ == str(models_dir)

    def test_file_outside_models_dir_updates_dir(self, tmp_path: Path) -> None:
        """Picking a file not reachable from models_dir sets models_dir to its parent."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        model = other_dir / "custom.gguf"
        model.touch()

        name, dir_ = resolve_model_selection(str(model), str(models_dir))
        assert name == "custom.gguf"
        assert dir_ == str(other_dir)

    def test_no_models_dir_sets_from_file_parent(self, tmp_path: Path) -> None:
        """When models_dir is empty, use the selected file's parent."""
        some_dir = tmp_path / "somewhere"
        some_dir.mkdir()
        model = some_dir / "model.gguf"
        model.touch()

        name, dir_ = resolve_model_selection(str(model), "")
        assert name == "model.gguf"
        assert dir_ == str(some_dir)

    def test_second_pick_preserves_models_dir(self, tmp_path: Path) -> None:
        """Simulates picking brain from blobs, then vision from models_dir.

        Both files share the same real target via symlink -- models_dir
        should stay stable and both names should resolve locally.
        """
        blobs_dir = tmp_path / "blobs"
        blobs_dir.mkdir()
        brain_blob = blobs_dir / "sha256-brain"
        brain_blob.touch()
        vision_file = tmp_path / "models" / "moondream.gguf"

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "Qwen.gguf").symlink_to(brain_blob)
        vision_file.touch()

        # First pick: brain blob
        name1, dir1 = resolve_model_selection(str(brain_blob), str(models_dir))
        assert name1 == "Qwen.gguf"
        assert dir1 == str(models_dir)

        # Second pick: vision file from same models_dir
        name2, dir2 = resolve_model_selection(str(vision_file), dir1)
        assert name2 == "moondream.gguf"
        assert dir2 == str(models_dir)


# -- Step tests -------------------------------------------------------------

@pytest.mark.timeout(10)
class TestStepEnvironment:
    """Environment step should set top-level environment field."""

    @pytest.mark.parametrize(
        ("input_choice", "expected_env"),
        [
            ("1", "desktop"),
            ("2", "server"),
            ("3", "iot"),
            ("", "server"),  # default
        ],
        ids=["desktop", "server", "iot", "default-is-server"],
    )
    def test_environment_choices(
        self, fresh_config: dict, input_choice: str, expected_env: str
    ) -> None:
        answers = _InputSequence([input_choice])
        with patch("builtins.input", answers):
            _step_environment(fresh_config)
        assert fresh_config["environment"] == expected_env


@pytest.mark.timeout(10)
class TestStepBrainModel:
    """Brain model step should set provider, models, roles, and prompt for API keys.

    The wizard now asks: "Same provider for all roles, or configure each separately?"
    before the provider selection. All tests below use mode=1 (uniform) unless testing
    per-role mode explicitly.
    """

    @patch("arqitect.cli.wizard._open_file_dialog", return_value="/models/test.gguf")
    def test_gguf_same_model_all_roles(self, _mock_dialog: MagicMock, fresh_config: dict) -> None:
        fresh_config["environment"] = "desktop"
        answers = _InputSequence([
            "1",               # mode: same for all
            "1",               # provider: GGUF (desktop option 1)
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"] == IsPartialDict(
            provider="gguf",
            models_dir="/models",
            models=IsPartialDict(brain="test.gguf", nerve="test.gguf"),
        )
        # Per-role config also written
        assert fresh_config["inference"]["roles"]["brain"]["provider"] == "gguf"
        assert fresh_config["inference"]["roles"]["nerve"]["model"] == "test.gguf"

    @patch("arqitect.cli.wizard._open_file_dialog")
    def test_gguf_different_model_per_role(self, mock_dialog: MagicMock, fresh_config: dict) -> None:
        fresh_config["environment"] = "desktop"
        mock_dialog.side_effect = [
            "/models/brain.gguf",
            "/models/nerve.gguf",
            "/models/coder.gguf",
            "/models/creative.gguf",
            "/models/comm.gguf",
        ]
        answers = _InputSequence([
            "2",               # mode: per-role
            "1",               # brain provider: GGUF
            "1",               # nerve provider: GGUF
            "1",               # coder provider: GGUF
            "1",               # creative provider: GGUF
            "1",               # communication provider: GGUF
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"]["models"] == IsPartialDict(
            brain="brain.gguf",
            nerve="nerve.gguf",
            coder="coder.gguf",
            creative="creative.gguf",
            communication="comm.gguf",
        )

    @patch("arqitect.cli.wizard._open_file_dialog", return_value="/custom/models/q.gguf")
    def test_gguf_models_dir_derived_from_file(self, _mock_file: MagicMock, fresh_config: dict) -> None:
        fresh_config["environment"] = "desktop"
        answers = _InputSequence([
            "1",               # mode: same for all
            "1",               # GGUF
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"]["models_dir"] == "/custom/models"
        assert fresh_config["inference"]["models"]["brain"] == "q.gguf"

    def test_anthropic_from_server(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "server"
        answers = _InputSequence([
            "1",               # mode: same for all
            "1",               # Anthropic (server option 1)
            "",                # default model
            "sk-test-key",     # API key (required)
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"]["provider"] == "anthropic"
        assert fresh_config["secrets"]["anthropic_api_key"] == "sk-test-key"

    def test_anthropic_from_desktop(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "desktop"
        answers = _InputSequence([
            "1",               # mode: same for all
            "2",               # Anthropic (desktop option 2: GGUF=1, Anthropic=2)
            "",                # default model
            "sk-key",          # API key (required)
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"]["provider"] == "anthropic"

    def test_anthropic_api_key_required(self, fresh_config: dict) -> None:
        """Cloud providers now require an API key -- empty key prompts again."""
        fresh_config["environment"] = "server"
        answers = _InputSequence([
            "1",               # mode: same for all
            "1",               # Anthropic (server option 1)
            "claude-haiku-4-5-20251001",
            "",                # first attempt: empty
            "sk-finally",      # second attempt: valid
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"]["models"]["brain"] == "claude-haiku-4-5-20251001"
        assert fresh_config["secrets"]["anthropic_api_key"] == "sk-finally"

    def test_openai_provider(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "server"
        answers = _InputSequence([
            "1",               # mode: same for all
            "2",               # OpenAI (server option 2)
            "https://my-proxy.com/v1",  # base URL (prompted first if unset)
            "gpt-4o-mini",     # model name
            "sk-openai-key",   # API key (required)
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"] == IsPartialDict(provider="openai")
        assert fresh_config["inference"]["models"]["brain"] == "gpt-4o-mini"
        assert fresh_config["secrets"] == IsPartialDict(
            openai_base_url="https://my-proxy.com/v1",
            openai_api_key="sk-openai-key",
        )

    def test_groq_provider(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "server"
        answers = _InputSequence([
            "1",               # mode: same for all
            "3",               # Groq (server option 3)
            "llama-3.1-8b",
            "gsk_testkey",     # API key (required)
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"] == IsPartialDict(provider="groq")
        assert fresh_config["inference"]["models"]["brain"] == "llama-3.1-8b"
        assert fresh_config["secrets"]["groq_api_key"] == "gsk_testkey"

    def test_deepseek_provider(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "desktop"
        answers = _InputSequence([
            "1",               # mode: same for all
            "5",               # DeepSeek (desktop: GGUF=1, Anthropic=2, OpenAI=3, Groq=4, DeepSeek=5)
            "",                # default model
            "sk-ds-key",       # API key (required)
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"]["provider"] == "deepseek"

    def test_server_includes_local_options(self, fresh_config: dict) -> None:
        """Server shows cloud providers first, then local (GGUF last)."""
        fresh_config["environment"] = "server"
        answers = _InputSequence([
            "1",               # mode: same for all
            "3",               # Groq (server: Anthropic=1, OpenAI=2, Groq=3)
            "",                # default model
            "gsk_key",         # API key
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"]["provider"] == "groq"

    def test_per_role_mixed_providers(self, fresh_config: dict) -> None:
        """Per-role mode allows mixing cloud and local providers."""
        fresh_config["environment"] = "server"
        answers = _InputSequence([
            "2",               # mode: per-role
            # brain: anthropic
            "1", "", "sk-ant-key",
            # nerve: groq
            "3", "", "gsk-key",
            # coder: groq (key already set, skipped)
            "3", "",
            # creative: anthropic (key already set, skipped)
            "1", "",
            # communication: anthropic (key already set, skipped)
            "1", "",
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"]["roles"]["brain"]["provider"] == "anthropic"
        assert fresh_config["inference"]["roles"]["nerve"]["provider"] == "groq"
        assert fresh_config["inference"]["roles"]["coder"]["provider"] == "groq"
        # Top-level uses brain's provider
        assert fresh_config["inference"]["provider"] == "anthropic"


@pytest.mark.timeout(10)
class TestStepVision:
    """Vision step should set sight sense config."""

    def test_disable_vision(self, fresh_config: dict) -> None:
        # For non-anthropic provider: 1=moondream, 2=custom, 3=skip
        answers = _InputSequence(["3"])
        with patch("builtins.input", answers):
            _step_vision(fresh_config)

        assert fresh_config["senses"]["sight"]["enabled"] is False

    @patch("arqitect.cli.wizard._open_file_dialog", return_value="/models/moondream2.gguf")
    def test_moondream_vision(self, _mock_dialog: MagicMock, fresh_config: dict) -> None:
        answers = _InputSequence(["1"])
        with patch("builtins.input", answers):
            _step_vision(fresh_config)

        assert fresh_config["senses"]["sight"] == IsPartialDict(
            enabled=True,
            provider="moondream",
        )
        assert fresh_config["inference"]["models"]["vision"] == "moondream2.gguf"
        assert fresh_config["inference"]["models_dir"] == "/models"

    def test_anthropic_vision_option_appears(self, fresh_config: dict) -> None:
        fresh_config["inference"]["provider"] = "anthropic"
        # With anthropic brain: 1=anthropic vision, 2=moondream, 3=custom, 4=skip
        answers = _InputSequence(["1"])
        with patch("builtins.input", answers):
            _step_vision(fresh_config)

        assert fresh_config["senses"]["sight"] == IsPartialDict(
            enabled=True,
            provider="anthropic",
        )

    def test_custom_vision_endpoint(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "2",               # custom endpoint (non-anthropic)
            "http://vision:8080/predict",
        ])
        with patch("builtins.input", answers):
            _step_vision(fresh_config)

        assert fresh_config["senses"]["sight"] == IsPartialDict(
            provider="custom",
            endpoint="http://vision:8080/predict",
        )


@pytest.mark.timeout(10)
class TestStepHearing:
    """Hearing step should set STT/TTS config."""

    def test_disable_hearing(self, fresh_config: dict) -> None:
        answers = _InputSequence(["3"])
        with patch("builtins.input", answers):
            _step_hearing(fresh_config)

        assert fresh_config["senses"]["hearing"]["enabled"] is False

    def test_whisper_with_tts(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "1",               # whisper
            "y",               # enable TTS
            "3",               # elevenlabs
        ])
        with patch("builtins.input", answers):
            _step_hearing(fresh_config)

        assert fresh_config["senses"]["hearing"] == IsPartialDict(
            stt="whisper",
            tts="elevenlabs",
        )

    def test_cloud_stt_without_tts(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "2",               # cloud STT
            "deepgram",
            "n",               # no TTS
        ])
        with patch("builtins.input", answers):
            _step_hearing(fresh_config)

        assert fresh_config["senses"]["hearing"]["stt"] == "deepgram"
        assert fresh_config["senses"]["hearing"].get("tts", "") == ""


@pytest.mark.timeout(10)
class TestStepPersonality:
    """Personality step should set name, tone, traits."""

    def test_preset_professional(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "MyBot",           # name
            "1",               # professional preset
            "2",               # casual formality
        ])
        with patch("builtins.input", answers):
            _step_personality(fresh_config)

        assert fresh_config["name"] == "MyBot"
        assert fresh_config["personality"] == IsPartialDict(
            preset="professional",
            tone="clear, direct, and helpful",
        )

    def test_custom_personality(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "Aria",
            "4",               # custom
            "warm and curious",
            "empathetic, witty",
            "1",               # formal
        ])
        with patch("builtins.input", answers):
            _step_personality(fresh_config)

        assert fresh_config["personality"]["tone"] == "warm and curious"
        assert fresh_config["personality"]["traits"] == ["empathetic", "witty"]
        assert fresh_config["personality"]["communication_style"]["formality"] == "formal"


@pytest.mark.timeout(10)
class TestStepConnectors:
    """Connector step should set telegram/whatsapp config."""

    def test_no_connectors(self, fresh_config: dict) -> None:
        answers = _InputSequence(["n", "n"])
        with patch("builtins.input", answers):
            _step_connectors(fresh_config)

        assert fresh_config["connectors"]["telegram"]["enabled"] is False
        assert fresh_config["connectors"]["whatsapp"]["enabled"] is False

    def test_telegram_enabled(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "y",               # enable telegram
            "mytoken123",      # bot token
            "TestBot",         # bot name
            "tb, testbot",     # aliases
            "n",               # no whatsapp
        ])
        with patch("builtins.input", answers):
            _step_connectors(fresh_config)

        assert fresh_config["connectors"]["telegram"]["enabled"] is True
        assert fresh_config["secrets"]["telegram_bot_token"] == "mytoken123"
        assert fresh_config["connectors"]["telegram"]["bot_aliases"] == ["tb", "testbot"]

    def test_whatsapp_enabled(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "n",               # no telegram
            "y",               # enable whatsapp
            "WaBot",           # bot name
        ])
        with patch("builtins.input", answers):
            _step_connectors(fresh_config)

        assert fresh_config["connectors"]["telegram"]["enabled"] is False
        assert fresh_config["connectors"]["whatsapp"] == IsPartialDict(
            enabled=True,
            bot_name="WaBot",
        )

    def test_both_connectors(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "y",               # telegram
            "tok123",          # token
            "TBot",            # bot name
            "",                # no aliases
            "y",               # whatsapp
            "WBot",
        ])
        with patch("builtins.input", answers):
            _step_connectors(fresh_config)

        assert fresh_config["connectors"]["telegram"]["enabled"] is True
        assert fresh_config["connectors"]["whatsapp"]["enabled"] is True


@pytest.mark.timeout(10)
class TestStepTouch:
    """Touch step should set environment and filesystem config."""

    def test_server_sandboxed(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "server"
        answers = _InputSequence([
            "2",               # sandboxed
            "./data",          # sandbox dir (fallback from cancelled picker)
            "2",               # allowlisted commands
            "python,git",
        ])
        with patch("builtins.input", answers), \
             patch("arqitect.cli.wizard._open_dir_dialog", return_value=""):
            _step_touch(fresh_config)

        assert fresh_config["senses"]["touch"] == IsPartialDict(
            environment="server",
            filesystem=IsPartialDict(access="sandboxed", root="./data"),
            execution=IsPartialDict(allowlist=["python", "git"]),
        )

    def test_desktop_full_access_any_command(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "desktop"
        answers = _InputSequence([
            "1",               # full access
            "1",               # any command
        ])
        with patch("builtins.input", answers):
            _step_touch(fresh_config)

        assert fresh_config["senses"]["touch"] == IsPartialDict(
            environment="desktop",
            filesystem=IsPartialDict(access="full"),
            execution=IsPartialDict(enabled=True),
        )

    def test_server_no_execution(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "server"
        answers = _InputSequence([
            "3",               # read-only
            "3",               # no execution
        ])
        with patch("builtins.input", answers):
            _step_touch(fresh_config)

        assert fresh_config["senses"]["touch"]["filesystem"]["access"] == "readonly"
        assert fresh_config["senses"]["touch"]["execution"]["enabled"] is False

    def test_iot_environment(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "iot"
        answers = _InputSequence([])
        with patch("builtins.input", answers):
            _step_touch(fresh_config)

        assert fresh_config["senses"]["touch"] == IsPartialDict(
            environment="iot",
            execution=IsPartialDict(enabled=False),
        )


@pytest.mark.timeout(10)
class TestStepEmbeddings:
    """Embeddings step should set provider and model."""

    @pytest.mark.parametrize(
        ("input_choice", "expected_provider", "expected_model"),
        [
            ("1", "local", "nomic-embed-text"),
            ("2", "local", "all-MiniLM-L6-v2"),
            ("4", "openai", "text-embedding-3-small"),
            ("5", "none", ""),
        ],
        ids=["nomic", "minilm", "openai", "none"],
    )
    def test_embedding_choices(
        self,
        fresh_config: dict,
        input_choice: str,
        expected_provider: str,
        expected_model: str,
    ) -> None:
        answers = _InputSequence([input_choice])
        with patch("builtins.input", answers):
            _step_embeddings(fresh_config)

        assert fresh_config["embeddings"] == IsPartialDict(
            provider=expected_provider,
            model=expected_model,
        )

    def test_gguf_embeddings_uses_brain_model(self, fresh_config: dict) -> None:
        fresh_config["inference"]["models"]["brain"] = "my-model.gguf"
        answers = _InputSequence(["3"])
        with patch("builtins.input", answers):
            _step_embeddings(fresh_config)

        assert fresh_config["embeddings"]["model"] == "my-model.gguf"


@pytest.mark.timeout(10)
class TestStepStorage:
    """Storage step should set Redis, cold DB, and ports."""

    def test_default_storage(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "1",               # local redis
            "1",               # sqlite
            "",                # default db name
            "",                # default episodes name
            "",                # default mcp port
            "",                # default bridge port
        ])
        with patch("builtins.input", answers):
            _step_storage(fresh_config)

        assert fresh_config["storage"]["hot"]["url"] == "redis://localhost:6379"
        assert fresh_config["storage"]["cold"]["path"] == "arqitect_memory.db"
        assert fresh_config["ports"]["mcp"] == 8100

    def test_remote_redis(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "2",               # remote redis
            "redis://my-redis:6380",
            "1",               # sqlite
            "",                # default db name
            "",                # default episodes
            "9100",            # custom mcp port
            "4000",            # custom bridge port
        ])
        with patch("builtins.input", answers):
            _step_storage(fresh_config)

        assert fresh_config["storage"]["hot"]["url"] == "redis://my-redis:6380"
        assert fresh_config["ports"] == IsPartialDict(mcp=9100, bridge=4000)

    def test_postgres_cold_storage(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "1",               # local redis
            "2",               # postgres
            "postgresql://u:p@host/db",
            "",                # default episodes
            "",                # default mcp
            "",                # default bridge
        ])
        with patch("builtins.input", answers):
            _step_storage(fresh_config)

        assert fresh_config["storage"]["cold"] == IsPartialDict(
            type="postgres",
            url="postgresql://u:p@host/db",
        )


@pytest.mark.timeout(10)
class TestStepAdmin:
    """Admin step should set name, email, and auto-generate JWT."""

    def test_admin_generates_jwt(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "Admin",
            "admin@test.com",
            "n",               # no SMTP
        ])
        with patch("builtins.input", answers):
            _step_admin(fresh_config)

        assert fresh_config["admin"] == IsPartialDict(
            name="Admin",
            email="admin@test.com",
        )
        # JWT secret is a non-trivial random string
        assert fresh_config["secrets"]["jwt_secret"] == IsStr(min_length=20)

    def test_admin_preserves_existing_jwt(self, fresh_config: dict) -> None:
        fresh_config["secrets"]["jwt_secret"] = "existing-secret"
        answers = _InputSequence([
            "Admin",
            "admin@test.com",
            "n",
        ])
        with patch("builtins.input", answers):
            _step_admin(fresh_config)

        assert fresh_config["secrets"]["jwt_secret"] == "existing-secret"

    def test_admin_with_smtp(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "Admin",
            "admin@test.com",
            "y",               # configure SMTP
            "smtp.example.com",
            "465",
            "user@example.com",
            "s3cret",
            "noreply@example.com",
        ])
        with patch("builtins.input", answers):
            _step_admin(fresh_config)

        assert fresh_config["secrets"]["smtp"] == IsPartialDict(
            host="smtp.example.com",
            port=465,
            user="user@example.com",
            password="s3cret",
        )
        assert fresh_config["secrets"]["smtp"]["from"] == "noreply@example.com"


# -- Config writer ----------------------------------------------------------

@pytest.mark.timeout(10)
class TestWriteConfig:
    """Config writer should produce valid YAML with ordered keys."""

    def test_roundtrip(self, fresh_config: dict, tmp_path: Path) -> None:
        yaml_path = tmp_path / "arqitect.yaml"
        _write_config(fresh_config, yaml_path)

        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["name"] == fresh_config["name"]
        assert loaded["inference"]["provider"] == fresh_config["inference"]["provider"]
        assert loaded["secrets"]["jwt_secret"] == fresh_config["secrets"]["jwt_secret"]

    def test_key_order(self, fresh_config: dict, tmp_path: Path) -> None:
        yaml_path = tmp_path / "arqitect.yaml"
        _write_config(fresh_config, yaml_path)

        with open(yaml_path) as f:
            content = f.read()

        name_pos = content.index("name:")
        inference_pos = content.index("inference:")
        secrets_pos = content.index("secrets:")
        assert name_pos < inference_pos < secrets_pos


# -- File dialog helpers ----------------------------------------------------

@pytest.mark.timeout(10)
class TestFileDialogs:
    """File/dir dialog helpers should handle all OS types."""

    @pytest.mark.parametrize(
        ("os_type", "initial_dir", "stdout", "expected_result", "expected_cmd_fragment"),
        [
            ("macos", "/tmp", "/path/to/model.gguf\n", "/path/to/model.gguf", "osascript"),
            ("windows", "C:\\models", "C:\\models\\test.gguf\n", "C:\\models\\test.gguf", "powershell"),
            ("linux", "/home/user", "/home/user/model.gguf\n", "/home/user/model.gguf", "zenity"),
        ],
        ids=["macos", "windows", "linux"],
    )
    @patch("subprocess.run")
    def test_file_dialog_per_os(
        self,
        mock_run: MagicMock,
        os_type: str,
        initial_dir: str,
        stdout: str,
        expected_result: str,
        expected_cmd_fragment: str,
    ) -> None:
        from arqitect.cli.wizard import _open_file_dialog
        mock_run.return_value = MagicMock(returncode=0, stdout=stdout)
        with patch("arqitect.cli.wizard._detect_os", return_value=os_type):
            result = _open_file_dialog("Pick", initial_dir=initial_dir)
        assert result == expected_result
        assert expected_cmd_fragment in mock_run.call_args[0][0]

    @patch("arqitect.cli.wizard._detect_os", return_value="macos")
    @patch("subprocess.run")
    def test_file_dialog_cancelled(self, mock_run: MagicMock, _mock_os: MagicMock) -> None:
        from arqitect.cli.wizard import _open_file_dialog
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = _open_file_dialog("Pick")
        assert result == ""

    @patch("arqitect.cli.wizard._detect_os", return_value="macos")
    @patch("subprocess.run")
    def test_dir_dialog_macos(self, mock_run: MagicMock, _mock_os: MagicMock) -> None:
        from arqitect.cli.wizard import _open_dir_dialog
        mock_run.return_value = MagicMock(returncode=0, stdout="/Users/test/models/\n")
        result = _open_dir_dialog("Pick dir", initial_dir="/Users/test")
        assert result == "/Users/test/models"


# -- Full wizard integration -----------------------------------------------

@pytest.mark.timeout(10)
class TestRunWizard:
    """Full wizard run should produce a valid arqitect.yaml."""

    @patch("arqitect.cli.wizard._open_file_dialog", return_value="/models/brain.gguf")
    @patch("arqitect.cli.wizard.get_project_root")
    def test_interrupt_saves_progress(self, mock_root: MagicMock, _mock_dialog: MagicMock, tmp_path: Path) -> None:
        mock_root.return_value = tmp_path

        call_count = 0
        def interrupt_after_brain(*args: str) -> str:
            nonlocal call_count
            call_count += 1
            # Let environment + brain model steps complete, then interrupt
            # 1=desktop(env), 2=mode(same for all), 3=GGUF(provider)
            if call_count <= 3:
                return ["1", "1", "1"][call_count - 1]
            raise KeyboardInterrupt

        with patch("builtins.input", side_effect=interrupt_after_brain):
            run_wizard()

        yaml_path = tmp_path / "arqitect.yaml"
        assert yaml_path.exists()
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        # Brain model step completed before interrupt
        assert config["inference"]["provider"] == "gguf"

    @patch("arqitect.cli.wizard._open_dir_dialog", return_value="")
    @patch("arqitect.cli.wizard._open_file_dialog", return_value="/models/brain.gguf")
    @patch("arqitect.cli.wizard.get_project_root")
    def test_full_run_anthropic(self, mock_root: MagicMock, _mock_dialog: MagicMock, _mock_dir: MagicMock, tmp_path: Path) -> None:
        mock_root.return_value = tmp_path
        answers = _InputSequence([
            # environment
            "2",               # server
            # brain model
            "1",               # mode: same for all
            "1",               # Anthropic (server option 1)
            "",                # default model
            "sk-test",         # API key (required)
            # vision -- anthropic vision (option 1 when brain is anthropic)
            "1",
            # hearing -- text only
            "3",
            # personality
            "CloudBot", "2", "2",  # friendly, casual
            # connectors
            "n", "n",
            # touch -- uses environment from step 1 (server)
            "2", "", "3",      # sandboxed, default dir, no exec
            # embeddings
            "4",               # openai
            # storage
            "1", "1", "", "", "", "",
            # admin
            "Admin", "a@b.com", "y",
            # smtp
            "", "", "", "", "",
            # github
            "n",
        ])
        with patch("builtins.input", answers):
            run_wizard()

        yaml_path = tmp_path / "arqitect.yaml"
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        assert config == IsPartialDict(
            name="CloudBot",
            inference=IsPartialDict(provider="anthropic"),
            senses=IsPartialDict(sight=IsPartialDict(provider="anthropic")),
        )

    @patch("arqitect.cli.wizard._open_dir_dialog", return_value="")
    @patch("arqitect.cli.wizard._open_file_dialog", return_value="/models/brain.gguf")
    @patch("arqitect.cli.wizard.get_project_root")
    def test_full_run(self, mock_root: MagicMock, _mock_dialog: MagicMock, _mock_dir: MagicMock, tmp_path: Path) -> None:
        mock_root.return_value = tmp_path
        answers = _InputSequence([
            # environment
            "1",               # desktop
            # brain model
            "1",               # mode: same for all
            "1",               # GGUF (desktop option 1)
            # file dialog mocked -- models_dir derived from file
            # vision
            "3",               # skip (last non-anthropic option is index 3)
            # hearing
            "3",               # text only
            # personality
            "TestBot", "1", "2",  # name, professional, casual
            # connectors
            "n", "n",
            # touch -- uses environment from step 1 (desktop)
            "2", "", "2", "",  # sandboxed, default dir, allowlisted, no commands
            # embeddings
            "5",               # none
            # storage
            "1", "1", "", "", "", "",
            # admin
            "Tester", "t@t.com", "y",
            # smtp
            "", "", "", "", "",
            # github
            "n",
        ])
        with patch("builtins.input", answers):
            run_wizard()

        yaml_path = tmp_path / "arqitect.yaml"
        assert yaml_path.exists()
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "TestBot"
        assert config["admin"]["email"] == "t@t.com"
        assert config["secrets"]["jwt_secret"] == IsStr(min_length=20)
