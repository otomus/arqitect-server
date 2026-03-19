"""Tests for the arqitect init wizard.

Mocks all interactive I/O (input + file dialogs) so tests run headlessly.
"""

import copy
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

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
    run_wizard,
    STEPS,
)
from arqitect.config.defaults import DEFAULTS


@pytest.fixture()
def fresh_config() -> dict:
    """Return a deep copy of DEFAULTS for each test."""
    return copy.deepcopy(DEFAULTS)


# ── Helpers ──────────────────────────────────────────────────────────────

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


# ── Step tests ───────────────────────────────────────────────────────────

class TestStepEnvironment:
    """Environment step should set top-level environment field."""

    def test_desktop(self, fresh_config: dict) -> None:
        answers = _InputSequence(["1"])
        with patch("builtins.input", answers):
            _step_environment(fresh_config)
        assert fresh_config["environment"] == "desktop"

    def test_server(self, fresh_config: dict) -> None:
        answers = _InputSequence(["2"])
        with patch("builtins.input", answers):
            _step_environment(fresh_config)
        assert fresh_config["environment"] == "server"

    def test_iot(self, fresh_config: dict) -> None:
        answers = _InputSequence(["3"])
        with patch("builtins.input", answers):
            _step_environment(fresh_config)
        assert fresh_config["environment"] == "iot"

    def test_default_is_server(self, fresh_config: dict) -> None:
        answers = _InputSequence([""])
        with patch("builtins.input", answers):
            _step_environment(fresh_config)
        assert fresh_config["environment"] == "server"


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

        assert fresh_config["inference"]["provider"] == "gguf"
        assert fresh_config["inference"]["models"]["brain"] == "/models/test.gguf"
        assert fresh_config["inference"]["models"]["nerve"] == "/models/test.gguf"
        assert fresh_config["inference"]["models_dir"] == "/models"
        # Per-role config also written
        assert fresh_config["inference"]["roles"]["brain"]["provider"] == "gguf"
        assert fresh_config["inference"]["roles"]["nerve"]["model"] == "/models/test.gguf"

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

        assert fresh_config["inference"]["models"]["brain"] == "/models/brain.gguf"
        assert fresh_config["inference"]["models"]["nerve"] == "/models/nerve.gguf"
        assert fresh_config["inference"]["models"]["coder"] == "/models/coder.gguf"
        assert fresh_config["inference"]["models"]["creative"] == "/models/creative.gguf"
        assert fresh_config["inference"]["models"]["communication"] == "/models/comm.gguf"

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
        assert fresh_config["inference"]["models"]["brain"] == "/custom/models/q.gguf"

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
            "3",               # Anthropic (desktop option 3)
            "",                # default model
            "sk-key",          # API key (required)
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"]["provider"] == "anthropic"

    def test_anthropic_api_key_required(self, fresh_config: dict) -> None:
        """Cloud providers now require an API key — empty key prompts again."""
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

        assert fresh_config["inference"]["provider"] == "openai"
        assert fresh_config["inference"]["models"]["brain"] == "gpt-4o-mini"
        assert fresh_config["secrets"]["openai_base_url"] == "https://my-proxy.com/v1"
        assert fresh_config["secrets"]["openai_api_key"] == "sk-openai-key"

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

        assert fresh_config["inference"]["provider"] == "groq"
        assert fresh_config["inference"]["models"]["brain"] == "llama-3.1-8b"
        assert fresh_config["secrets"]["groq_api_key"] == "gsk_testkey"

    def test_ollama_provider(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "desktop"
        answers = _InputSequence([
            "1",               # mode: same for all
            "2",               # Ollama (desktop option 2)
            "mistral:7b",
            "http://myhost:11434",
        ])
        with patch("builtins.input", answers):
            _step_brain_model(fresh_config)

        assert fresh_config["inference"]["provider"] == "ollama"
        assert fresh_config["inference"]["models"]["brain"] == "mistral:7b"
        assert fresh_config["inference"]["ollama_host"] == "http://myhost:11434"

    def test_server_includes_local_options(self, fresh_config: dict) -> None:
        """Server now shows all 5 providers (cloud first, then local)."""
        fresh_config["environment"] = "server"
        answers = _InputSequence([
            "1",               # mode: same for all
            "3",               # Groq (server option 3)
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
        # Flat backward-compat uses brain's provider
        assert fresh_config["inference"]["provider"] == "anthropic"


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

        assert fresh_config["senses"]["sight"]["enabled"] is True
        assert fresh_config["senses"]["sight"]["provider"] == "moondream"
        assert fresh_config["inference"]["models"]["vision"] == "/models/moondream2.gguf"

    def test_anthropic_vision_option_appears(self, fresh_config: dict) -> None:
        fresh_config["inference"]["provider"] = "anthropic"
        # With anthropic brain: 1=anthropic vision, 2=moondream, 3=custom, 4=skip
        answers = _InputSequence(["1"])
        with patch("builtins.input", answers):
            _step_vision(fresh_config)

        assert fresh_config["senses"]["sight"]["enabled"] is True
        assert fresh_config["senses"]["sight"]["provider"] == "anthropic"

    def test_custom_vision_endpoint(self, fresh_config: dict) -> None:
        answers = _InputSequence([
            "2",               # custom endpoint (non-anthropic)
            "http://vision:8080/predict",
        ])
        with patch("builtins.input", answers):
            _step_vision(fresh_config)

        assert fresh_config["senses"]["sight"]["provider"] == "custom"
        assert fresh_config["senses"]["sight"]["endpoint"] == "http://vision:8080/predict"


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

        assert fresh_config["senses"]["hearing"]["stt"] == "whisper"
        assert fresh_config["senses"]["hearing"]["tts"] == "elevenlabs"

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
        assert fresh_config["personality"]["preset"] == "professional"
        assert fresh_config["personality"]["tone"] == "clear, direct, and helpful"

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
        assert fresh_config["connectors"]["whatsapp"]["enabled"] is True
        assert fresh_config["connectors"]["whatsapp"]["bot_name"] == "WaBot"

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


class TestStepTouch:
    """Touch step should set environment and filesystem config."""

    def test_server_sandboxed(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "server"
        answers = _InputSequence([
            "2",               # sandboxed
            "./data",          # sandbox dir
            "2",               # allowlisted commands
            "python,git",
        ])
        with patch("builtins.input", answers):
            _step_touch(fresh_config)

        assert fresh_config["senses"]["touch"]["environment"] == "server"
        assert fresh_config["senses"]["touch"]["filesystem"]["access"] == "sandboxed"
        assert fresh_config["senses"]["touch"]["filesystem"]["root"] == "./data"
        assert fresh_config["senses"]["touch"]["execution"]["allowlist"] == ["python", "git"]

    def test_desktop_full_access_any_command(self, fresh_config: dict) -> None:
        fresh_config["environment"] = "desktop"
        answers = _InputSequence([
            "1",               # full access
            "1",               # any command
        ])
        with patch("builtins.input", answers):
            _step_touch(fresh_config)

        assert fresh_config["senses"]["touch"]["environment"] == "desktop"
        assert fresh_config["senses"]["touch"]["filesystem"]["access"] == "full"
        assert fresh_config["senses"]["touch"]["execution"]["enabled"] is True

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

        assert fresh_config["senses"]["touch"]["environment"] == "iot"
        assert fresh_config["senses"]["touch"]["execution"]["enabled"] is False


class TestStepEmbeddings:
    """Embeddings step should set provider and model."""

    def test_none_embeddings(self, fresh_config: dict) -> None:
        answers = _InputSequence(["5"])
        with patch("builtins.input", answers):
            _step_embeddings(fresh_config)

        assert fresh_config["embeddings"]["provider"] == "none"

    def test_gguf_embeddings_uses_brain_model(self, fresh_config: dict) -> None:
        fresh_config["inference"]["models"]["brain"] = "my-model.gguf"
        answers = _InputSequence(["3"])
        with patch("builtins.input", answers):
            _step_embeddings(fresh_config)

        assert fresh_config["embeddings"]["model"] == "my-model.gguf"

    def test_nomic_embed(self, fresh_config: dict) -> None:
        answers = _InputSequence(["1"])
        with patch("builtins.input", answers):
            _step_embeddings(fresh_config)

        assert fresh_config["embeddings"]["provider"] == "local"
        assert fresh_config["embeddings"]["model"] == "nomic-embed-text"

    def test_minilm_embed(self, fresh_config: dict) -> None:
        answers = _InputSequence(["2"])
        with patch("builtins.input", answers):
            _step_embeddings(fresh_config)

        assert fresh_config["embeddings"]["model"] == "all-MiniLM-L6-v2"

    def test_openai_embed(self, fresh_config: dict) -> None:
        answers = _InputSequence(["4"])
        with patch("builtins.input", answers):
            _step_embeddings(fresh_config)

        assert fresh_config["embeddings"]["provider"] == "openai"
        assert fresh_config["embeddings"]["model"] == "text-embedding-3-small"


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
        assert fresh_config["ports"]["mcp"] == 9100
        assert fresh_config["ports"]["bridge"] == 4000

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

        assert fresh_config["storage"]["cold"]["type"] == "postgres"
        assert fresh_config["storage"]["cold"]["url"] == "postgresql://u:p@host/db"


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

        assert fresh_config["admin"]["name"] == "Admin"
        assert fresh_config["admin"]["email"] == "admin@test.com"
        assert len(fresh_config["secrets"]["jwt_secret"]) > 20

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

        smtp = fresh_config["secrets"]["smtp"]
        assert smtp["host"] == "smtp.example.com"
        assert smtp["port"] == 465
        assert smtp["user"] == "user@example.com"
        assert smtp["password"] == "s3cret"
        assert smtp["from"] == "noreply@example.com"


# ── Config writer ────────────────────────────────────────────────────────

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


# ── File dialog helpers ──────────────────────────────────────────────────

class TestFileDialogs:
    """File/dir dialog helpers should handle all OS types."""

    @patch("arqitect.cli.wizard._detect_os", return_value="macos")
    @patch("subprocess.run")
    def test_file_dialog_macos(self, mock_run: MagicMock, _mock_os: MagicMock) -> None:
        from arqitect.cli.wizard import _open_file_dialog
        mock_run.return_value = MagicMock(returncode=0, stdout="/path/to/model.gguf\n")
        result = _open_file_dialog("Pick", initial_dir="/tmp")
        assert result == "/path/to/model.gguf"
        assert "osascript" in mock_run.call_args[0][0]

    @patch("arqitect.cli.wizard._detect_os", return_value="windows")
    @patch("subprocess.run")
    def test_file_dialog_windows(self, mock_run: MagicMock, _mock_os: MagicMock) -> None:
        from arqitect.cli.wizard import _open_file_dialog
        mock_run.return_value = MagicMock(returncode=0, stdout="C:\\models\\test.gguf\n")
        result = _open_file_dialog("Pick", initial_dir="C:\\models")
        assert result == "C:\\models\\test.gguf"
        assert "powershell" in mock_run.call_args[0][0]

    @patch("arqitect.cli.wizard._detect_os", return_value="linux")
    @patch("subprocess.run")
    def test_file_dialog_linux(self, mock_run: MagicMock, _mock_os: MagicMock) -> None:
        from arqitect.cli.wizard import _open_file_dialog
        mock_run.return_value = MagicMock(returncode=0, stdout="/home/user/model.gguf\n")
        result = _open_file_dialog("Pick", initial_dir="/home/user")
        assert result == "/home/user/model.gguf"
        assert "zenity" in mock_run.call_args[0][0]

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


# ── Full wizard integration ─────────────────────────────────────────────

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

    @patch("arqitect.cli.wizard._open_file_dialog", return_value="/models/brain.gguf")
    @patch("arqitect.cli.wizard.get_project_root")
    def test_full_run_anthropic(self, mock_root: MagicMock, _mock_dialog: MagicMock, tmp_path: Path) -> None:
        mock_root.return_value = tmp_path
        answers = _InputSequence([
            # environment
            "2",               # server
            # brain model
            "1",               # mode: same for all
            "1",               # Anthropic (server option 1)
            "",                # default model
            "sk-test",         # API key (required)
            # vision — anthropic vision (option 1 when brain is anthropic)
            "1",
            # hearing — text only
            "3",
            # personality
            "CloudBot", "2", "2",  # friendly, casual
            # connectors
            "n", "n",
            # touch — uses environment from step 1 (server)
            "2", "", "3",      # sandboxed, default dir, no exec
            # embeddings
            "4",               # openai
            # storage
            "1", "1", "", "", "", "",
            # admin
            "Admin", "a@b.com", "n",
        ])
        with patch("builtins.input", answers):
            run_wizard()

        yaml_path = tmp_path / "arqitect.yaml"
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        assert config["inference"]["provider"] == "anthropic"
        assert config["senses"]["sight"]["provider"] == "anthropic"
        assert config["name"] == "CloudBot"

    @patch("arqitect.cli.wizard._open_file_dialog", return_value="/models/brain.gguf")
    @patch("arqitect.cli.wizard.get_project_root")
    def test_full_run(self, mock_root: MagicMock, _mock_dialog: MagicMock, tmp_path: Path) -> None:
        mock_root.return_value = tmp_path
        answers = _InputSequence([
            # environment
            "1",               # desktop
            # brain model
            "1",               # mode: same for all
            "1",               # GGUF (desktop option 1)
            # file dialog mocked — models_dir derived from file
            # vision
            "3",               # skip (last non-anthropic option is index 3)
            # hearing
            "3",               # text only
            # personality
            "TestBot", "1", "2",  # name, professional, casual
            # connectors
            "n", "n",
            # touch — uses environment from step 1 (desktop)
            "2", "", "2", "",  # sandboxed, default dir, allowlisted, no commands
            # embeddings
            "5",               # none
            # storage
            "1", "1", "", "", "", "",
            # admin
            "Tester", "t@t.com", "n",
        ])
        with patch("builtins.input", answers):
            run_wizard()

        yaml_path = tmp_path / "arqitect.yaml"
        assert yaml_path.exists()
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "TestBot"
        assert config["admin"]["email"] == "t@t.com"
        assert len(config["secrets"]["jwt_secret"]) > 20
