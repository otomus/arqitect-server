"""Tests for arqitect.inference.config -- readiness checks, status reporting, setup guide."""

from unittest.mock import patch

import pytest
from dirty_equals import IsPositive
from hypothesis import given, settings, strategies as st

from arqitect.inference.config import (
    get_backend_type,
    get_model_name,
    get_models_dir,
    check_gguf_ready,
    print_setup_guide,
    print_status_report,
    REQUIRED_ROLES,
)


# ---------------------------------------------------------------------------
# REQUIRED_ROLES
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestRequiredRoles:
    """Verify REQUIRED_ROLES includes expected entries."""

    def test_contains_vision(self):
        assert "vision" in REQUIRED_ROLES

    def test_contains_inference_roles(self):
        for role in ("brain", "nerve", "coder", "creative", "communication"):
            assert role in REQUIRED_ROLES

    def test_required_roles_is_non_empty_list(self):
        assert len(REQUIRED_ROLES) == IsPositive


# ---------------------------------------------------------------------------
# get_backend_type
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGetBackendType:
    """Tests for get_backend_type."""

    @patch("arqitect.inference.config.get_inference_provider", return_value="gguf")
    def test_returns_provider(self, _mock):
        assert get_backend_type() == "gguf"

    @patch("arqitect.inference.config.get_inference_provider", return_value="openai")
    def test_returns_alternative_provider(self, _mock):
        assert get_backend_type() == "openai"

    @given(provider=st.sampled_from(["gguf", "openai", "anthropic", "groq", "together_ai"]))
    @settings(max_examples=10)
    def test_returns_whatever_provider_is_configured(self, provider):
        """get_backend_type is a pure passthrough to the config loader."""
        with patch("arqitect.inference.config.get_inference_provider", return_value=provider):
            assert get_backend_type() == provider


# ---------------------------------------------------------------------------
# get_model_name
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGetModelName:
    """Tests for get_model_name."""

    @patch("arqitect.inference.config.get_model_for_role", return_value="brain.gguf")
    def test_returns_configured_model(self, _mock):
        assert get_model_name("brain") == "brain.gguf"

    @patch("arqitect.inference.config.get_model_for_role", return_value=None)
    def test_falls_back_to_role_name(self, _mock):
        assert get_model_name("vision") == "vision"

    @patch("arqitect.inference.config.get_model_for_role", return_value="")
    def test_empty_string_falls_back(self, _mock):
        """Empty string is falsy, so falls back to role name."""
        assert get_model_name("nerve") == "nerve"

    @given(role=st.from_regex(r"[a-z_]{3,15}", fullmatch=True))
    @settings(max_examples=15)
    def test_fallback_always_returns_role_name(self, role):
        """When config returns nothing, the role name itself is returned."""
        with patch("arqitect.inference.config.get_model_for_role", return_value=None):
            assert get_model_name(role) == role


# ---------------------------------------------------------------------------
# get_models_dir
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGetModelsDir:
    """Tests for get_models_dir."""

    @patch("arqitect.inference.config._loader_get_models_dir", return_value="/data/models")
    def test_delegates_to_loader(self, _mock):
        assert get_models_dir() == "/data/models"


# ---------------------------------------------------------------------------
# check_gguf_ready
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestCheckGgufReady:
    """Tests for the GGUF readiness checker."""

    @patch("arqitect.inference.config.MODEL_REGISTRY", {})
    @patch("arqitect.inference.config.get_model_name")
    @patch("arqitect.inference.config.get_models_dir")
    def test_all_present(self, mock_dir, mock_name, tmp_path):
        mock_dir.return_value = str(tmp_path)
        for role in REQUIRED_ROLES:
            mock_name.side_effect = lambda r: f"{r}.gguf"
            (tmp_path / f"{role}.gguf").touch()

        mock_name.side_effect = lambda r: f"{r}.gguf"
        ready, missing = check_gguf_ready()
        assert ready is True
        assert missing == []

    @patch("arqitect.inference.config.MODEL_REGISTRY", {})
    @patch("arqitect.inference.config.get_model_name", return_value="nonexistent.gguf")
    @patch("arqitect.inference.config.get_models_dir", return_value="/tmp/empty")
    def test_all_missing(self, _dir, _name):
        ready, missing = check_gguf_ready()
        assert ready is False
        assert len(missing) == len(REQUIRED_ROLES)

    @patch("arqitect.inference.config.MODEL_REGISTRY", {"brain": {"source": "hf/brain"}})
    @patch("arqitect.inference.config.get_model_name")
    @patch("arqitect.inference.config.get_models_dir")
    def test_partial_missing_includes_source(self, mock_dir, mock_name, tmp_path):
        mock_dir.return_value = str(tmp_path)
        (tmp_path / "brain.gguf").touch()
        mock_name.side_effect = lambda r: f"{r}.gguf"

        ready, missing = check_gguf_ready()
        assert ready is False
        assert len(missing) == len(REQUIRED_ROLES) - 1


# ---------------------------------------------------------------------------
# print_setup_guide
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestPrintSetupGuide:
    """Tests for the setup guide output."""

    def test_prints_to_stderr(self, capsys):
        print_setup_guide()
        captured = capsys.readouterr()
        assert "ARQITECT" in captured.err
        assert "First Run Setup" in captured.err
        assert "arqitect.yaml" in captured.err

    def test_includes_install_instructions(self, capsys):
        print_setup_guide()
        captured = capsys.readouterr()
        assert "pip install" in captured.err
        assert "setup" in captured.err


# ---------------------------------------------------------------------------
# print_status_report
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestPrintStatusReport:
    """Tests for the status report printer."""

    @patch("arqitect.inference.config.get_models_dir", return_value="/m")
    def test_ready_report(self, _gmd, capsys):
        print_status_report("gguf", True, [])
        captured = capsys.readouterr()
        assert "all models ready" in captured.err
        assert "gguf" in captured.err

    @patch("arqitect.inference.config.get_models_dir", return_value="/m")
    def test_not_ready_report_lists_missing(self, _gmd, capsys):
        missing = ["vision (v.gguf) from hf/vision"]
        print_status_report("gguf", False, missing)
        captured = capsys.readouterr()
        assert "missing models detected" in captured.err
        assert "vision" in captured.err

    @patch("arqitect.inference.config.get_models_dir", return_value="/data/models")
    def test_not_ready_shows_fix_instructions(self, _gmd, capsys):
        print_status_report("gguf", False, ["x"])
        captured = capsys.readouterr()
        assert "/data/models" in captured.err
        assert "setup" in captured.err
