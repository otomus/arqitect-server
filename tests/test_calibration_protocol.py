"""Tests for arqitect.senses.calibration_protocol."""

import json
from unittest.mock import patch

import pytest

from arqitect.senses.calibration_protocol import (
    build_result,
    check_binary,
    check_python_module,
    derive_status,
    load_calibration,
    save_calibration,
)


class TestCheckBinary:
    """Tests for check_binary — verifies PATH lookup via shutil.which."""

    def test_binary_found(self):
        with patch("arqitect.senses.calibration_protocol.shutil.which", return_value="/usr/bin/ffmpeg"):
            result = check_binary("ffmpeg")
        assert result["installed"] is True
        assert result["path"] == "/usr/bin/ffmpeg"

    def test_binary_not_found(self):
        with patch("arqitect.senses.calibration_protocol.shutil.which", return_value=None):
            result = check_binary("nonexistent", install_hint="brew install nonexistent")
        assert result["installed"] is False
        assert result["path"] == ""
        assert result["install_hint"] == "brew install nonexistent"

    def test_custom_install_hint(self):
        with patch("arqitect.senses.calibration_protocol.shutil.which", return_value=None):
            result = check_binary("sox", install_hint="apt-get install sox")
        assert result["install_hint"] == "apt-get install sox"

    def test_empty_hint_when_found(self):
        with patch("arqitect.senses.calibration_protocol.shutil.which", return_value="/usr/bin/sox"):
            result = check_binary("sox", install_hint="apt-get install sox")
        assert result["install_hint"] == ""


class TestCheckPythonModule:
    """Tests for check_python_module — verifies __import__ probing."""

    def test_module_importable(self):
        result = check_python_module("json")
        assert result["installed"] is True
        assert result["install_hint"] == ""

    def test_module_not_found(self):
        result = check_python_module("nonexistent_module_xyz_12345")
        assert result["installed"] is False

    def test_custom_install_hint(self):
        result = check_python_module(
            "nonexistent_module_xyz_12345",
            install_hint="pip install my-special-package",
        )
        assert result["install_hint"] == "pip install my-special-package"

    def test_default_install_hint(self):
        result = check_python_module("nonexistent_module_xyz_12345")
        assert result["install_hint"] == "pip install nonexistent_module_xyz_12345"


class TestDeriveStatus:
    """Tests for derive_status — maps capability availability to status strings."""

    def test_empty_capabilities_unavailable(self):
        assert derive_status({}) == "unavailable"

    def test_all_available_operational(self):
        caps = {
            "mic": {"available": True},
            "speaker": {"available": True},
        }
        assert derive_status(caps) == "operational"

    def test_some_available_degraded(self):
        caps = {
            "mic": {"available": True},
            "speaker": {"available": False},
        }
        assert derive_status(caps) == "degraded"

    def test_none_available_unavailable(self):
        caps = {
            "mic": {"available": False},
            "speaker": {"available": False},
        }
        assert derive_status(caps) == "unavailable"

    def test_missing_available_key_treated_as_unavailable(self):
        caps = {
            "mic": {"resolution": "high"},
            "speaker": {"resolution": "low"},
        }
        assert derive_status(caps) == "unavailable"


class TestCalibrationPersistence:
    """Tests for save_calibration / load_calibration round-trip via filesystem."""

    def test_save_and_load_roundtrip(self, tmp_path):
        data = {"sense": "hearing", "status": "operational"}
        save_calibration(str(tmp_path), data)
        loaded = load_calibration(str(tmp_path))
        assert loaded == data

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_calibration(str(tmp_path)) is None

    def test_load_corrupted_json_returns_none(self, tmp_path):
        bad_file = tmp_path / "calibration.json"
        bad_file.write_text("{not valid json!!!")
        assert load_calibration(str(tmp_path)) is None


class TestBuildResult:
    """Tests for build_result — standardized calibration result builder."""

    def test_minimal_result(self):
        caps = {"mic": {"available": True}}
        deps = {"sox": {"installed": True}}
        result = build_result("hearing", caps, deps)

        assert result["sense"] == "hearing"
        assert result["status"] == "operational"
        assert result["capabilities"] == caps
        assert result["dependencies"] == deps
        assert result["config"] == {}
        assert result["user_action_needed"] == []
        assert result["auto_installable"] == []

    def test_full_result_with_all_params(self):
        caps = {"camera": {"available": True}}
        deps = {"opencv": {"installed": True}}
        config = {"resolution": "1080p"}
        actions = ["Grant camera permission"]
        auto = ["opencv-python"]

        result = build_result(
            "sight",
            caps,
            deps,
            config=config,
            user_actions=actions,
            auto_installable=auto,
        )

        assert result["config"] == config
        assert result["user_action_needed"] == actions
        assert result["auto_installable"] == auto

    def test_none_optionals_default_to_empty(self):
        result = build_result(
            "touch",
            {"haptic": {"available": False}},
            {},
            config=None,
            user_actions=None,
            auto_installable=None,
        )
        assert result["config"] == {}
        assert result["user_action_needed"] == []
        assert result["auto_installable"] == []

    def test_timestamp_is_numeric(self):
        result = build_result("hearing", {}, {})
        assert isinstance(result["timestamp"], (int, float))

    def test_platform_is_string(self):
        result = build_result("hearing", {}, {})
        assert isinstance(result["platform"], str)
        assert len(result["platform"]) > 0
