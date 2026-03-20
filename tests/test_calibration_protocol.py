"""Tests for arqitect.senses.calibration_protocol."""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest
import time_machine
from dirty_equals import IsInstance, IsPositiveFloat, IsStr
from hypothesis import given, settings
from hypothesis import strategies as st

from arqitect.senses.calibration_protocol import (
    build_result,
    check_binary,
    check_python_module,
    derive_status,
    load_calibration,
    save_calibration,
)

FROZEN_TIME = "2026-03-20T12:00:00Z"
FROZEN_TIMESTAMP = 1774008000.0


@pytest.mark.timeout(10)
class TestCheckBinary:
    """Tests for check_binary — verifies PATH lookup via shutil.which."""

    def test_binary_found(self):
        with patch("arqitect.senses.calibration_protocol.shutil.which", return_value="/usr/bin/ffmpeg"):
            result = check_binary("ffmpeg")
        assert result == {
            "installed": True,
            "path": IsStr(regex=r".+"),
            "install_hint": "",
        }
        assert result["path"] == "/usr/bin/ffmpeg"

    def test_binary_not_found(self):
        with patch("arqitect.senses.calibration_protocol.shutil.which", return_value=None):
            result = check_binary("nonexistent", install_hint="brew install nonexistent")
        assert result == {
            "installed": False,
            "path": "",
            "install_hint": "brew install nonexistent",
        }

    def test_custom_install_hint(self):
        with patch("arqitect.senses.calibration_protocol.shutil.which", return_value=None):
            result = check_binary("sox", install_hint="apt-get install sox")
        assert result["install_hint"] == "apt-get install sox"

    def test_empty_hint_when_found(self):
        with patch("arqitect.senses.calibration_protocol.shutil.which", return_value="/usr/bin/sox"):
            result = check_binary("sox", install_hint="apt-get install sox")
        assert result["install_hint"] == ""


@pytest.mark.timeout(10)
class TestCheckPythonModule:
    """Tests for check_python_module — verifies __import__ probing."""

    def test_module_importable(self):
        result = check_python_module("json")
        assert result == {
            "installed": True,
            "install_hint": "",
        }

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


@pytest.mark.timeout(10)
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

    @given(
        available_count=st.integers(min_value=0, max_value=10),
        unavailable_count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50)
    def test_status_matches_availability_ratio(self, available_count: int, unavailable_count: int):
        """Property: derive_status always returns one of the three valid statuses,
        and the status correctly reflects the ratio of available capabilities."""
        caps = {}
        for i in range(available_count):
            caps[f"available_{i}"] = {"available": True}
        for i in range(unavailable_count):
            caps[f"unavailable_{i}"] = {"available": False}

        status = derive_status(caps)
        assert status in {"operational", "degraded", "unavailable"}

        total = available_count + unavailable_count
        if total == 0:
            assert status == "unavailable"
        elif available_count == total:
            assert status == "operational"
        elif available_count == 0:
            assert status == "unavailable"
        else:
            assert status == "degraded"


@pytest.mark.timeout(10)
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

    @given(
        sense_name=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N"))),
        status=st.sampled_from(["operational", "degraded", "unavailable"]),
    )
    @settings(max_examples=20)
    def test_roundtrip_preserves_arbitrary_data(self, sense_name: str, status: str):
        """Property: any JSON-serializable calibration dict survives a save/load cycle."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data = {"sense": sense_name, "status": status}
            save_calibration(tmp_dir, data)
            loaded = load_calibration(tmp_dir)
            assert loaded == data


@pytest.mark.timeout(10)
class TestBuildResult:
    """Tests for build_result — standardized calibration result builder."""

    def test_minimal_result(self):
        caps = {"mic": {"available": True}}
        deps = {"sox": {"installed": True}}
        result = build_result("hearing", caps, deps)

        assert result == {
            "sense": "hearing",
            "timestamp": IsPositiveFloat,
            "platform": IsStr(min_length=1),
            "status": "operational",
            "capabilities": caps,
            "dependencies": deps,
            "config": {},
            "user_action_needed": [],
            "auto_installable": [],
        }

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

    @time_machine.travel(FROZEN_TIME)
    def test_timestamp_matches_frozen_time(self):
        """Verify that the timestamp in build_result matches the frozen clock."""
        result = build_result("hearing", {}, {})
        assert result["timestamp"] == IsPositiveFloat
        assert result["timestamp"] == pytest.approx(FROZEN_TIMESTAMP, abs=1.0)

    def test_platform_is_string(self):
        result = build_result("hearing", {}, {})
        assert result["platform"] == IsStr(min_length=1)

    @given(
        sense=st.sampled_from(["hearing", "sight", "touch", "awareness", "communication"]),
        score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_build_result_always_has_required_keys(self, sense: str, score: float):
        """Property: build_result always produces a dict with the full set of
        required keys regardless of input values."""
        caps = {"sensor": {"available": score > 0.5}}
        deps = {"lib": {"installed": score > 0.3}}
        result = build_result(sense, caps, deps)

        required_keys = {
            "sense", "timestamp", "platform", "status",
            "capabilities", "dependencies", "config",
            "user_action_needed", "auto_installable",
        }
        assert set(result.keys()) == required_keys
        assert result["sense"] == sense
        assert result["timestamp"] == IsPositiveFloat
        assert result["platform"] == IsStr(min_length=1)
        assert result["status"] in {"operational", "degraded", "unavailable"}
        assert result["config"] == IsInstance(dict)
        assert result["user_action_needed"] == IsInstance(list)
        assert result["auto_installable"] == IsInstance(list)

    @given(
        available_count=st.integers(min_value=0, max_value=5),
        unavailable_count=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=30)
    def test_build_result_status_reflects_capabilities(self, available_count: int, unavailable_count: int):
        """Property: the status field in build_result is consistent with
        derive_status applied to the same capabilities."""
        caps = {}
        for i in range(available_count):
            caps[f"cap_{i}"] = {"available": True}
        for i in range(unavailable_count):
            caps[f"miss_{i}"] = {"available": False}

        result = build_result("test_sense", caps, {})
        assert result["status"] == derive_status(caps)
