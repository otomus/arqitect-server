"""Tests for arqitect/brain/bootstrap.py — session bootstrap, sense calibration, and user sessions."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


MODULE = "arqitect.brain.bootstrap"


@pytest.fixture()
def mock_mem():
    """Provide a mocked MemoryManager with hot and cold sub-mocks."""
    mem = MagicMock()
    mem.hot.get_session.return_value = {}
    mem.cold.get_facts.return_value = {}
    mem.cold.get_user_facts.return_value = {}
    with patch(f"{MODULE}.mem", mem):
        yield mem


@pytest.fixture()
def mock_redis():
    """Provide a mocked Redis client."""
    r = MagicMock()
    with patch(f"{MODULE}.r", r):
        yield r


@pytest.fixture()
def mock_events():
    """Silence event publishing."""
    with patch(f"{MODULE}.publish_event") as pe, \
         patch(f"{MODULE}.publish_memory_state") as pms:
        yield pe, pms


@pytest.fixture()
def mock_requests():
    """Provide a mocked requests module."""
    with patch(f"{MODULE}.requests") as req:
        yield req


# ── bootstrap_session ────────────────────────────────────────────────────


class TestBootstrapSession:
    """Tests for startup session initialization."""

    def test_already_bootstrapped_returns_early(self, mock_mem, mock_events):
        mock_mem.hot.get_session.return_value = {"city": "Tel Aviv", "timezone": "Asia/Jerusalem"}
        from arqitect.brain.bootstrap import bootstrap_session
        bootstrap_session()
        mock_mem.cold.get_facts.assert_not_called()

    def test_restores_from_cold_memory(self, mock_mem, mock_events):
        mock_mem.hot.get_session.return_value = {}
        cold_facts = {"city": "London", "timezone": "Europe/London"}
        mock_mem.cold.get_facts.return_value = cold_facts
        from arqitect.brain.bootstrap import bootstrap_session
        bootstrap_session()
        mock_mem.hot.set_session.assert_called_once_with(cold_facts)
        _, publish_memory_state = mock_events
        publish_memory_state.assert_called_once()

    def test_ip_geolocation_success(self, mock_mem, mock_events, mock_requests):
        mock_mem.hot.get_session.return_value = {}
        mock_mem.cold.get_facts.return_value = {}
        geo_data = {
            "city": "Berlin",
            "region": "Berlin",
            "country_name": "Germany",
            "country_code": "DE",
            "timezone": "Europe/Berlin",
            "latitude": 52.52,
            "longitude": 13.405,
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = geo_data
        mock_requests.get.return_value = mock_resp
        from arqitect.brain.bootstrap import bootstrap_session
        bootstrap_session()
        mock_mem.hot.set_session.assert_called_once()
        session_arg = mock_mem.hot.set_session.call_args[0][0]
        assert session_arg["city"] == "Berlin"
        assert session_arg["timezone"] == "Europe/Berlin"

    def test_ip_geolocation_failure_does_not_raise(self, mock_mem, mock_events, mock_requests):
        mock_mem.hot.get_session.return_value = {}
        mock_mem.cold.get_facts.return_value = {}
        mock_requests.get.side_effect = Exception("network error")
        from arqitect.brain.bootstrap import bootstrap_session
        bootstrap_session()
        mock_mem.hot.set_session.assert_not_called()

    def test_cold_facts_stored_on_geo_success(self, mock_mem, mock_events, mock_requests):
        mock_mem.hot.get_session.return_value = {}
        mock_mem.cold.get_facts.return_value = {}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"city": "Paris", "timezone": "Europe/Paris"}
        mock_requests.get.return_value = mock_resp
        from arqitect.brain.bootstrap import bootstrap_session
        bootstrap_session()
        assert mock_mem.cold.set_fact.call_count >= 1


# ── calibrate_sense ──────────────────────────────────────────────────────


class TestCalibrateSense:
    """Tests for individual sense calibration."""

    def test_nerve_missing_returns_unavailable(self):
        with patch(f"{MODULE}.os.path.isfile", return_value=False), \
             patch(f"{MODULE}.SENSES_DIR", "/fake/senses"):
            from arqitect.brain.bootstrap import calibrate_sense
            result = calibrate_sense("sight")
        assert result["status"] == "unavailable"
        assert "not found" in result["error"]

    def test_subprocess_success_returns_parsed_json(self):
        cal_result = {"sense": "sight", "status": "available", "capabilities": {}}
        proc = MagicMock()
        proc.stdout = json.dumps(cal_result) + "\n"
        proc.stderr = ""
        with patch(f"{MODULE}.os.path.isfile", return_value=True), \
             patch(f"{MODULE}.subprocess.run", return_value=proc), \
             patch(f"{MODULE}.SENSES_DIR", "/fake/senses"), \
             patch(f"{MODULE}.SANDBOX_DIR", "/fake/sandbox"):
            from arqitect.brain.bootstrap import calibrate_sense
            result = calibrate_sense("sight")
        assert result["status"] == "available"

    def test_subprocess_timeout_returns_timed_out(self):
        with patch(f"{MODULE}.os.path.isfile", return_value=True), \
             patch(f"{MODULE}.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=30)), \
             patch(f"{MODULE}.SENSES_DIR", "/fake/senses"), \
             patch(f"{MODULE}.SANDBOX_DIR", "/fake/sandbox"):
            from arqitect.brain.bootstrap import calibrate_sense
            result = calibrate_sense("sight")
        assert result["status"] == "unavailable"
        assert "timed out" in result["error"]

    def test_subprocess_error_with_module_not_found(self):
        proc = MagicMock()
        proc.stdout = ""
        proc.stderr = "ModuleNotFoundError: No module named 'arqitect'"
        with patch(f"{MODULE}.os.path.isfile", return_value=True), \
             patch(f"{MODULE}.subprocess.run", return_value=proc), \
             patch(f"{MODULE}.SENSES_DIR", "/fake/senses"), \
             patch(f"{MODULE}.SANDBOX_DIR", "/fake/sandbox"):
            from arqitect.brain.bootstrap import calibrate_sense
            result = calibrate_sense("sight")
        assert result["status"] == "unavailable"
        assert "ModuleNotFoundError" in result["error"]

    def test_subprocess_generic_exception(self):
        with patch(f"{MODULE}.os.path.isfile", return_value=True), \
             patch(f"{MODULE}.subprocess.run", side_effect=OSError("disk failure")), \
             patch(f"{MODULE}.SENSES_DIR", "/fake/senses"), \
             patch(f"{MODULE}.SANDBOX_DIR", "/fake/sandbox"):
            from arqitect.brain.bootstrap import calibrate_sense
            result = calibrate_sense("sight")
        assert result["status"] == "unavailable"
        assert "disk failure" in result["error"]


# ── calibrate_all_senses ─────────────────────────────────────────────────


class TestCalibrateAllSenses:
    """Tests for bulk sense calibration."""

    def test_all_senses_available(self):
        cal_result = {"sense": "x", "status": "available", "capabilities": {}}
        proc = MagicMock()
        proc.stdout = json.dumps(cal_result)
        proc.stderr = ""
        senses = ["awareness", "communication", "hearing", "sight", "touch"]
        with patch(f"{MODULE}.os.path.isfile", return_value=True), \
             patch(f"{MODULE}.subprocess.run", return_value=proc), \
             patch(f"{MODULE}.CORE_SENSES", senses), \
             patch(f"{MODULE}.SENSES_DIR", "/fake/senses"), \
             patch(f"{MODULE}.SANDBOX_DIR", "/fake/sandbox"):
            from arqitect.brain.bootstrap import calibrate_all_senses
            results = calibrate_all_senses()
        assert len(results) == 5
        for name in senses:
            assert results[name]["status"] == "available"

    def test_module_not_found_forced_available(self):
        proc = MagicMock()
        proc.stdout = ""
        proc.stderr = "ModuleNotFoundError: No module named 'arqitect'"
        senses = ["sight"]
        with patch(f"{MODULE}.os.path.isfile", return_value=True), \
             patch(f"{MODULE}.subprocess.run", return_value=proc), \
             patch(f"{MODULE}.CORE_SENSES", senses), \
             patch(f"{MODULE}.SENSES_DIR", "/fake/senses"), \
             patch(f"{MODULE}.SANDBOX_DIR", "/fake/sandbox"):
            from arqitect.brain.bootstrap import calibrate_all_senses
            results = calibrate_all_senses()
        assert results["sight"]["status"] == "available"

    def test_non_module_error_stays_unavailable(self):
        proc = MagicMock()
        proc.stdout = ""
        proc.stderr = "SomeOtherError: bad things"
        senses = ["hearing"]
        with patch(f"{MODULE}.os.path.isfile", return_value=True), \
             patch(f"{MODULE}.subprocess.run", return_value=proc), \
             patch(f"{MODULE}.CORE_SENSES", senses), \
             patch(f"{MODULE}.SENSES_DIR", "/fake/senses"), \
             patch(f"{MODULE}.SANDBOX_DIR", "/fake/sandbox"):
            from arqitect.brain.bootstrap import calibrate_all_senses
            results = calibrate_all_senses()
        assert results["hearing"]["status"] == "unavailable"


# ── bootstrap_user_session ───────────────────────────────────────────────


class TestBootstrapUserSession:
    """Tests for per-user session initialization."""

    def test_no_user_id_returns_early(self, mock_mem):
        from arqitect.brain.bootstrap import bootstrap_user_session
        bootstrap_user_session("")
        mock_mem.hot.get_session.assert_not_called()

    def test_already_bootstrapped_returns_early(self, mock_mem):
        mock_mem.hot.get_session.return_value = {"city": "NYC"}
        from arqitect.brain.bootstrap import bootstrap_user_session
        bootstrap_user_session("user-123")
        mock_mem.cold.get_user_facts.assert_not_called()

    def test_restores_from_cold_user_facts(self, mock_mem):
        mock_mem.hot.get_session.return_value = {}
        mock_mem.cold.get_user_facts.return_value = {"city": "Tokyo", "timezone": "Asia/Tokyo"}
        from arqitect.brain.bootstrap import bootstrap_user_session
        bootstrap_user_session("user-456")
        mock_mem.hot.set_session.assert_called_once()
        call_kwargs = mock_mem.hot.set_session.call_args
        assert call_kwargs[0][0]["city"] == "Tokyo"
        assert call_kwargs[1]["user_id"] == "user-456"

    def test_falls_back_to_server_session(self, mock_mem):
        # First call (user session) returns empty, second call (server session) returns data
        mock_mem.hot.get_session.side_effect = [
            {},                                           # user session — empty
            {"city": "London", "timezone": "Europe/London"},  # server session
        ]
        mock_mem.cold.get_user_facts.return_value = {}
        from arqitect.brain.bootstrap import bootstrap_user_session
        bootstrap_user_session("user-789")
        mock_mem.hot.set_session.assert_called_once()
        call_args = mock_mem.hot.set_session.call_args
        assert call_args[0][0]["city"] == "London"

    def test_no_server_session_either(self, mock_mem):
        mock_mem.hot.get_session.return_value = {}
        mock_mem.cold.get_user_facts.return_value = {}
        from arqitect.brain.bootstrap import bootstrap_user_session
        bootstrap_user_session("user-000")
        mock_mem.hot.set_session.assert_not_called()


# ── _store_calibration_in_memory ─────────────────────────────────────────


class TestStoreCalibrationInMemory:
    """Tests for storing calibration results in cold + Redis."""

    def test_stores_in_cold_and_redis(self, mock_mem, mock_redis, mock_events):
        results = {
            "sight": {
                "sense": "sight",
                "status": "available",
                "capabilities": {"camera": {"available": True}},
            },
        }
        from arqitect.brain.bootstrap import _store_calibration_in_memory
        _store_calibration_in_memory(results)
        mock_mem.cold.set_fact.assert_called_once()
        fact_args = mock_mem.cold.set_fact.call_args[0]
        assert fact_args[0] == "sense_calibration"
        assert fact_args[1] == "sight"
        mock_redis.hset.assert_called_once()

    def test_handles_redis_failure_gracefully(self, mock_mem, mock_redis, mock_events):
        mock_redis.hset.side_effect = Exception("connection refused")
        results = {
            "touch": {
                "sense": "touch",
                "status": "available",
                "capabilities": {},
            },
        }
        from arqitect.brain.bootstrap import _store_calibration_in_memory
        _store_calibration_in_memory(results)
        # Cold memory should still be written
        mock_mem.cold.set_fact.assert_called_once()

    def test_publishes_event_for_user_action_needed(self, mock_mem, mock_redis, mock_events):
        publish_event, _ = mock_events
        results = {
            "hearing": {
                "sense": "hearing",
                "status": "available",
                "capabilities": {},
                "user_action_needed": [
                    {"key": "mic", "prompt": "Select microphone", "options": ["mic1", "mic2"]},
                ],
            },
        }
        from arqitect.brain.bootstrap import _store_calibration_in_memory
        _store_calibration_in_memory(results)
        publish_event.assert_called_once()
        event_data = publish_event.call_args[0][1]
        assert len(event_data["user_action_needed"]) == 1
        assert event_data["user_action_needed"][0]["sense"] == "hearing"

    def test_no_event_when_no_actions(self, mock_mem, mock_redis, mock_events):
        publish_event, _ = mock_events
        results = {
            "sight": {"sense": "sight", "status": "available", "capabilities": {}},
        }
        from arqitect.brain.bootstrap import _store_calibration_in_memory
        _store_calibration_in_memory(results)
        publish_event.assert_not_called()

    def test_summary_includes_missing_capabilities(self, mock_mem, mock_redis, mock_events):
        results = {
            "sight": {
                "sense": "sight",
                "status": "available",
                "capabilities": {
                    "camera": {"available": True},
                    "ocr": {"available": False},
                },
            },
        }
        from arqitect.brain.bootstrap import _store_calibration_in_memory
        _store_calibration_in_memory(results)
        summary = mock_mem.cold.set_fact.call_args[0][2]
        assert "camera" in summary
        assert "missing:ocr" in summary
