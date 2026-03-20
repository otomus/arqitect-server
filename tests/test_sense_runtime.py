"""Tests for arqitect/senses/sense_runtime.py — subprocess invocation and sense wrappers."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

MODULE = "arqitect.senses.sense_runtime"


@pytest.fixture()
def fake_senses_dir(tmp_path):
    """Patch get_senses_dir to return a temp directory."""
    with patch("arqitect.config.loader.get_senses_dir", return_value=str(tmp_path)):
        yield tmp_path


# ── _invoke_sense ────────────────────────────────────────────────────────


class TestInvokeSense:
    """Tests for the core subprocess invocation."""

    def test_path_does_not_exist_returns_error(self, fake_senses_dir):
        from arqitect.senses.sense_runtime import _invoke_sense
        result = _invoke_sense("sight", {"prompt": "describe"})
        assert "error" in result
        assert "not found" in result["error"]

    def test_subprocess_returns_valid_json(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "sight"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        expected = {"response": "A cat", "confidence": 0.95}
        proc = MagicMock()
        proc.stdout = json.dumps(expected) + "\n"
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc):
            from arqitect.senses.sense_runtime import _invoke_sense
            result = _invoke_sense("sight", {"prompt": "describe"})
        assert result == expected

    def test_subprocess_timeout_returns_error(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "hearing"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        with patch(f"{MODULE}.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=60)):
            from arqitect.senses.sense_runtime import _invoke_sense
            result = _invoke_sense("hearing", {})
        assert "error" in result
        assert "timed out" in result["error"]

    def test_subprocess_returns_non_json_falls_back(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "touch"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        proc = MagicMock()
        proc.stdout = "plain text output\n"
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc):
            from arqitect.senses.sense_runtime import _invoke_sense
            result = _invoke_sense("touch", {})
        assert result == {"response": "plain text output"}

    def test_subprocess_empty_stdout_returns_stderr(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "awareness"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        proc = MagicMock()
        proc.stdout = ""
        proc.stderr = "import error details"

        with patch(f"{MODULE}.subprocess.run", return_value=proc):
            from arqitect.senses.sense_runtime import _invoke_sense
            result = _invoke_sense("awareness", {})
        assert result == {"error": "import error details"}

    def test_subprocess_empty_stdout_and_stderr(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "sight"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        proc = MagicMock()
        proc.stdout = "  "
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc):
            from arqitect.senses.sense_runtime import _invoke_sense
            result = _invoke_sense("sight", {})
        assert result == {"error": "No output from sense"}

    def test_generic_exception_returns_error(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "sight"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        with patch(f"{MODULE}.subprocess.run", side_effect=OSError("spawn failed")):
            from arqitect.senses.sense_runtime import _invoke_sense
            result = _invoke_sense("sight", {})
        assert result == {"error": "spawn failed"}


# ── see ──────────────────────────────────────────────────────────────────


class TestSee:
    """Tests for the sight sense wrapper."""

    def test_with_image_path(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "sight"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        proc = MagicMock()
        proc.stdout = json.dumps({"description": "a dog"})
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc) as mock_run:
            from arqitect.senses.sense_runtime import see
            see(image_path="/tmp/photo.jpg", prompt="what is this")

        call_args = mock_run.call_args[0][0]
        passed_json = json.loads(call_args[2])
        assert passed_json["image_path"] == "/tmp/photo.jpg"
        assert passed_json["prompt"] == "what is this"

    def test_without_image_path(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "sight"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        proc = MagicMock()
        proc.stdout = json.dumps({"description": "nothing"})
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc) as mock_run:
            from arqitect.senses.sense_runtime import see
            see()

        call_args = mock_run.call_args[0][0]
        passed_json = json.loads(call_args[2])
        assert "image_path" not in passed_json
        assert passed_json["prompt"] == "Describe this image"


# ── hear ─────────────────────────────────────────────────────────────────


class TestHear:
    """Tests for the hearing sense wrapper."""

    def test_with_audio_path(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "hearing"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        proc = MagicMock()
        proc.stdout = json.dumps({"text": "hello world"})
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc) as mock_run:
            from arqitect.senses.sense_runtime import hear
            hear(audio_path="/tmp/audio.wav")

        call_args = mock_run.call_args[0][0]
        passed_json = json.loads(call_args[2])
        assert passed_json["mode"] == "stt"
        assert passed_json["audio_path"] == "/tmp/audio.wav"


# ── speak ────────────────────────────────────────────────────────────────


class TestSpeak:
    """Tests for the TTS wrapper."""

    def test_text_and_voice_forwarded(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "hearing"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        proc = MagicMock()
        proc.stdout = json.dumps({"audio_path": "/tmp/out.wav"})
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc) as mock_run:
            from arqitect.senses.sense_runtime import speak
            speak("Hello there", voice="alloy")

        call_args = mock_run.call_args[0][0]
        passed_json = json.loads(call_args[2])
        assert passed_json["mode"] == "tts"
        assert passed_json["text"] == "Hello there"
        assert passed_json["voice"] == "alloy"


# ── touch ────────────────────────────────────────────────────────────────


class TestTouch:
    """Tests for the file/OS operations wrapper."""

    def test_command_and_kwargs_merged(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "touch"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        proc = MagicMock()
        proc.stdout = json.dumps({"content": "file data"})
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc) as mock_run:
            from arqitect.senses.sense_runtime import touch
            touch(command="write", path="/tmp/file.txt", data="hello")

        call_args = mock_run.call_args[0][0]
        passed_json = json.loads(call_args[2])
        assert passed_json["command"] == "write"
        assert passed_json["path"] == "/tmp/file.txt"
        assert passed_json["data"] == "hello"


# ── check_awareness ─────────────────────────────────────────────────────


class TestCheckAwareness:
    """Tests for the permission-check wrapper."""

    def test_action_and_context_forwarded(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "awareness"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        proc = MagicMock()
        proc.stdout = json.dumps({"allowed": True})
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc) as mock_run:
            from arqitect.senses.sense_runtime import check_awareness
            check_awareness("delete_file", context="user requested")

        call_args = mock_run.call_args[0][0]
        passed_json = json.loads(call_args[2])
        assert passed_json["action"] == "delete_file"
        assert passed_json["context"] == "user requested"


# ── express ──────────────────────────────────────────────────────────────


class TestExpress:
    """Tests for the communication sense wrapper."""

    def test_message_tone_format_forwarded(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "communication"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        proc = MagicMock()
        proc.stdout = json.dumps({"formatted": "Hi!"})
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc) as mock_run:
            from arqitect.senses.sense_runtime import express
            express("Hi", tone="casual", fmt="markdown")

        call_args = mock_run.call_args[0][0]
        passed_json = json.loads(call_args[2])
        assert passed_json["message"] == "Hi"
        assert passed_json["tone"] == "casual"
        assert passed_json["format"] == "markdown"


# ── call_sense ───────────────────────────────────────────────────────────


class TestCallSense:
    """Tests for the generic sense invocation wrapper."""

    def test_delegates_to_invoke_sense(self, fake_senses_dir):
        sense_dir = fake_senses_dir / "custom_sense"
        sense_dir.mkdir()
        (sense_dir / "nerve.py").write_text("# stub")

        expected = {"result": "ok"}
        proc = MagicMock()
        proc.stdout = json.dumps(expected)
        proc.stderr = ""

        with patch(f"{MODULE}.subprocess.run", return_value=proc):
            from arqitect.senses.sense_runtime import call_sense
            result = call_sense("custom_sense", {"key": "val"})
        assert result == expected

    def test_call_sense_propagates_error(self, fake_senses_dir):
        from arqitect.senses.sense_runtime import call_sense
        result = call_sense("nonexistent", {})
        assert "error" in result
        assert "not found" in result["error"]
