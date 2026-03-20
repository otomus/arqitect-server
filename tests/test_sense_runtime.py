"""Tests for arqitect/senses/sense_runtime.py — subprocess invocation and sense wrappers.

Uses pytest-subprocess (fake_process / fp) instead of MagicMock for subprocess.run,
and hypothesis for property-based edge-case coverage.
"""

import json
import sys
from unittest.mock import patch

import pytest
from dirty_equals import IsPartialDict
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

MODULE = "arqitect.senses.sense_runtime"


@pytest.fixture()
def fake_senses_dir(tmp_path):
    """Patch get_senses_dir to return a temp directory."""
    with patch("arqitect.config.loader.get_senses_dir", return_value=str(tmp_path)):
        yield tmp_path


def _make_sense_dir(base_path, sense_name):
    """Create a sense directory with a stub nerve.py so _invoke_sense finds it.

    Args:
        base_path: The fake senses root directory.
        sense_name: Name of the sense subdirectory.

    Returns:
        Path to the created nerve.py file.
    """
    sense_dir = base_path / sense_name
    sense_dir.mkdir(exist_ok=True)
    nerve_path = sense_dir / "nerve.py"
    nerve_path.write_text("# stub")
    return str(nerve_path)


# -- _invoke_sense -----------------------------------------------------------


@pytest.mark.timeout(10)
class TestInvokeSense:
    """Tests for the core subprocess invocation."""

    def test_path_does_not_exist_returns_error(self, fake_senses_dir):
        """When the nerve.py file does not exist, return an error dict."""
        from arqitect.senses.sense_runtime import _invoke_sense

        result = _invoke_sense("sight", {"prompt": "describe"})
        assert "error" in result
        assert "not found" in result["error"]

    def test_subprocess_returns_valid_json(self, fake_senses_dir, fake_process):
        """Valid JSON stdout from the subprocess is parsed and returned."""
        nerve_path = _make_sense_dir(fake_senses_dir, "sight")
        expected = {"response": "A cat", "confidence": 0.95}

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps(expected) + "\n",
            stderr="",
        )

        from arqitect.senses.sense_runtime import _invoke_sense

        result = _invoke_sense("sight", {"prompt": "describe"})
        assert result == expected

    def test_subprocess_timeout_returns_error(self, fake_senses_dir, fake_process):
        """When the subprocess times out, return an error mentioning 'timed out'."""
        import subprocess

        nerve_path = _make_sense_dir(fake_senses_dir, "hearing")

        def _timeout_callback(process):
            raise subprocess.TimeoutExpired(cmd="nerve.py", timeout=60)

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            callback=_timeout_callback,
        )

        from arqitect.senses.sense_runtime import _invoke_sense

        result = _invoke_sense("hearing", {})
        assert "error" in result
        assert "timed out" in result["error"]

    def test_subprocess_returns_non_json_falls_back(self, fake_senses_dir, fake_process):
        """Non-JSON stdout is wrapped in a response dict."""
        nerve_path = _make_sense_dir(fake_senses_dir, "touch")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout="plain text output\n",
            stderr="",
        )

        from arqitect.senses.sense_runtime import _invoke_sense

        result = _invoke_sense("touch", {})
        assert result == {"response": "plain text output"}

    def test_subprocess_empty_stdout_returns_stderr(self, fake_senses_dir, fake_process):
        """Empty stdout with non-empty stderr returns the stderr as error."""
        nerve_path = _make_sense_dir(fake_senses_dir, "awareness")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout="",
            stderr="import error details",
        )

        from arqitect.senses.sense_runtime import _invoke_sense

        result = _invoke_sense("awareness", {})
        assert result == {"error": "import error details"}

    def test_subprocess_empty_stdout_and_stderr(self, fake_senses_dir, fake_process):
        """Both stdout and stderr empty returns a 'No output' error."""
        nerve_path = _make_sense_dir(fake_senses_dir, "sight")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout="  ",
            stderr="",
        )

        from arqitect.senses.sense_runtime import _invoke_sense

        result = _invoke_sense("sight", {})
        assert result == {"error": "No output from sense"}

    def test_generic_exception_returns_error(self, fake_senses_dir, fake_process):
        """An OSError from subprocess.run is caught and returned as an error dict."""
        nerve_path = _make_sense_dir(fake_senses_dir, "sight")

        def _raise_os_error(process):
            raise OSError("spawn failed")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            callback=_raise_os_error,
        )

        from arqitect.senses.sense_runtime import _invoke_sense

        result = _invoke_sense("sight", {})
        assert result == {"error": "spawn failed"}


# -- Hypothesis property tests -----------------------------------------------


# Strategy for arbitrary JSON-serializable dicts
json_value_strategy = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-10_000, max_value=10_000),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=50),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=10), children, max_size=5),
    ),
    max_leaves=10,
)

json_dict_strategy = st.dictionaries(
    st.text(min_size=1, max_size=20),
    json_value_strategy,
    max_size=5,
)


@pytest.mark.timeout(30)
class TestInvokeSensePropertyBased:
    """Hypothesis property-based tests for _invoke_sense edge cases."""

    @given(payload=json_dict_strategy)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arbitrary_json_payload_never_crashes(self, payload, fake_senses_dir, fake_process):
        """_invoke_sense should always return a dict, never raise, for any JSON payload."""
        nerve_path = _make_sense_dir(fake_senses_dir, "sight")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps({"response": "ok"}),
            stderr="",
        )

        from arqitect.senses.sense_runtime import _invoke_sense

        result = _invoke_sense("sight", payload)
        assert isinstance(result, dict)

    @given(sense_name=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N", "Pc"))))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_random_sense_name_returns_not_found(self, sense_name, fake_senses_dir):
        """Any sense name without a corresponding nerve.py should return 'not found'."""
        from arqitect.senses.sense_runtime import _invoke_sense

        result = _invoke_sense(sense_name, {})
        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"]

    @given(payload=json_dict_strategy)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_arbitrary_payload_echoed_back_through_subprocess(self, payload, fake_senses_dir, fake_process):
        """The payload dict is JSON-serialized and passed as the third CLI arg."""
        nerve_path = _make_sense_dir(fake_senses_dir, "sight")

        # Echo back whatever was received so we can verify round-trip
        echoed = {"echoed": True}
        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps(echoed),
            stderr="",
        )

        from arqitect.senses.sense_runtime import _invoke_sense

        result = _invoke_sense("sight", payload)
        assert result == echoed


# -- see ---------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestSee:
    """Tests for the sight sense wrapper."""

    def test_with_image_path(self, fake_senses_dir, fake_process):
        """see() passes image_path and prompt to the sight nerve."""
        nerve_path = _make_sense_dir(fake_senses_dir, "sight")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps({"description": "a dog"}),
            stderr="",
        )

        from arqitect.senses.sense_runtime import see

        result = see(image_path="/tmp/photo.jpg", prompt="what is this")
        assert result == IsPartialDict({"description": "a dog"})

        # Verify the subprocess received the correct args
        call = fake_process.calls[0]
        passed_json = json.loads(call[2])
        assert passed_json["image_path"] == "/tmp/photo.jpg"
        assert passed_json["prompt"] == "what is this"

    def test_without_image_path(self, fake_senses_dir, fake_process):
        """see() without image_path uses default prompt and omits image_path."""
        nerve_path = _make_sense_dir(fake_senses_dir, "sight")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps({"description": "nothing"}),
            stderr="",
        )

        from arqitect.senses.sense_runtime import see

        see()

        call = fake_process.calls[0]
        passed_json = json.loads(call[2])
        assert "image_path" not in passed_json
        assert passed_json["prompt"] == "Describe this image"


# -- hear --------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestHear:
    """Tests for the hearing sense wrapper."""

    def test_with_audio_path(self, fake_senses_dir, fake_process):
        """hear() passes mode=stt and audio_path to the hearing nerve."""
        nerve_path = _make_sense_dir(fake_senses_dir, "hearing")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps({"text": "hello world"}),
            stderr="",
        )

        from arqitect.senses.sense_runtime import hear

        hear(audio_path="/tmp/audio.wav")

        call = fake_process.calls[0]
        passed_json = json.loads(call[2])
        assert passed_json["mode"] == "stt"
        assert passed_json["audio_path"] == "/tmp/audio.wav"


# -- speak -------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestSpeak:
    """Tests for the TTS wrapper."""

    def test_text_and_voice_forwarded(self, fake_senses_dir, fake_process):
        """speak() passes mode=tts, text, and voice to the hearing nerve."""
        nerve_path = _make_sense_dir(fake_senses_dir, "hearing")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps({"audio_path": "/tmp/out.wav"}),
            stderr="",
        )

        from arqitect.senses.sense_runtime import speak

        speak("Hello there", voice="alloy")

        call = fake_process.calls[0]
        passed_json = json.loads(call[2])
        assert passed_json["mode"] == "tts"
        assert passed_json["text"] == "Hello there"
        assert passed_json["voice"] == "alloy"


# -- touch -------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestTouch:
    """Tests for the file/OS operations wrapper."""

    def test_command_and_kwargs_merged(self, fake_senses_dir, fake_process):
        """touch() merges command, path, and extra kwargs into a single args dict."""
        nerve_path = _make_sense_dir(fake_senses_dir, "touch")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps({"content": "file data"}),
            stderr="",
        )

        from arqitect.senses.sense_runtime import touch

        touch(command="write", path="/tmp/file.txt", data="hello")

        call = fake_process.calls[0]
        passed_json = json.loads(call[2])
        assert passed_json["command"] == "write"
        assert passed_json["path"] == "/tmp/file.txt"
        assert passed_json["data"] == "hello"


# -- check_awareness --------------------------------------------------------


@pytest.mark.timeout(10)
class TestCheckAwareness:
    """Tests for the permission-check wrapper."""

    def test_action_and_context_forwarded(self, fake_senses_dir, fake_process):
        """check_awareness() passes action and context to the awareness nerve."""
        nerve_path = _make_sense_dir(fake_senses_dir, "awareness")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps({"allowed": True}),
            stderr="",
        )

        from arqitect.senses.sense_runtime import check_awareness

        check_awareness("delete_file", context="user requested")

        call = fake_process.calls[0]
        passed_json = json.loads(call[2])
        assert passed_json["action"] == "delete_file"
        assert passed_json["context"] == "user requested"


# -- express -----------------------------------------------------------------


@pytest.mark.timeout(10)
class TestExpress:
    """Tests for the communication sense wrapper."""

    def test_message_tone_format_forwarded(self, fake_senses_dir, fake_process):
        """express() passes message, tone, and format to the communication nerve."""
        nerve_path = _make_sense_dir(fake_senses_dir, "communication")

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps({"formatted": "Hi!"}),
            stderr="",
        )

        from arqitect.senses.sense_runtime import express

        express("Hi", tone="casual", fmt="markdown")

        call = fake_process.calls[0]
        passed_json = json.loads(call[2])
        assert passed_json["message"] == "Hi"
        assert passed_json["tone"] == "casual"
        assert passed_json["format"] == "markdown"


# -- call_sense --------------------------------------------------------------


@pytest.mark.timeout(10)
class TestCallSense:
    """Tests for the generic sense invocation wrapper."""

    def test_delegates_to_invoke_sense(self, fake_senses_dir, fake_process):
        """call_sense() delegates to _invoke_sense and returns its result."""
        nerve_path = _make_sense_dir(fake_senses_dir, "custom_sense")
        expected = {"result": "ok"}

        fake_process.register(
            [sys.executable, nerve_path, fake_process.any()],
            stdout=json.dumps(expected),
            stderr="",
        )

        from arqitect.senses.sense_runtime import call_sense

        result = call_sense("custom_sense", {"key": "val"})
        assert result == expected

    def test_call_sense_propagates_error(self, fake_senses_dir):
        """call_sense() for a nonexistent sense returns a 'not found' error."""
        from arqitect.senses.sense_runtime import call_sense

        result = call_sense("nonexistent", {})
        assert "error" in result
        assert "not found" in result["error"]
