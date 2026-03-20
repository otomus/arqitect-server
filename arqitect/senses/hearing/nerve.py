"""Sense: Hearing — audio input (STT via whisper) and output (TTS via system speech).

Modes:
  - stt: {"mode": "stt", "audio_path": "..."} → speech-to-text
  - tts: {"mode": "tts", "text": "...", "voice": "default"} → text-to-speech
  - tts_file: {"mode": "tts_file", "text": "..."} → generate audio file
  - voices: {"mode": "voices"} → list available TTS voices
  - record: {"mode": "record", "duration": 5} → record audio from mic
  - play: {"mode": "play", "audio_path": "..."} → play an audio file

Graceful fallback: returns install instructions if audio tools unavailable.
"""

import json
import os
import platform
import shutil
import subprocess
import sys

_SENSE_DIR = os.path.dirname(os.path.abspath(__file__))
from arqitect.config.loader import get_project_root, get_redis_host_port, get_sandbox_dir as _get_sandbox_dir
from arqitect.types import Sense
_PROJECT_ROOT = str(get_project_root())
_SANDBOX_DIR = _get_sandbox_dir()

SENSE_NAME = Sense.HEARING
def _load_adapter_description() -> str:
    try:
        from arqitect.brain.adapters import get_description
        desc = get_description("hearing")
        if desc:
            return desc
    except Exception:
        pass
    return "Audio input (speech-to-text) and output (text-to-speech)"

DESCRIPTION = _load_adapter_description()


def _check_whisper_available() -> str:
    """Check which whisper implementation is available. Returns 'python', 'cpp', or ''."""
    # Check Python whisper package
    try:
        import whisper  # noqa: F401
        return "python"
    except ImportError:
        pass
    # Check whisper.cpp
    if shutil.which("whisper-cpp") or shutil.which("main"):
        return "cpp"
    return ""


def _check_tts_available() -> str:
    """Check which TTS tool is available. Returns 'say', 'espeak', or ''."""
    system = platform.system()
    if system == "Darwin" and shutil.which("say"):
        return "say"
    if shutil.which("espeak"):
        return "espeak"
    if shutil.which("espeak-ng"):
        return "espeak-ng"
    return ""


def stt(audio_path: str) -> dict:
    """Speech-to-text using whisper."""
    # Resolve path
    resolved = os.path.expanduser(audio_path)
    if not os.path.isabs(resolved):
        resolved = os.path.join(_SANDBOX_DIR, resolved)
    if not os.path.exists(resolved):
        return {"error": f"Audio file not found: {resolved}", "sense": SENSE_NAME}

    whisper_type = _check_whisper_available()

    if whisper_type == "python":
        try:
            import whisper
            model_size = _get_saved_config("whisper_model_size") or "tiny"
            model = whisper.load_model(model_size)
            result = model.transcribe(resolved)
            return {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "audio_path": resolved,
                "engine": "whisper-python",
                "sense": SENSE_NAME,
            }
        except Exception as e:
            return {"error": f"Whisper transcription failed: {e}", "sense": SENSE_NAME}

    elif whisper_type == "cpp":
        try:
            cmd = shutil.which("whisper-cpp") or shutil.which("main")
            result = subprocess.run(
                [cmd, "-m", "models/ggml-tiny.bin", "-f", resolved],
                capture_output=True, text=True, timeout=60,
            )
            text = result.stdout.strip()
            return {
                "text": text,
                "audio_path": resolved,
                "engine": "whisper-cpp",
                "sense": SENSE_NAME,
            }
        except Exception as e:
            return {"error": f"Whisper.cpp failed: {e}", "sense": SENSE_NAME}

    else:
        return {
            "error": "No speech-to-text engine available",
            "install_options": [
                "pip install openai-whisper  (Python, ~150MB for tiny model)",
                "brew install whisper-cpp  (macOS, C++ implementation)",
            ],
            "sense": SENSE_NAME,
        }


def _get_saved_config(key: str) -> str:
    """Read a saved sense config value from Redis or cold memory."""
    try:
        import redis as _redis
        _host, _port = get_redis_host_port()
        _r = _redis.Redis(host=_host, port=_port, decode_responses=True)
        val = _r.hget("synapse:sense_config", f"hearing.{key}")
        if val:
            return val
    except Exception:
        pass
    return ""


def tts(text: str, voice: str = "default", rate: str = "") -> dict:
    """Text-to-speech using system tools."""
    if not text:
        return {"error": "No text provided for TTS", "sense": SENSE_NAME}

    if voice == "default":
        saved = _get_saved_config("preferred_voice")
        if saved:
            voice = saved
    if not rate:
        rate = _get_saved_config("speech_rate")

    tts_tool = _check_tts_available()

    if tts_tool == "say":
        try:
            cmd = ["say"]
            if voice and voice != "default":
                cmd.extend(["-v", voice])
            if rate:
                cmd.extend(["-r", str(rate)])
            cmd.append(text)
            subprocess.run(cmd, check=True, timeout=30)
            return {
                "spoken": True,
                "text": text,
                "voice": voice,
                "engine": "macOS say",
                "sense": SENSE_NAME,
            }
        except Exception as e:
            return {"error": f"TTS failed: {e}", "sense": SENSE_NAME}

    elif tts_tool in ("espeak", "espeak-ng"):
        try:
            cmd = [tts_tool]
            if voice and voice != "default":
                cmd.extend(["-v", voice])
            cmd.append(text)
            subprocess.run(cmd, check=True, timeout=30)
            return {
                "spoken": True,
                "text": text,
                "voice": voice,
                "engine": tts_tool,
                "sense": SENSE_NAME,
            }
        except Exception as e:
            return {"error": f"TTS failed: {e}", "sense": SENSE_NAME}

    else:
        system = platform.system()
        install = "pre-installed on macOS" if system == "Darwin" else "apt install espeak (Linux)"
        return {
            "error": "No text-to-speech engine available",
            "install": install,
            "sense": SENSE_NAME,
        }


def tts_file(text: str, voice: str = "default", rate: str = "") -> dict:
    """Text-to-speech to file — generates audio file for remote playback."""
    if not text:
        return {"error": "No text provided for TTS", "sense": SENSE_NAME}

    if voice == "default":
        saved = _get_saved_config("preferred_voice")
        if saved:
            voice = saved
    if not rate:
        rate = _get_saved_config("speech_rate")

    tts_tool = _check_tts_available()
    os.makedirs(_SANDBOX_DIR, exist_ok=True)
    audio_path = os.path.join(_SANDBOX_DIR, "tts_output.aiff")

    if tts_tool == "say":
        try:
            cmd = ["say", "-o", audio_path]
            if voice and voice != "default":
                cmd.extend(["-v", voice])
            if rate:
                cmd.extend(["-r", str(rate)])
            cmd.append(text)
            subprocess.run(cmd, check=True, timeout=30)
            # Convert AIFF to WAV for browser compatibility
            wav_path = os.path.join(_SANDBOX_DIR, "tts_output.wav")
            if shutil.which("ffmpeg"):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_path, wav_path],
                    capture_output=True, timeout=15,
                )
                if os.path.exists(wav_path):
                    audio_path = wav_path
            return {
                "audio_path": audio_path,
                "text": text,
                "voice": voice,
                "engine": "macOS say",
                "sense": SENSE_NAME,
            }
        except Exception as e:
            return {"error": f"TTS file generation failed: {e}", "sense": SENSE_NAME}

    elif tts_tool in ("espeak", "espeak-ng"):
        audio_path = os.path.join(_SANDBOX_DIR, "tts_output.wav")
        try:
            cmd = [tts_tool, "-w", audio_path]
            if voice and voice != "default":
                cmd.extend(["-v", voice])
            cmd.append(text)
            subprocess.run(cmd, check=True, timeout=30)
            return {
                "audio_path": audio_path,
                "text": text,
                "voice": voice,
                "engine": tts_tool,
                "sense": SENSE_NAME,
            }
        except Exception as e:
            return {"error": f"TTS file generation failed: {e}", "sense": SENSE_NAME}

    else:
        return {
            "error": "No text-to-speech engine available",
            "sense": SENSE_NAME,
        }


def record_audio(duration: int = 5) -> dict:
    """Record audio from the default microphone."""
    os.makedirs(_SANDBOX_DIR, exist_ok=True)
    audio_path = os.path.join(_SANDBOX_DIR, "recording.wav")
    system = platform.system()

    if system == "Darwin" and shutil.which("sox"):
        try:
            subprocess.run(
                ["sox", "-d", "-r", "16000", "-c", "1", audio_path, "trim", "0", str(duration)],
                check=True, timeout=duration + 10,
            )
            return {"audio_path": audio_path, "duration": duration, "engine": "sox", "sense": SENSE_NAME}
        except Exception as e:
            return {"error": f"Recording failed: {e}", "sense": SENSE_NAME}
    elif shutil.which("ffmpeg"):
        try:
            if system == "Darwin":
                input_device = ["-f", "avfoundation", "-i", ":0"]
            else:
                input_device = ["-f", "alsa", "-i", "default"]
            subprocess.run(
                ["ffmpeg", "-y"] + input_device + ["-t", str(duration), "-ar", "16000", "-ac", "1", audio_path],
                check=True, timeout=duration + 10, capture_output=True,
            )
            return {"audio_path": audio_path, "duration": duration, "engine": "ffmpeg", "sense": SENSE_NAME}
        except Exception as e:
            return {"error": f"Recording failed: {e}", "sense": SENSE_NAME}
    elif shutil.which("arecord"):
        try:
            subprocess.run(
                ["arecord", "-d", str(duration), "-f", "cd", "-t", "wav", audio_path],
                check=True, timeout=duration + 10,
            )
            return {"audio_path": audio_path, "duration": duration, "engine": "arecord", "sense": SENSE_NAME}
        except Exception as e:
            return {"error": f"Recording failed: {e}", "sense": SENSE_NAME}
    else:
        return {
            "error": "No recording tool available",
            "install_options": [
                "brew install sox  (macOS)",
                "brew install ffmpeg  (macOS)",
                "apt install alsa-utils  (Linux)",
            ],
            "sense": SENSE_NAME,
        }


def play_audio(audio_path: str) -> dict:
    """Play an audio file through the system speaker."""
    resolved = os.path.expanduser(audio_path)
    if not os.path.isabs(resolved):
        resolved = os.path.join(_SANDBOX_DIR, resolved)
    if not os.path.exists(resolved):
        return {"error": f"Audio file not found: {resolved}", "sense": SENSE_NAME}

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", resolved], check=True, timeout=60)
        elif shutil.which("aplay"):
            subprocess.run(["aplay", resolved], check=True, timeout=60)
        elif shutil.which("ffplay"):
            subprocess.run(["ffplay", "-nodisp", "-autoexit", resolved],
                           check=True, timeout=60, capture_output=True)
        else:
            return {"error": "No audio player available", "sense": SENSE_NAME}
        return {"played": True, "audio_path": resolved, "sense": SENSE_NAME}
    except Exception as e:
        return {"error": f"Playback failed: {e}", "sense": SENSE_NAME}


_NOVELTY_VOICES = frozenset({
    "bad news", "bahh", "bells", "boing", "bubbles", "cellos", "good news",
    "jester", "organ", "superstar", "trinoids", "whisper", "zarvox",
    "deranged", "hysterical", "princess", "wobble", "albert", "fred",
    "junior", "kathy", "ralph", "grandma", "grandpa",
})


def list_voices() -> dict:
    """List available TTS voices with language/locale info.

    Returns voices as list of strings (name only) for backward compat,
    and voices_detailed as list of {name, locale, sample, type} dicts.
    Natural/human voices are sorted first.
    """
    tts_tool = _check_tts_available()
    if tts_tool == "say":
        try:
            result = subprocess.run(
                ["say", "-v", "?"], capture_output=True, text=True, timeout=10,
            )
            all_voices = []
            for line in result.stdout.strip().split("\n"):
                # Format: "Name              locale   # Sample text"
                if not line.strip():
                    continue
                # Split on # to get sample
                parts = line.split("#", 1)
                left = parts[0].strip()
                sample = parts[1].strip() if len(parts) > 1 else ""
                # Split left into name and locale
                tokens = left.rsplit(None, 1)
                if len(tokens) >= 2:
                    name = tokens[0].strip()
                    locale = tokens[1].strip()
                elif tokens:
                    name = tokens[0].strip()
                    locale = ""
                else:
                    continue
                name_lower = name.lower()
                is_novelty = name_lower in _NOVELTY_VOICES or any(name_lower.startswith(n) for n in _NOVELTY_VOICES)
                vtype = "novelty" if is_novelty else "human"
                all_voices.append({"name": name, "locale": locale, "sample": sample, "type": vtype})

            # Sort: human voices first, then novelty; within each group alphabetical
            all_voices.sort(key=lambda v: (0 if v["type"] == "human" else 1, v["name"]))

            voices = [v["name"] for v in all_voices]
            return {
                "voices": voices,
                "voices_detailed": all_voices,
                "engine": "macOS say",
                "sense": SENSE_NAME,
            }
        except Exception:
            pass
    elif tts_tool in ("espeak", "espeak-ng"):
        try:
            result = subprocess.run(
                [tts_tool, "--voices"], capture_output=True, text=True, timeout=10,
            )
            voices = []
            voices_detailed = []
            for line in result.stdout.strip().split("\n")[1:]:
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[3]
                    locale = parts[1] if len(parts) > 1 else ""
                    voices.append(name)
                    voices_detailed.append({"name": name, "locale": locale, "sample": ""})
            return {
                "voices": voices[:30],
                "voices_detailed": voices_detailed[:30],
                "engine": tts_tool,
                "sense": SENSE_NAME,
            }
        except Exception:
            pass
    return {"voices": [], "voices_detailed": [], "error": "No TTS engine available", "sense": SENSE_NAME}


def calibrate() -> dict:
    """Probe hearing sense capabilities."""
    from arqitect.senses.calibration_protocol import check_binary, check_python_module, build_result, save_calibration

    whisper_type = _check_whisper_available()
    tts_tool = _check_tts_available()

    # Check recording tools
    has_sox = bool(shutil.which("sox"))
    has_ffmpeg = bool(shutil.which("ffmpeg"))
    has_arecord = bool(shutil.which("arecord"))
    has_record = has_sox or has_ffmpeg or has_arecord
    record_provider = "sox" if has_sox else ("ffmpeg" if has_ffmpeg else ("arecord" if has_arecord else None))

    # Check playback tools
    system = platform.system()
    has_play = (system == "Darwin") or bool(shutil.which("aplay")) or bool(shutil.which("ffplay"))
    play_provider = "afplay" if system == "Darwin" else ("aplay" if shutil.which("aplay") else ("ffplay" if shutil.which("ffplay") else None))

    capabilities = {
        "stt": {
            "available": bool(whisper_type),
            "provider": f"whisper-{whisper_type}" if whisper_type else None,
            "notes": "" if whisper_type else "Install: pip install openai-whisper OR brew install whisper-cpp",
        },
        "tts": {
            "available": bool(tts_tool),
            "provider": tts_tool or None,
            "notes": "" if tts_tool else ("Pre-installed on macOS" if system == "Darwin" else "Install: apt install espeak"),
        },
        "voices": {
            "available": bool(tts_tool),
            "provider": tts_tool or None,
            "notes": "",
        },
        "record": {
            "available": has_record,
            "provider": record_provider,
            "notes": "" if has_record else "Install: brew install sox (macOS) or apt install alsa-utils (Linux)",
        },
        "play": {
            "available": has_play,
            "provider": play_provider,
            "notes": "" if has_play else "Install: apt install alsa-utils or ffmpeg",
        },
    }

    deps = {}
    # Whisper deps
    whisper_py = check_python_module("whisper", "pip install openai-whisper")
    deps["whisper-python"] = whisper_py
    whisper_cpp = check_binary("whisper-cpp", "brew install whisper-cpp")
    deps["whisper-cpp"] = whisper_cpp
    # TTS deps
    if platform.system() == "Darwin":
        deps["say"] = check_binary("say")
    else:
        deps["espeak"] = check_binary("espeak", "apt install espeak")
        deps["espeak-ng"] = check_binary("espeak-ng", "apt install espeak-ng")

    # Discover voices for user config
    user_actions = []
    if tts_tool:
        voices_result = list_voices()
        voices = voices_result.get("voices", [])
        voices_detailed = voices_result.get("voices_detailed", [])
        if voices:
            user_actions.append({
                "key": "preferred_voice",
                "prompt": "Voice",
                "options": voices,
                "options_detailed": voices_detailed,
                "required_for": ["tts"],
            })

    user_actions.append({
        "key": "speech_rate",
        "prompt": "Speech rate (words per minute)",
        "options": ["120", "150", "175", "200", "230", "260"],
        "required_for": ["tts"],
    })

    user_actions.append({
        "key": "whisper_model_size",
        "prompt": "STT model size (smaller = faster, larger = accurate)",
        "options": ["tiny", "base", "small"],
        "required_for": ["stt"],
    })

    result = build_result(SENSE_NAME, capabilities, deps, user_actions=user_actions)
    save_calibration(_SENSE_DIR, result)
    return result


def main():
    raw = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "{}"
    try:
        input_data = json.loads(raw)
    except json.JSONDecodeError:
        input_data = {"mode": "tts", "text": raw}

    mode = input_data.get("mode", "stt").lower()

    # Calibration mode
    if mode == "calibrate":
        print(json.dumps(calibrate()))
        return

    if mode == "stt":
        audio_path = input_data.get("audio_path", "")
        if not audio_path:
            result = {"error": "No audio_path provided for STT", "sense": SENSE_NAME}
        else:
            result = stt(audio_path)
    elif mode == "tts":
        text = input_data.get("text", "")
        voice = input_data.get("voice", "default")
        result = tts(text, voice)
    elif mode == "tts_file":
        text = input_data.get("text", "")
        voice = input_data.get("voice", "default")
        result = tts_file(text, voice)
    elif mode == "voices":
        result = list_voices()
    elif mode == "record":
        duration = input_data.get("duration", 5)
        result = record_audio(duration)
    elif mode == "play":
        audio_path = input_data.get("audio_path", "")
        if not audio_path:
            result = {"error": "No audio_path provided for playback", "sense": SENSE_NAME}
        else:
            result = play_audio(audio_path)
    else:
        result = {
            "error": f"Unknown mode: {mode}",
            "available_modes": ["stt", "tts", "tts_file", "voices", "record", "play"],
            "sense": SENSE_NAME,
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
