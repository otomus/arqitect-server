"""Sense runtime — convenience wrappers for nerves to invoke senses.

Nerves call these functions to interact with the five core senses
without needing to know the invocation mechanics.
"""

import json
import os
import subprocess
import sys


def _invoke_sense(sense_name: str, args: dict) -> dict:
    """Invoke a sense nerve as a subprocess, returning parsed JSON result."""
    from arqitect.config.loader import get_senses_dir
    sense_path = os.path.join(get_senses_dir(), sense_name, "nerve.py")
    if not os.path.exists(sense_path):
        return {"error": f"Sense '{sense_name}' not found at {sense_path}"}

    env = os.environ.copy()
    try:
        proc = subprocess.run(
            [sys.executable, sense_path, json.dumps(args)],
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=60, env=env,
        )
        output = proc.stdout.strip()
        if not output:
            return {"error": proc.stderr.strip() or "No output from sense"}
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"response": output}
    except subprocess.TimeoutExpired:
        return {"error": f"Sense '{sense_name}' timed out"}
    except Exception as e:
        return {"error": str(e)}


def see(image_path: str = "", prompt: str = "Describe this image") -> dict:
    """Analyze an image file using the sight sense."""
    args = {"prompt": prompt}
    if image_path:
        args["image_path"] = image_path
    return _invoke_sense("sight", args)


def see_screenshot(prompt: str = "Describe what's on the screen") -> dict:
    """Capture a screenshot and analyze it."""
    return _invoke_sense("sight", {"mode": "screenshot", "prompt": prompt})


def hear(audio_path: str = "") -> dict:
    """Speech-to-text from an audio file."""
    return _invoke_sense("hearing", {"mode": "stt", "audio_path": audio_path})


def speak(text: str, voice: str = "default") -> dict:
    """Text-to-speech."""
    return _invoke_sense("hearing", {"mode": "tts", "text": text, "voice": voice})


def touch(command: str = "read", path: str = "", **kwargs) -> dict:
    """File/OS operations via the touch sense."""
    args = {"command": command, "path": path}
    args.update(kwargs)
    return _invoke_sense("touch", args)


def check_awareness(action: str, context: str = "") -> dict:
    """Permission check before destructive operations."""
    return _invoke_sense("awareness", {"action": action, "context": context})


def express(message: str, tone: str = "neutral", fmt: str = "text") -> dict:
    """Format a message with tone and style via the communication sense."""
    return _invoke_sense("communication", {
        "message": message, "tone": tone, "format": fmt,
    })


def call_sense(sense_name: str, args: dict) -> dict:
    """Generic sense invocation by name."""
    return _invoke_sense(sense_name, args)
