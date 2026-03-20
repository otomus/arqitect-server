"""Sense: Sight — vision and image understanding via moondream model.

Modes:
  - image (default): {"image_path": "..."} or {"base64": "..."} → analyze with moondream
  - stream: {"mode": "stream", "source": "camera|url", "interval": 5} → continuous frame analysis
  - screenshot: {"mode": "screenshot"} → capture screen + analyze
  - stop_stream: {"mode": "stop_stream"} → stop background stream

Image generation is handled by the image_generator MCP tool, not this sense.
Graceful fallback: if moondream not installed, returns install instructions.
"""

import json
import os
import platform
import subprocess
import sys

_SENSE_DIR = os.path.dirname(os.path.abspath(__file__))
from arqitect.config.loader import get_project_root, get_redis_host_port, get_sandbox_dir as _get_sandbox_dir
from arqitect.types import Sense
_PROJECT_ROOT = str(get_project_root())
_SANDBOX_DIR = _get_sandbox_dir()

SENSE_NAME = Sense.SIGHT
def _load_adapter_description() -> str:
    try:
        from arqitect.brain.adapters import get_description
        desc = get_description("vision")
        if desc:
            return desc
    except Exception:
        pass
    return "Image analysis and description — examines photos, screenshots, and visual content."

DESCRIPTION = _load_adapter_description()
VISION_MODEL = "vision"


def _check_model_available() -> bool:
    """Check if the vision model files exist (without loading the full engine)."""
    try:
        from arqitect.inference.config import get_model_name, get_models_dir
        from arqitect.inference.model_registry import MODEL_REGISTRY
        models_dir = get_models_dir()
        model_file = get_model_name(VISION_MODEL)
        mmproj = MODEL_REGISTRY.get(VISION_MODEL, {}).get("mmproj", "")
        if not mmproj:
            return os.path.exists(os.path.join(models_dir, model_file))
        return (os.path.exists(os.path.join(models_dir, model_file))
                and os.path.exists(os.path.join(models_dir, mmproj)))
    except Exception:
        return False


def _model_not_available_msg() -> dict:
    """Return helpful message when vision model is not loaded."""
    return {
        "error": f"Vision model '{VISION_MODEL}' not loaded",
        "note": "The moondream GGUF model needs to be downloaded. Restart the system to auto-download.",
        "sense": SENSE_NAME,
    }


def _get_vision_prompt() -> str:
    """Load vision prompt from community adapter, fallback to hardcoded default."""
    try:
        from arqitect.brain.adapters import resolve_prompt
        adapter = resolve_prompt("vision")
        if adapter and adapter.get("system_prompt"):
            return adapter["system_prompt"]
    except Exception:
        pass
    return "Describe this image in detail."


def _analyze_image(image_path: str = "", base64_data: str = "", prompt: str = "") -> dict:
    """Analyze an image using the vision model via in-process inference."""
    if not prompt:
        prompt = _get_vision_prompt()
    if not _check_model_available():
        return _model_not_available_msg()

    # Resolve image path
    resolved_path = ""
    if image_path:
        resolved_path = os.path.expanduser(image_path)
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(_SANDBOX_DIR, resolved_path)
        if not os.path.exists(resolved_path):
            return {"error": f"Image not found: {resolved_path}", "sense": SENSE_NAME}

    try:
        from arqitect.inference.engine import get_engine
        description = get_engine().generate_vision(
            image_path=resolved_path,
            base64_data=base64_data,
            prompt=prompt,
        )
        if description.startswith("Error:"):
            return {"error": description, "sense": SENSE_NAME}
        return {
            "response": description,
            "image_path": image_path or "(base64 input)",
            "model": VISION_MODEL,
            "sense": SENSE_NAME,
        }
    except Exception as e:
        return {"error": f"Vision analysis failed: {e}", "sense": SENSE_NAME}


def _capture_screenshot() -> str:
    """Capture a screenshot and return the file path."""
    os.makedirs(_SANDBOX_DIR, exist_ok=True)
    screenshot_path = os.path.join(_SANDBOX_DIR, "screenshot.png")
    system = platform.system()

    try:
        if system == "Darwin":
            subprocess.run(["screencapture", "-x", screenshot_path], check=True, timeout=10)
        elif system == "Linux":
            # Try scrot, then gnome-screenshot, then import (ImageMagick)
            for cmd in [
                ["scrot", screenshot_path],
                ["gnome-screenshot", "-f", screenshot_path],
                ["import", "-window", "root", screenshot_path],
            ]:
                try:
                    subprocess.run(cmd, check=True, timeout=10)
                    break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            else:
                return ""
        else:
            return ""

        return screenshot_path if os.path.exists(screenshot_path) else ""
    except Exception:
        return ""


def _start_stream(source: str, interval: int, url: str = "") -> dict:
    """Start background video stream analysis."""
    if not _check_model_available():
        return _model_not_available_msg()

    # Build the stream worker script path
    stream_script = os.path.join(_SENSE_DIR, "_stream_worker.py")

    # Create a simple stream worker if it doesn't exist
    if not os.path.exists(stream_script):
        worker_code = '''"""Background stream worker for sight sense — uses in-process inference."""
import json, os, sys, time, base64, subprocess, tempfile

# Add project root to path so inference module is importable

SANDBOX_DIR = os.environ.get("SANDBOX_DIR", "/tmp/arqitect_sandbox")

def capture_frame(source, url=""):
    """Capture a single frame from camera or URL."""
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    frame_path = os.path.join(SANDBOX_DIR, "stream_frame.jpg")
    try:
        if source == "camera":
            import platform, shutil
            if platform.system() == "Darwin" and shutil.which("imagesnap"):
                subprocess.run(["imagesnap", "-q", frame_path], check=True, timeout=10)
            elif platform.system() == "Darwin" and shutil.which("ffmpeg"):
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "avfoundation", "-framerate", "1",
                     "-i", "0", "-frames:v", "1", frame_path],
                    check=True, timeout=10, capture_output=True,
                )
            else:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "v4l2", "-i", "/dev/video0",
                     "-frames:v", "1", frame_path],
                    check=True, timeout=10, capture_output=True,
                )
        elif source == "url" and url:
            subprocess.run(
                ["ffmpeg", "-y", "-i", url, "-frames:v", "1", frame_path],
                check=True, timeout=15, capture_output=True,
            )
        return frame_path if os.path.exists(frame_path) else None
    except Exception:
        return None

def analyze_frame(frame_path):
    """Analyze a frame with the vision model via in-process inference."""
    from arqitect.inference.engine import get_engine
    return get_engine().generate_vision(image_path=frame_path, prompt="Describe what you see.")

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "camera"
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    url = sys.argv[3] if len(sys.argv) > 3 else ""

    import redis
    _host, _port = get_redis_host_port()
    r = redis.Redis(host=_host, port=_port, decode_responses=True)

    while True:
        frame = capture_frame(source, url)
        if frame:
            try:
                desc = analyze_frame(frame)
                r.publish("sense:sight:stream", json.dumps({
                    "observation": desc, "source": source, "timestamp": time.time(),
                }))
            except Exception as e:
                r.publish("sense:sight:stream", json.dumps({"error": str(e)}))
        time.sleep(interval)

if __name__ == "__main__":
    main()
'''
        with open(stream_script, "w") as f:
            f.write(worker_code)

    # Launch as background process
    env = os.environ.copy()
    env["SANDBOX_DIR"] = _SANDBOX_DIR
    try:
        proc = subprocess.Popen(
            [sys.executable, stream_script, source, str(interval), url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        # Save PID for stop_stream
        pid_file = os.path.join(_SENSE_DIR, ".stream_pid")
        with open(pid_file, "w") as f:
            f.write(str(proc.pid))
        return {
            "streaming": True,
            "pid": proc.pid,
            "source": source,
            "interval": interval,
            "channel": "sense:sight:stream",
            "sense": SENSE_NAME,
        }
    except Exception as e:
        return {"error": f"Failed to start stream: {e}", "sense": SENSE_NAME}


def _stop_stream() -> dict:
    """Stop a background video stream."""
    pid_file = os.path.join(_SENSE_DIR, ".stream_pid")
    if not os.path.exists(pid_file):
        return {"error": "No active stream found", "sense": SENSE_NAME}
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 9)
        os.remove(pid_file)
        return {"stopped": True, "pid": pid, "sense": SENSE_NAME}
    except ProcessLookupError:
        if os.path.exists(pid_file):
            os.remove(pid_file)
        return {"stopped": True, "note": "Process already ended", "sense": SENSE_NAME}
    except Exception as e:
        return {"error": f"Failed to stop stream: {e}", "sense": SENSE_NAME}


def _discover_cameras() -> list[dict]:
    """Discover available camera devices."""
    cameras = []
    system = platform.system()
    try:
        if system == "Darwin":
            # Use system_profiler on macOS
            result = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                import json as _json
                data = _json.loads(result.stdout)
                for cam in data.get("SPCameraDataType", []):
                    cameras.append({
                        "id": cam.get("_name", "unknown"),
                        "name": cam.get("_name", "unknown"),
                    })
        else:
            # Linux: check /dev/video* devices
            import glob
            for dev in sorted(glob.glob("/dev/video*")):
                cameras.append({"id": dev, "name": dev})
    except Exception:
        pass
    return cameras


def calibrate() -> dict:
    """Probe sight sense capabilities."""
    from arqitect.senses.calibration_protocol import (
        check_binary, build_result, save_calibration,
    )

    system = platform.system()
    # Check model file existence (avoids loading full engine in calibration subprocess)
    has_moondream = _check_model_available()
    moondream = {"installed": has_moondream, "version": "latest" if has_moondream else "", "install_hint": ""}

    # Screenshot tool
    if system == "Darwin":
        screenshot_tool = check_binary("screencapture")
    else:
        scrot = check_binary("scrot", "apt install scrot")
        gnome_ss = check_binary("gnome-screenshot", "apt install gnome-screenshot")
        screenshot_tool = scrot if scrot["installed"] else gnome_ss

    # Camera tools
    imagesnap = check_binary("imagesnap", "brew install imagesnap")
    ffmpeg = check_binary("ffmpeg", "brew install ffmpeg" if system == "Darwin" else "apt install ffmpeg")
    has_camera_tool = imagesnap["installed"] or ffmpeg["installed"]

    cameras = _discover_cameras()

    capabilities = {
        "image_analysis": {
            "available": has_moondream,
            "provider": VISION_MODEL if has_moondream else None,
            "notes": "" if has_moondream else "Vision GGUF model not downloaded. Restart to auto-download.",
        },
        "screenshot": {
            "available": screenshot_tool["installed"],
            "provider": screenshot_tool.get("path", "").split("/")[-1] if screenshot_tool["installed"] else None,
            "notes": "" if screenshot_tool["installed"] else screenshot_tool.get("install_hint", ""),
        },
        "camera_capture": {
            "available": has_camera_tool and len(cameras) > 0,
            "provider": "imagesnap" if imagesnap["installed"] else ("ffmpeg" if ffmpeg["installed"] else None),
            "notes": "" if has_camera_tool else "Install: brew install imagesnap",
        },
        "video_stream": {
            "available": has_moondream and has_camera_tool,
            "provider": VISION_MODEL if has_moondream and has_camera_tool else None,
            "notes": "",
        },
    }

    deps = {
        VISION_MODEL: moondream,
        "ffmpeg": ffmpeg,
    }
    if system == "Darwin":
        deps["imagesnap"] = imagesnap
        deps["screencapture"] = check_binary("screencapture")
    else:
        deps["scrot"] = check_binary("scrot", "apt install scrot")

    # User config for camera device
    user_actions = []
    if cameras:
        user_actions.append({
            "key": "camera_device",
            "prompt": "Which camera device should I use?",
            "options": [f"{c['id']}: {c['name']}" for c in cameras],
            "required_for": ["camera_capture", "video_stream"],
        })

    config = {"camera_device": None}

    result = build_result(
        SENSE_NAME, capabilities, deps, config=config,
        user_actions=user_actions,
    )
    save_calibration(_SENSE_DIR, result)
    return result


def main():
    raw = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "{}"
    try:
        input_data = json.loads(raw)
    except json.JSONDecodeError:
        input_data = {"image_path": raw}

    mode = input_data.get("mode", "image").lower()

    # Calibration mode
    if mode == "calibrate":
        print(json.dumps(calibrate()))
        return

    if mode == "screenshot":
        path = _capture_screenshot()
        if not path:
            tools = "screencapture (macOS)" if platform.system() == "Darwin" else "scrot or gnome-screenshot (Linux)"
            result = {
                "error": f"Screenshot capture failed. Install: {tools}",
                "sense": SENSE_NAME,
            }
        else:
            result = _analyze_image(image_path=path)
            result["mode"] = "screenshot"
    elif mode == "stream":
        result = _start_stream(
            input_data.get("source", "camera"),
            input_data.get("interval", 5),
            input_data.get("url", ""),
        )
    elif mode == "stop_stream":
        result = _stop_stream()
    else:
        # Default: image analysis
        result = _analyze_image(
            image_path=input_data.get("image_path", ""),
            base64_data=input_data.get("base64", ""),
            prompt=input_data.get("prompt", "Describe this image in detail."),
        )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
