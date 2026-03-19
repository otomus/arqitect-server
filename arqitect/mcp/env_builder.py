"""Environment Builder — creates and maintains per-tool isolated dependency environments.

Handles Python venvs, Node node_modules, Go/Rust compilation, and Docker images.
Each tool directory in mcp_tools/ gets its own environment based on the runtime
declared in tool.json.

Environments are built during community seeding or dream state, never during active chat
(except as a fallback for first-call-before-build).
"""

import json
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

# Supported runtime values in tool.json
SUPPORTED_RUNTIMES = frozenset({"python", "node", "go", "rust", "binary", "docker"})


def _read_tool_json(tool_dir: str) -> dict | None:
    """Read and parse tool.json from a tool directory.

    Args:
        tool_dir: Path to the tool directory.

    Returns:
        Parsed tool.json dict, or None if missing/invalid.
    """
    path = os.path.join(tool_dir, "tool.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def env_ready(tool_dir: str) -> bool:
    """Check if a tool's environment is built and matches its tool.json version.

    Args:
        tool_dir: Path to the tool directory.

    Returns:
        True if the environment exists and the version matches.
    """
    manifest = _read_tool_json(tool_dir)
    if not manifest:
        return False

    runtime = manifest.get("runtime", "python")
    version = manifest.get("version", "0.0.0")
    version_file = os.path.join(tool_dir, ".env_version")

    if not os.path.isfile(version_file):
        return False

    try:
        with open(version_file) as f:
            installed_version = f.read().strip()
    except OSError:
        return False

    if installed_version != version:
        return False

    if runtime == "python":
        return os.path.isdir(os.path.join(tool_dir, ".venv"))
    if runtime == "node":
        return os.path.isdir(os.path.join(tool_dir, "node_modules"))
    if runtime in ("go", "rust"):
        return os.path.isfile(os.path.join(tool_dir, "run"))
    if runtime == "binary":
        return os.path.isfile(os.path.join(tool_dir, "run"))
    if runtime == "docker":
        name = manifest.get("name", os.path.basename(tool_dir))
        result = subprocess.run(
            ["docker", "image", "inspect", f"arqitect-tool-{name}"],
            capture_output=True,
        )
        return result.returncode == 0

    return False


def build_env(tool_dir: str) -> bool:
    """Build the dependency environment for a tool.

    Reads tool.json to determine the runtime, then creates the appropriate
    environment (venv, node_modules, compiled binary, or Docker image).

    Args:
        tool_dir: Path to the tool directory.

    Returns:
        True if the environment was built successfully.
    """
    manifest = _read_tool_json(tool_dir)
    if not manifest:
        logger.warning("[ENV_BUILDER] No tool.json in %s", tool_dir)
        return False

    runtime = manifest.get("runtime", "python")
    name = manifest.get("name", os.path.basename(tool_dir))

    logger.info("[ENV_BUILDER] Building %s environment for %s", runtime, name)

    builders = {
        "python": _build_python,
        "node": _build_node,
        "go": _build_go,
        "rust": _build_rust,
        "binary": _build_binary,
        "docker": _build_docker,
    }

    builder = builders.get(runtime)
    if not builder:
        logger.error("[ENV_BUILDER] Unsupported runtime '%s' for %s", runtime, name)
        return False

    try:
        success = builder(tool_dir, manifest)
    except Exception as e:
        logger.error("[ENV_BUILDER] Failed to build %s for %s: %s", runtime, name, e)
        return False

    if success:
        _write_version(tool_dir, manifest.get("version", "0.0.0"))
        logger.info("[ENV_BUILDER] Built %s environment for %s", runtime, name)

    return success


def rebuild_env(tool_dir: str) -> bool:
    """Tear down and rebuild a tool's environment.

    Used when a tool's dependencies change (version bump detected).

    Args:
        tool_dir: Path to the tool directory.

    Returns:
        True if the environment was rebuilt successfully.
    """
    cleanup_env(tool_dir)
    return build_env(tool_dir)


def cleanup_env(tool_dir: str) -> None:
    """Remove a tool's environment artifacts.

    Removes venvs, node_modules, compiled binaries, Docker images, and
    the version tracking file.

    Args:
        tool_dir: Path to the tool directory.
    """
    manifest = _read_tool_json(tool_dir)

    # Remove common environment artifacts
    for subdir in (".venv", "node_modules", "target"):
        path = os.path.join(tool_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

    # Remove compiled binary (but not if it's the source entry point for binary runtime)
    run_path = os.path.join(tool_dir, "run")
    if manifest and manifest.get("runtime") in ("go", "rust") and os.path.isfile(run_path):
        os.remove(run_path)

    # Remove Docker image
    if manifest and manifest.get("runtime") == "docker":
        name = manifest.get("name", os.path.basename(tool_dir))
        subprocess.run(
            ["docker", "rmi", f"arqitect-tool-{name}"],
            capture_output=True,
        )

    # Remove version tracking file
    version_file = os.path.join(tool_dir, ".env_version")
    if os.path.isfile(version_file):
        os.remove(version_file)


def _write_version(tool_dir: str, version: str) -> None:
    """Write the installed version to a tracking file.

    Args:
        tool_dir: Path to the tool directory.
        version: Version string to write.
    """
    with open(os.path.join(tool_dir, ".env_version"), "w") as f:
        f.write(version)


def _build_python(tool_dir: str, manifest: dict) -> bool:
    """Create a Python virtualenv and install requirements.

    Args:
        tool_dir: Path to the tool directory.
        manifest: Parsed tool.json.

    Returns:
        True if the environment was created successfully.
    """
    venv_dir = os.path.join(tool_dir, ".venv")
    result = subprocess.run(
        [sys.executable, "-m", "venv", venv_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("[ENV_BUILDER] venv creation failed: %s", result.stderr)
        return False

    # Install requirements if present
    req_file = os.path.join(tool_dir, "requirements.txt")
    if os.path.isfile(req_file):
        pip = os.path.join(venv_dir, "bin", "pip")
        result = subprocess.run(
            [pip, "install", "-r", req_file, "--quiet"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.error("[ENV_BUILDER] pip install failed: %s", result.stderr)
            return False

    return True


def _build_node(tool_dir: str, manifest: dict) -> bool:
    """Install Node.js dependencies via npm.

    Args:
        tool_dir: Path to the tool directory.
        manifest: Parsed tool.json.

    Returns:
        True if npm install succeeded.
    """
    if not os.path.isfile(os.path.join(tool_dir, "package.json")):
        return True  # No deps to install

    result = subprocess.run(
        ["npm", "install", "--prefix", tool_dir, "--quiet"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        logger.error("[ENV_BUILDER] npm install failed: %s", result.stderr)
        return False
    return True


def _build_go(tool_dir: str, manifest: dict) -> bool:
    """Compile a Go tool to a static binary.

    Args:
        tool_dir: Path to the tool directory.
        manifest: Parsed tool.json.

    Returns:
        True if go build succeeded.
    """
    result = subprocess.run(
        ["go", "build", "-o", "run", "."],
        cwd=tool_dir, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        logger.error("[ENV_BUILDER] go build failed: %s", result.stderr)
        return False
    return True


def _build_rust(tool_dir: str, manifest: dict) -> bool:
    """Compile a Rust tool via cargo.

    Args:
        tool_dir: Path to the tool directory.
        manifest: Parsed tool.json.

    Returns:
        True if cargo build succeeded.
    """
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=tool_dir, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        logger.error("[ENV_BUILDER] cargo build failed: %s", result.stderr)
        return False

    # Copy release binary to ./run for consistent interface
    binary_name = manifest.get("name", os.path.basename(tool_dir))
    src = os.path.join(tool_dir, "target", "release", binary_name)
    dst = os.path.join(tool_dir, "run")
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
    return True


def _build_binary(tool_dir: str, manifest: dict) -> bool:
    """Validate that a pre-compiled binary exists and is executable.

    Args:
        tool_dir: Path to the tool directory.
        manifest: Parsed tool.json.

    Returns:
        True if the binary exists.
    """
    entry = manifest.get("entry", "run")
    path = os.path.join(tool_dir, entry)
    if not os.path.isfile(path):
        logger.error("[ENV_BUILDER] Binary not found: %s", path)
        return False
    os.chmod(path, 0o755)
    return True


def _build_docker(tool_dir: str, manifest: dict) -> bool:
    """Build a Docker image for the tool.

    Args:
        tool_dir: Path to the tool directory.
        manifest: Parsed tool.json.

    Returns:
        True if docker build succeeded.
    """
    name = manifest.get("name", os.path.basename(tool_dir))
    image_tag = f"arqitect-tool-{name}"
    result = subprocess.run(
        ["docker", "build", "-t", image_tag, "."],
        cwd=tool_dir, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        logger.error("[ENV_BUILDER] docker build failed: %s", result.stderr)
        return False
    return True
