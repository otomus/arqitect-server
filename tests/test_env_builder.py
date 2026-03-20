"""Tests for arqitect.mcp.env_builder — per-tool isolated dependency environments."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from dirty_equals import IsInstance
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from arqitect.mcp.env_builder import (
    SUPPORTED_RUNTIMES,
    _read_tool_json,
    env_ready,
    build_env,
    rebuild_env,
    cleanup_env,
    _write_version,
    _build_python,
    _build_node,
    _build_go,
    _build_rust,
    _build_binary,
    _build_docker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tool_json(tool_dir: str, manifest: dict) -> None:
    """Write a tool.json into tool_dir."""
    path = os.path.join(tool_dir, "tool.json")
    with open(path, "w") as f:
        json.dump(manifest, f)


def _write_env_version(tool_dir: str, version: str) -> None:
    """Write a .env_version file into tool_dir."""
    with open(os.path.join(tool_dir, ".env_version"), "w") as f:
        f.write(version)


# Strategy for valid runtime strings
VALID_RUNTIMES = st.sampled_from(sorted(SUPPORTED_RUNTIMES))

# Strategy for version strings (semver-like)
VERSION_STRATEGY = st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True)

# Strategy for tool names (simple alphanumeric)
TOOL_NAME_STRATEGY = st.from_regex(r"[a-z][a-z0-9_]{1,15}", fullmatch=True)


# ---------------------------------------------------------------------------
# _read_tool_json
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestReadToolJson:
    """Contract: _read_tool_json returns parsed dict for valid JSON, None otherwise."""

    def test_reads_valid_json(self, tmp_path: object) -> None:
        """Reads a well-formed tool.json."""
        d = str(tmp_path / "mytool")
        os.makedirs(d)
        _write_tool_json(d, {"name": "mytool", "runtime": "python"})
        result = _read_tool_json(d)
        assert result == {"name": "mytool", "runtime": "python"}

    def test_returns_none_when_missing(self, tmp_path: object) -> None:
        """Returns None when tool.json does not exist."""
        assert _read_tool_json(str(tmp_path)) is None

    def test_returns_none_on_invalid_json(self, tmp_path: object) -> None:
        """Returns None when tool.json is malformed."""
        d = str(tmp_path / "bad")
        os.makedirs(d)
        with open(os.path.join(d, "tool.json"), "w") as f:
            f.write("not json {{{")
        assert _read_tool_json(d) is None

    def test_returns_none_on_directory_as_tool_json(self, tmp_path: object) -> None:
        """Returns None when tool.json path is a directory."""
        d = str(tmp_path / "weird")
        os.makedirs(os.path.join(d, "tool.json"))  # directory, not file
        assert _read_tool_json(d) is None

    @given(
        name=TOOL_NAME_STRATEGY,
        runtime=VALID_RUNTIMES,
        version=VERSION_STRATEGY,
    )
    @settings(max_examples=20)
    def test_roundtrip_any_valid_manifest(
        self, name: str, runtime: str, version: str
    ) -> None:
        """Any well-formed manifest round-trips through write/read."""
        d = tempfile.mkdtemp()
        try:
            manifest = {"name": name, "runtime": runtime, "version": version}
            _write_tool_json(d, manifest)
            result = _read_tool_json(d)
            assert result == manifest
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# env_ready
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestEnvReady:
    """Contract: env_ready returns True only when runtime artifacts AND version match."""

    def test_python_env_ready(self, tmp_path: object) -> None:
        """Python env is ready when .venv exists and version matches."""
        d = str(tmp_path / "py_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0"})
        _write_env_version(d, "1.0")
        os.makedirs(os.path.join(d, ".venv"))
        assert env_ready(d) is True

    def test_python_env_not_ready_version_mismatch(self, tmp_path: object) -> None:
        """Version mismatch means not ready."""
        d = str(tmp_path / "py_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "2.0"})
        _write_env_version(d, "1.0")
        os.makedirs(os.path.join(d, ".venv"))
        assert env_ready(d) is False

    def test_python_env_not_ready_no_venv(self, tmp_path: object) -> None:
        """No .venv directory means not ready."""
        d = str(tmp_path / "py_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0"})
        _write_env_version(d, "1.0")
        assert env_ready(d) is False

    def test_node_env_ready(self, tmp_path: object) -> None:
        """Node env is ready when node_modules exists and version matches."""
        d = str(tmp_path / "node_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "node", "version": "1.0"})
        _write_env_version(d, "1.0")
        os.makedirs(os.path.join(d, "node_modules"))
        assert env_ready(d) is True

    def test_go_env_ready(self, tmp_path: object) -> None:
        """Go env is ready when run binary exists and version matches."""
        d = str(tmp_path / "go_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "go", "version": "1.0"})
        _write_env_version(d, "1.0")
        with open(os.path.join(d, "run"), "w") as f:
            f.write("binary")
        assert env_ready(d) is True

    def test_rust_env_ready(self, tmp_path: object) -> None:
        """Rust env is ready when run binary exists and version matches."""
        d = str(tmp_path / "rust_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "rust", "version": "1.0"})
        _write_env_version(d, "1.0")
        with open(os.path.join(d, "run"), "w") as f:
            f.write("binary")
        assert env_ready(d) is True

    def test_binary_env_ready(self, tmp_path: object) -> None:
        """Binary env is ready when run file exists and version matches."""
        d = str(tmp_path / "bin_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "binary", "version": "1.0"})
        _write_env_version(d, "1.0")
        with open(os.path.join(d, "run"), "w") as f:
            f.write("binary")
        assert env_ready(d) is True

    def test_docker_env_ready(self, tmp_path: object) -> None:
        """Docker env is ready when image exists."""
        d = str(tmp_path / "docker_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "docker", "version": "1.0", "name": "dt"})
        _write_env_version(d, "1.0")
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert env_ready(d) is True
            mock_run.assert_called_once()
            assert "arqitect-tool-dt" in mock_run.call_args[0][0]

    def test_docker_env_not_ready_no_image(self, tmp_path: object) -> None:
        """Docker env not ready when image does not exist."""
        d = str(tmp_path / "docker_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "docker", "version": "1.0", "name": "dt"})
        _write_env_version(d, "1.0")
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert env_ready(d) is False

    def test_no_tool_json_not_ready(self, tmp_path: object) -> None:
        """No tool.json means not ready."""
        assert env_ready(str(tmp_path)) is False

    def test_no_version_file_not_ready(self, tmp_path: object) -> None:
        """No .env_version file means not ready."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0"})
        assert env_ready(d) is False

    def test_unknown_runtime_not_ready(self, tmp_path: object) -> None:
        """Unknown runtime falls through to False."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "brainfuck", "version": "1.0"})
        _write_env_version(d, "1.0")
        assert env_ready(d) is False

    def test_default_runtime_is_python(self, tmp_path: object) -> None:
        """When runtime is omitted, defaults to python."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"version": "1.0"})
        _write_env_version(d, "1.0")
        os.makedirs(os.path.join(d, ".venv"))
        assert env_ready(d) is True

    @given(
        version_a=VERSION_STRATEGY,
        version_b=VERSION_STRATEGY,
    )
    @settings(max_examples=20)
    def test_version_mismatch_always_not_ready(
        self, version_a: str, version_b: str
    ) -> None:
        """Any version mismatch causes env_ready to return False, regardless of artifacts."""
        assume(version_a != version_b)
        d = tempfile.mkdtemp()
        try:
            _write_tool_json(d, {"runtime": "python", "version": version_a})
            _write_env_version(d, version_b)
            os.makedirs(os.path.join(d, ".venv"))
            assert env_ready(d) is False
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# build_env
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestBuildEnv:
    """Contract: build_env returns bool and writes .env_version only on success."""

    def test_no_tool_json_returns_false(self, tmp_path: object) -> None:
        """Returns False when tool.json is missing."""
        assert build_env(str(tmp_path)) is False

    def test_unsupported_runtime_returns_false(self, tmp_path: object) -> None:
        """Returns False for unknown runtime."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "lua", "name": "t"})
        assert build_env(d) is False

    def test_python_build_creates_venv(self, tmp_path: object) -> None:
        """Python build creates venv and writes version."""
        d = str(tmp_path / "pytool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0", "name": "pytool"})
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = build_env(d)
        assert result is True
        version_path = os.path.join(d, ".env_version")
        assert os.path.isfile(version_path)
        with open(version_path) as f:
            assert f.read() == "1.0"

    def test_python_build_venv_failure(self, tmp_path: object) -> None:
        """Python build returns False when venv creation fails."""
        d = str(tmp_path / "pytool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0", "name": "pytool"})
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="venv error")
            assert build_env(d) is False

    def test_builder_exception_returns_false(self, tmp_path: object) -> None:
        """Returns False when builder raises an exception."""
        d = str(tmp_path / "pytool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0", "name": "pytool"})
        with patch("arqitect.mcp.env_builder._build_python", side_effect=OSError("disk full")):
            assert build_env(d) is False

    def test_version_written_on_success(self, tmp_path: object) -> None:
        """Version file is written after successful build."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "2.5", "name": "t"})
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            build_env(d)
        with open(os.path.join(d, ".env_version")) as f:
            assert f.read() == "2.5"

    def test_version_not_written_on_failure(self, tmp_path: object) -> None:
        """Version file is NOT written when build fails."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0", "name": "t"})
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="err")
            build_env(d)
        assert not os.path.isfile(os.path.join(d, ".env_version"))

    @given(runtime=st.text(min_size=1).filter(lambda r: r not in SUPPORTED_RUNTIMES))
    @settings(max_examples=10)
    def test_unsupported_runtime_never_writes_version(
        self, runtime: str
    ) -> None:
        """No unsupported runtime should ever produce a .env_version file."""
        d = tempfile.mkdtemp()
        try:
            _write_tool_json(d, {"runtime": runtime, "name": "fuzz"})
            build_env(d)
            assert not os.path.isfile(os.path.join(d, ".env_version"))
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Individual builders
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestBuildPython:
    """Contract: _build_python creates venv, optionally installs deps, returns bool."""

    def test_installs_requirements_when_present(self, tmp_path: object) -> None:
        """Runs pip install when requirements.txt exists."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with open(os.path.join(d, "requirements.txt"), "w") as f:
            f.write("requests\n")
        manifest = {"runtime": "python", "name": "t"}
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = _build_python(d, manifest)
        assert result is True
        # Should be called twice: venv creation + pip install
        assert mock_run.call_count == 2

    def test_skips_pip_when_no_requirements(self, tmp_path: object) -> None:
        """Skips pip install when no requirements.txt."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        manifest = {"runtime": "python", "name": "t"}
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _build_python(d, manifest)
        assert mock_run.call_count == 1  # Only venv creation

    def test_pip_install_failure(self, tmp_path: object) -> None:
        """Returns False when pip install fails."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with open(os.path.join(d, "requirements.txt"), "w") as f:
            f.write("nonexistent-pkg\n")
        manifest = {"runtime": "python", "name": "t"}
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(returncode=0)  # venv ok
            return MagicMock(returncode=1, stderr="pip error")

        with patch("arqitect.mcp.env_builder.subprocess.run", side_effect=side_effect):
            assert _build_python(d, manifest) is False


@pytest.mark.timeout(10)
class TestBuildNode:
    """Contract: _build_node runs npm install only if package.json exists."""

    def test_no_package_json_returns_true(self, tmp_path: object) -> None:
        """No package.json means no deps to install, returns True."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        assert _build_node(d, {}) is True

    def test_npm_install_success(self, tmp_path: object) -> None:
        """Runs npm install when package.json exists."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with open(os.path.join(d, "package.json"), "w") as f:
            json.dump({"dependencies": {}}, f)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _build_node(d, {}) is True

    def test_npm_install_failure(self, tmp_path: object) -> None:
        """Returns False when npm install fails."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with open(os.path.join(d, "package.json"), "w") as f:
            json.dump({}, f)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="npm err")
            assert _build_node(d, {}) is False


@pytest.mark.timeout(10)
class TestBuildGo:
    """Contract: _build_go compiles to a 'run' binary, returns bool."""

    def test_go_build_success(self, tmp_path: object) -> None:
        """Returns True on successful go build."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _build_go(d, {}) is True

    def test_go_build_failure(self, tmp_path: object) -> None:
        """Returns False on go build failure."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="go err")
            assert _build_go(d, {}) is False


@pytest.mark.timeout(10)
class TestBuildRust:
    """Contract: _build_rust compiles via cargo and copies release binary to ./run."""

    def test_cargo_build_copies_binary(self, tmp_path: object) -> None:
        """Copies release binary to ./run on success."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        # Create the expected release binary
        rel_dir = os.path.join(d, "target", "release")
        os.makedirs(rel_dir)
        with open(os.path.join(rel_dir, "mytool"), "w") as f:
            f.write("binary")
        manifest = {"name": "mytool"}
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _build_rust(d, manifest) is True
        # The run binary should exist
        assert os.path.isfile(os.path.join(d, "run"))

    def test_cargo_build_failure(self, tmp_path: object) -> None:
        """Returns False on cargo build failure."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="cargo err")
            assert _build_rust(d, {}) is False


@pytest.mark.timeout(10)
class TestBuildBinary:
    """Contract: _build_binary validates the binary exists and makes it executable."""

    def test_binary_exists(self, tmp_path: object) -> None:
        """Returns True when binary exists."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with open(os.path.join(d, "run"), "w") as f:
            f.write("bin")
        assert _build_binary(d, {}) is True

    def test_binary_missing(self, tmp_path: object) -> None:
        """Returns False when binary is missing."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        assert _build_binary(d, {}) is False

    def test_custom_entry(self, tmp_path: object) -> None:
        """Uses custom entry point from manifest."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with open(os.path.join(d, "mybin"), "w") as f:
            f.write("bin")
        assert _build_binary(d, {"entry": "mybin"}) is True

    def test_binary_is_made_executable(self, tmp_path: object) -> None:
        """Binary gets chmod 0o755 after _build_binary succeeds."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        bin_path = os.path.join(d, "run")
        with open(bin_path, "w") as f:
            f.write("bin")
        _build_binary(d, {})
        mode = os.stat(bin_path).st_mode
        assert mode & 0o755 == 0o755


@pytest.mark.timeout(10)
class TestBuildDocker:
    """Contract: _build_docker builds a Docker image tagged arqitect-tool-<name>."""

    def test_docker_build_success(self, tmp_path: object) -> None:
        """Returns True on docker build success."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _build_docker(d, {"name": "dt"}) is True
            # Verify image tag
            cmd = mock_run.call_args[0][0]
            assert "arqitect-tool-dt" in cmd

    def test_docker_build_failure(self, tmp_path: object) -> None:
        """Returns False on docker build failure."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="docker err")
            assert _build_docker(d, {"name": "dt"}) is False

    @given(name=TOOL_NAME_STRATEGY)
    @settings(max_examples=10)
    def test_image_tag_always_prefixed(self, name: str) -> None:
        """Docker image tag always uses the arqitect-tool- prefix."""
        d = tempfile.mkdtemp()
        try:
            with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                _build_docker(d, {"name": name})
                cmd = mock_run.call_args[0][0]
                assert f"arqitect-tool-{name}" in cmd
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# cleanup_env
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestCleanupEnv:
    """Contract: cleanup_env removes all runtime artifacts and version tracking."""

    def test_removes_venv(self, tmp_path: object) -> None:
        """Removes .venv directory."""
        d = str(tmp_path / "t")
        os.makedirs(os.path.join(d, ".venv"))
        _write_tool_json(d, {"runtime": "python"})
        cleanup_env(d)
        assert not os.path.isdir(os.path.join(d, ".venv"))

    def test_removes_node_modules(self, tmp_path: object) -> None:
        """Removes node_modules directory."""
        d = str(tmp_path / "t")
        os.makedirs(os.path.join(d, "node_modules"))
        _write_tool_json(d, {"runtime": "node"})
        cleanup_env(d)
        assert not os.path.isdir(os.path.join(d, "node_modules"))

    def test_removes_go_binary(self, tmp_path: object) -> None:
        """Removes compiled Go binary."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "go"})
        with open(os.path.join(d, "run"), "w") as f:
            f.write("bin")
        cleanup_env(d)
        assert not os.path.isfile(os.path.join(d, "run"))

    def test_removes_version_file(self, tmp_path: object) -> None:
        """Removes .env_version."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python"})
        _write_env_version(d, "1.0")
        cleanup_env(d)
        assert not os.path.isfile(os.path.join(d, ".env_version"))

    def test_docker_cleanup_removes_image(self, tmp_path: object) -> None:
        """Docker cleanup calls docker rmi."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "docker", "name": "dt"})
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            cleanup_env(d)
            mock_run.assert_called_once()
            assert "arqitect-tool-dt" in mock_run.call_args[0][0]

    def test_cleanup_no_tool_json_removes_artifacts(self, tmp_path: object) -> None:
        """Cleanup still removes common directories even without tool.json."""
        d = str(tmp_path / "t")
        os.makedirs(os.path.join(d, ".venv"))
        os.makedirs(os.path.join(d, "node_modules"))
        cleanup_env(d)
        assert not os.path.isdir(os.path.join(d, ".venv"))
        assert not os.path.isdir(os.path.join(d, "node_modules"))


# ---------------------------------------------------------------------------
# rebuild_env
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestRebuildEnv:
    """Contract: rebuild_env tears down then builds, returning build result."""

    def test_rebuild_cleans_then_builds(self, tmp_path: object) -> None:
        """Rebuild calls cleanup then build."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0", "name": "t"})
        with patch("arqitect.mcp.env_builder.cleanup_env") as mock_cleanup, \
             patch("arqitect.mcp.env_builder.build_env", return_value=True) as mock_build:
            result = rebuild_env(d)
        assert result is True
        mock_cleanup.assert_called_once_with(d)
        mock_build.assert_called_once_with(d)


# ---------------------------------------------------------------------------
# _write_version
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestWriteVersion:
    """Contract: _write_version writes a version string to .env_version."""

    def test_writes_version_string(self, tmp_path: object) -> None:
        """Writes version to .env_version file."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_version(d, "3.2.1")
        with open(os.path.join(d, ".env_version")) as f:
            assert f.read() == "3.2.1"

    @given(version=VERSION_STRATEGY)
    @settings(max_examples=20)
    def test_any_semver_roundtrips(self, version: str) -> None:
        """Any semver-like version string round-trips through write/read."""
        d = tempfile.mkdtemp()
        try:
            _write_version(d, version)
            with open(os.path.join(d, ".env_version")) as f:
                assert f.read() == version
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# SUPPORTED_RUNTIMES
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestSupportedRuntimes:
    """Contract: SUPPORTED_RUNTIMES is an immutable set of known runtime strings."""

    def test_contains_all_expected(self) -> None:
        """All documented runtimes are in SUPPORTED_RUNTIMES."""
        expected = {"python", "node", "go", "rust", "binary", "docker"}
        assert expected == SUPPORTED_RUNTIMES

    def test_is_frozenset(self) -> None:
        """SUPPORTED_RUNTIMES is immutable."""
        assert SUPPORTED_RUNTIMES == IsInstance(frozenset)
