"""Tests for arqitect.mcp.env_builder — per-tool isolated dependency environments."""

import json
import os
import stat
from unittest.mock import patch, MagicMock

import pytest

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

def _write_tool_json(tool_dir, manifest):
    """Write a tool.json into tool_dir."""
    path = os.path.join(tool_dir, "tool.json")
    with open(path, "w") as f:
        json.dump(manifest, f)


def _write_env_version(tool_dir, version):
    """Write a .env_version file into tool_dir."""
    with open(os.path.join(tool_dir, ".env_version"), "w") as f:
        f.write(version)


# ---------------------------------------------------------------------------
# _read_tool_json
# ---------------------------------------------------------------------------

class TestReadToolJson:
    def test_reads_valid_json(self, tmp_path):
        """Reads a well-formed tool.json."""
        d = str(tmp_path / "mytool")
        os.makedirs(d)
        _write_tool_json(d, {"name": "mytool", "runtime": "python"})
        result = _read_tool_json(d)
        assert result["name"] == "mytool"
        assert result["runtime"] == "python"

    def test_returns_none_when_missing(self, tmp_path):
        """Returns None when tool.json does not exist."""
        assert _read_tool_json(str(tmp_path)) is None

    def test_returns_none_on_invalid_json(self, tmp_path):
        """Returns None when tool.json is malformed."""
        d = str(tmp_path / "bad")
        os.makedirs(d)
        with open(os.path.join(d, "tool.json"), "w") as f:
            f.write("not json {{{")
        assert _read_tool_json(d) is None

    def test_returns_none_on_directory_as_tool_json(self, tmp_path):
        """Returns None when tool.json path is a directory."""
        d = str(tmp_path / "weird")
        os.makedirs(os.path.join(d, "tool.json"))  # directory, not file
        assert _read_tool_json(d) is None


# ---------------------------------------------------------------------------
# env_ready
# ---------------------------------------------------------------------------

class TestEnvReady:
    def test_python_env_ready(self, tmp_path):
        """Python env is ready when .venv exists and version matches."""
        d = str(tmp_path / "py_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0"})
        _write_env_version(d, "1.0")
        os.makedirs(os.path.join(d, ".venv"))
        assert env_ready(d) is True

    def test_python_env_not_ready_version_mismatch(self, tmp_path):
        """Version mismatch means not ready."""
        d = str(tmp_path / "py_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "2.0"})
        _write_env_version(d, "1.0")
        os.makedirs(os.path.join(d, ".venv"))
        assert env_ready(d) is False

    def test_python_env_not_ready_no_venv(self, tmp_path):
        """No .venv directory means not ready."""
        d = str(tmp_path / "py_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0"})
        _write_env_version(d, "1.0")
        assert env_ready(d) is False

    def test_node_env_ready(self, tmp_path):
        """Node env is ready when node_modules exists and version matches."""
        d = str(tmp_path / "node_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "node", "version": "1.0"})
        _write_env_version(d, "1.0")
        os.makedirs(os.path.join(d, "node_modules"))
        assert env_ready(d) is True

    def test_go_env_ready(self, tmp_path):
        """Go env is ready when run binary exists and version matches."""
        d = str(tmp_path / "go_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "go", "version": "1.0"})
        _write_env_version(d, "1.0")
        with open(os.path.join(d, "run"), "w") as f:
            f.write("binary")
        assert env_ready(d) is True

    def test_rust_env_ready(self, tmp_path):
        """Rust env is ready when run binary exists and version matches."""
        d = str(tmp_path / "rust_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "rust", "version": "1.0"})
        _write_env_version(d, "1.0")
        with open(os.path.join(d, "run"), "w") as f:
            f.write("binary")
        assert env_ready(d) is True

    def test_binary_env_ready(self, tmp_path):
        """Binary env is ready when run file exists and version matches."""
        d = str(tmp_path / "bin_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "binary", "version": "1.0"})
        _write_env_version(d, "1.0")
        with open(os.path.join(d, "run"), "w") as f:
            f.write("binary")
        assert env_ready(d) is True

    def test_docker_env_ready(self, tmp_path):
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

    def test_docker_env_not_ready_no_image(self, tmp_path):
        """Docker env not ready when image does not exist."""
        d = str(tmp_path / "docker_tool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "docker", "version": "1.0", "name": "dt"})
        _write_env_version(d, "1.0")
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert env_ready(d) is False

    def test_no_tool_json_not_ready(self, tmp_path):
        """No tool.json means not ready."""
        assert env_ready(str(tmp_path)) is False

    def test_no_version_file_not_ready(self, tmp_path):
        """No .env_version file means not ready."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0"})
        assert env_ready(d) is False

    def test_unknown_runtime_not_ready(self, tmp_path):
        """Unknown runtime falls through to False."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "brainfuck", "version": "1.0"})
        _write_env_version(d, "1.0")
        assert env_ready(d) is False

    def test_default_runtime_is_python(self, tmp_path):
        """When runtime is omitted, defaults to python."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"version": "1.0"})
        _write_env_version(d, "1.0")
        os.makedirs(os.path.join(d, ".venv"))
        assert env_ready(d) is True


# ---------------------------------------------------------------------------
# build_env
# ---------------------------------------------------------------------------

class TestBuildEnv:
    def test_no_tool_json_returns_false(self, tmp_path):
        """Returns False when tool.json is missing."""
        assert build_env(str(tmp_path)) is False

    def test_unsupported_runtime_returns_false(self, tmp_path):
        """Returns False for unknown runtime."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "lua", "name": "t"})
        assert build_env(d) is False

    def test_python_build_creates_venv(self, tmp_path):
        """Python build creates venv and writes version."""
        d = str(tmp_path / "pytool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0", "name": "pytool"})
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = build_env(d)
        assert result is True
        # Version file should be written
        assert os.path.isfile(os.path.join(d, ".env_version"))
        with open(os.path.join(d, ".env_version")) as f:
            assert f.read() == "1.0"

    def test_python_build_venv_failure(self, tmp_path):
        """Python build returns False when venv creation fails."""
        d = str(tmp_path / "pytool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0", "name": "pytool"})
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="venv error")
            assert build_env(d) is False

    def test_builder_exception_returns_false(self, tmp_path):
        """Returns False when builder raises an exception."""
        d = str(tmp_path / "pytool")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0", "name": "pytool"})
        with patch("arqitect.mcp.env_builder._build_python", side_effect=OSError("disk full")):
            assert build_env(d) is False

    def test_version_written_on_success(self, tmp_path):
        """Version file is written after successful build."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "2.5", "name": "t"})
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            build_env(d)
        with open(os.path.join(d, ".env_version")) as f:
            assert f.read() == "2.5"

    def test_version_not_written_on_failure(self, tmp_path):
        """Version file is NOT written when build fails."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python", "version": "1.0", "name": "t"})
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="err")
            build_env(d)
        assert not os.path.isfile(os.path.join(d, ".env_version"))


# ---------------------------------------------------------------------------
# Individual builders
# ---------------------------------------------------------------------------

class TestBuildPython:
    def test_installs_requirements_when_present(self, tmp_path):
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

    def test_skips_pip_when_no_requirements(self, tmp_path):
        """Skips pip install when no requirements.txt."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        manifest = {"runtime": "python", "name": "t"}
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _build_python(d, manifest)
        assert mock_run.call_count == 1  # Only venv creation

    def test_pip_install_failure(self, tmp_path):
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


class TestBuildNode:
    def test_no_package_json_returns_true(self, tmp_path):
        """No package.json means no deps to install, returns True."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        assert _build_node(d, {}) is True

    def test_npm_install_success(self, tmp_path):
        """Runs npm install when package.json exists."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with open(os.path.join(d, "package.json"), "w") as f:
            json.dump({"dependencies": {}}, f)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _build_node(d, {}) is True

    def test_npm_install_failure(self, tmp_path):
        """Returns False when npm install fails."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with open(os.path.join(d, "package.json"), "w") as f:
            json.dump({}, f)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="npm err")
            assert _build_node(d, {}) is False


class TestBuildGo:
    def test_go_build_success(self, tmp_path):
        """Returns True on successful go build."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _build_go(d, {}) is True

    def test_go_build_failure(self, tmp_path):
        """Returns False on go build failure."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="go err")
            assert _build_go(d, {}) is False


class TestBuildRust:
    def test_cargo_build_copies_binary(self, tmp_path):
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

    def test_cargo_build_failure(self, tmp_path):
        """Returns False on cargo build failure."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="cargo err")
            assert _build_rust(d, {}) is False


class TestBuildBinary:
    def test_binary_exists(self, tmp_path):
        """Returns True when binary exists."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with open(os.path.join(d, "run"), "w") as f:
            f.write("bin")
        assert _build_binary(d, {}) is True

    def test_binary_missing(self, tmp_path):
        """Returns False when binary is missing."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        assert _build_binary(d, {}) is False

    def test_custom_entry(self, tmp_path):
        """Uses custom entry point from manifest."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with open(os.path.join(d, "mybin"), "w") as f:
            f.write("bin")
        assert _build_binary(d, {"entry": "mybin"}) is True


class TestBuildDocker:
    def test_docker_build_success(self, tmp_path):
        """Returns True on docker build success."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _build_docker(d, {"name": "dt"}) is True
            # Verify image tag
            cmd = mock_run.call_args[0][0]
            assert "arqitect-tool-dt" in cmd

    def test_docker_build_failure(self, tmp_path):
        """Returns False on docker build failure."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="docker err")
            assert _build_docker(d, {"name": "dt"}) is False


# ---------------------------------------------------------------------------
# cleanup_env
# ---------------------------------------------------------------------------

class TestCleanupEnv:
    def test_removes_venv(self, tmp_path):
        """Removes .venv directory."""
        d = str(tmp_path / "t")
        os.makedirs(os.path.join(d, ".venv"))
        _write_tool_json(d, {"runtime": "python"})
        cleanup_env(d)
        assert not os.path.isdir(os.path.join(d, ".venv"))

    def test_removes_node_modules(self, tmp_path):
        """Removes node_modules directory."""
        d = str(tmp_path / "t")
        os.makedirs(os.path.join(d, "node_modules"))
        _write_tool_json(d, {"runtime": "node"})
        cleanup_env(d)
        assert not os.path.isdir(os.path.join(d, "node_modules"))

    def test_removes_go_binary(self, tmp_path):
        """Removes compiled Go binary."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "go"})
        with open(os.path.join(d, "run"), "w") as f:
            f.write("bin")
        cleanup_env(d)
        assert not os.path.isfile(os.path.join(d, "run"))

    def test_removes_version_file(self, tmp_path):
        """Removes .env_version."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "python"})
        _write_env_version(d, "1.0")
        cleanup_env(d)
        assert not os.path.isfile(os.path.join(d, ".env_version"))

    def test_docker_cleanup_removes_image(self, tmp_path):
        """Docker cleanup calls docker rmi."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_tool_json(d, {"runtime": "docker", "name": "dt"})
        with patch("arqitect.mcp.env_builder.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            cleanup_env(d)
            mock_run.assert_called_once()
            assert "arqitect-tool-dt" in mock_run.call_args[0][0]

    def test_cleanup_no_tool_json_removes_artifacts(self, tmp_path):
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

class TestRebuildEnv:
    def test_rebuild_cleans_then_builds(self, tmp_path):
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

class TestWriteVersion:
    def test_writes_version_string(self, tmp_path):
        """Writes version to .env_version file."""
        d = str(tmp_path / "t")
        os.makedirs(d)
        _write_version(d, "3.2.1")
        with open(os.path.join(d, ".env_version")) as f:
            assert f.read() == "3.2.1"


# ---------------------------------------------------------------------------
# SUPPORTED_RUNTIMES
# ---------------------------------------------------------------------------

class TestSupportedRuntimes:
    def test_contains_all_expected(self):
        """All documented runtimes are in SUPPORTED_RUNTIMES."""
        expected = {"python", "node", "go", "rust", "binary", "docker"}
        assert expected == SUPPORTED_RUNTIMES

    def test_is_frozenset(self):
        """SUPPORTED_RUNTIMES is immutable."""
        assert isinstance(SUPPORTED_RUNTIMES, frozenset)
