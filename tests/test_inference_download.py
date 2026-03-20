"""Tests for arqitect.inference.download -- model downloading from HuggingFace Hub."""

import os
from unittest.mock import patch, MagicMock

import pytest


from arqitect.inference.download import ensure_model, download_gguf
import arqitect.inference.download as download_mod


# ---------------------------------------------------------------------------
# ensure_model
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestEnsureModel:
    """Tests for the ensure_model function."""

    @patch.object(download_mod, "MODEL_REGISTRY", {})
    def test_returns_none_when_not_in_registry(self):
        assert ensure_model("unknown_role", "/tmp/m") is None

    @patch.object(download_mod, "MODEL_REGISTRY", {"brain": {"source": "hf/brain"}})
    def test_returns_none_when_entry_has_no_file(self):
        assert ensure_model("brain", "/tmp/m") is None

    @patch.object(download_mod, "MODEL_REGISTRY", {"brain": {"file": "", "source": "hf/brain"}})
    def test_returns_none_when_file_is_empty_string(self):
        assert ensure_model("brain", "/tmp/m") is None

    @patch.object(download_mod, "MODEL_REGISTRY", {"brain": {"file": "brain.gguf", "source": "hf/brain"}})
    def test_returns_path_when_file_exists(self, tmp_path):
        model_file = tmp_path / "brain.gguf"
        model_file.touch()
        result = ensure_model("brain", str(tmp_path))
        assert result == str(model_file)

    @patch.object(download_mod, "MODEL_REGISTRY", {"brain": {"file": "brain.gguf"}})
    def test_returns_none_when_missing_and_no_source(self, tmp_path):
        result = ensure_model("brain", str(tmp_path))
        assert result is None

    @patch("arqitect.inference.download.download_gguf", return_value="/tmp/m/brain.gguf")
    @patch.object(download_mod, "MODEL_REGISTRY", {"brain": {"file": "brain.gguf", "source": "hf/brain"}})
    def test_downloads_when_missing_with_source(self, mock_dl, tmp_path):
        result = ensure_model("brain", str(tmp_path))
        mock_dl.assert_called_once_with("hf/brain", "brain.gguf", str(tmp_path))
        assert result == "/tmp/m/brain.gguf"

    @patch("arqitect.inference.download.download_gguf", return_value=None)
    @patch.object(download_mod, "MODEL_REGISTRY", {"brain": {"file": "brain.gguf", "source": "hf/brain"}})
    def test_returns_none_when_download_fails(self, mock_dl, tmp_path):
        result = ensure_model("brain", str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# download_gguf
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestDownloadGguf:
    """Tests for the download_gguf function."""

    @patch("arqitect.inference.download.hf_hub_download", create=True)
    def test_successful_download(self, tmp_path):
        """Successful download returns the path."""
        dest = tmp_path / "models"
        with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}) as _:
            import sys
            mock_hf = MagicMock()
            mock_hf.hf_hub_download.return_value = str(dest / "model.gguf")
            sys.modules["huggingface_hub"] = mock_hf

            result = download_gguf("org/repo", "model.gguf", str(dest))
            assert result == str(dest / "model.gguf")
            mock_hf.hf_hub_download.assert_called_once_with(
                repo_id="org/repo",
                filename="model.gguf",
                local_dir=str(dest),
                local_dir_use_symlinks=False,
            )

    def test_creates_dest_dir(self, tmp_path):
        dest = tmp_path / "new_dir" / "models"
        with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}) as _:
            import sys
            mock_hf = MagicMock()
            mock_hf.hf_hub_download.return_value = str(dest / "m.gguf")
            sys.modules["huggingface_hub"] = mock_hf

            download_gguf("repo", "m.gguf", str(dest))
            assert dest.exists()

    def test_returns_none_on_import_error(self, tmp_path):
        """When huggingface_hub is not installed, returns None."""
        dest = str(tmp_path / "models")

        import sys
        original = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = None  # force ImportError on from-import

        with patch.dict("sys.modules", {"huggingface_hub": None}):
            result = download_gguf("repo", "m.gguf", dest)
            assert result is None

        if original is not None:
            sys.modules["huggingface_hub"] = original

    def test_returns_none_on_download_exception(self, tmp_path):
        """Network errors return None."""
        dest = str(tmp_path / "models")
        mock_hf = MagicMock()
        mock_hf.hf_hub_download.side_effect = ConnectionError("no network")

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            result = download_gguf("repo", "m.gguf", dest)
            assert result is None

    def test_prints_progress_to_stderr(self, tmp_path, capsys):
        dest = str(tmp_path / "models")
        mock_hf = MagicMock()
        mock_hf.hf_hub_download.return_value = os.path.join(dest, "m.gguf")

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            download_gguf("org/repo", "m.gguf", dest)

        captured = capsys.readouterr()
        assert "Fetching" in captured.err
        assert "m.gguf" in captured.err
