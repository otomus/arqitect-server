"""Tests for arqitect.github_app.

Covers GitHub App configuration, JWT generation, token caching,
git auth setup, and the setup flow (manifest + manual).
"""

import time
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest


# ── is_configured ────────────────────────────────────────────────────────

class TestIsConfigured:

    @patch("arqitect.github_app.get_secret")
    def test_returns_true_when_both_set(self, mock_secret):
        mock_secret.side_effect = lambda path, default="": {
            "github.app_id": "12345",
            "github.private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
        }.get(path, default)

        from arqitect.github_app import is_configured
        assert is_configured() is True

    @patch("arqitect.github_app.get_secret")
    def test_returns_false_when_missing(self, mock_secret):
        mock_secret.return_value = ""

        from arqitect.github_app import is_configured
        assert is_configured() is False

    @patch("arqitect.github_app.get_secret")
    def test_returns_false_when_only_app_id(self, mock_secret):
        mock_secret.side_effect = lambda path, default="": {
            "github.app_id": "12345",
        }.get(path, default)

        from arqitect.github_app import is_configured
        assert is_configured() is False


# ── _build_jwt ───────────────────────────────────────────────────────────

class TestBuildJwt:

    def test_jwt_contains_iss_claim(self):
        """JWT should contain the app_id as issuer."""
        import jwt as pyjwt
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization

        # Generate a test RSA key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()

        from arqitect.github_app import _build_jwt
        token = _build_jwt("99999", pem)

        decoded = pyjwt.decode(token, private_key.public_key(), algorithms=["RS256"])
        assert decoded["iss"] == "99999"
        assert "exp" in decoded
        assert "iat" in decoded


# ── Token Caching ────────────────────────────────────────────────────────

class TestTokenCache:

    def test_cache_returns_valid_token(self):
        """Cached tokens should be returned without re-minting."""
        from arqitect.github_app import _token_cache

        _token_cache["test/repo"] = ("cached_token", time.time() + 3000)

        from arqitect.github_app import get_installation_token

        with patch("arqitect.github_app.get_secret") as mock_secret:
            mock_secret.side_effect = lambda path, default="": {
                "github.app_id": "123",
                "github.private_key": "key",
            }.get(path, default)

            token = get_installation_token("test/repo")
            assert token == "cached_token"

        # Cleanup
        _token_cache.pop("test/repo", None)

    def test_expired_cache_is_refreshed(self):
        """Expired cached tokens should trigger a refresh."""
        from arqitect.github_app import _token_cache

        _token_cache["expired/repo"] = ("old_token", time.time() - 100)

        # Without proper keys, the refresh will fail gracefully
        with patch("arqitect.github_app.get_secret", return_value=""):
            from arqitect.github_app import get_installation_token
            token = get_installation_token("expired/repo")
            assert token is None

        _token_cache.pop("expired/repo", None)


# ── configure_git_auth ───────────────────────────────────────────────────

class TestConfigureGitAuth:

    @patch("arqitect.github_app.get_installation_token", return_value="ghs_test123")
    @patch("arqitect.github_app.get_config", return_value="TestBot")
    @patch("arqitect.github_app.get_secret")
    @patch("arqitect.github_app._run_git")
    def test_sets_remote_url_with_token(self, mock_git, mock_secret, mock_config, mock_token):
        mock_secret.return_value = "12345"

        from arqitect.github_app import configure_git_auth
        result = configure_git_auth("/tmp/repo", "owner/repo")

        assert result is True
        # Check remote URL was set with token
        calls = [str(c) for c in mock_git.call_args_list]
        assert any("x-access-token:ghs_test123" in str(c) for c in mock_git.call_args_list)

    @patch("arqitect.github_app.get_installation_token", return_value=None)
    def test_returns_false_without_token(self, mock_token):
        from arqitect.github_app import configure_git_auth
        result = configure_git_auth("/tmp/repo", "owner/repo")
        assert result is False


# ── get_app_slug ─────────────────────────────────────────────────────────

class TestGetAppSlug:

    @patch("arqitect.github_app.get_secret", return_value="")
    def test_returns_empty_when_not_configured(self, mock_secret):
        from arqitect.github_app import get_app_slug
        assert get_app_slug() == ""


# ── _build_manifest ─────────────────────────────────────────────────────

class TestBuildManifest:

    def test_contains_required_fields(self):
        from arqitect.github_app import _build_manifest
        m = _build_manifest("arqitect-ob1", "http://localhost:9999")

        assert m["name"] == "arqitect-ob1"
        assert m["redirect_url"] == "http://localhost:9999"
        assert m["hook_attributes"]["active"] is False
        assert m["public"] is False

    def test_permissions_include_contents_and_prs(self):
        from arqitect.github_app import _build_manifest
        m = _build_manifest("test", "http://localhost:1234")

        assert m["default_permissions"]["contents"] == "write"
        assert m["default_permissions"]["pull_requests"] == "write"
        assert m["default_permissions"]["metadata"] == "read"


# ── _exchange_manifest_code ─────────────────────────────────────────────

class TestExchangeManifestCode:

    @patch("arqitect.github_app.requests.post")
    def test_returns_app_data_on_success(self, mock_post):
        mock_post.return_value.json.return_value = {"id": 42, "pem": "KEY"}
        mock_post.return_value.raise_for_status = MagicMock()

        from arqitect.github_app import _exchange_manifest_code
        result = _exchange_manifest_code("test-code")

        assert result == {"id": 42, "pem": "KEY"}
        mock_post.assert_called_once()
        assert "test-code" in mock_post.call_args[0][0]

    @patch("arqitect.github_app.requests.post", side_effect=Exception("timeout"))
    def test_returns_none_on_failure(self, mock_post):
        from arqitect.github_app import _exchange_manifest_code
        assert _exchange_manifest_code("bad-code") is None


# ── setup_github_app ────────────────────────────────────────────────────

class TestSetupGithubApp:

    @patch("arqitect.github_app._setup_via_manifest", return_value=True)
    def test_uses_manifest_flow_first(self, mock_manifest):
        from arqitect.github_app import setup_github_app
        assert setup_github_app("test-app") is True
        mock_manifest.assert_called_once_with("test-app")

    @patch("arqitect.github_app._setup_via_manifest", side_effect=Exception("no browser"))
    @patch("arqitect.github_app._setup_manually", return_value=True)
    def test_falls_back_to_manual_on_manifest_failure(self, mock_manual, mock_manifest):
        from arqitect.github_app import setup_github_app
        assert setup_github_app("test-app") is True
        mock_manual.assert_called_once_with("test-app")

    @patch("arqitect.github_app.set_secret")
    @patch("arqitect.github_app._store_private_key")
    @patch("arqitect.github_app._open_pem_file_dialog", return_value="")
    def test_manual_setup_stores_key_to_secrets_dir(self, _mock_dialog, mock_store, mock_set):
        """_setup_manually saves the PEM to .secrets/ and stores the path."""
        from arqitect.github_app import _setup_manually

        mock_store.return_value = Path("/project/.secrets/github_app.pem")

        with patch("builtins.input", side_effect=["12345", "/tmp/key.pem"]), \
             patch("builtins.open", mock_open(read_data="-----BEGIN RSA-----\nKEY")):
            result = _setup_manually("arqitect-ob1")

        assert result is True
        mock_store.assert_called_once()
        mock_set.assert_any_call("github.app_id", "12345")
        mock_set.assert_any_call("github.private_key_path", "/project/.secrets/github_app.pem")
        mock_set.assert_any_call("github.app_slug", "arqitect-ob1")
