"""Tests for the secure credential collection flow.

Covers:
- build_credential_request() schema
- store_credentials() / get_credential() / has_credentials() round-trip
- set_secret() writes to arqitect.yaml and clears cache
- publish_response() includes request_credentials in envelope
- Bridge handles credential submission and notifies brain
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from arqitect.brain.credentials import (
    build_credential_request,
    store_credentials,
    get_credential,
    has_credentials,
)


# ── build_credential_request ─────────────────────────────────────────────

class TestBuildCredentialRequest:

    def test_minimal_request(self):
        result = build_credential_request(
            service="kaggle",
            fields=[{"key": "api_key"}],
        )
        assert result["service"] == "kaggle"
        assert len(result["fields"]) == 1
        assert result["fields"][0]["key"] == "api_key"
        assert result["fields"][0]["type"] == "password"
        assert result["reason"] == ""

    def test_full_request(self):
        result = build_credential_request(
            service="gmail",
            fields=[
                {"key": "client_id", "label": "Client ID", "type": "text"},
                {"key": "client_secret", "label": "Client Secret", "type": "password"},
            ],
            reason="To read your emails and summarize them",
        )
        assert result["service"] == "gmail"
        assert len(result["fields"]) == 2
        assert result["fields"][0]["label"] == "Client ID"
        assert result["fields"][0]["type"] == "text"
        assert result["fields"][1]["type"] == "password"
        assert "summarize" in result["reason"]

    def test_label_defaults_to_key(self):
        result = build_credential_request(
            service="test",
            fields=[{"key": "token"}],
        )
        assert result["fields"][0]["label"] == "token"


# ── store / get / has credentials ────────────────────────────────────────

class TestCredentialStorage:

    @patch("arqitect.brain.credentials.set_secret")
    def test_store_credentials_calls_set_secret(self, mock_set):
        store_credentials("kaggle", {"username": "alice", "api_key": "abc123"})
        assert mock_set.call_count == 2
        mock_set.assert_any_call("kaggle.username", "alice")
        mock_set.assert_any_call("kaggle.api_key", "abc123")

    @patch("arqitect.brain.credentials.get_secret", return_value="mykey")
    def test_get_credential(self, mock_get):
        result = get_credential("kaggle", "api_key")
        mock_get.assert_called_once_with("kaggle.api_key", "")
        assert result == "mykey"

    @patch("arqitect.brain.credentials.get_secret", return_value="")
    def test_get_credential_default(self, mock_get):
        result = get_credential("kaggle", "api_key", "fallback")
        mock_get.assert_called_once_with("kaggle.api_key", "fallback")

    @patch("arqitect.brain.credentials.get_secret")
    def test_has_credentials_all_present(self, mock_get):
        mock_get.side_effect = lambda path, default="": "value"
        assert has_credentials("kaggle", ["username", "api_key"]) is True

    @patch("arqitect.brain.credentials.get_secret")
    def test_has_credentials_missing(self, mock_get):
        mock_get.side_effect = lambda path, default="": "" if "api_key" in path else "value"
        assert has_credentials("kaggle", ["username", "api_key"]) is False


# ── set_secret round-trip ────────────────────────────────────────────────

class TestSetSecret:

    def test_set_secret_writes_yaml(self, tmp_path):
        """set_secret writes to arqitect.yaml and clears cache."""
        import yaml
        yaml_path = tmp_path / "arqitect.yaml"
        yaml_path.write_text(yaml.dump({"secrets": {"existing": "keep"}}))

        with patch("arqitect.config.loader.get_project_root", return_value=tmp_path), \
             patch("arqitect.config.loader.load_config") as mock_cache:
            mock_cache.cache_clear = MagicMock()
            from arqitect.config.loader import set_secret
            set_secret("kaggle.api_key", "test123")

        result = yaml.safe_load(yaml_path.read_text())
        assert result["secrets"]["existing"] == "keep"
        assert result["secrets"]["kaggle"]["api_key"] == "test123"

    def test_set_secret_nested(self, tmp_path):
        """set_secret creates nested structure for dotted paths."""
        import yaml
        yaml_path = tmp_path / "arqitect.yaml"
        yaml_path.write_text(yaml.dump({"name": "test"}))

        with patch("arqitect.config.loader.get_project_root", return_value=tmp_path), \
             patch("arqitect.config.loader.load_config") as mock_cache:
            mock_cache.cache_clear = MagicMock()
            from arqitect.config.loader import set_secret
            set_secret("service.nested.key", "val")

        result = yaml.safe_load(yaml_path.read_text())
        assert result["secrets"]["service"]["nested"]["key"] == "val"


# ── publish_response with request_credentials ────────────────────────────

class TestPublishResponseCredentials:

    @patch("arqitect.brain.events._record_personality_signal")
    @patch("arqitect.brain.events.publish_event")
    @patch("arqitect.brain.safety.generate_for_role", return_value='{"safe": true}')
    @patch("arqitect.brain.events.mem")
    def test_envelope_includes_request_credentials(
        self, mock_mem, mock_safety, mock_publish, mock_signal,
    ):
        mock_mem.hot.get_conversation.return_value = []
        from arqitect.brain.events import publish_response
        cred_request = build_credential_request(
            service="kaggle",
            fields=[{"key": "api_key", "label": "API Key"}],
            reason="For competition submission",
        )
        publish_response(
            "I need your Kaggle credentials to continue.",
            request_credentials=cred_request,
        )
        envelope = mock_publish.call_args[0][1]
        assert "request_credentials" in envelope
        assert envelope["request_credentials"]["service"] == "kaggle"
        assert len(envelope["request_credentials"]["fields"]) == 1

    @patch("arqitect.brain.events._record_personality_signal")
    @patch("arqitect.brain.events.publish_event")
    @patch("arqitect.brain.safety.generate_for_role", return_value='{"safe": true}')
    @patch("arqitect.brain.events.mem")
    def test_envelope_without_credentials(
        self, mock_mem, mock_safety, mock_publish, mock_signal,
    ):
        mock_mem.hot.get_conversation.return_value = []
        from arqitect.brain.events import publish_response
        publish_response("Normal response")
        envelope = mock_publish.call_args[0][1]
        assert "request_credentials" not in envelope


# ── Bridge credential handling ───────────────────────────────────────────

class TestBridgeCredentials:

    @pytest.fixture
    def bridge_setup(self):
        """Set up bridge module state for testing."""
        from arqitect.bridge import server
        ws = MagicMock()
        ws.send = MagicMock(return_value=None)
        session_id = "dash_test123"
        server._client_sessions[ws] = session_id
        server._session_clients[session_id] = ws
        yield server, ws, session_id
        server._client_sessions.pop(ws, None)
        server._session_clients.pop(session_id, None)

    @pytest.mark.asyncio
    async def test_handle_credentials_stores_and_notifies(self, bridge_setup):
        server, ws, session_id = bridge_setup

        # Make ws.send a coroutine
        async def fake_send(msg):
            pass
        ws.send = fake_send

        with patch("arqitect.bridge.server.store_credentials") as mock_store, \
             patch("arqitect.bridge.server._init_redis") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r

            await server._handle_credentials(
                {"service": "kaggle", "credentials": {"api_key": "k123"}},
                ws,
            )
            mock_store.assert_called_once_with("kaggle", {"api_key": "k123"})
            # Verify brain was notified via Redis
            publish_call = mock_r.publish.call_args
            assert publish_call[0][0] == "brain:credentials"
            payload = json.loads(publish_call[0][1])
            assert payload["service"] == "kaggle"
            assert "api_key" in payload["keys"]

    @pytest.mark.asyncio
    async def test_handle_credentials_rejects_empty(self, bridge_setup):
        server, ws, session_id = bridge_setup
        sent_messages = []

        async def fake_send(msg):
            sent_messages.append(json.loads(msg))
        ws.send = fake_send

        await server._handle_credentials({"service": "", "credentials": {}}, ws)
        assert sent_messages[0]["type"] == "credentials_error"


# ── Route detects request_credentials ────────────────────────────────────

class TestRouteCredentialDetection:

    @pytest.mark.asyncio
    async def test_route_detects_credential_request(self):
        from arqitect.bridge import server
        server._credentials_pending.clear()
        data = {
            "message": "I need your credentials",
            "request_credentials": {"service": "kaggle", "fields": []},
            "chat_id": "dash_abc",
        }
        payload = json.dumps({"channel": "brain:response", "data": data})
        with patch.object(server, "broadcast", return_value=None):
            await server._route_or_broadcast("brain:response", data, payload)
        assert "dash_abc" in server._credentials_pending
        server._credentials_pending.discard("dash_abc")
