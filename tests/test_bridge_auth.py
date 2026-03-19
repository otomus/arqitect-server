"""Tests for bridge server JWT auth, onboarding guard, and per-client routing."""

import asyncio
import json
import os
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from tests.conftest import make_mem

JWT_SECRET = "test-bridge-secret"


@pytest.fixture(autouse=True)
def _set_jwt_secret():
    with patch.dict(os.environ, {"ARQITECT_JWT_SECRET": JWT_SECRET}):
        yield


@pytest.fixture
def bridge_state():
    """Reset bridge module-level state before each test."""
    import arqitect.bridge.server as srv
    srv._client_user.clear()
    srv._onboarding_active.clear()
    srv._session_clients.clear()
    srv._pending_refresh.clear()
    srv._client_sessions.clear()
    srv.connected_clients.clear()
    srv._cold = None
    yield srv


@pytest.fixture
def fake_ws():
    """Create a fake WebSocket object."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    return ws


# ---------------------------------------------------------------------------
# Auth message handling
# ---------------------------------------------------------------------------

class TestAuthMessage:
    @pytest.mark.asyncio
    async def test_valid_token_returns_auth_ok(self, bridge_state, fake_ws, test_redis, tmp_memory_dir):
        srv = bridge_state
        from arqitect.auth.token import create_token

        mem = make_mem(test_redis)
        user_id = mem.cold.create_user_with_email("a@b.com", "dashboard", "dash_abc")
        mem.cold.set_user_display_name(user_id, "Alice")

        token = create_token(user_id, "a@b.com", "user", "Alice")
        srv._client_sessions[fake_ws] = "dash_abc"
        srv.connected_clients.add(fake_ws)

        with patch.object(srv, "_get_cold", return_value=mem.cold):
            await srv.handle_client_message({"type": "auth", "token": token}, fake_ws)

        fake_ws.send.assert_called_once()
        response = json.loads(fake_ws.send.call_args[0][0])
        assert response["type"] == "auth_ok"
        assert response["user"]["id"] == user_id
        assert srv._client_user[fake_ws] is not None

    @pytest.mark.asyncio
    async def test_invalid_token_returns_auth_error(self, bridge_state, fake_ws):
        srv = bridge_state
        srv._client_sessions[fake_ws] = "dash_abc"

        await srv.handle_client_message({"type": "auth", "token": "garbage"}, fake_ws)

        fake_ws.send.assert_called_once()
        response = json.loads(fake_ws.send.call_args[0][0])
        assert response["type"] == "auth_error"
        assert srv._client_user.get(fake_ws) is None

    @pytest.mark.asyncio
    async def test_expired_token_returns_auth_error(self, bridge_state, fake_ws):
        srv = bridge_state
        srv._client_sessions[fake_ws] = "dash_abc"
        from jose import jwt as jose_jwt

        expired = jose_jwt.encode(
            {"sub": "uid_1", "email": "a@b.com", "role": "user",
             "name": "Alice", "iat": 1000, "exp": 1001},
            JWT_SECRET, algorithm="HS256",
        )
        await srv.handle_client_message({"type": "auth", "token": expired}, fake_ws)

        response = json.loads(fake_ws.send.call_args[0][0])
        assert response["type"] == "auth_error"


# ---------------------------------------------------------------------------
# Task identity resolution
# ---------------------------------------------------------------------------

class TestTaskIdentity:
    @pytest.mark.asyncio
    async def test_authenticated_task_has_connector_user_id(self, bridge_state, fake_ws):
        srv = bridge_state
        from arqitect.auth.token import decode_token, create_token

        token = create_token("uid_1", "a@b.com", "user", "Alice")
        claims = decode_token(token)
        srv._client_sessions[fake_ws] = "dash_abc"
        srv._client_user[fake_ws] = claims

        mock_redis = MagicMock()
        with patch.object(srv, "_init_redis", return_value=mock_redis):
            await srv.handle_client_message({"type": "task", "task": "hello"}, fake_ws)

        mock_redis.publish.assert_called_once()
        payload = json.loads(mock_redis.publish.call_args[0][1])
        assert payload["connector_user_id"] == "usr_uid_1"

    @pytest.mark.asyncio
    async def test_anon_task_has_empty_connector_user_id(self, bridge_state, fake_ws):
        srv = bridge_state
        srv._client_sessions[fake_ws] = "dash_abc"
        srv._client_user[fake_ws] = None

        mock_redis = MagicMock()
        with patch.object(srv, "_init_redis", return_value=mock_redis):
            await srv.handle_client_message({"type": "task", "task": "hello"}, fake_ws)

        payload = json.loads(mock_redis.publish.call_args[0][1])
        assert payload["connector_user_id"] == ""


# ---------------------------------------------------------------------------
# Onboarding guard
# ---------------------------------------------------------------------------

class TestOnboardingGuard:
    @pytest.mark.asyncio
    async def test_request_identity_activates_onboarding(self, bridge_state, fake_ws):
        """request_identity signal in response should activate onboarding for session."""
        srv = bridge_state
        srv._client_sessions[fake_ws] = "dash_abc"
        srv._session_clients["dash_abc"] = fake_ws
        srv.connected_clients.add(fake_ws)

        data = {"message": "send your email", "request_identity": True, "chat_id": "dash_abc"}
        payload = json.dumps({"channel": "brain:response", "data": data, "timestamp": time.time()})

        await srv._route_or_broadcast("brain:response", data, payload)

        assert "dash_abc" in srv._onboarding_active

    @pytest.mark.asyncio
    async def test_onboarding_messages_not_published_to_brain(self, bridge_state, fake_ws, test_redis, tmp_memory_dir):
        """During onboarding, messages go to client directly, NOT to brain."""
        srv = bridge_state
        srv._client_sessions[fake_ws] = "dash_abc"
        srv._onboarding_active.add("dash_abc")
        srv.connected_clients.add(fake_ws)

        mem = make_mem(test_redis)
        mock_redis = MagicMock()

        with patch.object(srv, "_get_cold", return_value=mem.cold), \
             patch.object(srv, "_init_redis", return_value=mock_redis), \
             patch("arqitect.bridge.server.handle_onboarding", return_value=("Please enter the code", "")):
            await srv.handle_client_message({"type": "task", "task": "test@test.com"}, fake_ws)

        # Should NOT publish to brain
        mock_redis.publish.assert_not_called()
        # Should send directly to client
        fake_ws.send.assert_called_once()
        response = json.loads(fake_ws.send.call_args[0][0])
        assert "code" in response["data"]["message"].lower() or "enter" in response["data"]["message"].lower()

    @pytest.mark.asyncio
    async def test_onboarding_complete_mints_jwt(self, bridge_state, fake_ws, test_redis, tmp_memory_dir):
        """When onboarding finishes (user_id returned), bridge mints JWT and sends auth_ok."""
        srv = bridge_state
        srv._client_sessions[fake_ws] = "dash_abc"
        srv._onboarding_active.add("dash_abc")
        srv.connected_clients.add(fake_ws)

        mem = make_mem(test_redis)
        user_id = mem.cold.create_user_with_email("a@b.com", "dashboard", "dash_abc")
        mem.cold.set_user_display_name(user_id, "Alice")
        mem.cold.set_user_role(user_id, "user")

        with patch.object(srv, "_get_cold", return_value=mem.cold), \
             patch("arqitect.bridge.server.handle_onboarding", return_value=("", user_id)):
            await srv.handle_client_message({"type": "task", "task": "Alice"}, fake_ws)

        # Should have sent auth_ok with token
        fake_ws.send.assert_called()
        calls = [json.loads(c[0][0]) for c in fake_ws.send.call_args_list]
        auth_ok = [c for c in calls if c.get("type") == "auth_ok"]
        assert len(auth_ok) == 1
        assert "token" in auth_ok[0]
        assert auth_ok[0]["user"]["name"] == "Alice"
        # Onboarding should be cleared
        assert "dash_abc" not in srv._onboarding_active
        # User claims should be stored
        assert srv._client_user[fake_ws] is not None


# ---------------------------------------------------------------------------
# Per-client routing
# ---------------------------------------------------------------------------

class TestPerClientRouting:
    @pytest.mark.asyncio
    async def test_response_with_chat_id_routes_to_client(self, bridge_state, fake_ws):
        """Response with chat_id should go to that specific client, not broadcast."""
        srv = bridge_state
        other_ws = AsyncMock()
        srv._session_clients["dash_abc"] = fake_ws
        srv._session_clients["dash_xyz"] = other_ws
        srv.connected_clients.update({fake_ws, other_ws})

        data = {"message": "hello", "chat_id": "dash_abc"}
        payload = json.dumps({"channel": "brain:response", "data": data, "timestamp": time.time()})

        await srv._route_or_broadcast("brain:response", data, payload)

        fake_ws.send.assert_called_once()
        other_ws.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_channel_goes_to_all(self, bridge_state, fake_ws):
        """System status should broadcast to all clients."""
        srv = bridge_state
        other_ws = AsyncMock()
        srv.connected_clients.update({fake_ws, other_ws})

        data = {"cpu": 50}
        payload = json.dumps({"channel": "system:status", "data": data, "timestamp": time.time()})

        await srv._route_or_broadcast("system:status", data, payload)

        fake_ws.send.assert_called_once()
        other_ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_token_refresh_injected_in_routed_response(self, bridge_state, fake_ws):
        """Pending refresh token should be injected into routed response."""
        srv = bridge_state
        srv._session_clients["dash_abc"] = fake_ws
        srv.connected_clients.add(fake_ws)
        srv._pending_refresh["dash_abc"] = "fresh-token-123"

        data = {"message": "result", "chat_id": "dash_abc"}
        payload = json.dumps({"channel": "brain:response", "data": data, "timestamp": time.time()})

        await srv._route_or_broadcast("brain:response", data, payload)

        sent = json.loads(fake_ws.send.call_args[0][0])
        assert sent["data"]["token"] == "fresh-token-123"
        assert "dash_abc" not in srv._pending_refresh
