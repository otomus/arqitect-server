"""Python WebSocket bridge -- the dashboard/SDK connector.

Bridges Redis Pub/Sub to WebSocket clients (dashboard).
Handles its own identity resolution via JWT — same contract as
Telegram/WhatsApp connectors. The brain never sees onboarding
messages from bridge clients.
"""

import asyncio
import json
import os
import time
import uuid

import redis
import websockets
from websockets.server import serve

from arqitect.config.loader import (
    get_bridge_port, get_redis_host_port, get_ssl_context,
)
from arqitect.types import Channel
from arqitect.auth.token import create_token, decode_token, should_refresh
from arqitect.brain.onboarding import handle_onboarding, get_onboarding_state, VERIFIED

# Module-level Redis client (created once at startup via _init_redis)
_redis_client: redis.Redis | None = None


def _init_redis() -> redis.Redis:
    """Create and cache a module-level Redis client."""
    global _redis_client
    if _redis_client is None:
        host, port = get_redis_host_port()
        _redis_client = redis.Redis(host=host, port=port, decode_responses=True)
    return _redis_client


# Bridge's own ColdMemory instance (lazy-initialized)
_cold = None


def _get_cold():
    """Lazy-init a ColdMemory for onboarding queries."""
    global _cold
    if _cold is None:
        from arqitect.memory.cold import ColdMemory
        _cold = ColdMemory()
    return _cold


# Redis channels to subscribe to
CHANNELS = [
    Channel.BRAIN_THOUGHT, Channel.BRAIN_ACTION, Channel.BRAIN_RESPONSE,
    Channel.NERVE_RESULT, Channel.NERVE_QUALIFICATION,
    Channel.SYSTEM_STATUS,
    Channel.MEMORY_EPISODE, Channel.MEMORY_TOOL_LEARNED,
    Channel.SENSE_SIGHT_FRAME, Channel.SENSE_STT_RESULT,
]

# Channels that should be routed per-client (not broadcast)
_ROUTED_CHANNELS = frozenset({
    Channel.BRAIN_RESPONSE, Channel.BRAIN_THOUGHT,
    Channel.BRAIN_ACTION, Channel.NERVE_RESULT,
})

connected_clients: set = set()

# Map WebSocket → session ID (persists for the lifetime of the connection)
_client_sessions: dict = {}
# Reverse: session_id → WebSocket
_session_clients: dict = {}
# WebSocket → decoded JWT claims (None = anon)
_client_user: dict = {}
# Session IDs with active onboarding
_onboarding_active: set = set()
# session_id → fresh token to piggyback on next response
_pending_refresh: dict = {}


async def handler(websocket):
    """Handle a new WebSocket connection."""
    session_id = f"dash_{uuid.uuid4().hex[:12]}"
    connected_clients.add(websocket)
    _client_sessions[websocket] = session_id
    _session_clients[session_id] = websocket
    _client_user[websocket] = None
    print(f"[BRIDGE] Client connected ({len(connected_clients)} total) session={session_id}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_client_message(data, websocket)
            except json.JSONDecodeError:
                pass
    except websockets.ConnectionClosed:
        pass
    finally:
        session_id = _client_sessions.pop(websocket, "")
        _session_clients.pop(session_id, None)
        _client_user.pop(websocket, None)
        _onboarding_active.discard(session_id)
        _pending_refresh.pop(session_id, None)
        connected_clients.discard(websocket)
        print(f"[BRIDGE] Client disconnected ({len(connected_clients)} total)")


async def handle_client_message(data: dict, websocket=None):
    """Handle a message from a WebSocket client."""
    msg_type = data.get("type", "")
    r = _init_redis()

    if msg_type == "auth":
        await _handle_auth(data, websocket)
        return

    if msg_type == "nerve_details":
        name = data.get("name", "")
        if name and websocket:
            raw = await asyncio.to_thread(r.hget, "synapse:nerve_details", name)
            if raw:
                detail = json.loads(raw)
            else:
                detail = {"name": name, "error": "not found"}
            await websocket.send(json.dumps({"channel": "nerve:details", "data": detail}))

    elif msg_type == "task":
        task = data.get("task", "")
        if task:
            session_id = _client_sessions.get(websocket, "")

            # Onboarding guard — handle flow entirely in bridge
            if session_id in _onboarding_active:
                await _handle_onboarding_message(websocket, session_id, task)
                return

            # Resolve identity from JWT claims
            claims = _client_user.get(websocket)
            connector_user_id = f"usr_{claims['sub']}" if claims else ""

            await asyncio.to_thread(
                r.publish, Channel.BRAIN_TASK, json.dumps({
                    "task": task,
                    "source": "dashboard",
                    "chat_id": session_id,
                    "connector_user_id": connector_user_id,
                }),
            )

            # Sliding refresh check
            if claims and should_refresh(claims):
                _pending_refresh[session_id] = create_token(
                    claims["sub"], claims["role"], claims["name"],
                )

    elif msg_type == "peek":
        source = data.get("source", "screenshot")
        await asyncio.to_thread(
            r.publish, Channel.SENSE_PEEK, json.dumps({"source": source}),
        )

    elif msg_type == "voice":
        audio_b64 = data.get("audio_b64", "")
        if audio_b64:
            await asyncio.to_thread(
                r.publish, Channel.SENSE_VOICE, json.dumps({"audio_b64": audio_b64}),
            )

    elif msg_type == "image":
        image_b64 = data.get("image_b64", "")
        prompt = data.get("prompt", "")
        if image_b64:
            await asyncio.to_thread(
                r.publish, Channel.SENSE_IMAGE, json.dumps({
                    "image_b64": image_b64,
                    "prompt": prompt,
                }),
            )

    elif msg_type == "sense_config":
        sense = data.get("sense", "")
        key = data.get("key", "")
        value = data.get("value", "")
        if sense and key:
            config_key = f"{sense}.{key}"
            await asyncio.to_thread(r.hset, "synapse:sense_config", config_key, value)
            await asyncio.to_thread(
                r.publish, "sense:config", json.dumps({
                    "sense": sense, "key": key, "value": value,
                }),
            )

    elif msg_type == "kill":
        await asyncio.to_thread(r.publish, Channel.SYSTEM_KILL, "kill")


async def _handle_auth(data: dict, websocket):
    """Validate JWT and store claims for the connection."""
    token = data.get("token", "")
    claims = decode_token(token)
    if not claims:
        await websocket.send(json.dumps({
            "type": "auth_error",
            "reason": "invalid_token",
        }))
        return

    # Verify user still exists in cold storage
    cold = _get_cold()
    user_id = claims.get("sub", "")
    user = cold.get_user(user_id)
    if not user:
        await websocket.send(json.dumps({
            "type": "auth_error",
            "reason": "user_not_found",
        }))
        return

    _client_user[websocket] = claims
    await websocket.send(json.dumps({
        "type": "auth_ok",
        "user": {
            "id": user_id,
            "name": claims.get("name", ""),
            "role": claims.get("role", "user"),
        },
    }))


async def _handle_onboarding_message(websocket, session_id: str, task: str):
    """Route an onboarding message through the shared state machine."""
    cold = _get_cold()
    msg, user_id = handle_onboarding(cold, "dashboard", session_id, task)

    if user_id and not msg:
        # Onboarding complete — mint JWT and send auth_ok
        _complete_onboarding(websocket, session_id, user_id, cold)
        return

    if msg:
        await websocket.send(json.dumps({
            "channel": "brain:response",
            "data": {"message": msg},
        }))


def _complete_onboarding(websocket, session_id: str, user_id: str, cold):
    """Mint JWT, store claims, clear onboarding, and notify client."""
    user = cold.get_user(user_id) or {}
    role = user.get("role", "user")
    display_name = user.get("display_name", "")

    token = create_token(user_id, role, display_name)
    _client_user[websocket] = decode_token(token)
    _onboarding_active.discard(session_id)

    asyncio.ensure_future(websocket.send(json.dumps({
        "type": "auth_ok",
        "token": token,
        "user": {
            "id": user_id,
            "name": display_name,
            "role": role,
        },
    })))

    welcome = f"Welcome, {display_name}! How can I help you?" if display_name else "Welcome! How can I help you?"
    asyncio.ensure_future(websocket.send(json.dumps({
        "channel": "brain:response",
        "data": {"message": welcome},
    })))


async def _route_or_broadcast(channel: str, data, payload: str):
    """Route per-client or broadcast depending on channel and chat_id."""
    chat_id = data.get("chat_id", "") if isinstance(data, dict) else ""

    # Detect identity request signal — activate onboarding
    if isinstance(data, dict) and data.get("request_identity") and chat_id:
        _onboarding_active.add(chat_id)

    if channel in _ROUTED_CHANNELS and chat_id and chat_id in _session_clients:
        # Inject pending token refresh
        if chat_id in _pending_refresh:
            if isinstance(data, dict):
                data["token"] = _pending_refresh.pop(chat_id)
                payload = json.dumps({
                    "channel": channel, "data": data,
                    "timestamp": time.time(),
                })
        await _session_clients[chat_id].send(payload)
    else:
        await broadcast(payload)


async def broadcast(message: str):
    """Broadcast a message to all connected WebSocket clients."""
    clients = list(connected_clients)
    if clients:
        await asyncio.gather(
            *[client.send(message) for client in clients],
            return_exceptions=True,
        )


def _poll_redis_message(pubsub):
    """Blocking call to get next Redis pub/sub message (runs in thread)."""
    return pubsub.get_message(timeout=0.1)


async def redis_listener():
    """Listen to Redis Pub/Sub and route/broadcast to WebSocket clients."""
    r = _init_redis()
    pubsub = r.pubsub()
    pubsub.subscribe(*CHANNELS)

    while True:
        message = await asyncio.to_thread(_poll_redis_message, pubsub)
        if message and message["type"] == "message":
            channel = message["channel"]
            try:
                data = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                data = {"raw": message["data"]}

            payload = json.dumps({
                "channel": channel,
                "data": data,
                "timestamp": time.time(),
            })
            await _route_or_broadcast(channel, data, payload)
        else:
            await asyncio.sleep(0.05)


async def stats_broadcaster():
    """Periodically broadcast system stats."""
    import psutil
    while True:
        if connected_clients:
            stats = {
                "cpu": psutil.cpu_percent(),
                "memory": psutil.virtual_memory().percent,
                "uptime": time.time(),
            }
            payload = json.dumps({
                "channel": Channel.SYSTEM_STATUS,
                "data": stats,
                "timestamp": time.time(),
            })
            await broadcast(payload)
        await asyncio.sleep(2)


async def main():
    """Start the bridge server."""
    _init_redis()

    port = get_bridge_port()
    ssl_context = get_ssl_context()
    scheme = "wss" if ssl_context else "ws"
    print(f"[BRIDGE] Starting Python WebSocket bridge on port {port} ({scheme})")

    async with serve(handler, "0.0.0.0", port, ssl=ssl_context):
        # Run Redis listener and stats broadcaster concurrently
        await asyncio.gather(
            redis_listener(),
            stats_broadcaster(),
        )


if __name__ == "__main__":
    asyncio.run(main())
