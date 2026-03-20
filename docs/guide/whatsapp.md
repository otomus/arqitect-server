# WhatsApp

The WhatsApp connector bridges WhatsApp messages to the brain via Redis. It uses the [Baileys](https://github.com/WhiskeySockets/Baileys) library to emulate a WhatsApp Web client -- no official API access or business account required.

## Setup

1. Enable the connector in `arqitect.yaml`:

```yaml
connectors:
  whatsapp:
    enabled: true
    bot_name: Arqitect
```

2. Start arqitect. On first run, a QR code appears in the terminal.
3. Open WhatsApp on your phone, go to **Linked Devices**, tap **Link a Device**, and scan the QR code.
4. The connector logs `[WA] Connected to WhatsApp` when pairing succeeds.

Session credentials are persisted in `connectors/whatsapp/auth_store/`. Subsequent restarts reconnect automatically without re-scanning.

## Configuration

All settings live under `connectors.whatsapp` in `arqitect.yaml`.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable the WhatsApp connector |
| `bot_name` | string | `Arqitect` | Name used for addressing in groups |
| `bot_aliases` | list | `[]` | Alternative names the bot responds to |
| `whitelisted_users` | list | `[]` | Phone numbers allowed to interact. Empty = all users |
| `whitelisted_groups` | list | `[]` | Group IDs the bot responds in. Empty = all groups |
| `monitor_groups` | list | `[]` | Group IDs to observe silently (read-only, no responses) |

No API key or bot token is needed. Authentication happens via the QR code pairing.

## How it works

The connector runs as a standalone Node.js process. On startup it connects to Redis (two clients: publish and subscribe) and initializes a Baileys WebSocket connection to WhatsApp's servers.

**Inbound**: Incoming messages pass through the restriction model. Valid messages become `brain:task` payloads published to Redis.

**Outbound**: The connector subscribes to `brain:response`. When a response arrives with `source: "whatsapp"`, it routes it back to the originating chat via the Baileys socket.

**Important**: Messages from `status@broadcast` are always ignored.

## Media support

### Incoming

| Type | Supported | Notes |
|---|---|---|
| Text | Yes | Conversation text, extended text, and captions |
| Image | Yes | Saved as `.jpg`, base64-encoded for vision |
| Video | Yes | Saved as `.mp4` |
| Audio | Yes | Saved as `.ogg`, base64-encoded for hearing |
| Sticker | Yes | Saved as `.webp`, base64-encoded |
| Document | Yes | Any file type |
| Location | Yes | Latitude, longitude, name, and address |
| Live location | Yes | Treated the same as static location |
| Contact | Yes | Single or multiple contacts with vCard data |
| Poll creation | Yes | Question, options, and selectable count |

### Outgoing

| Type | Supported | Notes |
|---|---|---|
| Text | Yes | Plain text |
| Image | Yes | From file path or base64, with caption |
| GIF | Yes | Downloaded as MP4, sent with `gifPlayback: true`. Tenor URL conversion built-in |
| Audio/Voice | Yes | Sent as push-to-talk voice note (`ptt: true`) |
| Sticker | Yes | From base64 |
| Document | Yes | From base64 with filename and MIME type |
| Location | Yes | Latitude, longitude, name, and address |
| Contact | Yes | vCard format, single or multiple |
| Poll | Yes | Question, options, configurable selectable count |
| Card | Yes | Formatted text with bold title and italic footer |
| Reaction | Yes | Emoji reaction on the user's message |

## Group handling

In direct messages, every message is processed. In groups, the bot only responds when addressed by name.

### Addressing

Start your message with the bot name or any alias (e.g., `Arqitect, check the weather`). The name is case-insensitive. The prefix is stripped before the message reaches the brain.

Unlike Telegram, there is no @mention or /command syntax -- name prefix is the only addressing method in WhatsApp groups.

### Monitor groups

Groups listed in `monitor_groups` are observed silently. Messages are published to `whatsapp:monitor` but never receive a response.

## Whitelisting

Both whitelists are open by default (empty list = allow all).

**Users**: Add phone numbers (without `+` or spaces) to `whitelisted_users`. Partial matching is supported -- a whitelist entry of `1555` matches any number containing that string.

**Groups**: Add group identifiers to `whitelisted_groups`. Partial matching works the same way.

## Reactions

The WhatsApp connector has full emoji reaction support. The brain can react to the user's message with any emoji by including a `reactions` array in the response. Reactions are sent before any text or media.

Incoming reactions from users are published to `brain:event` with type `reaction`.

## Presence

The connector sends typing indicators (`composing`) while the brain is processing a response, and clears them (`paused`) after the reply is sent. Presence is subscribed per-chat.

## Media storage

Downloaded media files are saved to `sandbox/wa_media/` with the naming pattern `wa_{type}_{timestamp}.{ext}`.

## Identity

Each WhatsApp user is identified by their phone number, extracted from the JID (e.g., `14155551234@s.whatsapp.net` becomes `14155551234`). This phone number is sent as `connector_user_id` in the brain task payload.

The payload also includes `sender_name` from the WhatsApp push name.

## Response timeout

Pending responses are tracked per chat with a **120-second timeout**. A background cleanup runs every 30 seconds to discard stale entries. This prevents orphaned response mappings if the brain fails to reply.

## Reconnection

If the WhatsApp connection drops, the connector automatically reconnects after 3 seconds -- unless the disconnection reason is `loggedOut`, which means the device was unlinked from WhatsApp. In that case, you need to re-scan the QR code.

The Redis subscription is set up only once and reuses the module-level socket reference, so reconnections do not create duplicate listeners.

## Data flow

```json
{
  "task": "check the weather",
  "source": "whatsapp",
  "chat_id": "14155551234@g.us",
  "connector_user_id": "14155551234",
  "sender_name": "John",
  "msg_key": { "remoteJid": "...", "id": "..." }
}
```

The brain processes this and publishes a response to `brain:response` with the same `source` and `chat_id`, which the connector picks up and delivers back to WhatsApp.

::: tip Related pages
- [Bridge (Dashboard)](/guide/bridge) -- dashboard connector
- [Telegram](/guide/telegram) -- Telegram connector
- [Configuration](/guide/configuration) -- full config reference
:::
