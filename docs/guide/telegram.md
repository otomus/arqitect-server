# Telegram

The Telegram connector bridges Telegram messages to the brain via Redis. It uses the [Telegraf](https://telegraf.js.org/) library with long polling -- no webhooks, no public URL required.

## Setup

1. Open Telegram and message [@BotFather](https://t.me/BotFather).
2. Send `/newbot`, follow the prompts, and copy the bot token.
3. Add the token to your config:

```yaml
secrets:
  telegram_bot_token: 'YOUR_BOT_TOKEN'
```

4. Enable the connector:

```yaml
connectors:
  telegram:
    enabled: true
    bot_name: Arqitect
```

5. Restart arqitect. The connector logs `[TG] Bot launched. Listening for messages...` when ready.

## Configuration

All settings live under `connectors.telegram` in `arqitect.yaml`.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable the Telegram connector |
| `bot_name` | string | `Arqitect` | Name used for addressing in groups |
| `bot_aliases` | list | `[]` | Alternative names the bot responds to |
| `whitelisted_users` | list | `[]` | Telegram user IDs allowed to interact. Empty = all users |
| `whitelisted_groups` | list | `[]` | Group chat IDs the bot responds in. Empty = all groups |
| `monitor_groups` | list | `[]` | Group IDs to observe silently (read-only, no responses) |

The bot token goes in `secrets.telegram_bot_token`.

## How it works

The connector runs as a standalone Node.js process. On startup it connects to Redis (two clients: one for publishing, one for subscribing) and launches Telegraf in long-polling mode.

**Inbound**: Every incoming Telegram message is evaluated against the restriction model. If it passes, the connector builds a `brain:task` payload and publishes it to Redis.

**Outbound**: The connector subscribes to `brain:response`. When a response arrives with `source: "telegram"`, it routes it back to the originating chat via the Telegram Bot API.

## Media support

### Incoming

| Type | Supported | Notes |
|---|---|---|
| Text | Yes | Plain text and captions |
| Photo | Yes | Highest resolution selected, saved as `.jpg` |
| Video | Yes | Saved as `.mp4` |
| Audio | Yes | Saved as `.ogg` |
| Voice | Yes | Saved as `.ogg`, base64-encoded for senses |
| Sticker | Yes | Saved as `.webp` |
| Animation (GIF) | Yes | Saved as `.mp4` |
| Document | Yes | Any file type |
| Location | Yes | Latitude/longitude coordinates |
| Contact | Yes | Name and phone number |
| Poll | Yes | Question and options |

### Outgoing

| Type | Supported | Notes |
|---|---|---|
| Text | Yes | Markdown formatting with plain-text fallback |
| Photo | Yes | From file path or base64 |
| GIF/Animation | Yes | Sent via URL |
| Audio/Voice | Yes | Sent as voice note from base64 |
| Sticker | Yes | From base64 |
| Document | Yes | From base64 with filename |
| Location | Yes | Latitude/longitude |
| Contact | Yes | Phone and name |
| Poll | Yes | Question, options, multi-select support |
| Card | Yes | Formatted text with bold title and italic footer |

## Group handling

In direct messages, every message is processed. In groups, the bot only responds when explicitly addressed.

### Addressing methods

- **Name prefix** -- start your message with the bot name or any alias (e.g., `Arqitect, what time is it?`)
- **@mention** -- mention the bot's username (e.g., `@your_bot what time is it?`)
- **/command** -- any slash command (e.g., `/ask what time is it?`)

The bot name and aliases are case-insensitive. The prefix is stripped before the message reaches the brain.

### Monitor groups

Groups listed in `monitor_groups` are observed silently. Messages are collected and published to `telegram:monitor` but never receive a response. Use this for passive data collection from group conversations.

## Whitelisting

Both whitelists are open by default (empty list = allow all).

**Users**: Add Telegram user IDs (numbers) to `whitelisted_users`. Only these users can trigger the bot.

**Groups**: Add group chat IDs (negative numbers) to `whitelisted_groups`. The bot ignores messages from unlisted groups.

## Media storage

Downloaded media files are saved to `sandbox/tg_media/` with the naming pattern `tg_{type}_{timestamp}.{ext}`.

## Identity

Each Telegram user is identified by their numeric Telegram user ID. The connector sends this as `connector_user_id` in the brain task payload. The brain maps it to a canonical user identity via cold memory.

The payload also includes `sender_name` (first + last name) and `language_code` from the Telegram user profile.

## Data flow

```json
{
  "task": "what time is it?",
  "source": "telegram",
  "chat_id": "123456789",
  "connector_user_id": "987654321",
  "sender_name": "John Doe",
  "language_code": "en",
  "msg_id": 42
}
```

The brain processes this and publishes a response to `brain:response` with the same `source` and `chat_id`, which the connector picks up and delivers back to Telegram.

::: tip Related pages
- [Bridge (Dashboard)](/guide/bridge) -- dashboard connector
- [WhatsApp](/guide/whatsapp) -- WhatsApp connector
- [Configuration](/guide/configuration) -- full config reference
:::
