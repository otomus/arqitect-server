"""Response Envelope — standard rich response format for any client.

Every brain:response goes through this envelope. Clients (dashboard, mobile,
WhatsApp, Telegram, API) render what they support and ignore the rest.

Envelope schema:
{
    "message": "The main text content",
    "markdown": true,                    # hint: message may contain markdown
    "tone": "casual",                    # tone used
    "media": {
        "gif_url": "https://...",        # optional GIF
        "audio_b64": "base64...",        # optional TTS audio (aiff/wav)
        "audio_mime": "audio/aiff",      # MIME type for audio
        "image_b64": "base64...",        # optional image
        "image_mime": "image/png",       # MIME type for image
        "sticker_b64": "base64...",      # optional sticker (webp)
        "document_b64": "base64...",     # optional document
        "document_mime": "application/pdf",
        "document_name": "file.pdf",     # filename for document
    },
    "location": {                        # optional location pin
        "latitude": 32.0853,
        "longitude": 34.7818,
        "name": "Tel Aviv",
    },
    "contacts": [                        # optional contact cards
        {"name": "John", "phone": "+1234567890"},
    ],
    "poll": {                            # optional poll
        "name": "What do you prefer?",
        "options": ["Option A", "Option B"],
        "selectable_count": 1,
    },
    "card": {                            # optional structured card
        "title": "...",
        "body": "...",
        "footer": "— Sentient",
    },
    "actions": [                         # optional interactive buttons
        {"label": "Yes", "value": "yes"},
        {"label": "No", "value": "no"},
    ],
    "reactions": ["👍", "🎉"],           # optional quick reactions
    "react_to": "msg_key...",            # react to a specific message
}
"""


def build_envelope(
    message: str,
    tone: str = "neutral",
    markdown: bool = False,
    gif_url: str = "",
    audio_b64: str = "",
    audio_mime: str = "",
    image_b64: str = "",
    image_mime: str = "",
    sticker_b64: str = "",
    document_b64: str = "",
    document_mime: str = "",
    document_name: str = "",
    location: dict = None,
    contacts: list = None,
    poll: dict = None,
    card: dict = None,
    actions: list = None,
    reactions: list = None,
    react_to: str = "",
) -> dict:
    """Build a standard response envelope."""
    envelope = {
        "message": message,
        "tone": tone,
        "markdown": markdown,
    }

    media = {}
    if gif_url:
        media["gif_url"] = gif_url
    if audio_b64:
        media["audio_b64"] = audio_b64
        media["audio_mime"] = audio_mime or "audio/aiff"
    if image_b64:
        media["image_b64"] = image_b64
        media["image_mime"] = image_mime or "image/png"
    if sticker_b64:
        media["sticker_b64"] = sticker_b64
    if document_b64:
        media["document_b64"] = document_b64
        media["document_mime"] = document_mime or "application/octet-stream"
        if document_name:
            media["document_name"] = document_name
    if media:
        envelope["media"] = media

    if location:
        envelope["location"] = location
    if contacts:
        envelope["contacts"] = contacts
    if poll:
        envelope["poll"] = poll
    if card:
        envelope["card"] = card
    if actions:
        envelope["actions"] = actions
    if reactions:
        envelope["reactions"] = reactions
    if react_to:
        envelope["react_to"] = react_to

    return envelope


def merge_nerve_result_into_envelope(envelope: dict, nerve_result: dict) -> dict:
    """Merge media fields from a nerve result into an existing envelope."""
    if not isinstance(nerve_result, dict):
        return envelope

    # GIF from communication sense
    if nerve_result.get("gif_url"):
        envelope.setdefault("media", {})["gif_url"] = nerve_result["gif_url"]

    # Card from communication sense
    if nerve_result.get("card"):
        envelope["card"] = nerve_result["card"]

    # Format hint
    if nerve_result.get("format") in ("gif", "card", "emoji"):
        envelope["format"] = nerve_result["format"]

    # Audio from hearing sense (tts_file mode)
    if nerve_result.get("audio_b64"):
        envelope.setdefault("media", {})["audio_b64"] = nerve_result["audio_b64"]
        envelope.setdefault("media", {})["audio_mime"] = nerve_result.get("audio_mime", "audio/aiff")

    # Image from any nerve — prefer file path (lightweight), fall back to base64
    if nerve_result.get("image_path"):
        envelope.setdefault("media", {})["image_path"] = nerve_result["image_path"]
        envelope.setdefault("media", {})["image_mime"] = nerve_result.get("image_mime", "image/png")
    elif nerve_result.get("image_b64"):
        envelope.setdefault("media", {})["image_b64"] = nerve_result["image_b64"]
        envelope.setdefault("media", {})["image_mime"] = nerve_result.get("image_mime", "image/png")

    # Sticker
    if nerve_result.get("sticker_b64"):
        envelope.setdefault("media", {})["sticker_b64"] = nerve_result["sticker_b64"]

    # Document
    if nerve_result.get("document_b64"):
        envelope.setdefault("media", {})["document_b64"] = nerve_result["document_b64"]
        envelope.setdefault("media", {})["document_mime"] = nerve_result.get("document_mime", "application/octet-stream")
        if nerve_result.get("document_name"):
            envelope.setdefault("media", {})["document_name"] = nerve_result["document_name"]

    # Location
    if nerve_result.get("location"):
        envelope["location"] = nerve_result["location"]

    # Contacts
    if nerve_result.get("contacts"):
        envelope["contacts"] = nerve_result["contacts"]

    # Poll
    if nerve_result.get("poll"):
        envelope["poll"] = nerve_result["poll"]

    # Reaction to a specific message
    if nerve_result.get("react_to"):
        envelope["react_to"] = nerve_result["react_to"]

    return envelope
