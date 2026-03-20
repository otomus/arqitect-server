"""Tests for the response envelope builder and nerve result merger.

Covers:
- build_envelope with all parameter combinations
- Media field inclusion (gif, audio, image, sticker, document)
- Optional fields (location, contacts, poll, card, actions, reactions, react_to)
- Default MIME types when not specified
- merge_nerve_result_into_envelope for all media types
- Merge with non-dict nerve results
- Merge with missing/empty fields
- Envelope immutability across merges
"""

import pytest

from arqitect.senses.communication.envelope import (
    build_envelope,
    merge_nerve_result_into_envelope,
)


# ── build_envelope ────────────────────────────────────────────────────────────

class TestBuildEnvelope:
    """Standard response envelope construction."""

    def test_minimal_envelope(self):
        """Minimal envelope contains message, tone, and markdown flag."""
        env = build_envelope("Hello")
        assert env["message"] == "Hello"
        assert env["tone"] == "neutral"
        assert env["markdown"] is False
        assert "media" not in env

    def test_custom_tone_and_markdown(self):
        """Custom tone and markdown flag are set."""
        env = build_envelope("Hi", tone="casual", markdown=True)
        assert env["tone"] == "casual"
        assert env["markdown"] is True

    def test_gif_url_creates_media(self):
        """A gif_url populates the media dict."""
        env = build_envelope("funny", gif_url="https://example.com/cat.gif")
        assert env["media"]["gif_url"] == "https://example.com/cat.gif"

    def test_audio_with_default_mime(self):
        """Audio defaults to audio/aiff when no mime is given."""
        env = build_envelope("listen", audio_b64="base64data")
        assert env["media"]["audio_b64"] == "base64data"
        assert env["media"]["audio_mime"] == "audio/aiff"

    def test_audio_with_custom_mime(self):
        """Audio uses the provided MIME type."""
        env = build_envelope("listen", audio_b64="data", audio_mime="audio/mp3")
        assert env["media"]["audio_mime"] == "audio/mp3"

    def test_image_with_default_mime(self):
        """Image defaults to image/png."""
        env = build_envelope("look", image_b64="imgdata")
        assert env["media"]["image_b64"] == "imgdata"
        assert env["media"]["image_mime"] == "image/png"

    def test_image_with_custom_mime(self):
        """Image uses the provided MIME type."""
        env = build_envelope("look", image_b64="data", image_mime="image/jpeg")
        assert env["media"]["image_mime"] == "image/jpeg"

    def test_sticker(self):
        """Sticker base64 is included in media."""
        env = build_envelope("sticker", sticker_b64="stkdata")
        assert env["media"]["sticker_b64"] == "stkdata"

    def test_document_with_defaults(self):
        """Document defaults to application/octet-stream."""
        env = build_envelope("doc", document_b64="docdata")
        assert env["media"]["document_b64"] == "docdata"
        assert env["media"]["document_mime"] == "application/octet-stream"
        assert "document_name" not in env["media"]

    def test_document_with_name_and_mime(self):
        """Document with explicit name and MIME type."""
        env = build_envelope(
            "doc", document_b64="docdata",
            document_mime="application/pdf", document_name="report.pdf",
        )
        assert env["media"]["document_mime"] == "application/pdf"
        assert env["media"]["document_name"] == "report.pdf"

    def test_location(self):
        """Location dict is included."""
        loc = {"latitude": 32.0, "longitude": 34.7, "name": "Tel Aviv"}
        env = build_envelope("here", location=loc)
        assert env["location"] == loc

    def test_contacts(self):
        """Contact cards are included."""
        contacts = [{"name": "Alice", "phone": "+123"}]
        env = build_envelope("contact", contacts=contacts)
        assert env["contacts"] == contacts

    def test_poll(self):
        """Poll is included."""
        poll = {"name": "Q?", "options": ["A", "B"], "selectable_count": 1}
        env = build_envelope("vote", poll=poll)
        assert env["poll"] == poll

    def test_card(self):
        """Card is included."""
        card = {"title": "T", "body": "B", "footer": "F"}
        env = build_envelope("card", card=card)
        assert env["card"] == card

    def test_actions(self):
        """Action buttons are included."""
        actions = [{"label": "Yes", "value": "yes"}]
        env = build_envelope("choose", actions=actions)
        assert env["actions"] == actions

    def test_reactions(self):
        """Reactions are included."""
        env = build_envelope("react", reactions=["thumbs_up"])
        assert env["reactions"] == ["thumbs_up"]

    def test_react_to(self):
        """React-to message key is included."""
        env = build_envelope("react", react_to="msg_123")
        assert env["react_to"] == "msg_123"

    def test_no_media_when_all_empty(self):
        """No media key when all media fields are empty strings."""
        env = build_envelope("plain", gif_url="", audio_b64="", image_b64="")
        assert "media" not in env

    def test_none_optional_fields_excluded(self):
        """None values for optional fields are not included."""
        env = build_envelope("test", location=None, contacts=None, poll=None)
        assert "location" not in env
        assert "contacts" not in env
        assert "poll" not in env

    def test_empty_list_optional_fields_excluded(self):
        """Empty lists for optional fields are not included."""
        env = build_envelope("test", actions=[], reactions=[])
        assert "actions" not in env
        assert "reactions" not in env

    def test_multiple_media_types(self):
        """Multiple media types coexist in the same envelope."""
        env = build_envelope(
            "mixed", gif_url="https://gif.com/x.gif",
            audio_b64="audiodata", image_b64="imgdata",
        )
        assert "gif_url" in env["media"]
        assert "audio_b64" in env["media"]
        assert "image_b64" in env["media"]


# ── merge_nerve_result_into_envelope ─────────────────────────────────────────

class TestMergeNerveResult:
    """Merging nerve results into an existing envelope."""

    def _base_envelope(self) -> dict:
        """Create a base envelope for merge tests."""
        return build_envelope("Hello", tone="casual")

    def test_merge_non_dict_is_noop(self):
        """Non-dict nerve results do not modify the envelope."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, "not a dict")
        assert result == env

    def test_merge_none_is_noop(self):
        """None nerve result does not modify the envelope."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, None)
        assert result == env

    def test_merge_gif_url(self):
        """GIF URL from nerve result is merged into envelope."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"gif_url": "https://gif.com/x.gif"})
        assert result["media"]["gif_url"] == "https://gif.com/x.gif"

    def test_merge_card(self):
        """Card from nerve result replaces envelope card."""
        env = self._base_envelope()
        card = {"title": "T", "body": "B"}
        result = merge_nerve_result_into_envelope(env, {"card": card})
        assert result["card"] == card

    def test_merge_format_hint(self):
        """Format hint is transferred from nerve result."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"format": "gif"})
        assert result["format"] == "gif"

    def test_merge_format_hint_only_for_known_values(self):
        """Format hint is only set for gif, card, or emoji."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"format": "unknown"})
        assert "format" not in result

    def test_merge_audio(self):
        """Audio from nerve result is merged with default MIME."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"audio_b64": "data"})
        assert result["media"]["audio_b64"] == "data"
        assert result["media"]["audio_mime"] == "audio/aiff"

    def test_merge_audio_with_custom_mime(self):
        """Audio MIME is preserved from nerve result."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"audio_b64": "data", "audio_mime": "audio/mp3"})
        assert result["media"]["audio_mime"] == "audio/mp3"

    def test_merge_image_path_preferred_over_b64(self):
        """image_path takes priority over image_b64."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {
            "image_path": "/tmp/img.png",
            "image_b64": "b64data",
        })
        assert result["media"]["image_path"] == "/tmp/img.png"
        assert "image_b64" not in result["media"]

    def test_merge_image_b64_fallback(self):
        """image_b64 is used when image_path is absent."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"image_b64": "b64data"})
        assert result["media"]["image_b64"] == "b64data"
        assert result["media"]["image_mime"] == "image/png"

    def test_merge_sticker(self):
        """Sticker from nerve result is merged."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"sticker_b64": "stk"})
        assert result["media"]["sticker_b64"] == "stk"

    def test_merge_document(self):
        """Document from nerve result is merged with defaults."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"document_b64": "doc"})
        assert result["media"]["document_b64"] == "doc"
        assert result["media"]["document_mime"] == "application/octet-stream"

    def test_merge_document_with_name(self):
        """Document name from nerve result is included."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {
            "document_b64": "doc", "document_name": "report.pdf",
        })
        assert result["media"]["document_name"] == "report.pdf"

    def test_merge_location(self):
        """Location from nerve result is merged."""
        env = self._base_envelope()
        loc = {"latitude": 40.7, "longitude": -74.0, "name": "NYC"}
        result = merge_nerve_result_into_envelope(env, {"location": loc})
        assert result["location"] == loc

    def test_merge_contacts(self):
        """Contacts from nerve result are merged."""
        env = self._base_envelope()
        contacts = [{"name": "Bob", "phone": "+456"}]
        result = merge_nerve_result_into_envelope(env, {"contacts": contacts})
        assert result["contacts"] == contacts

    def test_merge_poll(self):
        """Poll from nerve result is merged."""
        env = self._base_envelope()
        poll = {"name": "Q?", "options": ["X", "Y"]}
        result = merge_nerve_result_into_envelope(env, {"poll": poll})
        assert result["poll"] == poll

    def test_merge_react_to(self):
        """React-to key from nerve result is merged."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"react_to": "msg_abc"})
        assert result["react_to"] == "msg_abc"

    def test_merge_empty_nerve_result(self):
        """Empty nerve result dict does not modify envelope."""
        env = self._base_envelope()
        original_keys = set(env.keys())
        result = merge_nerve_result_into_envelope(env, {})
        assert set(result.keys()) == original_keys

    def test_merge_preserves_existing_media(self):
        """Merging new media does not erase existing media fields."""
        env = build_envelope("mixed", gif_url="https://old.gif")
        result = merge_nerve_result_into_envelope(env, {"audio_b64": "audio"})
        assert result["media"]["gif_url"] == "https://old.gif"
        assert result["media"]["audio_b64"] == "audio"

    def test_merge_returns_same_object(self):
        """Merge modifies and returns the same envelope object."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"gif_url": "https://x.gif"})
        assert result is env
