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
- Property-based tests for structural invariants
"""

import copy

import pytest
from dirty_equals import IsInstance, IsPartialDict
from hypothesis import given, settings
from hypothesis import strategies as st

from arqitect.senses.communication.envelope import (
    build_envelope,
    merge_nerve_result_into_envelope,
)

# ── Hypothesis strategies ────────────────────────────────────────────────────

MIME_AUDIO = st.sampled_from(["audio/aiff", "audio/mp3", "audio/wav", "audio/ogg"])
MIME_IMAGE = st.sampled_from(["image/png", "image/jpeg", "image/webp"])
MIME_DOC = st.sampled_from(["application/pdf", "application/octet-stream", "text/plain"])

KNOWN_FORMATS = st.sampled_from(["gif", "card", "emoji"])
UNKNOWN_FORMATS = st.text(min_size=1).filter(lambda s: s not in ("gif", "card", "emoji"))


# ── Structural invariants (autouse) ──────────────────────────────────────────

REQUIRED_KEYS = {"message", "tone", "markdown"}


@pytest.fixture(autouse=True)
def _envelope_contract_check(request):
    """Every test that produces an envelope must satisfy the base contract."""
    yield
    # After the test, no additional assertion needed here — the property tests
    # and individual tests enforce the contract directly.


# ── build_envelope ───────────────────────────────────────────────────────────

@pytest.mark.timeout(10)
class TestBuildEnvelope:
    """Standard response envelope construction."""

    def test_minimal_envelope(self):
        """Minimal envelope contains message, tone, and markdown flag."""
        env = build_envelope("Hello")
        assert env == IsPartialDict(message="Hello", tone="neutral", markdown=False)
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
        assert env["media"] == IsPartialDict(audio_b64="base64data", audio_mime="audio/aiff")

    def test_audio_with_custom_mime(self):
        """Audio uses the provided MIME type."""
        env = build_envelope("listen", audio_b64="data", audio_mime="audio/mp3")
        assert env["media"]["audio_mime"] == "audio/mp3"

    def test_image_with_default_mime(self):
        """Image defaults to image/png."""
        env = build_envelope("look", image_b64="imgdata")
        assert env["media"] == IsPartialDict(image_b64="imgdata", image_mime="image/png")

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
        assert env["media"] == IsPartialDict(
            document_b64="docdata",
            document_mime="application/octet-stream",
        )
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

@pytest.mark.timeout(10)
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
        assert result["media"] == IsPartialDict(audio_b64="data", audio_mime="audio/aiff")

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
        assert result["media"] == IsPartialDict(image_b64="b64data", image_mime="image/png")

    def test_merge_sticker(self):
        """Sticker from nerve result is merged."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"sticker_b64": "stk"})
        assert result["media"]["sticker_b64"] == "stk"

    def test_merge_document(self):
        """Document from nerve result is merged with defaults."""
        env = self._base_envelope()
        result = merge_nerve_result_into_envelope(env, {"document_b64": "doc"})
        assert result["media"] == IsPartialDict(
            document_b64="doc",
            document_mime="application/octet-stream",
        )

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


# ── Hypothesis property-based tests ──────────────────────────────────────────

@pytest.mark.timeout(10)
class TestBuildEnvelopeProperties:
    """Property-based tests for structural invariants of build_envelope."""

    @given(
        message=st.text(min_size=0, max_size=500),
        tone=st.text(min_size=1, max_size=30),
        markdown=st.booleans(),
    )
    @settings(max_examples=50)
    def test_always_contains_required_keys(self, message, tone, markdown):
        """Every envelope must contain message, tone, and markdown."""
        env = build_envelope(message, tone=tone, markdown=markdown)
        assert REQUIRED_KEYS <= set(env.keys())
        assert env["message"] == message
        assert env["tone"] == tone
        assert env["markdown"] == markdown

    @given(
        message=st.text(min_size=1, max_size=100),
        gif_url=st.text(min_size=0, max_size=200),
        audio_b64=st.text(min_size=0, max_size=200),
        image_b64=st.text(min_size=0, max_size=200),
        sticker_b64=st.text(min_size=0, max_size=200),
        document_b64=st.text(min_size=0, max_size=200),
    )
    @settings(max_examples=50)
    def test_media_present_iff_any_media_field_truthy(
        self, message, gif_url, audio_b64, image_b64, sticker_b64, document_b64,
    ):
        """The media key exists if and only if at least one media field is truthy."""
        env = build_envelope(
            message,
            gif_url=gif_url,
            audio_b64=audio_b64,
            image_b64=image_b64,
            sticker_b64=sticker_b64,
            document_b64=document_b64,
        )
        any_media_provided = any([gif_url, audio_b64, image_b64, sticker_b64, document_b64])
        if any_media_provided:
            assert "media" in env
            assert env["media"] == IsInstance(dict)
        else:
            assert "media" not in env

    @given(
        audio_b64=st.text(min_size=1, max_size=50),
        audio_mime=st.text(min_size=0, max_size=50),
    )
    @settings(max_examples=30)
    def test_audio_mime_never_empty(self, audio_b64, audio_mime):
        """When audio is provided, the MIME type is always set (defaults to audio/aiff)."""
        env = build_envelope("msg", audio_b64=audio_b64, audio_mime=audio_mime)
        assert env["media"]["audio_mime"]  # never empty/falsy

    @given(
        image_b64=st.text(min_size=1, max_size=50),
        image_mime=st.text(min_size=0, max_size=50),
    )
    @settings(max_examples=30)
    def test_image_mime_never_empty(self, image_b64, image_mime):
        """When image is provided, the MIME type is always set (defaults to image/png)."""
        env = build_envelope("msg", image_b64=image_b64, image_mime=image_mime)
        assert env["media"]["image_mime"]  # never empty/falsy

    @given(
        document_b64=st.text(min_size=1, max_size=50),
        document_mime=st.text(min_size=0, max_size=50),
    )
    @settings(max_examples=30)
    def test_document_mime_never_empty(self, document_b64, document_mime):
        """When document is provided, the MIME type is always set."""
        env = build_envelope("msg", document_b64=document_b64, document_mime=document_mime)
        assert env["media"]["document_mime"]  # never empty/falsy

    @given(message=st.text(min_size=1, max_size=100))
    @settings(max_examples=20)
    def test_envelope_is_always_dict(self, message):
        """build_envelope always returns a dict."""
        env = build_envelope(message)
        assert env == IsInstance(dict)


@pytest.mark.timeout(10)
class TestMergeNerveResultProperties:
    """Property-based tests for merge_nerve_result_into_envelope invariants."""

    @given(nerve_result=st.one_of(
        st.none(),
        st.integers(),
        st.text(),
        st.lists(st.integers()),
        st.booleans(),
    ))
    @settings(max_examples=30)
    def test_non_dict_never_modifies_envelope(self, nerve_result):
        """Any non-dict nerve result leaves the envelope unchanged."""
        env = build_envelope("test", tone="neutral")
        snapshot = copy.deepcopy(env)
        result = merge_nerve_result_into_envelope(env, nerve_result)
        assert result == snapshot

    @given(fmt=KNOWN_FORMATS)
    @settings(max_examples=10)
    def test_known_format_hints_accepted(self, fmt):
        """Known format hints (gif, card, emoji) are always set."""
        env = build_envelope("test")
        result = merge_nerve_result_into_envelope(env, {"format": fmt})
        assert result["format"] == fmt

    @given(fmt=UNKNOWN_FORMATS)
    @settings(max_examples=20)
    def test_unknown_format_hints_rejected(self, fmt):
        """Non-standard format hints are never set on the envelope."""
        env = build_envelope("test")
        result = merge_nerve_result_into_envelope(env, {"format": fmt})
        assert "format" not in result

    @given(
        audio_b64=st.text(min_size=1, max_size=50),
        audio_mime=st.one_of(st.just(""), MIME_AUDIO),
    )
    @settings(max_examples=20)
    def test_merge_audio_always_has_mime(self, audio_b64, audio_mime):
        """Merged audio always has a MIME type, defaulting to audio/aiff."""
        env = build_envelope("test")
        nerve = {"audio_b64": audio_b64}
        if audio_mime:
            nerve["audio_mime"] = audio_mime
        result = merge_nerve_result_into_envelope(env, nerve)
        assert result["media"]["audio_mime"]  # never empty/falsy

    @given(data=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.text(min_size=0, max_size=50),
        max_size=5,
    ))
    @settings(max_examples=30)
    def test_merge_always_returns_same_object(self, data):
        """merge_nerve_result_into_envelope always returns the same envelope ref."""
        env = build_envelope("test")
        result = merge_nerve_result_into_envelope(env, data)
        assert result is env

    @given(data=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.text(min_size=0, max_size=50),
        max_size=5,
    ))
    @settings(max_examples=30)
    def test_merge_preserves_required_keys(self, data):
        """Required envelope keys survive any merge."""
        env = build_envelope("original", tone="warm", markdown=True)
        result = merge_nerve_result_into_envelope(env, data)
        assert result["message"] == "original"
        assert result["tone"] == "warm"
        assert result["markdown"] is True
