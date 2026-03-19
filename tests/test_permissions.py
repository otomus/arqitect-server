"""Tests for permission checks — nerve access and synthesis gating.

TDD: these tests define the expected behavior for synthesis permission
enforcement. Anon users can use existing nerves but cannot fabricate new ones.
"""

import pytest

from arqitect.brain.permissions import (
    can_use_nerve,
    can_synthesize_nerve,
    get_restriction_message,
    get_synthesis_restriction_message,
)


# ---------------------------------------------------------------------------
# Existing: can_use_nerve (sanity — ensure no regression)
# ---------------------------------------------------------------------------

class TestCanUseNerve:
    """Existing nerve access checks must continue to work."""

    def test_anon_can_use_regular_nerve(self):
        assert can_use_nerve("anon", "joke_nerve") is True

    def test_anon_cannot_use_touch(self):
        assert can_use_nerve("anon", "touch") is False

    def test_user_can_use_touch(self):
        assert can_use_nerve("user", "touch") is True

    def test_admin_can_use_anything(self):
        assert can_use_nerve("admin", "touch") is True
        assert can_use_nerve("admin", "code", "code") is True


# ---------------------------------------------------------------------------
# New: can_synthesize_nerve — anon cannot fabricate nerves
# ---------------------------------------------------------------------------

class TestCanSynthesizeNerve:
    """Only identified users (role >= 'user') may synthesize new nerves."""

    def test_anon_cannot_synthesize(self):
        assert can_synthesize_nerve("anon") is False

    def test_registered_user_can_synthesize(self):
        assert can_synthesize_nerve("user") is True

    def test_admin_can_synthesize(self):
        assert can_synthesize_nerve("admin") is True

    def test_owner_can_synthesize(self):
        assert can_synthesize_nerve("owner") is True

    def test_unknown_role_cannot_synthesize(self):
        """Empty or unrecognized role defaults to denied."""
        assert can_synthesize_nerve("") is False

    def test_garbage_role_cannot_synthesize(self):
        assert can_synthesize_nerve("superuser") is False


# ---------------------------------------------------------------------------
# Restriction messages
# ---------------------------------------------------------------------------

class TestSynthesisRestrictionMessage:
    """Denial message must tell the user how to unblock themselves."""

    def test_message_is_non_empty(self):
        msg = get_synthesis_restriction_message()
        assert len(msg) > 0

    def test_message_asks_for_email(self):
        """Denial message must tell the user exactly what to do: send their email."""
        msg = get_synthesis_restriction_message()
        assert "email" in msg.lower()
