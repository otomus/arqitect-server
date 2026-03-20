"""Tests for permission checks -- nerve access and synthesis gating.

Covers:
- can_use_nerve access control by role
- can_synthesize_nerve role gating
- Restriction messages for denied access
"""

import pytest
from hypothesis import given, settings, strategies as st

from arqitect.brain.permissions import (
    can_use_nerve,
    can_synthesize_nerve,
    get_restriction_message,
    get_synthesis_restriction_message,
)


# ---------------------------------------------------------------------------
# Existing: can_use_nerve (sanity -- ensure no regression)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
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

    @given(
        nerve_name=st.from_regex(r"[a-z_]{3,15}_nerve", fullmatch=True),
    )
    @settings(max_examples=20)
    def test_admin_can_always_use_any_nerve(self, nerve_name):
        """Admin role must have unrestricted access to all nerves."""
        assert can_use_nerve("admin", nerve_name) is True

    @given(
        nerve_name=st.from_regex(r"[a-z_]{3,15}_nerve", fullmatch=True),
    )
    @settings(max_examples=20)
    def test_anon_can_use_non_restricted_nerves(self, nerve_name):
        """Anon users can use any nerve that is not in the restricted set."""
        assert can_use_nerve("anon", nerve_name) is True


# ---------------------------------------------------------------------------
# New: can_synthesize_nerve -- anon cannot fabricate nerves
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
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

    @given(
        role=st.sampled_from(["user", "admin", "owner"]),
    )
    @settings(max_examples=10)
    def test_identified_roles_can_always_synthesize(self, role):
        """All identified roles must be allowed to synthesize."""
        assert can_synthesize_nerve(role) is True

    @given(
        role=st.from_regex(r"[a-z]{5,15}", fullmatch=True).filter(
            lambda r: r not in ("user", "admin", "owner")
        ),
    )
    @settings(max_examples=15)
    def test_unknown_roles_cannot_synthesize(self, role):
        """Roles not in the known hierarchy are denied synthesis."""
        assert can_synthesize_nerve(role) is False


# ---------------------------------------------------------------------------
# Restriction messages
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestRestrictionMessage:
    """Denial messages must be informative."""

    def test_touch_restriction_mentions_file_system(self):
        msg = get_restriction_message("touch")
        assert "file system" in msg.lower() or "authenticated" in msg.lower()

    def test_generic_restriction_mentions_nerve_name(self):
        msg = get_restriction_message("custom_nerve")
        assert "custom_nerve" in msg


@pytest.mark.timeout(10)
class TestSynthesisRestrictionMessage:
    """Denial message must tell the user how to unblock themselves."""

    def test_message_is_non_empty(self):
        msg = get_synthesis_restriction_message()
        assert len(msg) > 0

    def test_message_asks_for_email(self):
        """Denial message must tell the user exactly what to do: send their email."""
        msg = get_synthesis_restriction_message()
        assert "email" in msg.lower()
