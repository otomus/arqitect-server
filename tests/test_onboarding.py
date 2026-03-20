"""Tests for arqitect.brain.onboarding — email verification + profile setup.

Uses the real ColdMemory (via conftest ``mem`` fixture) backed by a temp
SQLite database.  SMTP I/O is the only thing mocked — everything else
runs against the real implementation.
"""

import smtplib
from unittest.mock import patch, MagicMock

import pytest
from dirty_equals import IsStr
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from arqitect.brain.onboarding import (
    ASK_EMAIL,
    ASK_NAME,
    AWAITING_CODE,
    VERIFIED,
    get_onboarding_state,
    handle_onboarding,
    _handle_email_submission,
    _send_verification_email,
)


# ---------------------------------------------------------------------------
# Helpers — seed the real ColdMemory with deterministic state
# ---------------------------------------------------------------------------

def _seed_user(cold, user_id: str, display_name: str = "", email: str = "") -> str:
    """Insert a user row directly for test setup.

    Returns the user_id for convenience.
    """
    with cold._lock:
        cold.conn.execute(
            "INSERT INTO users (user_id, display_name, email) VALUES (?, ?, ?)",
            (user_id, display_name, email),
        )
        cold.conn.commit()
    return user_id


def _link_user(cold, connector: str, connector_id: str, user_id: str) -> None:
    """Create a user_links row directly for test setup."""
    with cold._lock:
        cold.conn.execute(
            "INSERT INTO user_links (connector, connector_id, user_id) VALUES (?, ?, ?)",
            (connector, connector_id, user_id),
        )
        cold.conn.commit()


# ---------------------------------------------------------------------------
# Fixture — extract cold from the conftest ``mem`` fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def cold(mem):
    """Real ColdMemory backed by a temp SQLite database."""
    return mem.cold


# ---------------------------------------------------------------------------
# get_onboarding_state
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGetOnboardingState:
    """Tests for determining the current onboarding state."""

    def test_returns_ask_email_for_unknown_identity(self, cold):
        assert get_onboarding_state(cold, "telegram", "999") == ASK_EMAIL

    def test_returns_awaiting_code_when_verification_pending(self, cold):
        cold.store_verification_code("telegram", "999", "a@b.com", "123456")
        assert get_onboarding_state(cold, "telegram", "999") == AWAITING_CODE

    def test_returns_ask_name_when_linked_but_no_display_name(self, cold):
        _seed_user(cold, "u1", display_name="")
        _link_user(cold, "telegram", "999", "u1")
        assert get_onboarding_state(cold, "telegram", "999") == ASK_NAME

    def test_returns_verified_when_linked_with_display_name(self, cold):
        _seed_user(cold, "u1", display_name="Alice")
        _link_user(cold, "telegram", "999", "u1")
        assert get_onboarding_state(cold, "telegram", "999") == VERIFIED

    def test_different_connectors_have_independent_state(self, cold):
        _seed_user(cold, "u1", display_name="Alice")
        _link_user(cold, "telegram", "999", "u1")
        # Same connector_id on a different connector is not linked
        assert get_onboarding_state(cold, "discord", "999") == ASK_EMAIL
        assert get_onboarding_state(cold, "telegram", "999") == VERIFIED

    @given(
        connector=st.sampled_from(["telegram", "discord", "slack", "whatsapp"]),
        connector_id=st.text(min_size=1, max_size=20),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_unknown_identity_always_returns_ask_email(self, cold, connector, connector_id):
        """Property: any unseen (connector, connector_id) pair yields ASK_EMAIL.

        Safe to reuse fixture: read-only query on empty database.
        """
        assert get_onboarding_state(cold, connector, connector_id) == ASK_EMAIL


# ---------------------------------------------------------------------------
# handle_onboarding — VERIFIED state
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestHandleOnboardingVerified:
    """When the user is fully verified, onboarding is a no-op."""

    def test_returns_empty_message_and_user_id(self, cold):
        _seed_user(cold, "u1", display_name="Bob")
        _link_user(cold, "telegram", "10", "u1")
        msg, uid = handle_onboarding(cold, "telegram", "10", "hello")
        assert msg == ""
        assert uid == "u1"


# ---------------------------------------------------------------------------
# handle_onboarding — ASK_NAME state
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestHandleOnboardingAskName:
    """User is linked but has no display name yet."""

    def test_prompts_for_name_on_empty_message(self, cold):
        _seed_user(cold, "u1", display_name="")
        _link_user(cold, "telegram", "10", "u1")
        msg, uid = handle_onboarding(cold, "telegram", "10", "")
        assert "name" in msg.lower()
        assert uid == "u1"

    def test_prompts_for_name_on_whitespace_only(self, cold):
        _seed_user(cold, "u1", display_name="")
        _link_user(cold, "telegram", "10", "u1")
        msg, uid = handle_onboarding(cold, "telegram", "10", "   ")
        assert "name" in msg.lower()

    def test_sets_display_name_and_returns_greeting(self, cold):
        _seed_user(cold, "u1", display_name="")
        _link_user(cold, "telegram", "10", "u1")
        msg, uid = handle_onboarding(cold, "telegram", "10", "Alice")
        assert "Alice" in msg
        assert uid == "u1"
        user = cold.get_user("u1")
        assert user["display_name"] == "Alice"

    def test_strips_quotes_from_name(self, cold):
        _seed_user(cold, "u1", display_name="")
        _link_user(cold, "telegram", "10", "u1")
        msg, _ = handle_onboarding(cold, "telegram", "10", '"Bob"')
        assert "Bob" in msg
        user = cold.get_user("u1")
        assert user["display_name"] == "Bob"

    def test_stores_preferred_name_fact(self, cold):
        _seed_user(cold, "u1", display_name="")
        _link_user(cold, "telegram", "10", "u1")
        handle_onboarding(cold, "telegram", "10", "Charlie")
        row = cold.conn.execute(
            "SELECT value FROM facts WHERE category='user:u1' AND key='preferred_name'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "Charlie"

    @given(name=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_any_nonempty_name_is_stored(self, cold, name):
        """Property: any non-blank name is accepted and persisted.

        Safe to reuse fixture: rows are deleted between iterations.
        """
        _seed_user(cold, "u1", display_name="")
        _link_user(cold, "telegram", "10", "u1")
        msg, uid = handle_onboarding(cold, "telegram", "10", name)
        assert uid == "u1"
        assert msg == IsStr  # some greeting is returned
        # Clean up for next hypothesis example
        cold.conn.execute("DELETE FROM users")
        cold.conn.execute("DELETE FROM user_links")
        cold.conn.commit()


# ---------------------------------------------------------------------------
# handle_onboarding — ASK_EMAIL state
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestHandleOnboardingAskEmail:
    """User is completely new — not linked and no pending code."""

    def test_asks_for_email_on_non_email_message(self, cold):
        msg, uid = handle_onboarding(cold, "telegram", "10", "hi there")
        assert "email" in msg.lower()
        assert uid == ""

    @patch("arqitect.brain.onboarding._send_verification_email", return_value=True)
    def test_sends_code_when_email_provided(self, mock_send, cold):
        msg, uid = handle_onboarding(cold, "telegram", "10", "alice@example.com")
        assert "verification code" in msg.lower()
        assert uid == ""
        mock_send.assert_called_once()

    @patch("arqitect.brain.onboarding._send_verification_email", return_value=False)
    def test_reports_failure_when_email_send_fails(self, mock_send, cold):
        msg, uid = handle_onboarding(cold, "telegram", "10", "bad@example.com")
        assert "couldn't send" in msg.lower()
        assert uid == ""


# ---------------------------------------------------------------------------
# handle_onboarding — AWAITING_CODE state
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestHandleOnboardingAwaitingCode:
    """User has a pending verification code."""

    def test_verifies_correct_code_new_user(self, cold):
        cold.store_verification_code("tg", "1", "a@b.com", "123456")
        msg, uid = handle_onboarding(cold, "tg", "1", "123456")
        assert uid != ""
        assert "name" in msg.lower()

    def test_verifies_correct_code_existing_user_with_name(self, cold):
        # Pre-create a user with the same email and a display name
        _seed_user(cold, "u1", display_name="Alice", email="a@b.com")
        cold.store_verification_code("tg", "1", "a@b.com", "123456")
        msg, uid = handle_onboarding(cold, "tg", "1", "123456")
        assert uid == "u1"
        assert "Welcome back" in msg

    def test_rejects_wrong_code(self, cold):
        cold.store_verification_code("tg", "1", "a@b.com", "123456")
        msg, uid = handle_onboarding(cold, "tg", "1", "000000")
        assert "invalid" in msg.lower() or "expired" in msg.lower()
        assert uid == ""

    def test_extracts_digits_from_noisy_input(self, cold):
        cold.store_verification_code("tg", "1", "a@b.com", "123456")
        # Code embedded in text — only first 6 digits extracted
        msg, uid = handle_onboarding(cold, "tg", "1", "my code is 123456!")
        assert uid != ""

    def test_prompts_for_code_on_short_digit_string(self, cold):
        cold.store_verification_code("tg", "1", "a@b.com", "123456")
        msg, uid = handle_onboarding(cold, "tg", "1", "12345")
        assert "6-digit" in msg
        assert uid == ""

    def test_prompts_for_code_on_non_digit_input(self, cold):
        cold.store_verification_code("tg", "1", "a@b.com", "123456")
        msg, uid = handle_onboarding(cold, "tg", "1", "hello")
        assert "6-digit" in msg
        assert uid == ""

    @patch("arqitect.brain.onboarding._send_verification_email", return_value=True)
    def test_allows_email_change_during_awaiting_code(self, mock_send, cold):
        cold.store_verification_code("tg", "1", "old@b.com", "111111")
        msg, uid = handle_onboarding(cold, "tg", "1", "new@b.com")
        assert "verification code" in msg.lower()
        assert uid == ""

    @given(wrong_code=st.from_regex(r"[0-9]{6}", fullmatch=True))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_random_wrong_codes_are_rejected(self, cold, wrong_code):
        """Property: any 6-digit code that isn't the stored one is rejected.

        Safe to reuse fixture: verification_codes are deleted between iterations.
        """
        stored_code = "999999"
        assume(wrong_code != stored_code)
        cold.store_verification_code("tg", "1", "a@b.com", stored_code)
        msg, uid = handle_onboarding(cold, "tg", "1", wrong_code)
        assert uid == ""
        # Clean up for next hypothesis example
        cold.conn.execute("DELETE FROM verification_codes")
        cold.conn.commit()


# ---------------------------------------------------------------------------
# _handle_email_submission
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestHandleEmailSubmission:
    """Tests for the internal email submission helper."""

    @patch("arqitect.brain.onboarding._send_verification_email", return_value=True)
    def test_stores_code_and_reports_success(self, mock_send, cold):
        msg, uid = _handle_email_submission(cold, "tg", "1", "x@y.com")
        assert "verification code" in msg.lower()
        assert "x@y.com" in msg
        assert uid == ""
        # Code was stored
        row = cold.conn.execute(
            "SELECT code FROM verification_codes WHERE connector='tg' AND connector_id='1'"
        ).fetchone()
        assert row is not None
        assert len(row["code"]) == 6

    @patch("arqitect.brain.onboarding._send_verification_email", return_value=False)
    def test_reports_send_failure(self, mock_send, cold):
        msg, uid = _handle_email_submission(cold, "tg", "1", "x@y.com")
        assert "couldn't send" in msg.lower()
        assert uid == ""


# ---------------------------------------------------------------------------
# _send_verification_email
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestSendVerificationEmail:
    """Tests for SMTP email sending (all SMTP I/O is mocked)."""

    @patch("arqitect.brain.onboarding.get_secret")
    def test_returns_false_when_smtp_not_configured(self, mock_secret):
        def secret_side_effect(key, default=""):
            if key == "smtp.port":
                return 587
            return ""
        mock_secret.side_effect = secret_side_effect
        assert _send_verification_email("a@b.com", "123456") is False

    @patch("arqitect.brain.onboarding.smtplib.SMTP")
    @patch("arqitect.brain.onboarding.get_secret")
    def test_returns_true_on_successful_send(self, mock_secret, mock_smtp_cls):
        def secret_side_effect(key, default=""):
            return {
                "smtp.host": "smtp.test.com",
                "smtp.port": "587",
                "smtp.user": "user@test.com",
                "smtp.password": "pass",
                "smtp.from": "noreply@test.com",
            }.get(key, default)

        mock_secret.side_effect = secret_side_effect
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        assert _send_verification_email("a@b.com", "123456") is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()

    @patch("arqitect.brain.onboarding.smtplib.SMTP")
    @patch("arqitect.brain.onboarding.get_secret")
    def test_returns_false_on_smtp_exception(self, mock_secret, mock_smtp_cls):
        def secret_side_effect(key, default=""):
            return {
                "smtp.host": "smtp.test.com",
                "smtp.port": "587",
                "smtp.user": "user@test.com",
                "smtp.password": "pass",
                "smtp.from": "noreply@test.com",
            }.get(key, default)

        mock_secret.side_effect = secret_side_effect
        mock_smtp_cls.return_value.__enter__ = MagicMock(
            side_effect=smtplib.SMTPException("connection refused")
        )
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        assert _send_verification_email("a@b.com", "123456") is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestOnboardingEdgeCases:
    """Boundary and edge-case coverage."""

    def test_email_validation_requires_at_and_dot(self, cold):
        # "nodot@example" has no dot after @ — treated as non-email
        msg, _ = handle_onboarding(cold, "tg", "1", "nodot@example")
        assert "email" in msg.lower()

    def test_email_validation_accepts_subdomain(self, cold):
        with patch("arqitect.brain.onboarding._send_verification_email", return_value=True):
            msg, _ = handle_onboarding(cold, "tg", "1", "user@sub.example.com")
            assert "verification code" in msg.lower()

    def test_state_constants_are_distinct(self):
        states = {ASK_EMAIL, AWAITING_CODE, ASK_NAME, VERIFIED}
        assert len(states) == 4

    @given(
        text=st.text(min_size=0, max_size=100).filter(
            lambda s: "@" not in s or "." not in s.split("@")[-1]
        ),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_non_email_text_never_triggers_verification(self, cold, text):
        """Property: text that doesn't match the email pattern never sends a code.

        Safe to reuse fixture: read-only query on empty database.
        """
        msg, uid = handle_onboarding(cold, "tg", "1", text)
        assert uid == ""
        # Should either ask for email or prompt for 6-digit code (never "verification code sent")
        assert "verification code has been sent" not in msg.lower()
