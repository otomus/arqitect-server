"""Tests for arqitect.brain.onboarding — email verification + profile setup."""

import smtplib
import sqlite3
import threading
from unittest.mock import patch, MagicMock

import pytest

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
# In-memory ColdMemory stand-in (uses real SQLite, no mocking)
# ---------------------------------------------------------------------------

class InMemoryCold:
    """Minimal ColdMemory backed by an in-memory SQLite database.

    Implements the exact methods that onboarding.py calls on cold, using the
    same SQL schema as the real ColdMemory class.
    """

    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT DEFAULT '',
                email TEXT DEFAULT '',
                role TEXT DEFAULT 'user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                secrets TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS user_links (
                connector TEXT NOT NULL,
                connector_id TEXT NOT NULL,
                user_id TEXT NOT NULL REFERENCES users(user_id),
                linked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (connector, connector_id)
            );
            CREATE TABLE IF NOT EXISTS verification_codes (
                connector TEXT NOT NULL,
                connector_id TEXT NOT NULL,
                email TEXT NOT NULL,
                code TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (connector, connector_id)
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                UNIQUE(category, key)
            );
        """)

    # ── Methods called by onboarding.py ──────────────────────────────────

    def resolve_user(self, connector: str, connector_id: str) -> str:
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id FROM user_links WHERE connector=? AND connector_id=?",
                (connector, connector_id),
            ).fetchone()
        return row["user_id"] if row else ""

    def get_user(self, user_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM users WHERE user_id=?", (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_user_with_email(self, email: str, connector: str, connector_id: str) -> str:
        import uuid
        email = email.lower().strip()
        with self._lock:
            existing = self.conn.execute(
                "SELECT user_id FROM users WHERE email=?", (email,),
            ).fetchone()
            if existing:
                user_id = existing["user_id"]
                self.conn.execute(
                    "INSERT INTO user_links (connector, connector_id, user_id) VALUES (?, ?, ?) "
                    "ON CONFLICT(connector, connector_id) DO UPDATE SET user_id=excluded.user_id",
                    (connector, connector_id, user_id),
                )
                self.conn.commit()
                return user_id
            user_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO users (user_id, email) VALUES (?, ?)", (user_id, email),
            )
            self.conn.execute(
                "INSERT INTO user_links (connector, connector_id, user_id) VALUES (?, ?, ?)",
                (connector, connector_id, user_id),
            )
            self.conn.commit()
            return user_id

    def set_user_display_name(self, user_id: str, name: str):
        with self._lock:
            self.conn.execute(
                "UPDATE users SET display_name=? WHERE user_id=?", (name, user_id),
            )
            self.conn.commit()

    def set_user_fact(self, user_id: str, key: str, value: str):
        category = f"user:{user_id}"
        with self._lock:
            self.conn.execute(
                "INSERT INTO facts (category, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(category, key) DO UPDATE SET value=excluded.value",
                (category, key, value),
            )
            self.conn.commit()

    def store_verification_code(self, connector: str, connector_id: str, email: str, code: str):
        with self._lock:
            self.conn.execute(
                "INSERT INTO verification_codes (connector, connector_id, email, code) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(connector, connector_id) DO UPDATE SET email=excluded.email, code=excluded.code, "
                "created_at=CURRENT_TIMESTAMP",
                (connector, connector_id, email.lower().strip(), code),
            )
            self.conn.commit()

    def verify_code(self, connector: str, connector_id: str, code: str) -> str | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT email, code FROM verification_codes WHERE connector=? AND connector_id=?",
                (connector, connector_id),
            ).fetchone()
            if not row or row["code"] != code.strip():
                return None
            email = row["email"]
            self.conn.execute(
                "DELETE FROM verification_codes WHERE connector=? AND connector_id=?",
                (connector, connector_id),
            )
            self.conn.commit()
            return email

    # ── Helpers for test setup ───────────────────────────────────────────

    def _insert_user(self, user_id: str, display_name: str = "", email: str = ""):
        self.conn.execute(
            "INSERT INTO users (user_id, display_name, email) VALUES (?, ?, ?)",
            (user_id, display_name, email),
        )
        self.conn.commit()

    def _link_user(self, connector: str, connector_id: str, user_id: str):
        self.conn.execute(
            "INSERT INTO user_links (connector, connector_id, user_id) VALUES (?, ?, ?)",
            (connector, connector_id, user_id),
        )
        self.conn.commit()


@pytest.fixture()
def cold():
    """Fresh in-memory cold storage for each test."""
    return InMemoryCold()


# ---------------------------------------------------------------------------
# get_onboarding_state
# ---------------------------------------------------------------------------

class TestGetOnboardingState:
    """Tests for determining the current onboarding state."""

    def test_returns_ask_email_for_unknown_identity(self, cold):
        assert get_onboarding_state(cold, "telegram", "999") == ASK_EMAIL

    def test_returns_awaiting_code_when_verification_pending(self, cold):
        cold.store_verification_code("telegram", "999", "a@b.com", "123456")
        assert get_onboarding_state(cold, "telegram", "999") == AWAITING_CODE

    def test_returns_ask_name_when_linked_but_no_display_name(self, cold):
        cold._insert_user("u1", display_name="")
        cold._link_user("telegram", "999", "u1")
        assert get_onboarding_state(cold, "telegram", "999") == ASK_NAME

    def test_returns_verified_when_linked_with_display_name(self, cold):
        cold._insert_user("u1", display_name="Alice")
        cold._link_user("telegram", "999", "u1")
        assert get_onboarding_state(cold, "telegram", "999") == VERIFIED

    def test_different_connectors_have_independent_state(self, cold):
        cold._insert_user("u1", display_name="Alice")
        cold._link_user("telegram", "999", "u1")
        # Same connector_id on a different connector is not linked
        assert get_onboarding_state(cold, "discord", "999") == ASK_EMAIL
        assert get_onboarding_state(cold, "telegram", "999") == VERIFIED


# ---------------------------------------------------------------------------
# handle_onboarding — VERIFIED state
# ---------------------------------------------------------------------------

class TestHandleOnboardingVerified:
    """When the user is fully verified, onboarding is a no-op."""

    def test_returns_empty_message_and_user_id(self, cold):
        cold._insert_user("u1", display_name="Bob")
        cold._link_user("telegram", "10", "u1")
        msg, uid = handle_onboarding(cold, "telegram", "10", "hello")
        assert msg == ""
        assert uid == "u1"


# ---------------------------------------------------------------------------
# handle_onboarding — ASK_NAME state
# ---------------------------------------------------------------------------

class TestHandleOnboardingAskName:
    """User is linked but has no display name yet."""

    def test_prompts_for_name_on_empty_message(self, cold):
        cold._insert_user("u1", display_name="")
        cold._link_user("telegram", "10", "u1")
        msg, uid = handle_onboarding(cold, "telegram", "10", "")
        assert "name" in msg.lower()
        assert uid == "u1"

    def test_prompts_for_name_on_whitespace_only(self, cold):
        cold._insert_user("u1", display_name="")
        cold._link_user("telegram", "10", "u1")
        msg, uid = handle_onboarding(cold, "telegram", "10", "   ")
        assert "name" in msg.lower()

    def test_sets_display_name_and_returns_greeting(self, cold):
        cold._insert_user("u1", display_name="")
        cold._link_user("telegram", "10", "u1")
        msg, uid = handle_onboarding(cold, "telegram", "10", "Alice")
        assert "Alice" in msg
        assert uid == "u1"
        user = cold.get_user("u1")
        assert user["display_name"] == "Alice"

    def test_strips_quotes_from_name(self, cold):
        cold._insert_user("u1", display_name="")
        cold._link_user("telegram", "10", "u1")
        msg, _ = handle_onboarding(cold, "telegram", "10", '"Bob"')
        assert "Bob" in msg
        user = cold.get_user("u1")
        assert user["display_name"] == "Bob"

    def test_stores_preferred_name_fact(self, cold):
        cold._insert_user("u1", display_name="")
        cold._link_user("telegram", "10", "u1")
        handle_onboarding(cold, "telegram", "10", "Charlie")
        row = cold.conn.execute(
            "SELECT value FROM facts WHERE category='user:u1' AND key='preferred_name'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "Charlie"


# ---------------------------------------------------------------------------
# handle_onboarding — ASK_EMAIL state
# ---------------------------------------------------------------------------

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

class TestHandleOnboardingAwaitingCode:
    """User has a pending verification code."""

    def test_verifies_correct_code_new_user(self, cold):
        cold.store_verification_code("tg", "1", "a@b.com", "123456")
        msg, uid = handle_onboarding(cold, "tg", "1", "123456")
        assert uid != ""
        assert "name" in msg.lower()

    def test_verifies_correct_code_existing_user_with_name(self, cold):
        # Pre-create a user with the same email and a display name
        cold._insert_user("u1", display_name="Alice", email="a@b.com")
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


# ---------------------------------------------------------------------------
# _handle_email_submission
# ---------------------------------------------------------------------------

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
