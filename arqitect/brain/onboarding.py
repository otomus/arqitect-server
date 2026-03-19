"""User onboarding — email verification + profile setup for new users.

Flow:
1. New connector_id arrives, not in user_links → ask for email
2. User provides email → generate 6-digit code, send via email, store in verification_codes
3. User provides code → verify, create_user_with_email (links to existing if email matches)
4. Ask user their name and how they'd like to be addressed
5. Store profile → proceed normally

State machine per (connector, connector_id):
  - No entry in user_links and no verification_codes → state: ASK_EMAIL
  - Entry in verification_codes → state: AWAITING_CODE
  - Entry in user_links but no display_name → state: ASK_NAME
  - Entry in user_links with display_name → state: VERIFIED (skip onboarding)
"""

import random
import smtplib
from email.message import EmailMessage

from arqitect.config.loader import get_secret


# Onboarding states
ASK_EMAIL = "ask_email"
AWAITING_CODE = "awaiting_code"
ASK_NAME = "ask_name"
VERIFIED = "verified"


def get_onboarding_state(cold, connector: str, connector_id: str) -> str:
    """Determine the onboarding state for a connector identity."""
    # Already linked?
    user_id = cold.resolve_user(connector, connector_id)
    if user_id:
        user = cold.get_user(user_id)
        if user and user.get("display_name"):
            return VERIFIED
        return ASK_NAME
    # Has a pending verification?
    with cold._lock:
        row = cold.conn.execute(
            "SELECT 1 FROM verification_codes WHERE connector=? AND connector_id=?",
            (connector, connector_id),
        ).fetchone()
    if row:
        return AWAITING_CODE
    return ASK_EMAIL


def handle_onboarding(cold, connector: str, connector_id: str, user_message: str) -> tuple[str, str]:
    """Handle an onboarding step. Returns (response_message, user_id).

    user_id is empty string until verification completes.
    response_message is the text to send back to the user.
    """
    state = get_onboarding_state(cold, connector, connector_id)

    if state == VERIFIED:
        user_id = cold.resolve_user(connector, connector_id)
        return ("", user_id)  # No onboarding message, proceed normally

    if state == ASK_NAME:
        user_id = cold.resolve_user(connector, connector_id)
        msg = user_message.strip()
        if msg:
            # Parse the name — take whatever they say as their preferred name
            display_name = msg.strip().strip('"').strip("'")
            cold.set_user_display_name(user_id, display_name)
            cold.set_user_fact(user_id, "preferred_name", display_name)
            return (
                f"Nice to meet you, {display_name}! How can I help you today?",
                user_id,
            )
        return (
            "What's your name? And how would you like me to address you?",
            user_id,
        )

    if state == ASK_EMAIL:
        # Check if the message looks like an email
        msg = user_message.strip()
        if "@" in msg and "." in msg.split("@")[-1]:
            # User provided email directly
            return _handle_email_submission(cold, connector, connector_id, msg)
        # Ask for email
        return (
            "Welcome! To get started, please share your email address. "
            "This helps us keep your experience consistent across platforms.",
            "",
        )

    if state == AWAITING_CODE:
        msg = user_message.strip()
        # Check if user is providing a new email instead
        if "@" in msg and "." in msg.split("@")[-1]:
            # They're changing their email, restart with new email
            return _handle_email_submission(cold, connector, connector_id, msg)
        # Try to verify the code
        code = "".join(c for c in msg if c.isdigit())[:6]
        if len(code) == 6:
            email = cold.verify_code(connector, connector_id, code)
            if email:
                user_id = cold.create_user_with_email(email, connector, connector_id)
                user = cold.get_user(user_id)
                if user and user.get("display_name"):
                    name = user["display_name"]
                    return (
                        f"Welcome back, {name}! I've linked this platform to your account.",
                        user_id,
                    )
                return (
                    "Nice to meet you! What's your name?",
                    user_id,
                )
            else:
                return (
                    "That code is invalid or expired. Please check and try again, "
                    "or send your email address to get a new code.",
                    "",
                )
        return (
            "Please enter the 6-digit verification code sent to your email. "
            "Or send your email address again to get a new code.",
            "",
        )

    return ("Something went wrong. Please try again.", "")


def _handle_email_submission(cold, connector, connector_id, email):
    """Process an email submission: generate code, send it, store it."""
    code = f"{random.randint(0, 999999):06d}"
    cold.store_verification_code(connector, connector_id, email, code)

    sent = _send_verification_email(email, code)
    if sent:
        return (
            f"A verification code has been sent to {email}. "
            f"Please enter the 6-digit code to continue.",
            "",
        )
    else:
        return (
            f"I couldn't send an email to {email}. "
            f"Please check the address and try again.",
            "",
        )


def _send_verification_email(to_email: str, code: str) -> bool:
    """Send verification code via SMTP. Configure via environment variables."""
    smtp_host = get_secret("smtp.host", "smtp.gmail.com")
    smtp_port = int(get_secret("smtp.port", 587))
    smtp_user = get_secret("smtp.user", "")
    smtp_pass = get_secret("smtp.password", "")
    smtp_from = get_secret("smtp.from", smtp_user)

    if not smtp_host or not smtp_user:
        print(f"[ONBOARDING] SMTP not configured. Code for {to_email}: {code}")
        return False

    msg = EmailMessage()
    msg["Subject"] = f"Your Arqitect verification code: {code}"
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg.set_content(
        f"Your verification code is: {code}\n\n"
        f"Enter this code in your chat to complete verification.\n"
        f"This code expires in 10 minutes.\n\n"
        f"— Sentient"
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"[ONBOARDING] Verification email sent to {to_email}")
        return True
    except Exception as e:
        print(f"[ONBOARDING] Failed to send email to {to_email}: {e}")
        return False
