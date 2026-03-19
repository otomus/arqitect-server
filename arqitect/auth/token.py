"""JWT session tokens for bridge (dashboard/SDK) authentication.

The bridge connector uses JWT to track identified users across WebSocket
reconnections. Tokens are minted after onboarding completes and refreshed
with a sliding window before expiry.
"""

import time

from jose import jwt, JWTError

from arqitect.config.loader import get_secret

JWT_ALGORITHM = "HS256"
TOKEN_LIFETIME_SECONDS = 86400
REFRESH_THRESHOLD_SECONDS = 3600


def get_jwt_secret() -> str:
    """Read the JWT signing secret from arqitect.yaml.

    Returns:
        The secret string.

    Raises:
        ValueError: If secrets.jwt_secret is not set.
    """
    secret = get_secret("jwt_secret")
    if not secret:
        raise ValueError(
            "secrets.jwt_secret in arqitect.yaml is required but not set"
        )
    return secret


def create_token(
    user_id: str,
    role: str,
    display_name: str,
) -> str:
    """Mint a signed JWT for a verified bridge user.

    No PII (email, etc.) is stored in the token — only the opaque user ID,
    role, and display name. Email lives server-side in cold memory only.

    Args:
        user_id: Internal user identifier (stored as 'sub').
        role: User role (user, admin, owner).
        display_name: User's preferred display name.

    Returns:
        Encoded JWT string.
    """
    now = int(time.time())
    claims = {
        "sub": user_id,
        "role": role,
        "name": display_name,
        "iat": now,
        "exp": now + TOKEN_LIFETIME_SECONDS,
    }
    return jwt.encode(claims, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT.

    Args:
        token: Encoded JWT string.

    Returns:
        Claims dict if valid, None on any error (expired, tampered, garbage).
    """
    try:
        return jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
    except (JWTError, ValueError):
        return None


def should_refresh(claims: dict) -> bool:
    """Check whether a token is close enough to expiry to warrant a refresh.

    Args:
        claims: Decoded JWT claims dict (must contain 'exp').

    Returns:
        True if the token expires within REFRESH_THRESHOLD_SECONDS.
    """
    exp = claims.get("exp", 0)
    return (exp - time.time()) < REFRESH_THRESHOLD_SECONDS
