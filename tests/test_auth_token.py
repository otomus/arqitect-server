"""Tests for JWT session token create/decode/refresh."""

import os
import time
from unittest.mock import patch

import pytest

from arqitect.auth.token import (
    create_token,
    decode_token,
    get_jwt_secret,
    should_refresh,
    TOKEN_LIFETIME_SECONDS,
    REFRESH_THRESHOLD_SECONDS,
)

JWT_SECRET = "test-secret-for-unit-tests"


@pytest.fixture(autouse=True)
def _set_jwt_secret():
    """Every test in this file gets a valid JWT secret."""
    with patch.dict(os.environ, {"ARQITECT_JWT_SECRET": JWT_SECRET}):
        yield


class TestGetJwtSecret:
    def test_returns_secret_when_set(self):
        assert get_jwt_secret() == JWT_SECRET

    def test_raises_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ARQITECT_JWT_SECRET", None)
            with pytest.raises(ValueError, match="ARQITECT_JWT_SECRET"):
                get_jwt_secret()


class TestCreateToken:
    def test_produces_string(self):
        token = create_token("uid_1", "a@b.com", "user", "Alice")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_claims_round_trip(self):
        token = create_token("uid_1", "a@b.com", "user", "Alice")
        claims = decode_token(token)
        assert claims is not None
        assert claims["sub"] == "uid_1"
        assert claims["email"] == "a@b.com"
        assert claims["role"] == "user"
        assert claims["name"] == "Alice"

    def test_has_iat_and_exp(self):
        before = int(time.time())
        token = create_token("uid_1", "a@b.com", "user", "Alice")
        claims = decode_token(token)
        assert claims["iat"] >= before
        assert claims["exp"] == claims["iat"] + TOKEN_LIFETIME_SECONDS


class TestDecodeToken:
    def test_valid_token(self):
        token = create_token("uid_1", "a@b.com", "user", "Alice")
        claims = decode_token(token)
        assert claims is not None
        assert claims["sub"] == "uid_1"

    def test_expired_token_returns_none(self):
        from jose import jwt as jose_jwt

        expired_claims = {
            "sub": "uid_1", "email": "a@b.com", "role": "user",
            "name": "Alice", "iat": 1000, "exp": 1001,
        }
        token = jose_jwt.encode(expired_claims, JWT_SECRET, algorithm="HS256")
        assert decode_token(token) is None

    def test_tampered_token_returns_none(self):
        token = create_token("uid_1", "a@b.com", "user", "Alice")
        tampered = token[:-4] + "XXXX"
        assert decode_token(tampered) is None

    def test_garbage_returns_none(self):
        assert decode_token("not.a.jwt") is None
        assert decode_token("") is None

    def test_wrong_secret_returns_none(self):
        from jose import jwt as jose_jwt

        claims = {
            "sub": "uid_1", "email": "a@b.com", "role": "user",
            "name": "Alice", "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
        token = jose_jwt.encode(claims, "wrong-secret", algorithm="HS256")
        assert decode_token(token) is None


class TestShouldRefresh:
    def test_true_when_near_expiry(self):
        claims = {"exp": time.time() + 300}
        assert should_refresh(claims) is True

    def test_false_when_far_from_expiry(self):
        claims = {"exp": time.time() + 7200}
        assert should_refresh(claims) is False

    def test_boundary_at_threshold(self):
        claims = {"exp": time.time() + REFRESH_THRESHOLD_SECONDS - 1}
        assert should_refresh(claims) is True

    def test_already_expired(self):
        claims = {"exp": time.time() - 100}
        assert should_refresh(claims) is True
