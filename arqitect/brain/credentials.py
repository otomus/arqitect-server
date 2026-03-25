"""Secure credential collection — request, store, and retrieve service credentials.

When the brain needs API keys or secrets for an external service (e.g. Kaggle,
Gmail, GitHub), it sends a credential request through the envelope. The client
shows a secure form, the user fills it in, and the credentials are stored in
arqitect.yaml's secrets section — never passed through chat.

Flow:
1. Brain calls ``request_credentials(service, fields, reason)``
2. ``publish_response()`` includes ``request_credentials`` in the envelope
3. Bridge detects the field → routes a secure-form prompt to the client
4. Client submits credentials via ``{"type": "credentials", ...}``
5. Bridge calls ``store_credentials()`` → persists to arqitect.yaml secrets
6. Bridge publishes to ``brain:credentials`` so the brain can resume
7. Brain calls ``get_credential(service, key)`` to read the stored value
"""

from __future__ import annotations

from arqitect.config.loader import get_secret, set_secret


# ── Credential field schema ──────────────────────────────────────────────

def build_credential_request(
    service: str,
    fields: list[dict],
    reason: str = "",
) -> dict:
    """Build a credential request payload for the envelope.

    Args:
        service: Service identifier (e.g. "kaggle", "gmail", "github").
        fields: List of field descriptors, each with:
            - key: field identifier (e.g. "api_key")
            - label: human-readable label (e.g. "API Key")
            - type: "text" or "password" (default "password")
        reason: Why the credentials are needed (shown to user).

    Returns:
        Dict suitable for ``envelope["request_credentials"]``.
    """
    return {
        "service": service,
        "fields": [
            {
                "key": f.get("key", ""),
                "label": f.get("label", f.get("key", "")),
                "type": f.get("type", "password"),
            }
            for f in fields
        ],
        "reason": reason,
    }


# ── Storage ──────────────────────────────────────────────────────────────

def store_credentials(service: str, credentials: dict) -> None:
    """Persist credentials to arqitect.yaml secrets section.

    Each key-value pair is stored under ``secrets.<service>.<key>``.

    Args:
        service: Service identifier (e.g. "kaggle").
        credentials: Mapping of field key → secret value.
    """
    for key, value in credentials.items():
        set_secret(f"{service}.{key}", value)


def get_credential(service: str, key: str, default: str = "") -> str:
    """Retrieve a stored credential.

    Args:
        service: Service identifier (e.g. "kaggle").
        key: Field key (e.g. "api_key").
        default: Fallback if not found.

    Returns:
        The credential value, or default.
    """
    return get_secret(f"{service}.{key}", default)


def has_credentials(service: str, required_keys: list[str]) -> bool:
    """Check whether all required credentials for a service are present.

    Args:
        service: Service identifier.
        required_keys: List of field keys that must have non-empty values.

    Returns:
        True if all required keys have non-empty values.
    """
    return all(get_credential(service, k) for k in required_keys)
