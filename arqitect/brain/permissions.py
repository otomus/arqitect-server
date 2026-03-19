"""Permission checks for nerve access based on user role and model capability."""

# Nerves/senses that require at least 'user' role (anon cannot use)
_RESTRICTED_FOR_ANON = frozenset({"touch", "code"})

# Nerve roles that require at least 'user' role
_RESTRICTED_ROLES = frozenset({"code"})

# Role hierarchy (higher number = more permissions)
_ROLE_LEVEL = {
    "anon": 0,
    "user": 1,
    "admin": 2,
    "owner": 3,
}

# Model size classes with enough judgment for nerve fabrication and consolidation
_FABRICATION_CAPABLE_SIZES = frozenset({"medium", "large"})


def can_use_nerve(user_role: str, nerve_name: str, nerve_role: str = "tool") -> bool:
    """Check if a user role is permitted to invoke a nerve."""
    level = _ROLE_LEVEL.get(user_role, 0)
    if level >= 2:
        return True
    if level == 0:
        if nerve_name in _RESTRICTED_FOR_ANON:
            return False
        if nerve_role in _RESTRICTED_ROLES:
            return False
    return True


def can_synthesize_nerve(user_role: str) -> bool:
    """Check if a user role is permitted to synthesize (fabricate) new nerves.

    Only identified users (role level >= 1) may create new nerves.
    Anon users and unrecognized roles are denied.

    Args:
        user_role: The user's role string (anon, user, admin, owner).

    Returns:
        True if the role has synthesis permission.
    """
    return _ROLE_LEVEL.get(user_role, 0) >= 1


def can_model_fabricate() -> bool:
    """Check if the current brain model is capable enough to fabricate nerves.

    Only medium and large models have sufficient judgment for synthesis
    and consolidation decisions. Tinylm and small models must not attempt
    these operations.

    Returns:
        True if the brain model's size class is medium or large.
    """
    try:
        from arqitect.brain.adapters import get_model_size_class
        size = get_model_size_class("brain")
        return size in _FABRICATION_CAPABLE_SIZES
    except Exception:
        return False


def get_model_fabrication_message() -> str:
    """Return a message explaining why the model cannot fabricate nerves.

    Returns:
        User-friendly message about model size limitation.
    """
    return "This model is too small to create new capabilities. A medium or large model is required."


def get_restriction_message(nerve_name: str) -> str:
    """Return a user-friendly message explaining why access was denied."""
    if nerve_name == "touch":
        return "File system and OS operations require an authenticated account."
    return f"Access to {nerve_name} requires authentication. Please identify yourself first."


def get_synthesis_restriction_message() -> str:
    """Return a user-friendly message explaining why nerve synthesis was denied.

    Returns:
        Message telling the user to identify themselves before creating nerves.
    """
    return "Creating new capabilities requires an identified account. Please send your email address to get started."
