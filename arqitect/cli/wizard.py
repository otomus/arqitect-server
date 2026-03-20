"""Interactive setup wizard for arqitect.yaml configuration.

Walks the user through 10 steps to configure their Arqitect project.
All collected information is saved to arqitect.yaml.
"""

import copy
import secrets
from pathlib import Path
from typing import Any

import yaml

from arqitect.config.defaults import DEFAULTS
from arqitect.config.loader import get_project_root, load_config
from arqitect.inference.providers import PROVIDER_META, get_wizard_providers
from arqitect.types import InferenceRole, Sense


# ── Prompt helpers ───────────────────────────────────────────────────────

def _prompt_text(label: str, default: str = "") -> str:
    """Prompt the user for a text value with an optional default."""
    suffix = f" [{default}]" if default else ""
    value = input(f"  {label}{suffix}: ").strip()
    return value or default


def _prompt_choice(label: str, options: list[str], default: int = 1) -> int:
    """Prompt the user to pick from numbered options. Returns 1-based index."""
    print(f"\n  {label}")
    for i, opt in enumerate(options, 1):
        marker = "→" if i == default else " "
        print(f"    {marker} {i}. {opt}")
    while True:
        raw = input(f"  Choice [{default}]: ").strip()
        if not raw:
            return default
        try:
            choice = int(raw)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def _prompt_yes_no(label: str, default: bool = True) -> bool:
    """Prompt for a yes/no answer."""
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {label} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _prompt_list(label: str, hint: str = "comma-separated, or skip") -> list[str]:
    """Prompt for a comma-separated list of values."""
    raw = input(f"  {label} ({hint}): ").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _generate_jwt_secret() -> str:
    """Generate a cryptographically secure JWT secret."""
    return secrets.token_urlsafe(48)


def _detect_os() -> str:
    """Detect the current operating system.

    Returns 'macos', 'windows', or 'linux'.
    """
    import platform
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Windows":
        return "windows"
    return "linux"


def _open_file_dialog(title: str, initial_dir: str = "", file_type: str = "") -> str:
    """Open a native OS file picker dialog.

    Uses osascript on macOS, PowerShell on Windows, zenity on Linux.
    Returns the selected path or empty string if cancelled.
    """
    import subprocess

    resolved_dir = str(Path(initial_dir).expanduser().resolve()) if initial_dir else ""
    os_type = _detect_os()

    if os_type == "macos":
        # macOS 'choose file of type' expects UTIs, not file extensions.
        # Custom extensions like .gguf have no registered UTI, so the
        # filter silently hides them. Skip the type clause on macOS.
        script = f'set theFile to choose file with prompt "{title}"'
        if resolved_dir:
            script = (
                f'set theFile to choose file with prompt "{title}" '
                f'default location POSIX file "{resolved_dir}"'
            )
        result = subprocess.run(
            ["osascript", "-e", script, "-e", "return POSIX path of theFile"],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    if os_type == "windows":
        ps_filter = f"GGUF files (*.{file_type})|*.{file_type}|" if file_type else ""
        ps_script = (
            "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null;"
            "$dlg = New-Object System.Windows.Forms.OpenFileDialog;"
            f'$dlg.Title = "{title}";'
            f'$dlg.Filter = "{ps_filter}All files (*.*)|*.*";'
        )
        if resolved_dir:
            ps_script += f'$dlg.InitialDirectory = "{resolved_dir}";'
        ps_script += (
            "if ($dlg.ShowDialog() -eq 'OK') { $dlg.FileName } else { '' }"
        )
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    # Linux: zenity
    cmd = ["zenity", "--file-selection", f"--title={title}"]
    if resolved_dir:
        cmd.append(f"--filename={resolved_dir}/")
    if file_type:
        cmd.extend(["--file-filter", f"{file_type} | *.{file_type}"])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout.strip() if result.returncode == 0 else ""


def _open_dir_dialog(title: str, initial_dir: str = "") -> str:
    """Open a native OS directory picker dialog.

    Uses osascript on macOS, PowerShell on Windows, zenity on Linux.
    Returns the selected path or empty string if cancelled.
    """
    import subprocess

    resolved_dir = str(Path(initial_dir).expanduser().resolve()) if initial_dir else ""
    os_type = _detect_os()

    if os_type == "macos":
        script = f'set theFolder to choose folder with prompt "{title}"'
        if resolved_dir:
            script = (
                f'set theFolder to choose folder with prompt "{title}" '
                f'default location POSIX file "{resolved_dir}"'
            )
        result = subprocess.run(
            ["osascript", "-e", script, "-e", "return POSIX path of theFolder"],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip().rstrip("/") if result.returncode == 0 else ""

    if os_type == "windows":
        ps_script = (
            "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null;"
            "$dlg = New-Object System.Windows.Forms.FolderBrowserDialog;"
            f'$dlg.Description = "{title}";'
        )
        if resolved_dir:
            ps_script += f'$dlg.SelectedPath = "{resolved_dir}";'
        ps_script += (
            "if ($dlg.ShowDialog() -eq 'OK') { $dlg.SelectedPath } else { '' }"
        )
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    # Linux: zenity
    cmd = ["zenity", "--file-selection", "--directory", f"--title={title}"]
    if resolved_dir:
        cmd.append(f"--filename={resolved_dir}/")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout.strip() if result.returncode == 0 else ""


def _prompt_gguf_file(initial_dir: str = "") -> str:
    """Open a file picker for .gguf model files."""
    print("  Opening file picker...")
    path = _open_file_dialog(
        title="Select GGUF model file",
        file_type="gguf",
        initial_dir=initial_dir,
    )
    if path:
        print(f"  Selected: {path}")
        return path
    print("  No file selected.")
    return _prompt_text("Model file path")


# ── Data-driven provider config helpers ──────────────────────────────────

def _require_api_key(config: dict, provider_name: str) -> None:
    """Prompt for an API key and refuse to continue without one for cloud providers.

    Reads auth requirements from PROVIDER_META — no hardcoded key map.
    """
    meta = PROVIDER_META.get(provider_name, {})
    if meta.get("auth_type") != "api_key":
        return  # local provider, no key needed

    secret_key = meta["secret_key"]
    label = meta["secret_label"]

    existing = config.get("secrets", {}).get(secret_key, "")
    if existing:
        print(f"  {label}: already configured.")
        return

    while True:
        api_key = _prompt_text(f"{label} (required)")
        if api_key.strip():
            config.setdefault("secrets", {})[secret_key] = api_key.strip()
            return
        print(f"  API key is required for {provider_name}. Cannot skip.")


def _prompt_extra_config(config: dict, provider_name: str, *, before_model: bool) -> None:
    """Prompt for extra config fields defined in PROVIDER_META.

    Args:
        config: Mutable config dict to write values into.
        provider_name: Provider to read extra_config from.
        before_model: If True, only prompt fields with before_model=True.
                      If False, only prompt fields with before_model=False.
    """
    meta = PROVIDER_META.get(provider_name, {})
    for field in meta.get("extra_config", []):
        if field.get("before_model", False) != before_model:
            continue

        store_in = field["store_in"]
        key = field["key"]
        existing = config.get(store_in, {}).get(key, "")
        default = existing or field.get("default", "")

        value = _prompt_text(field["label"], default)
        if value:
            config.setdefault(store_in, {})[key] = value


def find_local_name(models_dir: Path, resolved_target: Path) -> str | None:
    """Find a filename in models_dir that resolves to the same file.

    Checks direct matches and symlinks so that picking a blob file
    returns the symlink name (e.g. model-file.gguf) instead
    of the raw target (sha256-60e05f...).

    Args:
        models_dir: Directory containing model files and symlinks.
        resolved_target: Fully resolved path of the selected file.

    Returns:
        The entry name in models_dir that points to resolved_target,
        or None if no match is found.
    """
    if not models_dir.is_dir():
        return None
    for entry in models_dir.iterdir():
        if entry.resolve() == resolved_target:
            return entry.name
    return None


def resolve_model_selection(
    selected_path: str, models_dir: str
) -> tuple[str, str]:
    """Resolve a file-picker selection into (model_name, models_dir).

    Resolution logic:
    1. If models_dir is set and contains a file (or symlink) that resolves
       to the same target as selected_path, return that entry's name.
    2. Otherwise, set models_dir to the selected file's parent directory
       and return the filename.

    Args:
        selected_path: Absolute path returned by the file picker.
        models_dir: Current models_dir from config (empty string if unset).

    Returns:
        Tuple of (model_name, models_dir) to store in config.
    """
    model_path = Path(selected_path)
    resolved_target = model_path.resolve()

    if models_dir:
        local_name = find_local_name(Path(models_dir), resolved_target)
        if local_name:
            return local_name, models_dir

    return model_path.name, str(model_path.parent)


def _prompt_model_for_provider(config: dict, provider_name: str, role: str = "") -> str:
    """Prompt for a model name/path appropriate to the provider.

    Data-driven: reads model_prompt type and default_model from PROVIDER_META.
    """
    meta = PROVIDER_META.get(provider_name, {})
    role_hint = f" for {role}" if role else ""

    # Prompt extra config that must come before model selection (e.g. base_url)
    _prompt_extra_config(config, provider_name, before_model=True)

    if meta.get("model_prompt") == "file_picker":
        existing_dir = config.get("inference", {}).get("models_dir", "")
        full_path = _prompt_gguf_file(existing_dir or "./models")
        model_name, resolved_dir = resolve_model_selection(full_path, existing_dir)
        config["inference"]["models_dir"] = resolved_dir
        return model_name

    # Text prompt with provider-specific default
    default_model = meta.get("default_model", "")
    model = _prompt_text(f"Model name{role_hint}", default_model)

    # Prompt extra config that comes after model selection
    _prompt_extra_config(config, provider_name, before_model=False)

    return model


# ── Step functions ───────────────────────────────────────────────────────

def _step_environment(config: dict) -> None:
    """Step 1: Deployment environment — drives nerve filtering and touch defaults."""
    print("\n━━━ Step 1: Environment ━━━")

    choice = _prompt_choice(
        "Where will this AI run?",
        [
            "Desktop/Laptop",
            "Server",
            "IoT/Embedded",
        ],
        default=2,
    )
    env_map = {1: "desktop", 2: "server", 3: "iot"}
    config["environment"] = env_map[choice]


def _step_brain_model(config: dict) -> None:
    """Step 2: Brain model selection — per-role provider routing."""
    print("\n━━━ Step 2: Inference Providers ━━━")

    environment = config.get("environment", "server")
    providers = get_wizard_providers(environment)

    config.setdefault("inference", {})
    config["inference"].setdefault("models", {})
    config["inference"].setdefault("roles", {})

    mode = _prompt_choice(
        "How would you like to configure inference?",
        [
            "Same provider for all roles (simpler)",
            "Configure each role separately (brain, nerve, coder, creative, communication)",
        ],
        default=1,
    )

    all_roles = tuple(InferenceRole)

    if mode == 1:
        _configure_provider_uniform(config, providers, all_roles)
    else:
        _configure_provider_per_role(config, providers, all_roles)


def _configure_provider_uniform(config: dict, providers: list, roles: tuple) -> None:
    """Configure a single provider + model for all roles."""
    choice = _prompt_choice(
        "Select inference provider:",
        [label for _, label in providers],
        default=1,
    )
    provider_name = providers[choice - 1][0]
    model = _prompt_model_for_provider(config, provider_name)

    # Require API key for cloud providers
    _require_api_key(config, provider_name)

    config["inference"]["provider"] = provider_name
    for role in roles:
        config["inference"]["roles"][role] = {"provider": provider_name, "model": model}
        config["inference"]["models"][role] = model


def _configure_provider_per_role(config: dict, providers: list, roles: tuple) -> None:
    """Configure provider + model independently for each role."""
    for role in roles:
        print(f"\n  ── {role.upper()} role ──")
        choice = _prompt_choice(
            f"Provider for {role}:",
            [label for _, label in providers],
            default=1,
        )
        provider_name = providers[choice - 1][0]
        model = _prompt_model_for_provider(config, provider_name, role=role)

        # Require API key for cloud providers (idempotent if already set)
        _require_api_key(config, provider_name)

        config["inference"]["roles"][role] = {"provider": provider_name, "model": model}
        config["inference"]["models"][role] = model

    # Use brain's provider as the top-level default
    brain_cfg = config["inference"]["roles"].get(InferenceRole.BRAIN, {})
    config["inference"]["provider"] = brain_cfg.get("provider", "gguf")


def _step_vision(config: dict) -> None:
    """Step 3: Vision sense configuration."""
    print("\n━━━ Step 3: Vision ━━━")

    provider = config.get("inference", {}).get("provider", "gguf")
    options = ["Yes — Local moondream2 (~2GB)"]
    if provider == "anthropic":
        options.insert(0, "Yes — Anthropic Claude Vision (uses brain provider)")
    options.append("Yes — Custom vision endpoint")
    options.append("No — Skip vision")

    sight = Sense.SIGHT
    choice = _prompt_choice("Enable vision sense?", options, default=1)
    config.setdefault("senses", {}).setdefault(sight, {})

    if options[choice - 1].startswith("No"):
        config["senses"][sight]["enabled"] = False
        config["senses"][sight]["provider"] = ""
        return

    config["senses"][sight]["enabled"] = True
    if "moondream" in options[choice - 1].lower():
        config["senses"][sight]["provider"] = "moondream"
        models_dir = config.get("inference", {}).get("models_dir", "./models")
        full_path = _prompt_gguf_file(models_dir)
        vision_name, resolved_dir = resolve_model_selection(full_path, models_dir)
        config["inference"]["models_dir"] = resolved_dir
        config["inference"]["models"]["vision"] = vision_name
    elif "anthropic" in options[choice - 1].lower():
        config["senses"][sight]["provider"] = "anthropic"
    else:
        endpoint = _prompt_text("Vision endpoint URL")
        config["senses"][sight]["provider"] = "custom"
        config["senses"][sight]["endpoint"] = endpoint


def _step_hearing(config: dict) -> None:
    """Step 4: Hearing sense configuration."""
    print("\n━━━ Step 4: Hearing ━━━")

    choice = _prompt_choice(
        "Enable hearing sense?",
        [
            "Yes — OpenAI Whisper (local)",
            "Yes — Cloud STT (Deepgram/other)",
            "No — Text only",
        ],
        default=3,
    )
    hearing = Sense.HEARING
    config.setdefault("senses", {}).setdefault(hearing, {})

    if choice == 3:
        config["senses"][hearing]["enabled"] = False
        return

    config["senses"][hearing]["enabled"] = True
    if choice == 1:
        config["senses"][hearing]["stt"] = "whisper"
    else:
        stt = _prompt_text("STT provider name", "deepgram")
        config["senses"][hearing]["stt"] = stt

    if _prompt_yes_no("Enable text-to-speech (TTS)?", default=False):
        tts_choice = _prompt_choice(
            "TTS provider (Whisper is STT only — TTS requires a separate engine):",
            [
                "OpenAI TTS (cloud)",
                "Piper (local, free)",
                "ElevenLabs (cloud)",
                "Custom",
            ],
            default=1,
        )
        tts_map = {1: "openai", 2: "piper", 3: "elevenlabs"}
        if tts_choice in tts_map:
            config["senses"][hearing]["tts"] = tts_map[tts_choice]
        else:
            config["senses"][hearing]["tts"] = _prompt_text("TTS provider name")


def _step_personality(config: dict) -> None:
    """Step 5: Personality configuration."""
    print("\n━━━ Step 5: Personality ━━━")

    ai_name = _prompt_text("Give your AI a name", config.get("name", "Arqitect"))
    config["name"] = ai_name
    config.setdefault("personality", {})["name"] = ai_name

    presets = [
        "Professional assistant",
        "Friendly companion",
        "Technical expert",
        "Custom...",
    ]
    choice = _prompt_choice("Personality preset:", presets, default=1)

    preset_map = {1: "professional", 2: "friendly", 3: "technical"}
    if choice in preset_map:
        config["personality"]["preset"] = preset_map[choice]
        config["personality"]["tone"] = {
            "professional": "clear, direct, and helpful",
            "friendly": "warm, approachable, and conversational",
            "technical": "precise, thorough, and analytical",
        }[preset_map[choice]]
    else:
        config["personality"]["preset"] = "custom"
        config["personality"]["tone"] = _prompt_text(
            "Tone of voice (e.g., 'warm and curious', 'direct and efficient')",
        )
        config["personality"]["traits"] = _prompt_list("Character traits")

    style = config["personality"].setdefault("communication_style", {})
    formality_choice = _prompt_choice(
        "Communication formality:",
        ["Formal", "Casual", "Adaptive"],
        default=2,
    )
    style["formality"] = ["formal", "casual", "adaptive"][formality_choice - 1]


def _step_connectors(config: dict) -> None:
    """Step 6: Communication connectors."""
    print("\n━━━ Step 6: Connectors ━━━")
    print("  Dashboard is always included.\n")

    config.setdefault("connectors", {})

    # Telegram
    if _prompt_yes_no("Enable Telegram connector?", default=False):
        config["connectors"].setdefault("telegram", {})["enabled"] = True
        token = _prompt_text("Telegram bot token (from @BotFather, or set later)")
        if token:
            config.setdefault("secrets", {})["telegram_bot_token"] = token
        bot_name = _prompt_text("Telegram bot display name", config.get("name", "Arqitect"))
        config["connectors"]["telegram"]["bot_name"] = bot_name
        aliases = _prompt_list("Bot aliases (nicknames users can mention)")
        if aliases:
            config["connectors"]["telegram"]["bot_aliases"] = aliases
    else:
        config["connectors"].setdefault("telegram", {})["enabled"] = False

    # WhatsApp
    if _prompt_yes_no("Enable WhatsApp connector?", default=False):
        config["connectors"].setdefault("whatsapp", {})["enabled"] = True
        bot_name = _prompt_text("WhatsApp bot display name", config.get("name", "Arqitect"))
        config["connectors"]["whatsapp"]["bot_name"] = bot_name
    else:
        config["connectors"].setdefault("whatsapp", {})["enabled"] = False


def _step_touch(config: dict) -> None:
    """Step 7: Touch — filesystem and execution settings based on environment."""
    print("\n━━━ Step 7: Touch ━━━")

    environment = config.get("environment", "server")
    print(f"  Environment: {environment}")

    touch = config.setdefault("senses", {}).setdefault("touch", {})
    touch["enabled"] = True
    touch["environment"] = environment

    if environment in ("desktop", "server"):
        fs_choice = _prompt_choice(
            "Filesystem access level:",
            ["Full access (home directory)", "Sandboxed (project directory only)", "Read-only", "No filesystem access"],
            default=2,
        )
        fs_map = {1: "full", 2: "sandboxed", 3: "readonly", 4: "none"}
        touch.setdefault("filesystem", {})["access"] = fs_map[fs_choice]

        if fs_map[fs_choice] == "sandboxed":
            touch["filesystem"]["root"] = _prompt_text("Sandbox directory", "./sandbox")

        exec_choice = _prompt_choice(
            "Allow command execution?",
            ["Yes — any command", "Yes — allowlisted commands only", "No"],
            default=2,
        )
        touch.setdefault("execution", {})
        if exec_choice == 3:
            touch["execution"]["enabled"] = False
        else:
            touch["execution"]["enabled"] = True
            if exec_choice == 2:
                touch["execution"]["allowlist"] = _prompt_list(
                    "Allowed commands",
                    "comma-separated, e.g. python,node,git,curl",
                )

    elif environment == "iot":
        touch.setdefault("filesystem", {})["access"] = "sandboxed"
        touch["filesystem"]["root"] = "./sandbox"
        touch.setdefault("execution", {})["enabled"] = False


def _step_embeddings(config: dict) -> None:
    """Step 8: Embeddings configuration."""
    print("\n━━━ Step 8: Embeddings ━━━")

    choice = _prompt_choice(
        "Embedding model for semantic matching:",
        [
            "Local — nomic-embed-text (recommended)",
            "Local — all-MiniLM (lighter)",
            "Local — GGUF model (same as brain)",
            "OpenAI embeddings (cloud)",
            "None — keyword matching only",
        ],
        default=3,
    )
    config.setdefault("embeddings", {})
    if choice == 1:
        config["embeddings"]["provider"] = "local"
        config["embeddings"]["model"] = "nomic-embed-text"
    elif choice == 2:
        config["embeddings"]["provider"] = "local"
        config["embeddings"]["model"] = "all-MiniLM-L6-v2"
    elif choice == 3:
        config["embeddings"]["provider"] = "gguf"
        brain_entry = config.get("inference", {}).get("models", {}).get("brain", "")
        brain_file = brain_entry.get("file", "") if isinstance(brain_entry, dict) else (brain_entry or "")
        config["embeddings"]["model"] = brain_file
    elif choice == 4:
        config["embeddings"]["provider"] = "openai"
        config["embeddings"]["model"] = "text-embedding-3-small"
    else:
        config["embeddings"]["provider"] = "none"
        config["embeddings"]["model"] = ""


def _step_storage(config: dict) -> None:
    """Step 9: Storage / database configuration."""
    print("\n━━━ Step 9: Storage ━━━")

    config.setdefault("storage", {})

    # Redis (hot memory)
    redis_choice = _prompt_choice(
        "Redis configuration (hot memory):",
        ["Local Redis (localhost:6379)", "Remote Redis (provide URL)"],
        default=1,
    )
    if redis_choice == 1:
        config["storage"]["hot"] = {"url": "redis://localhost:6379"}
    else:
        url = _prompt_text("Redis URL", "redis://localhost:6379")
        config["storage"]["hot"] = {"url": url}

    # Cold storage
    cold_choice = _prompt_choice(
        "Cold storage (memory database):",
        ["SQLite (default — zero config)", "PostgreSQL (provide connection string)"],
        default=1,
    )
    if cold_choice == 1:
        db_name = _prompt_text("SQLite database filename", "arqitect_memory.db")
        config["storage"]["cold"] = {"path": db_name}
    else:
        pg_url = _prompt_text("PostgreSQL connection URL")
        config["storage"]["cold"] = {"type": "postgres", "url": pg_url}

    # Warm storage
    config["storage"]["warm"] = {"path": _prompt_text("Episodes database filename", "episodes.db")}

    # Ports
    config.setdefault("ports", {})
    config["ports"]["mcp"] = int(_prompt_text("MCP server port", "8100"))
    config["ports"]["bridge"] = int(_prompt_text("Bridge/Dashboard port", "3000"))


def _step_admin(config: dict) -> None:
    """Step 10: Admin setup."""
    print("\n━━━ Step 10: Admin ━━━")

    config.setdefault("admin", {})
    config["admin"]["name"] = _prompt_text("Admin user name")
    config["admin"]["email"] = _prompt_text("Admin email")

    # Auto-generate JWT secret if empty
    existing_jwt = config.get("secrets", {}).get("jwt_secret", "")
    if not existing_jwt:
        config.setdefault("secrets", {})["jwt_secret"] = _generate_jwt_secret()
        print("  JWT secret auto-generated.")

    # SMTP (optional)
    if _prompt_yes_no("Configure SMTP for email notifications?", default=False):
        smtp = config.setdefault("secrets", {}).setdefault("smtp", {})
        smtp["host"] = _prompt_text("SMTP host", "smtp.gmail.com")
        smtp["port"] = int(_prompt_text("SMTP port", "587"))
        admin_email = config.get("admin", {}).get("email", "")
        smtp["user"] = _prompt_text("SMTP user (email)", admin_email)
        smtp["password"] = _prompt_text("SMTP password (or app password)")
        smtp["from"] = _prompt_text("From address", smtp.get("user", ""))


# ── Config writer ────────────────────────────────────────────────────────

def _write_config(config: dict, path: Path) -> None:
    """Write the config dict to arqitect.yaml, preserving key order."""
    # Define preferred key order for readability
    key_order = [
        "name", "environment", "inference", "personality", "senses",
        "connectors", "embeddings", "storage", "ports", "ssl",
        "admin", "secrets",
    ]
    ordered: dict[str, Any] = {}
    for key in key_order:
        if key in config:
            ordered[key] = config[key]
    # Append any keys not in the order list
    for key in config:
        if key not in ordered:
            ordered[key] = config[key]

    with open(path, "w") as f:
        yaml.dump(
            ordered,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


# ── Main wizard entry point ─────────────────────────────────────────────

STEPS = [
    _step_environment,
    _step_brain_model,
    _step_vision,
    _step_hearing,
    _step_personality,
    _step_connectors,
    _step_touch,
    _step_embeddings,
    _step_storage,
    _step_admin,
]


def run_wizard() -> None:
    """Run the interactive setup wizard and write arqitect.yaml."""
    root = get_project_root()
    yaml_path = root / "arqitect.yaml"

    print("\n╔══════════════════════════════════════╗")
    print("║       Arqitect Setup Wizard          ║")
    print("╚══════════════════════════════════════╝")

    # Start from existing config or defaults
    if yaml_path.exists():
        print(f"\n  Found existing {yaml_path.name} — updating in place.")
        print("  Existing values shown as defaults. Press Enter to keep them.\n")
        with open(yaml_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        print("\n  No arqitect.yaml found — creating a new one.\n")
        config = copy.deepcopy(DEFAULTS)

    for step_fn in STEPS:
        try:
            step_fn(config)
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Wizard interrupted. Progress saved so far.")
            _write_config(config, yaml_path)
            print(f"  Config written to {yaml_path}")
            return

    _write_config(config, yaml_path)
    print(f"\n  ✓ Config written to {yaml_path}")

    # Install dependencies and start services
    import subprocess
    print("\n  Installing dependencies and starting Arqitect...")
    subprocess.run(["make", "setup"], cwd=str(root))

    print("\n  Next steps:")
    print("  1. Open https://otomus.github.io/arqitect-dashboard/")
    print("  2. Go to Settings and add your server address to connect\n")
