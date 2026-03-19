"""Default configuration values for arqitect.yaml sections."""

DEFAULTS = {
    "name": "Arqitect",
    "environment": "server",
    "inference": {
        "provider": "gguf",
        "models": {
            "brain": "Qwen2.5-Coder-7B.gguf",
            "nerve": "Qwen2.5-Coder-7B.gguf",
            "coder": "Qwen2.5-Coder-7B.gguf",
            "creative": "Qwen2.5-Coder-7B.gguf",
            "communication": "Qwen2.5-Coder-7B.gguf",
            "vision": "moondream2-text-model-f16.gguf",
        },
        "roles": {
            "brain": {"provider": None, "model": None},
            "nerve": {"provider": None, "model": None},
            "coder": {"provider": None, "model": None},
            "creative": {"provider": None, "model": None},
            "communication": {"provider": None, "model": None},
        },
    },
    "personality": {
        "name": "",
        "tone": "",
        "traits": [],
        "preset": "",
        "communication_style": {
            "emoji_level": "moderate",
            "formality": "casual",
            "verbosity": "concise",
        },
    },
    "storage": {
        "hot": {"url": "redis://localhost:6379"},
        "cold": {"path": "arqitect_memory.db"},
        "warm": {"path": "episodes.db"},
    },
    "ports": {
        "mcp": 8100,
        "bridge": 3000,
    },
    "ssl": {
        "cert": "",
        "key": "",
    },
    "senses": {
        "sight": {
            "enabled": True,
            "provider": "",
        },
        "hearing": {
            "enabled": True,
            "stt": "",
            "tts": "",
        },
        "touch": {
            "enabled": True,
            "environment": "server",
            "filesystem": {
                "access": "sandboxed",
                "root": "./sandbox",
            },
            "execution": {
                "enabled": True,
                "allowlist": [],
            },
        },
        "awareness": {"enabled": True},
        "communication": {"enabled": True},
    },
    "connectors": {
        "whatsapp": {
            "enabled": False,
            "bot_name": "Arqitect",
            "bot_aliases": [],
            "whitelisted_users": [],
            "whitelisted_groups": [],
            "monitor_groups": [],
        },
        "telegram": {
            "enabled": False,
            "bot_name": "Arqitect",
            "bot_aliases": [],
            "whitelisted_users": [],
            "whitelisted_groups": [],
            "monitor_groups": [],
        },
    },
    "secrets": {
        "jwt_secret": "",
        "telegram_bot_token": "",
        "anthropic_api_key": "",
        "openai_api_key": "",
        "openai_base_url": "https://api.openai.com/v1",
        "groq_api_key": "",
        "deepseek_api_key": "",
        "mistral_api_key": "",
        "openrouter_api_key": "",
        "google_ai_api_key": "",
        "xai_api_key": "",
        "together_api_key": "",
        "smtp": {
            "host": "smtp.gmail.com",
            "port": 587,
            "user": "",
            "password": "",
            "from": "",
        },
    },
    "embeddings": {
        "provider": "gguf",
        "model": "Qwen2.5-Coder-7B.gguf",
    },
    "admin": {
        "name": "",
        "email": "",
    },
}
