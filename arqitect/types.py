"""Centralized type enums for the Arqitect nervous system."""

from enum import StrEnum


class Action(StrEnum):
    """Brain routing actions."""
    INVOKE_NERVE = "invoke_nerve"
    SYNTHESIZE_NERVE = "synthesize_nerve"
    CHAIN_NERVES = "chain_nerves"
    RESPOND = "respond"
    UPDATE_CONTEXT = "update_context"
    CLARIFY = "clarify"
    USE_SENSE = "use_sense"
    FABRICATE_TOOL = "fabricate_tool"
    FEEDBACK = "feedback"


class Sense(StrEnum):
    """Core senses."""
    SIGHT = "sight"
    HEARING = "hearing"
    TOUCH = "touch"
    AWARENESS = "awareness"
    COMMUNICATION = "communication"


class NerveRole(StrEnum):
    """Nerve model roles. Extensible — unknown roles fall back to default."""
    TOOL = "tool"
    CREATIVE = "creative"
    CODE = "code"


class Tone(StrEnum):
    """Communication tone values."""
    CASUAL = "casual"
    ENTHUSIASTIC = "enthusiastic"
    PROFESSIONAL = "professional"
    EMPATHETIC = "empathetic"
    NEUTRAL = "neutral"
    FORMAL = "formal"


class Channel(StrEnum):
    """Redis pub/sub event channels."""
    BRAIN_THOUGHT = "brain:thought"
    BRAIN_ACTION = "brain:action"
    BRAIN_RESPONSE = "brain:response"
    BRAIN_AUDIO = "brain:audio"
    BRAIN_TASK = "brain:task"
    BRAIN_CHECKLIST = "brain:checklist"
    NERVE_RESULT = "nerve:result"
    NERVE_QUALIFICATION = "nerve:qualification"
    SYSTEM_STATUS = "system:status"
    SYSTEM_KILL = "system:kill"
    MEMORY_UPDATE = "memory:update"
    MEMORY_EPISODE = "memory:episode"
    MEMORY_TOOL_LEARNED = "memory:tool_learned"
    SENSE_CALIBRATION = "sense:calibration"
    SENSE_CONFIG = "sense:config"
    SENSE_PEEK = "sense:peek"
    SENSE_VOICE = "sense:voice"
    SENSE_IMAGE = "sense:image"
    SENSE_SIGHT_FRAME = "sense:sight:frame"
    SENSE_STT_RESULT = "sense:stt:result"
    TOOL_LIFECYCLE = "tool:lifecycle"


class IntentType(StrEnum):
    """User intent classification."""
    WORKFLOW = "workflow"
    DIRECT = "direct"


class NerveStatus(StrEnum):
    """Nerve response status values."""
    OK = "ok"
    ERROR = "error"
    SUCCESS = "success"
    WRONG_NERVE = "wrong_nerve"
    NEEDS_DATA = "needs_data"


class RedisKey(StrEnum):
    """Redis key prefixes/names."""
    SESSION = "synapse:session"
    CONVERSATION = "synapse:conversation"
    SENSE_CALIBRATION = "synapse:sense_calibration"
    NERVE_STATUS = "synapse:nerve_status"
