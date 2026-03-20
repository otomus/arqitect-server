"""Type-safe test data factories for the brain dispatch pipeline.

Uses polyfactory to generate typed decision dataclasses and DispatchContext
instances. Every factory produces valid, fully-typed objects — no bare dicts.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from polyfactory.factories.dataclass_factory import DataclassFactory

from arqitect.brain.dispatch import (
    ChainDecision,
    ChainStep,
    ClarifyDecision,
    DispatchContext,
    FeedbackDecision,
    InvokeDecision,
    RespondDecision,
    SynthesizeDecision,
    UpdateContextDecision,
    UseSenseDecision,
)
from arqitect.types import Action


# ---------------------------------------------------------------------------
# Decision factories — one per action type
# ---------------------------------------------------------------------------

class InvokeDecisionFactory(DataclassFactory[InvokeDecision]):
    """Factory for invoke_nerve decisions."""
    __model__ = InvokeDecision

    action = Action.INVOKE_NERVE
    name = "joke_nerve"
    args = "tell me a joke"


class SynthesizeDecisionFactory(DataclassFactory[SynthesizeDecision]):
    """Factory for synthesize_nerve decisions."""
    __model__ = SynthesizeDecision

    action = Action.SYNTHESIZE_NERVE
    name = "astronomy_nerve"
    description = "answers space questions"


class ChainStepFactory(DataclassFactory[ChainStep]):
    """Factory for individual chain steps."""
    __model__ = ChainStep

    nerve = "joke_nerve"
    args = "tell a joke"


class ChainDecisionFactory(DataclassFactory[ChainDecision]):
    """Factory for chain_nerves decisions."""
    __model__ = ChainDecision

    action = Action.CHAIN_NERVES
    goal = "tell a joke then translate it"


class ClarifyDecisionFactory(DataclassFactory[ClarifyDecision]):
    """Factory for clarify decisions."""
    __model__ = ClarifyDecision

    action = Action.CLARIFY
    message = "What do you mean?"
    suggestions = ["option A", "option B"]


class FeedbackDecisionFactory(DataclassFactory[FeedbackDecision]):
    """Factory for feedback decisions."""
    __model__ = FeedbackDecision

    action = Action.FEEDBACK
    sentiment = "positive"
    message = "Great!"


class UpdateContextDecisionFactory(DataclassFactory[UpdateContextDecision]):
    """Factory for update_context decisions."""
    __model__ = UpdateContextDecision

    action = Action.UPDATE_CONTEXT
    message = "Noted!"


class RespondDecisionFactory(DataclassFactory[RespondDecision]):
    """Factory for respond decisions."""
    __model__ = RespondDecision

    action = Action.RESPOND
    message = "I am sentient"


class UseSenseDecisionFactory(DataclassFactory[UseSenseDecision]):
    """Factory for use_sense decisions."""
    __model__ = UseSenseDecision

    action = "use_sense"
    sense = "touch"
    args = "list files"


# ---------------------------------------------------------------------------
# DispatchContext factory
# ---------------------------------------------------------------------------

class DispatchContextFactory(DataclassFactory[DispatchContext]):
    """Factory for DispatchContext — the main dispatch pipeline input.

    Default nerve_catalog includes one nerve and one sense.
    think_fn defaults to a MagicMock returning 're-thought result'.
    """
    __model__ = DispatchContext
    __faker__ = None

    task = "hello"
    decision = {"action": "invoke_nerve", "name": "joke_nerve", "args": "tell me a joke"}
    user_id = ""
    depth = 0
    history = []
    nerve_catalog = {"joke_nerve": "tells jokes", "awareness": "identity"}
    available = ["joke_nerve", "awareness"]
    think_fn = MagicMock(return_value="re-thought result")

    @classmethod
    def build(cls, **kwargs):
        """Build with a fresh MagicMock for think_fn each time."""
        if "think_fn" not in kwargs:
            kwargs["think_fn"] = MagicMock(return_value="re-thought result")
        return super().build(**kwargs)


# ---------------------------------------------------------------------------
# Episode factory — for memory/warm tests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Episode:
    """A recorded episode in warm memory.

    Args:
        task: The user's original input.
        nerve: Nerve that handled the task.
        tool: Tool used by the nerve (empty if none).
        success: Whether the task succeeded.
        user_id: User who triggered it.
    """
    task: str = "tell me a joke"
    nerve: str = "joke_nerve"
    tool: str = ""
    success: bool = True
    user_id: str = ""


class EpisodeFactory(DataclassFactory[Episode]):
    """Factory for episode dicts used in memory tests."""
    __model__ = Episode

    task = "tell me a joke"
    nerve = "joke_nerve"
    tool = ""
    success = True
    user_id = ""


# ---------------------------------------------------------------------------
# Helpers — build typed dicts from decision dataclasses
# ---------------------------------------------------------------------------

def as_dict(decision: object) -> dict:
    """Convert a frozen decision dataclass to a mutable dict.

    This bridges typed factories with the dispatch pipeline which still
    accepts raw dicts at its boundary.

    Args:
        decision: A frozen decision dataclass instance.

    Returns:
        A plain dict suitable for DispatchContext.decision.
    """
    from dataclasses import asdict
    result = asdict(decision)
    # Convert ChainStep dataclasses back to plain dicts
    if "steps" in result and isinstance(result["steps"], list):
        result["steps"] = [
            s if isinstance(s, dict) else asdict(s)
            for s in result["steps"]
        ]
    return result
