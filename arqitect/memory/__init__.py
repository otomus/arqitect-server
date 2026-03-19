"""
Unified MemoryManager — facade over Hot (Redis), Warm (SQLite episodes),
and Cold (SQLite knowledge) memory tiers.
"""

import json
import time

from .hot import HotMemory
from .warm import WarmMemory
from .cold import ColdMemory


class MemoryManager:
    def __init__(self, redis_client):
        self.hot = HotMemory(redis_client)
        self.warm = WarmMemory()
        self.cold = ColdMemory()

    def get_context_for_task(self, task: str, user_id: str = "") -> dict:
        """Build full context for a task, scoped to user."""
        session = self.hot.get_session(user_id=user_id)
        episodes = self.warm.recall(task, limit=5, user_id=user_id)
        user_facts = self.cold.get_user_facts(user_id) if user_id else self.cold.get_facts("user")
        return {
            "session": session,
            "episodes": episodes,
            "facts": user_facts,
        }

    def record_episode(self, episode: dict):
        """Record an episode in warm memory and update cold stats."""
        episode.setdefault("timestamp", time.time())
        self.warm.record(episode)

        # Update cold stats
        nerve = episode.get("nerve")
        tool = episode.get("tool")
        success = episode.get("success", True)

        if nerve:
            self.cold.record_nerve_invocation(nerve, success)
        if tool:
            self.cold.record_tool_call(tool, success)
        if nerve and tool:
            self.cold.add_nerve_tool(nerve, tool)

    def get_env_for_nerve(self, nerve_name: str, task: str,
                          project_context: str = "", user_id: str = "") -> dict:
        """Build env vars to pass memory context to a nerve subprocess."""
        session = self.hot.get_session(user_id=user_id)
        episodes = self.warm.recall(task, limit=3, user_id=user_id)
        known_tools = self.cold.get_nerve_tools(nerve_name)

        # Simplify episodes for transport
        episode_hints = []
        for ep in episodes:
            episode_hints.append({
                "task": ep.get("task", ""),
                "nerve": ep.get("nerve", ""),
                "tool": ep.get("tool", ""),
                "success": bool(ep.get("success", 1)),
            })

        # Cold facts for fact-aware runtime
        user_facts = self.cold.get_user_facts(user_id) if user_id else self.cold.get_facts("user")

        # Nerve metadata (system_prompt + examples) from cold memory
        nerve_meta = self.cold.get_nerve_metadata(nerve_name)

        # Resolve model-specific prompt from community cache (context.json)
        try:
            from arqitect.brain.adapters import resolve_nerve_prompt
            role = nerve_meta.get("role", "tool")
            ctx = resolve_nerve_prompt(nerve_name, role)
            if ctx:
                if ctx.get("system_prompt"):
                    nerve_meta["system_prompt"] = ctx["system_prompt"]
                if ctx.get("few_shot_examples"):
                    nerve_meta["examples"] = ctx["few_shot_examples"]
        except (ImportError, ValueError):
            pass

        # User profile (name, gender, preferences) for personalized responses
        user_profile = self.cold.get_user_profile(user_id) if user_id else {}

        # Conversation history — sliding window sized by community adapter config
        from arqitect.brain.adapters import get_conversation_window
        nerve_role = nerve_meta.get("role", "nerve")
        messages = self.hot.get_conversation(limit=get_conversation_window(nerve_role), user_id=user_id)

        env = {
            "SYNAPSE_NERVE_NAME": nerve_name,
            "SYNAPSE_SESSION": json.dumps(session),
            "SYNAPSE_EPISODES": json.dumps(episode_hints),
            "SYNAPSE_KNOWN_TOOLS": json.dumps(known_tools),
            "SYNAPSE_FACTS": json.dumps(user_facts),
            "SYNAPSE_NERVE_META": json.dumps(nerve_meta),
            "SYNAPSE_USER_ID": user_id,
            "SYNAPSE_USER_PROFILE": json.dumps(user_profile),
            "SYNAPSE_MESSAGES": json.dumps(messages),
        }

        if project_context:
            env["SYNAPSE_PROJECT_CONTEXT"] = project_context

        return env
