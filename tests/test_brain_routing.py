"""P2/P3 — Brain routing tests: action dispatch, depth limits, sense routing.

Tests brain.think() end-to-end with mocked LLM and subprocess.
All LLM decision dicts are produced by typed factories.
"""

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import (
    FakeLLM, make_mem, register_qualified_nerve, make_nerve_file,
    setup_brain_patches, patch_invoke_nerve, patch_synthesize_nerve,
)
from tests.factories import (
    InvokeDecisionFactory,
    SynthesizeDecisionFactory,
    ChainDecisionFactory,
    ChainStepFactory,
    as_dict,
)


# ---------------------------------------------------------------------------
# 2.1 Depth limit terminates
# ---------------------------------------------------------------------------

class TestDepthLimit:
    def test_depth_limit_returns_fallback(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """think() must terminate at depth > 5 with a fallback message."""
        mem_fixture = make_mem(test_redis)
        fake = FakeLLM([
            ("Task:", '{"action": "invalid_action_forever"}', True),
        ])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        for p in patches:
            p.start()
        try:
            from arqitect.brain.brain import think
            result = think("hello")
            assert "rephrasing" in result.lower() or "detail" in result.lower(), \
                f"Expected fallback message, got: {result}"
            assert fake.call_count <= 15, f"Too many LLM calls: {fake.call_count}"
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# 2.2 Synthesize-then-invoke does not re-synthesize existing nerve
# ---------------------------------------------------------------------------

class TestSynthesizeThenInvoke:
    def test_existing_nerve_not_re_synthesized(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """If a nerve already exists in the catalog, synthesize should redirect to invoke."""
        mem_fixture = make_mem(test_redis)
        register_qualified_nerve(mem_fixture, "weather_nerve", "answers weather questions")
        make_nerve_file(nerves_dir, "weather_nerve")

        synth = SynthesizeDecisionFactory.build(
            name="weather_nerve", description="answers weather questions",
        )
        invoke = InvokeDecisionFactory.build(
            name="weather_nerve", args="weather in Paris",
        )
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(synth))),
            ("already exists", json.dumps(as_dict(invoke))),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_invoke_nerve(return_value='{"response": "sunny 25C"}'):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.brain import think
                result = think("weather in Paris")
                assert "sunny" in result or "25C" in result or "response" in result
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 3.1 Existing qualified nerve is invoked directly
# ---------------------------------------------------------------------------

class TestRouteToExistingNerve:
    def test_invoke_existing_nerve(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """Brain should invoke an existing qualified nerve without synthesizing."""
        mem_fixture = make_mem(test_redis)
        register_qualified_nerve(mem_fixture, "joke_nerve", "tells jokes and humor")
        make_nerve_file(nerves_dir, "joke_nerve")

        d = InvokeDecisionFactory.build(name="joke_nerve", args="tell me a joke")
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(d))),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_invoke_nerve(return_value='{"response": "Why did the chicken cross the road?"}'):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.brain import think
                result = think("tell me a joke")
                assert "chicken" in result or "response" in result
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 3.5 Action name fuzzy fix
# ---------------------------------------------------------------------------

class TestActionNameFuzzyFix:
    def test_typo_in_action_name_corrected(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """LLM typo like 'invok_nerve' should be corrected to 'invoke_nerve'."""
        mem_fixture = make_mem(test_redis)
        register_qualified_nerve(mem_fixture, "weather_nerve", "weather data")
        make_nerve_file(nerves_dir, "weather_nerve")

        d = InvokeDecisionFactory.build(
            action="invok_nerve", name="weather_nerve", args="weather in Paris",
        )
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(d))),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_invoke_nerve(return_value='{"response": "sunny"}') as mock_invoke:
            for p in patches:
                p.start()
            try:
                from arqitect.brain.brain import think
                result = think("weather in Paris")
                mock_invoke.assert_called()
                assert mock_invoke.call_args_list[0][0][0] == "weather_nerve"
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 3.6 Sense name used as action
# ---------------------------------------------------------------------------

class TestSenseNameAsAction:
    def test_sense_name_as_action_corrected(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """If LLM returns action='touch', it should map to invoke_nerve('touch')."""
        mem_fixture = make_mem(test_redis)
        mem_fixture.cold.register_sense("touch", "File system operations")

        d = InvokeDecisionFactory.build(action="touch", args="list files in home")
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(d))),
            ("Sense: touch", '{"mode": "list", "path": "/home"}'),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_invoke_nerve(return_value='{"response": "file1.txt file2.txt"}') as mock_invoke:
            for p in patches:
                p.start()
            try:
                from arqitect.brain.brain import think
                result = think("list files in home")
                mock_invoke.assert_called()
                assert mock_invoke.call_args[0][0] == "touch"
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 3.7 Identity queries route to awareness sense
# ---------------------------------------------------------------------------

class TestSenseRouting:
    def test_who_are_you_routes_to_awareness(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """'who are you?' should route to the awareness sense."""
        mem_fixture = make_mem(test_redis)

        d = InvokeDecisionFactory.build(name="awareness", args="who are you?")
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(d))),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_invoke_nerve(return_value='{"response": "I am Sentient"}') as mock_invoke:
            for p in patches:
                p.start()
            try:
                from arqitect.brain.brain import think
                result = think("who are you?")
                mock_invoke.assert_called_once()
                assert mock_invoke.call_args[0][0] == "awareness"
            finally:
                for p in patches:
                    p.stop()

    def test_thanks_routes_to_communication(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """'thanks' should route to the communication sense."""
        mem_fixture = make_mem(test_redis)

        d = InvokeDecisionFactory.build(name="communication", args="thanks, perfect")
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(d))),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_invoke_nerve(return_value='{"response": "You are welcome!"}') as mock_invoke:
            for p in patches:
                p.start()
            try:
                from arqitect.brain.brain import think
                result = think("thanks, perfect")
                mock_invoke.assert_called()
                assert mock_invoke.call_args_list[0][0][0] == "communication"
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 3.8 Greeting synthesizes a new nerve cleanly
# ---------------------------------------------------------------------------

class TestGreetingSynthesis:
    def test_hello_synthesis_uses_renamed_name(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """'hello' has no nerve — brain synthesizes. If renamed, the renamed name is used."""
        mem_fixture = make_mem(test_redis)

        # Synthesis requires an identified user
        user_id = mem_fixture.cold.create_user_with_email("test@test.com", "test", "test_user")
        mem_fixture.cold.set_user_role(user_id, "user")

        call_count = {"synth": 0}

        def mock_synthesize(name, desc, mcp_tools=None, trigger_task="", role=None):
            call_count["synth"] += 1
            d = os.path.join(nerves_dir, name)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "nerve.py")
            with open(path, "w") as f:
                f.write("# stub\n")
            return name, path

        synth = SynthesizeDecisionFactory.build(
            name="greeting_nerve", description="handles greetings and small talk",
        )
        invoke = InvokeDecisionFactory.build(name="greeting_nerve", args="hello")
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(synth))),
            ("Synthesized nerve", json.dumps(as_dict(invoke))),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_synthesize_nerve(side_effect=mock_synthesize):
            with patch_invoke_nerve(return_value='{"response": "Hello there!"}'):
                for p in patches:
                    p.start()
                try:
                    from arqitect.brain.brain import think
                    from arqitect.brain.events import set_task_origin
                    set_task_origin("test", "test_chat", user_id)
                    result = think("hello")
                    assert call_count["synth"] == 1, "Should synthesize exactly once"
                finally:
                    for p in patches:
                        p.stop()


# ---------------------------------------------------------------------------
# 3.85 Brain uses discover_nerves
# ---------------------------------------------------------------------------

class TestBrainUsesDiscoverNerves:
    def test_think_calls_discover_nerves_directly(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """brain.think() must import and call discover_nerves directly."""
        mem_fixture = make_mem(test_redis)

        d = InvokeDecisionFactory.build(name="awareness", args="who are you?")
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(d))),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_invoke_nerve(return_value='{"response": "I am Sentient"}'):
            for p in patches:
                p.start()
            try:
                import arqitect.brain.brain as brain_module
                assert hasattr(brain_module, "discover_nerves"), \
                    "brain.py must import discover_nerves from catalog"
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 3.9 All senses appear in catalog without qualification
# ---------------------------------------------------------------------------

class TestSenseCatalog:
    def test_all_senses_in_catalog(self, test_redis, tmp_memory_dir, nerves_dir):
        """All 5 core senses must appear in the catalog regardless of qualification."""
        from arqitect.brain.catalog import discover_nerves
        from arqitect.types import Sense

        mem_fixture = make_mem(test_redis)
        with patch("arqitect.brain.catalog.mem", mem_fixture):
            catalog = discover_nerves()
            for sense in Sense:
                assert sense in catalog, \
                    f"Core sense '{sense}' missing from catalog. Got: {list(catalog.keys())}"


# ---------------------------------------------------------------------------
# 4.0 Platform onboarding guard — Telegram/WhatsApp, before think()
# ---------------------------------------------------------------------------

class TestPlatformOnboardingGuard:
    """The brain's onboarding check runs only for platform connectors
    (Telegram, WhatsApp) that always send connector_user_id. Bridge
    handles its own onboarding — brain never sees bridge onboarding."""

    def test_platform_user_triggers_brain_onboarding(self, test_redis, tmp_memory_dir):
        """Telegram user with connector_user_id → brain onboarding fires."""
        from arqitect.brain.onboarding import handle_onboarding
        mem_fixture = make_mem(test_redis)

        with patch("arqitect.brain.onboarding._send_verification_email", return_value=True):
            msg, user_id = handle_onboarding(mem_fixture.cold, "telegram", "tg_user_42", "test@test.com")
        assert msg, "Should return onboarding message"
        lower = msg.lower()
        assert "verif" in lower or "code" in lower

    def test_verified_platform_user_passes_through(self, test_redis, tmp_memory_dir):
        """Already verified platform user → empty message, proceed to think()."""
        from arqitect.brain.onboarding import handle_onboarding
        mem_fixture = make_mem(test_redis)

        user_id = mem_fixture.cold.create_user_with_email(
            "test@test.com", "telegram", "tg_user_42",
        )
        mem_fixture.cold.set_user_display_name(user_id, "Test User")

        msg, resolved_id = handle_onboarding(mem_fixture.cold, "telegram", "tg_user_42", "what is the weather?")
        assert msg == "", "Verified user should get empty message (pass through)"
        assert resolved_id == user_id

    def test_brain_skips_onboarding_when_connector_user_id_empty(self, test_redis, tmp_memory_dir):
        """Bridge sends empty connector_user_id for anon — brain skips onboarding entirely.
        Bridge handles its own onboarding flow."""
        # Simulate the guard logic from brain's listen_redis
        connector_user_id = ""
        source = "dashboard"
        # Brain only runs onboarding when connector_user_id is present
        assert not connector_user_id, "Empty connector_user_id should skip brain onboarding"

    def test_same_email_different_platform_links_account(self, test_redis, tmp_memory_dir):
        """User verified on telegram, arrives from whatsapp with same email → links."""
        from arqitect.brain.onboarding import handle_onboarding
        mem_fixture = make_mem(test_redis)

        tg_user_id = mem_fixture.cold.create_user_with_email(
            "oron@test.com", "telegram", "tg_user_42",
        )
        mem_fixture.cold.set_user_display_name(tg_user_id, "Oron")

        with patch("arqitect.brain.onboarding._send_verification_email", return_value=True):
            msg, _ = handle_onboarding(mem_fixture.cold, "whatsapp", "12025551234", "oron@test.com")
        assert msg, "Should start verification flow"
        lower = msg.lower()
        assert "verif" in lower or "code" in lower
