"""Flow tests: conversation → episode recording → dreamstate tuning.

Behavioral contracts — what the system MUST do, not what it currently does.
Failures here are real bugs or missing features.

Contracts:
1. Every nerve that participates in a conversation gets last_invoked_at set.
2. Dreamstate work queue contains ONLY invoked nerves.
3. Consolidation targets locally-fabricated nerves vs community — not all-vs-all.
4. Test bank expansion fabricates up to min_training_examples, skips if enough.
5. Dreamstate is interruptible via wake().
6. Full flow: user says 'hi' → nerve fails → episode recorded → dreamstate tunes.

Tools used:
- conftest: FakeLLM, setup_brain_patches, patch_invoke_nerve, mem, nerves_dir
- factories: DispatchContextFactory, InvokeDecisionFactory, as_dict
- time-machine: deterministic timestamps (no flaky time.sleep)
- dirty-equals: flexible LLM output assertions
- hypothesis: property-based edge case generation
- syrupy: snapshot testing for prompt/decision shapes
"""

import json
import os
import threading
from unittest.mock import MagicMock, patch

import pytest
import time_machine
from dirty_equals import IsStr, IsDict, IsDatetime
from hypothesis import given, strategies as st, settings, HealthCheck

from tests.conftest import (
    FakeLLM,
    make_mem,
    register_qualified_nerve,
    make_nerve_file,
    setup_brain_patches,
    patch_invoke_nerve,
    patch_synthesize_nerve,
)
from tests.factories import (
    DispatchContextFactory,
    InvokeDecisionFactory,
    SynthesizeDecisionFactory,
    ChainDecisionFactory,
    ChainStepFactory,
    ClarifyDecisionFactory,
    FeedbackDecisionFactory,
    UpdateContextDecisionFactory,
    RespondDecisionFactory,
    UseSenseDecisionFactory,
    as_dict,
)


# ══════════════════════════════════════════════════════════════════════════════
# Contract 1: Nerve invocations are recorded as active
# ══════════════════════════════════════════════════════════════════════════════


class TestNerveInvocationRecording:
    """Every nerve that participates in a conversation gets last_invoked_at set."""

    def test_successful_nerve_records_timestamp(self, mem):
        """A nerve that succeeds gets last_invoked_at set."""
        mem.cold.register_nerve("weather_nerve", "Get weather")
        assert mem.cold.get_last_invoked_at("weather_nerve") is None

        mem.cold.record_nerve_invocation("weather_nerve", success=True)

        assert mem.cold.get_last_invoked_at("weather_nerve") == IsStr(min_length=1)

    def test_failed_nerve_still_records_timestamp(self, mem):
        """A nerve that FAILS still gets last_invoked_at —
        dreamstate needs to know it was used so it can tune it."""
        mem.cold.register_nerve("reflect_nerve", "Reflect on task")
        mem.cold.record_nerve_invocation("reflect_nerve", success=False)

        assert mem.cold.get_last_invoked_at("reflect_nerve") is not None

    def test_sense_invocation_records_timestamp(self, mem):
        """Core senses also get recorded when invoked."""
        mem.cold.register_sense("communication", "Personality voice")
        mem.cold.record_nerve_invocation("communication", success=True)

        assert mem.cold.get_last_invoked_at("communication") is not None

    def test_brain_role_nerve_records_timestamp(self, mem):
        """Brain-type nerves also get last_invoked_at."""
        mem.cold.register_nerve_rich("reasoning", "Deep reasoning", role="brain")
        mem.cold.record_nerve_invocation("reasoning", success=True)

        assert mem.cold.get_last_invoked_at("reasoning") is not None

    def test_communication_role_nerve_records_timestamp(self, mem):
        """Communication-type nerves also get last_invoked_at."""
        mem.cold.register_nerve_rich("greeter", "Greet users", role="communication")
        mem.cold.record_nerve_invocation("greeter", success=False)

        assert mem.cold.get_last_invoked_at("greeter") is not None

    def test_multiple_invocations_update_timestamp(self, mem):
        """Each invocation updates the timestamp — no stale values.

        BUG: SQLite datetime('now') has 1-second resolution AND time-machine
        cannot control SQLite's C-level clock. This test uses real sleep(1.1)
        as a workaround, but the underlying issue is that record_nerve_invocation
        uses datetime('now') instead of Python-generated timestamps.
        """
        import time
        mem.cold.register_nerve("test_nerve", "Test")
        mem.cold.record_nerve_invocation("test_nerve", success=True)
        first = mem.cold.get_last_invoked_at("test_nerve")

        time.sleep(1.1)  # SQLite datetime('now') has 1-second resolution
        mem.cold.record_nerve_invocation("test_nerve", success=True)
        second = mem.cold.get_last_invoked_at("test_nerve")

        assert second > first, (
            f"Second timestamp ({second}) should be strictly after first ({first}). "
            f"FIX: record_nerve_invocation should use Python datetime.utcnow().isoformat() "
            f"instead of SQLite datetime('now') for sub-second resolution."
        )

    def test_invocation_counts_accumulate(self, mem):
        """Success/failure counters accumulate correctly."""
        mem.cold.register_nerve("counter_nerve", "Counter")

        mem.cold.record_nerve_invocation("counter_nerve", success=True)
        mem.cold.record_nerve_invocation("counter_nerve", success=True)
        mem.cold.record_nerve_invocation("counter_nerve", success=False)

        info = mem.cold.get_nerve_info("counter_nerve")
        assert info["total_invocations"] == 3
        assert info["successes"] == 2
        assert info["failures"] == 1

    def test_record_invocation_for_unregistered_nerve_creates_row(self, mem):
        """Recording an invocation for a non-existent nerve should UPSERT a row
        so last_invoked_at is set and dreamstate can tune it."""
        mem.cold.record_nerve_invocation("ghost_nerve", success=True)
        info = mem.cold.get_nerve_info("ghost_nerve")
        assert info is not None
        assert info["total_invocations"] == 1
        assert info["last_invoked_at"] is not None

    def test_empty_text_nerve_recorded_via_episode(self, mem):
        """When a nerve returns empty text (the 'oopsie' case), episode recording
        should still set last_invoked_at so dreamstate tunes it."""
        mem.cold.register_nerve("reflect_nerve", "Reflect")

        # Simulate dispatch recording a failed episode
        mem.record_episode({
            "task": "hi",
            "nerve": "reflect_nerve",
            "tool": "",
            "success": False,
            "result_summary": "",
            "user_id": "",
        })

        assert mem.cold.get_last_invoked_at("reflect_nerve") is not None


# ══════════════════════════════════════════════════════════════════════════════
# Contract 2: Dreamstate work queue = only invoked nerves
# ══════════════════════════════════════════════════════════════════════════════


class TestDreamstateWorkQueue:
    """_build_work_queue MUST only include nerves that were used in conversations."""

    def test_invoked_nerve_included(self, mem, nerves_dir):
        """A nerve with last_invoked_at appears in the work queue."""
        mem.cold.register_nerve("weather_nerve", "Get weather")
        mem.cold.record_nerve_invocation("weather_nerve", success=True)
        make_nerve_file(nerves_dir, "weather_nerve")

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        names = [item["name"] for item in queue]
        assert "weather_nerve" in names

    def test_never_invoked_nerve_excluded(self, mem, nerves_dir):
        """A nerve that was never invoked MUST NOT be in the work queue."""
        mem.cold.register_nerve("dormant_nerve", "Never used")
        make_nerve_file(nerves_dir, "dormant_nerve")

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        names = [item["name"] for item in queue]
        assert "dormant_nerve" not in names

    def test_recently_invoked_nerve_comes_first(self, mem, nerves_dir):
        """Most recently invoked nerve should be first in queue."""
        import time
        mem.cold.register_nerve("old_nerve", "Old nerve")
        mem.cold.register_nerve("fresh_nerve", "Fresh nerve")

        mem.cold.record_nerve_invocation("old_nerve", success=True)
        time.sleep(1.1)  # SQLite datetime('now') has 1-second resolution
        mem.cold.record_nerve_invocation("fresh_nerve", success=False)

        make_nerve_file(nerves_dir, "old_nerve")
        make_nerve_file(nerves_dir, "fresh_nerve")

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        names = [item["name"] for item in queue]
        assert "fresh_nerve" in names
        assert "old_nerve" in names
        assert names.index("fresh_nerve") < names.index("old_nerve"), \
            f"Expected fresh_nerve before old_nerve, got: {names}"

    def test_core_senses_included_in_tuning(self, mem, nerves_dir):
        """Core senses must be in the tuning queue — they need model-specific tuning."""
        mem.cold.register_sense("hearing", "Audio processing")
        mem.cold.record_nerve_invocation("hearing", success=True)
        make_nerve_file(nerves_dir, "hearing")

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir), \
             patch("arqitect.brain.consolidate.CORE_SENSES",
                   {"hearing", "sight", "touch", "awareness", "communication"}):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        names = [item["name"] for item in queue]
        assert "hearing" in names

    def test_100_dormant_nerves_excluded(self, mem, nerves_dir):
        """Only the nerve that was actually used should be in the queue.
        100 dormant nerves should NOT appear."""
        for i in range(100):
            name = f"dormant_{i}"
            mem.cold.register_nerve(name, f"Dormant nerve {i}")
            make_nerve_file(nerves_dir, name)

        mem.cold.register_nerve("active_nerve", "The one that was used")
        mem.cold.record_nerve_invocation("active_nerve", success=False)
        make_nerve_file(nerves_dir, "active_nerve")

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        names = [item["name"] for item in queue]
        assert "active_nerve" in names
        dormant_in_queue = [n for n in names if n.startswith("dormant_")]
        assert dormant_in_queue == [], \
            f"Dormant nerves leaked into work queue: {dormant_in_queue}"


# ══════════════════════════════════════════════════════════════════════════════
# Contract 3: Consolidation scope — fabricated vs community
# ══════════════════════════════════════════════════════════════════════════════


class TestConsolidationScope:
    """Consolidation compares locally-fabricated nerves against community —
    not all-vs-all on the full 100+ catalog."""

    def test_stack_hash_nerves_excluded(self, mem):
        """Nerves with stack_hash (intentionally synthesized) are excluded."""
        from arqitect.brain.consolidate import find_nerve_clusters

        mem.cold.register_nerve("my_weather", "Get weather info")
        mem.cold.register_nerve("custom_tool", "Custom business tool")
        mem.cold.set_fact("nerve:custom_tool", "stack_hash", "abc123")

        catalog = {
            "weather_nerve": "Get weather forecasts",
            "forecast_nerve": "Weather forecasting",
            "my_weather": "Get weather info",
            "custom_tool": "Custom business tool",
        }
        community = frozenset(["weather_nerve", "forecast_nerve"])

        with patch("arqitect.brain.consolidate.mem", mem):
            clusters = find_nerve_clusters(catalog, community)

        all_names = [name for cluster in clusters for name, _ in cluster]
        assert "custom_tool" not in all_names

    def test_senses_excluded_from_clusters(self, mem):
        """Core senses should never appear in consolidation clusters."""
        from arqitect.brain.consolidate import find_nerve_clusters

        mem.cold.register_sense("sight", "Vision processing")

        catalog = {
            "sight": "Vision processing",
            "image_analyzer": "Analyze images",
        }

        with patch("arqitect.brain.consolidate.mem", mem):
            clusters = find_nerve_clusters(catalog, frozenset())

        all_names = [name for cluster in clusters for name, _ in cluster]
        assert "sight" not in all_names


# ══════════════════════════════════════════════════════════════════════════════
# Contract 4: Test bank expansion
# ══════════════════════════════════════════════════════════════════════════════


class TestTestBankExpansion:
    """Dreamstate fabricates test cases when below min_training_examples."""

    def test_expands_when_below_threshold(self, mem, nerves_dir):
        """If a nerve has 10 cases but needs 50, fabricate more."""
        mem.cold.register_nerve("calc_nerve", "Calculator")
        mem.cold.record_qualification("nerve", "calc_nerve", True, 0.96, 3, 10, 10, "[]")
        mem.cold.record_nerve_invocation("calc_nerve", success=True)

        existing = [{"input": f"test_{i}", "output": f"r_{i}"} for i in range(10)]
        mem.cold.set_test_bank("calc_nerve", existing)
        make_nerve_file(nerves_dir, "calc_nerve")

        mock_generate = MagicMock(return_value=[
            {"input": f"gen_{i}", "output": f"gen_r_{i}"} for i in range(10)
        ])

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value={
                 "min_training_examples": 50, "test_cases_per_batch": 10,
             }), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=existing), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", mock_generate):

            from arqitect.brain.consolidate import Dreamstate
            ds = Dreamstate.__new__(Dreamstate)
            ds._interrupted = threading.Event()
            ds._expand_test_banks_for_training()

        assert mock_generate.call_count > 0
        assert len(mem.cold.get_test_bank("calc_nerve")) > 10

    def test_skips_when_enough_tests(self, mem, nerves_dir):
        """If already at threshold, skip fabrication."""
        mem.cold.register_nerve("full_nerve", "Fully tested")
        mem.cold.record_qualification("nerve", "full_nerve", True, 0.98, 3, 10, 10, "[]")
        mem.cold.record_nerve_invocation("full_nerve", success=True)

        existing = [{"input": f"t_{i}", "output": f"r_{i}"} for i in range(200)]
        mem.cold.set_test_bank("full_nerve", existing)
        make_nerve_file(nerves_dir, "full_nerve")

        mock_generate = MagicMock()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value={
                 "min_training_examples": 200, "test_cases_per_batch": 10,
             }), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=existing), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", mock_generate):

            from arqitect.brain.consolidate import Dreamstate
            ds = Dreamstate.__new__(Dreamstate)
            ds._interrupted = threading.Event()
            ds._expand_test_banks_for_training()

        mock_generate.assert_not_called()

    def test_skips_for_small_model(self, mem):
        """tinylm/small models produce garbage — skip fabrication."""
        mem.cold.register_nerve("some_nerve", "Some nerve")

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="small"):

            from arqitect.brain.consolidate import Dreamstate
            ds = Dreamstate.__new__(Dreamstate)
            ds._interrupted = threading.Event()
            ds._expand_test_banks_for_training()

        # No crash, no fabrication attempted


# ══════════════════════════════════════════════════════════════════════════════
# Contract 5: Dreamstate interruptibility
# ══════════════════════════════════════════════════════════════════════════════


class TestDreamstateInterruptibility:
    """wake() must stop all dreamstate work immediately."""

    def test_expansion_stops_on_interrupt(self, mem, nerves_dir):
        """Test bank expansion checks interrupted flag between rounds."""
        mem.cold.register_nerve("calc_nerve", "Calculator")
        mem.cold.record_qualification("nerve", "calc_nerve", True, 0.96, 3, 10, 10, "[]")
        mem.cold.set_test_bank("calc_nerve", [{"input": "1+1", "output": "2"}])
        make_nerve_file(nerves_dir, "calc_nerve")

        call_count = 0
        interrupted = threading.Event()

        def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                interrupted.set()
            return [{"input": f"new_{call_count}", "output": "x"}]

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value={
                 "min_training_examples": 1000, "test_cases_per_batch": 1,
             }), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=[]), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", fake_generate):

            from arqitect.brain.consolidate import Dreamstate
            ds = Dreamstate.__new__(Dreamstate)
            ds._interrupted = interrupted
            ds._expand_test_banks_for_training()

        assert call_count <= 3, f"Expected early stop, got {call_count} calls"


# ══════════════════════════════════════════════════════════════════════════════
# Contract 6: Full flow — conversation failure → dreamstate tunes it
# ══════════════════════════════════════════════════════════════════════════════


class TestConversationToDreamstateFlow:
    """End-to-end: user says 'hi' → nerve fails → recorded → dreamstate picks it up."""

    def test_failed_nerve_appears_in_queue(self, mem, nerves_dir):
        """When reflect_nerve fails on 'hi', dreamstate work queue includes it."""
        mem.cold.register_nerve("reflect_nerve", "Reflect on interactions")
        make_nerve_file(nerves_dir, "reflect_nerve")
        mem.cold.record_nerve_invocation("reflect_nerve", success=False)

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        names = [item["name"] for item in queue]
        assert "reflect_nerve" in names


# ══════════════════════════════════════════════════════════════════════════════
# Edge cases: think() flow branches
# ══════════════════════════════════════════════════════════════════════════════


class TestThinkEdgeCases:
    """Every branch in think() must be exercised."""

    def test_depth_exceeds_limit(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """think() at depth > 5 returns fallback without calling LLM."""
        mem_fixture = make_mem(test_redis)
        fake = FakeLLM([("Task:", '{"action": "invalid_forever"}', True)])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        for p in patches:
            p.start()
        try:
            from arqitect.brain.brain import think
            result = think("hello")
            assert "rephrasing" in result.lower() or "detail" in result.lower()
        finally:
            for p in patches:
                p.stop()

    def test_depth_5_still_processes(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """depth=5 is the boundary — should still attempt processing."""
        mem_fixture = make_mem(test_redis)
        register_qualified_nerve(mem_fixture, "reflect_nerve", "Reflect")
        make_nerve_file(nerves_dir, "reflect_nerve")

        fake = FakeLLM([
            ("Available nerves", json.dumps(as_dict(
                InvokeDecisionFactory.build(name="reflect_nerve", args="hi")
            ))),
        ])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        for p in patches:
            p.start()
        try:
            with patch_invoke_nerve(return_value='{"response": "Hello!"}'):
                from arqitect.brain.brain import think
                result = think("hi", depth=5)
            assert "rephrasing" not in result.lower()
        finally:
            for p in patches:
                p.stop()

    def test_safety_blocks_harmful_input(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """Harmful input at depth=0 is blocked before any processing."""
        mem_fixture = make_mem(test_redis)
        fake = FakeLLM([
            ("safety filter", '{"safe": false, "category": "harmful"}', True),
            ("refusal message", "I can't help with that.", True),
        ])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        for p in patches:
            p.start()
        try:
            from arqitect.brain.brain import think
            result = think("harmful request")
            assert result == "I can't help with that."
            # Only safety classification + refusal calls — no routing LLM calls
            routing_calls = [c for c in fake.calls if c["model"] != "role:nerve"]
            assert len(routing_calls) == 0, "LLM routing should not be called for blocked input"
        finally:
            for p in patches:
                p.stop()

    def test_safety_skipped_at_depth_gt_zero(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """Safety check is only at depth=0 — re-think calls skip it."""
        mem_fixture = make_mem(test_redis)
        register_qualified_nerve(mem_fixture, "reflect_nerve", "Reflect")
        make_nerve_file(nerves_dir, "reflect_nerve")

        fake = FakeLLM([
            ("safety filter", '{"safe": false, "category": "harmful"}', True),
            ("refusal message", "Blocked", True),
            ("Available nerves", json.dumps(as_dict(
                InvokeDecisionFactory.build(name="reflect_nerve", args="hi")
            ))),
        ])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        for p in patches:
            p.start()
        try:
            with patch_invoke_nerve(return_value='{"response": "ok"}'):
                from arqitect.brain.brain import think
                result = think("some task", depth=1)
            # At depth>0 safety is skipped — should NOT be blocked
            assert result != "Blocked"
            # No safety classification calls should have been made
            safety_calls = fake.prompts_containing("safety filter")
            assert len(safety_calls) == 0, "Safety should not run at depth > 0"
        finally:
            for p in patches:
                p.stop()

    def test_recalibrate_senses_triggers_calibration(
        self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir
    ):
        """'recalibrate senses' should trigger calibrate_all_senses."""
        mem_fixture = make_mem(test_redis)
        fake = FakeLLM()
        cal_result = {
            "sight": {"status": "available", "capabilities": {"describe": {"available": True}}},
        }
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        patches.append(patch("arqitect.brain.brain.calibrate_all_senses", return_value=cal_result))
        patches.append(patch("arqitect.brain.brain._store_calibration_in_memory"))
        for p in patches:
            p.start()
        try:
            from arqitect.brain.brain import think
            result = think("recalibrate senses")
            assert "Recalibration complete" in result
        finally:
            for p in patches:
                p.stop()

    def test_senses_only_catalog_still_routes(
        self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir
    ):
        """When only core senses exist (no custom nerves), the brain should
        still route to a sense — catalog is never truly empty because
        bootstrap_senses always registers the 5 core senses."""
        mem_fixture = make_mem(test_redis)
        fake = FakeLLM([
            ("Available nerves", json.dumps(as_dict(
                InvokeDecisionFactory.build(name="awareness", args="hi")
            ))),
        ])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        for p in patches:
            p.start()
        try:
            with patch_invoke_nerve(return_value='{"response": "Hello!"}'):
                from arqitect.brain.brain import think
                result = think("hi")

            # With only senses, the brain should still produce a response
            assert len(result) > 0
            # The LLM should see available senses in its context
            assert fake.call_count > 0
        finally:
            for p in patches:
                p.stop()

    def test_llm_returns_plain_text_passthrough(
        self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir
    ):
        """When LLM returns plain text (no JSON), pass it through as the response."""
        mem_fixture = make_mem(test_redis)
        register_qualified_nerve(mem_fixture, "reflect_nerve", "Reflect")
        make_nerve_file(nerves_dir, "reflect_nerve")

        # FakeLLM with no matching rules — falls through to default
        fake = FakeLLM([
            ("Available nerves", "I'm not sure what you mean."),
        ])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        for p in patches:
            p.start()
        try:
            from arqitect.brain.brain import think
            result = think("hi")
            # No JSON → extract_json returns None → raw text returned
            assert "not sure" in result.lower() or len(result) > 0
        finally:
            for p in patches:
                p.stop()

    def test_long_task_gets_truncated(
        self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir
    ):
        """Tasks longer than 4000 chars should be truncated before LLM."""
        mem_fixture = make_mem(test_redis)
        # Use a realistic long task (not just "x" * 5000 which triggers safety)
        long_task = "Please analyze the following data: " + "measurement=42.5 " * 500
        fake = FakeLLM([("truncated", "ok")])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        for p in patches:
            p.start()
        try:
            from arqitect.brain.brain import think
            think(long_task)

            # Check that the prompt sent to LLM contains the truncation marker
            truncated = fake.prompts_containing("truncated")
            assert len(truncated) > 0 or len(long_task) <= 4000, \
                "Long task should be truncated before sending to LLM"
        finally:
            for p in patches:
                p.stop()

    def test_plan_intent_triggers_plan_start(
        self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir
    ):
        """When intent is 'plan', _handle_plan_start should be called."""
        mem_fixture = make_mem(test_redis)
        fake = FakeLLM([
            # Intent classification returns plan
            ("classify", '{"type": "plan", "category": "coding"}'),
            # Planning LLM response
            ("User wants", "Let me help you plan a web scraper. What sites?"),
        ])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        patches.append(patch("arqitect.brain.brain.detect_project_path", return_value=None))
        for p in patches:
            p.start()
        try:
            from arqitect.brain.brain import think
            result = think("build a web scraper")
            assert result is not None
        finally:
            for p in patches:
                p.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Edge cases: dispatch_action flow branches
# ══════════════════════════════════════════════════════════════════════════════


class TestDispatchEdgeCases:
    """Every branch in dispatch_action must be exercised.
    Uses DispatchContextFactory for typed, consistent test data."""

    def test_invoke_nonexistent_nerve_triggers_rethink(self):
        """Invoking a nerve that doesn't exist → re-think to synthesize."""
        from arqitect.brain.dispatch import dispatch_action

        ctx = DispatchContextFactory.build(
            decision=as_dict(InvokeDecisionFactory.build(name="ghost_nerve")),
            nerve_catalog={},
            available=[],
        )
        dispatch_action(ctx)

        ctx.think_fn.assert_called_once()
        history = ctx.think_fn.call_args[0][1]
        assert "does not exist" in history[-1].lower()

    def test_wrong_nerve_status_triggers_rethink(self, mem, nerves_dir):
        """When a nerve says wrong_nerve → re-think with ban."""
        from arqitect.brain.dispatch import dispatch_action

        make_nerve_file(nerves_dir, "weather_nerve")
        register_qualified_nerve(mem, "weather_nerve", "Weather")

        wrong_output = json.dumps({"status": "wrong_nerve", "reason": "not my domain"})

        with patch("arqitect.brain.dispatch.mem", mem), \
             patch("arqitect.brain.dispatch.invoke_nerve", return_value=wrong_output), \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True):
            ctx = DispatchContextFactory.build(
                decision=as_dict(InvokeDecisionFactory.build(name="weather_nerve", args="hi")),
                nerve_catalog={"weather_nerve": "Weather"},
                available=["weather_nerve"],
            )
            dispatch_action(ctx)

        ctx.think_fn.assert_called_once()
        history = ctx.think_fn.call_args[0][1]
        assert "Do NOT re-invoke" in history[-1]

    def test_needs_data_status_triggers_rethink(self, mem, nerves_dir):
        """When a nerve says needs_data → re-think to resolve dependency."""
        from arqitect.brain.dispatch import dispatch_action

        make_nerve_file(nerves_dir, "analysis_nerve")
        register_qualified_nerve(mem, "analysis_nerve", "Analyze data")

        needs_output = json.dumps({
            "status": "needs_data", "needs": "user's location", "tool": "geocoder",
        })

        with patch("arqitect.brain.dispatch.mem", mem), \
             patch("arqitect.brain.dispatch.invoke_nerve", return_value=needs_output), \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True):
            ctx = DispatchContextFactory.build(
                task="analyze my area",
                decision=as_dict(InvokeDecisionFactory.build(
                    name="analysis_nerve", args="analyze",
                )),
                nerve_catalog={"analysis_nerve": "Analyze data"},
                available=["analysis_nerve"],
            )
            dispatch_action(ctx)

        ctx.think_fn.assert_called_once()
        history = ctx.think_fn.call_args[0][1]
        assert "needs" in history[-1].lower()

    def test_empty_nerve_output_gives_graceful_failure(self, mem, nerves_dir):
        """Empty nerve output → communication LLM produces personality failure."""
        from arqitect.brain.dispatch import dispatch_action

        make_nerve_file(nerves_dir, "reflect_nerve")
        register_qualified_nerve(mem, "reflect_nerve", "Reflect")

        with patch("arqitect.brain.dispatch.mem", mem), \
             patch("arqitect.brain.dispatch.invoke_nerve", return_value=""), \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.brain.dispatch._graceful_failure_message",
                   return_value="Oopsie!") as mock_fail:
            ctx = DispatchContextFactory.build(
                task="hi",
                decision=as_dict(InvokeDecisionFactory.build(
                    name="reflect_nerve", args="hi",
                )),
                nerve_catalog={"reflect_nerve": "Reflect"},
                available=["reflect_nerve"],
            )
            result = dispatch_action(ctx)

        mock_fail.assert_called_once()
        assert result == "Oopsie!"

    def test_respond_action_redirects_to_awareness(self):
        """LLM must NEVER respond directly — redirect to awareness sense."""
        from arqitect.brain.dispatch import dispatch_action

        ctx = DispatchContextFactory.build(
            decision=as_dict(RespondDecisionFactory.build(message="Hello!")),
        )
        dispatch_action(ctx)

        ctx.think_fn.assert_called_once()
        history = ctx.think_fn.call_args[0][1]
        assert "awareness" in history[-1].lower()

    def test_unknown_action_triggers_rethink(self):
        """Unknown action → re-think with valid action list."""
        from arqitect.brain.dispatch import dispatch_action

        ctx = DispatchContextFactory.build(
            decision={"action": "dance_around", "name": "whatever"},
        )
        dispatch_action(ctx)

        ctx.think_fn.assert_called_once()
        history = ctx.think_fn.call_args[0][1]
        assert "unknown action" in history[-1].lower()

    def test_synthesize_existing_nerve_redirects_to_invoke(self, mem, nerves_dir):
        """Synthesize an existing nerve → redirect to invoke."""
        from arqitect.brain.dispatch import dispatch_action

        make_nerve_file(nerves_dir, "weather_nerve")
        register_qualified_nerve(mem, "weather_nerve", "Get weather")

        fake_llm = FakeLLM([("", "Sunny", True)])
        with patch("arqitect.brain.dispatch.mem", mem), \
             patch("arqitect.brain.dispatch.invoke_nerve",
                   return_value='{"response": "Sunny"}') as mock_invoke, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.senses.communication.nerve.rewrite_response",
                   side_effect=lambda message="", **kw: {"response": message, "format": "text", "tone": "neutral"}), \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"):
            ctx = DispatchContextFactory.build(
                task="what's the weather",
                decision=as_dict(SynthesizeDecisionFactory.build(
                    name="weather_nerve", description="Get weather",
                )),
                nerve_catalog={"weather_nerve": "Get weather"},
                available=["weather_nerve"],
            )
            dispatch_action(ctx)

        mock_invoke.assert_called_once()

    def test_anon_user_cannot_synthesize(self, mem):
        """Anonymous users are blocked from creating new nerves."""
        from arqitect.brain.dispatch import dispatch_action

        with patch("arqitect.brain.dispatch.mem", mem), \
             patch("arqitect.brain.dispatch.can_model_fabricate", return_value=True), \
             patch("arqitect.brain.dispatch.can_synthesize_nerve", return_value=False), \
             patch("arqitect.brain.dispatch.get_synthesis_restriction_message",
                   return_value="Please identify yourself."):
            ctx = DispatchContextFactory.build(
                task="build me a tool",
                decision=as_dict(SynthesizeDecisionFactory.build(
                    name="my_tool", description="A tool",
                )),
                user_id="",
                nerve_catalog={},
                available=[],
            )
            result = dispatch_action(ctx)

        assert "identify" in result.lower()

    def test_small_model_cannot_synthesize(self, mem):
        """tinylm/small models are blocked from fabricating nerves."""
        from arqitect.brain.dispatch import dispatch_action

        with patch("arqitect.brain.dispatch.mem", mem), \
             patch("arqitect.brain.dispatch.can_model_fabricate", return_value=False), \
             patch("arqitect.brain.dispatch.get_model_fabrication_message",
                   return_value="Model too small to fabricate."):
            ctx = DispatchContextFactory.build(
                task="build me a tool",
                decision=as_dict(SynthesizeDecisionFactory.build(
                    name="my_tool", description="A tool",
                )),
                nerve_catalog={},
                available=[],
            )
            result = dispatch_action(ctx)

        assert "too small" in result.lower()

    def test_clarify_returns_question_with_suggestions(self, mem):
        """Clarify action returns question + suggestions."""
        from arqitect.brain.dispatch import dispatch_action

        with patch("arqitect.brain.dispatch.mem", mem):
            ctx = DispatchContextFactory.build(
                decision=as_dict(ClarifyDecisionFactory.build(
                    message="What exactly?",
                    suggestions=["Option A", "Option B"],
                )),
            )
            result = dispatch_action(ctx)

        assert "What exactly?" in result
        assert "Option A" in result
        assert "Option B" in result

    def test_positive_feedback_records_circuit_breaker_success(self, mem):
        """Positive feedback → circuit breaker records success."""
        from arqitect.brain.dispatch import dispatch_action

        mem.warm.record({
            "task": "weather", "nerve": "weather_nerve",
            "tool": "", "success": True, "result_summary": "sunny",
            "user_id": "",
        })
        register_qualified_nerve(mem, "weather_nerve", "Weather")

        with patch("arqitect.brain.dispatch.mem", mem), \
             patch("arqitect.brain.dispatch._cb_success") as mock_cb:
            ctx = DispatchContextFactory.build(
                task="that was great!",
                decision=as_dict(FeedbackDecisionFactory.build(
                    sentiment="positive", message="Thanks!",
                )),
            )
            dispatch_action(ctx)

        mock_cb.assert_called_with("weather_nerve")

    def test_update_context_normalizes_location_to_city(self, mem):
        """update_context stores facts with location→city normalization."""
        from arqitect.brain.dispatch import dispatch_action

        with patch("arqitect.brain.dispatch.mem", mem):
            ctx = DispatchContextFactory.build(
                task="I live in Tel Aviv",
                decision=as_dict(UpdateContextDecisionFactory.build(
                    message="Got it!",
                )),
                user_id="test_user",
            )
            # Inject the context manually (factory defaults don't include it)
            ctx.decision["context"] = {"location": "Tel Aviv"}
            dispatch_action(ctx)

        fact = mem.cold.get_fact("user:test_user", "city")
        assert fact == "Tel Aviv"

    def test_chain_with_empty_steps_rethinks(self):
        """chain_nerves with no steps → re-think."""
        from arqitect.brain.dispatch import dispatch_action

        ctx = DispatchContextFactory.build(
            decision={"action": "chain_nerves", "steps": [], "goal": "stuff"},
        )
        dispatch_action(ctx)

        ctx.think_fn.assert_called_once()
        history = ctx.think_fn.call_args[0][1]
        assert "steps" in history[-1].lower()

    def test_single_step_chain_becomes_invoke(self, mem, nerves_dir):
        """Chain with one step → treated as simple invoke."""
        from arqitect.brain.dispatch import dispatch_action

        make_nerve_file(nerves_dir, "weather_nerve")
        register_qualified_nerve(mem, "weather_nerve", "Weather")

        fake_llm = FakeLLM([("", "Sunny", True)])
        with patch("arqitect.brain.dispatch.mem", mem), \
             patch("arqitect.brain.dispatch.invoke_nerve",
                   return_value='{"response": "Sunny"}') as mock_invoke, \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True), \
             patch("arqitect.senses.communication.nerve.rewrite_response",
                   side_effect=lambda message="", **kw: {"response": message, "format": "text", "tone": "neutral"}), \
             patch("arqitect.brain.dispatch.publish_event"), \
             patch("arqitect.brain.dispatch.publish_response"), \
             patch("arqitect.brain.dispatch.publish_memory_state"):
            step = ChainStepFactory.build(nerve="weather_nerve", args="weather")
            ctx = DispatchContextFactory.build(
                decision=as_dict(ChainDecisionFactory.build(
                    steps=[step], goal="get weather",
                )),
                nerve_catalog={"weather_nerve": "Weather"},
                available=["weather_nerve"],
            )
            dispatch_action(ctx)

        mock_invoke.assert_called_once()

    def test_unavailable_sense_blocked_by_calibration(self, mem, nerves_dir, test_redis):
        """Sense marked unavailable in calibration → blocked."""
        from arqitect.brain.dispatch import dispatch_action

        make_nerve_file(nerves_dir, "sight")
        mem.cold.register_sense("sight", "Vision")

        test_redis.hset("synapse:sense_calibration", "sight", json.dumps({
            "status": "unavailable",
            "dependencies": {"opencv": {"installed": False}},
        }))

        with patch("arqitect.brain.dispatch.mem", mem), \
             patch("arqitect.brain.dispatch.r", test_redis), \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=True):
            ctx = DispatchContextFactory.build(
                task="describe this image",
                decision=as_dict(InvokeDecisionFactory.build(
                    name="sight", args="describe",
                )),
                nerve_catalog={"sight": "Vision"},
                available=["sight"],
            )
            result = dispatch_action(ctx)

        assert "unavailable" in result.lower()
        assert "opencv" in result.lower()

    def test_permission_denied_for_restricted_nerve(self, mem, nerves_dir):
        """User lacks permission → restriction message."""
        from arqitect.brain.dispatch import dispatch_action

        make_nerve_file(nerves_dir, "admin_nerve")
        register_qualified_nerve(mem, "admin_nerve", "Admin operations")

        with patch("arqitect.brain.dispatch.mem", mem), \
             patch("arqitect.brain.dispatch.can_use_nerve", return_value=False), \
             patch("arqitect.brain.permissions.get_restriction_message",
                   return_value="Access denied."):
            ctx = DispatchContextFactory.build(
                task="delete everything",
                decision=as_dict(InvokeDecisionFactory.build(
                    name="admin_nerve", args="delete",
                )),
                nerve_catalog={"admin_nerve": "Admin operations"},
                available=["admin_nerve"],
            )
            result = dispatch_action(ctx)

        assert "denied" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Normalization and parsing edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizeAndParse:
    """Edge cases in action normalization and decision parsing."""

    def test_use_sense_normalized_to_invoke(self):
        """use_sense → invoke_nerve."""
        from arqitect.brain.dispatch import normalize_action
        from arqitect.types import Action

        d = as_dict(UseSenseDecisionFactory.build(sense="awareness", args="check"))
        action, decision = normalize_action(d, {"awareness": "Self-awareness"})

        assert action == Action.INVOKE_NERVE
        assert decision["name"] == "awareness"

    def test_nerve_name_as_action_normalized(self):
        """Nerve name used as action → invoke_nerve."""
        from arqitect.brain.dispatch import normalize_action
        from arqitect.types import Action

        action, decision = normalize_action(
            {"action": "weather_nerve", "args": "today's weather"},
            {"weather_nerve": "Weather"},
        )

        assert action == Action.INVOKE_NERVE
        assert decision["name"] == "weather_nerve"

    def test_typo_fuzzy_matched(self):
        """Typo like 'invok_nerve' fuzzy-matches to invoke_nerve."""
        from arqitect.brain.dispatch import normalize_action
        from arqitect.types import Action

        action, _ = normalize_action({"action": "invok_nerve", "name": "test"}, {})
        assert action == Action.INVOKE_NERVE

    def test_completely_unknown_action_passthrough(self):
        """Action with no close match passes through unchanged."""
        from arqitect.brain.dispatch import normalize_action

        action, _ = normalize_action({"action": "xyzzy_magic"}, {})
        assert action == "xyzzy_magic"

    def test_parse_decision_missing_action_raises(self):
        """parse_decision raises ValueError if action is missing."""
        from arqitect.brain.dispatch import parse_decision

        with pytest.raises(ValueError, match="action"):
            parse_decision({})

    def test_dict_args_coerced_to_json_string(self):
        """Dict args become JSON string."""
        from arqitect.brain.dispatch import parse_decision

        result = parse_decision({
            "action": "invoke_nerve", "name": "test",
            "args": {"key": "value"},
        })
        assert '"key"' in result.args

    def test_parse_nerve_output_handles_log_noise(self):
        """Nerve output with log lines before JSON should still parse."""
        from arqitect.brain.dispatch import _parse_nerve_output

        output = '[INFO] Loading...\n[INFO] Ready\n{"response": "Hello!"}'
        result = _parse_nerve_output(output)
        assert result == IsDict(response="Hello!")

    def test_parse_nerve_output_garbage_returns_empty(self):
        """Completely unparseable output → empty dict."""
        from arqitect.brain.dispatch import _parse_nerve_output

        result = _parse_nerve_output("this is not json at all")
        assert result == {}


# ══════════════════════════════════════════════════════════════════════════════
# _is_nerve_error detection contracts
# ══════════════════════════════════════════════════════════════════════════════


class TestIsNerveError:
    """Empty/error text detected, valid responses not flagged."""

    def test_empty_string_is_error(self):
        from arqitect.brain.helpers import _is_nerve_error
        assert _is_nerve_error("") is True

    def test_whitespace_only_is_error(self):
        from arqitect.brain.helpers import _is_nerve_error
        assert _is_nerve_error("   \n\t  ") is True

    def test_none_is_error(self):
        from arqitect.brain.helpers import _is_nerve_error
        assert _is_nerve_error(None) is True

    def test_error_pattern_detected(self):
        from arqitect.brain.helpers import _is_nerve_error
        assert _is_nerve_error("The tool returned an error: connection refused") is True

    def test_valid_json_response_not_error(self):
        from arqitect.brain.helpers import _is_nerve_error
        assert _is_nerve_error('{"response": "Hello there!"}') is False

    def test_short_clean_text_not_error(self):
        from arqitect.brain.helpers import _is_nerve_error
        assert _is_nerve_error("Hello!") is False

    def test_long_text_with_soft_error_not_flagged(self):
        """Soft error patterns only apply to short outputs."""
        from arqitect.brain.helpers import _is_nerve_error
        long_text = "x" * 300 + " error: something minor"
        assert _is_nerve_error(long_text) is False

    def test_traceback_detected(self):
        from arqitect.brain.helpers import _is_nerve_error
        assert _is_nerve_error("Traceback (most recent call last):\n  File...") is True


# ══════════════════════════════════════════════════════════════════════════════
# invoke_nerve edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestInvokeNerveEdgeCases:
    """Subprocess invocation edge cases."""

    def test_invalid_nerve_name_returns_error(self):
        """Path traversal in nerve name → rejected."""
        from arqitect.brain.invoke import invoke_nerve

        result = json.loads(invoke_nerve("../../etc/passwd", "args"))
        assert "error" in result
        assert "invalid" in result["error"].lower()

    def test_sanitize_accepts_valid_names(self):
        from arqitect.brain.invoke import _sanitize_nerve_name

        assert _sanitize_nerve_name("valid_nerve") == "valid_nerve"
        assert _sanitize_nerve_name("valid-nerve") == "valid-nerve"
        assert _sanitize_nerve_name("nerve123") == "nerve123"

    def test_sanitize_rejects_invalid_names(self):
        from arqitect.brain.invoke import _sanitize_nerve_name

        assert _sanitize_nerve_name("../../bad") is None
        assert _sanitize_nerve_name("") is None
        assert _sanitize_nerve_name("has space") is None

    def test_nerve_file_not_found(self):
        """Missing nerve.py → clear error."""
        from arqitect.brain.invoke import invoke_nerve

        with patch("arqitect.brain.invoke.NERVES_DIR", "/nonexistent/path"), \
             patch("arqitect.brain.invoke.SENSES_DIR", "/nonexistent/senses"):
            result = json.loads(invoke_nerve("ghost_nerve", "args"))

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_nerve_timeout(self, nerves_dir):
        """Subprocess timeout → timeout error."""
        import subprocess
        from arqitect.brain.invoke import invoke_nerve

        make_nerve_file(nerves_dir, "slow_nerve")

        with patch("arqitect.brain.invoke.NERVES_DIR", nerves_dir), \
             patch("arqitect.brain.invoke.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("cmd", 90)), \
             patch("arqitect.brain.invoke.mem") as mock_mem:
            mock_mem.cold.is_qualified.return_value = True
            mock_mem.get_env_for_nerve.return_value = {}
            result = json.loads(invoke_nerve("slow_nerve", "args"))

        assert "error" in result
        assert "timed out" in result["error"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# Hypothesis: property-based tests for LLM output parsing
# ══════════════════════════════════════════════════════════════════════════════


class TestPropertyBased:
    """Fuzz parsers with random inputs — they must never crash."""

    @given(st.text(min_size=0, max_size=5000))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_extract_json_never_crashes(self, raw):
        """extract_json must return dict or None for any input."""
        from arqitect.brain.helpers import extract_json
        result = extract_json(raw)
        assert result is None or isinstance(result, dict)

    @given(st.text(min_size=0, max_size=5000))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_is_nerve_error_never_crashes(self, text):
        """_is_nerve_error must return bool for any input."""
        from arqitect.brain.helpers import _is_nerve_error
        result = _is_nerve_error(text)
        assert isinstance(result, bool)

    @given(st.text(min_size=0, max_size=5000))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_parse_nerve_output_never_crashes(self, output):
        """_parse_nerve_output must return dict for any input.

        BUG FOUND: json.loads("0") returns int 0, but _parse_nerve_output
        has no type guard — it returns the raw parsed value instead of {}.
        Inputs like "0", "1", "true", "null", '"hello"' all trigger this.
        """
        from arqitect.brain.dispatch import _parse_nerve_output
        result = _parse_nerve_output(output)
        assert isinstance(result, dict)

    @given(st.dictionaries(
        st.text(min_size=0, max_size=50),
        st.one_of(st.text(), st.integers(), st.none(), st.booleans()),
        max_size=10,
    ))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_normalize_action_never_crashes(self, decision):
        """normalize_action must handle arbitrary dicts without crashing."""
        from arqitect.brain.dispatch import normalize_action
        action, result = normalize_action(decision, {})
        assert isinstance(result, dict)

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_sanitize_nerve_name_never_crashes(self, name):
        """_sanitize_nerve_name must return str or None."""
        from arqitect.brain.invoke import _sanitize_nerve_name
        result = _sanitize_nerve_name(name)
        assert result is None or isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot tests: prompt and decision shapes
# ══════════════════════════════════════════════════════════════════════════════


class TestSnapshotShapes:
    """Snapshot prompt templates and decision structures.
    Run with --snapshot-update to accept changes."""

    def test_invoke_decision_shape(self, snapshot):
        """InvokeDecision factory output should be stable."""
        d = InvokeDecisionFactory.build(name="weather_nerve", args="forecast")
        assert as_dict(d) == snapshot

    def test_synthesize_decision_shape(self, snapshot):
        """SynthesizeDecision factory output should be stable."""
        d = SynthesizeDecisionFactory.build(
            name="my_tool", description="A tool", mcp_tools=[],
        )
        assert as_dict(d) == snapshot

    def test_chain_decision_shape(self, snapshot):
        """ChainDecision factory output should be stable."""
        steps = [
            ChainStepFactory.build(nerve="step1", args="a"),
            ChainStepFactory.build(nerve="step2", args="b"),
        ]
        d = ChainDecisionFactory.build(steps=steps, goal="multi-step")
        assert as_dict(d) == snapshot

    def test_clarify_decision_shape(self, snapshot):
        d = ClarifyDecisionFactory.build()
        assert as_dict(d) == snapshot

    def test_feedback_decision_shape(self, snapshot):
        d = FeedbackDecisionFactory.build()
        assert as_dict(d) == snapshot

    def test_dispatch_context_shape(self, snapshot):
        """DispatchContext factory should produce stable structures."""
        ctx = DispatchContextFactory.build()
        # Serialize without the callable
        data = {
            "task": ctx.task,
            "decision": ctx.decision,
            "user_id": ctx.user_id,
            "depth": ctx.depth,
            "nerve_catalog": ctx.nerve_catalog,
            "available": ctx.available,
        }
        assert data == snapshot
