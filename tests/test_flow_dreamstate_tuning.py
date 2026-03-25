"""Flow tests — dreamstate tuning with real memory and real control flow.

These test the 4 production flows the user described:

1. Dreamstate only tunes nerves that were actually used in conversation
   (_build_work_queue filters by last_invoked_at != NULL).

2. Even when conversation switches between nerves, every nerve that participated
   gets recorded as active (record_nerve_invocation sets last_invoked_at).

3. Medium-brain dreamstate fabricates 200 test cases for a nerve that has fewer
   (_expand_test_banks_for_training generates in batches).

4. Medium-brain dreamstate skips fabrication when the nerve already has 200 cases
   (len(current_data) >= min_needed → continue).

All tests use:
- Real MemoryManager (fakeredis + temp SQLite) — no memory mocks
- Real _build_work_queue / _expand_test_banks_for_training logic
- Mocked only: LLM calls, filesystem, adapters config
"""

from __future__ import annotations

import json
import os
import threading
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import make_nerve_file, register_qualified_nerve


# ---------------------------------------------------------------------------
# Flow 1: Dreamstate only tunes nerves that were actually invoked
# ---------------------------------------------------------------------------

@pytest.mark.timeout(30)
class TestDreamstateOnlyTunesInvokedNerves:
    """_build_work_queue must exclude nerves that were never invoked.

    Production scenario: brain has 10 registered nerves but only 2 were used
    in conversation. Dreamstate should only spend LLM calls tuning those 2.
    """

    def test_never_invoked_nerve_excluded_from_work_queue(self, mem, nerves_dir):
        """A registered nerve with no invocations must NOT appear in the queue."""
        register_qualified_nerve(mem, "unused_nerve", "does nothing")
        make_nerve_file(nerves_dir, "unused_nerve")

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir), \
             patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        nerve_names = [item["name"] for item in queue]
        assert "unused_nerve" not in nerve_names

    def test_invoked_nerve_included_in_work_queue(self, mem, nerves_dir):
        """A nerve that was invoked (has last_invoked_at) MUST appear in the queue."""
        register_qualified_nerve(mem, "active_nerve", "does stuff")
        make_nerve_file(nerves_dir, "active_nerve")

        # Simulate a real invocation — this sets last_invoked_at via SQLite datetime('now')
        mem.cold.record_nerve_invocation("active_nerve", success=True)

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir), \
             patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        nerve_names = [item["name"] for item in queue]
        assert "active_nerve" in nerve_names

    def test_mix_of_invoked_and_dormant(self, mem, nerves_dir):
        """With 5 nerves, only the 2 that were invoked appear in the queue."""
        all_nerves = ["weather", "calendar", "email", "notes", "timer"]
        invoked = {"weather", "email"}

        for name in all_nerves:
            register_qualified_nerve(mem, name, f"{name} nerve")
            make_nerve_file(nerves_dir, name)

        # Only invoke 2 of them
        for name in invoked:
            mem.cold.record_nerve_invocation(name, success=True)

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir), \
             patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        queued_names = {item["name"] for item in queue}
        assert queued_names == invoked

    def test_senses_included_when_invoked(self, mem, nerves_dir):
        """Core senses must appear in the tuning queue when invoked —
        they need model-specific prompt tuning just like nerves."""
        from arqitect.brain.config import CORE_SENSES

        for sense in CORE_SENSES:
            mem.cold.register_sense(sense, f"{sense} sense")
            make_nerve_file(nerves_dir, sense)
            mem.cold.record_nerve_invocation(sense, success=True)

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir), \
             patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        queued_names = {item["name"] for item in queue}
        for sense in CORE_SENSES:
            assert sense in queued_names

    def test_high_scoring_nerve_excluded_even_if_invoked(self, mem, nerves_dir):
        """A nerve scoring above the improvement threshold is already good — skip it."""
        mem.cold.register_nerve("perfect_nerve", "already great")
        mem.cold.record_qualification("nerve", "perfect_nerve", qualified=True,
                                       score=0.98, iterations=5, test_count=20, pass_count=20)
        make_nerve_file(nerves_dir, "perfect_nerve")
        mem.cold.record_nerve_invocation("perfect_nerve", success=True)

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir), \
             patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        assert "perfect_nerve" not in [item["name"] for item in queue]

    def test_expand_test_banks_only_processes_invoked_nerves(self, mem):
        """BUG TEST: _expand_test_banks_for_training should only process nerves
        that were actually invoked, not ALL non-sense nerves.

        Production bug: brain registers 10 community nerves, user invokes 1,
        but dreamstate tries to fabricate test cases for all 10 — wasting LLM calls
        and processing nerves the user has never touched.
        """
        # Register 3 qualified nerves — only 1 was invoked
        for name in ["weather", "calendar", "email"]:
            register_qualified_nerve(mem, name, f"{name} nerve")

        # Only weather was actually used in conversation
        mem.cold.record_nerve_invocation("weather", success=True)

        generate_calls = []
        batch_counter = [0]
        def fake_generate(name, description, role="tool", existing_inputs=None):
            generate_calls.append(name)
            batch_counter[0] += 1
            start = batch_counter[0] * 15
            return [{"input": f"gen_{start + i}", "expected": "ok"} for i in range(15)]

        fake_cfg = {
            "min_training_examples": 200,
            "test_cases_per_batch": 15,
            "improvement_threshold": 0.95,
        }

        from arqitect.brain.consolidate import Dreamstate
        with patch.object(Dreamstate, "__init__", lambda self: None):
            worker = Dreamstate.__new__(Dreamstate)
            worker._interrupted = threading.Event()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_cfg), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=[]), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", side_effect=fake_generate):
            worker._expand_test_banks_for_training()

        # Only weather should have had test cases generated
        nerves_that_got_generated = set(generate_calls)
        assert "calendar" not in nerves_that_got_generated, \
            "calendar was never invoked — should NOT get test cases generated"
        assert "email" not in nerves_that_got_generated, \
            "email was never invoked — should NOT get test cases generated"
        assert "weather" in nerves_that_got_generated, \
            "weather WAS invoked — SHOULD get test cases generated"


# ---------------------------------------------------------------------------
# Flow 2: Conversation with nerve switching records all active nerves
# ---------------------------------------------------------------------------

@pytest.mark.timeout(30)
class TestConversationRecordsActiveNerves:
    """record_episode → cold.record_nerve_invocation → sets last_invoked_at.

    Production scenario: user asks "what's the weather and set a timer".
    Brain routes to weather_nerve first, then timer_nerve.
    Both must get last_invoked_at set so dreamstate can tune them.
    """

    def test_single_invocation_sets_last_invoked_at(self, mem):
        """After one episode recording, the nerve has a non-NULL last_invoked_at."""
        mem.cold.register_nerve("weather", "get weather")

        # Before invocation — no timestamp
        assert mem.cold.get_last_invoked_at("weather") is None

        # Record episode (same path as dispatch._handle_invoke)
        mem.record_episode({
            "task": "what's the weather",
            "nerve": "weather",
            "tool": "",
            "success": True,
            "result_summary": "sunny 25C",
        })

        assert mem.cold.get_last_invoked_at("weather") is not None

    def test_switching_nerves_records_both(self, mem):
        """When conversation switches between nerves, BOTH get last_invoked_at."""
        mem.cold.register_nerve("weather", "get weather")
        mem.cold.register_nerve("timer", "set timer")

        # First nerve in conversation
        mem.record_episode({
            "task": "weather and timer",
            "nerve": "weather",
            "tool": "",
            "success": True,
            "result_summary": "sunny",
        })

        # Second nerve in same conversation
        mem.record_episode({
            "task": "weather and timer",
            "nerve": "timer",
            "tool": "",
            "success": True,
            "result_summary": "timer set",
        })

        assert mem.cold.get_last_invoked_at("weather") is not None
        assert mem.cold.get_last_invoked_at("timer") is not None

    def test_failed_invocation_also_records_last_invoked_at(self, mem):
        """Even a failed nerve invocation must set last_invoked_at for tuning."""
        mem.cold.register_nerve("flaky_nerve", "sometimes fails")

        mem.record_episode({
            "task": "do something",
            "nerve": "flaky_nerve",
            "tool": "",
            "success": False,
            "result_summary": "error: timeout",
        })

        # Failed invocation still counts — dreamstate should tune this nerve
        assert mem.cold.get_last_invoked_at("flaky_nerve") is not None

    def test_multiple_nerves_in_chain_all_recorded(self, mem):
        """A 4-nerve chain must record all 4 as active for tuning."""
        nerves = ["lookup", "transform", "validate", "store"]
        for name in nerves:
            mem.cold.register_nerve(name, f"{name} step")

        # Simulate chain execution — each step records an episode
        for name in nerves:
            mem.record_episode({
                "task": "process data pipeline",
                "nerve": name,
                "tool": "",
                "success": True,
                "result_summary": f"{name} done",
            })

        for name in nerves:
            assert mem.cold.get_last_invoked_at(name) is not None, \
                f"Nerve '{name}' should have last_invoked_at after chain execution"

    def test_uninvolved_nerve_stays_null_during_conversation(self, mem):
        """Nerves not part of the conversation must keep NULL last_invoked_at."""
        mem.cold.register_nerve("weather", "get weather")
        mem.cold.register_nerve("bystander", "not used")

        mem.record_episode({
            "task": "weather check",
            "nerve": "weather",
            "tool": "",
            "success": True,
            "result_summary": "sunny",
        })

        assert mem.cold.get_last_invoked_at("weather") is not None
        assert mem.cold.get_last_invoked_at("bystander") is None

    def test_filesystem_only_nerve_gets_last_invoked_at(self, mem, nerves_dir):
        """BUG TEST: A nerve on disk but NOT in nerve_registry must still get
        last_invoked_at set when invoked.

        Production bug: nerve exists as nerves/weather/nerve.py (from synthesis
        or manual creation), gets invoked by dispatch, record_episode calls
        cold.record_nerve_invocation — but that does UPDATE WHERE name=?, which
        matches 0 rows because the nerve was never register_nerve'd.
        Result: last_invoked_at stays NULL, dreamstate never tunes it.
        """
        # Nerve exists on disk but NOT registered in DB
        make_nerve_file(nerves_dir, "fs_only_nerve")

        # Simulate what dispatch._handle_invoke does after successful invocation
        mem.record_episode({
            "task": "do the thing",
            "nerve": "fs_only_nerve",
            "tool": "",
            "success": True,
            "result_summary": "done",
        })

        # This SHOULD be non-NULL — the nerve was used in conversation
        assert mem.cold.get_last_invoked_at("fs_only_nerve") is not None, \
            "record_nerve_invocation does UPDATE (not UPSERT) — " \
            "a nerve not in nerve_registry silently gets 0 rows updated, " \
            "so last_invoked_at stays NULL and dreamstate never tunes it"

    def test_invoked_nerves_appear_in_work_queue_after_conversation(self, mem, nerves_dir):
        """End-to-end: record episodes → build_work_queue → only active nerves queued."""
        active = ["weather", "email"]
        dormant = ["calendar", "notes"]

        for name in active + dormant:
            register_qualified_nerve(mem, name, f"{name} nerve")
            make_nerve_file(nerves_dir, name)

        # Simulate conversation that uses weather then email
        for name in active:
            mem.record_episode({
                "task": "check weather and send email",
                "nerve": name,
                "tool": "",
                "success": True,
                "result_summary": f"{name} done",
            })

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.config.loader.get_nerves_dir", return_value=nerves_dir), \
             patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95):
            from arqitect.brain.consolidate import _build_work_queue
            queue = _build_work_queue()

        queued = {item["name"] for item in queue}
        assert queued == set(active), \
            f"Only conversation-active nerves should be queued, got {queued}"


# ---------------------------------------------------------------------------
# Flow 3: Medium brain fabricates 200 test cases when nerve has fewer
# ---------------------------------------------------------------------------

@pytest.mark.timeout(30)
class TestDreamstateExpandsTestBank:
    """_expand_test_banks_for_training generates test cases up to min_training_examples.

    Production scenario: qualified nerve "weather" has only 10 test cases.
    Brain is medium-sized. Dreamstate should call generate_test_cases in batches
    of 15 until the bank reaches 200.
    """

    def _make_consolidator(self):
        """Create a Dreamstate instance without triggering timers."""
        from arqitect.brain.consolidate import Dreamstate
        with patch.object(Dreamstate, "__init__", lambda self: None):
            worker = Dreamstate.__new__(Dreamstate)
            worker._interrupted = threading.Event()
        return worker

    def test_medium_brain_generates_test_cases_for_underfilled_nerve(self, mem):
        """A medium brain must generate test cases until bank reaches 200."""
        # Register a qualified nerve with only 10 test cases, mark as invoked
        register_qualified_nerve(mem, "weather", "get weather")
        mem.cold.record_nerve_invocation("weather", success=True)
        existing_tests = [{"input": f"test_{i}", "expected": "ok"} for i in range(10)]
        mem.cold.set_test_bank("weather", existing_tests)

        # generate_test_cases returns batches of 15 new unique tests
        batch_counter = [0]
        def fake_generate(name, description, role="tool", existing_inputs=None):
            batch_counter[0] += 1
            start = 10 + (batch_counter[0] - 1) * 15
            return [{"input": f"gen_{start + i}", "expected": "ok"} for i in range(15)]

        fake_tuning_cfg = {
            "min_training_examples": 200,
            "test_cases_per_batch": 15,
            "improvement_threshold": 0.95,
        }

        worker = self._make_consolidator()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_tuning_cfg), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=[]), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", side_effect=fake_generate):
            worker._expand_test_banks_for_training()

        final_bank = mem.cold.get_test_bank("weather")
        assert len(final_bank) >= 200, \
            f"Bank should have >= 200 tests after expansion, got {len(final_bank)}"

    def test_generate_called_in_batches_of_15(self, mem):
        """generate_test_cases must be called with batch_size=15 (from config)."""
        register_qualified_nerve(mem, "timer", "set timer")
        mem.cold.record_nerve_invocation("timer", success=True)
        mem.cold.set_test_bank("timer", [])

        generate_calls = []
        def fake_generate(name, description, role="tool", existing_inputs=None):
            call_num = len(generate_calls)
            generate_calls.append({"name": name, "existing_inputs": existing_inputs})
            start = call_num * 15
            return [{"input": f"gen_{start + i}", "expected": "ok"} for i in range(15)]

        fake_cfg = {
            "min_training_examples": 200,
            "test_cases_per_batch": 15,
            "improvement_threshold": 0.95,
        }

        worker = self._make_consolidator()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_cfg), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=[]), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", side_effect=fake_generate):
            worker._expand_test_banks_for_training()

        # At least ceil(200/15) = 14 calls needed to fill 200 from empty
        assert len(generate_calls) >= 13, \
            f"Expected many batch calls, got {len(generate_calls)}"
        # Each call should pass growing existing_inputs to avoid duplicates
        for i, call in enumerate(generate_calls):
            if i > 0:
                assert len(call["existing_inputs"]) > 0, \
                    f"Call {i} should have non-empty existing_inputs"

    def test_small_brain_does_not_generate_test_cases(self, mem):
        """A small brain must NOT generate test cases (garbage output)."""
        register_qualified_nerve(mem, "weather", "get weather")
        mem.cold.record_nerve_invocation("weather", success=True)
        mem.cold.set_test_bank("weather", [])

        generate_mock = MagicMock()
        fake_cfg = {"min_training_examples": 200, "test_cases_per_batch": 15, "improvement_threshold": 0.95}

        worker = self._make_consolidator()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="small"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_cfg), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=[]), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", generate_mock):
            worker._expand_test_banks_for_training()

        generate_mock.assert_not_called()
        assert len(mem.cold.get_test_bank("weather")) == 0

    def test_large_brain_also_generates(self, mem):
        """A large brain should also generate test cases (not just medium)."""
        register_qualified_nerve(mem, "email", "send email")
        mem.cold.record_nerve_invocation("email", success=True)
        mem.cold.set_test_bank("email", [])

        batch_counter = [0]
        def fake_generate(name, description, role="tool", existing_inputs=None):
            batch_counter[0] += 1
            start = (batch_counter[0] - 1) * 15
            return [{"input": f"gen_{start + i}", "expected": "ok"} for i in range(15)]

        fake_cfg = {"min_training_examples": 200, "test_cases_per_batch": 15, "improvement_threshold": 0.95}

        worker = self._make_consolidator()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="large"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_cfg), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=[]), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", side_effect=fake_generate):
            worker._expand_test_banks_for_training()

        assert len(mem.cold.get_test_bank("email")) >= 200

    def test_unqualified_nerve_skipped(self, mem):
        """Unqualified nerves must not get test cases generated."""
        # Register but do NOT qualify
        mem.cold.register_nerve("draft_nerve", "work in progress")
        mem.cold.set_test_bank("draft_nerve", [])

        generate_mock = MagicMock()
        fake_cfg = {"min_training_examples": 200, "test_cases_per_batch": 15, "improvement_threshold": 0.95}

        worker = self._make_consolidator()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_cfg), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=[]), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", generate_mock):
            worker._expand_test_banks_for_training()

        generate_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Flow 4: Medium brain skips fabrication when nerve already has 200 cases
# ---------------------------------------------------------------------------

@pytest.mark.timeout(30)
class TestDreamstateSkipsFullTestBank:
    """_expand_test_banks_for_training skips when len(current_data) >= min_needed.

    Production scenario: qualified nerve "weather" already has 200+ test cases.
    Dreamstate should detect this and not waste LLM calls generating more.
    """

    def _make_consolidator(self):
        from arqitect.brain.consolidate import Dreamstate
        with patch.object(Dreamstate, "__init__", lambda self: None):
            worker = Dreamstate.__new__(Dreamstate)
            worker._interrupted = threading.Event()
        return worker

    def test_nerve_with_200_training_examples_skipped(self, mem):
        """When collect_training_data returns 200 items, generate_test_cases is NOT called."""
        register_qualified_nerve(mem, "weather", "get weather")
        mem.cold.record_nerve_invocation("weather", success=True)
        mem.cold.set_test_bank("weather", [{"input": f"t{i}"} for i in range(50)])

        generate_mock = MagicMock()
        # 200 existing training examples — already enough
        existing_data = [{"input": f"data_{i}", "output": "ok"} for i in range(200)]
        fake_cfg = {"min_training_examples": 200, "test_cases_per_batch": 15, "improvement_threshold": 0.95}

        worker = self._make_consolidator()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_cfg), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=existing_data), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", generate_mock):
            worker._expand_test_banks_for_training()

        generate_mock.assert_not_called()

    def test_nerve_with_more_than_200_also_skipped(self, mem):
        """Even with 300 examples (well over 200), generation is skipped."""
        register_qualified_nerve(mem, "email", "send email")
        mem.cold.record_nerve_invocation("email", success=True)

        generate_mock = MagicMock()
        existing_data = [{"input": f"d_{i}", "output": "ok"} for i in range(300)]
        fake_cfg = {"min_training_examples": 200, "test_cases_per_batch": 15, "improvement_threshold": 0.95}

        worker = self._make_consolidator()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_cfg), \
             patch("arqitect.inference.tuner.collect_training_data", return_value=existing_data), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", generate_mock):
            worker._expand_test_banks_for_training()

        generate_mock.assert_not_called()

    def test_one_nerve_full_one_underfilled(self, mem):
        """When one nerve has 200 and another has 10, only the underfilled one gets expanded."""
        register_qualified_nerve(mem, "full_nerve", "already full")
        register_qualified_nerve(mem, "empty_nerve", "needs tests")
        mem.cold.record_nerve_invocation("full_nerve", success=True)
        mem.cold.record_nerve_invocation("empty_nerve", success=True)
        mem.cold.set_test_bank("full_nerve", [])
        mem.cold.set_test_bank("empty_nerve", [])

        def fake_collect(name):
            if name == "full_nerve":
                return [{"input": f"d_{i}"} for i in range(200)]
            return []  # empty_nerve has no training data

        generate_calls = []
        def fake_generate(name, description, role="tool", existing_inputs=None):
            generate_calls.append(name)
            return [{"input": f"gen_{len(generate_calls)}_{i}", "expected": "ok"} for i in range(15)]

        fake_cfg = {"min_training_examples": 200, "test_cases_per_batch": 15, "improvement_threshold": 0.95}

        worker = self._make_consolidator()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_cfg), \
             patch("arqitect.inference.tuner.collect_training_data", side_effect=fake_collect), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", side_effect=fake_generate):
            worker._expand_test_banks_for_training()

        # full_nerve should never appear in generate calls
        assert "full_nerve" not in generate_calls, \
            "full_nerve already has 200 examples — should NOT generate more"
        # empty_nerve should have been expanded
        assert "empty_nerve" in generate_calls, \
            "empty_nerve has 0 examples — SHOULD generate test cases"

    def test_exactly_at_threshold_skipped(self, mem):
        """Exactly 200 examples (== min_needed) should also skip."""
        register_qualified_nerve(mem, "exact", "exactly at threshold")
        mem.cold.record_nerve_invocation("exact", success=True)

        generate_mock = MagicMock()
        fake_cfg = {"min_training_examples": 200, "test_cases_per_batch": 15, "improvement_threshold": 0.95}

        worker = self._make_consolidator()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_cfg), \
             patch("arqitect.inference.tuner.collect_training_data",
                   return_value=[{"input": f"d_{i}"} for i in range(200)]), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", generate_mock):
            worker._expand_test_banks_for_training()

        generate_mock.assert_not_called()

    def test_199_examples_triggers_generation(self, mem):
        """199 examples (one short of 200) must trigger generation."""
        register_qualified_nerve(mem, "almost", "almost there")
        mem.cold.record_nerve_invocation("almost", success=True)
        mem.cold.set_test_bank("almost", [])

        generate_calls = []
        def fake_generate(name, description, role="tool", existing_inputs=None):
            generate_calls.append(name)
            return [{"input": f"gen_{len(generate_calls)}_{i}", "expected": "ok"} for i in range(15)]

        fake_cfg = {"min_training_examples": 200, "test_cases_per_batch": 15, "improvement_threshold": 0.95}

        worker = self._make_consolidator()

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"), \
             patch("arqitect.brain.adapters.get_tuning_config", return_value=fake_cfg), \
             patch("arqitect.inference.tuner.collect_training_data",
                   return_value=[{"input": f"d_{i}"} for i in range(199)]), \
             patch("arqitect.critic.qualify_nerve.generate_test_cases", side_effect=fake_generate):
            worker._expand_test_banks_for_training()

        assert "almost" in generate_calls, \
            "199 examples is one short of 200 — SHOULD trigger generation"
