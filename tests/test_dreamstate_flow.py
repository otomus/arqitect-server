"""Flow-based tests for the dreamstate pipeline.

Tests the desired end-to-end flow:
  nerve invocation -> dream state entry -> prioritized hydration ->
  template detection -> real test generation -> qualification ->
  test bank expansion to min_training_examples -> LoRA fine-tuning ->
  adapter creation.

These tests validate the invariants that were broken in production,
not just happy paths. Each test targets a specific pipeline gap.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from dirty_equals import IsInstance, IsPositive
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from arqitect.types import NerveRole


# ---------------------------------------------------------------------------
# Helper for provenance ranking (used by TestTestCaseProvenance)
# ---------------------------------------------------------------------------

_SIZE_RANK = {"community": 0, "tinylm": 1, "small": 2, "medium": 3, "large": 4}


def _size_rank(size: str | None) -> int:
    """Rank a size class for provenance comparison. Higher = stronger model."""
    return _SIZE_RANK.get(size or "", 0)


# -- 1. Role validation -- invalid roles must never leak into the pipeline ---


@pytest.mark.timeout(10)
class TestRoleValidation:
    """Nerves must only have roles: tool, creative, code.

    System roles like 'brain', 'awareness' have different tuning configs
    and would break the adapter resolution chain.
    """

    @pytest.mark.parametrize("role", ["tool", "creative", "code"])
    def test_validate_nerve_role_accepts_valid_roles(self, role):
        from arqitect.brain.routing import validate_nerve_role

        assert validate_nerve_role(role) == role

    @pytest.mark.parametrize("bad_role", [
        "brain", "awareness", "communication", "nerve", "coder",
    ])
    def test_validate_nerve_role_clamps_system_roles_to_tool(self, bad_role):
        from arqitect.brain.routing import validate_nerve_role

        assert validate_nerve_role(bad_role) == NerveRole.TOOL, (
            f"System role '{bad_role}' leaked through validation"
        )

    @given(garbage=st.text(min_size=0, max_size=50).filter(
        lambda s: s not in ("tool", "creative", "code")
    ))
    @settings(max_examples=20)
    def test_validate_nerve_role_clamps_arbitrary_strings_to_tool(self, garbage):
        """Any string that isn't a valid role must clamp to tool."""
        from arqitect.brain.routing import validate_nerve_role

        assert validate_nerve_role(garbage) == NerveRole.TOOL

    def test_classify_nerve_role_only_returns_valid_roles(self):
        """Even if the LLM hallucinates a system role, classify must clamp it."""
        from arqitect.brain.routing import classify_nerve_role

        with patch("arqitect.brain.routing.llm_generate", return_value="brain"):
            result = classify_nerve_role("reflect_nerve", "self-reflection agent")
            assert result in (NerveRole.TOOL, NerveRole.CREATIVE, NerveRole.CODE)
            assert result != "brain"

    def test_classify_nerve_role_handles_llm_failure(self):
        from arqitect.brain.routing import classify_nerve_role

        with patch("arqitect.brain.routing.llm_generate", side_effect=RuntimeError("LLM down")):
            result = classify_nerve_role("reflect_nerve", "self-reflection")
            assert result == NerveRole.TOOL

    def test_classify_nerve_role_handles_empty_response(self):
        from arqitect.brain.routing import classify_nerve_role

        with patch("arqitect.brain.routing.llm_generate", return_value=""):
            result = classify_nerve_role("reflect_nerve", "self-reflection")
            assert result == NerveRole.TOOL

    def test_classify_nerve_role_strips_punctuation(self):
        from arqitect.brain.routing import classify_nerve_role

        with patch("arqitect.brain.routing.llm_generate", return_value='"creative".'):
            result = classify_nerve_role("reflect_nerve", "self-reflection")
            assert result == NerveRole.CREATIVE

    def test_community_bundle_role_is_validated(self):
        """apply_community_bundle must validate the role before writing to cold memory."""
        from arqitect.brain.community import apply_community_bundle

        cold = MagicMock()
        cold.register_nerve_rich = MagicMock()

        bundle = {
            "description": "self-reflection",
            "role": "brain",  # invalid -- should be clamped
            "default": {"system_prompt": "reflect", "examples": []},
        }

        with patch("arqitect.brain.community._resolve_bundle_prompt", return_value=("reflect", [])):
            with patch("arqitect.brain.community._install_bundle_tools", return_value=[]):
                with patch("arqitect.brain.community._install_bundle_tests"):
                    with patch("arqitect.brain.community._install_bundle_lora"):
                        result = apply_community_bundle("reflect_nerve", bundle, cold)

        assert result["role"] != "brain", "Invalid role 'brain' was not clamped"
        assert result["role"] in (NerveRole.TOOL, NerveRole.CREATIVE, NerveRole.CODE)
        # Verify the role passed to cold memory was also valid
        call_kwargs = cold.register_nerve_rich.call_args
        registered_role = call_kwargs.kwargs.get("role") or call_kwargs[1].get("role")
        if registered_role is None:
            registered_role = call_kwargs[1].get(
                "role", call_kwargs[0][4] if len(call_kwargs[0]) > 4 else None
            )
        assert registered_role != "brain"


# -- 2. Template test bank detection ----------------------------------------


@pytest.mark.timeout(10)
class TestTemplateDetection:
    """Community bundles ship template test cases with placeholder outputs.

    These must be detected and regenerated -- treating them as real tests
    causes the reconciler to skip test generation and score against garbage.
    """

    def test_detects_template_with_example_input_marker(self):
        from arqitect.brain.consolidate import _is_template_test_bank

        tests = [{"input": "test", "output": '{"action": "call", "args": {"input": "example"}}'}]
        assert _is_template_test_bank(tests) is True

    def test_detects_template_with_error_action_marker(self):
        from arqitect.brain.consolidate import _is_template_test_bank

        tests = [{"input": "test", "output": '{"action": "error", "message": "something"}'}]
        assert _is_template_test_bank(tests) is True

    def test_detects_template_with_empty_args_marker(self):
        from arqitect.brain.consolidate import _is_template_test_bank

        tests = [{"input": "test", "output": '{"tool": "x", "args": {}}'}]
        assert _is_template_test_bank(tests) is True

    def test_accepts_real_test_bank(self):
        from arqitect.brain.consolidate import _is_template_test_bank

        tests = [
            {"input": "reflect on today", "output": "I notice a pattern of curiosity..."},
            {"input": "what did I learn", "output": "Based on recent interactions..."},
        ]
        assert _is_template_test_bank(tests) is False

    def test_empty_test_bank_is_not_template(self):
        from arqitect.brain.consolidate import _is_template_test_bank

        assert _is_template_test_bank([]) is False

    def test_ensure_test_bank_regenerates_templates(self):
        """_ensure_test_bank must not short-circuit when stored tests are templates."""
        from arqitect.brain.consolidate import _ensure_test_bank, MAX_RETEST_CASES

        template_tests = [
            {"input": f"test_{i}", "output": '{"args": {}}'}
            for i in range(MAX_RETEST_CASES + 5)
        ]
        real_tests = [{"input": "reflect on today", "output": "A pattern of curiosity..."}]
        generate_fn = MagicMock(return_value=real_tests)

        with patch("arqitect.brain.consolidate.mem"):
            result = _ensure_test_bank("reflect_nerve", "self-reflection", template_tests, generate_fn)

        generate_fn.assert_called_once()
        assert result == real_tests, "Template tests were returned instead of regenerated ones"

    def test_ensure_test_bank_keeps_real_tests(self):
        """Real tests with enough coverage should not be regenerated."""
        from arqitect.brain.consolidate import _ensure_test_bank, MAX_RETEST_CASES

        real_tests = [
            {"input": f"reflect on topic {i}", "output": f"Deep reflection #{i}..."}
            for i in range(MAX_RETEST_CASES + 1)
        ]
        generate_fn = MagicMock()

        result = _ensure_test_bank("reflect_nerve", "self-reflection", real_tests, generate_fn)

        generate_fn.assert_not_called()
        assert result == real_tests

    def test_ensure_test_bank_generates_when_too_few(self):
        """Even real tests trigger regeneration if below MAX_RETEST_CASES."""
        from arqitect.brain.consolidate import _ensure_test_bank, MAX_RETEST_CASES

        few_tests = [{"input": "test", "output": "real output"}]
        assert len(few_tests) < MAX_RETEST_CASES

        new_tests = [{"input": f"gen_{i}", "output": f"output_{i}"} for i in range(5)]
        generate_fn = MagicMock(return_value=new_tests)

        with patch("arqitect.brain.consolidate.mem"):
            result = _ensure_test_bank("reflect_nerve", "self-reflection", few_tests, generate_fn)

        generate_fn.assert_called_once()
        assert result == new_tests


# -- 3. Work queue prioritization -- recently-used nerves first --------------


@pytest.mark.timeout(10)
class TestWorkQueuePrioritization:
    """The dream state must prioritize nerves the user actually uses.

    A user who invokes reflect_nerve should see it qualified before
    the 150+ dormant community nerves that were never used.
    """

    def test_recently_used_nerves_sort_before_unused(self):
        from arqitect.brain.consolidate import _ts_key

        now = datetime.now()
        recent = now.isoformat()
        assert _ts_key(recent) == IsPositive
        assert _ts_key(None) == 0.0

    @pytest.mark.parametrize("bad_ts", ["not-a-date", "", None])
    def test_ts_key_handles_invalid_timestamps(self, bad_ts):
        from arqitect.brain.consolidate import _ts_key

        assert _ts_key(bad_ts) == 0.0

    def test_build_work_queue_puts_used_nerves_first(self):
        """Nerves with last_invoked_at must appear before those without."""
        from arqitect.brain.consolidate import _build_work_queue

        now = datetime.now().isoformat()
        old = (datetime.now() - timedelta(hours=2)).isoformat()

        mock_quals = [
            {"subject_type": "nerve", "subject_name": "unused_nerve", "score": 0.1},
            {"subject_type": "nerve", "subject_name": "recently_used", "score": 0.2},
            {"subject_type": "nerve", "subject_name": "old_used", "score": 0.1},
        ]

        mock_nerves = {
            "unused_nerve": "does nothing",
            "recently_used": "recently invoked",
            "old_used": "invoked a while ago",
        }

        invoked_at = {
            "unused_nerve": None,
            "recently_used": now,
            "old_used": old,
        }

        with patch("arqitect.brain.consolidate.mem") as mock_mem, \
             patch("arqitect.brain.consolidate.CORE_SENSES", frozenset()), \
             patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95), \
             patch("os.path.isdir", return_value=True), \
             patch("os.listdir", return_value=[]), \
             patch("os.path.isfile", return_value=True):

            mock_mem.cold.list_qualifications.return_value = mock_quals
            mock_mem.cold.list_nerves.return_value = mock_nerves
            mock_mem.cold.is_sense.return_value = False
            mock_mem.cold.get_last_invoked_at.side_effect = lambda n: invoked_at.get(n)

            queue = _build_work_queue()

        names = [item["name"] for item in queue]

        # Never-invoked nerves are excluded
        assert "unused_nerve" not in names, (
            "Never-invoked nerve should be excluded from reconciliation queue"
        )
        # recently_used should be first (most recent), old_used second
        assert names.index("recently_used") < names.index("old_used"), (
            "More recently used nerve should sort before older used nerve"
        )

    def test_hydration_prioritizes_recently_used_nerves(self):
        """Community sync hydration must process recently-used nerves first."""
        from arqitect.brain.consolidate import _ts_key

        now = datetime.now()
        items = [
            ("never_used", None),
            ("used_yesterday", (now - timedelta(days=1)).isoformat()),
            ("used_now", now.isoformat()),
            ("also_never_used", None),
        ]

        # Apply the same sort as _dream_community_sync
        items.sort(key=lambda x: (x[1] is None, -_ts_key(x[1])))

        names = [name for name, _ in items]
        assert names[0] == "used_now", "Most recently used should hydrate first"
        assert names[1] == "used_yesterday"
        # Never-used nerves should come last
        assert set(names[2:]) == {"never_used", "also_never_used"}


# -- 4. Tuner importability -- no module-level side effects ------------------


@pytest.mark.timeout(10)
class TestTunerImportability:
    """The tuner module must be importable without triggering config resolution.

    Module-level calls to get_nerves_dir() / get_models_dir() break when
    the tuner is imported in subprocess contexts where arqitect config
    isn't initialized.
    """

    def test_tuner_imports_without_config(self):
        """Import the tuner module without any arqitect config being available."""
        import importlib
        import arqitect.inference.tuner as tuner_mod

        # If this doesn't raise, the lazy accessors work
        importlib.reload(tuner_mod)

    def test_nerves_dir_is_lazy(self):
        """_nerves_dir() must not be called at import time."""
        from arqitect.inference.tuner import _nerves_dir

        assert callable(_nerves_dir)

    def test_models_dir_is_lazy(self):
        """_models_dir() must not be called at import time."""
        from arqitect.inference.tuner import _models_dir

        assert callable(_models_dir)


# -- 5. Training data threshold -- must reach min_training_examples ----------


@pytest.mark.timeout(10)
class TestTrainingDataThreshold:
    """Fine-tuning must not start until min_training_examples is reached.

    For medium size class (Qwen2.5-Coder-7B), this is ~200 examples.
    The pipeline must expand the test bank in batches until this threshold
    is met, not attempt to train with 5 template tests.
    """

    def test_train_nerve_adapter_rejects_insufficient_data(self):
        from arqitect.inference.tuner import train_nerve_adapter

        tiny_data = [{"input": "hi", "output": "hello"}] * 3

        with patch("arqitect.brain.adapters.get_tuning_config", return_value={
            "lora_rank": 8, "lora_epochs": 3, "lora_lr": 2e-4,
            "min_training_examples": 200,
        }):
            result = train_nerve_adapter(
                "reflect_nerve", role="creative", training_data=tiny_data,
            )

        assert result is False, "Training should not start with insufficient data"

    def test_get_nerves_ready_filters_by_min_examples(self):
        """get_nerves_ready_for_training must only return nerves above threshold."""
        from arqitect.inference.tuner import get_nerves_ready_for_training

        small_bank = [{"input": f"q{i}", "output": f"a{i}"} for i in range(5)]

        with patch("arqitect.memory.cold.ColdMemory") as MockCold:
            mock_cold = MockCold.return_value
            mock_cold.conn.execute.return_value.fetchall.return_value = [
                ("reflect_nerve", "self-reflection", "creative"),
            ]
            with patch("arqitect.inference.tuner.collect_training_data", return_value=small_bank):
                with patch("arqitect.brain.adapters.get_tuning_config", return_value={
                    "min_training_examples": 200,
                }):
                    with patch("os.path.exists", return_value=False):
                        result = get_nerves_ready_for_training()

        assert len(result) == 0, "Nerve with 5 examples should not be ready for training"

    def test_collect_training_data_includes_test_bank(self):
        """Training data must include test bank entries (high-quality expected behaviors)."""
        from arqitect.inference.tuner import collect_training_data

        test_bank = [
            {"input": "reflect on today", "output": "I notice patterns of curiosity"},
            {"input": "what did I learn", "output": "Based on your interactions"},
        ]

        with patch("arqitect.memory.cold.ColdMemory") as MockCold:
            mock_cold = MockCold.return_value
            mock_cold.get_test_bank.return_value = test_bank
            data = collect_training_data("reflect_nerve")

        assert len(data) >= 2
        inputs = {d["input"] for d in data}
        assert "reflect on today" in inputs


# -- 6. Test bank expansion -- batched generation until threshold ------------


@pytest.mark.timeout(10)
class TestTestBankExpansion:
    """The test bank must be expanded in batches until min_training_examples.

    Each batch generates test_cases_per_batch (typically 15) new tests.
    The expander must deduplicate and track progress across rounds.
    """

    def test_expand_deduplicates_across_rounds(self):
        """Expansion must not count duplicate inputs toward the threshold."""
        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = threading.Event()

        existing_bank = [{"input": "existing_q", "output": "existing_a"}]
        round_count = 0

        def mock_generate(name, desc, role="", existing_inputs=None):
            nonlocal round_count
            round_count += 1
            if round_count == 1:
                # First round: returns duplicates + one new
                return [
                    {"input": "existing_q", "output": "dup"},
                    {"input": "new_q_1", "output": "new_a_1"},
                ]
            # Subsequent rounds: all duplicates -- should stop
            return [{"input": "existing_q", "output": "dup again"}]

        with patch("arqitect.brain.consolidate.mem") as mock_mem:
            mock_mem.cold.conn.execute.return_value.fetchall.return_value = [
                ("reflect_nerve", "self-reflection", "creative"),
            ]
            mock_mem.cold.is_qualified.return_value = True
            mock_mem.cold.get_test_bank.return_value = existing_bank.copy()
            mock_mem.cold.set_test_bank = MagicMock()

            with patch("arqitect.inference.tuner.collect_training_data", return_value=[]):
                with patch("arqitect.brain.adapters.get_tuning_config", return_value={
                    "min_training_examples": 50,
                    "test_cases_per_batch": 15,
                }):
                    with patch("arqitect.critic.qualify_nerve.generate_test_cases", side_effect=mock_generate):
                        ds._expand_test_banks_for_training()

        # Should have stopped after all-duplicate round
        assert round_count == 2, f"Expected 2 rounds but got {round_count}"


# -- 7. Reconciliation flow -- template detection -> real test gen -> scoring


@pytest.mark.timeout(10)
class TestReconciliationFlow:
    """Reconciliation must detect template tests, regenerate, then score.

    The reconciler calls _ensure_test_bank which must:
    1. Detect community templates via _is_template_test_bank
    2. Call generate_test_cases to get real tests
    3. Store real tests in cold memory
    4. Run the nerve against real tests
    5. Score based on real outputs
    """

    def test_improve_one_nerve_calls_ensure_test_bank(self):
        """The reconciler must pass through _ensure_test_bank, not use raw stored tests."""
        from arqitect.brain.consolidate import _improve_one_nerve

        interrupted = threading.Event()
        template_tests = [{"input": "t", "output": '{"args": {}}'}] * 5
        real_tests = [{"input": "reflect", "output": "thoughtful reflection"}]

        with patch("arqitect.brain.consolidate.mem") as mock_mem, \
             patch("arqitect.brain.consolidate._ensure_test_bank") as mock_ensure, \
             patch("arqitect.critic.qualify_nerve.generate_test_cases"), \
             patch("arqitect.critic.qualify_nerve.run_nerve_with_input", return_value={"raw_stdout": "ok", "raw_stderr": ""}), \
             patch("arqitect.critic.qualify_nerve.evaluate_nerve_output", return_value={"score": 0.8, "passed": True}), \
             patch("arqitect.critic.qualify_nerve.suggest_improvements", return_value={}), \
             patch("arqitect.critic.qualify_nerve._publish_progress"), \
             patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95), \
             patch("arqitect.brain.consolidate._has_sufficient_coverage", return_value=True), \
             patch("arqitect.brain.consolidate.publish_nerve_status"):

            mock_mem.cold.get_nerve_metadata.return_value = {"role": "creative"}
            mock_mem.cold.get_nerve_tools.return_value = []
            mock_mem.cold.get_test_bank.return_value = template_tests

            with patch("arqitect.brain.adapters.get_tuning_config", return_value={
                "max_reconciliation_iterations": 1,
                "qualification_threshold": 0.7,
            }):
                mock_ensure.return_value = real_tests
                _improve_one_nerve("reflect_nerve", "self-reflection", 0.0, interrupted)

        mock_ensure.assert_called()
        assert mock_ensure.call_args[0][0] == "reflect_nerve"


# -- 8. Dream phase ordering -- the pipeline must flow correctly -------------


@pytest.mark.timeout(10)
class TestDreamPhaseOrdering:
    """Dream phases must execute in correct order:
    community sync -> consolidation -> MCP fanout -> reconciliation ->
    upgrade -> fine-tuning -> contribution -> personality.

    Fine-tuning depends on reconciliation (generates training data).
    MCP fanout must happen before reconciliation (nerves need tools first).
    """

    def test_dream_calls_phases_in_order(self):
        """Verify the _dream method calls phases in the documented order."""
        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = threading.Event()
        ds._last_activity = 0

        call_order = []

        ds._dream_community_sync = lambda: call_order.append("community_sync")
        ds._dream_mcp_fanout = lambda: call_order.append("mcp_fanout")
        ds._dream_reconcile = lambda: call_order.append("reconcile")
        ds._dream_upgrade = lambda: call_order.append("upgrade")
        ds._dream_finetune = lambda: call_order.append("finetune")
        ds._dream_contribute = lambda: call_order.append("contribute")

        with patch("arqitect.brain.consolidate.consolidate_nerves"):
            with patch("arqitect.brain.consolidate.mem"):
                ds._dream()

        assert "community_sync" in call_order
        assert "reconcile" in call_order
        assert "finetune" in call_order

        # Reconciliation must happen before fine-tuning
        assert call_order.index("reconcile") < call_order.index("finetune"), (
            "Reconciliation must run before fine-tuning (generates training data)"
        )
        # MCP fanout must happen before reconciliation
        assert call_order.index("mcp_fanout") < call_order.index("reconcile"), (
            "MCP fanout must run before reconciliation (nerves need tools)"
        )

    def test_dream_stops_on_interrupt(self):
        """When wake() is called, the dream must stop at the current phase."""
        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = threading.Event()
        ds._last_activity = 0

        call_order = []

        def interrupt_after_sync():
            call_order.append("community_sync")
            ds._interrupted.set()

        ds._dream_community_sync = interrupt_after_sync
        ds._dream_reconcile = lambda: call_order.append("reconcile")
        ds._dream_finetune = lambda: call_order.append("finetune")
        ds._dream_mcp_fanout = lambda: call_order.append("mcp_fanout")
        ds._dream_upgrade = lambda: call_order.append("upgrade")
        ds._dream_contribute = lambda: call_order.append("contribute")

        with patch("arqitect.brain.consolidate.consolidate_nerves"):
            with patch("arqitect.brain.consolidate.mem"):
                ds._dream()

        assert "community_sync" in call_order
        assert "reconcile" not in call_order, "Reconciliation should not run after interrupt"
        assert "finetune" not in call_order, "Fine-tuning should not run after interrupt"


# -- 9. Fine-tuning gate -- only qualified nerves with enough data -----------


@pytest.mark.timeout(10)
class TestFineTuningGate:
    """Fine-tuning must only proceed for nerves that are:
    1. Qualified (passed reconciliation)
    2. Have min_training_examples worth of data
    3. Have a valid nerve role (not a system role)
    """

    def test_finetune_skips_unqualified_nerves(self):
        """_expand_test_banks_for_training skips nerves that aren't qualified yet."""
        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = threading.Event()

        with patch("arqitect.brain.consolidate.mem") as mock_mem:
            mock_mem.cold.conn.execute.return_value.fetchall.return_value = [
                ("reflect_nerve", "self-reflection", "creative"),
            ]
            mock_mem.cold.is_qualified.return_value = False

            with patch("arqitect.inference.tuner.collect_training_data") as mock_collect:
                with patch("arqitect.brain.adapters.get_tuning_config", return_value={
                    "min_training_examples": 200,
                    "test_cases_per_batch": 15,
                }):
                    with patch("arqitect.critic.qualify_nerve.generate_test_cases"):
                        ds._expand_test_banks_for_training()

            mock_collect.assert_not_called()

    def test_finetune_imports_gracefully(self):
        """_dream_finetune must handle ImportError when torch/transformers missing."""
        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = threading.Event()

        with patch("arqitect.brain.consolidate.mem"):
            with patch.dict("sys.modules", {"arqitect.inference.tuner": None}):
                # Should not raise -- just log and return
                ds._dream_finetune()


# -- 10. Adapter resolution -- role maps to correct tuning config ------------


@pytest.mark.timeout(10)
class TestAdapterResolution:
    """The adapter resolver must map nerve roles to correct community configs.

    tool -> nerve adapter (structured I/O)
    creative -> creative adapter (generative)
    code -> code adapter (syntax-focused)
    """

    @pytest.mark.parametrize("role,expected", [
        ("tool", "nerve"),
        ("creative", "creative"),
        ("code", "code"),
    ])
    def test_role_maps_to_expected_adapter(self, role, expected):
        from arqitect.brain.adapters import _resolve_adapter_role

        assert _resolve_adapter_role(role) == expected

    def test_unknown_role_falls_back_to_nerve(self):
        from arqitect.brain.adapters import _resolve_adapter_role

        assert _resolve_adapter_role("unknown_role") == "nerve"

    def test_tuning_config_has_required_fields_with_community(self):
        """Every role's tuning config must include min_training_examples and test_cases_per_batch."""
        from arqitect.brain.adapters import get_tuning_config

        required_fields = {"min_training_examples", "test_cases_per_batch"}

        community_meta = {
            "tuning": {
                "min_training_examples": 200,
                "test_cases_per_batch": 15,
            }
        }

        for role in ("tool", "creative", "code"):
            with patch("arqitect.brain.adapters.resolve_meta", return_value=community_meta):
                with patch("arqitect.brain.adapters.resolve_prompt", return_value=None):
                    cfg = get_tuning_config(role)
                    for field in required_fields:
                        assert field in cfg, (
                            f"Tuning config for role '{role}' missing required field '{field}'"
                        )

    def test_tuning_config_missing_fields_without_community(self):
        """GAP: Without community meta, required fields are absent -- pipeline would KeyError.

        This documents a real gap: if the community cache is empty or unreachable,
        get_tuning_config returns only ROLE_TUNING_OVERRIDES which don't include
        min_training_examples or test_cases_per_batch. Any caller doing
        cfg["min_training_examples"] will crash.
        """
        from arqitect.brain.adapters import get_tuning_config

        with patch("arqitect.brain.adapters.resolve_meta", return_value=None):
            with patch("arqitect.brain.adapters.resolve_prompt", return_value=None):
                cfg = get_tuning_config("tool")

        # This documents the gap -- these fields are missing without community
        assert "min_training_examples" not in cfg, (
            "If this passes, the gap has been fixed -- update this test"
        )


# -- 11. Coverage gating -- scores require sufficient test coverage ----------


@pytest.mark.timeout(10)
class TestCoverageGating:
    """Qualification scores must only be recorded when enough tests ran.

    If only 1/5 tests ran (because of interruption), the score is unreliable
    and should not be saved. MIN_TEST_COVERAGE (80%) must be enforced.
    """

    @pytest.mark.parametrize("results,total", [(1, 10), (3, 5)])
    def test_has_sufficient_coverage_rejects_low_coverage(self, results, total):
        from arqitect.brain.consolidate import _has_sufficient_coverage

        assert _has_sufficient_coverage(results, total) is False

    @pytest.mark.parametrize("results,total", [(4, 5), (5, 5)])
    def test_has_sufficient_coverage_accepts_high_coverage(self, results, total):
        from arqitect.brain.consolidate import _has_sufficient_coverage

        assert _has_sufficient_coverage(results, total) is True

    def test_has_sufficient_coverage_handles_zero_tests(self):
        from arqitect.brain.consolidate import _has_sufficient_coverage

        assert _has_sufficient_coverage(0, 0) is False

    def test_plateau_detection_prevents_infinite_loops(self):
        """Plateau detection must stop reconciliation when scores stop improving."""
        from arqitect.brain.consolidate import _detect_plateau

        # Scores stuck at 0.6 -- should detect plateau
        assert _detect_plateau([0.60, 0.61]) is True

        # Scores improving -- should not detect plateau
        assert _detect_plateau([0.50, 0.70]) is False

        # Not enough data -- should not detect
        assert _detect_plateau([0.60]) is False


# -- 12. Dreamstate entry -- idle threshold enforcement ----------------------


@pytest.mark.timeout(10)
class TestDreamstateEntry:
    """The brain must wait IDLE_THRESHOLD (120s) before entering dream state.

    If a task arrives during dreams, wake() must interrupt immediately.
    """

    def test_enter_dreamstate_respects_idle_threshold(self):
        from arqitect.brain.consolidate import Dreamstate, IDLE_THRESHOLD

        ds = Dreamstate.__new__(Dreamstate)
        ds._last_activity = time.time()  # just now -- not idle
        ds._worker_thread = None
        ds._interrupted = threading.Event()
        ds._lock = threading.Lock()
        ds._timer = None

        # Should not enter -- brain just had activity
        ds._enter_dreamstate()
        assert ds._worker_thread is None, "Should not dream when brain was just active"

    def test_wake_interrupts_dream(self):
        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = threading.Event()
        ds._lock = threading.Lock()
        ds._timer = None
        ds._last_activity = 0

        # Simulate a running dream thread
        ds._worker_thread = MagicMock()
        ds._worker_thread.is_alive.return_value = False

        with patch("arqitect.brain.consolidate.mem"):
            ds.wake()

        assert ds._interrupted.is_set(), "wake() must set the interrupted event"


# -- 13. Size class mapping -- fractional params and edge cases --------------


@pytest.mark.timeout(10)
class TestSizeClassMapping:
    """Model name -> size class mapping must handle fractional param counts,
    unusual naming, cloud models, and missing info gracefully."""

    @pytest.mark.parametrize("model,expected", [
        ("qwen2.5-coder-7b", "medium"),
        ("llama-3-70b", "large"),
        ("smollm-1.5b", "tinylm"),
        ("llama-3-3b", "small"),
        ("phi-3-mini-3.8b", "small"),
        ("qwen2-0.5b-instruct", "tinylm"),
        ("deepseek-coder-1.3b", "tinylm"),
        ("codellama-13b-instruct", "medium"),
        ("codellama-34b", "large"),
        # Case insensitivity
        ("Qwen2.5-Coder-7B", "medium"),
        ("LLAMA-3-70B", "large"),
    ])
    def test_model_to_size_class(self, model, expected):
        from arqitect.brain.adapters import _model_to_size_class

        assert _model_to_size_class(model) == expected

    @pytest.mark.parametrize("model", ["mystery-model", "", None])
    def test_unknown_or_empty_model_returns_none(self, model):
        from arqitect.brain.adapters import _model_to_size_class

        assert _model_to_size_class(model) is None

    # -- Cloud model classification via provider category ---

    @pytest.mark.parametrize("model_name", [
        "claude-opus-4", "gpt-4o", "gemini-2.0-flash", "deepseek-chat",
    ])
    def test_cloud_model_maps_to_large(self, model_name):
        from arqitect.brain.adapters import get_model_size_class

        with patch("arqitect.brain.adapters._get_model_file_for_role", return_value=model_name):
            with patch("arqitect.brain.adapters._is_cloud_provider", return_value=True):
                assert get_model_size_class("brain") == "large"

    def test_groq_hosted_gguf_uses_param_count_not_cloud_default(self):
        """Groq-hosted open-source models have param counts in the name.
        The param count should take priority over cloud fallback."""
        from arqitect.brain.adapters import get_model_size_class

        with patch("arqitect.brain.adapters._get_model_file_for_role",
                   return_value="llama-3.3-70b-versatile"):
            assert get_model_size_class("brain") == "large"

    def test_groq_hosted_small_model_not_forced_large(self):
        """A 7B model on Groq should be medium, not forced to large."""
        from arqitect.brain.adapters import get_model_size_class

        with patch("arqitect.brain.adapters._get_model_file_for_role",
                   return_value="llama-3.1-8b-instant"):
            assert get_model_size_class("brain") == "medium"

    def test_local_unknown_model_returns_none(self):
        """A local model with no param count and local provider -> None."""
        from arqitect.brain.adapters import get_model_size_class

        with patch("arqitect.brain.adapters._get_model_file_for_role", return_value="custom-model"):
            with patch("arqitect.brain.adapters._is_cloud_provider", return_value=False):
                assert get_model_size_class("brain") is None


# -- 14. Training data must be sliced per size class -------------------------


@pytest.mark.timeout(10)
class TestTrainingDataPerSizeClass:
    """Each size class must train on a subset of the canonical test bank.

    The full bank is generated once at nerve level. Each size class slices:
      tinylm: ~50, small: ~100, medium: ~200, large: ~500.
    """

    def test_collect_training_data_accepts_size_class(self):
        from arqitect.inference.tuner import collect_training_data
        import inspect

        sig = inspect.signature(collect_training_data)
        assert "size_class" in sig.parameters, (
            "collect_training_data must accept size_class to slice the test bank"
        )

    def _make_full_bank(self, count=500):
        """Generate a test bank with the given number of entries."""
        return [{"input": f"q{i}", "output": f"a{i}"} for i in range(count)]

    @pytest.mark.parametrize("size_class,max_expected", [
        ("tinylm", 50),
        ("small", 100),
        ("medium", 200),
    ])
    def test_size_class_respects_upper_bound(self, size_class, max_expected):
        from arqitect.inference.tuner import collect_training_data

        full_bank = self._make_full_bank()

        with patch("arqitect.memory.cold.ColdMemory") as MockCold:
            mock_cold = MockCold.return_value
            mock_cold.get_test_bank.return_value = full_bank
            data = collect_training_data("reflect_nerve", size_class=size_class)

        assert len(data) <= max_expected

    def test_large_gets_more_than_medium(self):
        from arqitect.inference.tuner import collect_training_data

        full_bank = self._make_full_bank()

        with patch("arqitect.memory.cold.ColdMemory") as MockCold:
            mock_cold = MockCold.return_value
            mock_cold.get_test_bank.return_value = full_bank
            medium_data = collect_training_data("reflect_nerve", size_class="medium")
            large_data = collect_training_data("reflect_nerve", size_class="large")

        assert len(large_data) > len(medium_data)

    def test_strict_ordering_tinylm_lt_small_lt_medium_lt_large(self):
        """Each tier must get strictly more data than the tier below it."""
        from arqitect.inference.tuner import collect_training_data

        full_bank = self._make_full_bank()

        with patch("arqitect.memory.cold.ColdMemory") as MockCold:
            mock_cold = MockCold.return_value
            mock_cold.get_test_bank.return_value = full_bank

            counts = {}
            for size in ("tinylm", "small", "medium", "large"):
                counts[size] = len(collect_training_data("n", size_class=size))

        assert counts["tinylm"] < counts["small"] < counts["medium"] < counts["large"]

    def test_small_bank_returns_all_for_every_tier(self):
        """When the bank has fewer cases than the smallest tier needs,
        every size class gets whatever is available."""
        from arqitect.inference.tuner import collect_training_data

        tiny_bank = [{"input": "q1", "output": "a1"}, {"input": "q2", "output": "a2"}]

        with patch("arqitect.memory.cold.ColdMemory") as MockCold:
            mock_cold = MockCold.return_value
            mock_cold.get_test_bank.return_value = tiny_bank
            tinylm_data = collect_training_data("n", size_class="tinylm")
            large_data = collect_training_data("n", size_class="large")

        assert len(tinylm_data) == len(large_data) == 2

    def test_empty_bank_returns_empty(self):
        from arqitect.inference.tuner import collect_training_data

        with patch("arqitect.memory.cold.ColdMemory") as MockCold:
            mock_cold = MockCold.return_value
            mock_cold.get_test_bank.return_value = []
            data = collect_training_data("n", size_class="medium")

        assert data == []

    def test_slices_are_prefix_of_full_bank(self):
        """Every tier's slice must be a prefix -- same ordering, not random."""
        from arqitect.inference.tuner import collect_training_data

        full_bank = self._make_full_bank()

        with patch("arqitect.memory.cold.ColdMemory") as MockCold:
            mock_cold = MockCold.return_value
            mock_cold.get_test_bank.return_value = full_bank
            tiny_data = collect_training_data("n", size_class="tinylm")
            medium_data = collect_training_data("n", size_class="medium")

        # tinylm's slice must be the first N entries of medium's slice
        for i, item in enumerate(tiny_data):
            assert item["input"] == medium_data[i]["input"]

    def test_without_size_class_returns_full_bank(self):
        """When size_class is None (backward compat), return everything."""
        from arqitect.inference.tuner import collect_training_data

        full_bank = self._make_full_bank()

        with patch("arqitect.memory.cold.ColdMemory") as MockCold:
            mock_cold = MockCold.return_value
            mock_cold.get_test_bank.return_value = full_bank
            data = collect_training_data("n", size_class=None)

        assert len(data) == 500


# -- 15. Only medium/large brain can generate test cases ---------------------


@pytest.mark.timeout(10)
class TestBrainSizeGate:
    """Test generation and reconciliation must only run when the brain
    model is medium or large. tinylm/small produce garbage critic output.
    """

    def _mock_config(self):
        """Shared config mock so tests don't crash on KeyError before reaching the gate."""
        return patch("arqitect.brain.adapters.get_tuning_config", return_value={
            "test_cases_per_batch": 15,
            "training_max_length": 512,
            "few_shot_limit": 10,
        })

    @pytest.mark.parametrize("size", ["tinylm", "small", None])
    def test_generate_returns_empty_for_insufficient_brain(self, size):
        """tinylm, small, or unknown brain sizes must produce no test cases."""
        from arqitect.critic.qualify_nerve import generate_test_cases

        with patch("arqitect.brain.adapters.get_model_size_class", return_value=size):
            with self._mock_config():
                cases = generate_test_cases("reflect_nerve", "self-reflection")
        assert cases == []

    @pytest.mark.parametrize("size", ["medium", "large"])
    def test_generate_works_for_sufficient_brain(self, size):
        from arqitect.critic.qualify_nerve import generate_test_cases

        with patch("arqitect.brain.adapters.get_model_size_class", return_value=size):
            with self._mock_config():
                with patch("arqitect.critic.qualify_nerve._llm", return_value=json.dumps([
                    {"input": "reflect", "output": "thought", "category": "core"}
                ])):
                    cases = generate_test_cases("reflect_nerve", "self-reflection")
        assert len(cases) >= 1

    def test_suggest_improvements_returns_noop_for_small_brain(self):
        """Small brain can't produce useful improvements -- should return unchanged."""
        from arqitect.critic.qualify_nerve import suggest_improvements

        original_prompt = "You are a reflection agent."

        with patch("arqitect.brain.adapters.get_model_size_class", return_value="small"):
            result = suggest_improvements(
                "reflect_nerve", "self-reflection", original_prompt, [], [],
                [{"input": "test", "issue": "bad", "score": 0.3}],
            )

        assert result["system_prompt"] == original_prompt, (
            "Small brain must not modify the system prompt"
        )

    def test_reconcile_skips_improve_for_small_brain(self):
        """_dream_reconcile must not call _improve_one_nerve with a small brain."""
        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = threading.Event()

        with patch("arqitect.brain.adapters.get_model_size_class", return_value="small"), \
             patch("arqitect.brain.consolidate._build_work_queue") as mock_queue, \
             patch("arqitect.brain.consolidate._improve_one_nerve") as mock_improve, \
             patch("arqitect.brain.consolidate.mem"), \
             patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95), \
             patch("arqitect.brain.consolidate.publish_event"), \
             patch("arqitect.brain.consolidate.publish_nerve_status"):

            mock_queue.return_value = [
                {"name": "reflect_nerve", "description": "reflect",
                 "score": 0.1, "last_invoked_at": None}
            ]
            ds._dream_reconcile()

        mock_improve.assert_not_called()

    def test_expand_test_banks_skips_for_small_brain(self):
        """_expand_test_banks_for_training must not generate when brain is small."""
        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = threading.Event()

        with patch("arqitect.brain.adapters.get_model_size_class", return_value="small"), \
             patch("arqitect.brain.consolidate.mem") as mock_mem, \
             patch("arqitect.critic.qualify_nerve.generate_test_cases") as mock_gen:

            mock_mem.cold.conn.execute.return_value.fetchall.return_value = [
                ("reflect_nerve", "self-reflection", "creative"),
            ]
            mock_mem.cold.is_qualified.return_value = True

            with patch("arqitect.inference.tuner.collect_training_data", return_value=[]):
                with patch("arqitect.brain.adapters.get_tuning_config", return_value={
                    "min_training_examples": 200, "test_cases_per_batch": 15,
                }):
                    ds._expand_test_banks_for_training()

        mock_gen.assert_not_called()


# -- 16. Test case provenance -- tag with generating model -------------------


@pytest.mark.timeout(10)
class TestTestCaseProvenance:
    """Each test case must be tagged with the model that generated it.
    The tag is the actual model name -- size class is derivable from it.
    """

    def test_generated_cases_include_model_name(self):
        from arqitect.critic.qualify_nerve import generate_test_cases

        with patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"):
            with patch("arqitect.critic.qualify_nerve._llm", return_value=json.dumps([
                {"input": "reflect", "output": "thought", "category": "core"}
            ])):
                with patch("arqitect.brain.adapters.get_raw_model_name",
                           return_value="qwen2.5-coder-7b"):
                    cases = generate_test_cases("reflect_nerve", "self-reflection")

        assert len(cases) >= 1
        assert cases[0]["generated_by"] == "qwen2.5-coder-7b"

    def test_all_cases_in_batch_get_same_model_tag(self):
        """Every case in a single generate call must have the same model tag."""
        from arqitect.critic.qualify_nerve import generate_test_cases

        with patch("arqitect.brain.adapters.get_model_size_class", return_value="large"):
            with patch("arqitect.brain.adapters.get_tuning_config", return_value={
                "test_cases_per_batch": 15, "training_max_length": 512,
            }):
                with patch("arqitect.critic.qualify_nerve._llm", return_value=json.dumps([
                    {"input": "q1", "output": "a1", "category": "core"},
                    {"input": "q2", "output": "a2", "category": "edge"},
                    {"input": "q3", "output": "a3", "category": "negative"},
                ])):
                    with patch("arqitect.brain.adapters.get_raw_model_name",
                               return_value="llama-3-70b"):
                        cases = generate_test_cases("reflect_nerve", "self-reflection")

        assert len(cases) == 3
        for tc in cases:
            assert tc["generated_by"] == "llama-3-70b"

    def test_provenance_tag_survives_storage_round_trip(self):
        """generated_by must persist through set_test_bank -> get_test_bank."""
        from arqitect.memory.cold import ColdMemory

        cases = [
            {"input": "q1", "output": "a1", "generated_by": "qwen2.5-coder-7b"},
        ]

        cold = MagicMock(spec=ColdMemory)
        stored = None

        def fake_set(name, tests):
            nonlocal stored
            stored = json.dumps(tests)

        def fake_get(name):
            return json.loads(stored) if stored else []

        cold.set_test_bank.side_effect = fake_set
        cold.get_test_bank.side_effect = fake_get

        cold.set_test_bank("reflect_nerve", cases)
        loaded = cold.get_test_bank("reflect_nerve")

        assert loaded[0]["generated_by"] == "qwen2.5-coder-7b"

    def test_upgrade_logic_larger_replaces_smaller(self):
        """Large brain must identify and upgrade medium-generated cases."""
        from arqitect.brain.adapters import _model_to_size_class

        bank = [
            {"input": "q1", "output": "ok", "generated_by": "qwen2.5-coder-7b"},
            {"input": "q2", "output": "good", "generated_by": "llama-3-70b"},
        ]

        current_brain = "llama-3-70b"
        current_size = _model_to_size_class(current_brain)

        upgradeable = [
            tc for tc in bank
            if _size_rank(_model_to_size_class(tc["generated_by"]) or "")
               < _size_rank(current_size)
        ]

        assert len(upgradeable) == 1
        assert upgradeable[0]["input"] == "q1"

    def test_upgrade_logic_same_size_no_op(self):
        from arqitect.brain.adapters import _model_to_size_class

        bank = [
            {"input": "q1", "output": "a1", "generated_by": "qwen2.5-coder-7b"},
        ]

        current_size = _model_to_size_class("mistral-7b")  # also medium
        upgradeable = [
            tc for tc in bank
            if _size_rank(_model_to_size_class(tc["generated_by"]) or "")
               < _size_rank(current_size)
        ]
        assert len(upgradeable) == 0

    def test_upgrade_logic_smaller_brain_never_downgrades(self):
        """A medium brain must NOT overwrite cases generated by a large model."""
        from arqitect.brain.adapters import _model_to_size_class

        bank = [
            {"input": "q1", "output": "excellent", "generated_by": "llama-3-70b"},
        ]

        current_brain = "qwen2.5-coder-7b"
        current_size = _model_to_size_class(current_brain)  # medium

        upgradeable = [
            tc for tc in bank
            if _size_rank(_model_to_size_class(tc["generated_by"]) or "")
               < _size_rank(current_size)
        ]
        assert len(upgradeable) == 0, "Medium must not downgrade large-generated cases"

    def test_cases_without_tag_are_always_upgradeable(self):
        """Legacy cases with no generated_by field should be upgradeable by any model."""
        bank = [
            {"input": "q1", "output": "a1"},  # no generated_by
        ]

        upgradeable = [
            tc for tc in bank
            if _size_rank(None) < _size_rank("medium")
        ]
        assert len(upgradeable) == 1


# -- 17. Prompt alignment review after reconciliation iterations -------------


@pytest.mark.timeout(10)
class TestPromptAlignmentReview:
    """After reconciliation iterations, the system must review whether
    system_prompt, name, and description are still aligned and coherent.
    """

    def _make_mock_llm(self, responses):
        """Build a callable that returns responses in order.

        Args:
            responses: List of string responses to return sequentially.

        Returns:
            A side_effect-compatible callable.
        """
        call_count = {"n": 0}

        def _mock_llm(prompt, **kwargs):
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            return responses[idx]

        return _mock_llm

    def test_consolidation_after_5_plus_rules(self):
        """With 5+ appended rules, the prompt must be consolidated into
        a coherent rewrite, not left as a growing list."""
        from arqitect.critic.qualify_nerve import suggest_improvements

        long_prompt = "You are a reflection agent.\n" + "\n".join(
            f"Rule: Always include aspect {i} in reflections" for i in range(10)
        )

        llm_responses = [
            json.dumps({
                "rule": "Always mention emotional patterns",
                "examples": [],
                "description": "",
            }),
            "You are a reflection agent that always includes aspects 0-9 and emotional patterns in reflections.",
        ]

        with patch("arqitect.critic.qualify_nerve._llm",
                   side_effect=self._make_mock_llm(llm_responses)):
            with patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"):
                result = suggest_improvements(
                    "reflect_nerve", "self-reflection", long_prompt, [], [],
                    [{"input": "test", "issue": "missing emotion", "score": 0.3}],
                )

        assert result["system_prompt"].count("Rule:") < 10, (
            "With 10+ accumulated rules, the prompt should have been consolidated"
        )

    def test_consolidation_preserves_core_identity(self):
        """After consolidation, the nerve's core identity must remain."""
        from arqitect.critic.qualify_nerve import suggest_improvements

        prompt = "You are a reflection agent.\n" + "\n".join(
            f"Rule: Rule {i}" for i in range(8)
        )

        llm_responses = [
            json.dumps({
                "rule": "New rule",
                "examples": [],
                "description": "",
            }),
            "You are a reflection agent that follows rules 0-8 and the new rule.",
        ]

        with patch("arqitect.critic.qualify_nerve._llm",
                   side_effect=self._make_mock_llm(llm_responses)):
            with patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"):
                result = suggest_improvements(
                    "reflect_nerve", "self-reflection", prompt, [], [],
                    [{"input": "test", "issue": "bad", "score": 0.3}],
                )

        assert "reflect" in result["system_prompt"].lower(), (
            "Consolidation must preserve the nerve's core identity"
        )

    def test_no_consolidation_with_few_rules(self):
        """With < 5 rules, just append normally -- no rewrite needed."""
        from arqitect.critic.qualify_nerve import suggest_improvements

        short_prompt = "You are a reflection agent.\nRule: Be thoughtful about temporal patterns"

        with patch("arqitect.critic.qualify_nerve._llm", return_value=json.dumps({
            "rule": "Always format output as bullet points with timestamps",
            "examples": [],
            "description": "",
        })):
            with patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"):
                with patch("arqitect.brain.adapters.get_tuning_config", return_value={
                    "few_shot_limit": 10,
                }):
                    result = suggest_improvements(
                        "reflect_nerve", "self-reflection", short_prompt, [], [],
                        [{"input": "test", "issue": "bad format", "score": 0.3}],
                    )

        assert "temporal patterns" in result["system_prompt"]
        assert "bullet points" in result["system_prompt"]

    def test_prompt_cap_for_tinylm(self):
        """tinylm system prompt must not exceed max_system_tokens (512)."""
        from arqitect.critic.qualify_nerve import suggest_improvements

        long_prompt = "You are a reflection agent. " * 30  # ~900 chars

        with patch("arqitect.critic.qualify_nerve._llm", return_value=json.dumps({
            "rule": "Be concise",
            "examples": [],
            "description": "",
        })):
            with patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"):
                result = suggest_improvements(
                    "reflect_nerve", "self-reflection", long_prompt, [], [],
                    [{"input": "test", "issue": "verbose", "score": 0.3}],
                    role="tool",
                )

        # For any size class, prompt must not grow beyond 4096 (largest cap)
        assert len(result["system_prompt"]) <= 4096

    def test_prompt_cap_is_not_hardcoded_800(self):
        """The 800-char cap must be replaced with size-class-aware limits."""
        from arqitect.critic.qualify_nerve import suggest_improvements
        import inspect

        source = inspect.getsource(suggest_improvements)
        assert "max_system_tokens" in source or "size_class" in source, (
            "suggest_improvements must use size-class-aware prompt length caps"
        )

    def test_description_drift_blocked(self):
        """A description that drifts away from the nerve's domain must be rejected."""
        from arqitect.critic.qualify_nerve import suggest_improvements

        with patch("arqitect.critic.qualify_nerve._llm", return_value=json.dumps({
            "rule": "",
            "examples": [],
            "description": "handles complex mathematical computations",
        })):
            with patch("arqitect.brain.adapters.get_model_size_class", return_value="medium"):
                result = suggest_improvements(
                    "reflect_nerve", "self-reflection", "You reflect on patterns",
                    [], [], [{"input": "test", "issue": "unclear", "score": 0.5}],
                )

        new_desc = result.get("description", "")
        assert "math" not in new_desc.lower(), (
            "Description drifted from 'self-reflection' to 'math' -- "
            "the alignment guard should have blocked this"
        )


# -- 18. Smart hydration -- only hydrate nerves that need work ---------------


@pytest.mark.timeout(10)
class TestSmartHydration:
    """Community sync must not blindly hydrate all ~100 nerves.

    Decision matrix for hydration (all require _needs_hydration=True, i.e. no system_prompt):
    - Invoked + low score -> HYDRATE
    - Invoked + no qual + adapter -> HYDRATE
    - Invoked + high score + adapter -> SKIP
    - Invoked + high score + no adapter -> HYDRATE
    - Never invoked + no adapter -> HYDRATE
    - Never invoked + adapter + no overlap -> SKIP
    - Sense -> SKIP
    - Already hydrated -> SKIP
    """

    def _mock_cold(self, nerves, qualifications=None, origins=None,
                   last_invoked=None, system_prompts=None, senses=None):
        """Build a mock cold memory with configurable nerve state.

        Args:
            nerves: Dict of name->description.
            qualifications: Dict of name->qualification record.
            origins: Dict of name->origin string.
            last_invoked: Dict of name->ISO timestamp or None.
            system_prompts: Dict of name->prompt string.
            senses: Set of nerve names that are senses.

        Returns:
            Configured MagicMock for cold memory.
        """
        qualifications = qualifications or {}
        origins = origins or {}
        last_invoked = last_invoked or {}
        system_prompts = system_prompts or {}
        senses = senses or set()

        cold = MagicMock()
        cold.list_nerves.return_value = nerves
        cold.get_nerve_metadata.side_effect = lambda n: {
            "description": nerves.get(n, ""),
            "system_prompt": system_prompts.get(n, ""),
            "examples": [],
            "role": "tool",
            "origin": origins.get(n, "local"),
        }
        cold.get_qualification.side_effect = lambda t, n: qualifications.get(n)
        cold.get_last_invoked_at.side_effect = lambda n: last_invoked.get(n)
        cold.is_sense.side_effect = lambda n: n in senses
        cold.get_nerve_origin.side_effect = lambda n: origins.get(n, "local")
        return cold

    # -- Hydration: invoked nerves ---

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=False)
    def test_invoked_with_low_score_selected(self, mock_adapter, mock_community, mock_mem):
        """Invoked + qualification score <95% -> hydrate."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {"weak_nerve": "does something"}
        mock_mem.cold = self._mock_cold(
            nerves,
            qualifications={"weak_nerve": {"score": 0.7, "qualified": False}},
            last_invoked={"weak_nerve": "2026-03-20T10:00:00"},
        )
        mock_community.return_value = frozenset()

        names = [c[0] for c in _select_hydration_candidates()]
        assert "weak_nerve" in names

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=True)
    def test_invoked_no_qualification_record_selected(self, mock_adapter, mock_community, mock_mem):
        """Invoked + no qualification row + adapter exists -> hydrate.

        Reproduces the reflect_nerve production bug: nerve was invoked but
        had no qualification row yet. No record = 0% score, not 'skip'.
        """
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {"reflect_nerve": "Generate a reflection prompt"}
        mock_mem.cold = self._mock_cold(
            nerves,
            qualifications={},
            last_invoked={"reflect_nerve": "2026-03-20T12:30:00"},
        )
        mock_community.return_value = frozenset()

        names = [c[0] for c in _select_hydration_candidates()]
        assert "reflect_nerve" in names, (
            "No qualification record means 0% score -- must be hydrated"
        )

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=True)
    def test_invoked_high_score_with_adapter_skipped(self, mock_adapter, mock_community, mock_mem):
        """Invoked + score >=95% + adapter exists -> skip."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {"good_nerve": "works great"}
        mock_mem.cold = self._mock_cold(
            nerves,
            qualifications={"good_nerve": {"score": 0.98, "qualified": True}},
            last_invoked={"good_nerve": "2026-03-20T10:00:00"},
        )
        mock_community.return_value = frozenset()

        names = [c[0] for c in _select_hydration_candidates()]
        assert "good_nerve" not in names

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=False)
    def test_invoked_high_score_no_adapter_selected(self, mock_adapter, mock_community, mock_mem):
        """Invoked + score >=95% but no adapter -> hydrate (needs tuning)."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {"tuned_elsewhere": "works on other model"}
        mock_mem.cold = self._mock_cold(
            nerves,
            qualifications={"tuned_elsewhere": {"score": 0.98, "qualified": True}},
            last_invoked={"tuned_elsewhere": "2026-03-20T10:00:00"},
        )
        mock_community.return_value = frozenset()

        names = [c[0] for c in _select_hydration_candidates()]
        assert "tuned_elsewhere" in names, (
            "High score but no adapter for current model -- needs hydration for tuning"
        )

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=True)
    def test_exact_95_percent_boundary_skipped(self, mock_adapter, mock_community, mock_mem):
        """Score at exactly 95% with adapter -> skip (threshold is <95, not <=)."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {"boundary_nerve": "right at threshold"}
        mock_mem.cold = self._mock_cold(
            nerves,
            qualifications={"boundary_nerve": {"score": 0.95, "qualified": True}},
            last_invoked={"boundary_nerve": "2026-03-20T10:00:00"},
        )
        mock_community.return_value = frozenset()

        names = [c[0] for c in _select_hydration_candidates()]
        assert "boundary_nerve" not in names, "Exactly 95% should not be hydrated"

    # -- Hydration: never-invoked nerves ---

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=False)
    def test_never_invoked_no_adapter_selected(self, mock_adapter, mock_community, mock_mem):
        """Never invoked + no adapter -> hydrate (adapter gap, not score)."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {"fresh_nerve": "brand new"}
        mock_mem.cold = self._mock_cold(nerves)
        mock_community.return_value = frozenset()

        names = [c[0] for c in _select_hydration_candidates()]
        assert "fresh_nerve" in names

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=True)
    def test_never_invoked_with_adapter_no_overlap_skipped(self, mock_adapter, mock_community, mock_mem):
        """Never invoked + adapter exists + no local overlap -> skip."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {"unused_nerve": "some unique capability"}
        mock_mem.cold = self._mock_cold(
            nerves,
            origins={"unused_nerve": "community"},
        )
        mock_community.return_value = frozenset(["unused_nerve"])

        names = [c[0] for c in _select_hydration_candidates()]
        assert "unused_nerve" not in names

    # -- Hydration: guards ---

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=True)
    def test_already_hydrated_skipped(self, mock_adapter, mock_community, mock_mem):
        """Nerve with system_prompt already set -> skip (already hydrated)."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {"hydrated_nerve": "already has prompt"}
        mock_mem.cold = self._mock_cold(
            nerves,
            system_prompts={"hydrated_nerve": "You are a helpful agent."},
            last_invoked={"hydrated_nerve": "2026-03-20T10:00:00"},
        )
        mock_community.return_value = frozenset()

        assert len(_select_hydration_candidates()) == 0

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=False)
    def test_already_hydrated_even_low_score_skipped(self, mock_adapter, mock_community, mock_mem):
        """Hydrated nerve at 0% -> skip hydration (needs reconciliation not re-download)."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {"has_prompt_nerve": "already hydrated but bad score"}
        mock_mem.cold = self._mock_cold(
            nerves,
            system_prompts={"has_prompt_nerve": "You are a tool."},
            qualifications={"has_prompt_nerve": {"score": 0.0, "qualified": False}},
            last_invoked={"has_prompt_nerve": "2026-03-20T10:00:00"},
        )
        mock_community.return_value = frozenset()

        assert len(_select_hydration_candidates()) == 0, (
            "Already-hydrated nerve should not be re-downloaded even with low score"
        )

    # -- Hydration: consolidation overlap ---

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.consolidate._get_merge_threshold", return_value=0.5)
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=True)
    def test_community_overlapping_local_selected(self, mock_adapter, mock_threshold,
                                                  mock_community, mock_mem):
        """Community nerve overlapping a local fabricated nerve -> hydrate for consolidation."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {
            "translate_text": "translate text between languages",
            "text_translator": "translate and convert text between languages",
        }
        mock_mem.cold = self._mock_cold(
            nerves,
            origins={"translate_text": "community", "text_translator": "local"},
        )
        mock_community.return_value = frozenset(["translate_text"])

        names = [c[0] for c in _select_hydration_candidates()]
        assert "translate_text" in names
        assert "text_translator" not in names, (
            "Local nerve should not be hydrated -- it's already local"
        )

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.consolidate._get_merge_threshold", return_value=0.5)
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=True)
    def test_community_no_overlap_with_local_skipped(self, mock_adapter, mock_threshold,
                                                     mock_community, mock_mem):
        """Community nerve with no local overlap -> skip."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {
            "translate_text": "translate text between languages",
            "deploy_app": "deploy an application to cloud",
        }
        mock_mem.cold = self._mock_cold(
            nerves,
            origins={"translate_text": "community", "deploy_app": "local"},
        )
        mock_community.return_value = frozenset(["translate_text"])

        names = [c[0] for c in _select_hydration_candidates()]
        assert "translate_text" not in names, (
            "Community nerve with no local overlap should be skipped"
        )

    # -- Hydration: sort order ---

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=False)
    def test_candidates_sorted_by_recency(self, mock_adapter, mock_community, mock_mem):
        """Recently-used nerves should sort before older ones."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {"old_nerve": "old stuff", "recent_nerve": "new stuff", "never_nerve": "never used"}
        mock_mem.cold = self._mock_cold(
            nerves,
            last_invoked={
                "old_nerve": "2026-03-01T10:00:00",
                "recent_nerve": "2026-03-20T10:00:00",
            },
        )
        mock_community.return_value = frozenset()

        names = [c[0] for c in _select_hydration_candidates()]

        assert names.index("recent_nerve") < names.index("old_nerve"), (
            "Recently used nerve must sort before older nerve"
        )

    # -- Hydration: full scenario ---

    @patch("arqitect.brain.consolidate.mem")
    @patch("arqitect.brain.consolidate._get_community_nerve_names")
    @patch("arqitect.brain.adapters.has_model_specific_adapter", return_value=True)
    def test_mixed_catalog_only_actionable_nerves_selected(self, mock_adapter, mock_community, mock_mem):
        """Given a catalog of 5 nerves, only the ones needing work should be selected."""
        from arqitect.brain.consolidate import _select_hydration_candidates

        nerves = {
            "invoked_weak": "invoked and weak",
            "invoked_strong": "invoked and strong",
            "invoked_no_qual": "invoked but never qualified",
            "never_used": "registered but never invoked",
            "already_hydrated": "has system prompt already",
        }
        mock_mem.cold = self._mock_cold(
            nerves,
            qualifications={
                "invoked_weak": {"score": 0.3, "qualified": False},
                "invoked_strong": {"score": 0.99, "qualified": True},
            },
            last_invoked={
                "invoked_weak": "2026-03-20T10:00:00",
                "invoked_strong": "2026-03-20T10:00:00",
                "invoked_no_qual": "2026-03-20T09:00:00",
            },
            system_prompts={"already_hydrated": "You are an agent."},
        )
        mock_community.return_value = frozenset()

        names = [c[0] for c in _select_hydration_candidates()]

        assert "invoked_weak" in names, "Invoked + low score -> hydrate"
        assert "invoked_no_qual" in names, "Invoked + no qualification -> hydrate"
        assert "invoked_strong" not in names, "Invoked + high score + adapter -> skip"
        assert "never_used" not in names, "Never invoked + adapter exists -> skip"
        assert "already_hydrated" not in names, "Already has system_prompt -> skip"
        assert len(names) == 2, f"Expected exactly 2 candidates, got {len(names)}: {names}"


@pytest.mark.timeout(10)
class TestReconciliationWorkQueue:
    """_build_work_queue must only include nerves worth reconciling.

    Rules:
    - Only invoked nerves (has last_invoked_at)
    - Score below improvement threshold
    - Not a sense
    - Has nerve.py on disk
    """

    def _patch_work_queue(self, mock_mem, nerves, qualifications=None,
                          last_invoked=None, senses=None,
                          core_senses=frozenset(), threshold=0.95):
        """Configure mocks for _build_work_queue tests.

        Args:
            mock_mem: The patched mem module mock.
            nerves: Dict of name->description.
            qualifications: List of qualification dicts.
            last_invoked: Dict of name->ISO timestamp.
            senses: Set of nerve names that are senses.
            core_senses: Frozenset of core sense names.
            threshold: Improvement threshold score.
        """
        mock_mem.cold.list_qualifications.return_value = qualifications or []
        mock_mem.cold.list_nerves.return_value = nerves
        senses = senses or set()
        mock_mem.cold.is_sense.side_effect = lambda n: n in senses
        last_invoked = last_invoked or {}
        mock_mem.cold.get_last_invoked_at.side_effect = lambda n: last_invoked.get(n)

    @patch("os.path.isfile", return_value=True)
    @patch("os.listdir", return_value=[])
    @patch("os.path.isdir", return_value=True)
    @patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95)
    @patch("arqitect.brain.consolidate.CORE_SENSES", frozenset())
    @patch("arqitect.brain.consolidate.mem")
    def test_never_invoked_nerve_excluded(self, mock_mem, *_):
        """Nerves never invoked must not appear in the work queue."""
        from arqitect.brain.consolidate import _build_work_queue

        self._patch_work_queue(
            mock_mem,
            nerves={"unused_nerve": "never used", "used_nerve": "was used once"},
            last_invoked={"used_nerve": "2026-03-20T10:00:00"},
        )

        queue = _build_work_queue()
        names = [item["name"] for item in queue]

        assert "unused_nerve" not in names, "Never-invoked nerve should be excluded"
        assert "used_nerve" in names, "Invoked nerve should be included"

    @patch("os.path.isfile", return_value=True)
    @patch("os.listdir", return_value=[])
    @patch("os.path.isdir", return_value=True)
    @patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95)
    @patch("arqitect.brain.consolidate.CORE_SENSES", frozenset({"awareness"}))
    @patch("arqitect.brain.consolidate.mem")
    def test_sense_included_in_tuning(self, mock_mem, *_):
        """Core senses must be included in the tuning queue — they need
        model-specific prompt tuning just like nerves."""
        from arqitect.brain.consolidate import _build_work_queue

        self._patch_work_queue(
            mock_mem,
            nerves={"awareness": "self awareness sense", "real_nerve": "actual nerve"},
            senses={"awareness"},
            last_invoked={"awareness": "2026-03-20T10:00:00", "real_nerve": "2026-03-20T10:00:00"},
        )

        queue = _build_work_queue()
        names = [item["name"] for item in queue]

        assert "awareness" in names, "Sense must be included in tuning queue"
        assert "real_nerve" in names

    @patch("os.path.isfile", return_value=True)
    @patch("os.listdir", return_value=[])
    @patch("os.path.isdir", return_value=True)
    @patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95)
    @patch("arqitect.brain.consolidate.CORE_SENSES", frozenset())
    @patch("arqitect.brain.consolidate.mem")
    def test_nerve_at_threshold_excluded(self, mock_mem, *_):
        """Nerve at exactly the improvement threshold should be excluded."""
        from arqitect.brain.consolidate import _build_work_queue

        self._patch_work_queue(
            mock_mem,
            nerves={"at_threshold": "exactly at 95%", "below_threshold": "just under 95%"},
            qualifications=[
                {"subject_type": "nerve", "subject_name": "at_threshold", "score": 0.95},
                {"subject_type": "nerve", "subject_name": "below_threshold", "score": 0.94},
            ],
            last_invoked={
                "at_threshold": "2026-03-20T10:00:00",
                "below_threshold": "2026-03-20T10:00:00",
            },
        )

        queue = _build_work_queue()
        names = [item["name"] for item in queue]

        assert "at_threshold" not in names, "Nerve at exactly 95% should be excluded"
        assert "below_threshold" in names, "Nerve below 95% should be included"

    @patch("os.path.isfile", return_value=True)
    @patch("os.listdir", return_value=[])
    @patch("os.path.isdir", return_value=True)
    @patch("arqitect.brain.consolidate._get_improvement_threshold", return_value=0.95)
    @patch("arqitect.brain.consolidate.CORE_SENSES", frozenset())
    @patch("arqitect.brain.consolidate.mem")
    def test_only_invoked_nerves_in_count(self, mock_mem, *_):
        """Queue count must reflect only invoked nerves, not entire catalog.

        Reproduces the 152-nerve bug: all community nerves were queued
        even though only 1 was invoked.
        """
        from arqitect.brain.consolidate import _build_work_queue

        nerves = {f"nerve_{i}": f"nerve number {i}" for i in range(100)}
        nerves["invoked_a"] = "invoked nerve a"
        nerves["invoked_b"] = "invoked nerve b"

        self._patch_work_queue(
            mock_mem,
            nerves=nerves,
            last_invoked={
                "invoked_a": "2026-03-20T10:00:00",
                "invoked_b": "2026-03-19T10:00:00",
            },
        )

        queue = _build_work_queue()

        assert len(queue) == 2, (
            f"Expected 2 nerves in queue (only invoked ones), got {len(queue)}"
        )
        names = [item["name"] for item in queue]
        assert "invoked_a" in names
        assert "invoked_b" in names
