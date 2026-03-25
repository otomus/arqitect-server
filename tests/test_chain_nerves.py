"""Chain execution tests: multi-step, single-step degradation, mid-chain synthesis.

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
    RespondDecisionFactory,
    ChainDecisionFactory,
    ChainStepFactory,
    as_dict,
)


# ---------------------------------------------------------------------------
# 4.1 Multi-step chain with context accumulation
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestMultiStepChain:
    """Multi-step chain with context passing between steps."""

    def test_chain_passes_context_between_steps(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """Each step in a chain should receive output from previous steps."""
        mem_fixture = make_mem(test_redis)
        register_qualified_nerve(mem_fixture, "fetch_nerve", "fetches data")
        register_qualified_nerve(mem_fixture, "format_nerve", "formats data")
        make_nerve_file(nerves_dir, "fetch_nerve")
        make_nerve_file(nerves_dir, "format_nerve")

        invoke_calls = []

        def mock_invoke(name, args, user_id=""):
            invoke_calls.append({"name": name, "args": args})
            if name == "fetch_nerve":
                return '{"response": "raw data from API"}'
            elif name == "format_nerve":
                return '{"response": "formatted nicely"}'
            return '{"response": "ok"}'

        steps = [
            ChainStepFactory.build(nerve="fetch_nerve", args="get weather data"),
            ChainStepFactory.build(nerve="format_nerve", args="format the data"),
        ]
        d = ChainDecisionFactory.build(steps=steps, goal="fetch and format weather data")
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(d))),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_invoke_nerve(side_effect=mock_invoke):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.brain import think
                think("fetch and format weather data")
                assert len(invoke_calls) == 2
                step2_args = invoke_calls[1]["args"]
                assert "Step 1 result" in step2_args or "raw data" in step2_args, \
                    f"Step 2 should receive step 1 context, got: {step2_args}"
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 4.2 Single-step chain degrades to invoke_nerve
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestSingleStepChain:
    """Single-step chains should degrade to a plain invoke."""

    def test_single_step_chain_converts_to_invoke(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """A chain with a single step should convert to a plain invoke_nerve."""
        mem_fixture = make_mem(test_redis)
        register_qualified_nerve(mem_fixture, "joke_nerve", "tells jokes")
        make_nerve_file(nerves_dir, "joke_nerve")

        step = ChainStepFactory.build(nerve="joke_nerve", args="tell a joke")
        chain = ChainDecisionFactory.build(steps=[step], goal="tell a joke")
        invoke = InvokeDecisionFactory.build(name="joke_nerve", args="tell a joke")
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(chain))),
            ("invoke_nerve", json.dumps(as_dict(invoke))),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_invoke_nerve(return_value='{"response": "knock knock"}') as mock_invoke:
            for p in patches:
                p.start()
            try:
                from arqitect.brain.brain import think
                think("tell a joke")
                mock_invoke.assert_called()
                assert mock_invoke.call_args[0][0] == "joke_nerve"
            finally:
                for p in patches:
                    p.stop()


# ---------------------------------------------------------------------------
# 4.4 Chain step synthesis with rename propagation
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestChainSynthesisRename:
    """Mid-chain synthesis with rename propagation."""

    def test_mid_chain_synthesis_uses_renamed_name(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """If a chain step synthesizes a nerve that gets renamed, invoke uses the renamed name."""
        mem_fixture = make_mem(test_redis)

        def mock_synthesize(name, desc, mcp_tools=None, trigger_task="", role=None):
            actual_name = "system_info_nerve" if name == "awareness_nerve" else name
            d = os.path.join(nerves_dir, actual_name)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "nerve.py")
            with open(path, "w") as f:
                f.write("# stub\n")
            return actual_name, path

        invoke_calls = []

        def mock_invoke(name, args, user_id=""):
            invoke_calls.append(name)
            return '{"response": "ok"}'

        step = ChainStepFactory.build(nerve="awareness_nerve", args="get system info")
        chain = ChainDecisionFactory.build(steps=[step], goal="system info")
        invoke = InvokeDecisionFactory.build(name="system_info_nerve", args="get system info")
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(chain))),
            ("invoke_nerve", json.dumps(as_dict(invoke))),
        ])

        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_synthesize_nerve(side_effect=mock_synthesize):
            with patch_invoke_nerve(side_effect=mock_invoke):
                for p in patches:
                    p.start()
                try:
                    from arqitect.brain.brain import think
                    think("get system info")
                    if invoke_calls:
                        assert "awareness_nerve" not in invoke_calls, \
                            f"Should not invoke original name. Calls: {invoke_calls}"
                finally:
                    for p in patches:
                        p.stop()


# ---------------------------------------------------------------------------
# Edge cases: empty, None, missing, and long chains
# ---------------------------------------------------------------------------

def _run_chain_think(steps_data, test_redis, nerves_dir, sandbox_dir,
                     extra_nerves=None, invoke_side_effect=None):
    """Run think() with a chain decision and a respond fallback.

    Registers any extra_nerves so they exist in the catalog. Returns
    (result, invoke_mock) so callers can inspect calls.

    Args:
        steps_data: List of ChainStep factory instances.
        test_redis: Redis fixture.
        nerves_dir: Nerves directory fixture.
        sandbox_dir: Sandbox directory fixture.
        extra_nerves: List of (name, description) tuples to register.
        invoke_side_effect: Optional side_effect for invoke_nerve mock.

    Returns:
        Tuple of (think result string, invoke mock object).
    """
    mem_fixture = make_mem(test_redis)
    for name, desc in (extra_nerves or []):
        register_qualified_nerve(mem_fixture, name, desc)
        make_nerve_file(nerves_dir, name)

    chain = ChainDecisionFactory.build(steps=steps_data, goal="edge case test")
    respond = RespondDecisionFactory.build(message="fallback response")
    fake = FakeLLM([
        ("Task:", json.dumps(as_dict(chain))),
        ("respond", json.dumps(as_dict(respond))),
    ])

    patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
    invoke_kw = {"side_effect": invoke_side_effect} if invoke_side_effect else {"return_value": '{"response": "ok"}'}
    with patch_invoke_nerve(**invoke_kw) as mock_inv:
        for p in patches:
            p.start()
        try:
            from arqitect.brain.brain import think
            result = think("edge case test")
        finally:
            for p in patches:
                p.stop()
    return result, mock_inv


@pytest.mark.timeout(10)
class TestChainEdgeCases:
    """Edge cases that try to break chain execution."""

    def test_empty_chain_no_steps(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """A chain with zero steps should re-think, not crash."""
        result, mock_inv = _run_chain_think(
            [], test_redis, nerves_dir, sandbox_dir,
        )
        # Empty chain triggers re-think with error message; invoke should not fire
        mock_inv.assert_not_called()
        assert result is not None

    def test_chain_with_none_nerve_names(self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir):
        """Steps with None/empty nerve names should be skipped without crashing."""
        steps = [
            ChainStepFactory.build(nerve="", args="empty nerve"),
            ChainStepFactory.build(nerve="", args="also empty"),
        ]
        result, mock_inv = _run_chain_think(
            steps, test_redis, nerves_dir, sandbox_dir,
        )
        # All steps had empty nerve names — dispatch skips them
        mock_inv.assert_not_called()
        assert result is not None

    def test_chain_with_nonexistent_nerve_attempts_synthesis(
        self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir,
    ):
        """A multi-step chain referencing a missing nerve should attempt synthesis."""
        mem_fixture = make_mem(test_redis)
        register_qualified_nerve(mem_fixture, "real_nerve", "does real things")
        make_nerve_file(nerves_dir, "real_nerve")

        def mock_synth(name, desc, mcp_tools=None, trigger_task="", role=None):
            d = os.path.join(nerves_dir, name)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "nerve.py")
            with open(path, "w") as f:
                f.write("# stub\n")
            return name, path

        # Two steps so it stays a chain (not degraded to invoke).
        # First step exists, second does not.
        steps = [
            ChainStepFactory.build(nerve="real_nerve", args="step one"),
            ChainStepFactory.build(nerve="totally_fake_nerve", args="step two"),
        ]
        chain = ChainDecisionFactory.build(steps=steps, goal="test missing nerve")
        fake = FakeLLM([
            ("Task:", json.dumps(as_dict(chain))),
        ])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        # can_model_fabricate returns False by default (no real model in tests),
        # so we patch it to allow synthesis in the chain handler.
        with patch("arqitect.brain.dispatch.can_model_fabricate", return_value=True), \
             patch("arqitect.brain.dispatch.can_synthesize_nerve", return_value=True), \
             patch_synthesize_nerve(side_effect=mock_synth) as mock_s, \
             patch_invoke_nerve(return_value='{"response": "ok"}'):
            for p in patches:
                p.start()
            try:
                from arqitect.brain.brain import think
                result = think("test missing nerve")
                # Synthesis should have been called for the unknown nerve
                mock_s.assert_called()
                synth_name = mock_s.call_args[0][0]
                assert synth_name == "totally_fake_nerve"
            finally:
                for p in patches:
                    p.stop()

    def test_long_chain_accumulates_all_context(
        self, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir,
    ):
        """A chain with 12 steps should invoke all of them and pass context forward."""
        num_steps = 12
        nerve_names = [f"step_{i}_nerve" for i in range(num_steps)]
        extra_nerves = [(n, f"step {i}") for i, n in enumerate(nerve_names)]
        steps = [
            ChainStepFactory.build(nerve=n, args=f"do step {i}")
            for i, n in enumerate(nerve_names)
        ]

        invoke_calls = []

        def track_invoke(name, args, user_id=""):
            invoke_calls.append(name)
            return json.dumps({"response": f"result from {name}"})

        result, _ = _run_chain_think(
            steps, test_redis, nerves_dir, sandbox_dir,
            extra_nerves=extra_nerves,
            invoke_side_effect=track_invoke,
        )
        assert len(invoke_calls) == num_steps, (
            f"Expected {num_steps} invocations, got {len(invoke_calls)}: {invoke_calls}"
        )
        # Verify ordering — each nerve should be called exactly once in order
        assert invoke_calls == nerve_names

    @pytest.mark.parametrize("bad_steps", [
        None,
        "not_a_list",
        42,
    ], ids=["none", "string", "integer"])
    def test_chain_with_invalid_steps_type(
        self, bad_steps, test_redis, tmp_memory_dir, nerves_dir, sandbox_dir,
    ):
        """Non-list 'steps' values should trigger re-think, not crash."""
        mem_fixture = make_mem(test_redis)
        # Build a raw chain dict with bad steps directly
        chain_dict = {"action": "chain_nerves", "steps": bad_steps, "goal": "bad input"}
        respond = RespondDecisionFactory.build(message="recovered")
        fake = FakeLLM([
            ("Task:", json.dumps(chain_dict)),
            ("respond", json.dumps(as_dict(respond))),
        ])
        patches = setup_brain_patches(fake, mem_fixture, test_redis, nerves_dir, sandbox_dir)
        with patch_invoke_nerve(return_value='{"response": "ok"}') as mock_inv:
            for p in patches:
                p.start()
            try:
                from arqitect.brain.brain import think
                result = think("bad input")
                # Should not crash; invoke should not be called
                mock_inv.assert_not_called()
                assert result is not None
            finally:
                for p in patches:
                    p.stop()
