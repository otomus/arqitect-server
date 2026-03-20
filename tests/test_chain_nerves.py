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
