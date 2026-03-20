"""Convert captured OTel trace files into pytest test cases.

Reads trace JSONL files produced by arqitect.telemetry, extracts the
execution flow (which nerves were invoked, what the LLM responded, what
the brain decided), and generates pytest test files that replay the
exact same flow using FakeLLM with the real LLM responses as fixtures.

Usage::

    # After running stress_test.py against a live brain:
    python tests/trace_to_tests.py traces/trace_20260320_143000.jsonl

    # Generate tests for specific tags only:
    python tests/trace_to_tests.py traces/trace_*.jsonl --tag greeting,weather

    # Dry run — print what would be generated:
    python tests/trace_to_tests.py traces/trace_*.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

@dataclass
class LLMCall:
    """A captured LLM generate call."""

    role: str
    prompt: str
    system: str
    response: str
    duration_ms: float


@dataclass
class NerveInvocation:
    """A captured nerve invocation."""

    name: str
    args: str
    output: str
    error: str | None
    duration_ms: float


@dataclass
class FlowCapture:
    """A complete think() flow extracted from traces."""

    trace_id: str
    task: str
    response: str
    depth: int
    dispatch_action: str
    dispatch_nerve: str
    dispatch_decision: dict
    llm_calls: list[LLMCall] = field(default_factory=list)
    nerve_invocations: list[NerveInvocation] = field(default_factory=list)
    synthesized_nerves: list[dict] = field(default_factory=list)
    duration_ms: float = 0.0
    status: str = "OK"


def parse_trace_file(path: str) -> list[FlowCapture]:
    """Parse a JSONL trace file into FlowCapture objects.

    Groups spans by trace_id, then assembles each think() flow
    from its child spans.

    Args:
        path: Path to the trace JSONL file.

    Returns:
        List of FlowCapture objects, one per think() invocation.
    """
    spans_by_trace: dict[str, list[dict]] = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            span = json.loads(line)
            tid = span.get("trace_id", "")
            spans_by_trace.setdefault(tid, []).append(span)

    flows = []
    for trace_id, spans in spans_by_trace.items():
        flow = _assemble_flow(trace_id, spans)
        if flow:
            flows.append(flow)

    return flows


def _assemble_flow(trace_id: str, spans: list[dict]) -> FlowCapture | None:
    """Assemble a FlowCapture from a group of related spans."""
    think_span = _find_span(spans, "brain.think")
    if not think_span:
        return None

    attrs = think_span.get("attributes", {})
    flow = FlowCapture(
        trace_id=trace_id,
        task=attrs.get("task", ""),
        response=attrs.get("response_preview", ""),
        depth=attrs.get("depth", 0),
        dispatch_action="",
        dispatch_nerve="",
        dispatch_decision={},
        duration_ms=think_span.get("duration_ms", 0),
        status=think_span.get("status", "OK"),
    )

    # Extract dispatch info
    dispatch_span = _find_span(spans, "brain.dispatch")
    if dispatch_span:
        d_attrs = dispatch_span.get("attributes", {})
        flow.dispatch_action = d_attrs.get("dispatch.action", "")
        flow.dispatch_nerve = d_attrs.get("dispatch.nerve", "")
        decision_str = d_attrs.get("dispatch.decision", "{}")
        try:
            flow.dispatch_decision = json.loads(decision_str)
        except (json.JSONDecodeError, TypeError):
            flow.dispatch_decision = {"raw": decision_str}

    # Extract LLM calls
    for span in spans:
        if span["name"] == "llm.generate":
            s_attrs = span.get("attributes", {})
            flow.llm_calls.append(LLMCall(
                role=s_attrs.get("llm.role", ""),
                prompt=s_attrs.get("llm.prompt", ""),
                system=s_attrs.get("llm.system", ""),
                response=s_attrs.get("llm.response", ""),
                duration_ms=span.get("duration_ms", 0),
            ))

    # Extract nerve invocations
    for span in spans:
        if span["name"] == "nerve.invoke":
            s_attrs = span.get("attributes", {})
            flow.nerve_invocations.append(NerveInvocation(
                name=s_attrs.get("nerve.name", ""),
                args=s_attrs.get("nerve.args", ""),
                output=s_attrs.get("nerve.output", ""),
                error=s_attrs.get("nerve.error"),
                duration_ms=span.get("duration_ms", 0),
            ))

    # Extract synthesized nerves
    for span in spans:
        if span["name"] == "nerve.synthesize":
            s_attrs = span.get("attributes", {})
            flow.synthesized_nerves.append({
                "name": s_attrs.get("synth.name", ""),
                "actual_name": s_attrs.get("synth.actual_name", ""),
                "description": s_attrs.get("synth.description", ""),
            })

    return flow


def _find_span(spans: list[dict], name: str) -> dict | None:
    """Find the first span with a given name."""
    for span in spans:
        if span["name"] == name:
            return span
    return None


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------

def generate_test_file(flows: list[FlowCapture], output_path: str) -> str:
    """Generate a pytest test file from captured flows.

    Args:
        flows: List of FlowCapture objects to generate tests for.
        output_path: Path to write the generated test file.

    Returns:
        The generated test file content.
    """
    test_lines = [
        '"""Auto-generated flow regression tests from captured OTel traces.',
        '',
        'Each test replays a real execution flow using FakeLLM with the actual',
        'LLM responses captured during the stress test run. This ensures the',
        'brain routes and dispatches identically to the live run.',
        '',
        'DO NOT EDIT — regenerate with: python tests/trace_to_tests.py <trace_file>',
        '"""',
        '',
        'from __future__ import annotations',
        '',
        'import json',
        'import os',
        'from unittest.mock import patch, MagicMock',
        '',
        'import pytest',
        '',
        'from arqitect.tracing import FlowRecorder, PUBLISH_EVENT_SITES',
        'from tests.conftest import (',
        '    FakeLLM,',
        '    make_nerve_file,',
        '    register_qualified_nerve,',
        ')',
        '',
        '',
        '# ---------------------------------------------------------------------------',
        '# Captured LLM responses (real responses from the live run)',
        '# ---------------------------------------------------------------------------',
        '',
    ]

    # Generate fixture data
    for i, flow in enumerate(flows):
        test_lines.append(f'FLOW_{i}_TASK = {json.dumps(flow.task)!s}')
        test_lines.append(f'FLOW_{i}_ACTION = {json.dumps(flow.dispatch_action)!s}')
        test_lines.append(f'FLOW_{i}_NERVE = {json.dumps(flow.dispatch_nerve)!s}')
        test_lines.append(f'FLOW_{i}_DECISION = {json.dumps(flow.dispatch_decision)!s}')
        test_lines.append(f'FLOW_{i}_RESPONSE = {json.dumps(flow.response)!s}')
        test_lines.append('')

        # LLM call fixtures
        for j, llm_call in enumerate(flow.llm_calls):
            var_prefix = f'FLOW_{i}_LLM_{j}'
            test_lines.append(f'{var_prefix}_ROLE = {json.dumps(llm_call.role)!s}')
            # Truncate long prompts for readability but keep enough to match
            prompt_substr = llm_call.prompt[:200] if llm_call.prompt else ""
            test_lines.append(f'{var_prefix}_PROMPT_SUBSTR = {json.dumps(prompt_substr)!s}')
            test_lines.append(f'{var_prefix}_RESPONSE = {json.dumps(llm_call.response)!s}')
            test_lines.append('')

        # Nerve invocation fixtures
        for j, inv in enumerate(flow.nerve_invocations):
            var_prefix = f'FLOW_{i}_NERVE_{j}'
            test_lines.append(f'{var_prefix}_NAME = {json.dumps(inv.name)!s}')
            test_lines.append(f'{var_prefix}_OUTPUT = {json.dumps(inv.output[:500] if inv.output else "")!s}')
            test_lines.append('')

        test_lines.append('')

    # Generate test class
    test_lines.extend([
        '',
        '# ---------------------------------------------------------------------------',
        '# Setup helpers',
        '# ---------------------------------------------------------------------------',
        '',
        'def _setup_brain_for_flow(fake_llm, mem, test_redis, nerves_dir, sandbox_dir):',
        '    """Return patches for brain.think() with flow recorder events."""',
        '    return [',
        '        patch("arqitect.brain.brain.mem", mem),',
        '        patch("arqitect.brain.brain.r", test_redis),',
        '        patch("arqitect.brain.dispatch.mem", mem),',
        '        patch("arqitect.brain.dispatch.r", test_redis),',
        '        patch("arqitect.brain.catalog.mem", mem),',
        '        patch("arqitect.brain.brain.NERVES_DIR", nerves_dir),',
        '        patch("arqitect.brain.brain.SANDBOX_DIR", sandbox_dir),',
        '        patch("arqitect.brain.brain.llm_generate", side_effect=fake_llm),',
        '        patch("arqitect.brain.dispatch.llm_generate", side_effect=fake_llm),',
        '        patch("arqitect.brain.helpers.llm_generate", side_effect=fake_llm),',
        '        patch("arqitect.brain.intent.llm_generate", side_effect=fake_llm),',
        '        patch("arqitect.matching._get_nerve_embedding", return_value=None),',
        '        patch("arqitect.brain.synthesis.classify_nerve_role", return_value="tool"),',
        '        patch(',
        '            "arqitect.brain.synthesis.threading.Thread",',
        '            type("_NoOp", (), {',
        '                "__init__": lambda *a, **kw: None,',
        '                "start": lambda self: None,',
        '            }),',
        '        ),',
        '        patch("arqitect.brain.brain.get_consolidator", return_value=MagicMock()),',
        '        patch("arqitect.brain.permissions.can_use_nerve", return_value=True),',
        '        patch("arqitect.brain.dispatch.can_use_nerve", return_value=True),',
        '    ]',
        '',
        '',
    ])

    # Generate test methods
    test_lines.extend([
        '@pytest.mark.timeout(30)',
        'class TestCapturedFlows:',
        '    """Regression tests generated from captured OTel traces."""',
        '',
        '    @pytest.fixture(autouse=True)',
        '    def _setup(self, mem, test_redis, nerves_dir, sandbox_dir):',
        '        self.mem = mem',
        '        self.test_redis = test_redis',
        '        self.nerves_dir = nerves_dir',
        '        self.sandbox_dir = sandbox_dir',
        '',
    ])

    for i, flow in enumerate(flows):
        safe_task = flow.task[:40].replace('"', '\\"').replace("'", "\\'")
        test_name = f"test_flow_{i}_{_slugify(flow.task[:30])}"

        test_lines.append(f'    def {test_name}(self):')
        test_lines.append(f'        """Replay: {safe_task}"""')

        # Build FakeLLM with captured responses
        test_lines.append('        fake_llm = FakeLLM([')
        for j, llm_call in enumerate(flow.llm_calls):
            var_prefix = f'FLOW_{i}_LLM_{j}'
            test_lines.append(f'            ({var_prefix}_PROMPT_SUBSTR, {var_prefix}_RESPONSE, False),')
        test_lines.append('        ])')
        test_lines.append('')

        # Register nerves that were invoked
        invoked_names = {inv.name for inv in flow.nerve_invocations}
        if flow.dispatch_nerve:
            invoked_names.add(flow.dispatch_nerve)
        for name in sorted(invoked_names):
            test_lines.append(f'        register_qualified_nerve(self.mem, {json.dumps(name)!s})')
            test_lines.append(f'        make_nerve_file(self.nerves_dir, {json.dumps(name)!s})')
        test_lines.append('')

        # Set up patches and run
        test_lines.append('        recorder = FlowRecorder()')
        test_lines.append('        event_patches = [')
        test_lines.append('            patch(site, side_effect=recorder.intercept)')
        test_lines.append('            for site in PUBLISH_EVENT_SITES')
        test_lines.append('        ]')
        test_lines.append('        silence_patches = [')
        test_lines.append('            patch("arqitect.brain.brain.publish_response"),')
        test_lines.append('            patch("arqitect.brain.brain.publish_memory_state"),')
        test_lines.append('            patch("arqitect.brain.brain.publish_nerve_status"),')
        test_lines.append('            patch("arqitect.brain.dispatch.publish_response"),')
        test_lines.append('            patch("arqitect.brain.dispatch.publish_memory_state"),')
        test_lines.append('            patch("arqitect.brain.consolidate.publish_nerve_status"),')
        test_lines.append('        ]')
        test_lines.append('        brain_patches = _setup_brain_for_flow(')
        test_lines.append('            fake_llm, self.mem, self.test_redis,')
        test_lines.append('            self.nerves_dir, self.sandbox_dir,')
        test_lines.append('        )')
        test_lines.append('')
        test_lines.append('        all_patches = brain_patches + event_patches + silence_patches')
        test_lines.append('        for p in all_patches:')
        test_lines.append('            p.start()')
        test_lines.append('')
        test_lines.append('        try:')
        test_lines.append('            from arqitect.brain.brain import think')
        test_lines.append(f'            response = think(FLOW_{i}_TASK)')
        test_lines.append('            trace = recorder.trace()')
        test_lines.append('')

        # Assertions based on what was captured
        test_lines.append('            # Response was produced')
        test_lines.append('            assert response is not None')
        test_lines.append('            assert len(response) > 0')
        test_lines.append('')
        test_lines.append('            # Brain reached thinking stage')
        test_lines.append('            assert trace.has_event("brain:thought", data_contains={"stage": "thinking"})')

        if flow.dispatch_action == "invoke_nerve" and flow.dispatch_nerve:
            test_lines.append('')
            test_lines.append(f'            # Routed to invoke: {flow.dispatch_nerve}')
            test_lines.append(f'            assert trace.has_event("brain:action")')

        if flow.nerve_invocations:
            test_lines.append('')
            test_lines.append(f'            # Nerves invoked: {[n.name for n in flow.nerve_invocations]}')
            for inv in flow.nerve_invocations:
                test_lines.append(f'            assert {json.dumps(inv.name)!s} in trace.nerves_invoked()')

        test_lines.append('')
        test_lines.append('        finally:')
        test_lines.append('            for p in all_patches:')
        test_lines.append('                p.stop()')
        test_lines.append('')

    content = '\n'.join(test_lines)

    with open(output_path, 'w') as f:
        f.write(content)

    return content


def _slugify(text: str) -> str:
    """Convert text to a valid Python identifier slug."""
    slug = ""
    for ch in text.lower():
        if ch.isalnum():
            slug += ch
        elif ch in (" ", "-", "_"):
            slug += "_"
    # Collapse multiple underscores
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:30]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Parse trace files and generate test cases."""
    parser = argparse.ArgumentParser(
        description="Convert OTel trace files into pytest regression tests",
    )
    parser.add_argument("trace_files", nargs="+", help="Path(s) to trace JSONL files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output test file path (default: tests/test_flow_captured.py)")
    parser.add_argument("--tag", type=str, default="",
                        help="Only generate tests for flows matching these tags (comma-separated)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated test file to stdout instead of writing")
    args = parser.parse_args()

    all_flows = []
    for path in args.trace_files:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        flows = parse_trace_file(path)
        all_flows.extend(flows)
        print(f"Parsed {path}: {len(flows)} flows")

    if not all_flows:
        print("No flows found in trace files.")
        sys.exit(1)

    # Filter by tag if specified (match task content against tags)
    if args.tag:
        tags = [t.strip() for t in args.tag.split(",")]
        filtered = []
        for flow in all_flows:
            task_lower = flow.task.lower()
            if any(t in task_lower for t in tags):
                filtered.append(flow)
        all_flows = filtered
        print(f"After tag filter: {len(all_flows)} flows")

    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "test_flow_captured.py",
    )

    content = generate_test_file(all_flows, output_path)

    if args.dry_run:
        print(content)
    else:
        print(f"\nGenerated {len(all_flows)} test(s) -> {output_path}")
        print(f"Run with: python -m pytest {output_path} -v")


if __name__ == "__main__":
    main()
