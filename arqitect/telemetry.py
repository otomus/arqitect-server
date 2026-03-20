"""OpenTelemetry instrumentation for the arqitect brain.

Instruments the critical path: think → dispatch → invoke/synthesize,
plus every LLM call through generate_for_role. Traces are exported to
a JSON file (one per session) for post-hoc analysis and test generation.

Enable by calling ``init_telemetry()`` before the brain starts listening.
Disabled by default — zero overhead when not initialized.

Usage::

    from arqitect.telemetry import init_telemetry
    init_telemetry()  # call once at startup

Environment variables:
    ARQITECT_TRACE_DIR: Directory for trace JSON files (default: ./traces)
    ARQITECT_TRACE_OTLP: If set, also export to OTLP endpoint (e.g. localhost:4317)
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode

_TRACE_DIR = os.environ.get("ARQITECT_TRACE_DIR", "./traces")
_initialized = False

tracer = trace.get_tracer("arqitect.brain")


# ---------------------------------------------------------------------------
# JSON file exporter — writes spans to a JSONL file for offline analysis
# ---------------------------------------------------------------------------

class JSONFileExporter(SpanExporter):
    """Export spans to a JSON-Lines file.

    Each line is a complete span dict with timing, attributes, events,
    and parent context. Files are named by session start time.

    Args:
        trace_dir: Directory to write trace files into.
    """

    def __init__(self, trace_dir: str):
        self._trace_dir = Path(trace_dir)
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        session_ts = time.strftime("%Y%m%d_%H%M%S")
        self._file_path = self._trace_dir / f"trace_{session_ts}.jsonl"
        self._file = open(self._file_path, "a")

    def export(self, spans: list) -> SpanExportResult:
        """Write each span as a JSON line."""
        for span in spans:
            record = _span_to_dict(span)
            self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Close the trace file."""
        self._file.close()

    def force_flush(self, timeout_millis: int = 0) -> bool:
        """Flush pending writes."""
        self._file.flush()
        return True

    @property
    def file_path(self) -> Path:
        """Return the path to the current trace file."""
        return self._file_path


def _span_to_dict(span) -> dict:
    """Convert an OTel ReadableSpan to a JSON-serializable dict."""
    ctx = span.get_span_context()
    parent = span.parent
    return {
        "name": span.name,
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": format(parent.span_id, "016x") if parent else None,
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "duration_ms": round((span.end_time - span.start_time) / 1e6, 2),
        "status": span.status.status_code.name,
        "attributes": dict(span.attributes) if span.attributes else {},
        "events": [
            {
                "name": e.name,
                "timestamp_ns": e.timestamp,
                "attributes": dict(e.attributes) if e.attributes else {},
            }
            for e in span.events
        ],
    }


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_telemetry() -> Path:
    """Initialize OpenTelemetry tracing with JSON file export.

    Safe to call multiple times — only initializes once.

    Returns:
        Path to the trace JSONL file being written.
    """
    global _initialized
    if _initialized:
        # Return existing file path
        provider = trace.get_tracer_provider()
        for processor in getattr(provider, '_active_span_processor', {}).get('_span_processors', []):
            if hasattr(processor, 'span_exporter') and isinstance(processor.span_exporter, JSONFileExporter):
                return processor.span_exporter.file_path
        return Path(_TRACE_DIR) / "trace_existing.jsonl"

    resource = Resource.create({"service.name": "arqitect-brain"})
    provider = TracerProvider(resource=resource)

    # Always export to JSON file
    json_exporter = JSONFileExporter(_TRACE_DIR)
    provider.add_span_processor(SimpleSpanProcessor(json_exporter))

    # Optionally export to OTLP collector (Jaeger, Grafana, etc.)
    otlp_endpoint = os.environ.get("ARQITECT_TRACE_OTLP")
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))

    trace.set_tracer_provider(provider)
    _initialized = True

    _patch_brain()
    _patch_inference()
    _patch_dispatch()
    _patch_invoke()
    _patch_synthesis()

    return json_exporter.file_path


# ---------------------------------------------------------------------------
# Monkey-patch instrumentation — wraps real functions with tracing spans
# ---------------------------------------------------------------------------

def _patch_brain():
    """Instrument brain.think() with a root span."""
    from arqitect.brain import brain as brain_mod

    original_think = brain_mod.think

    @functools.wraps(original_think)
    def traced_think(task: str, history: list[str] | None = None, depth: int = 0) -> str:
        with tracer.start_as_current_span("brain.think") as span:
            span.set_attribute("task", task[:500])
            span.set_attribute("depth", depth)
            span.set_attribute("has_history", bool(history))
            try:
                result = original_think(task, history=history, depth=depth)
                span.set_attribute("response_length", len(result) if result else 0)
                span.set_attribute("response_preview", (result or "")[:300])
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    brain_mod.think = traced_think


def _patch_inference():
    """Instrument generate_for_role() — captures every LLM call."""
    from arqitect.inference import router as router_mod

    original_generate = router_mod.generate_for_role

    @functools.wraps(original_generate)
    def traced_generate(role: str, prompt: str, system: str = "", **kwargs) -> str:
        with tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("llm.role", role)
            span.set_attribute("llm.prompt", prompt[:2000])
            span.set_attribute("llm.system", system[:1000])
            span.set_attribute("llm.prompt_length", len(prompt))
            for k, v in kwargs.items():
                if v is not None:
                    span.set_attribute(f"llm.{k}", str(v))
            try:
                result = original_generate(role, prompt, system=system, **kwargs)
                span.set_attribute("llm.response", result[:2000])
                span.set_attribute("llm.response_length", len(result))
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    router_mod.generate_for_role = traced_generate

    # Also patch the helpers wrapper so both paths are covered
    from arqitect.brain import helpers as helpers_mod
    original_helper = helpers_mod.llm_generate

    @functools.wraps(original_helper)
    def traced_helper(model: str, prompt: str, system: str = "") -> str:
        # The actual tracing happens in generate_for_role above,
        # but we add a thin span to track the caller context
        with tracer.start_as_current_span("brain.llm_generate") as span:
            span.set_attribute("llm.model", model)
            return original_helper(model, prompt, system=system)

    helpers_mod.llm_generate = traced_helper


def _patch_dispatch():
    """Instrument dispatch_action() — captures routing decisions."""
    from arqitect.brain import dispatch as dispatch_mod

    original_dispatch = dispatch_mod.dispatch_action

    @functools.wraps(original_dispatch)
    def traced_dispatch(ctx) -> str | None:
        with tracer.start_as_current_span("brain.dispatch") as span:
            action = ctx.decision.get("action", "unknown")
            nerve_name = ctx.decision.get("name", "")
            span.set_attribute("dispatch.action", action)
            span.set_attribute("dispatch.nerve", nerve_name)
            span.set_attribute("dispatch.depth", ctx.depth)
            span.set_attribute("dispatch.decision", json.dumps(ctx.decision, default=str)[:1000])
            try:
                result = original_dispatch(ctx)
                span.set_attribute("dispatch.response_length", len(result) if result else 0)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    dispatch_mod.dispatch_action = traced_dispatch


def _patch_invoke():
    """Instrument invoke_nerve() — captures nerve execution."""
    from arqitect.brain import invoke as invoke_mod

    original_invoke = invoke_mod.invoke_nerve

    @functools.wraps(original_invoke)
    def traced_invoke(name: str, args: str, user_id: str = "") -> str:
        with tracer.start_as_current_span("nerve.invoke") as span:
            span.set_attribute("nerve.name", name)
            span.set_attribute("nerve.args", args[:500])
            span.set_attribute("nerve.user_id", user_id)
            try:
                result = original_invoke(name, args, user_id=user_id)
                span.set_attribute("nerve.output", result[:2000] if result else "")
                span.set_attribute("nerve.output_length", len(result) if result else 0)
                # Check for error in output
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and "error" in parsed:
                        span.set_attribute("nerve.error", parsed["error"])
                        span.set_status(StatusCode.ERROR, parsed["error"])
                except (json.JSONDecodeError, TypeError):
                    pass
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    invoke_mod.invoke_nerve = traced_invoke

    # Also patch at import sites in brain.py and dispatch.py
    from arqitect.brain import brain as brain_mod
    from arqitect.brain import dispatch as dispatch_mod
    brain_mod.invoke_nerve = traced_invoke
    dispatch_mod.invoke_nerve = traced_invoke


def _patch_synthesis():
    """Instrument synthesize_nerve() — captures nerve creation."""
    from arqitect.brain import synthesis as synth_mod

    original_synth = synth_mod.synthesize_nerve

    @functools.wraps(original_synth)
    def traced_synthesize(name: str, description: str, mcp_tools=None,
                          trigger_task: str = "", role: str | None = None):
        with tracer.start_as_current_span("nerve.synthesize") as span:
            span.set_attribute("synth.name", name)
            span.set_attribute("synth.description", description[:500])
            span.set_attribute("synth.trigger_task", trigger_task[:500])
            if role:
                span.set_attribute("synth.role", role)
            try:
                actual_name, nerve_path = original_synth(
                    name, description, mcp_tools=mcp_tools,
                    trigger_task=trigger_task, role=role,
                )
                span.set_attribute("synth.actual_name", actual_name)
                span.set_attribute("synth.nerve_path", nerve_path)
                return actual_name, nerve_path
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    synth_mod.synthesize_nerve = traced_synthesize

    # Also patch at import sites
    from arqitect.brain import brain as brain_mod
    from arqitect.brain import dispatch as dispatch_mod
    brain_mod.synthesize_nerve = traced_synthesize
    dispatch_mod.synthesize_nerve = traced_synthesize
