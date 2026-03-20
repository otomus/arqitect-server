"""Flow tracing — capture execution flows for testing and debugging.

Records publish_event calls with parent-child span context, producing a
structured trace that tests can assert against. Zero external dependencies.

Usage in tests::

    def test_weather_flow(flow_recorder, ...):
        with flow_recorder.span("think"):
            result = think("what's the weather?")

        trace = flow_recorder.trace()
        assert trace.has_event(channel="brain:action", data_contains={"nerve": "weather"})

Usage for post-hoc analysis (no manual spans needed)::

    trace = flow_recorder.trace()
    for event in trace.events_for_channel("nerve:result"):
        print(event.data["nerve"], event.data["output"])
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator


# All module-level import sites for publish_event.
# Used by the fixture to patch every call site.
PUBLISH_EVENT_SITES: tuple[str, ...] = (
    "arqitect.brain.brain.publish_event",
    "arqitect.brain.dispatch.publish_event",
    "arqitect.brain.consolidate.publish_event",
    "arqitect.brain.synthesis.publish_event",
    "arqitect.brain.bootstrap.publish_event",
)


@dataclass(frozen=True)
class Span:
    """A named execution span with unique ID and optional parent.

    Args:
        span_id: Unique hex identifier.
        name: Human-readable span name (e.g. "think", "invoke:weather").
        parent_id: Parent span ID, or None for root spans.
    """

    span_id: str
    name: str
    parent_id: str | None


@dataclass(frozen=True)
class TraceEvent:
    """A single captured event within the trace.

    Args:
        timestamp: Wall-clock time (time.time()) for display.
        channel: Channel string (e.g. "brain:action").
        data: Snapshot of the event payload at capture time.
        span_id: Enclosing span, or None if recorded outside any span.
        parent_span_id: Parent of the enclosing span, for nesting queries.
        flow_id: Groups all events in one recorder session.
    """

    timestamp: float
    channel: str
    data: dict
    span_id: str | None
    parent_span_id: str | None
    flow_id: str


@dataclass
class FlowTrace:
    """Complete trace result with query helpers.

    Args:
        flow_id: Session identifier.
        events: Ordered list of captured events.
        spans: All spans created during the session.
    """

    flow_id: str
    events: list[TraceEvent] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)

    def events_for_channel(self, channel: str) -> list[TraceEvent]:
        """Return events matching a specific channel."""
        return [e for e in self.events if e.channel == channel]

    def events_in_span(self, span_name: str) -> list[TraceEvent]:
        """Return events that occurred inside a named span."""
        span_ids = {s.span_id for s in self.spans if s.name == span_name}
        return [e for e in self.events if e.span_id in span_ids]

    def child_spans(self, parent_name: str) -> list[Span]:
        """Return spans whose parent has the given name."""
        parent_ids = {s.span_id for s in self.spans if s.name == parent_name}
        return [s for s in self.spans if s.parent_id in parent_ids]

    def root_spans(self) -> list[Span]:
        """Return spans with no parent."""
        return [s for s in self.spans if s.parent_id is None]

    def has_event(self, channel: str, data_contains: dict | None = None) -> bool:
        """Check if an event exists matching channel and optional data subset.

        Args:
            channel: Channel to match.
            data_contains: If provided, every key-value pair must be present
                in the event's data dict.
        """
        for event in self.events:
            if event.channel != channel:
                continue
            if data_contains is None:
                return True
            if all(event.data.get(k) == v for k, v in data_contains.items()):
                return True
        return False

    def nerves_invoked(self) -> list[str]:
        """Return ordered list of nerve names from brain:action events."""
        return [
            e.data["nerve"]
            for e in self.events
            if e.channel == "brain:action" and "nerve" in e.data
        ]

    def nerves_resulted(self) -> list[dict]:
        """Return nerve:result event data in order."""
        return [e.data for e in self.events if e.channel == "nerve:result"]

    def dreamstate_stages(self) -> list[str]:
        """Return ordered list of dreamstate stage names."""
        return [
            e.data["stage"]
            for e in self.events
            if e.channel == "brain:thought" and "stage" in e.data
        ]


class FlowRecorder:
    """Captures publish_event calls with parent-child span context.

    Thread-safe. Each instance represents one recording session.
    Designed to be injected via pytest fixture — patches publish_event
    at all import sites and collects events into an in-memory trace.

    Args:
        flow_id: Optional session identifier. Generated if not provided.
    """

    def __init__(self, flow_id: str | None = None):
        self._flow_id = flow_id or uuid.uuid4().hex
        self._events: list[TraceEvent] = []
        self._spans: list[Span] = []
        self._span_stack: list[Span] = []
        self._lock = threading.Lock()

    @property
    def flow_id(self) -> str:
        """Return the session identifier."""
        return self._flow_id

    @contextlib.contextmanager
    def span(self, name: str) -> Iterator[Span]:
        """Open a named span for grouping events.

        Args:
            name: Human-readable span name.

        Yields:
            The created Span object.
        """
        span_id = uuid.uuid4().hex
        with self._lock:
            parent_id = self._span_stack[-1].span_id if self._span_stack else None
            span_obj = Span(span_id=span_id, name=name, parent_id=parent_id)
            self._spans.append(span_obj)
            self._span_stack.append(span_obj)
        try:
            yield span_obj
        finally:
            with self._lock:
                self._span_stack.pop()

    def record(self, channel: str, data: dict) -> None:
        """Record an event linked to the current span context.

        Args:
            channel: Event channel (e.g. "brain:action").
            data: Event payload. Shallow-copied to prevent mutation.
        """
        with self._lock:
            current_span = self._span_stack[-1] if self._span_stack else None
            event = TraceEvent(
                timestamp=time.time(),
                channel=channel,
                data=dict(data),
                span_id=current_span.span_id if current_span else None,
                parent_span_id=current_span.parent_id if current_span else None,
                flow_id=self._flow_id,
            )
            self._events.append(event)

    def intercept(self, channel: str, data: dict) -> None:
        """Drop-in replacement for publish_event. Records without publishing.

        This is the function passed as ``side_effect`` when patching
        publish_event at all import sites.

        Args:
            channel: Event channel.
            data: Event payload.
        """
        self.record(channel, data)

    def trace(self) -> FlowTrace:
        """Return the captured trace as an immutable snapshot."""
        with self._lock:
            return FlowTrace(
                flow_id=self._flow_id,
                events=list(self._events),
                spans=list(self._spans),
            )

    def clear(self) -> None:
        """Reset the recorder for reuse."""
        with self._lock:
            self._events.clear()
            self._spans.clear()
            self._span_stack.clear()
