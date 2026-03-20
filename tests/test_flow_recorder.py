"""Tests for arqitect.tracing — FlowRecorder, Span, TraceEvent, FlowTrace."""

from __future__ import annotations

import threading

import pytest

from arqitect.tracing import FlowRecorder, Span, TraceEvent, FlowTrace, PUBLISH_EVENT_SITES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestConstants:
    """Verify the publish_event site list stays in sync."""

    def test_all_sites_are_strings(self):
        for site in PUBLISH_EVENT_SITES:
            assert isinstance(site, str)
            assert site.startswith("arqitect.")

    def test_sites_cover_core_modules(self):
        modules = {s.rsplit(".", 1)[0] for s in PUBLISH_EVENT_SITES}
        assert "arqitect.brain.brain" in modules
        assert "arqitect.brain.dispatch" in modules
        assert "arqitect.brain.consolidate" in modules


# ---------------------------------------------------------------------------
# FlowRecorder — basic recording
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestRecordEvents:
    """Recording events without spans."""

    def test_record_captures_channel_and_data(self):
        recorder = FlowRecorder()
        recorder.record("brain:action", {"nerve": "weather"})

        trace = recorder.trace()
        assert len(trace.events) == 1
        assert trace.events[0].channel == "brain:action"
        assert trace.events[0].data == {"nerve": "weather"}

    def test_record_snapshots_data(self):
        """Mutation of the original dict must not affect captured data."""
        recorder = FlowRecorder()
        data = {"key": "original"}
        recorder.record("test", data)
        data["key"] = "mutated"

        assert recorder.trace().events[0].data["key"] == "original"

    def test_events_ordered_by_insertion(self):
        recorder = FlowRecorder()
        for i in range(5):
            recorder.record("ch", {"seq": i})

        trace = recorder.trace()
        assert [e.data["seq"] for e in trace.events] == [0, 1, 2, 3, 4]

    def test_intercept_is_alias_for_record(self):
        recorder = FlowRecorder()
        recorder.intercept("brain:thought", {"stage": "planning"})

        trace = recorder.trace()
        assert len(trace.events) == 1
        assert trace.events[0].channel == "brain:thought"

    def test_flow_id_assigned(self):
        recorder = FlowRecorder()
        recorder.record("ch", {})

        trace = recorder.trace()
        assert len(trace.flow_id) > 0
        assert trace.events[0].flow_id == trace.flow_id

    def test_custom_flow_id(self):
        recorder = FlowRecorder(flow_id="test-session-1")
        assert recorder.flow_id == "test-session-1"

    def test_clear_resets_state(self):
        recorder = FlowRecorder()
        recorder.record("ch", {"a": 1})
        with recorder.span("s"):
            recorder.record("ch", {"b": 2})

        recorder.clear()
        trace = recorder.trace()
        assert len(trace.events) == 0
        assert len(trace.spans) == 0


# ---------------------------------------------------------------------------
# FlowRecorder — span nesting
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestSpanNesting:
    """Parent-child span relationships."""

    def test_single_span(self):
        recorder = FlowRecorder()
        with recorder.span("think") as span:
            recorder.record("brain:action", {"nerve": "weather"})

        trace = recorder.trace()
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "think"
        assert trace.spans[0].parent_id is None
        assert trace.events[0].span_id == span.span_id

    def test_nested_spans(self):
        recorder = FlowRecorder()
        with recorder.span("think") as parent:
            with recorder.span("invoke:weather") as child:
                recorder.record("nerve:result", {"output": "sunny"})

        trace = recorder.trace()
        assert len(trace.spans) == 2
        assert child.parent_id == parent.span_id
        assert trace.events[0].span_id == child.span_id
        assert trace.events[0].parent_span_id == parent.span_id

    def test_three_levels_deep(self):
        recorder = FlowRecorder()
        with recorder.span("think") as root:
            with recorder.span("dispatch") as mid:
                with recorder.span("invoke:weather") as leaf:
                    recorder.record("nerve:result", {})

        assert root.parent_id is None
        assert mid.parent_id == root.span_id
        assert leaf.parent_id == mid.span_id

    def test_sequential_spans_are_siblings(self):
        recorder = FlowRecorder()
        with recorder.span("invoke:weather") as s1:
            recorder.record("nerve:result", {"nerve": "weather"})
        with recorder.span("invoke:timer") as s2:
            recorder.record("nerve:result", {"nerve": "timer"})

        assert s1.parent_id is None
        assert s2.parent_id is None

    def test_event_outside_span_has_no_span_id(self):
        recorder = FlowRecorder()
        recorder.record("standalone", {"key": "val"})

        event = recorder.trace().events[0]
        assert event.span_id is None
        assert event.parent_span_id is None

    def test_span_pops_on_exception(self):
        """Span stack must be restored even if body raises."""
        recorder = FlowRecorder()
        with pytest.raises(ValueError):
            with recorder.span("failing"):
                raise ValueError("boom")

        # Stack should be empty — next span has no parent
        with recorder.span("after") as s:
            pass
        assert s.parent_id is None


# ---------------------------------------------------------------------------
# FlowTrace — query helpers
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestFlowTraceQueries:
    """Query methods on FlowTrace."""

    def _sample_trace(self) -> FlowTrace:
        recorder = FlowRecorder(flow_id="test")
        with recorder.span("think"):
            recorder.record("brain:thought", {"stage": "planning"})
            recorder.record("brain:action", {"nerve": "weather"})
            with recorder.span("invoke:weather"):
                recorder.record("nerve:result", {"nerve": "weather", "output": "sunny"})
            recorder.record("brain:action", {"nerve": "timer"})
            with recorder.span("invoke:timer"):
                recorder.record("nerve:result", {"nerve": "timer", "output": "set"})
        recorder.record("brain:thought", {"stage": "reconciliation_start"})
        return recorder.trace()

    def test_events_for_channel(self):
        trace = self._sample_trace()
        actions = trace.events_for_channel("brain:action")
        assert len(actions) == 2
        assert actions[0].data["nerve"] == "weather"
        assert actions[1].data["nerve"] == "timer"

    def test_events_in_span(self):
        trace = self._sample_trace()
        events = trace.events_in_span("invoke:weather")
        assert len(events) == 1
        assert events[0].data["output"] == "sunny"

    def test_child_spans(self):
        trace = self._sample_trace()
        children = trace.child_spans("think")
        names = {s.name for s in children}
        assert "invoke:weather" in names
        assert "invoke:timer" in names

    def test_root_spans(self):
        trace = self._sample_trace()
        roots = trace.root_spans()
        assert len(roots) == 1
        assert roots[0].name == "think"

    def test_has_event_by_channel(self):
        trace = self._sample_trace()
        assert trace.has_event("brain:action")
        assert not trace.has_event("nonexistent:channel")

    def test_has_event_with_data_contains(self):
        trace = self._sample_trace()
        assert trace.has_event("brain:action", data_contains={"nerve": "weather"})
        assert not trace.has_event("brain:action", data_contains={"nerve": "missing"})

    def test_nerves_invoked(self):
        trace = self._sample_trace()
        assert trace.nerves_invoked() == ["weather", "timer"]

    def test_nerves_resulted(self):
        trace = self._sample_trace()
        results = trace.nerves_resulted()
        assert len(results) == 2
        assert results[0]["nerve"] == "weather"

    def test_dreamstate_stages(self):
        trace = self._sample_trace()
        stages = trace.dreamstate_stages()
        assert "planning" in stages
        assert "reconciliation_start" in stages

    def test_empty_trace(self):
        trace = FlowTrace(flow_id="empty")
        assert trace.events_for_channel("any") == []
        assert trace.nerves_invoked() == []
        assert trace.root_spans() == []
        assert not trace.has_event("any")


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestThreadSafety:
    """Concurrent recording must not lose events."""

    def test_concurrent_record(self):
        recorder = FlowRecorder()
        events_per_thread = 100
        thread_count = 4

        def record_batch(thread_id: int):
            for i in range(events_per_thread):
                recorder.record("ch", {"thread": thread_id, "seq": i})

        threads = [
            threading.Thread(target=record_batch, args=(t,))
            for t in range(thread_count)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        trace = recorder.trace()
        assert len(trace.events) == events_per_thread * thread_count

    def test_concurrent_spans(self):
        recorder = FlowRecorder()

        def span_work(name: str):
            with recorder.span(name):
                recorder.record("ch", {"span": name})

        threads = [
            threading.Thread(target=span_work, args=(f"span_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        trace = recorder.trace()
        assert len(trace.spans) == 10
        assert len(trace.events) == 10


# ---------------------------------------------------------------------------
# Fixture integration
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestFlowRecorderFixture:
    """Verify the flow_recorder fixture patches publish_event."""

    def test_fixture_captures_publish_event(self, flow_recorder):
        """Events published via a patched import site are captured."""
        # Import from a patched site (brain.brain), not events.py directly
        from arqitect.brain import brain as brain_mod
        brain_mod.publish_event("brain:thought", {"stage": "test"})

        trace = flow_recorder.trace()
        assert trace.has_event("brain:thought", data_contains={"stage": "test"})

    def test_fixture_returns_empty_trace_initially(self, flow_recorder):
        trace = flow_recorder.trace()
        assert len(trace.events) == 0
        assert len(trace.spans) == 0
