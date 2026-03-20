"""Usage tracking tests — nerve and tool calls recorded in dedicated monitoring.db.

Covers:
- MonitoringMemory: record_call, get_usage_report, get_error_details, prune
- Nerve invocation recording via invoke_nerve()
- Tool call recording via MCP server _record_tool_usage()
- Dream state usage report generation to community repo
"""

import json
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest
from dirty_equals import IsFloat, IsInstance, IsPositiveInt, IsStr
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from tests.conftest import make_nerve_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monitoring(tmp_memory_dir):
    """Provide an isolated MonitoringMemory backed by a temp database."""
    db_path = str(tmp_memory_dir / "monitoring.db")
    with patch("arqitect.memory.monitoring._DB_PATH", db_path):
        from arqitect.memory.monitoring import MonitoringMemory
        yield MonitoringMemory()


@pytest.fixture
def community_dir(tmp_path):
    """Create a fake community repo directory."""
    d = tmp_path / "arqitect-community"
    d.mkdir()
    (d / ".git").mkdir()
    return d


# ---------------------------------------------------------------------------
# MonitoringMemory — core operations
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestMonitoringMemory:
    """Dedicated monitoring database records per-call data."""

    def test_empty_report(self, monitoring):
        """Report is empty when no calls have been recorded."""
        report = monitoring.get_usage_report()
        assert report["nerves"] == []
        assert report["tools"] == []
        assert report["mcps"] == []

    def test_record_nerve_calls(self, monitoring):
        """Nerve calls are aggregated correctly in the report."""
        monitoring.record_call("nerve", "weather_nerve", success=True, latency_ms=120)
        monitoring.record_call("nerve", "weather_nerve", success=True, latency_ms=80)
        monitoring.record_call("nerve", "weather_nerve", success=False, latency_ms=200,
                               error_message="Model crashed")

        report = monitoring.get_usage_report()
        assert len(report["nerves"]) == 1
        assert report["tools"] == []

        nerve = report["nerves"][0]
        assert nerve["name"] == "weather_nerve"
        assert nerve["total"] == 3
        assert nerve["successes"] == 2
        assert nerve["failures"] == 1
        assert nerve["error_rate"] == pytest.approx(1 / 3, abs=0.001)
        assert nerve["avg_latency_ms"] == pytest.approx(133.3, abs=1)

    def test_record_tool_calls(self, monitoring):
        """Tool calls are aggregated with error rates."""
        monitoring.record_call("tool", "web_search", success=True, latency_ms=50)
        monitoring.record_call("tool", "web_search", success=False, latency_ms=5000,
                               error_message="Timeout")

        report = monitoring.get_usage_report()
        assert len(report["tools"]) == 1

        tool = report["tools"][0]
        assert tool["name"] == "web_search"
        assert tool["total"] == 2
        assert tool["error_rate"] == 0.5

    def test_sorted_by_total_descending(self, monitoring):
        """Report entries are sorted by total calls, most used first."""
        monitoring.record_call("nerve", "rare_nerve", success=True)
        for _ in range(5):
            monitoring.record_call("nerve", "popular_nerve", success=True)

        report = monitoring.get_usage_report()
        assert report["nerves"][0]["name"] == "popular_nerve"
        assert report["nerves"][1]["name"] == "rare_nerve"

    def test_report_since_filter(self, monitoring):
        """get_usage_report(since=) only includes calls after the timestamp."""
        monitoring.record_call("nerve", "old_nerve", success=True)
        cutoff = time.time()
        time.sleep(0.01)  # Ensure new_nerve timestamp is strictly after cutoff
        monitoring.record_call("nerve", "new_nerve", success=True)

        report = monitoring.get_usage_report(since=cutoff)
        names = [n["name"] for n in report["nerves"]]
        assert "new_nerve" in names
        assert "old_nerve" not in names

    def test_mcp_calls_tracked_separately(self, monitoring):
        """MCP calls appear in their own section, not mixed with tools."""
        monitoring.record_call("tool", "web_search", success=True)
        monitoring.record_call("mcp", "slack/send_message", success=True)
        monitoring.record_call("mcp", "slack/send_message", success=False,
                               error_message="Auth failed")

        report = monitoring.get_usage_report()
        assert len(report["tools"]) == 1
        assert len(report["mcps"]) == 1
        assert report["tools"][0]["name"] == "web_search"
        assert report["mcps"][0]["name"] == "slack/send_message"
        assert report["mcps"][0]["error_rate"] == 0.5

    def test_get_error_details(self, monitoring):
        """get_error_details returns recent errors for a specific subject."""
        monitoring.record_call("tool", "flaky_tool", success=False,
                               error_message="Connection refused")
        monitoring.record_call("tool", "flaky_tool", success=True)
        monitoring.record_call("tool", "flaky_tool", success=False,
                               error_message="Timeout")

        errors = monitoring.get_error_details("tool", "flaky_tool")
        assert len(errors) == 2
        # Most recent first
        assert errors[0]["error_message"] == "Timeout"
        assert errors[1]["error_message"] == "Connection refused"

    def test_prune_removes_old_entries(self, monitoring):
        """prune() deletes entries older than the specified number of days."""
        # Insert a call with a timestamp 60 days ago
        old_timestamp = time.time() - (60 * 86400)
        monitoring.conn.execute(
            "INSERT INTO call_log (timestamp, subject_type, subject_name, success) "
            "VALUES (?, 'nerve', 'ancient_nerve', 1)",
            (old_timestamp,),
        )
        monitoring.conn.commit()

        monitoring.record_call("nerve", "recent_nerve", success=True)

        monitoring.prune(older_than_days=30)

        report = monitoring.get_usage_report()
        names = [n["name"] for n in report["nerves"]]
        assert "recent_nerve" in names
        assert "ancient_nerve" not in names

    def test_error_message_truncated(self, monitoring):
        """Error messages longer than 500 chars are truncated."""
        long_error = "x" * 1000
        monitoring.record_call("tool", "verbose_tool", success=False,
                               error_message=long_error)
        errors = monitoring.get_error_details("tool", "verbose_tool")
        assert len(errors[0]["error_message"]) == 500

    @given(
        subject_type=st.sampled_from(["nerve", "tool", "mcp"]),
        name=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=("L", "N"), whitelist_characters="_-/"
        )),
        successes=st.integers(min_value=0, max_value=10),
        failures=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_report_entry_shape(self, subject_type, name, successes, failures, tmp_memory_dir):
        """Every report entry has the expected keys and value types."""
        assume(successes + failures > 0)
        import uuid
        db_path = str(tmp_memory_dir / f"monitoring_{uuid.uuid4().hex[:8]}.db")
        with patch("arqitect.memory.monitoring._DB_PATH", db_path):
            from arqitect.memory.monitoring import MonitoringMemory
            mon = MonitoringMemory()

            for _ in range(successes):
                mon.record_call(subject_type, name, success=True, latency_ms=10.0)
            for _ in range(failures):
                mon.record_call(subject_type, name, success=False, latency_ms=20.0,
                                error_message="err")

            report = mon.get_usage_report()

            # Find the correct bucket
            bucket_key = {"nerve": "nerves", "tool": "tools", "mcp": "mcps"}[subject_type]
            entries = report[bucket_key]
            assert len(entries) == 1

            entry = entries[0]
            assert entry["name"] == name
            assert entry["total"] == successes + failures
            assert entry["successes"] == successes
            assert entry["failures"] == failures
            assert entry["error_rate"] == IsFloat(ge=0.0, le=1.0)
            assert entry["avg_latency_ms"] == IsFloat(ge=0.0)
            assert entry["last_called_at"] == IsFloat(gt=0.0)


# ---------------------------------------------------------------------------
# Nerve invocation tracking via invoke_nerve()
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestNerveInvocationTracking:
    """invoke_nerve records calls in the monitoring database."""

    def test_successful_invocation_recorded(self, nerves_dir, sandbox_dir, mem, monitoring):
        """A nerve that exits 0 records a success in monitoring."""
        make_nerve_file(nerves_dir, "tracker_nerve")

        mock_result = MagicMock()
        mock_result.stdout = '{"response": "ok"}'
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("arqitect.brain.invoke.mem", mem), \
             patch("arqitect.brain.invoke._monitoring", monitoring), \
             patch("arqitect.brain.invoke.subprocess.run", return_value=mock_result):
            from arqitect.brain.invoke import invoke_nerve
            invoke_nerve("tracker_nerve", "test")

        report = monitoring.get_usage_report()
        assert len(report["nerves"]) == 1
        nerve = report["nerves"][0]
        assert nerve["name"] == "tracker_nerve"
        assert nerve["successes"] == 1
        assert nerve["avg_latency_ms"] == IsFloat(ge=0.0)

    def test_failed_invocation_recorded(self, nerves_dir, sandbox_dir, mem, monitoring):
        """A nerve that exits non-zero records a failure in monitoring."""
        make_nerve_file(nerves_dir, "failing_nerve")

        mock_result = MagicMock()
        mock_result.stdout = '{"error": "broke"}'
        mock_result.stderr = ""
        mock_result.returncode = 1

        with patch("arqitect.brain.invoke.mem", mem), \
             patch("arqitect.brain.invoke._monitoring", monitoring), \
             patch("arqitect.brain.invoke.subprocess.run", return_value=mock_result):
            from arqitect.brain.invoke import invoke_nerve
            invoke_nerve("failing_nerve", "test")

        report = monitoring.get_usage_report()
        assert report["nerves"][0]["failures"] == 1

    def test_timeout_recorded_as_failure(self, nerves_dir, sandbox_dir, mem, monitoring):
        """A nerve that times out records a failure with error message."""
        make_nerve_file(nerves_dir, "slow_nerve")

        def _timeout_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=90)

        with patch("arqitect.brain.invoke.mem", mem), \
             patch("arqitect.brain.invoke._monitoring", monitoring), \
             patch("arqitect.brain.invoke.subprocess.run", side_effect=_timeout_run):
            from arqitect.brain.invoke import invoke_nerve
            invoke_nerve("slow_nerve", "test")

        report = monitoring.get_usage_report()
        assert report["nerves"][0]["failures"] == 1

        errors = monitoring.get_error_details("nerve", "slow_nerve")
        assert "Timed out" in errors[0]["error_message"]

    def test_monitoring_failure_does_not_break_invocation(self, nerves_dir, sandbox_dir, mem):
        """If monitoring db fails, invoke_nerve still returns the nerve output."""
        make_nerve_file(nerves_dir, "robust_nerve")

        mock_result = MagicMock()
        mock_result.stdout = '{"response": "still works"}'
        mock_result.stderr = ""
        mock_result.returncode = 0

        broken_monitoring = MagicMock()
        broken_monitoring.record_call.side_effect = RuntimeError("DB locked")

        with patch("arqitect.brain.invoke.mem", mem), \
             patch("arqitect.brain.invoke._monitoring", broken_monitoring), \
             patch("arqitect.brain.invoke.subprocess.run", return_value=mock_result):
            from arqitect.brain.invoke import invoke_nerve
            result = invoke_nerve("robust_nerve", "test")

        assert "still works" in result


# ---------------------------------------------------------------------------
# Tool call tracking via MCP server
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestToolCallTracking:
    """_record_tool_usage persists stats in monitoring database."""

    def test_record_tool_success(self, monitoring):
        """Successful tool calls are recorded with latency."""
        from arqitect.mcp.server import _record_tool_usage

        with patch("arqitect.mcp.server._monitoring", monitoring):
            _record_tool_usage("web_search", success=True, latency_ms=45.2)
            _record_tool_usage("web_search", success=True, latency_ms=52.1)
            _record_tool_usage("web_search", success=False, latency_ms=5001,
                               error_message="Timeout")

        report = monitoring.get_usage_report()
        tool = report["tools"][0]
        assert tool["name"] == "web_search"
        assert tool["total"] == 3
        assert tool["successes"] == 2
        assert tool["failures"] == 1

    def test_record_mcp_usage(self, monitoring):
        """External MCP calls are tracked with server/tool name format."""
        from arqitect.mcp.server import _record_mcp_usage

        with patch("arqitect.mcp.server._monitoring", monitoring):
            _record_mcp_usage("slack", "send_message", success=True, latency_ms=200)
            _record_mcp_usage("github", "create_issue", success=False,
                              latency_ms=5000, error_message="Auth expired")

        report = monitoring.get_usage_report()
        assert len(report["mcps"]) == 2
        names = {m["name"] for m in report["mcps"]}
        assert "slack/send_message" in names
        assert "github/create_issue" in names

    def test_record_failure_is_silent(self):
        """If monitoring raises, _record_tool_usage does not propagate."""
        from arqitect.mcp.server import _record_tool_usage

        broken = MagicMock()
        broken.record_call.side_effect = RuntimeError("DB locked")

        with patch("arqitect.mcp.server._monitoring", broken):
            _record_tool_usage("broken_tool", success=True)  # Should not raise


# ---------------------------------------------------------------------------
# Dream state — usage report generation
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestDreamUsageReport:
    """Dream state reads from monitoring.db and writes reports to community."""

    def test_writes_report_to_community(self, mem, monitoring, community_dir):
        """_dream_usage_report writes a JSON file to community reports/."""
        monitoring.record_call("nerve", "test_nerve", success=True, latency_ms=100)
        monitoring.record_call("tool", "test_tool", success=True, latency_ms=50)
        monitoring.record_call("tool", "test_tool", success=False, latency_ms=5000,
                               error_message="Timeout")

        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = MagicMock()
        ds._interrupted.is_set.return_value = False

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.memory.monitoring._DB_PATH",
                   monitoring.conn.execute("PRAGMA database_list").fetchone()[2]):
            ds._find_community_dir = MagicMock(return_value=str(community_dir))
            ds._dream_usage_report()

        reports_dir = community_dir / "reports"
        assert reports_dir.exists()
        report_files = list(reports_dir.glob("usage_*.json"))
        assert len(report_files) == 1

        report = json.loads(report_files[0].read_text())
        assert len(report["nerves"]) == 1
        assert len(report["tools"]) == 1
        assert report["nerves"][0]["name"] == "test_nerve"
        assert report["nerves"][0]["avg_latency_ms"] == 100.0
        assert report["tools"][0]["name"] == "test_tool"
        assert report["tools"][0]["error_rate"] == 0.5
        assert report["generated_at"] == IsStr()
        assert report["instance_id"] == IsStr()

    def test_skips_when_no_community_dir(self, mem):
        """No error when community repo is not found."""
        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = MagicMock()
        ds._interrupted.is_set.return_value = False

        with patch("arqitect.brain.consolidate.mem", mem):
            ds._find_community_dir = MagicMock(return_value=None)
            ds._dream_usage_report()  # Should not raise

    def test_skips_when_no_usage_data(self, mem, monitoring, community_dir):
        """No report file written when there are no stats."""
        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = MagicMock()
        ds._interrupted.is_set.return_value = False

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.memory.monitoring._DB_PATH",
                   monitoring.conn.execute("PRAGMA database_list").fetchone()[2]):
            ds._find_community_dir = MagicMock(return_value=str(community_dir))
            ds._dream_usage_report()

        reports_dir = community_dir / "reports"
        assert not reports_dir.exists() or len(list(reports_dir.glob("*.json"))) == 0

    def test_stable_instance_id(self, mem, monitoring, community_dir):
        """Instance ID is generated once and reused across dream cycles."""
        monitoring.record_call("nerve", "n1", success=True)

        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = MagicMock()
        ds._interrupted.is_set.return_value = False

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.memory.monitoring._DB_PATH",
                   monitoring.conn.execute("PRAGMA database_list").fetchone()[2]):
            ds._find_community_dir = MagicMock(return_value=str(community_dir))
            ds._dream_usage_report()

            # Second run — same instance_id, overwrites same file
            monitoring.record_call("nerve", "n1", success=True)
            ds._dream_usage_report()

        reports_dir = community_dir / "reports"
        report_files = list(reports_dir.glob("usage_*.json"))
        assert len(report_files) == 1  # Same file overwritten

    def test_report_does_not_touch_knowledge_db(self, mem, monitoring, community_dir):
        """Usage report reads from monitoring.db, never from knowledge.db."""
        monitoring.record_call("nerve", "tracked_nerve", success=True)
        # Cold memory has a different nerve — should NOT appear in report
        mem.cold.register_nerve("cold_only_nerve", "Not tracked")
        mem.cold.record_nerve_invocation("cold_only_nerve", success=True)

        from arqitect.brain.consolidate import Dreamstate

        ds = Dreamstate.__new__(Dreamstate)
        ds._interrupted = MagicMock()
        ds._interrupted.is_set.return_value = False

        with patch("arqitect.brain.consolidate.mem", mem), \
             patch("arqitect.memory.monitoring._DB_PATH",
                   monitoring.conn.execute("PRAGMA database_list").fetchone()[2]):
            ds._find_community_dir = MagicMock(return_value=str(community_dir))
            ds._dream_usage_report()

        report = json.loads(
            list((community_dir / "reports").glob("*.json"))[0].read_text()
        )
        names = [n["name"] for n in report["nerves"]]
        assert "tracked_nerve" in names
        assert "cold_only_nerve" not in names
