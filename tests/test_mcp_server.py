"""Tests for arqitect.mcp.server — HTTP endpoints for MCP tool server."""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool_mgr_mock():
    """Mock ToolManager with controllable scan/list/call/get_meta."""
    mgr = MagicMock()
    mgr.scan.return_value = None
    mgr.list_tools.return_value = {}
    mgr.get_meta.return_value = None
    mgr.call.return_value = "tool result"
    return mgr


@pytest.fixture
def ext_mcp_mock():
    """Mock ExternalMCPManager with controllable list/call."""
    mgr = MagicMock()
    mgr.list_all_tools.return_value = {}
    mgr.get_server_for_tool.return_value = None
    mgr.call_tool.return_value = "ext result"
    mgr.install_from_npm.return_value = {"ok": True}
    mgr.install_from_github.return_value = {"ok": True}
    mgr.shutdown_all.return_value = None
    return mgr


@pytest.fixture
def client(tool_mgr_mock, ext_mcp_mock):
    """Provide a Starlette TestClient wired to the MCP server with mocked managers."""
    with patch("arqitect.mcp.server.tool_mgr", tool_mgr_mock), \
         patch("arqitect.mcp.server.ext_mcp", ext_mcp_mock), \
         patch("arqitect.mcp.server._publish_tool_event"), \
         patch("arqitect.mcp.server._record_tool_usage"), \
         patch("arqitect.mcp.server._record_mcp_usage"):
        from arqitect.mcp.server import mcp
        app = mcp.http_app()
        yield TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client, tool_mgr_mock, ext_mcp_mock):
        """Health endpoint returns status ok with tool lists."""
        tool_mgr_mock.list_tools.return_value = {"my_tool": {"description": "d"}}
        ext_mcp_mock.list_all_tools.return_value = {"ext_tool": {"description": "e"}}

        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "my_tool" in body["managed_tools"]
        assert "ext_tool" in body["external_tools"]
        assert "my_tool" in body["tools"]
        assert "ext_tool" in body["tools"]

    def test_health_scans_tools(self, client, tool_mgr_mock):
        """Health endpoint triggers a scan."""
        client.get("/health")
        tool_mgr_mock.scan.assert_called()

    def test_health_empty_tools(self, client):
        """Health endpoint works with zero tools."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["tools"] == []


# ---------------------------------------------------------------------------
# /tools
# ---------------------------------------------------------------------------

class TestListToolsEndpoint:
    def test_list_managed_tools(self, client, tool_mgr_mock):
        """Lists managed tools from ToolManager."""
        tool_mgr_mock.list_tools.return_value = {
            "weather": {"description": "Get weather", "params": {"city": "str"}},
        }
        resp = client.get("/tools")
        assert resp.status_code == 200
        tools = resp.json()["tools"]
        assert "weather" in tools

    def test_list_merges_external_tools(self, client, tool_mgr_mock, ext_mcp_mock):
        """External tools are merged into the list."""
        tool_mgr_mock.list_tools.return_value = {"local": {"description": "local"}}
        ext_mcp_mock.list_all_tools.return_value = {
            "remote": {"description": "remote tool", "params": {}, "server": "s1"},
        }
        resp = client.get("/tools")
        tools = resp.json()["tools"]
        assert "local" in tools
        assert "remote" in tools
        assert tools["remote"]["source"] == "external"

    def test_managed_tool_not_overwritten_by_external(self, client, tool_mgr_mock, ext_mcp_mock):
        """If a managed tool has the same name as an external, managed wins."""
        tool_mgr_mock.list_tools.return_value = {"dup": {"description": "managed version"}}
        ext_mcp_mock.list_all_tools.return_value = {
            "dup": {"description": "ext version", "params": {}, "server": "s"},
        }
        resp = client.get("/tools")
        tools = resp.json()["tools"]
        assert tools["dup"]["description"] == "managed version"


# ---------------------------------------------------------------------------
# /call/{tool_name}
# ---------------------------------------------------------------------------

class TestCallToolEndpoint:
    def test_call_managed_tool_success(self, client, tool_mgr_mock):
        """Calling a managed tool returns its result."""
        meta = MagicMock()
        meta.params = {"query": "str"}
        tool_mgr_mock.get_meta.return_value = meta
        tool_mgr_mock.call.return_value = "sunny 25C"

        resp = client.post("/call/weather", json={"query": "NYC"})
        assert resp.status_code == 200
        assert resp.json()["result"] == "sunny 25C"

    def test_call_managed_tool_error(self, client, tool_mgr_mock):
        """Managed tool raising exception returns 500."""
        meta = MagicMock()
        meta.params = {"q": "str"}
        tool_mgr_mock.get_meta.return_value = meta
        tool_mgr_mock.call.side_effect = RuntimeError("boom")

        resp = client.post("/call/broken", json={"q": "x"})
        assert resp.status_code == 500
        assert "Tool error" in resp.json()["error"]

    def test_call_external_tool_success(self, client, tool_mgr_mock, ext_mcp_mock):
        """Falls through to external MCP when managed tool not found."""
        tool_mgr_mock.get_meta.return_value = None
        ext_mcp_mock.get_server_for_tool.return_value = "ext_server"
        ext_mcp_mock.call_tool.return_value = "ext result"

        resp = client.post("/call/ext_tool", json={"a": 1})
        assert resp.status_code == 200
        assert resp.json()["result"] == "ext result"

    def test_call_external_tool_error(self, client, tool_mgr_mock, ext_mcp_mock):
        """External tool raising exception returns 500."""
        tool_mgr_mock.get_meta.return_value = None
        ext_mcp_mock.get_server_for_tool.return_value = "ext_server"
        ext_mcp_mock.call_tool.side_effect = ConnectionError("down")

        resp = client.post("/call/ext_tool", json={})
        assert resp.status_code == 500
        assert "External tool error" in resp.json()["error"]

    def test_call_unknown_tool_404(self, client, tool_mgr_mock, ext_mcp_mock):
        """Unknown tool returns 404."""
        tool_mgr_mock.get_meta.return_value = None
        ext_mcp_mock.get_server_for_tool.return_value = None

        resp = client.post("/call/nonexistent", json={})
        assert resp.status_code == 404
        assert "Unknown tool" in resp.json()["error"]

    def test_call_with_empty_body(self, client, tool_mgr_mock):
        """Call with no JSON body defaults to empty dict."""
        meta = MagicMock()
        meta.params = {}
        tool_mgr_mock.get_meta.return_value = meta
        tool_mgr_mock.call.return_value = "ok"

        resp = client.post("/call/noop", content=b"not json",
                           headers={"content-type": "application/json"})
        # Should not crash — body falls back to {}
        assert resp.status_code == 200

    def test_call_remaps_wrong_param_names(self, client, tool_mgr_mock):
        """When model sends wrong param names, args get remapped."""
        meta = MagicMock()
        meta.params = {"city": "str", "units": "str"}
        tool_mgr_mock.get_meta.return_value = meta
        tool_mgr_mock.call.return_value = "result"

        # Send with mismatched param names
        resp = client.post("/call/weather", json={"location": "NYC", "units": "metric"})
        assert resp.status_code == 200
        # The unmatched value "NYC" should be mapped to remaining param "city"
        call_args = tool_mgr_mock.call.call_args[0][1]
        assert call_args.get("city") == "NYC"
        assert call_args.get("units") == "metric"


# ---------------------------------------------------------------------------
# /install
# ---------------------------------------------------------------------------

class TestInstallEndpoint:
    def test_install_npm(self, client, ext_mcp_mock):
        """Install from npm."""
        resp = client.post("/install", json={"source": "npm", "package": "@foo/bar"})
        assert resp.status_code == 200
        ext_mcp_mock.install_from_npm.assert_called_once_with("@foo/bar", server_name=None)

    def test_install_github(self, client, ext_mcp_mock):
        """Install from github."""
        resp = client.post("/install", json={"source": "github", "repo_url": "https://github.com/a/b"})
        assert resp.status_code == 200
        ext_mcp_mock.install_from_github.assert_called_once()

    def test_install_missing_source_returns_400(self, client):
        """Missing package/repo_url returns 400."""
        resp = client.post("/install", json={"source": "npm"})
        assert resp.status_code == 400

    def test_install_invalid_json_returns_400(self, client):
        """Invalid JSON body returns 400."""
        resp = client.post("/install", content=b"not json",
                           headers={"content-type": "application/json"})
        assert resp.status_code == 400

    def test_install_with_custom_name(self, client, ext_mcp_mock):
        """Custom server name is forwarded."""
        resp = client.post("/install", json={
            "source": "npm", "package": "pkg", "name": "custom"
        })
        assert resp.status_code == 200
        ext_mcp_mock.install_from_npm.assert_called_once_with("pkg", server_name="custom")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestMonitoringHelpers:
    def test_record_tool_usage_silences_errors(self):
        """_record_tool_usage never raises."""
        with patch("arqitect.mcp.server._ensure_monitoring") as mock_mon:
            mock_mon.side_effect = RuntimeError("db down")
            from arqitect.mcp.server import _record_tool_usage
            # Should not raise
            _record_tool_usage("test", True, 100.0)

    def test_record_mcp_usage_silences_errors(self):
        """_record_mcp_usage never raises."""
        with patch("arqitect.mcp.server._ensure_monitoring") as mock_mon:
            mock_mon.side_effect = RuntimeError("db down")
            from arqitect.mcp.server import _record_mcp_usage
            _record_mcp_usage("srv", "tool", True, 50.0)

    def test_publish_tool_event_silences_errors(self):
        """_publish_tool_event never raises even when Redis is unavailable."""
        with patch("arqitect.mcp.server._redis", None):
            with patch.dict("sys.modules", {"redis": MagicMock(Redis=MagicMock(side_effect=ConnectionError))}):
                from arqitect.mcp.server import _publish_tool_event
                # Should not raise
                _publish_tool_event("t", {"a": "b"}, error="err")

    def test_ensure_monitoring_lazy_init(self):
        """_ensure_monitoring creates MonitoringMemory once."""
        with patch("arqitect.mcp.server._monitoring", None):
            mock_cls = MagicMock()
            with patch.dict("sys.modules", {"arqitect.memory.monitoring": MagicMock(MonitoringMemory=mock_cls)}):
                from arqitect.mcp.server import _ensure_monitoring
                result = _ensure_monitoring()
                assert result is not None
