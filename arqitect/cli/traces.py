"""Built-in trace server — serves a JSON API for the monitoring dashboard.

Exposes trace data from JSONL files via HTTP with CORS support so that
the GitHub Pages monitoring dashboard can connect to the local server.

Usage::

    python -m arqitect.cli.traces          # default port 7681
    python -m arqitect.cli.traces --port 9999
"""

from __future__ import annotations

import http.server
import json
import os
import urllib.parse
from pathlib import Path
from typing import Any

_TRACE_DIR = os.environ.get("ARQITECT_TRACE_DIR", "./traces")

# ---------------------------------------------------------------------------
# Trace file reading
# ---------------------------------------------------------------------------


def list_trace_files(trace_dir: str) -> list[dict[str, str]]:
    """Return available trace files sorted newest-first.

    Args:
        trace_dir: Directory containing .jsonl trace files.

    Returns:
        List of dicts with 'name', 'path', and 'size' keys.
    """
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        return []
    files = sorted(trace_path.glob("trace_*.jsonl"), reverse=True)
    return [
        {
            "name": f.name,
            "path": str(f),
            "size": f"{f.stat().st_size / 1024:.1f} KB",
        }
        for f in files
    ]


def read_trace_file(file_path: str) -> list[dict[str, Any]]:
    """Parse a JSONL trace file into a list of span dicts.

    Args:
        file_path: Path to a .jsonl trace file.

    Returns:
        List of span dictionaries.
    """
    spans = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans


def compute_stats(trace_dir: str) -> dict[str, Any]:
    """Compute aggregate stats across all trace files.

    Args:
        trace_dir: Directory containing .jsonl trace files.

    Returns:
        Summary statistics dict.
    """
    files = list_trace_files(trace_dir)
    total_spans = 0
    total_errors = 0
    span_type_counts: dict[str, int] = {}
    span_type_durations: dict[str, list[float]] = {}
    latest_spans: list[dict[str, Any]] = []

    for f_info in files[:10]:
        spans = read_trace_file(os.path.join(trace_dir, f_info["name"]))
        total_spans += len(spans)
        for s in spans:
            if s.get("status") == "ERROR":
                total_errors += 1
            name = s.get("name", "unknown")
            category = _span_category(name)
            span_type_counts[category] = span_type_counts.get(category, 0) + 1
            durations = span_type_durations.setdefault(category, [])
            durations.append(s.get("duration_ms", 0))

        if not latest_spans:
            latest_spans = spans

    type_stats = {}
    for category, durations in span_type_durations.items():
        sorted_d = sorted(durations)
        count = len(sorted_d)
        type_stats[category] = {
            "count": count,
            "avg_ms": round(sum(sorted_d) / count, 1) if count else 0,
            "p50_ms": round(sorted_d[count // 2], 1) if count else 0,
            "p95_ms": round(sorted_d[int(count * 0.95)], 1) if count else 0,
            "p99_ms": round(sorted_d[int(count * 0.99)], 1) if count else 0,
            "max_ms": round(sorted_d[-1], 1) if count else 0,
        }

    return {
        "total_files": len(files),
        "total_spans": total_spans,
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_spans * 100, 2) if total_spans else 0,
        "by_type": type_stats,
    }


def _span_category(name: str) -> str:
    """Classify a span name into a category.

    Args:
        name: The span name (e.g. 'llm.generate', 'nerve.invoke').

    Returns:
        Category string.
    """
    if name.startswith("llm."):
        return "llm"
    if name.startswith("nerve.synth") or name.startswith("synth."):
        return "synthesis"
    if name.startswith("nerve."):
        return "nerve"
    if name.startswith("dreamstate."):
        return "dreamstate"
    if name.startswith("brain."):
        return "brain"
    return "other"


# ---------------------------------------------------------------------------
# HTTP handler with CORS
# ---------------------------------------------------------------------------


class _TraceHandler(http.server.BaseHTTPRequestHandler):
    """Serves the trace JSON API with CORS for the GitHub Pages dashboard."""

    trace_dir: str = _TRACE_DIR

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        """Route GET requests to the appropriate handler."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api/files":
            self._json_response(list_trace_files(self.trace_dir))
        elif path == "/api/trace":
            params = urllib.parse.parse_qs(parsed.query)
            filename = params.get("file", [""])[0]
            self._serve_trace(filename)
        elif path == "/api/stats":
            self._json_response(compute_stats(self.trace_dir))
        elif path == "/api/health":
            self._json_response({"status": "ok"})
        else:
            self._json_response({"error": "not found"}, status=404)

    def _serve_trace(self, filename: str) -> None:
        if not filename or ".." in filename or "/" in filename:
            self._json_response({"error": "invalid filename"}, status=400)
            return
        file_path = os.path.join(self.trace_dir, filename)
        if not os.path.exists(file_path):
            self._json_response({"error": "file not found"}, status=404)
            return
        self._json_response(read_trace_file(file_path))

    def _json_response(self, data: Any, status: int = 200) -> None:
        """Send a JSON response with CORS headers.

        Args:
            data: JSON-serializable data.
            status: HTTP status code.
        """
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default request logging."""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(port: int = 7681, trace_dir: str = _TRACE_DIR) -> None:
    """Start the trace server.

    Args:
        port: Port to listen on.
        trace_dir: Directory containing .jsonl trace files.
    """
    _TraceHandler.trace_dir = trace_dir
    server = http.server.HTTPServer(("127.0.0.1", port), _TraceHandler)
    print(f"Trace server: http://localhost:{port}")
    print(f"Reading traces from: {os.path.abspath(trace_dir)}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arqitect Trace Server")
    parser.add_argument("--port", type=int, default=7681, help="Port (default: 7681)")
    parser.add_argument(
        "--trace-dir", default=_TRACE_DIR, help="Trace directory (default: ./traces)"
    )
    args = parser.parse_args()
    serve(port=args.port, trace_dir=args.trace_dir)
