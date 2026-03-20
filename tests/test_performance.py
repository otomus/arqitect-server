"""Performance tests for core arqitect modules.

Verifies that critical operations complete within acceptable time bounds.
All external dependencies (LLM, Redis, subprocesses) are mocked.
Uses time.perf_counter() for high-resolution timing.
"""

import json
import time
import threading
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cold(tmp_path):
    """Create a ColdMemory backed by a temporary SQLite database."""
    db_path = str(tmp_path / "knowledge.db")
    with patch("arqitect.memory.cold._DB_PATH", db_path):
        from arqitect.memory.cold import ColdMemory
        return ColdMemory()


@pytest.fixture
def warm(tmp_path):
    """Create a WarmMemory backed by a temporary SQLite database."""
    db_path = str(tmp_path / "episodes.db")
    with patch("arqitect.memory.warm._DB_PATH", db_path):
        from arqitect.memory.warm import WarmMemory
        return WarmMemory()


@pytest.fixture
def hot():
    """Create a HotMemory backed by fakeredis."""
    import fakeredis
    from arqitect.memory.hot import HotMemory
    client = fakeredis.FakeRedis(decode_responses=True)
    return HotMemory(client)


# ---------------------------------------------------------------------------
# 1. Matching Performance
# ---------------------------------------------------------------------------

class TestMatchingPerformance:
    """Verify that keyword-scoring functions scale to realistic catalog sizes."""

    @patch("arqitect.matching._get_nerve_embedding", return_value=None)
    def test_match_score_throughput(self, _mock_emb):
        """match_score x1000 should complete in < 1 second."""
        from arqitect.matching import match_score

        start = time.perf_counter()
        for _ in range(1000):
            match_score(
                "weather forecast today",
                "weather_nerve",
                "Get weather forecasts and current conditions",
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"match_score x1000 took {elapsed:.2f}s, expected < 1.0s"

    @patch("arqitect.matching._get_nerve_embedding", return_value=None)
    def test_match_score_with_params(self, _mock_emb):
        """match_score with params x1000 should complete in < 1.5 seconds."""
        from arqitect.matching import match_score

        params = {"location": "city name", "units": "metric or imperial"}
        start = time.perf_counter()
        for _ in range(1000):
            match_score(
                "get temperature in celsius for new york",
                "temperature_converter",
                "Convert temperatures between different units and fetch forecasts",
                params=params,
            )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.5, f"match_score with params x1000 took {elapsed:.2f}s, expected < 1.5s"

    @patch("arqitect.matching._get_nerve_embedding", return_value=None)
    @patch("arqitect.inference.engine.get_engine", side_effect=ImportError("no engine in test"))
    def test_match_nerves_100(self, _mock_engine, _mock_emb):
        """match_nerves with 100 nerves should complete in < 2 seconds."""
        from arqitect.matching import match_nerves

        catalog = {
            f"nerve_{i}": f"This nerve handles task category {i} with domain-specific logic"
            for i in range(100)
        }
        start = time.perf_counter()
        match_nerves("handle task category 42 with domain logic", catalog, threshold=0.5)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"match_nerves x100 took {elapsed:.2f}s, expected < 2.0s"

    @patch("arqitect.matching._get_nerve_embedding", return_value=None)
    def test_match_tools_50(self, _mock_emb):
        """match_tools with 50 tools should complete in < 1 second."""
        from arqitect.matching import match_tools

        tools = {
            f"tool_{i}": {"description": f"Perform action {i} on the system resource"}
            for i in range(50)
        }
        start = time.perf_counter()
        match_tools("perform action on system resource", tools, threshold=0.5)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"match_tools x50 took {elapsed:.2f}s, expected < 1.0s"

    @patch("arqitect.matching._get_nerve_embedding", return_value=None)
    def test_find_duplicate_nerves_50(self, _mock_emb):
        """find_duplicate_nerves with 50 nerves should complete in < 3 seconds."""
        from arqitect.matching import find_duplicate_nerves

        catalog = {
            f"nerve_{i}": f"Handle requests related to domain {i} and process data"
            for i in range(50)
        }
        start = time.perf_counter()
        find_duplicate_nerves(catalog, threshold=3.0)
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"find_duplicate_nerves x50 took {elapsed:.2f}s, expected < 3.0s"

    @patch("arqitect.matching._get_nerve_embedding", return_value=None)
    @patch("arqitect.inference.engine.get_engine", side_effect=ImportError("no engine in test"))
    def test_match_nerves_repeated(self, _mock_engine, _mock_emb):
        """match_nerves called 10 times on 50-nerve catalog should complete in < 3 seconds."""
        from arqitect.matching import match_nerves

        catalog = {
            f"nerve_{i}": f"Nerve {i} processes weather data and location lookups"
            for i in range(50)
        }
        queries = [
            "weather forecast", "location lookup", "data processing",
            "temperature conversion", "wind speed analysis",
            "humidity reading", "barometric pressure", "sunrise sunset",
            "UV index check", "air quality monitor",
        ]
        start = time.perf_counter()
        for query in queries:
            match_nerves(query, catalog, threshold=0.5)
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"match_nerves x10 on 50-nerve catalog took {elapsed:.2f}s, expected < 3.0s"


# ---------------------------------------------------------------------------
# 2. Cold Memory Performance
# ---------------------------------------------------------------------------

class TestColdMemoryPerformance:
    """Verify ColdMemory operations scale within acceptable bounds."""

    def test_insert_and_retrieve_500_facts(self, cold):
        """Insert 500 facts, then retrieve all in a category: should complete in < 2 seconds."""
        category = "perf_test"
        start = time.perf_counter()
        for i in range(500):
            cold.set_fact(category, f"key_{i}", f"value_{i}")
        facts = cold.get_facts(category)
        elapsed = time.perf_counter() - start
        assert len(facts) == 500
        assert elapsed < 2.0, f"500 fact insert + retrieve took {elapsed:.2f}s, expected < 2.0s"

    def test_register_and_list_100_nerves(self, cold):
        """Register 100 nerves, then list_nerves: should complete in < 1 second."""
        start = time.perf_counter()
        for i in range(100):
            cold.register_nerve(f"nerve_{i}", f"Description for nerve {i}")
        nerves = cold.list_nerves()
        elapsed = time.perf_counter() - start
        assert len(nerves) == 100
        assert elapsed < 1.0, f"100 nerve register + list took {elapsed:.2f}s, expected < 1.0s"

    def test_get_all_nerve_data_100(self, cold):
        """get_all_nerve_data with 100 nerves: should complete in < 2 seconds."""
        for i in range(100):
            cold.register_nerve(f"nerve_{i}", f"Description for nerve {i}")
            cold.add_nerve_tool(f"nerve_{i}", f"tool_a_{i}")
            cold.add_nerve_tool(f"nerve_{i}", f"tool_b_{i}")

        start = time.perf_counter()
        data = cold.get_all_nerve_data()
        elapsed = time.perf_counter() - start
        assert len(data) == 100
        assert elapsed < 2.0, f"get_all_nerve_data x100 took {elapsed:.2f}s, expected < 2.0s"

    def test_nerve_invocation_recording(self, cold):
        """Record 500 nerve invocations: should complete in < 2 seconds."""
        cold.register_nerve("perf_nerve", "A nerve for performance testing")
        start = time.perf_counter()
        for i in range(500):
            cold.record_nerve_invocation("perf_nerve", success=(i % 3 != 0))
        elapsed = time.perf_counter() - start
        info = cold.get_nerve_info("perf_nerve")
        assert info["total_invocations"] == 500
        assert elapsed < 2.0, f"500 invocation records took {elapsed:.2f}s, expected < 2.0s"


# ---------------------------------------------------------------------------
# 3. Warm Memory Performance
# ---------------------------------------------------------------------------

class TestWarmMemoryPerformance:
    """Verify WarmMemory episodic storage scales within acceptable bounds."""

    @patch("arqitect.matching._get_nerve_embedding", return_value=None)
    def test_record_500_episodes_then_recall(self, _mock_emb, warm):
        """Record 500 episodes, then recall: should complete in < 2 seconds."""
        start = time.perf_counter()
        for i in range(500):
            warm.record({
                "timestamp": time.time() - (500 - i),
                "task": f"perform task {i} involving weather data",
                "nerve": f"nerve_{i % 10}",
                "tool": f"tool_{i % 5}",
                "args": {"param": f"value_{i}"},
                "result_summary": f"Completed task {i} successfully",
                "success": True,
                "tokens": 100 + i,
            })
        results = warm.recall("weather data task", limit=10)
        elapsed = time.perf_counter() - start
        assert len(results) <= 10
        assert elapsed < 2.0, f"500 episode record + recall took {elapsed:.2f}s, expected < 2.0s"

    @patch("arqitect.matching._get_nerve_embedding", return_value=None)
    def test_pruning_at_max_episodes(self, _mock_emb, warm):
        """Pruning at MAX_EPISODES boundary: should complete in < 1 second."""
        from arqitect.memory.warm import MAX_EPISODES

        # Fill to MAX_EPISODES
        for i in range(MAX_EPISODES):
            warm.record({
                "timestamp": time.time() - (MAX_EPISODES - i),
                "task": f"task {i}",
                "nerve": "test_nerve",
                "tool": "test_tool",
                "result_summary": "ok",
                "success": True,
            })

        # Now record 50 more — each triggers pruning
        start = time.perf_counter()
        for i in range(50):
            warm.record({
                "timestamp": time.time(),
                "task": f"overflow task {i}",
                "nerve": "test_nerve",
                "tool": "test_tool",
                "result_summary": "ok",
                "success": True,
            })
        elapsed = time.perf_counter() - start

        count = warm.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert count <= MAX_EPISODES
        assert elapsed < 1.0, f"50 prune-triggering inserts took {elapsed:.2f}s, expected < 1.0s"


# ---------------------------------------------------------------------------
# 4. Hot Memory Performance (fakeredis)
# ---------------------------------------------------------------------------

class TestHotMemoryPerformance:
    """Verify HotMemory Redis operations complete quickly with fakeredis."""

    def test_add_100_messages_and_get_conversation(self, hot):
        """Add 100 messages and get_conversation: should complete in < 1 second."""
        start = time.perf_counter()
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            hot.add_message(role, f"Message number {i} with some content", user_id="user_1")
        conversation = hot.get_conversation(limit=20, user_id="user_1")
        elapsed = time.perf_counter() - start
        assert len(conversation) == 20
        assert elapsed < 1.0, f"100 messages + get took {elapsed:.2f}s, expected < 1.0s"

    def test_set_get_50_session_values(self, hot):
        """Set/get 50 session values: should complete in < 0.5 seconds."""
        start = time.perf_counter()
        for i in range(50):
            hot.set_session({f"key_{i}": f"value_{i}"}, user_id="user_perf")
        for i in range(50):
            session = hot.get_session(user_id="user_perf")
        elapsed = time.perf_counter() - start
        assert session  # non-empty
        assert elapsed < 0.5, f"50 session set/get took {elapsed:.2f}s, expected < 0.5s"

    def test_multiple_users_conversation(self, hot):
        """Conversations for 20 users with 10 messages each: should complete in < 1 second."""
        start = time.perf_counter()
        for user_idx in range(20):
            user_id = f"user_{user_idx}"
            for msg_idx in range(10):
                hot.add_message("user", f"Hello from {user_id} msg {msg_idx}", user_id=user_id)
        # Retrieve all conversations
        for user_idx in range(20):
            hot.get_conversation(limit=10, user_id=f"user_{user_idx}")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"20 users x 10 messages + retrieve took {elapsed:.2f}s, expected < 1.0s"


# ---------------------------------------------------------------------------
# 5. JSON Extraction Performance
# ---------------------------------------------------------------------------

class TestJsonExtractionPerformance:
    """Verify extract_json handles volume without degradation."""

    def test_extract_json_1000_short_strings(self):
        """extract_json on 1000 short strings: should complete in < 1 second."""
        from arqitect.brain.helpers import extract_json

        samples = [
            f'Some preamble text. ###JSON: {{"action": "invoke", "id": {i}}}' for i in range(1000)
        ]
        start = time.perf_counter()
        for sample in samples:
            result = extract_json(sample)
            assert result is not None
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"extract_json x1000 short took {elapsed:.2f}s, expected < 1.0s"

    def test_extract_json_100_long_strings(self):
        """extract_json on 100 long strings (1000+ chars): should complete in < 2 seconds."""
        from arqitect.brain.helpers import extract_json

        padding = "x" * 900
        samples = [
            f"Reasoning: {padding} thinking about task {i} with extra context... "
            f'###JSON: {{"action": "invoke_nerve", "nerve": "nerve_{i}", '
            f'"args": {{"query": "some long query text for item {i} with more detail"}}}}'
            for i in range(100)
        ]
        assert all(len(s) > 1000 for s in samples), f"min len={min(len(s) for s in samples)}"

        start = time.perf_counter()
        for sample in samples:
            result = extract_json(sample)
            assert result is not None
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"extract_json x100 long took {elapsed:.2f}s, expected < 2.0s"

    def test_extract_json_nested_objects(self):
        """extract_json on 500 strings with nested JSON: should complete in < 1.5 seconds."""
        from arqitect.brain.helpers import extract_json

        samples = [
            f'{{"outer": {{"inner": {{"value": {i}, "list": [1, 2, 3]}}, "name": "test_{i}"}}}}'
            for i in range(500)
        ]
        start = time.perf_counter()
        for sample in samples:
            result = extract_json(sample)
            assert result is not None
        elapsed = time.perf_counter() - start
        assert elapsed < 1.5, f"extract_json x500 nested took {elapsed:.2f}s, expected < 1.5s"


# ---------------------------------------------------------------------------
# 6. Config Loader Performance
# ---------------------------------------------------------------------------

class TestConfigLoaderPerformance:
    """Verify config loading and access are fast, especially with caching."""

    def test_load_config_100_times_cached(self):
        """load_config called 100 times (cached): should complete in < 0.1 seconds."""
        from arqitect.config.loader import load_config

        # Warm the cache
        load_config()

        start = time.perf_counter()
        for _ in range(100):
            load_config()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"load_config x100 cached took {elapsed:.2f}s, expected < 0.1s"

    def test_get_config_deep_paths_1000(self):
        """get_config with deep paths 1000 times: should complete in < 0.5 seconds."""
        from arqitect.config.loader import get_config

        paths = [
            "inference.provider",
            "inference.models.brain",
            "paths.nerves",
            "storage.hot.url",
            "ports.mcp",
        ]
        start = time.perf_counter()
        for _ in range(200):
            for path in paths:
                get_config(path)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"get_config x1000 took {elapsed:.2f}s, expected < 0.5s"


# ---------------------------------------------------------------------------
# 7. Calibration Protocol Performance
# ---------------------------------------------------------------------------

class TestCalibrationProtocolPerformance:
    """Verify calibration helpers are fast enough for startup probes."""

    def test_derive_status_20_capabilities(self):
        """derive_status with 20 capabilities: should complete in < 0.01 seconds per call."""
        from arqitect.senses.calibration_protocol import derive_status

        capabilities = {
            f"cap_{i}": {"available": i % 3 != 0, "detail": f"Capability {i}"}
            for i in range(20)
        }
        start = time.perf_counter()
        for _ in range(100):
            derive_status(capabilities)
        elapsed = time.perf_counter() - start
        per_call = elapsed / 100
        assert per_call < 0.01, f"derive_status per call took {per_call:.4f}s, expected < 0.01s"

    def test_build_result_performance(self):
        """build_result: should complete in < 0.01 seconds per call."""
        from arqitect.senses.calibration_protocol import build_result

        capabilities = {
            f"cap_{i}": {"available": True} for i in range(10)
        }
        deps = {f"dep_{i}": {"installed": True, "path": f"/usr/bin/dep_{i}"} for i in range(5)}
        config = {"setting_a": "value_a", "setting_b": 42}

        start = time.perf_counter()
        for _ in range(100):
            build_result(
                sense="test_sense",
                capabilities=capabilities,
                deps=deps,
                config=config,
                user_actions=["Install dep_x"],
                auto_installable=["dep_y"],
            )
        elapsed = time.perf_counter() - start
        per_call = elapsed / 100
        assert per_call < 0.01, f"build_result per call took {per_call:.4f}s, expected < 0.01s"

    def test_derive_status_edge_cases(self):
        """derive_status with empty and full availability: should be near-instant."""
        from arqitect.senses.calibration_protocol import derive_status

        cases = [
            {},
            {f"c_{i}": {"available": True} for i in range(20)},
            {f"c_{i}": {"available": False} for i in range(20)},
            {f"c_{i}": {"available": i == 0} for i in range(20)},
        ]
        start = time.perf_counter()
        for _ in range(500):
            for case in cases:
                derive_status(case)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"derive_status x2000 edge cases took {elapsed:.2f}s, expected < 0.1s"


# ---------------------------------------------------------------------------
# 8. Concurrent Access Performance
# ---------------------------------------------------------------------------

class TestConcurrentAccessPerformance:
    """Verify thread-safety and performance under concurrent access."""

    def test_cold_memory_concurrent_writes(self, cold):
        """10 threads writing to ColdMemory simultaneously: should complete in < 3 seconds."""
        errors = []

        def writer(thread_id: int):
            try:
                for i in range(50):
                    cold.set_fact(
                        f"thread_{thread_id}",
                        f"key_{i}",
                        f"value_{thread_id}_{i}",
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        assert not errors, f"Concurrent writes raised errors: {errors}"
        # Verify data integrity — each thread wrote 50 facts
        for thread_id in range(10):
            facts = cold.get_facts(f"thread_{thread_id}")
            assert len(facts) == 50, f"Thread {thread_id} expected 50 facts, got {len(facts)}"
        assert elapsed < 3.0, f"10-thread concurrent writes took {elapsed:.2f}s, expected < 3.0s"

    def test_hot_memory_concurrent_reads(self, hot):
        """10 threads reading from HotMemory simultaneously: should complete in < 1 second."""
        # Seed conversation data
        for i in range(50):
            hot.add_message("user", f"Message {i}", user_id="shared_user")

        errors = []

        def reader(thread_id: int):
            try:
                for _ in range(50):
                    msgs = hot.get_conversation(limit=20, user_id="shared_user")
                    assert len(msgs) == 20
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader, args=(t,)) for t in range(10)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        assert not errors, f"Concurrent reads raised errors: {errors}"
        assert elapsed < 1.0, f"10-thread concurrent reads took {elapsed:.2f}s, expected < 1.0s"

    def test_cold_memory_mixed_read_write(self, cold):
        """5 writers + 5 readers on ColdMemory: should complete in < 3 seconds."""
        # Pre-seed some data for readers
        for i in range(100):
            cold.register_nerve(f"base_nerve_{i}", f"Base description {i}")

        errors = []

        def writer(thread_id: int):
            try:
                for i in range(20):
                    cold.register_nerve(
                        f"new_nerve_{thread_id}_{i}",
                        f"Thread {thread_id} nerve {i} description",
                    )
            except Exception as exc:
                errors.append(exc)

        def reader(thread_id: int):
            try:
                for _ in range(20):
                    cold.list_nerves()
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=writer, args=(t,)) for t in range(5)]
            + [threading.Thread(target=reader, args=(t,)) for t in range(5)]
        )
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        assert not errors, f"Mixed read/write raised errors: {errors}"
        assert elapsed < 3.0, f"Mixed concurrent access took {elapsed:.2f}s, expected < 3.0s"
