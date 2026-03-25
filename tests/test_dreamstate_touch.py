"""Tests for Dreamstate.touch() and wake() — activity-aware idle timer.

touch() resets the idle timer without interrupting dreamstate, preventing
premature dream entry while the brain is actively working on chains.

wake() signals dreamstate to yield and must never block the brain for more
than a brief moment — tasks always take priority over dream phases.
"""

import threading
import time
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture()
def dreamstate():
    """Create a Dreamstate instance with mocked dependencies.

    Patches heavy imports (mem, llm, etc.) so we only test timer logic.
    """
    with patch("arqitect.brain.consolidate.mem"):
        from arqitect.brain.consolidate import Dreamstate
        ds = Dreamstate()
        yield ds
        # Clean up timer to avoid leaked threads
        if ds._timer:
            ds._timer.cancel()


class TestDreamstateTouch:
    """Verify touch() resets idle timer without interrupting dreamstate."""

    def test_touch_updates_last_activity(self, dreamstate):
        """touch() must advance _last_activity to (approximately) now."""
        old_activity = dreamstate._last_activity
        time.sleep(0.05)
        dreamstate.touch()
        assert dreamstate._last_activity > old_activity

    def test_touch_does_not_set_interrupted(self, dreamstate):
        """touch() must NOT set _interrupted — only wake() should do that."""
        dreamstate._interrupted.clear()
        dreamstate.touch()
        assert not dreamstate._interrupted.is_set()

    def test_touch_reschedules_timer(self, dreamstate):
        """touch() must cancel the old timer and start a new one."""
        old_timer = dreamstate._timer
        dreamstate.touch()
        new_timer = dreamstate._timer
        assert new_timer is not old_timer
        assert new_timer is not None
        assert new_timer.is_alive()

    def test_touch_does_not_join_worker(self, dreamstate):
        """touch() must not block on the worker thread (unlike wake())."""
        mock_worker = MagicMock(spec=threading.Thread)
        mock_worker.is_alive.return_value = True
        dreamstate._worker_thread = mock_worker
        dreamstate.touch()
        mock_worker.join.assert_not_called()

    def test_wake_still_interrupts_after_touch(self, dreamstate):
        """After touch(), wake() must still set _interrupted."""
        dreamstate.touch()
        assert not dreamstate._interrupted.is_set()
        dreamstate.wake()
        assert dreamstate._interrupted.is_set()


class TestDreamstateWake:
    """Verify wake() never blocks the brain waiting for dreamstate.

    Regression: wake() used to join(timeout=30), blocking task processing
    for up to 30s when dreamstate was in a non-interruptible phase. Tasks
    must always proceed immediately after signaling.
    """

    def test_wake_sets_interrupted_flag(self, dreamstate):
        """wake() must set _interrupted so dreamstate yields at next checkpoint."""
        dreamstate._interrupted.clear()
        dreamstate.wake()
        assert dreamstate._interrupted.is_set()

    def test_wake_updates_last_activity(self, dreamstate):
        """wake() must advance _last_activity to prevent immediate re-dreaming."""
        old = dreamstate._last_activity
        time.sleep(0.05)
        dreamstate.wake()
        assert dreamstate._last_activity > old

    def test_wake_does_not_block_when_worker_alive(self, dreamstate):
        """wake() must return within 2s even if the worker thread is stuck.

        Regression: previously join(timeout=30) caused 30s blocking.
        """
        stuck_event = threading.Event()

        def stuck_worker():
            stuck_event.wait(timeout=60)  # simulate non-interruptible phase

        worker = threading.Thread(target=stuck_worker, daemon=True)
        worker.start()
        dreamstate._worker_thread = worker

        start = time.time()
        dreamstate.wake()
        elapsed = time.time() - start

        # wake() must return in well under 2s (the join timeout should be ≤0.5s)
        assert elapsed < 2.0, f"wake() blocked for {elapsed:.1f}s — must not block tasks"

        # Clean up
        stuck_event.set()
        worker.join(timeout=2)

    def test_wake_reschedules_timer(self, dreamstate):
        """wake() must reschedule the idle timer after signaling."""
        old_timer = dreamstate._timer
        dreamstate.wake()
        assert dreamstate._timer is not old_timer
        assert dreamstate._timer is not None

    def test_wake_clears_conversation(self, dreamstate):
        """wake() must clear stale dream conversation when worker was alive."""
        mock_worker = MagicMock(spec=threading.Thread)
        mock_worker.is_alive.return_value = True
        # join returns immediately (simulating fast yield)
        mock_worker.join.return_value = None
        # After join, is_alive returns False
        mock_worker.is_alive.side_effect = [True, False]
        dreamstate._worker_thread = mock_worker

        with patch("arqitect.brain.consolidate.mem") as mock_mem:
            dreamstate.wake()
            mock_mem.hot.clear_conversation.assert_called_once()

    def test_rapid_wake_calls_dont_block(self, dreamstate):
        """Multiple rapid wake() calls must not accumulate blocking time.

        Regression: during stress tests, each incoming task calls wake().
        If each blocks 30s, tasks pile up catastrophically.
        """
        stuck_event = threading.Event()

        def stuck_worker():
            stuck_event.wait(timeout=60)

        worker = threading.Thread(target=stuck_worker, daemon=True)
        worker.start()
        dreamstate._worker_thread = worker

        start = time.time()
        for _ in range(10):
            dreamstate.wake()
        elapsed = time.time() - start

        # 10 wake() calls must complete in well under 10s total
        assert elapsed < 8.0, f"10x wake() took {elapsed:.1f}s — must not accumulate blocking"

        stuck_event.set()
        worker.join(timeout=2)
