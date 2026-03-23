"""Tests for Dreamstate.touch() — activity-aware idle timer.

touch() resets the idle timer without interrupting dreamstate, preventing
premature dream entry while the brain is actively working on chains.
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
