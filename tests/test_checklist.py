"""Tests for arqitect.brain.checklist — generic task checklist with pass/fail checkpoints."""

import os

import pytest

from arqitect.brain.checklist import TaskChecklist


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_steps(*names: str) -> list[dict]:
    """Build a step list from step names."""
    return [{"name": n} for n in names]


@pytest.fixture()
def checklist():
    """A three-step checklist for general testing."""
    return TaskChecklist(
        task_id="t1",
        goal="Deploy widget",
        steps=_make_steps("Build", "Test", "Deploy"),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    """Verify initial state after creating a TaskChecklist."""

    def test_stores_task_id_and_goal(self, checklist):
        assert checklist.task_id == "t1"
        assert checklist.goal == "Deploy widget"

    def test_initializes_steps_as_pending(self, checklist):
        for step in checklist.steps:
            assert step["status"] == "pending"
            assert step["result"] == ""

    def test_preserves_step_names(self, checklist):
        names = [s["name"] for s in checklist.steps]
        assert names == ["Build", "Test", "Deploy"]

    def test_empty_steps_list(self):
        cl = TaskChecklist("t0", "nothing", [])
        assert cl.steps == []

    def test_step_missing_name_key_defaults_to_empty(self):
        cl = TaskChecklist("t0", "g", [{"description": "no name key"}])
        assert cl.steps[0]["name"] == ""

    def test_single_step(self):
        cl = TaskChecklist("t0", "one step", _make_steps("only"))
        assert len(cl.steps) == 1
        assert cl.steps[0]["name"] == "only"


# ---------------------------------------------------------------------------
# activate
# ---------------------------------------------------------------------------

class TestActivate:
    """Tests for marking a step as in-progress."""

    def test_activate_sets_status_to_active(self, checklist):
        checklist.activate(0)
        assert checklist.steps[0]["status"] == "active"

    def test_activate_does_not_affect_other_steps(self, checklist):
        checklist.activate(1)
        assert checklist.steps[0]["status"] == "pending"
        assert checklist.steps[2]["status"] == "pending"

    def test_activate_out_of_range_positive_is_no_op(self, checklist):
        checklist.activate(99)
        assert all(s["status"] == "pending" for s in checklist.steps)

    def test_activate_negative_index_is_no_op(self, checklist):
        # Negative indices satisfy 0 <= idx < len check for -1 only if
        # len > abs(idx), but the guard uses 0 <= step_index, so -1 is rejected.
        checklist.activate(-1)
        assert all(s["status"] == "pending" for s in checklist.steps)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

class TestCheck:
    """Tests for recording step pass/fail results."""

    def test_check_pass(self, checklist):
        checklist.check(0, True, "built ok")
        assert checklist.steps[0]["status"] == "pass"
        assert checklist.steps[0]["result"] == "built ok"

    def test_check_fail(self, checklist):
        checklist.check(1, False, "tests failed")
        assert checklist.steps[1]["status"] == "fail"
        assert checklist.steps[1]["result"] == "tests failed"

    def test_check_out_of_range_is_no_op(self, checklist):
        checklist.check(100, True, "nope")
        assert all(s["status"] == "pending" for s in checklist.steps)

    def test_check_negative_index_is_no_op(self, checklist):
        checklist.check(-1, True, "nope")
        assert all(s["status"] == "pending" for s in checklist.steps)

    def test_check_overwrites_previous_result(self, checklist):
        checklist.check(0, False, "first attempt failed")
        checklist.check(0, True, "retry succeeded")
        assert checklist.steps[0]["status"] == "pass"
        assert checklist.steps[0]["result"] == "retry succeeded"


# ---------------------------------------------------------------------------
# current_step
# ---------------------------------------------------------------------------

class TestCurrentStep:
    """Tests for finding the next pending/active step."""

    def test_returns_zero_initially(self, checklist):
        assert checklist.current_step() == 0

    def test_advances_past_passed_steps(self, checklist):
        checklist.check(0, True, "ok")
        assert checklist.current_step() == 1

    def test_stops_at_active_step(self, checklist):
        checklist.check(0, True, "ok")
        checklist.activate(1)
        assert checklist.current_step() == 1

    def test_returns_len_when_all_passed(self, checklist):
        for i in range(3):
            checklist.check(i, True, "ok")
        assert checklist.current_step() == 3

    def test_stops_at_failed_step_if_still_pending_after(self, checklist):
        checklist.check(0, True, "ok")
        checklist.check(1, False, "fail")
        # current_step finds first pending/active — step 2 is still pending
        assert checklist.current_step() == 2

    def test_empty_checklist_returns_zero(self):
        cl = TaskChecklist("t0", "empty", [])
        assert cl.current_step() == 0


# ---------------------------------------------------------------------------
# is_complete
# ---------------------------------------------------------------------------

class TestIsComplete:
    """Tests for completion detection."""

    def test_not_complete_initially(self, checklist):
        assert checklist.is_complete() is False

    def test_not_complete_with_partial_passes(self, checklist):
        checklist.check(0, True, "ok")
        checklist.check(1, True, "ok")
        assert checklist.is_complete() is False

    def test_complete_when_all_pass(self, checklist):
        for i in range(3):
            checklist.check(i, True, "ok")
        assert checklist.is_complete() is True

    def test_not_complete_with_failure(self, checklist):
        checklist.check(0, True, "ok")
        checklist.check(1, False, "fail")
        checklist.check(2, True, "ok")
        assert checklist.is_complete() is False

    def test_empty_checklist_is_complete(self):
        cl = TaskChecklist("t0", "empty", [])
        assert cl.is_complete() is True


# ---------------------------------------------------------------------------
# failed_step
# ---------------------------------------------------------------------------

class TestFailedStep:
    """Tests for finding the first failed step."""

    def test_none_when_no_failures(self, checklist):
        assert checklist.failed_step() is None

    def test_returns_first_failed_index(self, checklist):
        checklist.check(0, True, "ok")
        checklist.check(1, False, "fail")
        checklist.check(2, False, "also fail")
        assert checklist.failed_step() == 1

    def test_none_when_all_pass(self, checklist):
        for i in range(3):
            checklist.check(i, True, "ok")
        assert checklist.failed_step() is None

    def test_empty_checklist_returns_none(self):
        cl = TaskChecklist("t0", "empty", [])
        assert cl.failed_step() is None


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for human-readable summary generation."""

    def test_includes_goal(self, checklist):
        s = checklist.summary()
        assert "Deploy widget" in s

    def test_pending_icon(self, checklist):
        s = checklist.summary()
        assert "[ ]" in s

    def test_pass_icon(self, checklist):
        checklist.check(0, True, "ok")
        s = checklist.summary()
        assert "[x]" in s

    def test_fail_icon(self, checklist):
        checklist.check(0, False, "bad")
        s = checklist.summary()
        assert "[!]" in s

    def test_active_icon(self, checklist):
        checklist.activate(0)
        s = checklist.summary()
        assert "[~]" in s

    def test_result_appended_after_separator(self, checklist):
        checklist.check(0, True, "compiled in 2s")
        s = checklist.summary()
        assert "-- compiled in 2s" in s

    def test_no_separator_when_no_result(self, checklist):
        s = checklist.summary()
        assert "--" not in s

    def test_all_steps_appear_in_summary(self, checklist):
        s = checklist.summary()
        assert "Build" in s
        assert "Test" in s
        assert "Deploy" in s


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    """Tests for serialization."""

    def test_contains_required_keys(self, checklist):
        d = checklist.to_dict()
        assert d["task_id"] == "t1"
        assert d["goal"] == "Deploy widget"
        assert isinstance(d["steps"], list)
        assert "complete" in d

    def test_complete_reflects_state(self, checklist):
        assert checklist.to_dict()["complete"] is False
        for i in range(3):
            checklist.check(i, True, "ok")
        assert checklist.to_dict()["complete"] is True

    def test_steps_serialized_correctly(self, checklist):
        checklist.check(0, True, "built")
        checklist.activate(1)
        steps = checklist.to_dict()["steps"]
        assert steps[0]["status"] == "pass"
        assert steps[0]["result"] == "built"
        assert steps[1]["status"] == "active"
        assert steps[2]["status"] == "pending"


# ---------------------------------------------------------------------------
# verify_test_output (static)
# ---------------------------------------------------------------------------

class TestVerifyTestOutput:
    """Tests for test-runner output parsing."""

    def test_empty_output(self):
        passed, summary = TaskChecklist.verify_test_output("")
        assert passed is False
        assert summary == "No output"

    def test_none_output(self):
        passed, summary = TaskChecklist.verify_test_output(None)
        assert passed is False
        assert summary == "No output"

    def test_vitest_all_passed(self):
        output = "Tests  5 passed (5)\nDuration  1.23s"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is True
        assert "5" in summary

    def test_vitest_with_failures(self):
        output = "Tests  2 failed | 3 passed (5)\nDuration  2.00s"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False
        assert "2" in summary

    def test_jest_all_passed(self):
        output = "Test Suites: 1 passed, 1 total\nTests:       10 passed, 10 total"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is True

    def test_jest_with_failures(self):
        output = "Tests:       3 failed, 7 passed, 10 total"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    def test_pytest_all_passed(self):
        output = "===== 12 passed in 0.5s ====="
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is True
        assert "12" in summary

    def test_pytest_with_failures(self):
        output = "===== 1 failed, 11 passed in 1.2s ====="
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False
        assert "1" in summary

    def test_generic_pass_keyword(self):
        output = "All checks pass\nDone."
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is True
        assert summary == "Tests passed"

    def test_generic_fail_keyword(self):
        output = "FAIL: expected 1 got 2\nAssertionError"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    def test_generic_error_keyword(self):
        output = "Error: module not found"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    def test_generic_traceback_keyword(self):
        output = "Traceback (most recent call last):\n  File 'test.py', line 1"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    def test_unparseable_output(self):
        output = "lorem ipsum dolor sit amet"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False
        assert summary == "Could not parse test output"

    def test_pass_with_error_in_text_is_not_safe(self):
        # "pass" is present but so is "error" — should be unsafe
        output = "pass but also error happened"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    def test_exception_keyword_detected(self):
        output = "RuntimeException: something broke"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False


# ---------------------------------------------------------------------------
# verify_file_exists (static)
# ---------------------------------------------------------------------------

class TestVerifyFileExists:
    """Tests for file existence verification."""

    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        passed, summary = TaskChecklist.verify_file_exists(str(f))
        assert passed is True
        assert "5 bytes" in summary

    def test_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.txt")
        passed, summary = TaskChecklist.verify_file_exists(path)
        assert passed is False
        assert "not found" in summary.lower()

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        passed, summary = TaskChecklist.verify_file_exists(str(f))
        assert passed is True
        assert "0 bytes" in summary

    def test_directory_counts_as_exists(self, tmp_path):
        # os.path.exists returns True for directories too
        passed, summary = TaskChecklist.verify_file_exists(str(tmp_path))
        assert passed is True
