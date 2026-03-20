"""Tests for arqitect.brain.checklist — generic task checklist with pass/fail checkpoints."""

from __future__ import annotations

import os

import pytest
from dirty_equals import IsInstance, IsPartialDict
from hypothesis import given, settings
from hypothesis import strategies as st

from arqitect.brain.checklist import TaskChecklist


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

step_name_strategy = st.text(min_size=1, max_size=50)
step_list_strategy = st.lists(
    st.fixed_dictionaries({"name": step_name_strategy}),
    min_size=0,
    max_size=20,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_steps(*names: str) -> list[dict]:
    """Build a step list from step names."""
    return [{"name": n} for n in names]


@pytest.fixture()
def checklist() -> TaskChecklist:
    """A three-step checklist for general testing."""
    return TaskChecklist(
        task_id="t1",
        goal="Deploy widget",
        steps=_make_steps("Build", "Test", "Deploy"),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestConstruction:
    """Verify initial state after creating a TaskChecklist."""

    def test_stores_task_id_and_goal(self, checklist: TaskChecklist) -> None:
        assert checklist.task_id == "t1"
        assert checklist.goal == "Deploy widget"

    def test_initializes_steps_as_pending(self, checklist: TaskChecklist) -> None:
        for step in checklist.steps:
            assert step == IsPartialDict(status="pending", result="")

    def test_preserves_step_names(self, checklist: TaskChecklist) -> None:
        names = [s["name"] for s in checklist.steps]
        assert names == ["Build", "Test", "Deploy"]

    def test_empty_steps_list(self) -> None:
        cl = TaskChecklist("t0", "nothing", [])
        assert cl.steps == []

    def test_step_missing_name_key_defaults_to_empty(self) -> None:
        cl = TaskChecklist("t0", "g", [{"description": "no name key"}])
        assert cl.steps[0]["name"] == ""

    def test_single_step(self) -> None:
        cl = TaskChecklist("t0", "one step", _make_steps("only"))
        assert len(cl.steps) == 1
        assert cl.steps[0]["name"] == "only"

    @given(
        task_id=st.text(min_size=1, max_size=20),
        goal=st.text(min_size=1, max_size=100),
        steps=step_list_strategy,
    )
    @settings(max_examples=30)
    def test_arbitrary_construction_always_starts_pending(
        self, task_id: str, goal: str, steps: list[dict]
    ) -> None:
        """Every step starts pending regardless of inputs."""
        cl = TaskChecklist(task_id, goal, steps)
        assert cl.task_id == task_id
        assert cl.goal == goal
        assert len(cl.steps) == len(steps)
        for step in cl.steps:
            assert step["status"] == "pending"
            assert step["result"] == ""


# ---------------------------------------------------------------------------
# activate
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestActivate:
    """Tests for marking a step as in-progress."""

    def test_activate_sets_status_to_active(self, checklist: TaskChecklist) -> None:
        checklist.activate(0)
        assert checklist.steps[0]["status"] == "active"

    def test_activate_does_not_affect_other_steps(self, checklist: TaskChecklist) -> None:
        checklist.activate(1)
        assert checklist.steps[0]["status"] == "pending"
        assert checklist.steps[2]["status"] == "pending"

    def test_activate_out_of_range_positive_is_no_op(self, checklist: TaskChecklist) -> None:
        checklist.activate(99)
        assert all(s["status"] == "pending" for s in checklist.steps)

    def test_activate_negative_index_is_no_op(self, checklist: TaskChecklist) -> None:
        checklist.activate(-1)
        assert all(s["status"] == "pending" for s in checklist.steps)

    @given(idx=st.integers(min_value=3, max_value=1000))
    @settings(max_examples=20)
    def test_activate_any_out_of_range_is_no_op(self, idx: int) -> None:
        """Out-of-range indices never mutate state."""
        cl = TaskChecklist("t", "g", _make_steps("A", "B", "C"))
        cl.activate(idx)
        assert all(s["status"] == "pending" for s in cl.steps)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestCheck:
    """Tests for recording step pass/fail results."""

    def test_check_pass(self, checklist: TaskChecklist) -> None:
        checklist.check(0, True, "built ok")
        assert checklist.steps[0] == IsPartialDict(status="pass", result="built ok")

    def test_check_fail(self, checklist: TaskChecklist) -> None:
        checklist.check(1, False, "tests failed")
        assert checklist.steps[1] == IsPartialDict(status="fail", result="tests failed")

    def test_check_out_of_range_is_no_op(self, checklist: TaskChecklist) -> None:
        checklist.check(100, True, "nope")
        assert all(s["status"] == "pending" for s in checklist.steps)

    def test_check_negative_index_is_no_op(self, checklist: TaskChecklist) -> None:
        checklist.check(-1, True, "nope")
        assert all(s["status"] == "pending" for s in checklist.steps)

    def test_check_overwrites_previous_result(self, checklist: TaskChecklist) -> None:
        checklist.check(0, False, "first attempt failed")
        checklist.check(0, True, "retry succeeded")
        assert checklist.steps[0] == IsPartialDict(status="pass", result="retry succeeded")

    @given(
        idx=st.integers(min_value=0, max_value=2),
        passed=st.booleans(),
        result=st.text(max_size=100),
    )
    @settings(max_examples=30)
    def test_check_always_records_status_and_result(
        self, idx: int, passed: bool, result: str
    ) -> None:
        """Any valid index records exactly the given status and result."""
        cl = TaskChecklist("t", "g", _make_steps("A", "B", "C"))
        cl.check(idx, passed, result)
        expected_status = "pass" if passed else "fail"
        assert cl.steps[idx]["status"] == expected_status
        assert cl.steps[idx]["result"] == result


# ---------------------------------------------------------------------------
# current_step
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestCurrentStep:
    """Tests for finding the next pending/active step."""

    def test_returns_zero_initially(self, checklist: TaskChecklist) -> None:
        assert checklist.current_step() == 0

    def test_advances_past_passed_steps(self, checklist: TaskChecklist) -> None:
        checklist.check(0, True, "ok")
        assert checklist.current_step() == 1

    def test_stops_at_active_step(self, checklist: TaskChecklist) -> None:
        checklist.check(0, True, "ok")
        checklist.activate(1)
        assert checklist.current_step() == 1

    def test_returns_len_when_all_passed(self, checklist: TaskChecklist) -> None:
        for i in range(3):
            checklist.check(i, True, "ok")
        assert checklist.current_step() == 3

    def test_stops_at_failed_step_if_still_pending_after(self, checklist: TaskChecklist) -> None:
        checklist.check(0, True, "ok")
        checklist.check(1, False, "fail")
        # current_step finds first pending/active — step 2 is still pending
        assert checklist.current_step() == 2

    def test_empty_checklist_returns_zero(self) -> None:
        cl = TaskChecklist("t0", "empty", [])
        assert cl.current_step() == 0

    @given(n=st.integers(min_value=1, max_value=15))
    @settings(max_examples=20)
    def test_current_step_after_passing_all_equals_length(self, n: int) -> None:
        """Passing every step pushes current_step to len(steps)."""
        names = [f"step_{i}" for i in range(n)]
        cl = TaskChecklist("t", "g", _make_steps(*names))
        for i in range(n):
            cl.check(i, True, "ok")
        assert cl.current_step() == n


# ---------------------------------------------------------------------------
# is_complete
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestIsComplete:
    """Tests for completion detection."""

    def test_not_complete_initially(self, checklist: TaskChecklist) -> None:
        assert checklist.is_complete() is False

    def test_not_complete_with_partial_passes(self, checklist: TaskChecklist) -> None:
        checklist.check(0, True, "ok")
        checklist.check(1, True, "ok")
        assert checklist.is_complete() is False

    def test_complete_when_all_pass(self, checklist: TaskChecklist) -> None:
        for i in range(3):
            checklist.check(i, True, "ok")
        assert checklist.is_complete() is True

    def test_not_complete_with_failure(self, checklist: TaskChecklist) -> None:
        checklist.check(0, True, "ok")
        checklist.check(1, False, "fail")
        checklist.check(2, True, "ok")
        assert checklist.is_complete() is False

    def test_empty_checklist_is_complete(self) -> None:
        cl = TaskChecklist("t0", "empty", [])
        assert cl.is_complete() is True

    @given(n=st.integers(min_value=1, max_value=15))
    @settings(max_examples=20)
    def test_single_failure_prevents_completion(self, n: int) -> None:
        """Any single failure in an otherwise-passing checklist means incomplete."""
        names = [f"step_{i}" for i in range(n)]
        cl = TaskChecklist("t", "g", _make_steps(*names))
        for i in range(n):
            cl.check(i, True, "ok")
        # Fail the last step
        cl.check(n - 1, False, "broke")
        assert cl.is_complete() is False


# ---------------------------------------------------------------------------
# failed_step
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestFailedStep:
    """Tests for finding the first failed step."""

    def test_none_when_no_failures(self, checklist: TaskChecklist) -> None:
        assert checklist.failed_step() is None

    def test_returns_first_failed_index(self, checklist: TaskChecklist) -> None:
        checklist.check(0, True, "ok")
        checklist.check(1, False, "fail")
        checklist.check(2, False, "also fail")
        assert checklist.failed_step() == 1

    def test_none_when_all_pass(self, checklist: TaskChecklist) -> None:
        for i in range(3):
            checklist.check(i, True, "ok")
        assert checklist.failed_step() is None

    def test_empty_checklist_returns_none(self) -> None:
        cl = TaskChecklist("t0", "empty", [])
        assert cl.failed_step() is None


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestSummary:
    """Tests for human-readable summary generation."""

    def test_includes_goal(self, checklist: TaskChecklist) -> None:
        s = checklist.summary()
        assert "Deploy widget" in s

    def test_pending_icon(self, checklist: TaskChecklist) -> None:
        s = checklist.summary()
        assert "[ ]" in s

    def test_pass_icon(self, checklist: TaskChecklist) -> None:
        checklist.check(0, True, "ok")
        s = checklist.summary()
        assert "[x]" in s

    def test_fail_icon(self, checklist: TaskChecklist) -> None:
        checklist.check(0, False, "bad")
        s = checklist.summary()
        assert "[!]" in s

    def test_active_icon(self, checklist: TaskChecklist) -> None:
        checklist.activate(0)
        s = checklist.summary()
        assert "[~]" in s

    def test_result_appended_after_separator(self, checklist: TaskChecklist) -> None:
        checklist.check(0, True, "compiled in 2s")
        s = checklist.summary()
        assert "-- compiled in 2s" in s

    def test_no_separator_when_no_result(self, checklist: TaskChecklist) -> None:
        s = checklist.summary()
        assert "--" not in s

    def test_all_steps_appear_in_summary(self, checklist: TaskChecklist) -> None:
        s = checklist.summary()
        assert "Build" in s
        assert "Test" in s
        assert "Deploy" in s

    @given(
        names=st.lists(st.text(min_size=1, max_size=30, alphabet=st.characters(categories=("L", "N"))), min_size=1, max_size=10),
    )
    @settings(max_examples=20)
    def test_summary_contains_every_step_name(self, names: list[str]) -> None:
        """Every step name appears somewhere in the summary output."""
        cl = TaskChecklist("t", "goal", _make_steps(*names))
        s = cl.summary()
        for name in names:
            assert name in s


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestToDict:
    """Tests for serialization."""

    def test_contains_required_keys(self, checklist: TaskChecklist) -> None:
        d = checklist.to_dict()
        assert d["task_id"] == "t1"
        assert d["goal"] == "Deploy widget"
        assert d["steps"] == IsInstance(list)
        assert "complete" in d

    def test_complete_reflects_state(self, checklist: TaskChecklist) -> None:
        assert checklist.to_dict()["complete"] is False
        for i in range(3):
            checklist.check(i, True, "ok")
        assert checklist.to_dict()["complete"] is True

    def test_steps_serialized_correctly(self, checklist: TaskChecklist) -> None:
        checklist.check(0, True, "built")
        checklist.activate(1)
        steps = checklist.to_dict()["steps"]
        assert steps[0] == IsPartialDict(status="pass", result="built")
        assert steps[1] == IsPartialDict(status="active")
        assert steps[2] == IsPartialDict(status="pending")

    @given(steps=step_list_strategy)
    @settings(max_examples=20)
    def test_to_dict_step_count_matches(self, steps: list[dict]) -> None:
        """Serialized step count always matches input step count."""
        cl = TaskChecklist("t", "g", steps)
        d = cl.to_dict()
        assert len(d["steps"]) == len(steps)
        assert d["task_id"] == "t"
        assert d["goal"] == "g"
        assert "complete" in d


# ---------------------------------------------------------------------------
# verify_test_output (static)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestVerifyTestOutput:
    """Tests for test-runner output parsing."""

    def test_empty_output(self) -> None:
        passed, summary = TaskChecklist.verify_test_output("")
        assert passed is False
        assert summary == "No output"

    def test_none_output(self) -> None:
        passed, summary = TaskChecklist.verify_test_output(None)
        assert passed is False
        assert summary == "No output"

    def test_vitest_all_passed(self) -> None:
        output = "Tests  5 passed (5)\nDuration  1.23s"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is True
        assert "5" in summary

    def test_vitest_with_failures(self) -> None:
        output = "Tests  2 failed | 3 passed (5)\nDuration  2.00s"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False
        assert "2" in summary

    def test_jest_all_passed(self) -> None:
        output = "Test Suites: 1 passed, 1 total\nTests:       10 passed, 10 total"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is True

    def test_jest_with_failures(self) -> None:
        output = "Tests:       3 failed, 7 passed, 10 total"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    def test_pytest_all_passed(self) -> None:
        output = "===== 12 passed in 0.5s ====="
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is True
        assert "12" in summary

    def test_pytest_with_failures(self) -> None:
        output = "===== 1 failed, 11 passed in 1.2s ====="
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False
        assert "1" in summary

    def test_generic_pass_keyword(self) -> None:
        output = "All checks pass\nDone."
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is True
        assert summary == "Tests passed"

    def test_generic_fail_keyword(self) -> None:
        output = "FAIL: expected 1 got 2\nAssertionError"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    def test_generic_error_keyword(self) -> None:
        output = "Error: module not found"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    def test_generic_traceback_keyword(self) -> None:
        output = "Traceback (most recent call last):\n  File 'test.py', line 1"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    def test_unparseable_output(self) -> None:
        output = "lorem ipsum dolor sit amet"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False
        assert summary == "Could not parse test output"

    def test_pass_with_error_in_text_is_not_safe(self) -> None:
        # "pass" is present but so is "error" — should be unsafe
        output = "pass but also error happened"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    def test_exception_keyword_detected(self) -> None:
        output = "RuntimeException: something broke"
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False

    @given(n_passed=st.integers(min_value=1, max_value=500))
    @settings(max_examples=20)
    def test_pytest_style_pass_always_detected(self, n_passed: int) -> None:
        """Any 'N passed' pytest output is recognized as passing."""
        output = f"===== {n_passed} passed in 0.1s ====="
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is True
        assert str(n_passed) in summary

    @given(
        n_failed=st.integers(min_value=1, max_value=500),
        n_passed=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=20)
    def test_pytest_style_failure_always_detected(self, n_failed: int, n_passed: int) -> None:
        """Any 'N failed' pytest output is recognized as failing."""
        output = f"===== {n_failed} failed, {n_passed} passed in 0.1s ====="
        passed, summary = TaskChecklist.verify_test_output(output)
        assert passed is False
        assert str(n_failed) in summary

    def test_return_type_is_always_tuple(self) -> None:
        """verify_test_output always returns (bool, str)."""
        result = TaskChecklist.verify_test_output("anything")
        assert result == IsInstance(tuple)
        assert result[0] == IsInstance(bool)
        assert result[1] == IsInstance(str)


# ---------------------------------------------------------------------------
# verify_file_exists (static)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestVerifyFileExists:
    """Tests for file existence verification."""

    def test_existing_file(self, tmp_path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        passed, summary = TaskChecklist.verify_file_exists(str(f))
        assert passed is True
        assert "5 bytes" in summary

    def test_missing_file(self, tmp_path) -> None:
        path = str(tmp_path / "nonexistent.txt")
        passed, summary = TaskChecklist.verify_file_exists(path)
        assert passed is False
        assert "not found" in summary.lower()

    def test_empty_file(self, tmp_path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        passed, summary = TaskChecklist.verify_file_exists(str(f))
        assert passed is True
        assert "0 bytes" in summary

    def test_directory_counts_as_exists(self, tmp_path) -> None:
        # os.path.exists returns True for directories too
        passed, summary = TaskChecklist.verify_file_exists(str(tmp_path))
        assert passed is True

    def test_return_type_contract(self, tmp_path) -> None:
        """verify_file_exists always returns (bool, str)."""
        result = TaskChecklist.verify_file_exists(str(tmp_path / "nope"))
        assert result == IsInstance(tuple)
        assert result[0] == IsInstance(bool)
        assert result[1] == IsInstance(str)
