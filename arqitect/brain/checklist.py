"""Generic task checklist for tracking progress on complex chains.

NOT TDD-specific. Can be used for any multi-step chain that needs
pass/fail checkpoints and real-time progress reporting.
"""

import json
import os
import re


class TaskChecklist:
    """Track progress on a complex task with pass/fail checkpoints."""

    def __init__(self, task_id: str, goal: str, steps: list[dict]):
        self.task_id = task_id
        self.goal = goal
        # Each step: {"name": str, "status": "pending"|"active"|"pass"|"fail",
        #             "result": str}
        self.steps = []
        for s in steps:
            self.steps.append({
                "name": s.get("name", ""),
                "status": "pending",
                "result": "",
            })

    def activate(self, step_index: int):
        """Mark step as in-progress."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = "active"

    def check(self, step_index: int, passed: bool, result: str):
        """Mark step as pass/fail with result summary."""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = "pass" if passed else "fail"
            self.steps[step_index]["result"] = result

    def current_step(self) -> int:
        """Index of next pending step."""
        for i, s in enumerate(self.steps):
            if s["status"] in ("pending", "active"):
                return i
        return len(self.steps)

    def is_complete(self) -> bool:
        """All steps passed."""
        return all(s["status"] == "pass" for s in self.steps)

    def failed_step(self) -> int | None:
        """First failed step index, or None."""
        for i, s in enumerate(self.steps):
            if s["status"] == "fail":
                return i
        return None

    def summary(self) -> str:
        """Human-readable checklist."""
        _STATUS_ICON = {"pass": "[x]", "fail": "[!]", "active": "[~]", "pending": "[ ]"}
        lines = [f"Task: {self.goal}"]
        for s in self.steps:
            icon = _STATUS_ICON.get(s["status"], "[ ]")
            line = f"  {icon} {s['name']}"
            if s["result"]:
                line += f" -- {s['result']}"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for Redis publish + cold memory."""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "steps": self.steps,
            "complete": self.is_complete(),
        }

    @staticmethod
    def verify_test_output(output: str) -> tuple[bool, str]:
        """Parse test runner output. Returns (all_passed, summary).

        Supports vitest, jest, pytest output patterns.
        """
        if not output:
            return False, "No output"

        out_lower = output.lower()

        # Vitest / Jest patterns
        # "Tests  1 passed" or "Test Suites: 1 passed"
        pass_match = re.search(r'(\d+)\s+passed', out_lower)
        fail_match = re.search(r'(\d+)\s+failed', out_lower)

        if pass_match or fail_match:
            passed = int(pass_match.group(1)) if pass_match else 0
            failed = int(fail_match.group(1)) if fail_match else 0
            if failed > 0:
                return False, f"{failed} test(s) failed, {passed} passed"
            if passed > 0:
                return True, f"{passed} test(s) passed"

        # Pytest patterns: "1 passed" or "1 failed"
        pytest_pass = re.search(r'(\d+)\s+passed', out_lower)
        pytest_fail = re.search(r'(\d+)\s+failed', out_lower)
        if pytest_pass and not pytest_fail:
            return True, f"{pytest_pass.group(1)} test(s) passed"
        if pytest_fail:
            failed = pytest_fail.group(1)
            passed = pytest_pass.group(1) if pytest_pass else "0"
            return False, f"{failed} test(s) failed, {passed} passed"

        # Generic PASS/FAIL indicators
        if "pass" in out_lower and "fail" not in out_lower and "error" not in out_lower:
            return True, "Tests passed"
        if any(kw in out_lower for kw in ("fail", "error", "exception", "traceback")):
            # Extract first error line
            for line in output.splitlines():
                line_s = line.strip()
                if any(kw in line_s.lower() for kw in ("fail", "error", "assert", "expect")):
                    return False, line_s[:200]
            return False, "Tests failed"

        return False, "Could not parse test output"

    @staticmethod
    def verify_file_exists(path: str) -> tuple[bool, str]:
        """Check file was written."""
        if os.path.exists(path):
            size = os.path.getsize(path)
            return True, f"File exists ({size} bytes)"
        return False, f"File not found: {path}"
