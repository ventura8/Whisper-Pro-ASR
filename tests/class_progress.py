"""Class-grouped pytest progress: one finished line per TestClass.

Print ``module.py::TestClass ....`` only when that class/group has received
every expected result — not when an xdist worker drains, and not as a session
end dump of unfinished groups.

Skipped/failed setup outcomes are counted so expected totals still complete.
Progress state and printing run only on the controller (xdist workers skip).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

_EXPECTED: dict[str, int] = {}
_RESULTS: dict[str, str] = {}
_PRINTED: set[str] = set()
_SEEN: set[str] = set()
_RUNTIME: dict[str, Any] = {"is_controller": True, "terminal_writer": None}


def _group_for(nodeid: str) -> str:
    """Prefer module.py::ClassName; fall back to module basename for free tests."""
    parts = nodeid.split("::")
    if len(parts) >= 3:
        module = parts[0].rsplit("/", 1)[-1]
        return f"{module}::{parts[1]}"
    if len(parts) == 2:
        return parts[0].rsplit("/", 1)[-1]
    return nodeid


def _result_char(outcome: str) -> str:
    return {"passed": ".", "skipped": "s", "failed": "F", "error": "E"}.get(outcome, "?")


def _flush_group(group: str) -> None:
    if group in _PRINTED:
        return
    dots = _RESULTS.get(group, "")
    expected = _EXPECTED.get(group, 0)
    if expected and len(dots) >= expected:
        writer = _RUNTIME.get("terminal_writer")
        line = f"{group} {dots}\n"
        if writer is not None:
            writer.write(line)
            writer.flush()
        _PRINTED.add(group)


def _should_record(report: pytest.TestReport) -> bool:
    """One progress char per test: call result, or setup skip/fail (no call)."""
    if report.nodeid in _SEEN:
        return False
    if report.when == "call":
        return True
    return report.when == "setup" and report.outcome in {"skipped", "failed", "error"}


def _suppress_default_char(report: pytest.TestReport) -> bool:
    if report.when == "call":
        return True
    return report.when == "setup" and report.outcome != "passed"


def pytest_configure(config: pytest.Config) -> None:
    """Only the controller owns progress state; workers must not print."""
    _RUNTIME["is_controller"] = not hasattr(config, "workerinput")
    # get_terminal_writer() asserts when terminalreporter is absent (xdist workers).
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    _RUNTIME["terminal_writer"] = config.get_terminal_writer() if reporter is not None else None


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Record how many tests belong to each class/module group (controller only)."""
    del session
    if hasattr(config, "workerinput"):
        return
    _EXPECTED.clear()
    _RESULTS.clear()
    _PRINTED.clear()
    _SEEN.clear()
    _EXPECTED.update(Counter(_group_for(item.nodeid) for item in items))


@pytest.hookimpl(tryfirst=True)
def pytest_report_teststatus(report: pytest.TestReport, config: pytest.Config) -> tuple[str, str, str] | None:
    """Suppress default progress chars; print a class line when it completes."""
    if not _RUNTIME["is_controller"] or config.getoption("verbose") > 0:
        return None
    if _should_record(report):
        _SEEN.add(report.nodeid)
        group = _group_for(report.nodeid)
        _RESULTS[group] = _RESULTS.get(group, "") + _result_char(report.outcome)
        _flush_group(group)
    if _suppress_default_char(report):
        return report.outcome, "", report.outcome.upper()
    return None


def pytest_xdist_node_collection_finished(node: object, ids: list[str]) -> None:
    """Populate _EXPECTED from worker-collected node IDs when running under xdist.

    pytest-xdist calls this on the controller for each worker once that worker
    finishes collection. The hook is optional: if pytest-xdist is absent, it is
    simply never registered. Workers always skip progress-state updates (see
    pytest_configure), so this only runs on the controller.
    """
    if not _RUNTIME["is_controller"]:
        return
    del node
    _EXPECTED.update(Counter(_group_for(nodeid) for nodeid in ids))
