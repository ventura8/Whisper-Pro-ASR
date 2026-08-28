"""Bounded thread-join helpers shared by preemption concurrency tests."""

from __future__ import annotations

import threading


def thread_was_started(thread: threading.Thread | None) -> bool:
    """True only after Thread.start(); join() raises RuntimeError otherwise."""
    return thread is not None and thread.ident is not None


def join_if_started(thread: threading.Thread | None, timeout: float) -> None:
    """Join a thread only when it was actually started."""
    if thread_was_started(thread):
        thread.join(timeout=timeout)


def join_and_assert_terminated(thread: threading.Thread | None, timeout: float, label: str) -> None:
    """Join a started thread and fail if it outlives cleanup."""
    join_if_started(thread, timeout)
    if thread_was_started(thread):
        assert not thread.is_alive(), f"{label} did not terminate before cleanup"


def join_straggler_if_alive(thread: threading.Thread | None, timeout: float) -> None:
    """Bounded follow-up join for a still-running started thread."""
    if thread_was_started(thread) and thread.is_alive():
        thread.join(timeout=timeout)


def join_scenario_threads(
    t_std: threading.Thread | None,
    t_prio: threading.Thread | None,
    primary_timeout: float = 8.0,
    straggler_timeout: float = 30.0,
) -> None:
    """Join scenario threads, with a bounded follow-up wait for stragglers."""
    # Callers invoke this from finally even if start() failed -- skip unstarted
    # threads so cleanup does not mask the original failure with RuntimeError.
    join_if_started(t_std, primary_timeout)
    join_if_started(t_prio, primary_timeout)
    # Belt-and-suspenders: if either thread is still running past the primary
    # timeout, keep waiting (bounded, so a true deadlock still eventually surfaces
    # as a test failure/CI timeout rather than hanging forever) instead of letting
    # the caller's `with` block exit and restore patched globals while a straggler
    # thread is still executing against them.
    join_straggler_if_alive(t_std, straggler_timeout)
    join_straggler_if_alive(t_prio, straggler_timeout)
    if thread_was_started(t_std):
        assert not t_std.is_alive(), "t_std did not terminate before patched globals are restored"
    if thread_was_started(t_prio):
        assert not t_prio.is_alive(), "t_prio did not terminate before patched globals are restored"
