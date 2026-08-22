# Release v1.2.1 - Orchestration/Scheduling End-to-End Test Coverage, Two Live Bug Fixes, and Docs Accuracy Corrections

This release closes a full end-to-end test-coverage audit across the seven core orchestration, scheduling, and delivery capabilities: semaphore-backed top-level hardware orchestration with non-locking nested stages, hybrid accelerator assignment, priority preemption, FIFO ordering, the dashboard task timeline, Bazarr-compatible subtitle output, and telemetry. Along the way it fixes two real runtime bugs the new tests uncovered and corrects two places where product documentation overstated what the scheduler and locking layer actually do.

---

## Highlights

### Bug Fixes

- **ETA no longer counts down during a preemption pause.** `speed_status.js` was reusing a cached ETA/speed estimate whenever the processed position hadn't changed between polls, which made the remaining-time estimate shrink toward zero as if work continued even while a task sat paused under priority preemption. Estimate reuse is now bounded to a short poll-gap window; beyond it, a fresh calculation runs, correctly lowering the derived speed and growing the ETA while the task is stalled.
- **Crashed/killed tasks no longer show "active" forever on the dashboard.** There was no reaping mechanism for task-registry entries left behind by a worker that never reaches normal completion. `modules/monitoring/telemetry.py` now treats an "active" entry older than a bounded staleness window as a stale ghost and reports it as `failed` instead of perpetually `active`.

### Test Coverage — New Unit, Integration, and End-to-End Tests

- **Hardware locking**: proved `scheduler.STATE.model_lock` is a single global permit semaphore (not a reentrant/thread-local primitive), with per-unit assignment tracked separately via `STATE.hw_pool`, and that nested sub-stages avoid re-locking structurally via dedicated non-locking `_direct` entry points, not via lock re-entry. (`tests/inference/runtime/test_concurrency_reentrancy.py`)
- **Hybrid accelerator assignment**: GPU-side task throughput proven unaffected by continuous load on a separately-assigned unit, and a simulated mid-task device loss on one unit proven not to corrupt a concurrently running task's lock/registry state on another unit. (`tests/integration/concurrency/test_e2e_hybrid_split_isolation.py`)
- **Priority preemption**: a 1.5-second pause-confirmation SLA bound now asserted on the wall clock; all-units-saturated-with-priority-tasks now proven to queue rather than self-preempt; the natural-completion-vs-preemption race forced deterministically via a thread barrier across repeated runs; both preemption wait paths proven to self-heal rather than hang under a simulated worker crash. (`tests/inference/scheduler/priority/test_priority_preemption_sla.py`, `tests/inference/scheduler/priority/test_priority_preemption_self_heal.py`)
- **FIFO with priority yielding**: start_time tie-break proven non-flaky over 100 repeated runs; clock-skew resubmission proven to dispatch by start_time, not arrival order; a standard task proven not to starve behind 20 sequential priority arrivals; queued-task withdrawal proven not to corrupt remaining order. (`tests/inference/scheduler/priority/test_priority_fifo_ordering_regressions.py`, `tests/integration/concurrency/test_e2e_traffic_volume.py`)
- **Dashboard task timeline**: introduced a shared JSON ordering fixture consumed by both the Python (`_task_sort_key`) and JS (`_compareTaskOrder`, `_compareHistoryItems`) comparators, closing the risk of the two implementations silently diverging, and proving active→history transitions order by start_time+task_id rather than completion time. (`tests/fixtures/task_ordering_fixture.json`, `tests/monitoring/test_telemetry_loop.py`, `tests/js/dashboard_main.test.js`)
- **Bazarr output formats**: added a JSON response schema/contract pin, non-ASCII/CJK/RTL encoding round-trip tests, explicit out-of-order/overlapping timestamp behavior pinning, and a hand-written strict structural SRT/VTT validator standing in for third-party-parser round-trip verification. (`tests/unit/test_subtitles_bazarr_contract.py`, `tests/integration/test_bazarr_integration.py`)
- **Telemetry**: speed-multiplier and ETA math tested directly against fixtures (including mid-run partial computation and the pause case above); telemetry's reported hardware-unit state now cross-checked against the scheduler's actual internal unit assignment for the same run, closing the risk of the two subsystems drifting apart silently. (`tests/js/speed_status.test.js`, `tests/integration/concurrency/test_e2e_telemetry_ordering.py`)

### Documentation Accuracy

- Corrected "Re-entrant Hardware Orchestration ... thread-local locking" language in `README.md`, `docs/DOCKERHUB_DESCRIPTION.md`, `docs/ARCHITECTURE.md`, `docs/CONCURRENCY.md`, and `.agent/skills/runtime/concurrency_orchestration_skill.md` to accurately describe the real mechanism: a single global permit semaphore for top-level task dispatch (with per-unit assignment tracked separately via `STATE.hw_pool`), with nested sub-stages routed through dedicated non-locking `_direct` entry points instead of re-acquiring the lock.
- Corrected "Hybrid Split Architecture" language describing per-stage accelerator splitting *within* a single task to accurately describe what's actually implemented: per-task accelerator assignment, where the scheduler assigns each incoming task as a whole to whichever accelerator fits, and different tasks run concurrently and independently on different accelerators.

### Design Gaps Flagged for Follow-Up (Not Implemented This Release)

- No cancel/withdraw API exists anywhere in the scheduler for a queued-but-not-yet-dispatched task; the withdrawal test simulates it via direct registry removal.
- Neither preemption wait path has a named wall-clock timeout constant — both self-heal by re-checking task-registry state on a short poll interval instead. Documented as the intended mechanism rather than changed, given the correctness sensitivity of the concurrency code.

---

## Verification

Full Docker-parity CI gate (`scripts/ci/build-and-test.sh`) run to completion as part of release preparation, real process exit code confirmed via `PIPESTATUS` (not a `tee`-masked code):

- Backend: `pytest` (with coverage) — **1180 passed**, 0 failed. Coverage gate (`tests/check_coverage.py`, threshold 90%) — **96.04% total coverage**, passed.
- Frontend unit: `vitest run tests/js` — **105 passed**.
- Frontend E2E: Playwright — **25 passed**.
- Lint/format: Black, isort, Ruff (check + format), Flake8, Pylint — all clean, Pylint **10.00/10**.
- Complexity: Radon cyclomatic complexity, rank-A enforced repo-wide — zero violations.
- Security/audit: Bandit, `pip-audit`, `gitleaks`, `npm audit` (low threshold) — zero findings.
- Markdown/inline-ignore policy: `markdownlint-cli2`, `scripts/ci/check-inline-ignores.py` — zero violations.

Full scenario-by-scenario audit trail, coverage checklist, and priority rationale: `docs/E2E_TEST_PLAN_ORCHESTRATION.md`.
