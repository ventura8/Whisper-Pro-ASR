# Troubleshooting Playbook Skill

Use this skill for production-like bug triage: deadlocks, stuck queues, bad ordering, or performance regressions.

## Objective
Reproduce quickly, isolate root cause, and convert the fix into deterministic regression tests.

## Workflow
1. Capture exact request sequence (arrival order, endpoints, hardware layout).
2. Reproduce with focused test or add one if absent.
3. Inspect scheduler and concurrency state transitions (`queued`, `active`, and `unit_sync[unit_id]` pause/resume generation events).
4. Patch minimal logic in scheduler/concurrency layers.
5. Add/strengthen regression tests for the exact scenario.
6. Run focused tests, then full suite.

## Recommended Commands
```bash
.venv/bin/python -m pytest -o addopts='' tests/inference/priority/test_priority_concurrency.py -q
.venv/bin/python -m pytest tests/
```

## Done Criteria
- Repro test fails before fix and passes after fix.
- Full suite remains green.