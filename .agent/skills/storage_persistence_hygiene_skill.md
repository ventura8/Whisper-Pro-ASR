# Storage and Persistence Hygiene Skill

Use this skill when touching temp files, upload handling, history retention, or cleanup flows.

## Objective
Guarantee cleanup correctness and persistent-state integrity.

## Checklist
1. All transient files are registered and deleted in `finally` blocks.
2. Error paths clean partial outputs and descriptors.
3. Persistent artifacts are limited to intended locations (`model_cache`, state/history/logs).
4. Cleanup routines do not remove active-session artifacts.

## Validation Commands
```bash
.venv/bin/python -m pytest tests/test_utils_hygiene.py tests/monitoring/test_history_manager.py tests/integration/test_system_routes.py -q
```

## Done Criteria
- No storage leaks in happy/error paths.
- History and telemetry persistence remain stable.