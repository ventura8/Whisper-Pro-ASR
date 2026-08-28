# Storage and Persistence Hygiene Skill

Use this skill when touching temp files, upload handling, history retention, or cleanup flows.

## Objective

Guarantee cleanup correctness and persistent-state integrity.

## Checklist

1. All transient files are registered and deleted in `finally` blocks. For async routes that use AnyIO worker threads, initialize one request-scoped registry before dispatch so worker-created assets and route cleanup share ownership.
2. Error paths clean partial outputs and descriptors.
3. Persistent artifacts are limited to intended locations (`model_cache`, state/history/logs).
4. Cleanup routines do not remove active-session artifacts.
5. Upgrade compatibility is preserved: legacy history/analytics paths are imported into current state storage when present.

## Validation Commands

```bash
.venv/bin/python -m pytest tests/test_utils_hygiene.py tests/monitoring/test_history_manager.py tests/integration/test_system_routes.py -q
```

## Done Criteria

- No storage leaks in happy/error paths.
- History and telemetry persistence remain stable.
