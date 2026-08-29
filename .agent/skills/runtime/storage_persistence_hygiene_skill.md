# Storage and Persistence Hygiene Skill

Use this skill when touching temp files, upload handling, history retention, or cleanup flows.

## Objective

Guarantee cleanup correctness and persistent-state integrity.

## Checklist

1. All transient files are registered and deleted in `finally` blocks. For async routes that use AnyIO worker threads, initialize one request-scoped registry before dispatch so worker-created assets and route cleanup share ownership.
2. Error paths clean partial outputs and descriptors.
3. Persistent artifacts are limited to intended locations (`model_cache`, state/history/logs).
4. Temp-directory fallback selection validates the same free-space threshold for tmpfs and persistent storage. When **neither** meets it, `resolve_temp_dir` logs a warning and returns the persistent directory anyway -- it does not raise. That is deliberate: the function only *selects* a path and runs while resolving config on the request path, so raising turned a capacity warning into a hard failure for every request. Callers that need space to be guaranteed must check it themselves.
5. Cleanup routines do not remove active-session artifacts.
6. Upgrade compatibility is preserved: legacy history/analytics paths are imported into current state storage when present.

## Validation Commands

```bash
.venv/bin/python -m pytest tests/test_utils_hygiene.py tests/monitoring/test_history_manager.py tests/integration/test_system_routes.py -q
```

## Done Criteria

- No storage leaks in happy/error paths.
- History and telemetry persistence remain stable.
