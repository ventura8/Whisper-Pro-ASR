# Bazarr Integration Skill

Use this skill when changing API behavior, path handling, or deployment config that affects Bazarr clients.

## Objective
Keep Bazarr integration reliable for large media files and mapped-volume workflows.

## Checklist
1. Endpoint compatibility remains intact (`/asr`, OpenAI-compatible audio routes).
2. `local_path` behavior is preserved for zero-copy processing.
3. Volume path mappings are documented and validated.
4. Long-running requests remain stable with high timeout expectations.
5. When `local_path` is readable, uploads are not materialized to disk; when unreadable, upload fallback is used.

## Manual Validation
1. Submit one request via `local_path` and one via upload.
2. Verify outputs are downloadable and history entries are correct.
3. Confirm no path-resolution regressions in Dockerized environments.

## Done Criteria
- Both mapped-path and upload paths work.
- No regressions for common Bazarr provider settings.