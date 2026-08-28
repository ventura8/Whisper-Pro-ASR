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
6. Some Bazarr clients send JSON bodies where the media path is the object key (for example `{"/tv/show.mkv": ""}`) instead of a `local_path` field. Route parsing and history backfill must recover those paths.
7. Failed registered tasks must persist dashboard history fields: normalized `request_json.local_path`, `result.error` / `response_json`, and at least one execution log line via `record_task_failure()`.
8. `encode=false` (Bazarr raw PCM) applies raw-PCM input flags only to an uploaded audio stream. A readable mapped `local_path` or `video_file` takes precedence and clears those flags because it represents an original media container, which FFmpeg must auto-detect.

## Manual Validation

1. Submit one request via `local_path` and one via upload.
2. Verify outputs are downloadable and history entries are correct.
3. Confirm no path-resolution regressions in Dockerized environments.
4. Submit `encode=false` with an uploaded 16kHz mono s16le PCM audio stream and confirm FFmpeg receives raw-PCM input flags.
5. Submit the same request with a readable mapped media path and confirm the mapped path wins and FFmpeg auto-detects its container format.

## Done Criteria

- Both mapped-path and upload paths work.
- No regressions for common Bazarr provider settings.
- Uploaded raw PCM and readable mapped media both use the correct FFmpeg input interpretation.
