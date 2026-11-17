# Bazarr Integration Skill

Use this skill when changing API behavior, path handling, or deployment config that affects Bazarr clients.

## Objective

Keep Bazarr integration reliable for large media files and mapped-volume workflows.

## Checklist

1. Endpoint compatibility remains intact (`/asr`, OpenAI-compatible audio routes).
2. `local_path` / resolvable `video_file` behavior is preserved for zero-copy processing -- optional optimisation only.
3. Volume path mappings are documented and validated; **absence of media mounts must not break Bazarr**.
4. Long-running requests remain stable with high timeout expectations.
5. When `local_path` / `video_file` is readable under approved roots, uploads are not materialized to disk; when unreadable or unmounted, upload fallback is used. Both sides are pinned in `tests/integration/test_bazarr_upload_without_mounts_e2e.py` (mounts-absent upload survival + mounts-present mapped-path preference for ASR and detect-language) and `tests/integration/test_bazarr_wire_format.py`.
6. Some Bazarr clients send JSON bodies where the media path is the object key (for example `{"/tv/show.mkv": ""}`) instead of a `local_path` field. Route parsing and history backfill must recover those paths.
7. Failed registered tasks must persist dashboard history fields: normalized `request_json.local_path`, `result.error` / `response_json`, and at least one execution log line via `record_task_failure()`.
8. `encode=false` (Bazarr raw PCM) applies raw-PCM input flags only to an uploaded audio stream. A readable mapped `local_path` or `video_file` takes precedence and clears those flags because it represents an original media container, which FFmpeg must auto-detect.
9. `TEMP_DIR` / tmpfs must stay writable by the service user (`tmpfs` `mode=1777` in compose). Startup `_temp_dir()` falls back to a whisper-owned `whisper-runtime/temp` directory (never bare `/tmp`); runtime `resolve_temp_dir` may further select `PERSISTENT_TEMP_DIR` so materialization cannot collapse into `No audio source provided`.

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
