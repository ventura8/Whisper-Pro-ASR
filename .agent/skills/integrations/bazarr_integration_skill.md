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
8. `encode=false` (Bazarr raw PCM) only bypasses FFmpeg normalization when the mapped audio already matches the uniform ingestion spec: **16kHz, mono (1ch), s16le/16-bit PCM**.
   - If the mapped audio is not in that format, normalize it (run the encode/standardization path) or reject the request so only compliant audio reaches inference.
9. Default Compose host publish is localhost-only (`127.0.0.1:9000:9000`). Same-network Bazarr uses `http://whisper-pro-asr:9000`. Bazarr on another host requires `"9000:9000"` plus `API_KEY`.

## Manual Validation

1. Submit one request via `local_path` and one via upload.
2. Verify outputs are downloadable and history entries are correct.
3. Confirm no path-resolution regressions in Dockerized environments.
4. Submit `encode=false` with a mapped audio file that is already **16kHz mono s16le/16-bit PCM** and confirm FFmpeg normalization is skipped.
5. Submit `encode=false` with a nonconforming mapped audio file and confirm the service normalizes it (or rejects) instead of bypassing FFmpeg.

## Done Criteria

- Both mapped-path and upload paths work.
- No regressions for common Bazarr provider settings.
- `encode=false` bypasses FFmpeg when input is already 16 kHz mono PCM.
