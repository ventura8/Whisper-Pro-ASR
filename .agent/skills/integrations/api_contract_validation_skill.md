# API Contract Validation Skill

Use this skill whenever endpoint behavior, request/response schemas, status codes, or parameter forwarding might change.

## Objective

Protect external API compatibility for:

- `/asr`
- `/detect-language`
- `/detectlang`
- `/v1/audio/transcriptions`
- `/v1/audio/translations`
- `/status`, `/analytics`, `/settings`, `/history`

## Contract Checklist

1. Verify media input modes: `audio_file` and `local_path`.
2. Verify standardization assumptions: 16kHz mono WAV pipeline for ASR and detect flow.
3. Verify `/asr` parameter forwarding (`initial_prompt`, `vad_filter`, `word_timestamps`, diarization fields).
4. Verify subtitle controls (`max_line_width`, `max_line_count`) affect SRT/VTT output.
5. Verify expected status codes & access controls:
   - `400` for malformed input, invalid/disallowed `ASR_MODEL`, invalid `ASR_DEVICE`, out-of-bounds retention settings, or media handling failures
   - `401` for missing/invalid API key when `API_KEY` or `ADMIN_API_KEY` is configured, including `GET /logs/download` when `ADMIN_API_KEY` protection is enabled
   - `403` for untrusted cross-origin requests to administrative endpoints when unauthenticated, including requests with neither Origin nor Referer
   - `GET /logs/download` must require trusted Origin or Referer validation against configured origins when no admin key is configured, and reject both missing headers and untrusted values with `403`
   - Default Compose host publish is `127.0.0.1:9000:9000`; all-interface `"9000:9000"` requires `API_KEY`. Startup warns when both `API_KEY` and `ADMIN_API_KEY` are unset.
   - `503` for unavailable inference engine
6. Verify CORS resolution: wildcard CORS disabled by default; origins come from `CORS_ORIGINS`, or from an explicit `CORS_ALLOW_ALL=true` opt-in. Wildcard CORS must not skip administrative CSRF origin checks.
7. Verify `/status` task ordering guarantees used by dashboard consumers.
8. Verify endpoint normalization policy:
   - `/asr` and `/v1/audio/...` map to the same standard ASR execution behavior.
   - `/detect-language` and `/detectlang` map to identical high-priority language-ID behavior.
9. Verify ASR observability log parity:
   - Transcription and translation paths both emit a pre-inference log with the selected `unit_id`.

## Validation Commands

```bash
./scripts/ci/build-and-test.sh

# Or target the route suite inside the Docker test image:
docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
docker run --rm -e CI=true -v "$PWD/assets:/app/assets" -v "$PWD/reports:/reports" \
  whisper-pro-asr-test /bin/bash -lc \
  "python3 -m pytest tests/integration/test_routes.py tests/integration/test_routes_helpers.py tests/integration/test_server.py tests/integration/test_system_routes.py"
```

## Done Criteria

- Route tests pass.
- No schema regressions in returned JSON payloads.
- No undocumented API behavior changes.
