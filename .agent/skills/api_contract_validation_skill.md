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
5. Verify expected status codes:
   - `400` for malformed input/media handling failures
   - `503` for unavailable inference engine
6. Verify `/status` task ordering guarantees used by dashboard consumers.
7. Verify endpoint normalization policy:
   - `/asr` and `/v1/audio/...` map to the same standard ASR execution behavior.
   - `/detect-language` and `/detectlang` map to identical high-priority language-ID behavior.

## Validation Commands
```bash
.venv/bin/python -m pytest tests/integration/test_routes.py tests/integration/test_server.py tests/integration/test_system_routes.py
```

## Done Criteria
- Route tests pass.
- No schema regressions in returned JSON payloads.
- No undocumented API behavior changes.