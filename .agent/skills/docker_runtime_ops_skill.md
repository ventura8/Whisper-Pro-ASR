# Docker Runtime Operations Skill

Use this skill for container build/run/deploy changes, hardware passthrough, and runtime diagnostics.

## Objective
Ensure stable containerized operation across CPU, Intel, and NVIDIA hosts.

## Runtime Checklist
1. Confirm expected device mappings:
   - Intel: `/dev/dri`, `/dev/accel/accel0` (and `/dev/dxg` for WSL)
   - NVIDIA: container toolkit + GPU reservation
2. Confirm persistent volumes:
   - `model_cache` for model and compilation caches
   - `state`/`data` for history, telemetry, logs
3. Confirm temp-path configuration and fallback thresholds.
4. Confirm environment flags align with desired engine/device behavior.

## Validation Commands
```bash
docker build -t whisper-pro-asr-test -f Dockerfile.test .
docker run --rm whisper-pro-asr-test
```

## Done Criteria
- Containerized test image passes.
- No device-mapping or cache-path regressions.
- Startup logs reflect expected hardware selection.