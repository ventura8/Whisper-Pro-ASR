# Docker Runtime Operations Skill

Use this skill for container build/run/deploy changes, hardware passthrough, and runtime diagnostics.

## Objective

Ensure stable containerized operation across CPU, Intel, NVIDIA, and AMD hosts.

## Runtime Checklist

1. Confirm expected device mappings:
   - Intel Linux: `/dev/dri` and `/dev/accel`; when render/accel node ACLs require `group_add`, derive the host render/accel device GID (for example via `stat -c '%g' /dev/dri/renderD* /dev/accel/*`) and configure that value. `991` is environment-specific and only an example.
   - Intel Windows/WSL2: `/dev/dxg` plus `/dev/dri` and `/dev/accel` when WSL exposes them
   - NVIDIA: container toolkit + GPU reservation
   - AMD Linux (ROCm): `/dev/kfd` and `/dev/dri` device mapping
   - AMD Windows/WSL2 (DirectML): `/dev/dxg` detection-only (DirectML compute is not available inside the Linux container); UVR falls back to CPU
2. Confirm persistent volumes:
   - `model_cache` for model and compilation caches
   - `state`/`data` for history, telemetry, logs
3. Confirm temp-path configuration and fallback thresholds.
4. Confirm environment flags align with desired engine/device behavior.
5. Confirm compose build cache configuration remains enabled (`build.cache_from/cache_to` using `.buildx-cache`) and `.dockerignore` excludes volatile artifacts **including** `.docker-build-cache`, `.buildx-cache`, `.docker-build-cache.new`, and `.docker-build-cache.old` (local BuildKit cache dirs under the project root must never enter the build context).
6. Confirm build-time integrity checks:
   - Production (`Dockerfile`) and test (`Dockerfile.test`) images use the same signed upstream FFmpeg source release, currently pinned to **9.0.1**. Keep both `FFMPEG_VERSION` arguments synchronized whenever updating it.
   - ROCm apt repository key is checksum-validated before dearmor/install.
   - UVR model + Silero VAD ONNX downloads are SHA-256 verified before replacing target files.
   - Streaming HTTP responses are deterministically closed (preload scripts use `requests.get(..., stream=True)` inside a context manager).
   - Runtime `patchelf --clear-execstack` sweeps over Python `*.so*` trees (and `/app/libs/whisperx/`) track per-file failures and enforce a post-sweep check: the image build fails if any required object remains unpatched. Only an explicitly identified set of known-safe files (e.g. stub `.so` wrappers that carry no executable stack) may produce non-fatal failures; all other failures must abort the build.

## Validation Commands

```bash
docker build -t whisper-pro-asr-test -f Dockerfile.test .
docker run --rm whisper-pro-asr-test
```

## Done Criteria

- Containerized test image passes.
- No device-mapping or cache-path regressions.
- Startup logs reflect expected hardware selection.
