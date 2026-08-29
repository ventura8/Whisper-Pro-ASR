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
   - Compose tmpfs mounts for `/tmp/whisper` and `/tmp/numba-cache` must declare
     `mode=1777`. A `docker compose restart` remounts tmpfs and does not retain
     the image-layer `chmod`; the explicit mode keeps upload materialization
     writable for the non-root runtime user.
4. Confirm environment flags align with desired engine/device behavior.
   - With `ASR_ENGINE=AUTO`, an explicit `ASR_DEVICE` constrains engine
     resolution. `ASR_DEVICE=CPU` must select `FASTER-WHISPER`, regardless of
     Intel devices discovered in the container.
   - The base Compose file deliberately selects the pinned Ubuntu base for CPU
     compatibility. Every Intel-bearing override (`intel`, `intel-xpu`,
     `nvidia-intel`, and `full`) must explicitly restore the pinned OpenVINO base;
     otherwise an Intel build loses its bundled Level Zero runtime and attempts to
     install unavailable Ubuntu packages.
5. Confirm compose build cache configuration remains enabled (`build.cache_from/cache_to` using `.buildx-cache`) and `.dockerignore` excludes volatile artifacts **including** `.docker-build-cache`, `.buildx-cache`, `.docker-build-cache.new`, and `.docker-build-cache.old` (local BuildKit cache dirs under the project root must never enter the build context).
6. Confirm build-time integrity checks:
   - Production (`Dockerfile`) and test (`Dockerfile.test`) images use the same signed upstream FFmpeg source release, currently pinned to **9.0.1**. Keep both `FFMPEG_VERSION` arguments synchronized whenever updating it.
   - ROCm apt repository key is checksum-validated before dearmor/install.
   - UVR model + Silero VAD ONNX downloads are SHA-256 verified before replacing target files.
   - Streaming HTTP responses are deterministically closed (preload scripts use `requests.get(..., stream=True)` inside a context manager).
   - Models are **not** baked into images. `modules/core/model_provisioning.py` downloads them at startup into `/app/model_cache`; `scripts/preload_model.py` remains only for optional pre-baking and delegates its download bodies to that module so the two cannot drift.
   - UVR download staging uses the preload cache directory (falling back to `PERSISTENT_TEMP_DIR`), never the removed `config.CACHE_DIR` setting.
   - Runtime `patchelf --clear-execstack` sweeps over Python `*.so*` trees (and `/app/libs/whisperx/`) track per-file failures and enforce a post-sweep check: the image build fails if any required object remains unpatched. Only an explicitly identified set of known-safe files (e.g. stub `.so` wrappers that carry no executable stack) may produce non-fatal failures; all other failures must abort the build.
   - No build-only artifacts may ship: the compiler toolchain, `__pycache__`, static `.a`
     archives and build temp files are removed by `scripts/docker/strip_build_artifacts.sh`
     **inside the same RUN that created them** (a later `rm` only whites files out; the bytes
     still ship). Assert with `scripts/docker/verify_no_build_artifacts.sh`.
     - Per-vendor image targets, selected by `BUILD_TARGET` and tagged `whisper-pro-asr:${BUILD_TARGET}`.
     Vendor apt blocks live in `scripts/docker/install_{cuda,rocm,intel}.sh` so composed
     targets share one implementation. ONNX Runtime variants are built once in the
     `ort-builder` stage and copied per target, so no image carries another vendor's runtime.
     `/app/libs/cpu` ships in **every** target -- it is the fallback `modules/core/bootstrap.py`
     resolves to when a vendor runtime is absent.
     - `cpu` (~3.5 GB): CPU only, no accelerator stack.
     - `intel` (~4.5 GB): Intel iGPU/Arc/NPU via OpenVINO.
     - `nvidia` (~7.7 GB): NVIDIA CUDA.
     - `nvidia-intel` (~9.4 GB): hybrid NVIDIA + Intel.
     - `amd` (~17.7 GB): AMD ROCm.
     - `intel-xpu` (~11.2 GB): `intel` plus the XPU PyTorch build, so OPENAI-WHISPER runs on
       the Intel GPU. **Requires Arc (Alchemist) or newer** -- on older iGPUs torch reports
       XPU as available and then fails to execute, so those fall back to CPU.
     - `nvidia-whisperx` (~18.4 GB): `nvidia` plus WhisperX, i.e. speaker diarization
       without `full`'s Intel and AMD stacks. Needs an NVIDIA GPU; degrades to CPU without one.
     - `amd-rocm-torch` (~21.8 GB): `amd` plus the ROCm PyTorch build, so OPENAI-WHISPER runs
       on the AMD GPU. CTranslate2 still has no ROCm backend, so FASTER-WHISPER stays on CPU.
     - `full` (`target: full`, default, ~29.8 GB uncompressed on disk, measured locally): all vendors + CPU-optimized WhisperX diarization stack.
       The isolated WhisperX runtime is installed during the image build as root, then the
       final image returns to the non-root `nobody` runtime user. Its installation layer
       runs the shared artifact stripper, as do the AMD/ROCm layers, so `__pycache__` and
       static archives are not shipped.
     - CUDA targets retain Torch's complete CUDA dependency set, including NCCL, cuSPARSELt,
       and NVSHMEM: UVR dynamically loads those libraries even for a single-GPU session.
   - `LD_LIBRARY_PATH` is composed per target in the image (`ENV ...:${LD_LIBRARY_PATH}`) and
     **must not** be restated in `docker-compose.yml`. The old compose override silently
     re-broke the loader path whenever the two drifted -- it is why CUDA paths went missing and
     every GPU transcription failed with `libcublas.so.12 is not found`.
   - Device passthrough differs per vendor and lives in `docker-compose.<target>.yml` overrides,
     merged onto the base file. A CPU-only host must never require nvidia-container-toolkit.

## Audit the Host Before Validating Hardware

Run the audit tool **before** selecting a build target or claiming any hardware
validation, and then run only the validations the host can support:

```bash
scripts/audit_hardware.sh          # Linux host  (--env writes BUILD_TARGET/HOST_INTEL_RENDER_GID to .env, --json for CI)
scripts/audit_hardware.ps1         # Windows / Docker Desktop (WSL2) host
```

It probes `nvidia-smi` plus a real Docker GPU probe (CUDA), `/dev/dri/renderD*` and their GID (Intel/AMD render
nodes), `/dev/accel` (Intel NPU), `/dev/kfd` (AMD ROCm), `lspci` vendor IDs
(8086/10de/1002), `intel_gpu_top` availability and free disk on `/`, then prints a
recommended `BUILD_TARGET` + `docker-compose.<target>.yml`.

Pick `BUILD_TARGET` and the matching `docker-compose.<target>.yml` from the result. A
target whose accelerator is absent may only be reported as "boots and falls back to CPU
cleanly" -- never as hardware-validated. `/dev/accel` is absent on iGPU/Arc-only hosts, so
Intel NPU paths cannot be validated there.

Then run the real-engine accuracy check from
`.agent/skills/quality/pipeline_skill.md` (Local Hardware Validation):

```bash
docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
docker run --rm --network host -e RUN_REAL_ASR=1 -e WHISPER_BASE_URL \
  -v "$(pwd):/app" -w /app -u "$(id -u):$(id -g)" -e HOME=/tmp \
  whisper-pro-asr-test python3 -m pytest tests/integration/test_transcription_accuracy.py
```

Run through the test image, not host pytest: quality gates are Docker-only (AGENTS.md), and
the real engine needs the per-vendor ONNX Runtime under `/app/libs`, which a host
interpreter does not have. `-u`/`HOME` keep the container from writing root-owned files
into the working tree.

A correct transcript proves decoding, not acceleration -- CPU fallback transcribes
correctly too. Pair it with `nvidia-smi --query-compute-apps` (CUDA) or `intel_gpu_top`
(Intel) evidence.

## Validation Commands

```bash
docker build -t whisper-pro-asr-test -f Dockerfile.test .
docker run --rm whisper-pro-asr-test
```

## Done Criteria

- Containerized test image passes.
- No device-mapping or cache-path regressions.
- Startup logs reflect expected hardware selection.
