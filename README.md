# Whisper Pro ASR

![Main Language](https://img.shields.io/github/languages/top/ventura8/Whisper-Pro-ASR)
![Coverage](assets/coverage.svg)
![Pylint](https://img.shields.io/badge/Pylint-10.00%2F10-brightgreen)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Whisper Pro ASR** is a high-performance transcription microservice with **speaker diarization**, optimized for the **Whisper Large V3** model. It delivers enterprise-grade performance with native hardware acceleration for **Intel Core Ultra NPUs**, **Integrated GPUs**, **NVIDIA CUDA**, and **native Linux AMD ROCm** environments.

Engineered for seamless integration with **Bazarr** and the broader media automation stack, it offloads computationally intensive AI tasks from your primary system resources, providing industrial-strength transcription with speaker identification and rapid hardware context switching.

## Concurrency-First Priority

Concurrency correctness is the top project priority.

- Any change that can affect scheduling, locks, queues, events, or model lifecycle must preserve deadlock and livelock safety before feature throughput optimizations.
- Priority/preemption synchronization waits are intentionally unbounded (waiting indefinitely with periodic logging every 30 seconds to survive heavy load); requests must wait until hardware and preemption handoff are available instead of failing on scheduler timeouts.
- Concurrency-affecting changes require matching liveness regression tests and documentation updates in this repository.
- Guarantee model: practical high-confidence liveness with explicit assumptions and CI stress evidence, not absolute universal proof across all OS and third-party internals.

---

## ⚡ Quick Start

Deploy instantly using standard `docker-compose.yml`:

```yaml
services:
  whisper-pro-asr:
    # Choose the edition matching your hardware. Models are downloaded on first start
    # into ./model_cache, so images ship without weights:
    #   cpu | intel | intel-xpu | nvidia | nvidia-whisperx | full
    #   nvidia-intel | amd | amd-rocm-torch      (see the image table below)
    image: ventura8/whisper-pro-asr:latest
    container_name: whisper-pro-asr
    ports:
      - "9000:9000"
    restart: unless-stopped

    # 1. Intel Silicon (NPU/GPU)
    # Linux Intel hosts:
    # group_add:
    #   - "991"  # Intel render/accel group on Linux hosts
    # devices:
    #   - /dev/dri:/dev/dri # Intel iGPU / Arc (all render nodes)
    #   - /dev/accel:/dev/accel # Intel NPU (all accel nodes)
    # Windows 11 / WSL2 Intel hosts:
    # devices:
    #   - /dev/dxg:/dev/dxg # WSL GPU bridge
    #   - /dev/dri:/dev/dri # Optional if WSL exposes DRM render nodes
    #   - /dev/accel:/dev/accel # Optional if WSL exposes Intel NPU accel nodes
    # pid: host
    # privileged: true

    # 2. NVIDIA Silicon (CUDA)
    # Note: Requires NVIDIA Container Toolkit on the HOST for driver passthrough.
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [ gpu ]

    # 3. AMD GPU (native Linux ROCm via ONNX Runtime)
    # Published amd/full images support consumer Radeon RDNA2/RDNA3/RDNA4.
    # Linux AMD hosts:
    # devices:
    #   - /dev/kfd:/dev/kfd # AMD KFD (ROCm GPU driver)
    #   - /dev/dri:/dev/dri # DRM render nodes
    # Windows 11 / WSL2 AMD hosts:
    # devices:
    #   - /dev/dxg:/dev/dxg # WSL GPU bridge (detection only in this Linux container; UVR falls back to CPU)
    # If using WSL2 AMD detection, also apply `docker-compose.wsl.yml` to mount `/usr/lib/wsl:/usr/lib/wsl:ro` (WSL driver libraries, read-only).
    
    environment:
      # --- [SSD WRITE PROTECTION] ---
      - WHISPER_TEMP_DIR=/tmp/whisper

    tmpfs:
      - /tmp/whisper:size=2G,mode=1777

    volumes:
      # AI models are downloaded here on first start and reused on every restart.
      # Also holds pre-compiled hardware binaries (NPU) and the Hugging Face cache.
      - ./model_cache:/app/model_cache
      # Persistent storage for task history, telemetry, and system logs
      - ./data:/app/data
      # Recommended: Map your media volumes to enable instant (0-copy) local processing
      # The service will prioritize reading these files directly over network uploads.
      - /path/to/my/media:/media
      - /mnt/nas/tv:/tv
      - /mnt/nas/movies:/movies
```

1. Save the configuration.
2. Launch: `docker compose up -d`

> [!TIP]
> **Not sure which edition to pick?** Run `scripts/audit_hardware.sh` (Linux) or
> `scripts/audit_hardware.ps1` (Windows / Docker Desktop on WSL2). It inspects the host's
> GPUs, real Docker NVIDIA access, render nodes, Intel NPU and AMD ROCm nodes, and free disk,
> then prints the recommended `BUILD_TARGET` and `docker compose` command. Add `--env`
> (`-Env` in PowerShell) to write `BUILD_TARGET`/`HOST_INTEL_RENDER_GID` straight into `.env`.
>
> **Autonomous Hardware Resolution**: The engine automatically detects and adapts to your specific hardware (NVIDIA CUDA, native Linux AMD ROCm, Intel NPU, or Integrated GPU), optimizing the processing pipeline without requiring manual intervention. On WSL2 `/dev/dxg`, AMD detection still works, but UVR falls back to CPU in this Linux container.

## First Start

Images ship **without model weights**. On first start the service downloads the models it
needs into `./model_cache` and reuses them on every subsequent start.

- The container reports **healthy** immediately and the API stays reachable while the
  download runs -- it does not block startup.
- Transcription requests submitted during the download are **held in the queue** with the
  stage `Downloading Model (xx%)`, then run automatically once it completes. They are not
  rejected.
- `GET /status` reports `engines.whisper.status: "downloading"` while this is happening.
- Expect roughly 3-4.5 GB on first start, depending on the target and whether vocal
  separation is enabled. Later starts skip straight to serving.

Keep `./model_cache` on a persistent volume; deleting it forces a fresh download.

## Frontend Quality Gates

Dashboard UI quality is validated with ESLint, Stylelint, Vitest coverage gates, and mandatory Playwright E2E tests. All of these run exclusively inside the Docker test image via the repository's Docker quality wrapper — never directly on the host:

```bash
scripts/ci/build-and-test.sh
```

```powershell
./scripts/ci/build-and-test.ps1
```

These build `Dockerfile.test` and run `tests/run_suite.sh` inside it, which executes each step of the frontend gate list individually (HTML/JS/JS-complexity/CSS/TOML lint, `npm audit --audit-level=low`, Vitest, fixture-mock Playwright, and the real-backend Playwright project) — the same steps as the `npm run quality:frontend` aggregate script, run one by one rather than via that single command.

A second, real-backend Playwright project (`tests/e2e/real/`) runs the same dashboard/analytics/docs UI against the actual FastAPI app (`tests/e2e/real_backend/serve_real_app.py`) instead of the fixture mock server — only ASR inference and language-detection are patched to deterministic fakes, so routing, history/telemetry persistence, settings, and the auth middleware are exercised for real. Like the rest of the Playwright suite, this must run inside the Docker test image via the repository's Docker quality wrapper (`scripts/ci/build-and-test.sh` / `scripts/ci/build-and-test.ps1`, which build `Dockerfile.test` and run `tests/run_suite.sh` inside it) rather than `npm run test:e2e:real` directly on the host.

By default `tests/run_suite.sh` always runs this real-backend project (`npm run test:e2e:real`). Setting `SKIP_REAL_E2E=1` in the environment skips just that step, leaving every other gate (lint, Vitest, fixture-mock Playwright, coverage, etc.) mandatory; it's intended for local iteration only — CI and release workflows never set it, so the real-backend project always runs there.

### CI Parallelization & Caching

`tests/run_suite.sh` is stage-selectable via the `PIPELINE_STAGE` environment variable (`all` by default — used by the local wrappers above — or one of `lint`, `python-tests`, `js-unit-tests`, `e2e-fixture`, `e2e-real`). `.github/workflows/ci.yml` uses this to run each stage as its own parallel job (all depending on a `build-image` job that populates a shared `type=gha` BuildKit cache), instead of one long sequential job — a `publish` job then gates release/production-image steps on every stage job succeeding, same as before. The `lint` stage's ~24 independent tools also run concurrently against each other (not just across jobs) via background shell jobs. A named Docker volume (`whisper-pro-asr-tool-cache`) persists ESLint/Stylelint/ruff/pytest run-time caches across separate local runs; local Docker builds use `docker buildx build --cache-from/--cache-to=type=local` (mirroring CI's `type=gha` cache) so repeat local builds are fast too.

## Local Hardware Validation

Every automated test mocks the ASR engine, so a broken accelerator path -- wrong CUDA
major, missing ONNX Runtime, a model that loads but decodes garbage -- still passes them
all. **When testing on a local machine, always run the real-engine accuracy test:**

```bash
# 1. Bring up the stack for the target the audit recommended (BUILD_TARGET in .env):
#    cpu | intel | intel-xpu | nvidia | nvidia-intel | nvidia-whisperx | amd | amd-rocm-torch | full
#    .env has to be sourced: compose reads it for the container's environment, but the shell
#    expanding the filename below does not, so without this the override resolves to the
#    literal "docker-compose..yml" and compose fails on a file that does not exist.
set -a; . ./.env; set +a
docker compose -f docker-compose.yml -f "docker-compose.${BUILD_TARGET}.yml" up -d

# 2. Run the real-engine checks through the Docker test image, never on the host.
RUN_REAL_ASR=1 PIPELINE_STAGE=real-audio scripts/ci/build-and-test.sh          # smoke, <20 min
RUN_REAL_ASR=1 PIPELINE_STAGE=real-audio-stress scripts/ci/build-and-test.sh   # full matrix, ~2h
```

`docker-compose.nvidia.yml` is not a default: the override has to match `BUILD_TARGET`, or
an Intel, AMD or CPU host either fails to start or silently validates the wrong thing. Run
`scripts/audit_hardware.sh` first if you are unsure which one applies.

The `real-audio` stage posts `tests/e2e/fixtures/speech_known_text.wav` to the running
service and asserts the transcript contains both known sentences:

- *"The quick brown fox jumps over the lazy dog."*
- *"Whisper Pro ASR is running a hardware acceleration test on this machine."*

Skipped unless `RUN_REAL_ASR=1`, so it never slows CI. Override the target with
`WHISPER_BASE_URL`, and raise `REAL_ASR_TIMEOUT` if a cold-cache model download is slow.

That fixture is English only. `tests/real_audio/` extends the same live-service checks
across the multilingual audio matrix -- real neural speech per language, code-switched
clips, degraded and malformed audio, and a 20-minute long-form stress clip gated to NVIDIA
hosts:

That is what `PIPELINE_STAGE=real-audio` selects above -- a representative subset
finishing in under 20 minutes. `real-audio-stress` runs the full matrix (~2 hours) and the
20-minute long-form clip.

See [docs/SETUP.md](docs/SETUP.md) for the tier contract, the fixture generator and the
environment variables.

A correct transcript proves decoding works, not that the accelerator was used -- CPU
fallback transcribes correctly too. Confirm acceleration with `nvidia-smi
--query-compute-apps` (CUDA) or `intel_gpu_top` (Intel).

## Python Quality Gates

Backend quality checks run in CI and local parity scripts with a strict lint stack. Like the frontend gates above, these run exclusively inside the Docker test image via `scripts/ci/build-and-test.sh` / `scripts/ci/build-and-test.ps1` (which build `Dockerfile.test` and run `tests/run_suite.sh` inside it) — the commands below are the actual steps `tests/run_suite.sh` executes in-container, shown for reference, not meant to be run directly on the host.

Local parity pipeline runs (`scripts/ci/build-and-test.sh` and `scripts/ci/build-and-test.ps1`) always regenerate and overwrite `assets/coverage.svg` from the latest successful coverage results.

```bash
python3 -m ruff format --check .
python3 -m ruff check .
python3 -m flake8 modules whisper_pro_asr.py tests tests/check_coverage.py
python3 -m pylint modules whisper_pro_asr.py tests
git ls-files -z '*.py' | xargs -0 -r python3 -m radon cc -n B
hadolint Dockerfile Dockerfile.test
shellcheck scripts/ci/build-and-test.sh tests/run_suite.sh .agent/skills/workflow/resolve-pr-comments-run.sh
npm run lint:html
npm run lint:css
pwsh -NoLogo -NoProfile -Command "$issues = Invoke-ScriptAnalyzer -Path scripts -Recurse -IncludeDefaultRules -Severity Warning,Error,Information; if ($issues) { $issues | Format-Table ScriptName,Line,Severity,RuleName,Message -AutoSize; exit 1 }"
```

Cyclomatic complexity policy is strict: any Radon result with rank `B` or worse fails CI and local parity build pipelines. Required baseline is 100% rank `A` (complexity <= 5).

Ruff and Flake8 policy are strict at `140` columns, with no ignore directives.

## GitHub Releases

Pushing a semver tag `vMAJOR.MINOR.PATCH` runs `.github/workflows/ci.yml`, which:

1. Verifies `docs/releases/vMAJOR.MINOR.PATCH_github_description.md` exists.
2. Checks the tag version matches `pyproject.toml`, `package.json`, and `modules/core/config.py`.
3. Publishes the Docker image, then creates the GitHub Release via `gh release create` with that file as `--notes-file` (title = first `#` heading).

Do not rely on auto-generated GitHub release notes; curate the description before tagging.

Coverage policy for monitored dashboard and analytics JavaScript files (`modules/monitoring/templates/dashboard/**/*.js` and `modules/monitoring/templates/analytics/**/*.js`):

- Per-file minimum `90%` for `lines` and `statements`.
- CI fails when any monitored file drops below threshold.

CodeRabbit review guidance is stored in [.coderabbit.yaml](.coderabbit.yaml) and covers both dashboard JavaScript and Python modules.

## 🚀 Key Features

### 🗣 Speaker Diarization

- **WhisperX Integration**: Identify who said what with automatic speaker diarization powered by WhisperX alignment and PyAnnote speaker segmentation.
- **Speaker Labels**: Output formats (SRT, VTT, TXT, TSV) include speaker identification labels (e.g., `[SPEAKER_00]: Hello world`).
- **Configurable Speakers**: Control diarization with `min_speakers` and `max_speakers` parameters for optimal speaker count estimation.
- **Graceful Fallback**: If diarization fails or no token is configured (`DIARIZATION_HF_TOKEN`), the system seamlessly falls back to standard transcription.

### Precision Architecture

- **Multi-Backend Support**: Specialized optimization profiles for **NVIDIA CUDA**, **native Linux AMD ROCm**, **Intel OpenVINO**, and **Generic CPU** runtimes. WSL2 `/dev/dxg` enables AMD adapter detection for this image, but UVR still runs on CPU there.
- **Nesting-Safe Hardware Orchestration**: Complex pipelines (UVR → ASR → Diarization) share a single hardware claim without deadlocking, via dedicated non-locking "_direct" entry points for internal sub-stages. Top-level task dispatch is gated by the global `STATE.model_lock` semaphore (`model_lock_ctx` in `modules/inference/runtime/model_manager.py`), while which specific hardware unit gets assigned is tracked separately via `STATE.hw_pool`.
- **FFmpeg 9.0.1 Integration**: The production and Docker test images compile the signed upstream FFmpeg 9.0.1 release, with optimized hardware-accelerated decoding. All media (MKV, AVI, MP4, etc.) is automatically standardized to **16kHz Mono WAV** using the `utils.py` core before entering the AI pipeline for maximum accuracy.

### Advanced Intelligence

- **FIFO Fairness with Priority Yielding**: Tasks are processed in arrival order within the same priority tier. High-priority language detection still preempts ASR when needed, but detect-language requests are also processed FIFO among themselves.
- **Deterministic Dashboard Ordering**: Active and historical task cards are rendered in arrival order (`start_time`) so operators see the same sequence tasks entered the system.
- **Intel ASR Chunking & Streaming**: Refactored OpenVINO engine transcription to split long media files dynamically into structured chunks guided by speech VAD timestamps, ensuring stability on very long movies.
- **UVR Chunk Progress Tracking**: Computes and emits real-time preprocessing progress updates per UVR chunk to keep the dashboard progress bar fluid during vocal separation.
- **Graceful Temp-Storage Fallback**: Establishes a 2GB minimum free space threshold and 1.5x file-size headroom multiplier; both tmpfs and persistent fallback storage are validated so insufficient capacity fails early instead of causing an ENOSPC crash.
- **Cooperative Pre-emption**: High-priority operations (such as language detection) pause long-running ASR at deterministic checkpoints, including pre-vocal-separation, HQ-prep FFmpeg progress boundaries, and pre-inference, ensuring responsive API behavior under saturation.
- **Consolidated Batch Montage**: Consolidates multiple sampling targets into a single high-density montage. This allows for a **single-pass UVR isolation** across multiple non-contiguous segments, eliminating repeated model loading overhead.
- **Global VAD & In-Memory Slicing**: Features a unified Voice Activity Detection scan across the entire montage (built once via FFmpeg concat into a single file). Individual probe segments are then sliced from that montage as **NumPy arrays in memory** rather than re-extracted to disk per slice, significantly reducing VAD overhead.
- **Customizable ASR Parameters**: Fine-tune transcription with `initial_prompt` (context guidance), `vad_filter` (silence suppression), and `word_timestamps` (word-level timing).
- **Subtitle Layout Control**: Custom character-per-line wrapping (`max_line_width`) and max line block limits (`max_line_count`) for SRT/VTT output.
- **Plex-Compatible AI Subtitle Tagging**: All subtitle output filenames use the `<source>.<language>-ai.<format>` naming convention (e.g. `movie.en-ai.srt`). The `-ai` suffix leverages the ISO 3166-1 country code for Anguilla (`AI`), which Plex's regional layout parser maps to display subtitles as `<Language> (AI)` — e.g. `English (AI)`, `Spanish (AI)`. Works for all languages and both transcription and translation tasks, preventing fall-through to `xx (Unknown)` in Plex.
- **Subtitle Word Highlighting**: `subtitle_highlight_words=true` renders the currently-spoken word in a highlight color within SRT/VTT blocks, automatically enabling word-level timestamps.
- **Configurable Subtitle Promo Card**: Prepends a promo subtitle block (e.g. `"Made with Whisper Pro ASR"`) to SRT and WebVTT outputs. Customizable display duration and text are fully configurable via Docker Compose.
- **Smart Model Lifecycle**: Configurable `MODEL_IDLE_TIMEOUT` keeps models warm in memory for rapid response to bursty workloads. A deferred cleanup timer starts only after the last task completes, and is automatically cancelled and rescheduled when new tasks arrive.
- **Deferred Persistence Engine**: Protects SSD longevity by buffering task history and telemetry in RAM, only syncing to physical storage after 10 tasks or 1 hour of activity.
- **Fail-Safe Dual-Path VAD**: Intelligent logic that verifies speech presence on both isolated and raw audio, selecting the optimal path automatically based on signal clarity.
- **Squared-Confidence Voting**: Softmax probabilities are squared before aggregation, punishing low-confidence noise (e.g., spurious NO/NN hallucinations) so the dominant candidate wins the vote without a dedicated confusion-matrix lookup.
- **Unified Session Orchestration**: Integrated task and queue tracking ensures that hardware resources are only reclaimed when the system is fully idle (zero active or waiting tasks).
- **Proactive Resource Reclamation**: Automatically offloads heavy models and clears hardware caches (CUDA/NPU) only when the queue is empty, with reclaim logs reporting both process RSS and CUDA VRAM deltas when NVIDIA telemetry is available.
- **Weighted Multi-Segment Voting**: Aggregates probabilities from multiple zones with confidence-weighted averaging for industrial-strength accuracy.
- **Advanced Memory Hygiene**: Implements a "Nuclear Purge" strategy using `malloc_trim` and ctranslate2 cache clearing to keep idle memory low even after heavy ASR sessions.
- **Telemetry Downsampling**: Dual-layer downsampling (server-side and client-side) caps telemetry chart data at 300 points, ensuring smooth dashboard rendering even after extended operation.
- **Centralized Storage Hygiene**: Features a thread-local tracking system that registers every transient asset (uploads, HQ prep files, isolated stems) created during a request. The system ensures a **100% cleanup rate** by purging all tracked files immediately upon request completion or failure.
- **On-Demand History Tiering**: Implements a dual-tier storage strategy. The dashboard and RAM are strictly capped at the last 20 tasks, while a durable history of up to 1000 tasks is maintained on the persistent volume.
- **Model Download Integrity & Self-Healing**: Downloaded runtime models and assets with dedicated verification pipelines (Faster-Whisper, OpenVINO, Silero VAD, UVR vocal separation ONNX) undergo rigorous structural sanity, minimum size, and SHA-256 integrity verification. Any detected corruption triggers an automatic purge and one bounded reload attempt; failures remain visible without retry loops.
- **Hardened Diagnostic Logging**: System logs (`whisper_pro.log`) are redirected to the persistent state volume with real-time flush-to-disk logic. Log downloads are served via atomic in-memory reads to prevent `RuntimeError: Response content longer than Content-Length` failures that occur when the log file is actively written during download. Zero-caching headers ensure the latest diagnostic data is always delivered.

### Production Ready

- **OpenAI Standard API**: Drop-in compatible with the OpenAI whisper specification, allowing immediate integration with existing clients.
- **Endpoint Taxonomy (Contract)**: `/asr` and `/v1/audio/...` are equivalent standard-priority ASR surfaces, while `/detect-language` (and alias `/detectlang`) is the high-priority language-identification surface.
- **Interactive Documentation**: Full OpenAPI/Swagger interface available at `/docs` for testing and endpoint exploration.
- **Live SRT Streaming**: Features a real-time, auto-scrolling SubRip (SRT) display during processing, providing immediate visual feedback identical to the final output.
- **Persistent History Dashboard**: Maintains a durable log of all ASR and Language Detection tasks, including the hardware unit used for each completed task. Completed transcriptions are stored indefinitely and can be downloaded as `.srt` files directly from the dashboard.
- **Industrial Telemetry**: Real-time progress monitoring, including completion percentages (%), segment counts (`Seg 11 | 01:20 / 05:00`), active processing stages (e.g., UVR Preprocessing, Transcribing), and detailed hardware state reporting. NVIDIA usage is sourced from `nvidia-smi`; AMD GPU utilization is tracked by task and preprocessor activity inference (reporting `100%` when busy, `0%` when idle); Intel GPU and NPU utilization prefer native device counters before falling back to Windows performance counters or task/activity inference when needed.
- **Granular Performance Auditing**: Every task provides a detailed breakdown of its execution phases, including exact time spent in **Queue**, **Vocal Isolation**, and **AI Inference**.
- **Material Design Dashboard**: A comprehensive monitoring interface at `/dashboard` (or the root `/` when accessed via browser) featuring live task progress bars, system resource visualization, real-time auto-scrolling logs, and a **Live Refresh** toggle with fixed polling intervals (1s, 2s, 5s, 10s).
- **Bazarr Optimized**: Purpose-built for high-volume subtitle automation with stable SRT, VTT, and verbose JSON output formats.

---

### 🧩 Hardware Compatibility Matrix

| Pipeline Stage | CPU (Generic) | NVIDIA (CUDA) | AMD (native Linux ROCm) | Intel iGPU / Arc | Intel NPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Media Standardization** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Vocal Isolation (UVR)** | ✅ | ✅ | ❔ (native Linux ROCm via `/dev/kfd`); WSL2 `/dev/dxg` detects AMD but falls back to CPU | ✅ (OpenVINO) | ✅ (OpenVINO) |
| **VAD Verification** | ✅ | ✅ | ❔ | ✅ | ✅ |
| **Whisper ASR Inference** | ✅ | ✅ | ⚠️ (CPU Fallback) | ✅ *engine-dependent, see below* | ⚠️ (CPU Fallback) |
| **Speaker Diarization** | ✅ | ✅ | ❔ | ✅ | ✅ |

✅ measured on real hardware &nbsp;·&nbsp; ⚠️ works, but on CPU &nbsp;·&nbsp; ❔ **not yet validated on supported silicon**

**Intel ASR depends on the engine, not just the device.** Measured on a Core Ultra 255H
(Arc 140T iGPU + AI Boost NPU) against a 20-minute clip:

| Engine | Where ASR actually runs on Intel |
| :--- | :--- |
| `INTEL-WHISPER` (OpenVINO) | **On the iGPU.** The only engine that accelerates ASR on Intel, and ~1.7x faster than the CPU-fallback engines on the same box. |
| `FASTER-WHISPER` | CPU (int8). CTranslate2 has no Intel GPU backend. |
| `OPENAI-WHISPER` | CPU — unless you use the `intel-xpu` image, which adds the XPU torch build and runs it on the GPU. **Requires Arc (Alchemist) or newer**; on older iGPUs torch reports XPU as available and then fails to execute. |

The **NPU accelerates vocal isolation (UVR)**, not ASR: with `ASR_PREPROCESS_DEVICE=NPU`,
UVR runs on `Intel(R) AI Boost` while ASR goes to the iGPU or CPU per the table above.

> [!IMPORTANT]
> **AMD is unverified.** The ROCm paths are implemented and reasoned through, but have
> never been exercised on supported AMD silicon — the only Radeon available for testing
> was a `gfx1036` integrated part, which is not in the shipped ROCm kernel set. Treat every
> ❔ above as untested rather than working, and see
> [docs/REMOTE_VALIDATION.md](docs/REMOTE_VALIDATION.md) for how to validate it if you have
> a supported card.

### System Architecture

The service utilizes a **Heterogeneous Model Pool** to orchestrate tasks across NVIDIA GPUs, AMD GPUs, Intel NPUs, and CPUs simultaneously, with integrated WhisperX diarization and configurable model lifecycle management. For a deep dive into the processing pipelines, resource locking, and hardware acceleration logic, see the [Technical Architecture](docs/ARCHITECTURE.md) documentation.

> [!TIP]
> View the [Concurrency & Resource Orchestration](docs/CONCURRENCY.md) guide for details on parallel preprocessing and pre-emption.

---

## Prerequisites

- **Silicon**: Any CPU or Intel GPU/NPU or NVIDIA Pascal+ GPU.
- **Environment**: Docker Engine 20.10+ / Docker Desktop.
- **NPU Requirements**: Latest Intel NPU driver package (NPU Plugin).

## Configuration Reference

The service is highly tunable via environment variables in `docker-compose.yml`.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| **Runtime Control** | | |
| `ASR_DEVICE` | `AUTO` | Inference target: `AUTO`, `CUDA`, or `CPU`. |
| `ASR_PREPROCESS_DEVICE` | `AUTO` | Inference target: `AUTO`, `NPU`, `GPU`, or `CPU`. AUTO uses the next available Intel accelerator reported by OpenVINO and falls back to CPU when needed. |
| `ASR_MODEL` | `Systran/faster-whisper-large-v3` | Model ID (HuggingFace) or local path. |
| `ASR_ENGINE` | `AUTO` | Selects ASR backend engine. Options: `AUTO`, `FASTER-WHISPER`, `INTEL-WHISPER`, `OPENAI-WHISPER`, `WHISPERX`. `AUTO` resolves to `FASTER-WHISPER` on every host. Invalid values fail startup. |
| `HYBRID_ENGINES` | `false` | Off by default. On a host with both a CUDA/AMD GPU **and** an Intel GPU/NPU, `true` lets each unit run its native engine in its own worker so both accelerators stay busy -- at the cost of the engine depending on which unit serves a request. Ignored on single-vendor hosts. |
| `VOCAL_SEPARATION_MODEL` | `UVR-MDX-NET-Voc_FT` | Model ID (HuggingFace) or local path |
| `ASR_BATCH_SIZE` | `1` | Number of segments processed per pass. |
| `ASR_BEAM_SIZE` | `5` | Decoding beam width (Search depth). |
| `DEBUG` | `false` | Enables verbose stack traces and debug logging. |
| **Diarization** | | |
| `DIARIZATION_HF_TOKEN` | *(empty)* | Hugging Face token for speaker diarization (PyAnnote models). |
| **Transcription Tuning** | | |
| `INITIAL_PROMPT` | *(multilingual)* | Default context prompt to guide Whisper transcription. |
| `MODEL_IDLE_TIMEOUT` | `300` | Seconds to keep models loaded after last task (0 = immediate offload). |
| `INTEL_ASR_CHUNK_DURATION` | `300` | Chunk duration in seconds for Intel Whisper transcription. |
| `AGGRESSIVE_OFFLOAD` | `false` | Immediately unload models when idle (overridden by `MODEL_IDLE_TIMEOUT`). |
| **Subtitle Promo** | | |
| `SUBTITLE_PROMO_ENABLED` | `true` | Prepend a promo card "Made with Whisper Pro ASR" to SRT/VTT. |
| `SUBTITLE_PROMO_TEXT` | `Made with Whisper Pro ASR` | Text to display in the promo card. |
| `SUBTITLE_PROMO_DURATION` | `3.0` | Duration (in seconds) to display the promo card. |
| **Optimization** | | |
| `OV_PERFORMANCE_HINT` | `LATENCY` | OpenVINO scheduling hint (Latency/Throughput). |
| `OV_CACHE_DIR` | `./model_cache` | Persistent directory for downloaded models and compiled hardware blobs. |
| **Parallelism** | | |
| `ASR_THREADS` | `4` | CPU core allocation for inference (Auto-capped by hardware). |
| `ASR_PREPROCESS_THREADS` | `4` | CPU core allocation for UVR/ONNX (Auto-capped by hardware). |
| **SSD Protection** | | |
| `WHISPER_TEMP_DIR` | `/tmp/whisper` | Redirects transient I/O (uploads, WAVs, stems) to this path. |
| `WHISPER_TEMP_MIN_FREE_MB` | `2048` | Fallback threshold to disk if RAM-disk is full. |
| **Preprocessing** | | |
| `ENABLE_VOCAL_SEPARATION` | `false` | UVR background removal. Off by default -- measured on an RTX 5090 as 76% slower (RTF 0.063 -> 0.110) with no gain on clean speech and 1.7 points lost on harder audio. Enable for music-heavy source material. |
| `UVR_CHUNK_DURATION` | `600` | Chunk duration in seconds for UVR separation (0 to disable). |
| `ENABLE_LD_PREPROCESSING` | `true` | Toggles UVR background removal engine for language detection. |
| `LD_VAD_THRESHOLD` | `0.3` | Aggressiveness of VAD during language identification (0.0 to 1.0). |
| `SMART_SAMPLING_SEARCH` | `true` | Enables localized entropy-based signal searching in sparse audio. |
| `MAX_CUDA_UNITS` | `1` | Max NVIDIA GPUs to utilize (supports `ALL`, `AUTO`). |
| `MAX_GPU_UNITS` | `1` | Max Intel GPUs to utilize (supports `ALL`, `AUTO`). |
| `MAX_NPU_UNITS` | `1` | Max Intel NPUs to utilize (supports `ALL`, `AUTO`). |
| `MAX_CPU_UNITS` | `1` | Max concurrent CPU scheduler units (supports `ALL`, `AUTO`). Caps CPU fallbacks including AMD ASR/WSL UVR. |
| `FFMPEG_HWACCEL` | `none` | FFmpeg hardware acceleration target (`cuda`, `vaapi`, `qsv`). |
| `FFMPEG_FILTER` | `dynaudnorm` | Normalization filter: `dynaudnorm` (Standard) or `loudnorm` (Broadcast). |
| **Security & Access Control** | | |
| `API_KEY` / `WHISPER_API_KEY` | *(empty)* | Optional API key to authenticate transcription, language-ID, and telemetry API routes. |
| `ADMIN_API_KEY` | *(empty)* | Distinct admin API key for `/system/settings`, log downloads, and telemetry purge. Must be set explicitly; does not fall back to `API_KEY`. |
| `CORS_ORIGINS` | *(empty)* | Comma-separated list of allowed CORS origins (e.g. `http://localhost:3000`). |
| `CORS_ALLOW_ALL` | `false` | Enables wildcard CORS (`*`). Defaults to `false` for cross-origin security. |
| `ALLOWED_MODELS` | *(empty)* | Comma-separated list of additional allowed Hugging Face models for dynamic runtime loading. |

### 📦 Which Image Should I Pick?

Match the image to the hardware you have. **No image ships model weights** -- they download
on first start into `./model_cache`.

| Your hardware | Use this image |
| :--- | :--- |
| No GPU | **`cpu`** |
| Intel iGPU / Arc / NPU | **`intel`** |
| NVIDIA GPU | **`nvidia`** |
| NVIDIA GPU + you need **speaker diarization** | **`full`** |
| NVIDIA GPU **and** an Intel iGPU in the same box | **`nvidia-intel`** |
| AMD Radeon (native Linux ROCm) | **`amd`** |

Two extra images exist only if you want to run the **openai-whisper** engine *on the GPU*.
They are large, and most people do not need them -- the default engines are already
GPU-accelerated in the images above.

| Special case | Use this image |
| :--- | :--- |
| openai-whisper on an Intel GPU | **`intel-xpu`** |
| openai-whisper on an AMD GPU | **`amd-rocm-torch`** |

#### Capability Matrix

Sizes are uncompressed on-disk; Docker Hub reports a smaller compressed number.

| Image | Size | Transcription runs on | Vocal isolation (UVR) runs on | Engines available | Speaker diarization |
| :--- | ---: | :--- | :--- | :--- | :---: |
| **`cpu`** | 4.9 GB | CPU | CPU | Faster-Whisper, OpenAI-Whisper | — |
| **`intel`** | 5.2 GB | **Intel GPU** (OpenVINO); CPU fallback on NPU | **Intel GPU / NPU** (OpenVINO) | + Intel-Whisper | — |
| **`intel-xpu`** | 11.2 GB | **Intel GPU** (OpenVINO); CPU fallback on NPU | **Intel GPU / NPU** (OpenVINO) | + Intel-Whisper<br>OpenAI-Whisper also on **Intel GPU** **Requires Intel Arc (Alchemist) or newer** -- torch's XPU backend does not execute Whisper on older iGPUs (verified: UHD Graphics selects XPU but fails with a Level Zero error even for the `tiny` model). | — |
| **`nvidia`** | 17.5 GB | **NVIDIA GPU** (CUDA) | **NVIDIA GPU** (CUDA) | Faster-Whisper, OpenAI-Whisper | — |
| **`full`** | ~29.8 GB | **NVIDIA GPU** *and* **Intel GPU** (CUDA / OpenVINO); CPU fallback on NPU | either GPU, Intel NPU, or **AMD** (ROCm) | Faster-Whisper, Intel-Whisper, OpenAI-Whisper, WhisperX | ✅ |
| **`nvidia-whisperx`** | ~18.4 GB | **NVIDIA GPU** (CUDA) | **NVIDIA GPU** (CUDA) | + WhisperX | ✅ |
| **`nvidia-intel`** | 17.9 GB | **NVIDIA GPU** *and* **Intel GPU** at the same time | either GPU | Faster-Whisper, Intel-Whisper, OpenAI-Whisper | — |
| **`amd`** | 14.1 GB | CPU *(see note)* | **AMD GPU** (ROCm) | Faster-Whisper, OpenAI-Whisper | — |
| **`amd-rocm-torch`** | ~21.8 GB | CPU, except OpenAI-Whisper on **AMD GPU** | **AMD GPU** (ROCm) | Faster-Whisper, OpenAI-Whisper | — |

**Why AMD transcribes on the CPU:** the default engine is CTranslate2, which has no ROCm
backend at all. On AMD the GPU accelerates vocal isolation, and -- with `amd-rocm-torch` --
the openai-whisper engine. This is a limitation of the upstream engine, not of the image.

**Speaker diarization** needs WhisperX, which ships in **`full`** and in `nvidia-whisperx`.
Prefer `full` unless image size is the binding constraint: it carries every vendor's ONNX
Runtime (CPU, NVIDIA, Intel, AMD) in one image, so the same tag runs on whatever hardware
the host turns out to have, and `ASR_ENGINE`/`ASR_DEVICE` are the only things to change.
`nvidia-whisperx` is the smaller, NVIDIA-only alternative.

---

### ⚙️ ASR Backend Engines (ASR_ENGINE)

The service supports multiple ASR backend engines to run inference. You can configure this using the `ASR_ENGINE` environment variable. The following options are available:

- **`AUTO`** (default): Always resolves to **`FASTER-WHISPER`**, on every host. The engine no longer varies with the accelerators present, so the same deployment decodes identically across the fleet. Hardware still selects which *unit* the task runs on, in the order `CUDA` -> `AMD` -> `Intel GPU` -> `Intel NPU` -> `CPU`; when the chosen unit is one CTranslate2 cannot drive (AMD, Intel GPU/NPU), ASR reports and runs on the CPU while that unit stays available for vocal isolation. To use an accelerator-specific engine, ask for it explicitly.
  - `CUDA` -> `FASTER-WHISPER`
  - `Intel GPU` -> `INTEL-WHISPER`
  - `Intel NPU` -> CPU fallback (NPU remains available for vocal isolation)
  - `CPU` -> `FASTER-WHISPER`
  - An explicit `ASR_DEVICE` constrains this choice. For example,
    `ASR_DEVICE=CPU` always resolves `AUTO` to `FASTER-WHISPER`, even when
    Intel hardware is visible to the container.
- **`FASTER-WHISPER`**: Uses the CTranslate2 engine, and is what `AUTO` resolves to everywhere. Extremely fast with a low memory footprint on NVIDIA CUDA and CPU. CTranslate2 has no ROCm or OpenVINO backend, so on AMD or Intel hosts it decodes on the CPU -- the startup banner says so rather than naming a device it cannot address.
- **`INTEL-WHISPER`**: Uses the OpenVINO-based Intel Whisper engine (`IntelWhisperEngine`) on Intel Integrated/Arc GPUs, and is **the only engine that accelerates ASR on Intel** -- set `ASR_ENGINE=INTEL-WHISPER` to use it, as `AUTO` will not select it for you. Intel NPU is a vocal-isolation target, not an ASR target; without an Intel GPU, ASR falls back to CPU.
- **`OPENAI-WHISPER`**: Uses the reference OpenAI Whisper Python backend.
- **`WHISPERX`**: Uses the WhisperX backend, supporting batch inference.

When `ASR_ENGINE` is set explicitly, unsupported values are rejected at startup with a clear validation error.

---

## 📜 Full `docker-compose.yml` Example

For an exhaustive deployment featuring all optimization toggles and hardware passthrough options:

```yaml
services:
  whisper-pro-asr:
    image: ventura8/whisper-pro-asr:latest
    container_name: whisper-pro-asr
    restart: unless-stopped
    ports:
      - "9000:9000"
 
    # --- [HARDWARE ACCELERATION] ---
    # The application performs automated detection of both Intel and NVIDIA hardware.
    # To enable hardware passthrough, uncomment the appropriate sections below.
 
    # 1. Intel Silicon (iGPU / NPU) - Used for Preprocessing
    # Linux Intel hosts:
    # devices:
    #   - /dev/dri:/dev/dri # Integrated GPU / Arc (all render nodes)
    #   - /dev/accel:/dev/accel # Meteor/Lunar Lake NPU (all accel nodes)
    # Windows 11 / WSL2 Intel hosts:
    #   - /dev/dxg:/dev/dxg # WSL GPU bridge
    #   - /dev/dri:/dev/dri # Optional if WSL exposes DRM render nodes
    #   - /dev/accel:/dev/accel # Optional if WSL exposes Intel NPU accel nodes
 
    # 2. NVIDIA Silicon (CUDA)
    # Note: Requires NVIDIA Container Toolkit on the HOST for driver passthrough.
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
 
    environment:
      - DEBUG=false
 
      # --- [ENGINE CONFIGURATION] ---
      # Hardware Target: AUTO (Automated detection), CUDA (NVIDIA), CPU
      - ASR_DEVICE=AUTO
      # Computation Precision: AUTO, int8, float16 (default: AUTO)
      - ASR_COMPUTE_TYPE=AUTO
      # Model Weight Source (Faster-Whisper ID or local path)
      - ASR_MODEL=Systran/faster-whisper-large-v3
 
      # AUTO always resolves to FASTER-WHISPER, on every host. Hardware detection is a
      # separate decision: it selects the execution UNIT (CUDA > AMD > Intel GPU > NPU >
      # CPU), not the engine. An explicit ASR_DEVICE constrains which unit is chosen.
      - ASR_ENGINE=AUTO
      - INTEL_ASR_CHUNK_DURATION=300
      # --- [INFERENCE PARAMETERS] ---
      # Generation Search Breadth (Higher = more accurate, lower = faster)
      - ASR_BEAM_SIZE=5
      # Parallel segment batching (1 is recommended for single-GPU/NPU stability)
      - ASR_BATCH_SIZE=1
 
      # --- [PREPROCESSING (UVR / MDX-NET)] ---
      # Target Device: AUTO, CPU, CUDA (NVIDIA), GPU (Intel), NPU (Intel)
      - ASR_PREPROCESS_DEVICE=AUTO
      # Isolation Model Filename
      - VOCAL_SEPARATION_MODEL=UVR-MDX-NET-Inst_HQ_3.onnx
      # Vocal Separation Logic Toggles
      - ENABLE_VOCAL_SEPARATION=false   # see the table below; on costs 76% for little gain
      - ENABLE_LD_PREPROCESSING=true
      - LD_VAD_THRESHOLD=0.3
      - LD_MIN_CONFIDENCE_THRESHOLD=0.8
      - SMART_SAMPLING_SEARCH=false
 
      # --- [RESOURCE ALLOCATION] ---
      # Core limit for Whisper ASR logic
      - ASR_THREADS=4
      # Core limit for Preprocessing (ONNX Runtime)
      - ASR_PREPROCESS_THREADS=4
      # Core limit for Media Normalization (0 = auto-detect system-wide)
      - FFMPEG_THREADS=4
      # Max number of physical accelerators to use (default: all)
      - ASR_MAX_ACCEL_UNITS=1

      # --- [SSD WRITE PROTECTION] ---
      - WHISPER_TEMP_DIR=/tmp/whisper

    tmpfs:
      - /tmp/whisper:size=2G,mode=1777

    volumes:
      # Persistent cache for AI models and pre-compiled hardware binaries (NPU)
      - ./model_cache:/app/model_cache
      # Recommended: Map your media volumes to enable instant (0-copy) local processing
      # The service will prioritize reading these files directly over network uploads.
      - /path/to/my/media:/media
      - /mnt/nas/tv:/tv
      - /mnt/nas/movies:/movies
```

---

## API Reference

Comprehensive Swagger documentation is hosted at **`/docs`**.

### 1. Identify Language

**POST** `/detect-language`  
Performs multi-zone analysis to identify source language metadata. Returns full language names (e.g., "English") for Bazarr compatibility.

### 2. Transcribe Media

**POST** `/asr`  
**POST** `/v1/audio/transcriptions`  
Main entry point for generating subtitles with optional speaker diarization.

- **Formats**: `srt` (default), `vtt`, `txt`, `tsv`, `json` (with segments).
- **Diarization**: Add `diarize=true` to enable speaker identification (requires `DIARIZATION_HF_TOKEN` or request `hf_token`).
- **ASR Tuning**: `initial_prompt`, `vad_filter`, `word_timestamps` for fine-grained control.
- **Subtitle Layout**: `max_line_width` and `max_line_count` for custom subtitle formatting.
- **Word Highlighting**: `subtitle_highlight_words=true` highlights the active spoken word in SRT/VTT output.
- **Plex AI Tagging**: Subtitle files are named `<source>.<language>-ai.<ext>` so Plex displays them as `<Language> (AI)` for all languages.
- **Optimization**: Prioritizes local file access if the path exists (via volume mapping), otherwise accepts file uploads.

### 3. Service Analytics & Dashboard

**GET** `/status`  
Health-check endpoint returning model metadata, hardware status, and versioning information.

**GET** `/dashboard` (or **GET** `/` via Browser)  
Interactive Material Design interface for real-time monitoring of task progress, hardware utilization, and application memory.

**GET** `/analytics` (or **GET** `/analytics` via Browser)  
Cumulative and daily analytics dashboard with interactive charts, providing categorized breakdowns of task counts, durations, and usage patterns by endpoint (/asr, /detect-language, /v1/audio/...).

**GET/POST** `/settings`  
View or dynamically update service configuration (model, device, telemetry retention) at runtime without container restart.

---

## 📺 Bazarr Configuration

To use this service with **Bazarr**:

1. **Provider**: Choose **Whisper** (or `whisper-asr-webservice`).
2. **Endpoint**: `http://<YOUR_DOCKER_IP>:9000`
3. **Timeouts**: Should be set very high (54000) for long movies
4. **Pass video filename to Whisper**: Should be enabled for volume mapping to work correctly
5. **Volume Mapping (Highly Recommended)**:

- Ensure your Bazarr and Whisper-Pro-ASR containers share the same media paths (e.g., both map `/tv` to the same actual folder).
- When configured this way, Bazarr sends the *file path* to Whisper. Whisper Pro checks if it can read that path locally. If yes, it uses the mapped file directly and skips upload materialization.
- If paths don't match, Whisper Pro automatically falls back to handling the full file upload from Bazarr.

---

## Performance Notes

- **Golden Configuration**: We recommend **Large-V3** with **Batch=1** and **Beam=5** for the majority of CPU/GPU workloads.
- **VRAM/RAM Requirements**: Ensure at least **16GB of System RAM** when running both Vocal Isolation and Large-V3.

---

## 🛠 Project Structure

See the full annotated source tree in [Technical Architecture](docs/ARCHITECTURE.md#-project-structure).
