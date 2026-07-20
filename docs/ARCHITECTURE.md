# Technical Architecture

Whisper Pro ASR implements a **Heterogeneous Model Pool** architecture designed to extract maximum performance from modern hybrid silicon (Intel Meteor Lake, NVIDIA RTX, AMD Radeon), with integrated speaker diarization and configurable model lifecycle management.

## Concurrency Priority Policy

The architecture is concurrency-first: deadlock/livelock safety and bounded progress are treated as hard requirements.

- Throughput optimizations cannot weaken lock-safety or liveness guarantees.
- Priority/preemption pathways use indefinite waiting semantics under saturation; queued work must wait instead of failing on scheduler timeout.
- Scheduler changes must preserve documented lock-order constraints and be validated with liveness regression tests.

## 🧬 Module Ecosystem

### Core Runtime Modules (`modules/core/`)

All core runtime modules are consolidated under `modules/core/` for improved organization and import clarity:

| Component | Responsibility |
| :--- | :--- |
| `modules/core/bootstrap.py` | Hardware path patching and library redirection. Ensures correct hardware-optimized libraries are injected into `sys.path` before any AI modules are imported. |
| `modules/core/config.py` | Centralized hardware detection (CUDA/NPU/iGPU), unit pool initialization, and feature flags (`DIARIZATION_HF_TOKEN`, `MODEL_IDLE_TIMEOUT`, `INITIAL_PROMPT`). |
| `modules/core/logging_setup.py` | Orchestrates hardware banners and thread-local context filtering. |
| `modules/core/constants.py` | Static constants such as `HALLUCINATION_PHRASES` used across the codebase. |
| `modules/core/utils.py` | Managed FFmpeg normalization, **16kHz WAV Standardization**, subtitle generation with `wrap_text()` layout control, speaker label formatting, and cross-platform utilities. |
| `modules/core/subtitles.py` | Subtitle format generation (SRT, VTT, TSV, TXT) with text wrapping, speaker labels, and layout customization. |

### Application Modules

| Component | Responsibility |
| :--- | :--- |
| `modules/inference/` | Inference stack grouped by concern: `runtime/` (`model_manager`, `concurrency`, segment consumption), `scheduler/` (re-entrant locks, state/order/task helpers), `pipeline/` (`preprocessing/` package with orchestrator in `__init__.py` plus `helpers.py`, `provider.py`, `execution.py`, alongside `vad`, `language_detection`, `diarization`, `post_processing`), and `engines/` (`base`, `engine_factory`, `faster_whisper_engine`, `openai_whisper_engine`, `whisperx_engine`, `intel_engine`). |
| `modules/api/` | FastAPI application layer grouped by concern: `routes/` (`asr`, `detect`, `system`) and `support/` (`request_utils`, `upload_extraction`, `local_path`) for shared request/materialization/path-approval logic. |
| `modules/monitoring/` | `dashboard` & `dashboard_ui` (Material Design UI renderer loading manifest-ordered modules from `templates/dashboard_js_files.txt`), `analytics_ui` (analytics dashboard loaded from `templates/analytics_js_files.txt`), `telemetry` & `telemetry_manager` (persistent telemetry history), `history_manager` (task history with dual-tier storage), and `metrics_discovery` (hardware metrics). |

### 🧩 Hardware Compatibility Matrix

| Pipeline Stage | CPU (Generic) | NVIDIA (CUDA) | AMD (ROCm/DirectML) | Intel iGPU / Arc | Intel NPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Media Standardization** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Vocal Isolation (UVR)** | ✅ | ✅ | ✅ (ONNX ROCm/DirectML) | ✅ (OpenVINO) | ✅ (OpenVINO) |
| **VAD Verification** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Whisper ASR Inference** | ✅ | ✅ | ⚠️ (CPU Fallback) | ⚠️ (CPU Fallback) | ⚠️ (CPU Fallback) |
| **Speaker Diarization** | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 🏎 Processing Pipelines

### Transcription Flow (/asr)

```mermaid
graph TD
    EP["Endpoint Surface: /asr or /v1/audio/..."] --> A["Source Media"]
    A --> STD["Standardization: 16kHz WAV"]
    STD -->|Check| L{"Lang Given?"}
    L -->|No| LD["Optimized Language ID"]
    L -->|Yes| PRE["Preprocessing (UVR)"]
    LD --> PRE
    
    subgraph CORE ["Heterogeneous Engine Pool"]
    PRE -->|16kHz Stereo| C{"Isolation?"}
    C -->|Enabled| D["UVR Separation (Re-entrant Lock)"]
    C -->|Disabled| E["Standard Signal"]
    
    D --> VAD{"Single-Pass VAD"}
    VAD -->|Isolated Silent| E
    VAD -->|Isolated Speech| F["Processing Signal"]
    E --> F
    
    F --> G["Faster-Whisper Inference"]
    G -->|Heterogeneous Parallel| H{"Unit Pool"}
    H -->|NVIDIA| I["CUDA Acceleration"]
    H -->|AMD| AMD["ONNX ROCm/DirectML (UVR) + CPU ASR"]
    H -->|Intel| J["OpenVINO/CPU Pipeline"]
    I --> K["Segment Assembly"]
    AMD --> K
    J --> K
    end

    K --> DIAR{"Diarize?"}
    DIAR -->|No| FMT["Format Output"]
    DIAR -->|Yes| ALIGN["WhisperX Alignment"]
    ALIGN --> SPEAK["Speaker Diarization (PyAnnote)"]
    SPEAK --> ASSIGN["Speaker Assignment"]
    ASSIGN --> FMT
    FMT --> OUT["SRT / VTT / JSON / TXT / TSV"]
```

### Speaker Diarization Pipeline

```mermaid
graph LR
    SEG["Raw Segments"] --> ALIGN["whisperx.align()"]
    ALIGN --> DIAR["DiarizationPipeline (DIARIZATION_HF_TOKEN)"]
    DIAR --> ASSIGN["assign_word_speakers()"]
    ASSIGN --> LABELED["Speaker-Labeled Segments"]
    
    subgraph CACHE ["Model Cache Pools"]
    AP["_ALIGN_POOL (per unit)"]
    DP["_DIARIZE_POOL (per unit)"]
    end
```

### Priority Detection Flow (/detect-language)

```mermaid
graph TD
    START["Detection Request: /detect-language or /detectlang"] --> SAMPLING["Strategic Sampling: 1-15 Zones"]
    SAMPLING --> MONTAGE["Batch Montage: FFmpeg Concat (16kHz Stereo)"]
    
    subgraph BATCH ["Consolidated Batch Pipeline"]
    MONTAGE --> ISOLATE["UVR Isolation (Single Pass - Re-entrant)"]
    ISOLATE --> VAD["Global VAD Scan (One Pass)"]
    VAD --> BATCH_INF["Batch Inference Session"]
    
    subgraph LOOP ["In-Memory Slicing"]
    BATCH_INF --> SLICE["NumPy 30s Slice"]
    SLICE --> SPEECH{"Has Speech?"}
    SPEECH -->|Yes| ID["Whisper Identification (No VAD)"]
    SPEECH -->|No| NEXT["Next Slice"]
    ID --> NEXT
    end
    
    NEXT -->|All Done| VOTE["Squared Weighting Vote"]
    VOTE --> RETURN
    end
```

---

## 🔒 Granular Resource Orchestration

### 1. Re-entrant Hardware Locks

The system implements a **Thread-Local Re-entrant Locking Pattern** via `model_lock_ctx()`. This allows a high-level task (like a full transcription request) to "claim" a hardware unit once and share it across all internal sub-stages:

1. **Vocal Isolation (UVR)**
2. **Language Identification (Whisper)**
3. **ASR Transcription (Whisper)**
4. **Speaker Diarization (WhisperX)**

This prevents deadlocks where a task might release a unit between stages and be unable to reclaim it due to high queue volume.

### 2. Deadlock-Free Priority Resumption

The system utilizes a **Cooperative Yielding** pattern combined with an automated `release_priority` cleanup. High-priority tasks (like `/detect-language`) can signal active transcriptions to pause. Priority tasks are not globally serialized and may run in parallel across multiple available/borrowed units, while same-priority FIFO ordering is preserved at acquisition boundaries. Once a priority task completes, the context manager automatically triggers unit-scoped resumption signaling, ensuring paused tasks continue exactly where they left off.

- **Standard Task Yielding**: Standard tasks yield resource acquisition and loop-sleep instead of blocking on the model lock semaphore whenever priority tasks are present in the registry, preventing priority starvation.
- **Priority Preemption Bypass**: Running priority tasks ignore preemption requests, preventing them from pausing themselves if multiple priority tasks are queued.
- **Preemption Visibility**: Preempted tasks temporarily transition to `"queued"` status with a `"Paused for Priority Task"` stage, ensuring they display in the dashboard queue.
- **Unit-Scoped Gating Only**: Pause/resume gates that affect execution are limited to per-hardware-unit sync entries in `STATE.unit_sync[unit_id]`. Shared scheduler events are compatibility mirrors only and are not used as execution gates.
- **Arrival-Aware FIFO Acquisition**: The scheduler records a `task_arrival_order` timestamp for each task and uses `has_earlier_task()` to enforce FIFO only among tasks of the same priority tier at hardware-acquisition time.
- **Waiting-Only Blocking Rule**: Same-tier FIFO blocking applies only while earlier tasks are still waiting for hardware (`initializing`/`queued` without `unit_id`), avoiding starvation when earlier tasks are already actively executing on another accelerator.
- **Centralized Storage Hygiene**: Implements a `tracked_files` registry within the thread context. Every transient file (uploaded media, standardized WAVs, HQ prepared files, and isolated stems) is registered upon creation. A mandatory `cleanup_files()` call in the request's `finally` block ensures a **100% deletion rate**, eliminating storage leaks even after fatal errors.

### 3. Model Lifecycle & Idle Timeout

The system supports two model lifecycle strategies, configured via environment variables:

| Strategy | Config | Behavior |
| :--- | :--- | :--- |
| **Aggressive Offload** | `AGGRESSIVE_OFFLOAD=false` | Models are unloaded from memory immediately when active sessions drop to zero. |
| **Idle Timeout** | `MODEL_IDLE_TIMEOUT=300` (default) | A deferred `threading.Timer` is started after the last task completes. Models are only purged after the configured idle period (in seconds) elapses with zero active sessions. New incoming tasks cancel the pending timer, keeping models warm for bursty workloads. |

When `MODEL_IDLE_TIMEOUT > 0` (or defaults to `300`), it takes precedence over `AGGRESSIVE_OFFLOAD`. The timer is started lazily on the first session decrement that brings the active count to zero. If a new task arrives while the cleanup routine is actively executing, the system allows the cleanup to complete and re-initializes models on demand.

```mermaid
graph TD
    DEC["Session Decrement"] --> CHK{"Active == 0?"}
    CHK -->|No| DONE["Continue"]
    CHK -->|Yes| TIMEOUT{"IDLE_TIMEOUT > 0?"}
    TIMEOUT -->|Yes| CANCEL["Cancel Existing Timer (if any)"]
    CANCEL --> START["Start Deferred Timer"]
    TIMEOUT -->|No| AGG{"AGGRESSIVE_OFFLOAD?"}
    AGG -->|Yes| UNLOAD["unload_models()"]
    AGG -->|No| DONE
    START --> WAIT["Wait (IDLE_TIMEOUT seconds)"]
    WAIT -->|Timer Fires| LOCKCHK{"Active still == 0?"}
    LOCKCHK -->|Yes| UNLOAD
    LOCKCHK -->|No| DONE
    WAIT -->|New Task Arrives| CANCEL2["Timer Cancelled"]
    CANCEL2 --> DONE
```

### 4. Real-time Observability Engine

The system features a thread-aware logging and telemetry engine designed for industrial reliability:

- **Hardened Diagnostic Logging**: Implements a persistent, idempotent logging architecture. The `whisper_pro.log` stream is guaranteed across application lifecycles via a hardened initialization sequence that survives global resets.
- **Thread-Isolated Buffers**: Utilizing a custom `TaskLogFilter`, logs are redirected to a thread-local buffer (`TASK_LOGS`) in real-time. This allows the dashboard to display execution logs specific to an active task without inter-thread noise.
- **Real-Time Synchronization**: The log download endpoint features a mandatory flush-to-disk sequence and zero-caching headers, ensuring diagnostics are always current.
- **Telemetry Downsampling**: A dual-layer downsampling strategy caps telemetry data at 300 points for dashboard chart rendering. Server-side downsampling in `telemetry.py` reduces payloads before transmission, while client-side downsampling in `dashboard_ui.py` provides an additional safety net for chart performance.
- **Hardware Utilization Probes**: `metrics_discovery.py` assembles per-unit utilization for CUDA, Intel GPU, and Intel NPU. CUDA uses `nvidia-smi` as the primary source and falls back to activity inference when direct telemetry is unavailable. Intel GPU and NPU probe in strict order: native Linux device counters, then Windows performance counters, and only then task/activity inference. The dashboard consumes the resulting `telemetry.hardware_util` map keyed by unit id, with legacy fields kept only for compatibility.
- **Service Analytics**: The `/analytics` endpoint and dedicated analytics UI (`analytics_ui.py`) provide cumulative and daily breakdowns of task counts, durations, and usage patterns separated by HTTP endpoint surface (`/asr`, `/detect-language`/`/detectlang`, and `/v1/audio/...`) from persistent task history. The analytics dashboard composes modular HTML/CSS plus foldered JS under `modules/monitoring/templates/analytics/`, loaded in deterministic order from `templates/analytics_js_files.txt`.
- **Industrial Quality Standard**: The entire ecosystem is maintained with a strict **10.00/10 Pylint score**, strict **Ruff static analysis and formatting compliance** at `140` columns, strict **Flake8 compliance** (`max-line-length=140`, no ignore directives), and **>90% test coverage** across all modules and tests, representing a zero-regression baseline for enterprise deployments.
- **Incremental Dashboard Updates**: The monitoring UI utilizes an incremental DOM update pattern to maintain scroll positions in log buffers and live streams while polling the `/status` endpoint every 2 seconds. Dashboard JS is split across `modules/monitoring/templates/dashboard/core/`, `modules/monitoring/templates/dashboard/features/`, plus the orchestration entrypoint `modules/monitoring/templates/dashboard/main.js`, and concatenated in deterministic manifest order from `templates/dashboard_js_files.txt`.
- **O(1) Live Subtitle Updates**: Appends pre-formatted subtitle blocks incrementally to the live SRT display stream during processing instead of doing full $O(N^2)$ stream reconstructions, preventing performance bottlenecks and memory bloat on large media files.

### 5. Long-Movie Processing & Audio Chunking

- **Intel ASR Chunking & Streaming**: Refactored OpenVINO engine transcription (`IntelWhisperEngine`) to split long media files dynamically into structured chunks (configured via `INTEL_ASR_CHUNK_DURATION`, default 300 seconds), guided by speech VAD timestamps (`find_split_points()`), and auto-detecting/locking the language on the first chunk to ensure stability on very long movies.
- **UVR Chunk Progress Tracking**: Patches the UVR vocal separation process dynamically on the scheduler to compute and emit real-time chunk progress status according to `UVR_CHUNK_DURATION` (default 600 seconds) to prevent visual hangs.
- **Graceful Temp-Storage Fallback**: Establishes a 2GB minimum free space threshold and 1.5x file-size headroom multiplier to fallback gracefully to persistent storage (`PERSISTENT_TEMP_DIR`) when tmpfs runs low on space.

---

## 🏐 Hardware Interface & Host Dependencies

- **Intel NPU/GPU**: Leverages `/dev/dri` and `/dev/accel` nodes.
- **NVIDIA CUDA**: Requires the **NVIDIA Container Toolkit** on the host.
- **AMD GPU (ROCm/DirectML)**: Leverages `/dev/kfd` and `/dev/dri` on Linux; uses `/dev/dxg` (WSL GPU bridge) on Windows. `onnxruntime-rocm` is isolated under `/app/libs/amd` and loaded automatically when AMD hardware is detected. Whisper ASR runs on CPU while UVR vocal isolation offloads to the AMD GPU via ONNX Runtime ROCm/DirectML.
- **SSD Optimization**: All transient I/O is redirected to a RAM-backed `tmpfs` volume to prevent physical wear.
- **Standardization Layer**: All incoming media (MKV, AVI, MP4, etc.) is standardized to 16kHz Mono WAV before entering the pipeline, ensuring consistent results across all formats.
- **Diarization Models**: WhisperX alignment and PyAnnote diarization models are cached per hardware unit in `_ALIGN_POOL` and `_DIARIZE_POOL`. These are purged alongside Whisper models during `unload_models()`.
