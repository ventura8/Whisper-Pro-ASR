# Technical Architecture

Whisper Pro v1.0.4 implements a **Heterogeneous Model Pool** architecture designed to extract maximum performance from modern hybrid silicon (Intel Meteor Lake, NVIDIA RTX).

## 🧬 Module Ecosystem

| Component | Responsibility |
|:---|:---|
| `modules/config.py` | Centralized hardware detection (CUDA/NPU/iGPU) and unit pool initialization. |
| `modules/logging_setup.py` | Orchestrates hardware banners and thread-local context filtering. |
| `modules/inference/` | Core logic for `model_manager`, `scheduler` (re-entrant locks), `preprocessing` (UVR), `vad`, and `intel_engine`. |
| `modules/api/` | Flask API layer implementing `routes_asr`, `routes_detect`, and `routes_system`. |
| `modules/monitoring/` | `dashboard`, `telemetry`, and `metrics_discovery` for real-time observability. |
| `modules/utils.py` | Managed FFmpeg normalization and **16kHz WAV Standardization**. |

### 🧩 Hardware Compatibility Matrix
| Pipeline Stage | CPU (Generic) | NVIDIA (CUDA) | Intel iGPU / Arc | Intel NPU |
| :--- | :---: | :---: | :---: | :---: |
| **Media Standardization** | ✅ | ✅ | ✅ | ✅ |
| **Vocal Isolation (UVR)** | ✅ | ✅ | ✅ (OpenVINO) | ✅ (OpenVINO) |
| **VAD Verification** | ✅ | ✅ | ✅ | ✅ |
| **Whisper ASR Inference** | ✅ | ✅ | ⚠️ (CPU Fallback) | ⚠️ (CPU Fallback) |

---

## 🏎 Processing Pipelines

### Transcription Flow (/asr)
```mermaid
graph TD
    A["Source Media"] --> STD["Standardization: 16kHz WAV"]
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
    H -->|Intel| J["OpenVINO/CPU Pipeline"]
    I --> K["Final Assembly"]
    J --> K
    end
```

### Priority Detection Flow (/detect-language)
```mermaid
graph TD
    START["Detection Request"] --> SAMPLING["Strategic Sampling: 1-15 Zones"]
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
1.  **Vocal Isolation (UVR)**
2.  **Language Identification (Whisper)**
3.  **ASR Transcription (Whisper)**

This prevents deadlocks where a task might release a unit between stages and be unable to reclaim it due to high queue volume.

### 2. Deadlock-Free Priority Resumption
The system utilizes a **Cooperative Yielding** pattern combined with an automated `release_priority` cleanup. High-priority tasks (like `/detect-language`) can signal active transcriptions to pause. Once the priority task completes, the `early_task_registration` context manager automatically triggers a system-wide resumption signal (`resume_event`), ensuring that paused tasks continue immediately exactly where they left off.

### 3. SSD Write Protection & RAM Hygiene
To ensure system longevity and minimal idle footprint:
- **Transient State**: Telemetry snapshots and real-time logs are stored in `STATE_DIR` (RAM-disk).
- **History RAM Capping**: The active registry only maintains the **20 most recent tasks** in RAM. The full history is loaded from the SSD on-demand, preventing multi-gigabyte memory spikes during idle periods.
- **Nuclear RAM Reclamation**: On task completion, the system triggers a deep purge using `malloc_trim(0)` and `ctranslate2.clear_caches()`, forcing the Operating System to reclaim 100% of unused model memory.
- **Deferred History Sync**: Task history is synced to the physical `task_history.json` on the SSD only every 10 tasks or 1 hour.

### 4. Real-time Observability Engine
The system features a thread-aware logging and telemetry engine:
- **Thread-Isolated Logs**: Utilizing a custom `TaskLogFilter`, logs are redirected to a thread-local buffer (`TASK_LOGS`) in real-time. This allows the dashboard to display execution logs specific to an active task without inter-thread noise.
- **Incremental Dashboard Updates**: The monitoring UI utilizes an incremental DOM update pattern to maintain scroll positions in log buffers and live streams while polling the `/status` endpoint every 2 seconds.
- **Granular Performance Metrics**: Every stage (Queuing, Isolation, Inference) is timed and reported, enabling precise auditing of system throughput and hardware efficiency.

---

## 🏛 Hardware Interface & Host Dependencies

- **Intel NPU/GPU**: Leverages `/dev/dri` and `/dev/accel` nodes.
- **NVIDIA CUDA**: Requires the **NVIDIA Container Toolkit** on the host.
- **SSD Optimization**: All transient I/O is redirected to a RAM-backed `tmpfs` volume to prevent physical wear.
- **Standardization Layer**: All incoming media (MKV, AVI, MP4, etc.) is standardized to 16kHz Mono WAV before entering the pipeline, ensuring consistent results across all formats.
