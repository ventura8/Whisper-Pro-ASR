# Concurrency & Resource Orchestration

This document provides an exhaustive technical reference for the multithreading and resource management strategies implemented in **Whisper Pro ASR**. It is intended for developers and AI agents to understand the system's high-concurrency safeguards.

---

## 🏗 High-Level Architecture

Whisper Pro ASR uses a **Hybrid Concurrency Model** that balances I/O-bound tasks (FFmpeg extraction), CPU-bound tasks (Audio normalization), and Hardware-bound tasks (AI Inference).

### 1. The Locking Hierarchy
The system utilizes a multi-layered locking strategy to prevent race conditions and VRAM/NPU congestion:

| Lock Name | Type | Scope | Responsibility |
|:---|:---|:---|:---|
| `_MODEL_LOCK` | `threading.Lock` | Global | Ensures serial access to the Whisper Inference engine. |
| `_PRIORITY_LOCK` | `threading.Lock` | Global | Protects the priority request counter and pre-emption signal. |
| `_PRIORITY_SEQUENTIAL_LOCK` | `threading.RLock` | Global | Serializes multiple high-priority tasks (e.g., concurrent Detection calls). |
| `PreprocessingManager._lock` | `threading.Lock` | Instance | Serializes access to the UVR/MDX-NET isolation engine. |

### Resource Contention Visualization
```mermaid
graph TD
    REQ[Incoming Request] --> TYPE{Task Type?}
    
    subgraph "Transcription Flow"
    TYPE -->|ASR| ALOCK[_MODEL_LOCK]
    ALOCK --> AEXEC[Inference Execution]
    end
    
    subgraph "Priority Flow"
    TYPE -->|Detection| SLOCK[_PRIORITY_SEQUENTIAL_LOCK]
    SLOCK --> PLOCK[_PRIORITY_LOCK]
    PLOCK --> DLOCK[_MODEL_LOCK]
    DLOCK --> DEXEC[Fast Metadata Inference]
    end
    
    AEXEC -.->|Yield Point| DLOCK
```

---

## 🚦 Request Prioritization & Pre-emption

Whisper Pro implements a **Zero-Wait Detection** system that allows high-priority metadata tasks (Language Detection) to interrupt long-running transcriptions.

### The Yielding Workflow
1. **Priority Arrival**: A `/detect-language` request arrives and calls `model_manager.request_priority()`.
2. **Signal Broadcast**: The `_PAUSE_REQUESTED` event is set and `_RESUME_EVENT` is cleared.
3. **Yield Point**: The active Transcription thread checks for the pause signal at the end of every audio segment.
4. **Hardware Surrender**: The Transcription thread calls `wait_for_priority(model_lock=_MODEL_LOCK)`. It releases the Whisper engine lock and waits on `_RESUME_EVENT`.
5. **Priority Execution**: The Detection task acquires the locks, utilizes the hardware, and completes.
6. **Resumption**: The Detection task calls `release_priority()`, which sets `_RESUME_EVENT`. The Transcription thread re-acquires the lock and continues.

### Pre-emption Sequence Diagram
```mermaid
sequenceDiagram
    participant ASR as ASR Task (Thread A)
    participant SCHED as Scheduler (model_manager)
    participant DET as Detection Task (Thread B)
    participant HW as Hardware (NPU/GPU)

    ASR->>HW: Whisper Processing...
    DET->>SCHED: request_priority()
    SCHED->>SCHED: Set _PAUSE_REQUESTED
    ASR->>SCHED: wait_for_priority()
    Note over ASR,HW: At next segment boundary
    ASR->>HW: Release _MODEL_LOCK
    SCHED->>ASR: Suspend (wait for _RESUME_EVENT)
    DET->>HW: Acquire _MODEL_LOCK
    DET->>HW: Fast Inference
    DET->>HW: Release _MODEL_LOCK
    DET->>SCHED: release_priority()
    SCHED->>SCHED: Set _RESUME_EVENT
    SCHED->>ASR: Resume Signal
    ASR->>HW: Re-acquire _MODEL_LOCK
    ASR->>HW: Continue Transcription
```

---

## 📦 Parallel Preprocessing Pipeline

While AI Inference is serialized to prevent hardware crashes, **Data Preparation** is highly parallelized.

### 1. Zone Extraction
In the `/detect-language` endpoint, the system uses a `ThreadPoolExecutor` (sized by `ASR_PREPROCESS_THREADS`) to extract and isolate up to 5 strategic audio zones concurrently.
- **FFmpeg**: Parallelized at the OS level (static builds).
- **UVR/Isolation**: Serialized via internal locks, but "Extraction" of zones happens in parallel while the UVR engine is busy with the previous zone.

### Detection Pipeline (Multi-Zone Voting)
The following diagram illustrates how the system balances high-throughput data preparation with serialized AI inference:

```mermaid
graph TD
    START[Detection Request] --> SAMPLING[Strategic Sampling: Define 5 Zones]
    SAMPLING --> POOL{ThreadPoolExecutor}
    
    subgraph "Parallel Preparation (CPU Bound)"
    POOL --> Z0P[Zone 0: FFmpeg + VAD]
    POOL --> Z1P[Zone 1: FFmpeg + VAD]
    POOL --> Z2P[Zone 2: FFmpeg + VAD]
    end
    
    subgraph "Serialized Execution (Hardware Bound)"
    Z0P --> Z0I[UVR Isolation]
    Z0I --> Z0W[Whisper Inference]
    Z0W --> EXIT_CHECK{Conf >= 80%?}
    
    EXIT_CHECK -->|Yes| RETURN[Early Exit & Return]
    
    EXIT_CHECK -->|No| Z1I[UVR Isolation]
    Z1I --> Z1W[Whisper Inference]
    Z1W --> EXIT_CHECK_2{Conf >= 80%?}
    
    EXIT_CHECK_2 -->|Yes| RETURN
    EXIT_CHECK_2 -->|No| VOTING[Aggregate & Vote]
    VOTING --> RETURN
    end
    
    Z0I -.->|Internal Lock| Z1I
    Z1I -.->|Internal Lock| Z2I[Zone 2 Isolation]
```

### 2. FFmpeg Threading
The `FFMPEG_THREADS` configuration limit is applied to every individual FFmpeg call to prevent "Fork Bombs" that could starve the primary Flask service.

---

## 🛠 Resource Lifecycle & Keep-Alive

To ensure production-grade stability, Whisper Pro implements **Unified Session Tracking**.

### 1. Tracking Counters
- `_ACTIVE_SESSIONS`: Tracks tasks currently inside a route or core execution function.
- `_QUEUED_SESSIONS`: Tracks tasks currently blocked by a hardware lock or waiting for pre-emption.

### 2. Proactive Reclamation
The system triggers `_check_and_offload_resources()` every time a session ends.
- **Rule**: Reclamation (VRAM/RAM clearing) ONLY happens if `Active + Queued == 0`.
- **Optimization**: If a second task is waiting in the queue, the model remains resident, avoiding the expensive "Initialization Penalty" (NPU compilation/VRAM allocation).

### Lifecycle State Machine
```mermaid
stateDiagram-v2
    [*] --> Idle: _ACTIVE = 0, _QUEUED = 0
    Idle --> Loading: New Task Entry
    Loading --> Active: Lock Acquired
    Active --> Waiting: Pre-empted / Lock Contention
    Waiting --> Active: Lock Re-acquired
    Active --> Idle: Session Counter = 0
    Idle --> Offloading: _check_and_offload_resources()
    Offloading --> [*]: VRAM Purged
    
    Active --> Active: Task Finish (Counter > 0)
    Waiting --> Waiting: Still Queued
```

---

## 🧪 Concurrency Safeguards (Fail-Safe)

### Deadlock Prevention
- **Re-entrant Locks**: `_PRIORITY_SEQUENTIAL_LOCK` allows the same thread to acquire priority multiple times without locking itself.
- **Timeout Protection**: Internal `wait()` calls are balanced with state checks to ensure threads don't "hang" if a signal is missed.
- **Finally Blocks**: All lock acquisitions and session increments are wrapped in `try...finally` to ensure state balance even on catastrophic hardware failure.

### Hardware Isolation
The system distinguishes between **Isolation** (NPU/iGPU) and **Transcription** (CUDA/CPU). This "Split Architecture" allows UVR and Whisper to potentially run on different accelerators simultaneously, though internal locks still protect each specific resource from over-subscription.

---

## 📈 Configuration reference

| Variable | Target | Description |
|:---|:---|:---|
| `ASR_THREADS` | Whisper/PyTorch | Internal CPU threads for tensor operations. |
| `ASR_PREPROCESS_THREADS` | Parallel Pool | Number of concurrent FFmpeg extraction/isolation tasks. |
| `FFMPEG_THREADS` | FFmpeg Binary | Threads per extraction process. |
| `ASR_PARALLEL_LIMIT_ACCEL` | Global Queue | Maximum parallel requests allowed for hardware accelerators. |

---

> [!IMPORTANT]
> **Production Note**: Always ensure that `ASR_PREPROCESS_THREADS` * `FFMPEG_THREADS` does not exceed the physical core count of the host to maintain sub-second API responsiveness.
