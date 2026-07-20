# GitHub Release v1.2.0

## 🚀 AMD GPU Support – Dual-Hardware Acceleration

This release introduces automatic AMD GPU detection and hardware acceleration for UVR vocal isolation via `onnxruntime-rocm`, enabling heterogeneous dual-GPU pipelines (e.g., UVR on AMD, Whisper ASR on NVIDIA) with full dashboard monitoring support.

---

## ✨ New Features

### AMD GPU Hardware Acceleration

- **Automatic AMD Detection**: Runtime automatically detects AMD GPUs via `/dev/kfd`, DRM vendor ID (`0x1002`), and ONNX Runtime execution provider availability. No manual device override required.
- **Segregated ONNX Runtime**: `onnxruntime-rocm==1.22.2.post3` is installed under `/app/libs/amd`, fully isolated from the Intel OpenVINO and CPU ONNX paths.
- **UVR on AMD GPU**: Vocal isolation (UVR/MDX-NET) runs natively on the AMD GPU via ROCm/DirectML ONNX Runtime execution providers.
- **Whisper ASR CPU Fallback**: CTranslate2 does not support ROCm, so Whisper ASR inference automatically falls back to CPU with `int8` compute type on AMD units. No manual configuration needed.
- **Dual-GPU Parallel Execution**: When both NVIDIA and AMD GPUs are present, the runtime automatically assigns Whisper ASR to NVIDIA CUDA and UVR preprocessing to AMD GPU — leveraging both accelerators simultaneously.
- **AMD Scheduler Unit**: AMD GPUs are registered as `amd:0` hardware units in the resource pool, eligible for the standard priority queue alongside CUDA, Intel GPU, NPU, and CPU units.

### Dashboard AMD Pretty Printing

- **AMD GPU Label**: AMD hardware units display as `AMD GPU` in the dashboard active card, history, and hardware pool.
- **Lightning Bolt Icon**: AMD units use the `bolt` Material Icon (high-performance indicator) instead of the generic memory chip icon.
- **Utilization Charts**: AMD GPU utilization tracking added to hardware charts — reports `100%` when processing and `0%` when idle.

---

## 🔧 Changes

### Core

- **`modules/core/bootstrap.py`**: Added `_detect_amd_hardware`, `_has_amd_drm_vendor`, and dual-GPU path selection logic. AMD targets `/app/libs/amd`, NVIDIA targets `/app/libs/nvidia`. When both are detected, AMD path is loaded for ONNX Runtime while CTranslate2 binds directly to NVIDIA CUDA.
- **`modules/core/config_helpers.py`**: Added `_update_amd_state` — auto-sets `prep_device = "AMD"` when AMD hardware is detected and `MAX_AMD_UNITS >= 1`.
- **`modules/core/config.py`**: Added `MAX_AMD` environment variable parsing and AMD unit logging.
- **`modules/core/engine_registry.py`**: Added `"AMD"` to `AUTO_ENGINE_PRIORITY` ordering (after CUDA, before Intel GPU).

### Inference

- **`modules/inference/engines/engine_factory.py`**: Added `float16` → `int8` coercion for Whisper/WhisperX engines initializing on CPU slots (AMD units). Extracted `_create_whisperx_engine` helper to maintain Radon Rank A complexity.

### Monitoring & Telemetry

- **`modules/monitoring/metrics_discovery.py`**: Added `_resolve_amd_utilization` and AMD branch in `_resolve_unit_utilization`. AMD utilization reports activity-based inference (100% busy / 0% idle) via `_probe_activity_fallback`. Refactored `_resolve_unit_utilization` to dictionary dispatch for Radon Rank A compliance.
- **`modules/monitoring/templates/dashboard/core/utils.js`**: Fixed missing `const source = id || type;` variable declaration in `_normalizeHardwareFamily` (regression fix). Added AMD family normalization and `bolt` icon mapping.
- **`modules/monitoring/templates/dashboard/features/runtime.js`**: Added `_amdVisual` renderer for AMD hardware units.
- **`modules/monitoring/templates/dashboard/features/task_filter_history.js`**: Mapped AMD device type to `bolt` icon.

### Docker & Infrastructure

- **`Dockerfile`**: Added `pip install onnxruntime-rocm==1.22.2.post3 --target /app/libs/amd` stage (no-deps) for AMD path isolation.
- **`docker-compose.yml`**: Added NVIDIA device reservations, `/dev/dxg`, `/dev/kfd`, and `/dev/dri` passthrough mappings (uncommented for dual-GPU test hosts).

### Tests

- **`tests/unit/test_config_amd.py`**: 83-line AMD hardware detection unit test file covering `_detect_amd_hardware`, `_update_amd_state`, AMD scheduler unit registration, and fallback behavior.

### Documentation

- **`README.md`**: Updated description, hardware matrix, quick start compose, telemetry, and system architecture to include AMD GPU.
- **`docs/ARCHITECTURE.md`**: Updated silicon header, hardware matrix, transcription flow diagram, and hardware interface section.
- **`docs/DOCKERHUB_DESCRIPTION.md`**: Updated description, hardware matrix, quick start, and GPU/NPU Support section.
- **`docs/SETUP.md`**: Replaced "AMD telemetry readiness note" (stating AMD was not supported) with full AMD GPU configuration instructions, device mapping guide, and engine resolution order update.
- **`.agent/skills/runtime/amd_hardware_inference_skill.md`**: Created comprehensive AMD runtime skill documenting ONNX isolation, CPU fallback behavior, docker mapping, and verification procedure.

---

## ✅ Verification

| Gate | Status |
| :--- | :---: |
| Backend tests (908 tests) | ✅ Passed |
| Playwright E2E (21 tests) | ✅ Passed |
| Python coverage ≥ 90% | ✅ Passed |
| JS frontend coverage ≥ 90% | ✅ Passed |
| Pylint 10.00/10 | ✅ Passed |
| Ruff + Flake8 + Black | ✅ Passed |
| Radon Rank A (100%) | ✅ Passed |
| Bandit + pip-audit | ✅ No vulnerabilities |
| npm audit --audit-level=low | ✅ No vulnerabilities |
| Hadolint + ShellCheck + PSScriptAnalyzer | ✅ Passed |
| Live runtime test (NVIDIA + AMD) | ✅ Both GPUs detected and used |

**Tested on**: Dual-GPU host (NVIDIA GeForce RTX 5090 + AMD Radeon Graphics) under Windows 11 WSL2 Docker Desktop.
