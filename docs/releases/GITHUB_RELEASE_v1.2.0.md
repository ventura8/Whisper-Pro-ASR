# GitHub Release v1.2.0

## 🚀 AMD GPU Support & Security Hardening

This release introduces automatic AMD GPU detection and hardware acceleration for UVR vocal isolation via `onnxruntime-rocm`, enabling heterogeneous dual-GPU pipelines (e.g., UVR on AMD, Whisper ASR on NVIDIA) with full dashboard monitoring support. It also adds a comprehensive defense-in-depth security layer with CORS allowlisting, dual-tier API key authentication, anti-CSRF protection, and model supply chain guarding.

---

## ✨ New Features

### AMD GPU Hardware Acceleration

- **Automatic AMD Detection**: Runtime automatically detects AMD GPUs via `/dev/kfd`, DRM vendor ID (`0x1002`), and ONNX Runtime execution provider availability. No manual device override required.
- **Segregated ONNX Runtime**: `onnxruntime-rocm==1.22.2.post3` is installed under `/app/libs/amd`, fully isolated from the Intel OpenVINO and CPU ONNX paths.
- **UVR on AMD GPU**: Vocal isolation (UVR/MDX-NET) runs natively on the AMD GPU on native Linux ROCm hosts with `/dev/kfd`. WSL2 `/dev/dxg` still allows AMD adapter detection in this Linux container, but UVR falls back to CPU there.
- **Whisper ASR CPU Fallback**: CTranslate2 does not support ROCm, so Whisper ASR inference automatically falls back to CPU with `int8` compute type on AMD units. No manual configuration needed.
- **Dual-GPU Parallel Execution**: When both NVIDIA and AMD GPUs are present, the runtime automatically assigns Whisper ASR to NVIDIA CUDA and UVR preprocessing to AMD GPU — leveraging both accelerators simultaneously.
- **AMD Scheduler Unit**: AMD GPUs are registered as `amd:0` hardware units in the resource pool, eligible for the standard priority queue alongside CUDA, Intel GPU, NPU, and CPU units.

### Security & Access Control

- **Defense-in-Depth Security Layer**: New `modules/api/support/security.py` implementing CORS allowlist, API key authentication, anti-CSRF protection, model supply chain guarding, and audit logging.
- **No Wildcard CORS by Default**: Cross-origin access is denied unless explicitly configured via `CORS_ORIGINS` or `CORS_ALLOW_ALL=true`.
- **Dual-Tier Authentication**: Separate `API_KEY` (general API) and `ADMIN_API_KEY` (administrative endpoints) with constant-time token comparison.
- **Anti-CSRF Origin Validation**: State-modifying admin endpoints require Origin/Referer verification against the request host or CORS allowlist in unauthenticated mode.
- **Model Supply Chain Allowlist**: Dynamic model loading validates against standard Whisper models, trusted HuggingFace repositories, and explicit `ALLOWED_MODELS` entries. Rejects path traversals and arbitrary repositories.

### Dashboard AMD Pretty Printing

- **AMD GPU Label**: AMD hardware units display as `AMD GPU` in the dashboard active card, history, and hardware pool.
- **Lightning Bolt Icon**: AMD units use the `bolt` Material Icon (high-performance indicator) instead of the generic memory chip icon.
- **Utilization Charts**: AMD GPU utilization tracking added to hardware charts — reports `100%` when processing and `0%` when idle.

---

## 🔧 Changes

### Core

- **`modules/core/bootstrap.py`**: Added `_detect_amd_hardware`, `_has_amd_drm_vendor`, and dual-GPU path selection logic. AMD targets `/app/libs/amd`, NVIDIA targets `/app/libs/nvidia`. When both are detected, AMD path is loaded for ONNX Runtime while CTranslate2 binds directly to NVIDIA CUDA.
- **`modules/core/config_helpers.py`**: Added `_update_amd_state` — auto-sets `prep_device = "AMD"` when AMD hardware is detected and `MAX_AMD_UNITS >= 1`.
- **`modules/core/config.py`**: Added `MAX_AMD_UNITS` environment variable parsing and AMD unit logging.
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
- **`docker-compose.yml`**: Added optional commented device mappings and configurations for NVIDIA (CUDA), Intel (NPU/iGPU/Arc), and AMD (ROCm/DirectML) hardware acceleration. Out-of-the-box configuration runs seamlessly on default CPU environment.
- **`docker-compose.wsl.yml`**: Added WSL2-specific Docker Compose override for Windows 11 DirectX GPU bridge (`/dev/dxg`) and WSL library mounts.

### Tests

- **`tests/unit/test_config_amd.py`**: 83-line AMD hardware detection unit test file covering `_detect_amd_hardware`, `_update_amd_state`, AMD scheduler unit registration, and fallback behavior.

### Security & Access Control Module

- **`modules/api/support/security.py`**: New defense-in-depth security module implementing CORS allowlist (no wildcard by default), dual-tier API key authentication (`API_KEY` + `ADMIN_API_KEY`), anti-CSRF origin/referer verification for unauthenticated admin endpoints, model supply chain allowlisting (`is_valid_model_name`), device whitelisting, and audit logging for administrative mutations.
- **`tests/unit/test_security.py`**: 289-line security test suite covering CORS resolution, API key verification, admin auth, CSRF origin checks, model allowlist, and audit logging.

### Infrastructure

- **`docker-compose.wsl.yml`**: New WSL2-specific Docker Compose override for Windows 11 DirectX GPU bridge (`/dev/dxg`) and WSL library mounts.
- **`.gitattributes`**: Added line-ending normalization rules for cross-platform consistency.
- **Preprocessing refactor**: `modules/inference/pipeline/preprocessing/` refactored from single file to subpackage (`__init__.py`, `execution.py`, `helpers.py`, `provider.py`).
- **`modules/inference/pipeline/openvino_provider_dispatch.py`**: Extended with AMD ROCm/DirectML provider resolution, WSL2 `/dev/dxg` detection, and dictionary-based dispatch for Radon Rank A compliance.

### Documentation

- **`README.md`**: Updated description, hardware matrix, quick start compose, telemetry, and security configuration. Moved project structure tree to `docs/ARCHITECTURE.md`.
- **`docs/ARCHITECTURE.md`**: Updated silicon header, hardware matrix, transcription flow diagram, hardware interface section, security architecture section, and added full annotated project structure tree.
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
| Live runtime test (NVIDIA + AMD) | ✅ NVIDIA ASR used; AMD detected on WSL2 with UVR on CPU |

**Tested on**: Dual-GPU host (NVIDIA GeForce RTX 5090 + AMD Radeon Graphics) under Windows 11 WSL2 Docker Desktop. NVIDIA handled ASR; AMD UVR acceleration requires native Linux `/dev/kfd` (not available in this WSL2 container).
