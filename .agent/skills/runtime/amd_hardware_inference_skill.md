# AMD Hardware Inference & UVR Acceleration Skill

This skill documents instructions for configuring, debugging, and testing ROCm/DirectML-based execution on AMD GPU hardware.

## Architecture Summary

- **ONNX Runtime Segregation**: AMD uses `/app/libs/amd` pointing to `onnxruntime-rocm==1.22.2.post3`, isolated from Intel OpenVINO, NVIDIA CUDA, and CPU paths.
- **Split Execution Model**: UVR vocal isolation runs on AMD GPU via ONNX ROCm / MIGraphX (`MIGraphXExecutionProvider` or `ROCMExecutionProvider` on native Linux with `/dev/kfd`). Whisper ASR inference falls back to CPU (CTranslate2 has no ROCm backend) with `int8` compute type.
- **Dual-GPU Support**: When both NVIDIA and AMD are detected, bootstrap loads `/app/libs/amd` for ONNX Runtime (used by UVR), while CTranslate2 bypasses ONNX entirely and binds directly to NVIDIA CUDA for ASR.
- **Scheduler Unit ID**: AMD GPUs register as `amd:0` (index from `MAX_AMD_UNITS`). Device type is `"AMD"`, unit name is reported from ONNX execution provider.
- **Auto-Detection Triggers**: `_detect_amd_hardware` checks `/dev/kfd` presence or AMD DRM vendor `0x1002` in `/sys/class/drm`. On WSL2 (`/dev/dxg`), mounted AMD WSL driver artifacts (`amdxc64.so`) and `librocdxg.so` are strictly required to prevent false positives when running NVIDIA or Intel containers.

## ⚠️ Critical WSL2 / Windows GPU Limitation

**UVR cannot use the AMD GPU inside a Linux Docker container on WSL2, even with `/dev/dxg` mapped.**

The reasons are:

| Path | Requirement | WSL2 Status |
|---|---|---|
| `ROCMExecutionProvider` | `/dev/kfd` (ROCm Linux kernel driver) | ❌ Not available on standard WSL2 |
| `DmlExecutionProvider` | DirectML (Windows-only API) | ❌ Not available inside Linux containers |

- `/dev/dxg` provides DirectX/DirectML access to **Windows native processes only** — not Linux processes running inside Docker.
- `onnxruntime-rocm` requires `/dev/kfd` to call `hipGetDeviceProperties()`. Without it, loading `ROCMExecutionProvider` throws a C++ `terminate()` crash (`HIP failure 100: no ROCm-capable device is detected`).
- The app correctly guards against this crash by checking `/dev/kfd` existence before selecting `ROCMExecutionProvider`, then falls back to `CPUExecutionProvider` with a clear warning log.

**When will AMD GPU work for UVR?**
- Native Linux host (bare-metal or KVM) with `/dev/kfd` + `/dev/dri` mapped in docker-compose.
- Not available on WSL2 without a custom AMD WSL2 kernel with ROCm support.

## Docker Compose Passthrough

```yaml
# Linux AMD hosts (bare-metal / KVM) - enables ROCm GPU for UVR
devices:
  - /dev/kfd:/dev/kfd        # AMD KFD (ROCm GPU driver) - REQUIRED for GPU
  - /dev/dri:/dev/dri        # DRM render nodes

# Windows 11 / WSL2 AMD hosts - enables AMD detection only, UVR runs CPU
devices:
  - /dev/dxg:/dev/dxg        # WSL GPU bridge (enables AMD detection, NOT GPU compute)
```

> **Note**: `MAX_AMD_UNITS` is no longer required to enable AMD detection. Uncomment the `devices:` section in docker-compose.yml to enable AMD hardware detection.

## Bootstrap Path Resolution Order

`_resolve_target_library` in `modules/core/bootstrap.py` selects paths as follows:

1. If both NVIDIA and AMD detected → load `/app/libs/amd` (ONNX gets ROCm; CTranslate2 gets CUDA directly)
2. If only AMD detected → load `/app/libs/amd`
3. If only NVIDIA detected → load `/app/libs/nvidia`
4. If Intel detected → load `/app/libs/intel`
5. Fallback → load `/app/libs/cpu`

## CTranslate2 Float16 Constraint

CTranslate2 raises `ValueError` if initialized on CPU with `compute_type="float16"`. The engine factory coerces float16 to int8 for CPU-mode slots:

- In `_create_faster_whisper_engine` and `_create_whisperx_engine` in `modules/inference/engines/engine_factory.py`
- This affects any unit where `_resolve_device_str(unit)` returns `"cpu"` (including AMD units)

## Dashboard Integration

- **Icon**: `bolt` (Material Icons) — set in `core/utils.js`, `features/task_filter_history.js`, `features/runtime.js`
- **Label**: `AMD GPU` — resolved in `_normalizeHardwareFamily` and `getHwIconAndLabel` in `core/utils.js`
- **Utilization Charts**: Activity-inferred via `_resolve_amd_utilization` in `metrics_discovery.py`
  - Reports `100%` when AMD tasks are active (UVR stage) or preprocessor lock is held
  - Reports `0%` when idle

## Verification & Testing Procedure

### Execute AMD Configuration Tests

Verify that AMD device detection and fallbacks work correctly:

```bash
python3 -m pytest tests/unit/test_config_amd.py
```

### Runtime Verification

After `docker compose up --build`, check startup logs for:

```text
Context: AMD ROCm -> Path: /app/libs/amd
Successfully loaded ONNX <version> from /app/libs/amd
[config] AMD Radeon Graphics → amd:0 registered
```

For UVR provider diagnostics, look for these log lines per task:

```text
# When AMD GPU provider is selected (native Linux with /dev/kfd):
[UVR] AMD GPU provider selected: ROCMExecutionProvider (device_id=0)
[System] Initializing UVR (UVR-MDX-NET-Inst_HQ_3.onnx) on AMD GPU 0... [ONNX provider: ROCMExecutionProvider]
[UVR] Starting vocal isolation on AMD GPU 0 [ONNX: ROCMExecutionProvider]...

# When AMD GPU falls back to CPU (WSL2 without /dev/kfd):
[UVR] AMD GPU fallback to CPU: /dev/dxg (DirectX/WSL2) is present but onnxruntime-rocm does not
      support DirectML in Linux containers. ROCm requires /dev/kfd which is absent on this WSL2 host.
      UVR will run on CPU.
[System] Initializing UVR (UVR-MDX-NET-Inst_HQ_3.onnx) on AMD GPU 0... [ONNX provider: CPUExecutionProvider]
[UVR] Starting vocal isolation on AMD GPU 0 [ONNX: CPUExecutionProvider]...
```

### Troubleshooting

- **`amd:0` pool empty after init**: Check that `/dev/dxg` (Windows) or `/dev/kfd`+`/dev/dri` (Linux) are mapped in docker-compose.yml.
- **UVR using CPU instead of GPU (WSL2)**: This is expected behavior inside Linux containers — DirectML is not accessible from Linux Docker containers. For 10x+ realtime AMD GPU acceleration on Windows, run natively on Windows host Python (`pip install onnxruntime-directml audio-separator` and set `ASR_DEVICE=AMD`). UVR GPU acceleration inside Docker requires native Linux with `/dev/kfd`.
- **C++ `terminate()` crash on UVR load (HIP failure 100)**: This means `ROCMExecutionProvider` was passed to `ort.InferenceSession` without `/dev/kfd`. Ensure `_get_amd_provider_order()` in `openvino_provider_dispatch.py` gates `ROCMExecutionProvider` behind `/dev/kfd` existence check.
- **`libhipfft.so.0` or `librocm_smi64.so.1` not found**: These are runtime deps of `libonnxruntime_providers_rocm.so`. The Dockerfile installs `hipfft` and `rocm-smi-lib` packages from the ROCm 6.2 apt repo, then creates compat symlinks `libamdhip64.so.7` → `.so.6` and `librocm_smi64.so.1` → `.so.7`.
- **float16 compute type error on CPU fallback**: Ensure `_create_non_intel_engine` coercion is applied (check engine_factory.py).
