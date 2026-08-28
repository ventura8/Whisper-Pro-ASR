# Intel Hardware Inference Skill

This skill documents instructions for configuring, debugging, and testing OpenVINO-based ASR execution on Intel Meteor Lake NPU/GPU hardware.

## Runtime Resilience Notes

- Production runtime auto-falls back to writable `/tmp`-based paths when mounted state/cache directories are not writable, preventing UVR and telemetry failures from host ACL mismatches.
- OpenVINO device enumeration is authoritative for runnable Intel GPU/NPU scheduler units. Linux device nodes (`/dev/accel/accel0`, `/dev/dri`) are diagnostic signals only when OpenVINO reports no usable accelerator devices; do not register node-only Intel units for scheduling.
- Docker deployments must pass `/dev/dri` and `/dev/accel`. Compose uses non-root runtime (`user: "65534:65534"`) with `group_add: ["991"]` on Intel NUC hosts so Intel device-node access works without root.
- UVR OpenVINO provider options resolve preprocess targets against the OpenVINO runtime device list. Generic `GPU` requests should resolve to a concrete GPU slot in provider `device_type` (`GPU.0`, `GPU.1`, ...), while NPU slot selection is encoded via OpenVINO `load_config` using `DEVICE_ID` and provider `device_type` remains generic `NPU`, because ORT/OpenVINO session initialization rejects dotted NPU `device_type` values in this runtime.
- UVR OpenVINO first-load initialization is serialized per accelerator family (`GPU` and `NPU` lock independently) so first GPU and first NPU initialization paths do not block each other under mixed first-batch traffic.
- UVR OpenVINO provider cache is partitioned per accelerator family (`.../uvr/gpu`, `.../uvr/npu`) so first-batch GPU/NPU initialization does not contend on a shared cache path.
- UVR OpenVINO uses `num_streams=1` for preprocessing sessions to avoid first-batch GPU stall behavior under mixed GPU/NPU priority bursts.
- If OpenVINO initialization falls back to CPU for a family (`NPU` or `GPU`), a runtime circuit-breaker disables further OpenVINO attempts for that family in the current process and routes UVR directly to CPU to prevent repeated ORT provider errors.
- If ORT reports the global OpenVINO loader failure (`INTEL_OPENVINO_DIR is set but OpenVINO library wasn't able to be loaded`), runtime opens a process-wide circuit-breaker for both Intel families (`NPU` and `GPU`) immediately and aborts further OpenVINO retries in that process.
- ONNX Runtime loading is deterministic by path (`/app/libs/cpu`, `/app/libs/nvidia`, `/app/libs/intel`) to avoid transitive CPU-only ONNX packages overriding Intel OpenVINO execution.
- Runtime image policy pins OpenVINO 2026.3.1. Startup diagnostics must log ORT/OpenVINO versions, provider paths, OpenVINO devices, Intel runtime env (`INTEL_OPENVINO_DIR`, `LD_LIBRARY_PATH`, `LIBVA_DRIVER_NAME`, `ONEAPI_DEVICE_SELECTOR`, `ZE_AFFINITY_MASK`, `OCL_ICD_VENDORS`), and Linux node visibility before diagnosing Intel runtime failures. A successful hardware validation requires provider/device metadata identifying `GPU`; CPU fallback is reported as a validation failure.
- The Intel compose override passes through the runtime env knobs above and hard-sets `INTEL_OPENVINO_DIR=/opt/intel/openvino` so OpenVINO loader issues are visible immediately in banner logs.
- `INTEL_DEEP_OV_PROBE=true` enables a one-time startup probe for `GPU`, `GPU.0`, `NPU`, and `NPU.0`; keep it disabled by default unless you are actively troubleshooting target visibility.
- For explicit Intel preprocess targets (`NPU`, `GPU`, `OpenVINO`), UVR retries alternate available OpenVINO Intel devices when the requested device fails to initialize, logs any ORT CPU provider fallback, and then falls back to CPU preprocessing if Intel acceleration is unavailable. `AUTO` may select CPU when OpenVINO exposes no Intel accelerators.

## Objective

Enforce robust OpenVINO execution on long media streams by testing VAD-guided audio slicing, language identification locks, and silent region masking.

---

## Technical Flow & Configurations

- **Chunked Slicing**: Splits files longer than `INTEL_ASR_CHUNK_DURATION` (default 300 seconds) dynamically to prevent engine hangs.
- **VAD Split Identification**: Uses global Voice Activity Detection (VAD) via `find_split_points()` to split chunks precisely in speech gaps instead of hard time boundaries.
- **Language Lock**: Auto-detects the source language on the first chunk and forces it on subsequent chunks to prevent language drift.
- **Silent Masking**: Quiet chunks are skipped, while chunks containing speech are padded/masked to preserve timing alignment.

---

## Verification & Testing Procedure

### 1. Execute OpenVINO Tests

Verify that the slicing helper and chunk assembly logic are error-free:

```bash
python3 -m pytest tests/inference/engines/test_intel_engine.py
```

### 2. Verify VAD Slicing

- Validate `find_split_points()` by feeding it mock VAD speech segments.
- Assert that splits occur only within silence zones and that chunks do not exceed maximum duration limits.

### 3. Verify Silent Masking

- Verify that sections of chunks with no speech are successfully masked (zeroed out) in the numpy arrays before being sent to the inference engine.
- Verify that entirely silent chunks bypass the Whisper inference session entirely.

### 4. Verify Language Lock

- Verify that `lock_language=True` successfully propagates the detected language code of the first chunk to the remaining transcription iterations.
