# Intel Hardware Inference Skill

This skill documents instructions for configuring, debugging, and testing OpenVINO-based ASR execution on Intel Meteor Lake NPU/GPU hardware.

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
python3 -m pytest tests/inference/test_intel_engine.py
```

### 2. Verify VAD Slicing
- Validate `find_split_points()` by feeding it mock VAD speech segments.
- Assert that splits occur only within silence zones and that chunks do not exceed maximum duration limits.

### 3. Verify Silent Masking
- Verify that sections of chunks with no speech are successfully masked (zeroed out) in the numpy arrays before being sent to the inference engine.
- Verify that entirely silent chunks bypass the Whisper inference session entirely.

### 4. Verify Language Lock
- Verify that `lock_language=True` successfully propagates the detected language code of the first chunk to the remaining transcription iterations.
