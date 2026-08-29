# Model Lifecycle Management Skill

This skill documents the configuration, lifecycle stages, and verification methods for model pool warming, deferred timeouts, and memory offloading in Whisper Pro ASR.

## Objective

Prevent memory exhaustion and GPU/NPU/RAM leaks by configuring, verifying, and testing the system's memory release mechanisms (`AGGRESSIVE_OFFLOAD` and `MODEL_IDLE_TIMEOUT`).

---

## Lifecycle Strategies

| Mode | Environment Config | Description |
| :--- | :--- | :--- |
| **Aggressive Offload** | `AGGRESSIVE_OFFLOAD=true` | Immediately purges and unloads models from hardware RAM/VRAM when active session count reaches 0. |
| **Idle Timeout** | `MODEL_IDLE_TIMEOUT=300` | Lazily triggers a deferred `threading.Timer` on session count decrements. If a new request arrives before the timeout, the timer is cancelled and models stay warm. |

*Note*: If `MODEL_IDLE_TIMEOUT > 0`, it takes precedence over `AGGRESSIVE_OFFLOAD`.

---

## Model Download Integrity & Corruption Recovery

Every ingestion path is protected by `modules/core/model_integrity.py`. The unchecked
`_download_openvino_source()` fallback this section used to warn about is gone: it fetched
~22GB of unconverted OpenAI source weights into the directory the Intel engine reads and
reported success, leaving a tree `ov_genai.WhisperPipeline` cannot load, with no route to a
usable IR because the runtime image ships no `optimum-cli`. `scripts/preload_model.py` now
fails loudly there instead, which at build/provision time is the cheap outcome.

- **Structural & Checksum Verification**: Models are validated before use:
  - CTranslate2 (Faster-Whisper): requires `model.bin` or `model.safetensors` (>= 10 MB), `config.json`, `preprocessor_config.json`, and `tokenizer.json`.
  - OpenVINO (Intel Whisper): requires `openvino_encoder_model` and `openvino_decoder_model` XML/BIN pairs (>= 50 MB), `openvino_tokenizer` and `openvino_detokenizer` XML/BIN pairs, and `generation_config.json`.
  - UVR & Silero VAD: validated via minimum file size and SHA-256 checksums.
- **Auto-Purge & Recovery Behavior**:
  - **Faster-Whisper**: Directory-valued `model_id` paths are purged after validation failure but return False without attempting a reload; retry loading applies only to cached snapshot paths under `download_root` when `model_id` can be resolved for restoration.
  - **OpenVINO (Intel Whisper)**: If the local OpenVINO directory is corrupted, initialization purges it, re-provisions the IR through `model_provisioning.ensure_openvino_whisper` (which validates and bounded-retries), and makes one reload attempt. A failed purge, re-provision or reload preserves the original initialization failure without retrying indefinitely.
  - **UVR & Silero VAD**: Validated via SHA-256 checksums in both preload and runtime helpers with auto-purge and retry.

---

## Verification & Testing Procedure

### 1. Test Aggressive Offload

To verify that models are instantly purged when active count hits zero:

- Set up a mock pipeline that decrements standard sessions.
- Run tests in `tests/inference/runtime/test_model_manager.py` and `tests/inference/scheduler/test_scheduler.py` checking model unload/offload behavior.
- On NVIDIA hosts, verify reclaim logs show both `RAM(RSS)` and `CUDA VRAM` deltas so host-memory vs VRAM reclaim is distinguishable.

### 2. Test Idle Timeout

To test dynamic timer scheduling and cancellation:

- Set `MODEL_IDLE_TIMEOUT = 1.0` (or another small duration).
- Trigger session increment then decrement to start the timer.
- Assert the timer is actively scheduled in `STATE.idle_timer`.
- Trigger a second session increment before the timer fires, and assert that the timer is successfully cancelled.

### 3. Module Structure

All lifecycle logic lives in `modules/inference/runtime/model_manager.py` (kept under 500 lines). Key lifecycle functions:

- `_run_idle_cleanup`, `_schedule_idle_cleanup`, `_cancel_idle_cleanup` — Timer-based idle model purging.
- `unload_models` — Aggressive model purge from RAM/VRAM.
- `increment_active_session`, `decrement_active_session` — Session tracking with automatic idle scheduling.
- `wait_for_priority` — Priority task synchronization.
- Public aliases: `check_preemption`, `cancel_idle_cleanup`.

### 4. Verify Thread Safety

The lifecycle operations utilize thread locks to protect model pools from concurrent modifications:

- Ensure that if a task arrives *during* model unload execution, the unload lock prevents race conditions, allowing the unload to complete before models are reloaded on demand.

### 5. Execute Automated Verification

Run targeted end-to-end idle timeout reclamation and lifecycle tests:

```bash
docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
docker run --rm -v "$(pwd):/app" -w /app whisper-pro-asr-test python3 -m pytest tests/integration/concurrency/test_e2e_idle_timeout_reclamation.py -v
```

Or run the full Docker test pipeline wrapper:

```bash
scripts/ci/build-and-test.sh
```

Ensure per-file coverage >=90% and Rank-A complexity (Radon CC <= 5) across modified code blocks.
