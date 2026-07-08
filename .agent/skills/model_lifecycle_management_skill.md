# Model Lifecycle Management Skill

This skill documents the configuration, lifecycle stages, and verification methods for model pool warming, deferred timeouts, and memory offloading in Whisper Pro ASR.

## Objective
Prevent memory exhaustion and GPU/NPU/RAM leaks by configuring, verifying, and testing the system's memory release mechanisms (`AGGRESSIVE_OFFLOAD` and `MODEL_IDLE_TIMEOUT`).

---

## Lifecycle Strategies

| Mode | Environment Config | Description |
| :--- | :--- | :--- |
| **Aggressive Offload** | `AGGRESSIVE_OFFLOAD=true` | Immediately purges and unloads models from hardware RAM when active session count reaches 0. |
| **Idle Timeout** | `MODEL_IDLE_TIMEOUT=300` | Lazily triggers a deferred `threading.Timer` on session count decrements. If a new request arrives before the timeout, the timer is cancelled and models stay warm. |

*Note*: If `MODEL_IDLE_TIMEOUT > 0`, it takes precedence over `AGGRESSIVE_OFFLOAD`.

---

## Verification & Testing Procedure

### 1. Test Aggressive Offload
To verify that models are instantly purged when active count hits zero:
- Set up a mock pipeline that decrements standard sessions.
- Run tests in `tests/inference/test_model_manager.py` and `tests/inference/test_scheduler.py` checking model unload/offload behavior.

### 2. Test Idle Timeout
To test dynamic timer scheduling and cancellation:
- Set `MODEL_IDLE_TIMEOUT = 1.0` (or another small duration).
- Trigger session increment then decrement to start the timer.
- Assert the timer is actively scheduled in `STATE.idle_timer`.
- Trigger a second session increment before the timer fires, and assert that the timer is successfully cancelled.

### 3. Verify Thread Safety
The lifecycle operations utilize thread locks to protect model pools from concurrent modifications:
- Ensure that if a task arrives *during* model unload execution, the unload lock prevents race conditions, allowing the unload to complete before models are reloaded on demand.

### 4. Execute Automated Verification
Run unit tests targeting model managers and configurations:
```bash
python3 -m pytest tests/inference/test_model_manager.py tests/inference/test_scheduler.py -k "idle_timeout or offload"
```
Ensure a perfect **10.0/10.0** Pylint score is preserved across modified code blocks.
