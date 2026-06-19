# Release v1.1.2 - Deferred Memory Lifecycle, Dashboard Telemetry Accuracy & Static Analysis Hardening

This patch release refactors the model memory lifecycle to a deferred-timer pattern, fixes dashboard telemetry accuracy for hardware acceleration charts, and hardens the entire codebase to a verified **10.00/10** pylint score with zero suppression directives.

## 🚀 Key Improvements & Bug Fixes

### 🧠 Deferred Memory Lifecycle Management
- **Task-Aware Cleanup Timer**: Replaced the fixed-interval polling `ModelIdleMonitor` daemon thread with a deferred `threading.Timer` that starts only after the last task completes. New incoming tasks cancel and reschedule the timer, ensuring models remain warm during bursty workloads.
- **Edge-Case Safe**: If a new task arrives while the cleanup routine is actively running (not just waiting), the system allows the cleanup to complete, then re-initializes fresh models on demand for the new task.
- **Serialized Pool Access**: Added `_POOL_LOCK` to serialize all `unload_models` and engine initialization operations, preventing race conditions during concurrent engine state transitions.

### 📊 Dashboard Telemetry Accuracy
- **Hardware Acceleration Chart Fix**: Fixed a bug where the hardware utilization chart showed `0%` on initial page load and only displayed correct values after a manual browser refresh (F5). The metrics discovery module now emits correct initial values on first poll.
- **Engine-Aware Metrics**: Fixed metrics discovery to correctly report utilization only for engines that actually run on hardware accelerators. Whisper ASR running on CPU no longer incorrectly shows GPU/NPU utilization in the dashboard chart.
- **Language Detection Task Tracking**: Extended the metrics discovery and telemetry system to correctly account for `/detect-language` tasks alongside `/asr` tasks in utilization calculations.

### 🧹 Static Analysis Hardening
- **Zero Suppression Policy**: Removed all `# pylint: disable` comments from production code. All lint compliance is achieved through genuine code improvements (specific exception types, reduced local variable counts, clean module access patterns).
- **Specific Exception Handling**: Replaced broad `except Exception` blocks with targeted `RuntimeError`, `OSError`, and `ValueError` catches throughout `model_manager.py` and `routes_system.py`.
- **Clean Import Resolution**: Resolved mock initialization ordering in `tests/conftest.py` to prevent `ImportError` during test collection when `faster_whisper` submodules are accessed.

### 🧪 Full Verification & Validation
- **100% Pass Rate**: Passed all unit and integration tests successfully.
- **Pylint**: Maintained a perfect **10.00/10** score on all repository code files with zero suppression directives.
- **Coverage**: Maintained strong project test coverage, exceeding the 90% build-gate threshold.

---
*For deployment and configuration instructions, refer to the [README.md](README.md) or [ARCHITECTURE.md](docs/ARCHITECTURE.md).*
