# Release v1.1.4 - Clean Architecture Reorganization, Endpoint-Separated Analytics, Templates Split, Plex AI Subtitle Tagging & Dependency Upgrades

This release introduces a major structural refactoring with a dedicated `modules.core` package for core runtime modules and elimination of compatibility wrappers, combined with endpoint-separated analytics metrics, modular UI template assets, and dependency upgrades. The HTTP API remains backward compatible, but Python import consumers must migrate to canonical `modules.core.*` imports.

This release also includes scheduler fairness and observability hardening: same-tier FIFO task acquisition with preserved priority yielding, a deadlock/stuck-state fix for mixed ASR and detect-language bursts on Intel accelerator setups, deterministic dashboard ordering by arrival time, and expanded regression coverage for these concurrency paths.

It also corrects Intel iGPU/Arc utilization reporting in the monitoring charts so saturated Intel GPU load now displays as `100%` instead of capping at `99%`.

The latest release validation also captured a vocal-separation liveness fix: long UVR preprocessing chunks now expose cooperative yield checkpoints earlier, reducing time-to-pause for queued priority work while keeping pause/resume semantics intact.

Subtitle output now carries Plex-compatible `<language>-ai` naming so tracks appear as `<Language> (AI)` in Plex for all transcription and translation languages, and a log-download race condition that caused Firefox to report a download failure has been resolved.

---

## 🚀 Key Improvements & Bug Fixes

### 🏗 Core Module Reorganization (Architecture Refactoring)
- **Dedicated Core Package**: Consolidated all core runtime modules (`config`, `constants`, `utils`, `logging_setup`, `bootstrap`, `subtitles`) under a new `modules.core` sub-package for logical grouping and cleaner namespace management.
- **Eliminated Compatibility Wrappers (Breaking for Python Imports)**: Removed all top-level wrapper modules that previously re-exported from `modules.core`; Python code importing `modules.config`, `modules.utils`, etc. must switch to canonical `modules.core.*` paths.
- **Simplified Import Discipline**: All application code (~80 files), tests (~25 files), and documentation now use direct imports from `modules.core`, eliminating the cognitive overhead of dual import paths and potential version mismatches.
- **Priority Test Grouping**: Consolidated priority scheduling and preemption tests under a dedicated `tests/inference/priority/` sub-package, improving test discoverability and logical organization.

### 📊 Endpoint-Separated Analytics Metrics
- **Granular Category Tracking**: Implemented separate statistics aggregation for `/asr` requests, `/detect-language` (represented internally as `detectlang`), and OpenAI-compatible `/v1/audio/...` calls.
- **Categorization Engine**: Added `categorize_task()` in `history_manager.py` to route completed tasks to their corresponding categories using endpoint path triggers, request parameters, or fallback types.
- **Automated Cache Rebuilding**: Added `rebuild_analytics_from_history()` and automated backfilling logic to reconstruct segmented category metrics dynamically from persistent `task_history.json` logs when loading legacy analytics files on startup.

### 🎨 Modular UI Asset Split for Analytics
- **Maintainability Refactoring**: Deconstructed the monolithic `analytics.html` dashboard into separate, cohesive components under a dedicated templates directory:
  - `templates/analytics.html`: Structured HTML markup utilizing dynamic inline replacement tokens.
  - `templates/analytics.css`: Sleek stylesheet variables and modern adaptive design rules.
  - `templates/analytics.js`: Unified chart render logic, data formats, and export hooks.
- **Dynamic Inlining Loader**: Refactored `analytics_ui.py` to compile and inline these static assets on-the-fly, matching the modular pattern established for the main system dashboard.
- **Dashboard Enhancements**: Added an **Endpoint Cumulative Breakdown** section with color-coded breakdown cards, custom table columns, and stacked ApexCharts series tracking counts and transcription volumes.

### 🛠 Dependency Upgrades & CI/CD
- **FastAPI Upgrade**: Bumped the minimum FastAPI dependency constraint to `>=0.138.0` in `pyproject.toml` to support the latest web service improvements.
- **Swagger UI Upgrade**: Upgraded the offline Swagger UI source asset build target to `v5.32.6` inside the `Dockerfile` to provide the newest interactive API documentation features.
- **Node.js Deprecation Patch**: Upgraded `peter-evans/dockerhub-description` workflow action from `v4` to `v5` in `.github/workflows/ci.yml`. This forces the Docker Hub description synchronization runner to target the supported Node.js 24 runtime, eliminating GitHub runner warning logs.

### 🚦 Scheduler Fairness, Priority Yielding & Dashboard Ordering
- **Same-Tier FIFO Resource Acquisition**: Introduced arrival-aware scheduling semantics so tasks only wait behind earlier tasks of the same priority tier that are still waiting for hardware.
- **No False Blocking Behind Active Tasks**: Refined `has_earlier_task()` checks to avoid stalling later tasks behind earlier tasks that already hold a different hardware unit.
- **Priority Behavior Preserved**: Detect-language tasks continue to preempt ASR under saturation, while detect-language requests themselves are processed FIFO.
- **Stuck "Waiting for Hardware" Regression Fix**: Resolved sequence-specific deadlock/stuck behavior observed in mixed ASR + detect-language bursts on Intel NPU/GPU environments.
- **Deterministic Dashboard Ordering**: Active and historical task lists are explicitly sorted by `start_time` then `task_id` in both telemetry payload and dashboard rendering.
- **Concurrency Regression Coverage**: Added/updated tests in scheduler, priority orchestration, and telemetry ordering to lock in behavior.

### 🧵 Vocal-Separation Liveness Hardening
- **Early Cooperative Yielding**: UVR preprocessing now emits a yield checkpoint at the start of each chunk as well as after completion, so long-running separation work becomes observable sooner by the scheduler.
- **Stage-Aware Pause Confirmation**: Pause confirmation skips now avoid unnecessary waiting once the priority backlog has cleared and no detect-language task remains queued.
- **Single-Unit Resume Safety**: ASR flows paused during vocal separation resume cleanly once priority work drains, without reintroducing stuck active/waiting states.
- **Thread-Local Progress Isolation**: UVR segment progress metrics are now isolated inside thread-local contexts (`utils.THREAD_CONTEXT`). This prevents concurrently borrowing priority tasks from overwriting or unpatching the standard task's progress wrapper, resolving dashboard freeze bugs on resume.

### 🧱 Concurrency-First Reliability Hardening
- **Project Priority Policy**: Concurrency correctness is now explicitly treated as priority #1 in project and developer documentation.
- **Unit-Scoped Priority Execution**: Priority orchestration is enforced as single-permit sequencing per hardware unit to reduce preemption thrash risk under high contention while allowing parallel priority tasks across free units.
- **Cooperative Yielding & Same-Tier FIFO**: Integrated cooperative yielding checks and same-tier FIFO ordering to resolve race conditions and enforce strict preemption boundaries.
- **Indefinite Waits with Periodic Logging**: Handoff and pause confirmation paths wait indefinitely with periodic logging (every 30 seconds) to survive heavy load, rather than failing on short timeouts.
- **Lock-Safe FIFO Snapshotting**: Same-tier FIFO checks were hardened to avoid nested lock-order inversion risk by using snapshot-based reads.
- **Governance and CI Guidance**: Agent skills and quality-gate docs were updated to require lock-order review, liveness tests, and synchronized concurrency docs updates for scheduler-impacting changes.
- **Immediate Fallback Release**: Priority tasks that fallback-borrow a different unit immediately release their preemption hold on their original target unit. This allows standard tasks on the target unit to resume executing immediately instead of waiting for the priority task to finish on the fallback unit.

### 🎬 Plex-Compatible AI Subtitle Tagging
- **Smart Subtitle Filenames**: All subtitle responses (SRT, VTT, TXT, TSV) are now returned with the filename pattern `<source>.<language>-ai.<format>` (e.g. `movie.en-ai.srt`, `episode.es-ai.vtt`).
- **Plex Display**: The `-ai` regional suffix uses the ISO 3166-1 country code for **Anguilla** (`AI`). Plex's regional layout parser maps this to display subtitle tracks as **`<Language> (AI)`** — e.g. `English (AI)`, `Spanish (AI)`, `French (AI)` — for every language and for both transcription and translation tasks.
- **No More `xx (Unknown)`**: Previously, unrecognized or plain subtitle filenames would fall back to `xx (Unknown)` in Plex. The new naming convention guarantees correct language identification at all times.
- **Language Auto-Detection**: The language code is resolved from the Whisper inference result, falling back to the user-specified `language` parameter, ensuring the filename always carries the real detected language — not a hardcoded default.

### 🌟 Subtitle Word Highlighting
- **Active Word Highlighting**: New `subtitle_highlight_words` parameter renders the currently-spoken word in a distinct highlight color (`#E0E0E0`) within SRT/VTT blocks, producing a karaoke-style display.
- **Auto-Enables Word Timestamps**: Enabling word highlighting automatically activates `word_timestamps=true` for correct per-word timing data.
- **Speaker-Aware**: Highlighting is applied correctly even in diarized output, preserving `[SPEAKER_XX]:` prefixes.

### 📣 Configurable Subtitle Promo Card
- **Intro Promo Card**: Prepend a configurable promo block (defaults to `"Made with Whisper Pro ASR"`) to the beginning of SRT and WebVTT outputs.
- **Configurable Screen Time**: Configurable display duration (defaults to `3.0` seconds) and customizable text can be enabled, adjusted, or completely disabled using environment variables.
- **Docker Compose Integration**: Exposed via `SUBTITLE_PROMO_ENABLED`, `SUBTITLE_PROMO_TEXT`, and `SUBTITLE_PROMO_DURATION` environment variables.

### 🛠 Log Download Reliability Fix
- **Root Cause**: The `GET /logs/download` endpoint previously returned a `FileResponse` (streaming file). Since `whisper_pro.log` is written to continuously by the application, the file could grow between the moment the `Content-Length` was computed and the moment the body was fully sent, causing Uvicorn to raise `RuntimeError: Response content longer than Content-Length` and Firefox/browsers to report a download failure.
- **Fix**: The endpoint now reads the log file atomically into memory and returns a standard `Response` object. The body size is fixed at read time, making the response length deterministic throughout the transfer.


---

## 📋 Affected Components

| Component | Change |
|:---|:---|
| `modules/config.py` → `modules/core/config.py` | Core configuration manager (hardware detection, env vars, device pools). |
| `modules/constants.py` → `modules/core/constants.py` | Static constants (`HALLUCINATION_PHRASES`). |
| `modules/utils.py` → `modules/core/utils.py` | Cross-platform media utilities, FFmpeg integration, telemetry. |
| `modules/logging_setup.py` → `modules/core/logging_setup.py` | Logging configuration and hardware diagnostic banners. |
| `modules/bootstrap.py` → `modules/core/bootstrap.py` | Hardware path patching and library redirection. |
| `modules/subtitles.py` → `modules/core/subtitles.py` | Subtitle generation and text wrapping. |
| `modules/api/routes_asr.py` | Plex AI subtitle filename tagging (`<lang>-ai.<fmt>`), `subtitle_highlight_words` parameter. |
| `modules/api/routes_system.py` | Atomic in-memory log read for `GET /logs/download` (fixes download race condition). |
| `tests/inference/test_priority_*.py` | Relocated to `tests/inference/priority/test_priority_*.py`. |
| All imports across codebase | Updated from `modules.<name>` to `modules.core.<name>` (55 files). |

---

## 🧪 Full Verification & Validation

- **606/606 Tests Passing**: Full test suite passes with 100% pass rate across unit, integration, performance, and monitoring tests.
- **Test Coverage**: 94.99% overall project coverage, exceeding the 90% build-gate threshold.
- **Pylint Rating**: Perfect **10.00/10** score with **zero suppressions** across all Python source files.
- **Ruff Compliance**: All import ordering and formatting checks pass without exceptions.
- **Build-Gate Compliance**: Full pipeline (yamllint, ruff format, ruff check, pylint, pytest, coverage) passes without warnings or exceptions.

---

## 🔄 Migration Guide for Developers

If you have custom extensions or integrations, update imports as follows:

**Before (v1.1.3 and earlier)**:
```python
from modules import config, utils, logging_setup, bootstrap, subtitles
from modules import constants
```

**After (v1.1.4+)**:
```python
from modules.core import config, utils, logging_setup, bootstrap, subtitles, constants
```

For test fixtures:
```python
# Before
from unittest.mock import patch
patch('modules.config.HARDWARE_UNITS')

# After
from unittest.mock import patch
patch('modules.core.config.HARDWARE_UNITS')
```

### Runtime Notes
HTTP API endpoints and configuration variables remain identical. Monitoring charts now report Intel GPU saturation at `100%` when the device is fully busy, and Python import usage changed due to the import-path migration even though the API surface stayed stable.

---

*For deployment and configuration instructions, refer to the [README.md](../../README.md) or [ARCHITECTURE.md](../ARCHITECTURE.md).*
