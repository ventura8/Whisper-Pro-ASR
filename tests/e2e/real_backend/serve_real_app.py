"""Boots the REAL Whisper Pro ASR FastAPI app for real-backend e2e tests.

Per the test plan's "test the real system, not a stand-in" principle: this launcher
keeps routing, history_manager, telemetry_manager, settings persistence, and the
auth/admin-key middleware genuinely live. It patches only the two things that are
expensive/non-deterministic to run for real in CI — actual ASR model inference and
language-detection voting — with deterministic fakes, exactly the same seam
tests/integration/conftest.py uses for Python integration tests
(mock.patch("modules.api.routes.asr.model_manager"), etc.), just applied at process
start instead of per-test.

Usage: python tests/e2e/real_backend/serve_real_app.py
Reads WHISPER_E2E_PORT (default 9615) and WHISPER_STATE_DIR (REQUIRED -- must be set
by the caller to a fresh temp directory so history/telemetry state starts clean per
run; startup aborts if it's missing rather than silently falling through to the
app's real default state directory).
"""

from __future__ import annotations

import copy
import importlib
import os
import shutil
import time
from typing import Any, Optional

#: Kept comfortably above the dashboard's own client-side poll interval (2s,
#: see modules/monitoring/templates/dashboard/core/state.js's currentRefreshInterval)
#: so a real-backend e2e test asserting the dashboard UI shows a task as "active"
#: is guaranteed at least one full poll cycle lands while the fake task is still
#: running, rather than racing task completion against poll timing.
FAKE_TRANSCRIPTION_DELAY_SEC = float(os.environ.get("WHISPER_E2E_FAKE_DELAY_SEC", "3.0"))

_FAKE_TRANSCRIPTION_RESULT = {
    "text": "This is a real-backend e2e fixture transcription.",
    "segments": [
        {"start": 0.0, "end": 2.5, "text": "This is a real-backend e2e fixture"},
        {"start": 2.5, "end": 4.0, "text": "transcription."},
    ],
    "language": "en",
    "video_duration_sec": 4.0,
    "performance": {"queue_sec": 0.0, "isolation_sec": 0.0, "inference_sec": FAKE_TRANSCRIPTION_DELAY_SEC},
}

_FAKE_DETECTION_RESULT = {
    "confidence": 0.97,
    "detected_language": "en",
    "language": "en",
    "language_code": "en",
}


def _fake_run_transcription(
    audio_path: str,
    language: Optional[str],
    task: str,
    *,
    diarize: bool = False,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    hf_token: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    vad_filter: bool = True,
    word_timestamps: bool = False,
    batch_size: Optional[int] = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Deterministic stand-in for modules.inference.runtime.model_manager.run_transcription:
    same call signature and result shape as the real ASR inference call, with an
    artificial delay so real-backend e2e specs have a window to observe the task in
    the dashboard's 'active' state before it completes."""
    time.sleep(FAKE_TRANSCRIPTION_DELAY_SEC)
    # Deep copy: nested segments/performance objects must be independent per call so
    # concurrent or sequential requests (and any in-place downstream mutation of the
    # result) can't cross-contaminate each other via the shared module-level template.
    result = copy.deepcopy(_FAKE_TRANSCRIPTION_RESULT)
    if task == "translate":
        result["text"] = "[translated] " + result["text"]
    return result


def _fake_run_voting_detection(
    audio_path: str,
    model_manager_module: Any,
    start_time: Optional[float] = None,
) -> dict[str, Any]:
    """Deterministic stand-in for
    modules.inference.pipeline.language_detection.run_voting_detection."""
    return dict(_FAKE_DETECTION_RESULT)


def _patch_expensive_dependencies() -> None:
    from modules.inference.pipeline import language_detection
    from modules.inference.runtime import model_manager

    model_manager.run_transcription = _fake_run_transcription
    language_detection.run_voting_detection = _fake_run_voting_detection


def _isolate_from_legacy_history() -> bool:
    """history_manager.py resolves legacy-migration candidates relative to cwd
    (os.path.abspath("state")/"data") the moment it's first imported. Launched from
    the repo root, a fresh WHISPER_STATE_DIR would still get real dev/prod history
    migrated in from the repo's own data/ or state/ dirs. Chdir into the isolated
    state dir (which has neither) only for the duration of that one import — nothing
    on this box for the legacy-migration path to find there — then chdir back so the
    rest of app startup (e.g. the "static" relative-path StaticFiles mount) still
    resolves against the real repo root.

    WHISPER_STATE_DIR is required (checked by main() before this is ever called): a
    missing value here would silently skip isolation and let the app fall through to
    its real default state directory, potentially reading/writing real dev/prod
    history/telemetry during an e2e run.

    Returns whether this call created the directory -- main()'s cleanup must only
    remove it if so, otherwise an operator pointing WHISPER_STATE_DIR at an existing
    directory (e.g. the repo's own state/) would have it recursively deleted on exit."""
    state_dir = os.environ["WHISPER_STATE_DIR"]
    created_state_dir = not os.path.isdir(state_dir)
    os.makedirs(state_dir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(state_dir)
    try:
        history_manager = importlib.import_module("modules.monitoring.history_manager")
    finally:
        os.chdir(original_cwd)
    # Reference the module so the import isn't reported as unused; the point of
    # importing it here is purely its module-level legacy-path side effect above.
    # A plain `assert` is stripped under `python -O`, silently dropping the
    # reference (and the unused-import guard it provides) -- use `del` instead,
    # which always executes.
    del history_manager
    return created_state_dir


def main() -> None:
    if not os.environ.get("WHISPER_STATE_DIR"):
        raise SystemExit(
            "WHISPER_STATE_DIR is required (must point to a fresh temp directory) -- "
            "aborting before create_app() to avoid the real app reading/writing its "
            "default (potentially real dev/prod) state directory."
        )

    port = int(os.environ.get("WHISPER_E2E_PORT", "9615"))
    state_dir = os.environ["WHISPER_STATE_DIR"]

    created_state_dir = _isolate_from_legacy_history()
    try:
        _patch_expensive_dependencies()

        import uvicorn

        import whisper_pro_asr

        app = whisper_pro_asr.create_app(testing=False)
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        # Remove only this run's freshly-created state dir, so a failure during
        # dependency patching/import/create_app (not just a clean uvicorn.run
        # shutdown) still doesn't leak it, and repeated E2E runs don't leave
        # history/telemetry/log files accumulating in the host's temp directory --
        # while an operator pointing WHISPER_STATE_DIR at an existing directory
        # (e.g. the repo's own state/) never has it deleted.
        if created_state_dir:
            shutil.rmtree(state_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
