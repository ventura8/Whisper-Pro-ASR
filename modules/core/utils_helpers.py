"""
Utility Helper Functions for Whisper Pro ASR
"""

import importlib
import logging
import os
import shutil
import time

from modules.core import config, process_exec

logger = logging.getLogger(__name__)


def run_ffmpeg_standardization(
    source_path,
    output_path,
    duration,
    ffmpeg_env,
    *,
    input_flags=None,
    flags=None,
    yield_cb=None,
):
    """Execute FFmpeg command with progress tracking."""
    cfg = ffmpeg_env["cfg"]
    thread_context = ffmpeg_env["thread_context"]
    standard_ffmpeg_cond = ffmpeg_env["standard_ffmpeg_cond"]
    standard_ffmpeg_state = ffmpeg_env["standard_ffmpeg_state"]
    standard_audio_flags = ffmpeg_env["standard_audio_flags"]
    standard_normalization_filters = ffmpeg_env["standard_normalization_filters"]
    format_duration = ffmpeg_env["format_duration"]

    _ = cfg
    is_priority = getattr(thread_context, "is_priority", False)
    _mark_standard_ffmpeg_start(is_priority, standard_ffmpeg_cond, standard_ffmpeg_state)

    if input_flags is None:
        input_flags = getattr(thread_context, "input_flags", None)

    try:
        if flags is None:
            flags = standard_audio_flags
        command = _build_ffmpeg_standardization_cmd(
            source_path,
            output_path,
            flags,
            standard_normalization_filters,
            input_flags,
        )
        ffmpeg_timeout = _compute_ffmpeg_timeout(duration)
        _execute_ffmpeg_with_watchdog(command, duration, ffmpeg_timeout, format_duration, yield_cb=yield_cb)
    finally:
        _mark_standard_ffmpeg_end(is_priority, standard_ffmpeg_cond, standard_ffmpeg_state)


def _mark_standard_ffmpeg_start(is_priority: bool, standard_ffmpeg_cond, standard_ffmpeg_state):
    if is_priority:
        return
    with standard_ffmpeg_cond:
        standard_ffmpeg_state["count"] += 1
        logger.debug("[Prep] Incrementing standard FFmpeg count: %d", standard_ffmpeg_state["count"])


def _mark_standard_ffmpeg_end(is_priority: bool, standard_ffmpeg_cond, standard_ffmpeg_state):
    if is_priority:
        return
    with standard_ffmpeg_cond:
        standard_ffmpeg_state["count"] = max(0, standard_ffmpeg_state["count"] - 1)
        logger.debug("[Prep] Decrementing standard FFmpeg count: %d", standard_ffmpeg_state["count"])
        standard_ffmpeg_cond.notify_all()


def _build_ffmpeg_standardization_cmd(
    source_path,
    output_path,
    flags,
    standard_normalization_filters,
    input_flags=None,
) -> list[str]:
    command = [
        "ffmpeg",
        "-threads",
        str(config.FFMPEG_THREADS),
        "-thread_queue_size",
        "2048",
        "-err_detect",
        "ignore_err",
        "-fflags",
        "+genpts+discardcorrupt",
        "-y",
        "-loglevel",
        "error",
        "-filter_threads",
        str(config.FFMPEG_THREADS),
    ]
    if config.FFMPEG_HWACCEL.lower() != "none":
        command.extend(["-hwaccel", config.FFMPEG_HWACCEL])
    input_args = []
    if input_flags:
        input_args.extend(input_flags)
    input_args.extend(["-i", source_path])
    command.extend(input_args + ["-progress", "pipe:1"] + flags + ["-af", standard_normalization_filters, output_path])
    return command


def _compute_ffmpeg_timeout(duration: float) -> float:
    return max(300.0, duration * 5.0) if duration > 0 else 300.0


def _execute_ffmpeg_with_watchdog(command, duration, ffmpeg_timeout, format_duration, yield_cb=None):
    progress_state = {
        "last_logged_pct": -10.0,
        "last_stage_pct": -1,
        "final_speed": "N/A",
        "last_yield_time": 0.0,
    }

    def _on_line(line: str):
        _update_ffmpeg_progress_state(line, duration, format_duration, progress_state, yield_cb=yield_cb)

    try:
        process_exec.run_stream(command, timeout=ffmpeg_timeout, on_line=_on_line)
        logger.info("[Prep] FFmpeg finished | Speed: %s", progress_state["final_speed"])
    except process_exec.CommandTimeoutError as exc:
        logger.warning("[Prep] FFmpeg execution exceeded timeout (%.1fs).", ffmpeg_timeout)
        raise RuntimeError(f"FFmpeg standardization timed out after {ffmpeg_timeout}s") from exc
    except process_exec.CommandExecutionError as exc:
        logger.error("[Prep] FFmpeg failed execution: %s (stderr: %s)", exc, getattr(exc, "stderr", ""))
        raise RuntimeError(f"FFmpeg failed with return code {exc.returncode}") from exc


def _update_ffmpeg_progress_state(line, duration, format_duration, state: dict, yield_cb=None):
    state["final_speed"] = _parse_ffmpeg_speed_line(line, state["final_speed"])
    if "out_time_ms=" not in line:
        return
    if duration > 0:
        stage_pct, logged_pct = _handle_duration_progress(
            line,
            duration,
            (state["last_stage_pct"], state["last_logged_pct"]),
            format_duration,
            yield_cb,
        )
        state["last_stage_pct"] = stage_pct
        state["last_logged_pct"] = logged_pct
        return
    state["last_yield_time"] = _yield_periodically_without_duration(state["last_yield_time"], yield_cb)


def _handle_duration_progress(line, duration, pct_state, format_duration, yield_cb):
    """Parse time progress and publish updates when duration is available."""
    last_stage_pct, last_logged_pct = pct_state
    try:
        time_ms = int(line.split("=")[1].strip())
        time_sec = time_ms / 1000000.0
        pct = (time_sec / duration) * 100
        new_stage_pct = _publish_ffmpeg_stage_if_needed(pct, last_stage_pct, yield_cb)
        new_logged_pct = _log_ffmpeg_progress_if_needed(pct, last_logged_pct, time_sec, duration, format_duration)
        return new_stage_pct, new_logged_pct
    except (ValueError, IndexError):
        return last_stage_pct, last_logged_pct


def _publish_ffmpeg_stage_if_needed(pct: float, last_stage_pct: float, yield_cb=None):
    pct_int = max(0, min(100, int(pct)))
    if pct_int < last_stage_pct + 5:
        return last_stage_pct
    _update_scheduler_ffmpeg_stage(pct_int)
    _run_optional_yield(yield_cb)
    return pct_int


def _update_scheduler_ffmpeg_stage(pct_int: int):
    try:
        scheduler = importlib.import_module("modules.inference.scheduler")
        scheduler.update_task_progress(None, f"FFmpeg ({pct_int}%)")
    except (ImportError, RuntimeError, AttributeError, ValueError, TypeError, KeyError):
        pass


def _log_ffmpeg_progress_if_needed(pct: float, last_logged_pct: float, time_sec: float, duration: float, format_duration):
    if pct - last_logged_pct < 10:
        return last_logged_pct
    logger.info(
        "[Prep] FFmpeg Status: %5.1f%% | %s / %s",
        pct,
        format_duration(time_sec),
        format_duration(duration),
    )
    return pct


def parse_ffmpeg_progress(process, duration, format_duration, yield_cb=None):
    """Parse FFmpeg stdout for progress updates."""
    last_logged_pct = -10
    last_stage_pct = -1
    final_speed = "N/A"
    last_yield_time = 0.0
    while True:
        line = process.stdout.readline()
        if not line:
            break

        final_speed = _parse_ffmpeg_speed_line(line, final_speed)
        if "out_time_ms=" not in line:
            continue
        if duration > 0:
            last_stage_pct, last_logged_pct = _handle_duration_progress(
                line, duration, (last_stage_pct, last_logged_pct), format_duration, yield_cb
            )
        else:
            last_yield_time = _yield_periodically_without_duration(last_yield_time, yield_cb)
    return final_speed


def _parse_ffmpeg_speed_line(line: str, current_speed: str) -> str:
    if "speed=" not in line:
        return current_speed
    try:
        return line.split("=")[1].strip()
    except (ValueError, IndexError):
        return current_speed


def _yield_periodically_without_duration(last_yield_time: float, yield_cb=None) -> float:
    current_time = time.time()
    if current_time - last_yield_time >= 1.0:
        _run_optional_yield(yield_cb)
        return current_time
    return last_yield_time


def _run_optional_yield(yield_cb=None):
    if yield_cb:
        yield_cb()


def secure_remove(file_path):
    """Safely remove a file if it exists, ignoring errors."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except tuple([Exception]):
            pass


def get_pretty_model_name(model_path):
    """Convert technical model identifiers to human-readable names."""
    if not model_path:
        return "Unknown Engine"

    name = str(model_path).rsplit("/", maxsplit=1)[-1]

    # Common mappings (Sorted by specificity to avoid partial matches)
    mappings = {
        "faster-whisper-large-v3": "Whisper Large v3",
        "faster-whisper-medium": "Whisper Medium",
        "faster-whisper-small": "Whisper Small",
        "faster-whisper-base": "Whisper Base",
        "faster-whisper-tiny": "Whisper Tiny",
        "distil-large-v3": "Distil Large v3",
        "distil-medium": "Distil Medium",
        "distil-small": "Distil Small",
        "whisper-tiny": "Whisper Tiny",
        "whisper-base": "Whisper Base",
        "whisper-small": "Whisper Small",
        "whisper-medium": "Whisper Medium",
        "whisper-large": "Whisper Large",
        "UVR-MDX-NET-Inst_HQ_3.onnx": "Vocal Isolation HQ",
        "UVR-MDX-NET-Voc_FT.onnx": "Vocal Isolation FT",
        "silero_vad.onnx": "Silero VAD",
        "whisper": "Whisper Engine",
    }

    for key, pretty in mappings.items():
        if key in name:
            return pretty

    # Cleanup path and return (handle both dashes and underscores)
    return name.replace(".onnx", "").replace("_", " ").replace("-", " ").title()


def validate_audio(file_path):
    """Checks if the audio file is valid (exists and non-empty)."""
    if not file_path or not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) == 0:
        return False
    return True


def cleanup_old_files(directory, days=7):
    """Deletes files older than specified days in a directory."""
    if not os.path.exists(directory):
        return

    now = time.time()
    cutoff = now - (days * 86400)

    for root, _, files in os.walk(directory):
        for name in files:
            _prune_file_if_old(root, name, cutoff)


def _prune_file_if_old(root: str, name: str, cutoff: float):
    file_path = os.path.join(root, name)
    try:
        if os.path.getmtime(file_path) < cutoff:
            os.remove(file_path)
            logger.debug("[System] Pruned old file: %s", name)
    except tuple([Exception]) as e:
        logger.warning("[System] Failed to prune %s: %s", name, e)


def purge_temporary_assets():
    """Purge orphaned transcription files from the temp directory."""
    temp_dir = _resolve_temp_asset_dir()
    if os.path.exists(temp_dir):
        try:
            for name in os.listdir(temp_dir):
                try:
                    _remove_temporary_asset_entry(temp_dir, name)
                except OSError as exc:
                    logger.error("[System] Cleanup failed for %s: %s", name, exc)
            logger.info("[System] Purged temporary asset cache")
        except OSError as exc:
            logger.error("[System] Cleanup failed: %s", exc)


def _resolve_temp_asset_dir() -> str:
    return os.path.abspath(config.get_temp_dir())


def _remove_temporary_asset_entry(temp_dir: str, name: str):
    path = os.path.join(temp_dir, name)
    if os.path.isfile(path) and _should_remove_temp_file(name):
        os.remove(path)
    elif os.path.isdir(path) and _should_remove_temp_dir(name):
        shutil.rmtree(path)


def _should_remove_temp_file(name: str) -> bool:
    return name.startswith(("tmp_", "upload_", "whisper_", "processed_")) or name.endswith((".wav", ".mp3", ".tmp", ".json"))


def _should_remove_temp_dir(name: str) -> bool:
    return name.startswith("whisper_") or name in ["preprocessing"]
