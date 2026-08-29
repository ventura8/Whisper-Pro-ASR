"""Child-side host for an isolated ASR engine.

Serves transcription, translation and language detection -- everything the model does
on the accelerator -- for one engine type. One worker process per *engine*, holding the
same ``unit_id -> engine`` pool the parent used to keep, so isolation relocates resident
models instead of duplicating them.

**Spawn safety and device isolation.** Nothing heavier than the standard library is
imported at module import time. The engine stack is imported lazily inside
:func:`_load_model`, *after* that request's ``env`` overrides are applied, which is what
lets an Intel worker start with ``CUDA_VISIBLE_DEVICES=""`` and therefore never create a
CUDA context at all. Importing eagerly would initialise ``modules.core.config`` -- and
with it hardware detection -- before the isolation could take effect.

**No audio crosses the boundary.** Callers pass paths; decoding happens here. Results
stream back as small plain dicts, so a two-hour transcription never materialises as a
list in either process.
"""

import importlib
import logging
import os
from collections.abc import Iterator
from multiprocessing.connection import Connection
from typing import Any, Optional

from modules.inference.engines import worker_runtime

logger = logging.getLogger(__name__)

#: unit_id -> engine instance. The worker's entire resident footprint.
_ENGINES: dict[str, Any] = {}


def _apply_env(env: Optional[dict[str, str]]) -> None:
    """Apply device-visibility overrides before any runtime is imported."""
    for key, value in (env or {}).items():
        os.environ[key] = value


def _engine_device(handle: str) -> Optional[str]:
    """The device the loaded engine settled on, as the engine itself reports it.

    The parent cannot infer this. An engine may move itself after construction --
    IntelWhisperEngine rewrites its device to CPU when the NPU warmup fails -- and the
    torch engines resolve their device from whichever build the image ships, so
    OPENAI-WHISPER on an Intel unit is the XPU on the intel-xpu image and the CPU
    everywhere else. Only the object in this process knows which happened.
    """
    engine = _ENGINES.get(handle)
    return None if engine is None else getattr(engine, "device", None)


def _load_model(engine_type: str, model_id: str, unit: dict, env: Optional[dict] = None) -> str:
    """Instantiate an engine for ``unit`` and return its handle (the unit id)."""
    handle = unit["id"]
    if handle in _ENGINES:
        return handle

    _apply_env(env)
    # Deferred so _apply_env lands first -- see module docstring. importlib rather than a
    # function-level `import` statement, which would need a ruff PLC0415 and a pylint
    # import-outside-toplevel suppression; this repository bans both.
    engine_factory = importlib.import_module("modules.inference.engines.engine_factory")

    logger.info("[InferenceWorker] Loading %s (%s) on %s", engine_type, model_id, unit.get("name", handle))
    _ENGINES[handle] = engine_factory.create_engine(engine_type, model_id, unit)
    return handle


def _get_engine(handle: str):
    engine = _ENGINES.get(handle)
    if engine is None:
        raise KeyError(f"No engine loaded for handle '{handle}'")
    return engine


def _unload_model(handle: str) -> bool:
    engine = _ENGINES.pop(handle, None)
    if engine is None:
        return False
    if hasattr(engine, "unload"):
        engine.unload()
    return True


def _unload_all() -> int:
    handles = list(_ENGINES)
    for handle in handles:
        _unload_model(handle)
    return len(handles)


def _loaded_handles() -> list[str]:
    return list(_ENGINES)


# --- normalisation to plain, picklable payloads ----------------------------------


def _segment_to_dict(segment: Any) -> dict[str, Any]:
    """Flatten an engine segment. Engines yield dataclasses or Namespaces; both duck-type."""
    return {
        "start": float(getattr(segment, "start", 0.0) or 0.0),
        "end": float(getattr(segment, "end", 0.0) or 0.0),
        "text": getattr(segment, "text", "") or "",
        "words": _words_to_list(getattr(segment, "words", None)),
    }


def _num(obj: Any, name: str) -> float:
    """A float attribute, treating a missing or None value as 0.0.

    Engines disagree about absent fields: some omit the attribute, some set it to None.
    Folding both into one helper is what keeps the two converters below branch-free --
    every `or 0.0` inline was a separate decision point.
    """
    return float(getattr(obj, name, 0.0) or 0.0)


def _word_text(word: Any) -> str:
    """A word's text under either spelling engines use for it."""
    return getattr(word, "word", None) or getattr(word, "text", None) or ""


def _word_to_dict(word: Any) -> dict[str, Any]:
    """One word-level timing entry, in the shape the parent expects."""
    return {
        "start": _num(word, "start"),
        "end": _num(word, "end"),
        "word": _word_text(word),
        "probability": _num(word, "probability"),
    }


def _words_to_list(words: Any) -> Optional[list[dict[str, Any]]]:
    if not words:
        return None
    return [_word_to_dict(word) for word in words]


def _language_probs(info: Any) -> Optional[list[tuple[str, float]]]:
    """The full per-language probability list, when the engine reported one."""
    all_probs = getattr(info, "all_language_probs", None)
    if not all_probs:
        return None
    return [(str(code), float(prob)) for code, prob in all_probs]


def _info_to_dict(info: Any) -> dict[str, Any]:
    return {
        "language": getattr(info, "language", None) or "en",
        "language_probability": _num(info, "language_probability"),
        "duration": _num(info, "duration"),
        "all_language_probs": _language_probs(info),
    }


# --- streaming commands -----------------------------------------------------------


def _transcribe(handle: str, audio_path: str, params: Optional[dict] = None) -> Iterator[dict[str, Any]]:
    """Stream ``info`` first, then one event per segment.

    ``info`` leads because the parent needs the detected language and duration before it
    starts consuming segments, exactly as the in-process contract provides it.
    """
    engine = _get_engine(handle)
    segments, info = engine.transcribe(audio_path, **(params or {}))
    yield {"event": "info", "info": _info_to_dict(info)}
    for segment in segments:
        yield {"event": "segment", "segment": _segment_to_dict(segment)}


def _detect_language_batch(handle: str, audio_path: str, segment_count: int) -> Iterator[dict[str, Any]]:
    """Scan up to ``segment_count`` 30s windows, streaming one result per window.

    This mirrors ``language_detection_core._detect_segments`` but emits progress as
    events rather than writing to the parent's scheduler, which does not exist here. The
    audio is decoded once, in this process, so no arrays cross the pipe.
    """
    engine = _get_engine(handle)
    vad = importlib.import_module("modules.inference.pipeline.vad")
    run_language_detection_core = importlib.import_module("modules.inference.pipeline.language_detection_core").run_language_detection_core

    full_audio = vad.decode_audio(audio_path)
    try:
        segment_len = int(30 * 16000)
        for index in range(segment_count):
            start = index * segment_len
            if start >= len(full_audio):
                break
            # Bound hoisted out of the subscript: ruff formats a complex slice with spaces
            # around the colon, which flake8 then flags as E203. A plain name needs none.
            end = min(start + segment_len, len(full_audio))
            chunk = full_audio[start:end].copy()
            yield {
                "event": "detection",
                "index": index,
                "result": run_language_detection_core(engine, chunk, skip_vad=False),
            }
    finally:
        del full_audio


def _detect_language(handle: str, audio_path: str) -> dict[str, Any]:
    """Single-window detection, kept for interface parity with the in-process engine.

    Takes a path rather than samples so the decode happens here; the batch stream above
    is the path the transcription pipeline actually uses.
    """
    engine = _get_engine(handle)
    vad = importlib.import_module("modules.inference.pipeline.vad")

    audio = vad.decode_audio(audio_path)
    try:
        language, probability, all_probs = engine.detect_language(audio)
    finally:
        del audio
    return {
        "language": language,
        "probability": float(probability or 0.0),
        "all_probs": [(str(code), float(prob)) for code, prob in (all_probs or [])],
    }


def worker_main(conn: Connection) -> None:
    """Entry point for the spawned process."""
    worker_runtime.configure_worker_logging("worker")
    worker_runtime.serve(
        conn,
        handlers={
            "load_model": _load_model,
            "engine_device": _engine_device,
            "unload_model": _unload_model,
            "unload_all": _unload_all,
            "loaded_handles": _loaded_handles,
            "detect_language": _detect_language,
        },
        stream_handlers={
            "transcribe": _transcribe,
            "detect_language_batch": _detect_language_batch,
        },
    )
