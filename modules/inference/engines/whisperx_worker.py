"""Isolated WhisperX worker process entrypoint.

WhisperX 3.8.6 (the latest published release) hard-pins ``torch~=2.8.0``,
``torchaudio~=2.8.0``, ``torchvision~=0.23.0`` and ``huggingface-hub<1.0.0``,
which conflicts with the app's main ``transformers``/``huggingface-hub``/
``torch`` stack (kept on their own latest versions everywhere else). Those
packages are stateful C-extension-backed libraries (CUDA context, global
registries) that cannot be safely unloaded and reloaded from a different
path within one live interpreter, and the app already imports ``transformers``
at startup (see ``modules.core.logging_setup``) — so a ``sys.path`` swap in
the main process is not enough to get an isolated ``transformers`` import.

This module is therefore run in its own **subprocess**, started with the
``spawn`` start method so it gets a brand-new interpreter. The production
entry module (``whisper_pro_asr.py``) and package ``__init__`` files are
spawn-safe: when spawn re-imports app ``__main__`` as ``__mp_main__``, it
skips FastAPI/torch imports, and importing this module does not eagerly load
engine packages that would pull in the main stack. Before anything else is
imported for WhisperX work, ``worker_main`` prepends the segregated WhisperX
install directory (``/app/libs/whisperx`` by default, built in the
Dockerfile) to ``sys.path``, guaranteeing every subsequent import in this
process — including transitive ones pulled in by ``whisperx`` itself —
resolves against the isolated, whisperx-compatible dependency set rather
than the main environment's.

Critical: that ``sys.path`` mutation must live inside ``worker_main``, not
at module import time. ``whisperx_worker_client`` (running in the *main*
process) has to import this module too, just to get a reference to
``worker_main`` to hand to ``multiprocessing.Process(target=...)`` — and a
plain module-level ``sys.path.insert`` would fire during that import as
well, leaking the isolated path into the main process. That's what
previously caused ``torchvision`` to intermittently resolve against the
isolated (older, incompatible) copy in the main process instead of the
main environment's — do not undo this without re-checking that case.

Do not import this module for its symbols from the main process; only
``whisperx_worker_client`` should reference it, and only to pass
``worker_main`` to a new ``multiprocessing`` child process.
"""

import importlib
import itertools
import os
import pickle
import sys
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any, Optional

from modules.inference.engines import worker_runtime


def _get_whisperx():
    """Import whisperx lazily, from the isolated sys.path set up by worker_main.

    Only resolvable inside this worker subprocess (see module docstring); using
    importlib.import_module rather than a literal `import whisperx` statement
    matches this codebase's established convention for every other optional/
    hardware-specific dependency (see faster_whisper_engine.py, intel_engine.py,
    etc.), and keeps this genuinely-only-available-in-one-process import from
    being statically resolved (and reported missing) outside it.
    """
    return importlib.import_module("whisperx")


def _activate_isolated_lib_path() -> None:
    """Prepend the isolated WhisperX install dir to sys.path. Worker-process only."""
    lib_path = os.environ.get("WHISPERX_LIB_PATH", "/app/libs/whisperx")
    if os.path.isdir(lib_path) and lib_path not in sys.path:
        sys.path.insert(0, lib_path)


def worker_main(conn: Connection) -> None:
    """Blocking request/response loop. Runs entirely inside the child process."""
    _activate_isolated_lib_path()
    objects: dict[str, object] = {}
    handlers = _build_handlers(objects)

    for request in worker_runtime.iter_requests(conn):
        _send_response(conn, _dispatch(handlers, request))


def _send_response(conn: Connection, response: dict[str, Any]) -> None:
    """Send a dispatch response back to the parent, degrading to a serialized error
    reply (rather than crashing the worker's request loop) if the response itself
    turns out to contain something unpicklable -- e.g. a handler bug that returns a
    raw object instead of a handle/plain-data result."""
    try:
        conn.send(response)
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        try:
            conn.send({"id": response.get("id"), "ok": False, "error": f"Unserializable worker response: {type(exc).__name__}: {exc}"})
        except OSError:
            pass


def _dispatch(handlers: dict[str, Callable[..., Any]], request: dict[str, Any]) -> dict[str, Any]:
    """Ship every handler failure back to the parent as a wire-protocol error rather
    than crash the worker's request loop. Catches Exception broadly rather than an
    enumerated tuple of "realistic" failure types: a WhisperX/torch/CUDA handler call
    can raise types this list can't anticipate in advance (a new torch internal error,
    a third-party dependency's own exception type, etc.), and any of those escaping
    here would kill the whole worker process instead of surfacing as a normal error
    response. BaseException subclasses (KeyboardInterrupt, SystemExit) are deliberately
    not caught here and continue to propagate."""
    request_id = request.get("id")
    cmd = request.get("cmd")
    args = request.get("args", {})
    try:
        handler = handlers[cmd]
        result = handler(**args)
        return {"id": request_id, "ok": True, "result": result}
    except tuple([Exception]) as exc:
        return {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _build_handlers(objects: dict[str, object]) -> dict[str, Callable[..., Any]]:
    return {
        "load_model": lambda **kw: _load_model(objects, **kw),
        "transcribe": lambda **kw: _transcribe(objects, **kw),
        "detect_language": lambda **kw: _detect_language(objects, **kw),
        "unload_model": lambda **kw: _unload_model(objects, **kw),
        "load_audio": lambda **kw: _load_audio(objects, **kw),
        "load_align_model": lambda **kw: _load_align_model(objects, **kw),
        "align": lambda **kw: _align(objects, **kw),
        "load_diarization_pipeline": lambda **kw: _load_diarization_pipeline(objects, **kw),
        "run_diarization": lambda **kw: _run_diarization(objects, **kw),
        "assign_word_speakers": lambda **kw: _assign_word_speakers(objects, **kw),
        "release": lambda **kw: _release(objects, **kw),
        "ping": lambda **kw: "pong",
        "capabilities": lambda **kw: _capabilities(),
    }


def _capabilities() -> dict[str, Any]:
    """Report what this worker's stack can actually do.

    The image supplies the torch build (the bundled CPU copy is removed at build time so
    WhisperX reuses the CUDA one), so whether GPU is usable is a property of the worker,
    not something the parent can infer. Asking avoids the hard AssertionError that a
    CPU-only torch raises at model load when handed device="cuda".
    """
    try:
        torch = importlib.import_module("torch")
        return {"cuda": bool(torch.cuda.is_available()), "torch": torch.__version__}
    except (ImportError, AttributeError, RuntimeError) as exc:
        return {"cuda": False, "torch": None, "error": str(exc)}


#: Monotonic counter for handle generation. len(objects) is not safe here: it can repeat
#: after an entry is removed (_release/_unload_model), and combined with id(obj) -- which
#: CPython can also reuse once an object is garbage-collected -- len+id together could in
#: principle collide and silently alias two different stored objects under one handle.
_HANDLE_IDS = itertools.count()


def _put(objects: dict[str, object], obj: object) -> str:
    handle = f"h{next(_HANDLE_IDS)}"
    objects[handle] = obj
    return handle


def _load_model(objects: dict[str, object], model_id: str, device: str, compute_type: str) -> str:
    whisperx = _get_whisperx()

    model = whisperx.load_model(model_id, device=device, compute_type=compute_type)
    return _put(objects, model)


def _resolve_audio(audio_path: Optional[str], audio_array: Optional[Any]) -> Any:
    if audio_array is not None:
        return audio_array
    whisperx = _get_whisperx()

    return whisperx.load_audio(audio_path)


def _transcribe(
    objects: dict[str, object],
    model_handle: str,
    batch_size: int,
    language: Optional[str],
    task: str,
    *,
    audio_path: Optional[str] = None,
    audio_array: Optional[Any] = None,
) -> dict[str, Any]:
    model = objects[model_handle]
    audio = _resolve_audio(audio_path, audio_array)
    return model.transcribe(audio, batch_size=batch_size, language=language, task=task)


def _try_candidate_detect_language(
    candidate: Any,
    audio: Any,
) -> tuple[str, float, list[tuple[str, float]]] | None:
    if not candidate or not hasattr(candidate, "detect_language"):
        return None
    try:
        lang_code, lang_prob, all_probs_list = candidate.detect_language(audio)
        return lang_code, float(lang_prob), [(k, float(v)) for k, v in all_probs_list]
    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError):
        return None


def _fallback_transcribe_language(model: Any, audio: Any) -> tuple[str, float, list[tuple[str, float]]]:
    result = model.transcribe(audio, batch_size=1, task="transcribe")
    detected_lang = result.get("language", "en")
    return detected_lang, 1.0, [(detected_lang, 1.0)]


def _detect_language(
    objects: dict[str, object],
    model_handle: str,
    audio_path: Optional[str] = None,
    audio_array: Optional[Any] = None,
) -> tuple[str, float, list[tuple[str, float]]]:
    model = objects[model_handle]
    audio = _resolve_audio(audio_path, audio_array)

    for candidate in (getattr(model, "model", None), model):
        result = _try_candidate_detect_language(candidate, audio)
        if result is not None:
            return result

    return _fallback_transcribe_language(model, audio)


def _unload_model(objects: dict[str, object], model_handle: str) -> None:
    objects.pop(model_handle, None)


def _load_audio(objects: dict[str, object], path: str) -> str:
    whisperx = _get_whisperx()

    audio = whisperx.load_audio(path)
    return _put(objects, audio)


def _load_align_model(objects: dict[str, object], lang_code: str, device: str) -> str:
    whisperx = _get_whisperx()

    model_a, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
    return _put(objects, (model_a, metadata))


def _align(
    objects: dict[str, object],
    raw_segments: list[dict[str, Any]],
    align_handle: str,
    audio_handle: str,
    device: str,
) -> dict[str, Any]:
    whisperx = _get_whisperx()

    model_a, metadata = objects[align_handle]
    audio = objects[audio_handle]
    return whisperx.align(raw_segments, model_a, metadata, audio, device=device, return_char_alignments=False)


def _load_diarization_pipeline(objects: dict[str, object], token: str, device: str) -> str:
    # DiarizationPipeline lives in whisperx.diarize (not a "whisperx.diarization"
    # attribute -- the top-level whisperx package only lazily exposes the specific
    # functions listed in its __init__.py, not this submodule) and its WhisperX
    # 3.8.6 constructor takes `token`, not the older `use_auth_token` keyword.
    diarize_module = importlib.import_module("whisperx.diarize")
    pipeline = diarize_module.DiarizationPipeline(token=token, device=device)
    return _put(objects, pipeline)


def _run_diarization(
    objects: dict[str, object],
    pipeline_handle: str,
    audio_handle: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> str:
    pipeline = objects[pipeline_handle]
    audio = objects[audio_handle]
    diarize_segments = pipeline(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    return _put(objects, diarize_segments)


def _assign_word_speakers(
    objects: dict[str, object],
    diarize_handle: str,
    alignment_result: dict[str, Any],
) -> dict[str, Any]:
    whisperx = _get_whisperx()

    diarize_segments = objects[diarize_handle]
    return whisperx.assign_word_speakers(diarize_segments, alignment_result)


def _release(objects: dict[str, object], handle: str) -> None:
    objects.pop(handle, None)
