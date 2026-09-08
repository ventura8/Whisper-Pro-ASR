"""The transcribe settings a request resolves to, and what each consumer takes of them.

Split from model_manager, which sits at the module line cap. Nothing here touches the
scheduler or the pool: these are pure functions of the request's parameters and the
configuration, which is what makes them cheap to pin in tests.
"""

from modules.core import config


def for_decode(params: dict, was_auto_detect: bool) -> dict:
    """The transcribe settings for the decode, from the request's resolved parameters."""
    return {
        "language": params.get("language"),
        "task": params.get("task"),
        "beam_size": config.DEFAULT_BEAM_SIZE,
        "initial_prompt": params.get("initial_prompt"),
        "word_timestamps": params.get("word_timestamps"),
        "vad_parameters": {"min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS, "threshold": config.VAD_THRESHOLD},
        # `language` is always resolved by now, so only this says whether it was auto-detected:
        # such a language may be revised per window; a demanded one is honoured as given.
        "multilingual": was_auto_detect,
        "vad_filter": params.get("vad_filter"),
        # Forced transcription: runs in another language are translated (run_decoding).
        "force_transcription": bool(params.get("force_transcription")),
    }


def for_gaps(decode: dict) -> dict:
    """The decode settings gap filling passes through; the file language says whose prompt it
    holds and, with transcription forced, which gaps are translated instead."""
    keys = ("language", "task", "initial_prompt", "vad_filter", "word_timestamps", "force_transcription")
    return {key: decode[key] for key in keys}


def translates(task) -> bool:
    """Whether ``task`` is a translation, however it is spelt."""
    return str(task).lower() == "translate"


def force_transcription(override) -> bool:
    """The request's say on forced transcription (``force_transcription=``), else the deployment's."""
    return config.ASR_FORCE_TRANSCRIPTION if override is None else bool(override)
