"""Where each engine's weights live, and which engine drives which unit.

Whisper weights exist in three incompatible shapes -- a CTranslate2 directory, an OpenVINO
IR directory, and a bare model name openai-whisper resolves itself. A hybrid host runs two
engines at once and therefore needs two of them resolved simultaneously, which is why this
is a table rather than a single path.

Split out of ``config.py`` to keep that module inside the project's module-length limit.
"""

from __future__ import annotations

import os

from modules.core import engine_registry

#: openai-whisper takes a model *name* (or a .pt path), never a CTranslate2 directory.
#: Its weights are fetched by the library itself into the cache directory below.
OPENAI_WHISPER_DEFAULT_MODEL = "large-v3"


def resolve_model_paths(*, asr_engine: str, asr_env: str, model_id: str, default_model: str, ov_cache_dir: str, ov_model_path: str) -> dict:
    """Resolve the concrete weights path for ``asr_engine``, plus the per-engine table.

    Weights are provisioned at runtime into the persistent cache volume rather than baked
    into the image, so the sentinels the engine-selection stage leaves behind ("OpenVINO",
    or the default HuggingFace id) are turned into real directories here.
    """
    ct2_cache_dir = os.path.join(ov_cache_dir, "whisper")
    ov_cache_model_dir = os.path.join(ov_cache_dir, "whisper-openvino")
    openai_cache_dir = os.path.join(ov_cache_dir, "openai-whisper")

    # OV_MODEL_LEGACY ("/models/whisper-openvino") is never mounted by compose.
    resolved_ov_model_path = ov_model_path if os.path.exists(ov_model_path) else ov_cache_model_dir

    model_id = _resolve_model_id(
        model_id,
        asr_engine=asr_engine,
        asr_env=asr_env,
        default_model=default_model,
        ct2_cache_dir=ct2_cache_dir,
        ov_cache_model_dir=ov_cache_model_dir,
    )

    return {
        "MODEL_ID": model_id,
        "CT2_CACHE_DIR": ct2_cache_dir,
        "OV_CACHE_MODEL_DIR": ov_cache_model_dir,
        "OV_MODEL_PATH": resolved_ov_model_path,
        "OPENAI_WHISPER_CACHE_DIR": openai_cache_dir,
        #: Weights per engine, so a hybrid host can hand each unit the format its engine
        #: reads. Populated **only for default weights**: the comment used to say a custom
        #: ASR_MODEL keeps MODEL_ID, but the table was built unconditionally and
        #: model_id_for_engine reads it first, so in hybrid mode an operator's custom model
        #: was silently replaced by the default cache directory for every engine. Left empty
        #: for a custom model, which makes model_id_for_engine fall through to MODEL_ID.
        "MODEL_ID_BY_ENGINE": (
            {
                engine_registry.ENGINE_FASTER_WHISPER: ct2_cache_dir,
                engine_registry.ENGINE_INTEL_WHISPER: ov_cache_model_dir,
                engine_registry.ENGINE_OPENAI_WHISPER: OPENAI_WHISPER_DEFAULT_MODEL,
            }
            if asr_env == default_model
            else {}
        ),
    }


def _resolve_model_id(
    model_id: str, *, asr_engine: str, asr_env: str, default_model: str, ct2_cache_dir: str, ov_cache_model_dir: str
) -> str:
    """Turn the engine-selection stage's sentinels into a concrete weights location.

    Three shapes arrive here: the literal "OpenVINO" (handed straight to
    ov_genai.WhisperPipeline with nothing behind it), the default HuggingFace id, and a
    custom ASR_MODEL. Only the first two are rewritten; a custom model is passed through
    untouched, which is what makes an operator's own weights survive this stage.
    """
    if asr_engine == engine_registry.ENGINE_OPENAI_WHISPER and asr_env == default_model:
        # openai-whisper takes a model *name*, never a CTranslate2 directory.
        return OPENAI_WHISPER_DEFAULT_MODEL
    if model_id == "OpenVINO":
        return ov_cache_model_dir
    if model_id == default_model:
        # Default weights resolve to the cache dir the provisioner downloads into.
        return ct2_cache_dir
    return model_id


def engine_for_unit(unit: dict, *, hybrid_engines: bool, asr_engine: str) -> str:
    """Return the ASR engine that should run on ``unit``.

    Outside hybrid mode every unit runs the single resolved ASR_ENGINE, preserving the
    non-hybrid behaviour exactly. In hybrid mode each unit runs the engine native to its
    silicon, which is only safe because the engines are isolated in separate processes.
    """
    if not hybrid_engines:
        return asr_engine
    if unit.get("type") in ("GPU", "NPU"):
        return engine_registry.ENGINE_INTEL_WHISPER
    return engine_registry.ENGINE_FASTER_WHISPER


def model_id_for_engine(engine: str, *, hybrid_engines: bool, model_id: str, model_id_by_engine: dict) -> str:
    """Return the weights path ``engine`` should load."""
    if not hybrid_engines:
        return model_id
    return model_id_by_engine.get(engine, model_id)


def engines_in_use(units: list, *, hybrid_engines: bool, asr_engine: str) -> list[str]:
    """Every engine this deployment will actually load, for provisioning."""
    if not hybrid_engines:
        return [asr_engine]
    return sorted({engine_for_unit(unit, hybrid_engines=True, asr_engine=asr_engine) for unit in units})
