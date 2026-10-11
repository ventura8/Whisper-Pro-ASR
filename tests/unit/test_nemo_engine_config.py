"""Registry, catalog and settings for the PARAKEET / CANARY engines.

Covers modules/core/engine_registry.py (the NeMo additions), modules/core/nemo_catalog.py
and modules/core/config_nemo.py, which together decide what a NeMo engine loads and where.
"""

import importlib
from unittest import mock

import pytest

from modules.core import config_nemo, engine_registry, nemo_catalog

PARAKEET = engine_registry.ENGINE_PARAKEET
CANARY = engine_registry.ENGINE_CANARY


def test_nemo_engines_come_last_so_a_missing_onnx_asr_still_falls_back_to_faster_whisper():
    """Fallback picks the first installed engine in order."""
    engines = engine_registry.supported_engines()
    assert engines[0] == engine_registry.ENGINE_FASTER_WHISPER
    assert engines[-2:] == [PARAKEET, CANARY]


def test_auto_is_unchanged():
    """AUTO still resolves to FASTER-WHISPER; the NeMo engines are opt-in."""
    assert engine_registry.AUTO_DEFAULT_ENGINE == engine_registry.ENGINE_FASTER_WHISPER


@pytest.mark.parametrize("name", ["parakeet", " CANARY "])
def test_nemo_engine_names_validate(name):
    """The names validate case- and space-insensitively."""
    assert engine_registry.is_nemo_engine(engine_registry.normalize_and_validate_engine(name))


@pytest.mark.parametrize(
    ("engine", "unit_type", "supported"),
    [(PARAKEET, "CUDA", True), (PARAKEET, "GPU", True), (CANARY, "CPU", True), (PARAKEET, "NPU", False), (CANARY, "AMD", False)],
)
def test_unit_support(engine, unit_type, supported):
    """CUDA and the Intel GPU accelerate; NPU and AMD do not."""
    assert engine_registry.engine_supports_unit(engine, unit_type) is supported


def test_only_engines_that_decode_clips_are_given_them():
    """Clips go only to engines that decode them."""
    assert engine_registry.engine_takes_clips(PARAKEET)
    assert engine_registry.engine_takes_clips(engine_registry.ENGINE_FASTER_WHISPER)
    assert not engine_registry.engine_takes_clips(engine_registry.ENGINE_WHISPERX)
    assert not engine_registry.engine_takes_clips(engine_registry.ENGINE_INTEL_WHISPER)


@pytest.mark.parametrize(
    ("engine", "unit_type", "key"),
    [
        (PARAKEET, "CUDA", "PARAKEET-CUDA"),
        (PARAKEET, "GPU", "PARAKEET-INTEL"),
        (PARAKEET, "NPU", "PARAKEET-INTEL"),
        (CANARY, "AMD", "CANARY-AMD"),
        (CANARY, "CPU", "CANARY-CPU"),
        (CANARY, "", "CANARY-CPU"),
        # Whisper engines keep one worker per engine, exactly as before.
        (engine_registry.ENGINE_FASTER_WHISPER, "CUDA", "FASTER-WHISPER"),
        (engine_registry.ENGINE_INTEL_WHISPER, "NPU", "INTEL-WHISPER"),
    ],
)
def test_worker_key_separates_onnx_runtime_builds(engine, unit_type, key):
    """NeMo workers are keyed by ONNX Runtime build; Whisper engines are not."""
    assert engine_registry.worker_key(engine, unit_type) == key


def test_installation_is_probed_through_onnx_asr():
    """Installed means onnx_asr is importable."""
    with mock.patch("importlib.util.find_spec", return_value=None) as find_spec:
        assert not engine_registry.engine_is_installed(PARAKEET)
    find_spec.assert_called_once_with("onnx_asr")


# --- catalog ---------------------------------------------------------------------


def test_every_catalog_entry_is_pinned_and_belongs_to_one_family():
    """Every model is pinned to a commit and loadable by one engine."""
    families = [name for names in nemo_catalog.FAMILY_MODELS.values() for name in names]
    assert sorted(families) == sorted(nemo_catalog.CATALOG)
    for spec in nemo_catalog.CATALOG.values():
        assert len(spec.revision) == 40, f"{spec.name} must be pinned to a commit"
        assert spec.repo_id.startswith("istupakov/")
    for engine, default in nemo_catalog.FAMILY_DEFAULT.items():
        assert default in nemo_catalog.FAMILY_MODELS[engine]


def test_canary_needs_a_language_and_has_no_token_times():
    """The two families' contracts, as measured."""
    canary = nemo_catalog.CATALOG["nemo-canary-1b-v2"]
    assert canary.needs_language
    assert canary.translates_to_english
    assert not canary.word_timestamps
    parakeet = nemo_catalog.CATALOG["nemo-parakeet-tdt-0.6b-v3"]
    assert not parakeet.needs_language
    assert parakeet.word_timestamps
    assert not parakeet.translates_to_english


def test_eu25_is_twenty_five_two_letter_codes():
    """The coverage set is the 25 languages, as ISO 639-1 codes."""
    assert len(nemo_catalog.EU25) == 25
    assert all(len(code) == 2 for code in nemo_catalog.EU25)


@pytest.mark.parametrize(
    ("name", "quantization", "expected"),
    [
        (
            "nemo-parakeet-tdt-0.6b-v3",
            None,
            ["config.json", "vocab.txt", "encoder-model.onnx", "decoder_joint-model.onnx", "encoder-model.onnx.data"],
        ),
        ("nemo-parakeet-tdt-0.6b-v3", "int8", ["config.json", "vocab.txt", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"]),
        ("nemo-canary-1b-v2", "int8", ["config.json", "vocab.txt", "encoder-model.int8.onnx", "decoder-model.int8.onnx"]),
    ],
)
def test_required_files(name, quantization, expected):
    """The files onnx-asr opens, per architecture and quantization."""
    assert nemo_catalog.required_files(nemo_catalog.CATALOG[name], quantization) == expected


def test_required_files_for_several_quantizations_is_a_union_without_repeats():
    """fp32 and int8 together need seven distinct files."""
    files = nemo_catalog.required_files_for(nemo_catalog.CATALOG["nemo-canary-1b-v2"], [None, "int8"])
    assert len(files) == len(set(files)) == 7


# --- settings --------------------------------------------------------------------


@pytest.fixture(name="reload_settings")
def _reload_settings(monkeypatch):
    """Re-import config_nemo under a given environment."""

    def _reload(**env):
        """Clear the NEMO_* environment, apply ``env``, and re-import the settings."""
        for key in ("NEMO_MODEL", "NEMO_QUANTIZATION", "NEMO_MAX_CHUNK_SEC", "NEMO_BATCH_SIZE", "NEMO_COMPANION_DEVICE"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config_nemo)

    yield _reload
    monkeypatch.undo()
    importlib.reload(config_nemo)


def test_defaults(reload_settings):
    """Unset settings take the documented defaults."""
    settings = reload_settings()
    assert (settings.NEMO_MAX_CHUNK_SEC, settings.NEMO_BATCH_SIZE) == (20.0, 4)
    assert (settings.NEMO_QUANTIZATION, settings.NEMO_COMPANION_DEVICE) == ("AUTO", "AUTO")
    assert settings.model_spec(PARAKEET).name == "nemo-parakeet-tdt-0.6b-v3"
    assert settings.model_spec(CANARY).name == "nemo-canary-1b-v2"


def test_unusable_values_fall_back_with_a_warning(reload_settings, caplog):
    """A bad value is the default plus a warning, never a startup failure."""
    settings = reload_settings(NEMO_QUANTIZATION="fp8", NEMO_MAX_CHUNK_SEC="90", NEMO_BATCH_SIZE="0", NEMO_COMPANION_DEVICE="npu")
    assert (settings.NEMO_QUANTIZATION, settings.NEMO_MAX_CHUNK_SEC, settings.NEMO_BATCH_SIZE) == ("AUTO", 20.0, 4)
    assert settings.NEMO_COMPANION_DEVICE == "AUTO"
    assert "NEMO_QUANTIZATION" in caplog.text


def test_a_model_from_the_other_family_is_refused(reload_settings, caplog):
    """A Canary name under PARAKEET is refused."""
    settings = reload_settings(NEMO_MODEL="nemo-canary-1b-v2")
    assert settings.model_spec(PARAKEET).name == "nemo-parakeet-tdt-0.6b-v3"
    assert settings.model_spec(CANARY).name == "nemo-canary-1b-v2"
    assert "is not a PARAKEET model" in caplog.text


def test_a_refused_model_is_logged_once_not_on_every_call(reload_settings, caplog):
    """/status asks for the spec on every poll; the refusal must not flood the log."""
    settings = reload_settings(NEMO_MODEL="someone/else-onnx")
    for _ in range(5):
        settings.model_spec(PARAKEET)
    assert caplog.text.count("is not a PARAKEET model") == 1


def test_a_model_outside_the_catalog_is_never_loaded(reload_settings):
    """The catalog is the allowlist."""
    settings = reload_settings(NEMO_MODEL="someone/else-onnx")
    assert settings.model_spec(PARAKEET).name == "nemo-parakeet-tdt-0.6b-v3"


def test_the_english_parakeet_can_be_chosen(reload_settings):
    """NEMO_MODEL selects the English-only Parakeet."""
    assert reload_settings(NEMO_MODEL="nemo-parakeet-tdt-0.6b-v2").model_spec(PARAKEET).languages == frozenset({"en"})


@pytest.mark.parametrize(
    ("setting", "unit_type", "expected"),
    [
        ("AUTO", "CUDA", None),
        ("AUTO", "GPU", None),
        ("AUTO", "CPU", "int8"),
        ("AUTO", "npu", "int8"),
        ("INT8", "CUDA", "int8"),
        ("FP32", "CPU", None),
    ],
)
def test_quantization_per_unit(reload_settings, setting, unit_type, expected):
    """fp32 where an accelerator runs it, int8 on the CPU provider."""
    assert reload_settings(NEMO_QUANTIZATION=setting).quantization_for(unit_type) == expected


def test_quantizations_for_a_pool_are_distinct_and_never_empty(reload_settings):
    """A pool's quantizations, once each."""
    settings = reload_settings()
    assert settings.quantizations_for([{"type": "CUDA"}, {"type": "CPU"}, {"type": "GPU"}]) == [None, "int8"]
    assert settings.quantizations_for([]) == [None]


def test_model_dir_is_under_the_cache():
    """Weights live under <cache>/nemo/<model>."""
    spec = nemo_catalog.CATALOG["nemo-canary-1b-v2"]
    assert config_nemo.model_dir("/app/model_cache", spec) == "/app/model_cache/nemo/nemo-canary-1b-v2"


def test_banner_lines_name_the_model_and_the_companion(reload_settings):
    """The banner names the model, its revision and the companion."""
    settings = reload_settings(NEMO_COMPANION_DEVICE="cpu")
    rows = dict(settings.banner_lines(CANARY, "Systran/faster-whisper-large-v3"))
    assert rows["NeMo Model"].startswith("nemo-canary-1b-v2 (25 languages, ")
    assert "on CPU" in rows["Whisper Companion"]
    assert settings.banner_lines(engine_registry.ENGINE_FASTER_WHISPER, "x") == []
