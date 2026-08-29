"""The openai-whisper wrapper, and the Intel XPU decoding clamp it carries.

The guard is the reason this file needs its own tests. `intel-xpu` is the only image that
puts this engine on a GPU, and multi-candidate decoding *hangs* there rather than failing --
whisper's KV-cache does an `index_select` over the candidate dimension that Intel's XPU
backend does not implement, so a beam-search request never returns. A hang has no
traceback and no failing test; the only thing that catches a regression here is an
assertion that the parameters were rewritten before they reached the model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modules.inference.engines import openai_whisper_engine
from modules.inference.engines.openai_whisper_engine import OpenaiWhisperEngine, _is_multi_candidate


class TestIsMultiCandidate:
    """Either knob alone is enough to reach the path that hangs."""

    @pytest.mark.parametrize(("beam_size", "best_of"), [(5, None), (2, 1), (None, 5), (1, 3), (4, 4)])
    def test_more_than_one_candidate_is_detected(self, beam_size, best_of):
        """More than one candidate is detected."""
        assert _is_multi_candidate(beam_size, best_of) is True

    @pytest.mark.parametrize(("beam_size", "best_of"), [(None, None), (1, 1), (0, 0), (1, None), (None, 1)])
    def test_greedy_settings_are_not_multi_candidate(self, beam_size, best_of):
        """Greedy settings are not multi candidate."""
        assert _is_multi_candidate(beam_size, best_of) is False

    def test_best_of_alone_still_counts(self):
        """whisper's temperature-fallback loop raises best_of on its own, so a request that
        only set beam_size=1 can still reach the multi-candidate path."""
        assert _is_multi_candidate(1, 5) is True


@pytest.fixture(name="engine")
def _engine(monkeypatch):
    """Build the engine against a fake `whisper` module, without loading a model."""

    def make(device: str):
        """Build the object under test with the given settings."""
        loaded = {}
        fake_whisper = SimpleNamespace(load_model=lambda model_id, device, download_root: loaded.setdefault("m", SimpleNamespace()))
        fake_config = SimpleNamespace(OPENAI_WHISPER_CACHE_DIR="/app/model_cache/openai-whisper")
        monkeypatch.setattr(
            openai_whisper_engine.importlib,
            "import_module",
            lambda name: fake_whisper if name == "whisper" else fake_config,
        )
        return OpenaiWhisperEngine("large-v3", device=device)

    return make


class TestConstruction:
    """construction."""

    def test_weights_download_into_the_persistent_cache_not_home(self, monkeypatch):
        """The library's default lives under HOME, which is not a mounted volume, so every
        container restart would re-fetch multiple gigabytes."""
        captured = {}

        def load_model(model_id, device, download_root):
            """Record (or fail) the model load."""
            captured.update(model_id=model_id, device=device, download_root=download_root)
            return SimpleNamespace()

        monkeypatch.setattr(
            openai_whisper_engine.importlib,
            "import_module",
            lambda name: (
                SimpleNamespace(load_model=load_model)
                if name == "whisper"
                else SimpleNamespace(OPENAI_WHISPER_CACHE_DIR="/app/model_cache/openai-whisper")
            ),
        )
        OpenaiWhisperEngine("large-v3", device="cuda")

        assert captured == {"model_id": "large-v3", "device": "cuda", "download_root": "/app/model_cache/openai-whisper"}


class TestTheXpuDecodingGuard:
    """`_clamp_multi_candidate_decoding` rewrites params in place, before they reach the model."""

    def test_multi_candidate_decoding_is_forced_to_greedy_on_xpu(self, engine, caplog):
        """Multi candidate decoding is forced to greedy on xpu."""
        params = {"beam_size": 5, "best_of": 5, "language": "en"}
        with caplog.at_level("WARNING"):
            engine("xpu")._clamp_multi_candidate_decoding(params)

        assert params["beam_size"] is None
        assert params["best_of"] is None
        assert params["language"] == "en", "only the decoding knobs are touched"
        assert "Intel XPU" in caplog.text, "silently changing the request would hide it from the operator"

    def test_greedy_params_are_left_alone_on_xpu(self, engine):
        """Greedy params are left alone on xpu."""
        params = {"beam_size": 1, "best_of": None}
        engine("xpu")._clamp_multi_candidate_decoding(params)
        assert params == {"beam_size": 1, "best_of": None}

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_other_devices_keep_beam_search(self, device, engine):
        """The hang is specific to Intel's XPU backend; CUDA and CPU beam search are fine,
        and quietly degrading them would cost accuracy for no reason."""
        params = {"beam_size": 5, "best_of": 5}
        engine(device)._clamp_multi_candidate_decoding(params)
        assert params == {"beam_size": 5, "best_of": 5}

    def test_absent_knobs_on_xpu_are_not_invented(self, engine):
        """Absent knobs on xpu are not invented."""
        params = {"language": "fr"}
        engine("xpu")._clamp_multi_candidate_decoding(params)
        assert params == {"language": "fr"}
