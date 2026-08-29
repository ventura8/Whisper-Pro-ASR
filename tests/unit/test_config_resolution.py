"""The engine/pool resolution contract, exercised directly as a pure function.

``resolve_engine_and_pool`` decides which engine runs, which units stay in the scheduler
pool, and which device is claimed -- and it does all of it in a fixed order, where each
stage depends on what the one before it left behind. Every other test of this behaviour
goes through ``importlib.reload(config)``, which needs a faked machine underneath it and
asserts the *outcome*; none of them pins the ordering itself, so a stage moved past another
would keep those tests green while changing what a real host resolves to.

Nothing here touches hardware, the environment, or a vendor runtime: the function takes the
machine as an argument, which is the whole reason it lives outside config.py.
"""

from __future__ import annotations

import logging
from unittest import mock

import pytest

from modules.core import config_resolution, engine_registry

logger = logging.getLogger(__name__)


def _units(*types: str) -> list[dict]:
    """A scheduler pool of one unit per type, in the order given."""
    return [{"type": t, "id": f"{t.lower()}:0", "name": f"{t} unit"} for t in types]


def _resolve(**overrides):
    """Resolve with a CUDA+Intel host and AUTO everything, unless told otherwise."""
    kwargs = {
        "asr_engine_env": "AUTO",
        "asr_device_env": "AUTO",
        "device": "CUDA",
        "hardware_units": _units("CUDA", "GPU"),
        "isolate_engines": True,
        "hybrid_env": "",
        "logger": logger,
    }
    kwargs.update(overrides)
    return kwargs["hardware_units"], config_resolution.resolve_engine_and_pool(**kwargs)


class TestTheTwoPoolSnapshots:
    """DETECTED_HARDWARE_UNITS and DETECTED_UNITS are taken at different stages, on purpose."""

    def test_the_live_pool_is_narrowed_in_place(self):
        """Callers hold this list; returning a new one would leave them on the old pool."""
        units = _units("CUDA", "GPU")
        same_list, _result = _resolve(hardware_units=units)

        assert same_list is units

    def test_a_unit_the_engine_cannot_drive_stays_in_the_pool(self):
        """It is still a scheduler slot, and still the device its vocal isolation runs on.

        CTranslate2 has no OpenVINO backend, so an Intel GPU decodes on the CPU -- but UVR
        is ONNX Runtime and reaches the iGPU, so the unit can run a whole task: isolation on
        the accelerator, decoding on the CPU. Dropping it made a hybrid NVIDIA+Intel host
        run one task at a time when it could run two, and the CPU fallback it was dropped to
        prevent is now reported per unit by execution_status rather than being silent.
        """
        units, result = _resolve(hardware_units=_units("CUDA", "GPU"))

        assert [u["type"] for u in units] == ["CUDA", "GPU"]
        assert result["DEVICE"] == "CUDA", "the reported ASR device is still one the engine can drive"
        assert "ASR on CPU" in result["ASR_ENGINE_RESOLUTION"], "the kept slot must not read as accelerated"

    def test_detected_hardware_units_survives_every_narrowing(self):
        """Code asking what the machine HAS must not read a list filtered for one engine."""
        _units_list, result = _resolve()

        assert [u["type"] for u in result["DETECTED_HARDWARE_UNITS"]] == ["CUDA", "GPU"]

    def test_detected_units_keeps_what_the_engine_filter_removed(self):
        """Preprocessing reads this: UVR is ONNX Runtime and reaches the pruned iGPU.

        Taken after the Intel/device restrictions but *before* the engine's own filter, so
        an Intel unit dropped for FASTER-WHISPER's sake is still offered to UVR.
        """
        _units_list, result = _resolve()

        assert [u["type"] for u in result["DETECTED_UNITS"]] == ["CUDA", "GPU"]

    def test_the_snapshots_are_copies_not_aliases(self):
        """Narrowing the live pool in place must not reach back into a snapshot."""
        units, result = _resolve()
        units.clear()

        assert result["DETECTED_HARDWARE_UNITS"], "the snapshot aliased the live pool"
        assert result["DETECTED_UNITS"]


class TestStageOrdering:
    """Each stage depends on what the previous one left; the order is the contract."""

    def test_a_missing_engine_falls_back_before_the_pool_is_restricted(self):
        """The Intel restriction is keyed on the REQUESTED engine, so it must run second.

        Ordered the other way, ASR_ENGINE=INTEL-WHISPER on an image with no OpenVINO
        stripped the CUDA units out first and only then discovered the engine is not
        installed -- leaving the Faster-Whisper fallback with an Intel-only pool it cannot
        drive, and a host with a working GPU decoding on the CPU.
        """
        with mock.patch.object(engine_registry, "engine_is_installed", side_effect=lambda e: e != engine_registry.ENGINE_INTEL_WHISPER):
            units, result = _resolve(asr_engine_env="INTEL-WHISPER")

        assert result["ASR_ENGINE"] == engine_registry.ENGINE_FASTER_WHISPER
        # The subject is that the CUDA unit SURVIVES; it previously read as "CUDA is the only
        # unit left", which was the old drop-what-the-engine-cannot-drive rule showing through
        # rather than anything this test is about. The Intel GPU now stays too, for UVR.
        assert "CUDA" in [u["type"] for u in units], "the CUDA unit must survive an engine that was never loadable"
        assert result["DEVICE"] == "CUDA"

    def test_the_device_follows_the_pool_when_the_pool_is_narrowed(self):
        """A device left pointing at a pruned unit named hardware the runtime cannot address.

        Two units of the same vendor family, so _restrict_pool_to_requested_device narrows
        the pool and _align_device_with_engine settles DEVICE onto what is left.
        """
        _units_list, result = _resolve(asr_engine_env="INTEL-WHISPER", device="NPU", hardware_units=_units("GPU", "NPU"))

        assert result["ASR_ENGINE"] == engine_registry.ENGINE_INTEL_WHISPER
        assert result["DEVICE"] in ("GPU", "NPU"), "DEVICE must name a unit that is still in the pool"

    def test_intel_whisper_on_a_cuda_plus_intel_host_names_the_igpu(self):
        """The regression this stage was extended for: it used to report CPU.

        ASR_DEVICE=AUTO resolves DEVICE to CUDA (the priority order puts CUDA first), then
        _restrict_pool_to_intel drops the CUDA unit without moving DEVICE with it, and
        nothing else moved it either, so DEVICE fell through to CPU while
        tasks were dispatched to the Intel GPU and genuinely ran there: create_engine routes
        on the unit, not on DEVICE, so the banner read "OpenVINO (CPU)" for work happening
        on the iGPU.
        """
        _units_list, result = _resolve(asr_engine_env="INTEL-WHISPER", device="CUDA", hardware_units=_units("CUDA", "GPU"))

        assert result["ASR_ENGINE"] == engine_registry.ENGINE_INTEL_WHISPER
        assert result["DEVICE"] == "GPU"
        assert result["ASR_DEVICE_ENV"] == "GPU"

    def test_the_npu_is_not_dropped_as_a_side_effect_of_moving_the_device(self):
        """Moving DEVICE must not read back as an operator request and narrow again.

        The rejected fix for the above had _restrict_pool_to_intel move DEVICE itself, which
        rewrites asr_device_env -- and the later _restrict_pool_to_requested_device treats
        that as the operator asking for GPU, silently dropping the NPU from the pool as a
        side effect of an internal move.
        """
        units, result = _resolve(asr_engine_env="INTEL-WHISPER", device="CUDA", hardware_units=_units("CUDA", "GPU", "NPU"))

        assert result["DEVICE"] == "GPU"
        assert [u["type"] for u in units] == ["GPU", "NPU"], "the NPU is still a schedulable Intel unit"

    def test_nothing_drivable_still_falls_back_to_the_cpu(self):
        """CTranslate2 on an Intel-only host: there is no drivable unit to move to."""
        units, result = _resolve(asr_engine_env="FASTER-WHISPER", device="GPU", hardware_units=_units("GPU"))

        assert result["DEVICE"] == "CPU"
        assert [u["type"] for u in units] == ["GPU"], "the unit stays for vocal isolation"


class TestTheResolutionString:
    """ASR_ENGINE_RESOLUTION is the operator-facing account of every decision above."""

    def test_it_records_each_stage_that_fired(self):
        """It records each stage that fired."""
        _units_list, result = _resolve(asr_engine_env="INTEL-WHISPER")

        resolution = result["ASR_ENGINE_RESOLUTION"]
        assert "explicit -> INTEL-WHISPER" in resolution
        assert "pool -> Intel GPU/NPU only" in resolution

    def test_it_is_joined_only_after_the_last_stage(self):
        """Joined early, the late stages' entries were silently dropped from it."""
        with mock.patch.object(engine_registry, "engine_is_installed", side_effect=lambda e: e != engine_registry.ENGINE_INTEL_WHISPER):
            _units_list, result = _resolve(asr_engine_env="INTEL-WHISPER")

        assert "requested engine not installed" in result["ASR_ENGINE_RESOLUTION"]

    def test_an_untouched_auto_resolution_still_names_the_engine(self):
        """An untouched auto resolution still names the engine."""
        _units_list, result = _resolve(hardware_units=_units("CUDA"))

        assert result["ASR_ENGINE_SOURCE"] == "auto"
        assert "AUTO -> FASTER-WHISPER" in result["ASR_ENGINE_RESOLUTION"]


class TestTheEnabledHybridPath:
    """Hybrid is the one mode that deliberately keeps units the resolved engine cannot drive."""

    def test_both_vendors_stay_in_the_pool(self):
        """Both vendors stay in the pool."""
        units, result = _resolve(hybrid_env="true")

        assert result["HYBRID_ENGINES"] is True
        assert [u["type"] for u in units] == ["CUDA", "GPU"], "hybrid runs each unit's own engine, so neither is pruned"
        assert "hybrid -> per-unit engines" in result["ASR_ENGINE_RESOLUTION"]

    def test_it_is_ignored_on_a_single_vendor_host(self):
        """One interpreter cannot hold both contexts, so asking cannot conjure the hardware."""
        _units_list, result = _resolve(hybrid_env="true", hardware_units=_units("CUDA"))

        assert result["HYBRID_ENGINES"] is False

    def test_it_requires_isolation(self):
        """It requires isolation."""
        _units_list, result = _resolve(hybrid_env="true", isolate_engines=False)

        assert result["HYBRID_ENGINES"] is False

    @pytest.mark.parametrize("value", ["true", "TRUE", " True ", "1", "yes", "on"])
    def test_the_env_value_is_matched_case_and_space_insensitively(self, value):
        """The env value is matched case and space insensitively."""
        _units_list, result = _resolve(hybrid_env=value)

        assert result["HYBRID_ENGINES"] is True

    @pytest.mark.parametrize("value", ["", "false", "0", "no", "off", "maybe"])
    def test_anything_else_leaves_it_off(self, value):
        """Anything else leaves it off."""
        _units_list, result = _resolve(hybrid_env=value)

        assert result["HYBRID_ENGINES"] is False


class TestAnExplicitEngineIsNotDiscardedByHybridMode:
    """Both settings are operator requests, and they contradict each other.

    Hybrid mode assigns each unit the engine native to its silicon. With an explicit
    ASR_ENGINE also set, the named engine was silently discarded on every unit --
    ``ASR_ENGINE=OPENAI-WHISPER`` plus ``HYBRID_ENGINES=true`` ran Faster-Whisper on the
    CUDA unit and Intel-Whisper on the Intel one and the requested engine nowhere, while
    the banner still reported the request as honoured. The named engine is the narrower
    instruction, so it wins.
    """

    def test_hybrid_stays_off_when_the_engine_was_named(self):
        """Hybrid stays off when the engine was named."""
        _units_list, result = _resolve(asr_engine_env=engine_registry.ENGINE_FASTER_WHISPER, hybrid_env="true")

        assert result["ASR_ENGINE_SOURCE"] == "explicit"
        assert result["HYBRID_ENGINES"] is False
        assert result["ASR_ENGINE"] == engine_registry.ENGINE_FASTER_WHISPER

    def test_the_conflict_is_recorded_in_the_resolution_line(self):
        """Half the throughput of what was asked for must never be silent."""
        _units_list, result = _resolve(asr_engine_env=engine_registry.ENGINE_FASTER_WHISPER, hybrid_env="true")

        assert "hybrid -> off (explicit" in result["ASR_ENGINE_RESOLUTION"]

    def test_hybrid_still_turns_on_when_the_engine_was_left_to_auto(self):
        """The guard must only fire on a genuine conflict."""
        _units_list, result = _resolve(asr_engine_env="AUTO", hybrid_env="true")

        assert result["ASR_ENGINE_SOURCE"] == "auto"
        assert result["HYBRID_ENGINES"] is True


class TestWhichUnitsEarnAPoolSlot:
    """A unit belongs in the pool when it accelerates *either* stage, not just decoding.

    Decoding and vocal isolation are different runtimes reaching different silicon --
    CTranslate2 and ONNX Runtime -- so a unit useless to one is routinely valuable to the
    other. The pool is what the scheduler dispatches to and what the dashboard shows, so a
    unit that can accelerate something has to be in it.
    """

    def test_an_intel_gpu_stays_for_uvr_even_though_ctranslate2_cannot_decode_on_it(self):
        """The hybrid NVIDIA+Intel case: dropping it cost a whole scheduler slot."""
        units, result = _resolve(hardware_units=_units("CUDA", "GPU"))

        assert [u["type"] for u in units] == ["CUDA", "GPU"]
        assert result["DEVICE"] == "CUDA", "the reported ASR device is still one the engine can drive"
        assert "ASR on CPU" in result["ASR_ENGINE_RESOLUTION"], "the kept slot must not read as accelerated"

    def test_a_unit_that_accelerates_neither_stage_is_dropped(self):
        """An accelerator no runtime in this image can use is a CPU slot wearing its name."""
        units, _result = _resolve(hardware_units=_units("CUDA", "TPU"))

        assert [u["type"] for u in units] == ["CUDA"]

    def test_with_nothing_accelerable_a_named_cpu_unit_is_served(self):
        """The scheduler cannot run on an empty pool, and the dashboard must show what runs.

        Leaving the unusable accelerator in the pool would have the dashboard name silicon
        that never executes anything; a CPU unit says what actually will.
        """
        units, result = _resolve(asr_engine_env="FASTER-WHISPER", device="TPU", hardware_units=_units("TPU", "VPU"))

        assert [u["type"] for u in units] == ["CPU"]
        assert result["DEVICE"] == "CPU"
        assert "pool -> CPU" in result["ASR_ENGINE_RESOLUTION"]
