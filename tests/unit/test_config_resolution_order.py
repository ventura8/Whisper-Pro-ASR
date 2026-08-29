"""The order the engine/pool resolution stages run in, and what each snapshot captures.

Every stage here reads state a previous one wrote, so the order is the contract rather than
an implementation detail. Two orderings have already been wrong in ways that cost a working
accelerator: the installed-engine fallback running *after* the Intel restriction left a
Faster-Whisper fallback with an Intel-only pool it could not drive, and the two snapshots
taken at the wrong points made preprocessing inherit the ASR engine's device filter.
"""

# The unit under test is the module's internals; reaching them by name is the point.

import logging
from unittest import mock

import pytest

from modules.core import config_resolution, engine_registry

_STAGES = (
    "_select_engine",
    "_require_intel_hardware",
    "_fall_back_to_an_installed_engine",
    "_restrict_pool_to_intel",
    "_restrict_pool_to_requested_device",
    "_resolve_hybrid",
    "_keep_only_drivable_units",
    "_align_device_with_engine",
)


def _units():
    return [
        {"type": "CUDA", "id": "cuda:0", "name": "NVIDIA GPU 0"},
        {"type": "GPU", "id": "GPU.0", "name": "Intel Graphics"},
        {"type": "NPU", "id": "NPU", "name": "Intel NPU"},
    ]


def _resolve(**overrides):
    kwargs = {
        "asr_engine_env": "AUTO",
        "asr_device_env": "AUTO",
        "device": "CUDA",
        "hardware_units": _units(),
        "isolate_engines": True,
        "hybrid_env": "",
        "logger": logging.getLogger("test-resolution"),
    }
    kwargs.update(overrides)
    return kwargs, config_resolution.resolve_engine_and_pool(**kwargs)


def test_the_stages_run_in_the_documented_order():
    """Spied rather than reasoned about: each stage depends on what the previous one wrote.

    The installed-engine fallback in particular must precede both pool restrictions. Running
    it after meant ASR_ENGINE=INTEL-WHISPER on an image with no OpenVINO first stripped the
    CUDA units out of the pool and only then discovered the engine was never loadable --
    leaving the fallback engine an Intel-only pool it cannot drive, which the drivable-unit
    filter then emptied down to the CPU. A host with a working GPU decoded on the CPU
    because of an engine that could never have been used.
    """
    order = []
    with mock.patch.multiple(
        config_resolution,
        **{name: mock.DEFAULT for name in _STAGES},
    ) as patched:
        for name, stage in patched.items():
            # Default argument, not an immediately-invoked lambda: both capture `name` per
            # iteration, but only one of them is a construct pylint rejects (C3002).
            stage.side_effect = lambda *a, captured=name, **k: order.append(captured)
        config_resolution.resolve_engine_and_pool(
            asr_engine_env="AUTO",
            asr_device_env="AUTO",
            device="CUDA",
            hardware_units=_units(),
            isolate_engines=True,
            hybrid_env="",
            logger=logging.getLogger("test-resolution"),
        )

    assert order == list(_STAGES)


def test_detected_hardware_units_is_the_pool_before_any_narrowing():
    """Code asking what the machine *has* must not read a filtered list.

    Taken ahead of the Intel restriction as well, because that drops real units too -- a
    caller reading a post-restriction snapshot would conclude the CUDA card is absent on a
    machine that has one.
    """
    _kwargs, result = _resolve(asr_engine_env="INTEL-WHISPER")

    assert [unit["id"] for unit in result["DETECTED_HARDWARE_UNITS"]] == ["cuda:0", "GPU.0", "NPU"]


def test_detected_units_is_the_pool_after_the_device_filters_but_before_the_engine_filter():
    """The snapshot preprocessing reads, and the distinction is the whole point of having two.

    UVR runs on ONNX Runtime and reaches devices CTranslate2 never will, so it must not
    inherit the ASR engine's drivable-unit filter -- an Intel iGPU pruned for
    FASTER-WHISPER's sake is still a perfectly good place to run vocal isolation.
    """
    _kwargs, result = _resolve(asr_engine_env="INTEL-WHISPER", asr_device_env="GPU")

    # The Intel restriction and the explicit-device restriction have both been applied...
    assert [unit["id"] for unit in result["DETECTED_UNITS"]] == ["GPU.0"]
    # ...while the pre-restriction snapshot still holds everything detection found.
    assert len(result["DETECTED_HARDWARE_UNITS"]) == 3


def test_both_snapshots_are_copies_the_caller_cannot_corrupt():
    """They are handed out as config globals and read for the life of the process.

    Sharing dict objects with the live scheduler pool would let a later mutation of one unit
    silently rewrite what the snapshots claim detection found.
    """
    kwargs, result = _resolve()
    live_pool = kwargs["hardware_units"]

    for snapshot in (result["DETECTED_HARDWARE_UNITS"], result["DETECTED_UNITS"]):
        for unit in snapshot:
            assert not any(unit is live for live in live_pool), "snapshot shares a dict with the live pool"

    live_pool[0]["type"] = "MUTATED"
    assert all(unit["type"] != "MUTATED" for unit in result["DETECTED_HARDWARE_UNITS"])


def test_the_live_pool_is_narrowed_in_place():
    """Every other module reads this list object, so it has to be the one that changes."""
    kwargs, _result = _resolve(asr_engine_env="INTEL-WHISPER", asr_device_env="NPU")

    assert [unit["id"] for unit in kwargs["hardware_units"]] == ["NPU"]


@pytest.mark.parametrize("requested", ["GPU", "NPU"])
def test_an_explicit_intel_device_decides_which_unit_serves_the_task(requested):
    """GPU and NPU are two units behind one engine, so without this the scheduler picks either."""
    kwargs, _result = _resolve(asr_engine_env="INTEL-WHISPER", asr_device_env=requested)

    assert [unit["type"] for unit in kwargs["hardware_units"]] == [requested]


def test_the_resolution_summary_records_every_stage_that_acted():
    """Joined only after all stages have appended; joining earlier dropped later entries."""
    _kwargs, result = _resolve(asr_engine_env="INTEL-WHISPER", asr_device_env="NPU")

    summary = result["ASR_ENGINE_RESOLUTION"]
    assert "explicit -> " in summary
    assert "pool -> NPU only" in summary


def test_auto_on_a_cuda_host_resolves_to_the_single_default_engine():
    """AUTO is one engine everywhere; hardware picks the unit, not the engine."""
    _kwargs, result = _resolve()

    assert result["ASR_ENGINE"] == engine_registry.AUTO_DEFAULT_ENGINE
    assert result["ASR_ENGINE_SOURCE"] == "auto"
