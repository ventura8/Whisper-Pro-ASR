"""Engine-selection decisions, as pure functions.

These live outside ``config.py`` so they can be tested directly. Everything here is a
decision about *which* engine runs where; nothing in this module touches hardware, reads
the environment, or imports a vendor runtime, so a test can exercise every branch without
the machine it runs on changing the answer.
"""

from __future__ import annotations

from modules.core import engine_registry

#: Values of HYBRID_ENGINES that switch it on. Anything else -- unset, "false", or an
#: unrecognised word -- leaves it off, which is the default.
_FORCE_WORDS = ("true", "1", "yes", "on")


def resolve_hybrid_engines(
    *,
    isolate_engines: bool,
    has_intel_unit: bool,
    has_cuda_or_amd_unit: bool,
    env_value: str,
) -> bool:
    """Whether each unit should run its own native engine in its own worker process.

    **Off unless asked for.** Hybrid mode exists to give each accelerator the engine
    native to its silicon, which necessarily means the engine -- and therefore the
    decoding behaviour and the transcript -- varies with whichever unit a request lands
    on. AUTO now resolves to one engine on every host (see
    engine_registry.AUTO_DEFAULT_ENGINE), and a default that quietly reintroduced
    per-unit engines would contradict that. An operator who wants both accelerators
    running their own engines asks for it with ``HYBRID_ENGINES=true``.

    Enabling it still requires the hardware to support it: isolation, plus one unit of
    each vendor family. No override can satisfy those, because one interpreter genuinely
    cannot hold a CUDA context and an OpenVINO GPU context, nor a CUDA torch build and a
    ROCm one -- so ``HYBRID_ENGINES=true`` on a single-vendor host is simply ignored.

    ``env_value`` is matched case-insensitively.
    """
    normalized = (env_value or "").strip().lower()
    if normalized not in _FORCE_WORDS:
        return False
    return bool(isolate_engines and has_intel_unit and has_cuda_or_amd_unit)


def hybrid_is_possible(*, isolate_engines: bool, has_intel_unit: bool, has_cuda_or_amd_unit: bool) -> bool:
    """Whether the host could run hybrid engines, ignoring preference and overrides."""
    return bool(isolate_engines and has_intel_unit and has_cuda_or_amd_unit)


def hybrid_blocked_reason() -> str:
    """Say why hybrid mode is off on a host that could otherwise run it."""
    return "it is off by default; every unit runs the resolved ASR_ENGINE"


def _select_engine(state: dict, *, hardware_units: list, logger) -> None:
    """Resolve ASR_ENGINE from AUTO or an explicit request, recording how it was reached."""
    if state["asr_engine_env"] == "AUTO":
        state["engine_source"] = "auto"
        engine, tier = engine_registry.resolve_auto_engine(hardware_units, state["asr_device_env"])
        state["engine"] = engine
        state["parts"].append(f"AUTO -> {engine} ({tier})")
        if state["asr_device_env"] == "AUTO":
            state["device"] = engine_registry.resolve_auto_device(hardware_units)
        logger.info(
            "ASR_ENGINE=AUTO resolved to %s using hardware tier %s (ASR_DEVICE=%s; order: CUDA > AMD > GPU > NPU > CPU)",
            engine,
            tier,
            state["asr_device_env"],
        )
        return
    state["engine_source"] = "explicit"
    state["engine"] = engine_registry.normalize_and_validate_engine(state["asr_engine_env"])
    state["parts"].append(f"explicit -> {state['engine']}")
    if state["asr_device_env"] == "AUTO" and state["engine"] == engine_registry.ENGINE_INTEL_WHISPER:
        state["device"] = engine_registry.resolve_auto_device(hardware_units)


def _require_intel_hardware(state: dict, *, hardware_units: list, logger) -> None:
    """INTEL-WHISPER needs Intel silicon; without it use Faster-Whisper, not OpenVINO CPU."""
    if state["engine"] != engine_registry.ENGINE_INTEL_WHISPER:
        return
    if any(unit.get("type") in ["GPU", "NPU"] for unit in hardware_units):
        return
    logger.warning("INTEL-WHISPER requested but no Intel GPU/NPU available. Falling back to FASTER-WHISPER.")
    state["engine"] = engine_registry.ENGINE_FASTER_WHISPER
    state["parts"].append(f"fallback -> {state['engine']} (no Intel GPU/NPU)")


def _unit_ids(units: list) -> str:
    """Comma-separated unit ids, for a log line naming what was kept or dropped."""
    return ", ".join(unit.get("id", "?") for unit in units) or "empty"


def _narrow_pool(state: dict, hardware_units: list, keep: list, *, logger, message: str, part: str) -> bool:
    """Narrow the pool to ``keep``, logging what was dropped and recording the change.

    Shared by the three narrowing stages, which all do the same three things: replace the
    live pool, say which units went, and add a line to the resolution summary. Returning
    whether anything happened lets a caller do follow-up work only on a real change.

    Never empties the pool and never "narrows" to everything: a scheduler with no units
    cannot serve at all, which is strictly worse than a unit the engine drives poorly, and
    a no-op narrowing would log a drop that did not occur.
    """
    if not keep or len(keep) == len(hardware_units):
        return False
    dropped = [unit for unit in hardware_units if unit not in keep]
    logger.info(message, _unit_ids(keep), _unit_ids(dropped))
    hardware_units[:] = keep
    state["parts"].append(part)
    return True


def _restrict_pool_to_intel(state: dict, *, hardware_units: list, logger) -> None:
    """Keep a non-Intel unit from being handed an OpenVINO IR only the Intel engine reads.

    Leaving CUDA/AMD/CPU units in the pool let the scheduler give this task a non-Intel
    slot, where the factory silently degraded to Faster-Whisper, handed the OpenVINO
    directory to CTranslate2 -- which failed, decided the directory was a corrupt CT2
    model, and deleted the weights out from under the Intel unit.
    """
    if state["engine"] != engine_registry.ENGINE_INTEL_WHISPER:
        return
    _narrow_pool(
        state,
        hardware_units,
        [unit for unit in hardware_units if unit.get("type") in ("GPU", "NPU")],
        logger=logger,
        message="INTEL-WHISPER: restricting scheduler pool to Intel GPU/NPU units (%s); dropped non-Intel units: %s",
        part="pool -> Intel GPU/NPU only",
    )


def _restrict_pool_to_requested_device(state: dict, *, hardware_units: list, logger) -> None:
    """Make an explicit Intel ASR_DEVICE actually decide which unit runs the task.

    Scoped to Intel units on purpose. CUDA and AMD units stay in the pool even when the
    other is requested, because hybrid engines exist precisely so each unit can run its
    native engine. Intel is the case that breaks: GPU and NPU are two units behind one
    OpenVINO engine, so the scheduler will happily serve an NPU request from the iGPU while
    every log line still says the request was honoured.
    """
    requested = state["asr_device_env"]
    if requested not in ("GPU", "NPU"):
        return
    matching = [unit for unit in hardware_units if unit.get("type") == requested]
    if not matching:
        _warn_requested_device_absent(requested, hardware_units, logger=logger)
        return
    _narrow_pool(
        state,
        hardware_units,
        matching,
        logger=logger,
        message=f"ASR_DEVICE={requested}: restricting scheduler pool to %s; dropped: %s",
        part=f"pool -> {requested} only",
    )


def _warn_requested_device_absent(requested: str, hardware_units: list, *, logger) -> None:
    """Say that an explicit ASR_DEVICE matched nothing, rather than implying it took effect.

    Not fatal: DEVICE still applies, and CPU in particular has no pool entry on accelerator
    hosts. The request simply cannot narrow a pool that contains no such unit.
    """
    logger.warning(
        "ASR_DEVICE=%s was requested but no unit of that type is in the pool (%s). The scheduler will use the pool as-is.",
        requested,
        _unit_ids(hardware_units),
    )


def _resolve_hybrid(state: dict, *, hardware_units: list, logger) -> None:
    """Decide hybrid mode and say why, so half throughput is never silent."""
    has_intel = any(unit.get("type") in ("GPU", "NPU") for unit in hardware_units)
    has_cuda_or_amd = any(unit.get("type") in ("CUDA", "AMD") for unit in hardware_units)
    possible = hybrid_is_possible(isolate_engines=state["isolate_engines"], has_intel_unit=has_intel, has_cuda_or_amd_unit=has_cuda_or_amd)
    state["hybrid"] = resolve_hybrid_engines(
        isolate_engines=state["isolate_engines"],
        has_intel_unit=has_intel,
        has_cuda_or_amd_unit=has_cuda_or_amd,
        env_value=state["hybrid_env"],
    )
    if state["hybrid"]:
        logger.info(
            "Hybrid engines enabled: CUDA/AMD units run %s, Intel GPU/NPU units run %s, each in its own worker process.",
            engine_registry.ENGINE_FASTER_WHISPER,
            engine_registry.ENGINE_INTEL_WHISPER,
        )
        state["parts"].append("hybrid -> per-unit engines")
        return
    if possible:
        logger.info(
            "Hybrid engines are possible on this host but not enabled: %s. Set HYBRID_ENGINES=true to use both.",
            hybrid_blocked_reason(),
        )


def _fall_back_to_an_installed_engine(state: dict, *, logger) -> None:
    """Requesting an engine the image does not ship must not serve errors for every request.

    ASR_ENGINE=WHISPERX on the nvidia image, which contains no whisperx, produced a service
    that started healthy and returned 139 failures in 13 seconds.
    """
    if engine_registry.engine_is_installed(state["engine"]):
        return
    missing = engine_registry.ENGINE_REQUIRED_MODULE.get(state["engine"], "?")
    logger.error("ASR_ENGINE=%s is not installed in this image (missing module '%s').", state["engine"], missing)
    alternatives = [e for e in engine_registry.SUPPORTED_ASR_ENGINES if engine_registry.engine_is_installed(e)]
    if not alternatives:
        logger.error("No ASR engine is installed in this image; every request will fail.")
        return
    logger.error("Falling back to %s. ASR_ENGINE=%s was NOT honoured; use an image that ships it.", alternatives[0], state["engine"])
    state["engine"] = alternatives[0]
    state["parts"].append(f"fallback -> {state['engine']} (requested engine not installed)")


def _keep_only_drivable_units(state: dict, *, hardware_units: list, logger) -> None:
    """Drop units the resolved engine cannot drive, and move DEVICE to follow the pool.

    A unit the engine cannot drive is worse than no unit: the scheduler dispatches to it,
    the engine degrades to the CPU, and the logs still name the accelerator. Hybrid mode is
    exempt because there each unit runs its own native engine by design.
    """
    if state["hybrid"] or len(hardware_units) <= 1:
        return
    narrowed = _narrow_pool(
        state,
        hardware_units,
        _drivable_units(state["engine"], hardware_units),
        logger=logger,
        message=f"{state['engine']} kept %s in the scheduler pool; dropped units it cannot drive: %s",
        part=f"pool -> units {state['engine']} can drive",
    )
    if narrowed:
        _move_device_into_pool(state, hardware_units, logger=logger)


def _drivable_units(engine: str, hardware_units: list) -> list:
    """The units ``engine`` can actually execute on, per the runtime's own support table."""
    return [unit for unit in hardware_units if engine_registry.engine_supports_unit(engine, unit.get("type", ""))]


def _move_device_into_pool(state: dict, hardware_units: list, *, logger) -> None:
    """Point DEVICE at a unit that is still in the pool after a narrowing.

    DEVICE has to follow the pool. Filtering units while leaving DEVICE pointing at a device
    that is no longer there produced "ASR Runtime: OpenVINO (CUDA)" -- a runtime paired with
    hardware it cannot address, on a machine whose pool was by then Intel-only.
    """
    if any(u.get("type") == state["device"] for u in hardware_units):
        return
    previous = state["device"]
    state["device"] = hardware_units[0].get("type", previous)
    state["asr_device_env"] = state["device"]
    logger.info("ASR_DEVICE moved from %s to %s: %s cannot drive %s.", previous, state["device"], state["engine"], previous)


def _align_device_with_engine(state: dict, *, logger) -> None:
    """DEVICE must never name hardware the resolved engine cannot actually drive.

    _keep_only_drivable_units moves DEVICE when it prunes the pool, but it deliberately
    does nothing when there is only one unit or when *nothing* is drivable -- emptying the
    scheduler pool would be worse. That leaves the case this covers: an Intel-only host
    running the default engine, where CTranslate2 has no OpenVINO backend at all. DEVICE
    stayed "NPU", the banner announced an NPU, and every request decoded on the CPU.

    The unit stays in the pool, because it is still a perfectly good place to run vocal
    isolation -- UVR is ONNX Runtime and reaches devices CTranslate2 never will. Only the
    ASR device claim is corrected, and loudly: a silent CPU fallback wearing an
    accelerator's name is the failure mode this project keeps being bitten by.
    """
    if engine_registry.engine_supports_unit(state["engine"], state["device"]):
        return
    previous = state["device"]
    state["device"] = "CPU"
    state["asr_device_env"] = "CPU"
    state["parts"].append(f"device -> CPU ({state['engine']} cannot drive {previous})")
    logger.warning(
        "%s cannot drive %s, so ASR runs on the CPU. The %s unit stays in the pool for "
        "preprocessing; set ASR_ENGINE explicitly to use it for ASR.",
        state["engine"],
        previous,
        previous,
    )


def resolve_engine_and_pool(
    *,
    asr_engine_env: str,
    asr_device_env: str,
    device: str,
    hardware_units: list,
    isolate_engines: bool,
    hybrid_env: str,
    logger,
) -> dict:
    """Resolve the ASR engine, the scheduler pool, and hybrid mode, in that order.

    ``hardware_units`` is narrowed in place, because it is the live scheduler pool every
    other module reads. Two snapshots are returned alongside: the pool as detection found
    it, and the pool before the engine's own limits were applied -- preprocessing must not
    inherit an ASR engine's filter, since UVR runs on ONNX Runtime and reaches devices
    CTranslate2 never will.
    """
    state = {
        "asr_engine_env": asr_engine_env,
        "asr_device_env": asr_device_env,
        "device": device,
        "isolate_engines": isolate_engines,
        "hybrid_env": hybrid_env,
        "engine": "",
        "engine_source": "auto",
        "hybrid": False,
        "parts": [],
    }

    _select_engine(state, hardware_units=hardware_units, logger=logger)
    _require_intel_hardware(state, hardware_units=hardware_units, logger=logger)

    # Before _restrict_pool_to_intel, not after. That filter is keyed on the *requested*
    # engine, so ASR_ENGINE=INTEL-WHISPER on an image with no OpenVINO stripped the CUDA
    # units out of the pool and only then discovered the engine is not installed -- leaving
    # the Faster-Whisper fallback with an Intel-only pool it cannot drive, which
    # _keep_only_drivable_units then emptied down to the CPU. A host with a working GPU
    # decoded on the CPU because of an engine that was never loadable in the first place.
    _fall_back_to_an_installed_engine(state, logger=logger)

    # Snapshot before any narrowing: code that needs to know what the machine actually has
    # must not read a filtered list and conclude hardware is absent. Taken ahead of the
    # Intel restriction too -- that also drops real units.
    detected_hardware_units = [dict(unit) for unit in hardware_units]

    _restrict_pool_to_intel(state, hardware_units=hardware_units, logger=logger)
    _restrict_pool_to_requested_device(state, hardware_units=hardware_units, logger=logger)
    _resolve_hybrid(state, hardware_units=hardware_units, logger=logger)

    # Everything detection found, before the ASR engine's own limits are applied.
    detected_units = [dict(unit) for unit in hardware_units]

    _keep_only_drivable_units(state, hardware_units=hardware_units, logger=logger)
    _align_device_with_engine(state, logger=logger)

    return {
        "ASR_ENGINE": state["engine"],
        "ASR_ENGINE_SOURCE": state["engine_source"],
        "DEVICE": state["device"],
        "ASR_DEVICE_ENV": state["asr_device_env"],
        "HYBRID_ENGINES": state["hybrid"],
        "DETECTED_HARDWARE_UNITS": detected_hardware_units,
        "DETECTED_UNITS": detected_units,
        # Joined only after every stage has appended: joining earlier silently dropped the
        # "requested engine not installed" fallback and the engine/unit pool narrowing.
        "ASR_ENGINE_RESOLUTION": " | ".join(state["parts"]),
    }
