"""Decide whether an accelerator can execute the model, before the service claims it can.

The Intel NPU builds a WhisperPipeline and only fails at inference, so "the pipeline
loaded" is not evidence the device works. Without a check the service starts healthy,
prints `ASR Runtime: OpenVINO (NPU)`, and returns HTTP 500 for every request.

The check is metadata, not a trial run. Actually compiling Whisper on the NPU takes longer
than any sane startup budget (a first attempt blew a 180s timeout), and a timeout answers
"unknown" rather than "unusable". Reading the IR answers exactly the right question in
milliseconds: the NPU plugin rejects dynamic shapes, and this model's IR is dynamic, which
is precisely why execution fails with `ZE_RESULT_ERROR_UNKNOWN` at the Level Zero layer.

`read_model` parses XML on the host and creates no device context, so this is safe to call
in the main process -- unlike compiling, which is what engine isolation exists to keep out.
"""

import logging
import os

logger = logging.getLogger(__name__)

_IR_PARTS = ("openvino_encoder_model.xml", "openvino_decoder_model.xml")


def _dynamic_inputs(core, path: str, part: str) -> list[str] | None:
    """This IR file's dynamically-shaped inputs, empty when static, None when unreadable.

    The three answers have to stay distinct. An empty list previously meant both "inspected
    and fully static" and "could not be read at all", so a wholly unreadable IR counted as
    inspected-and-fine and the probe reported the NPU as usable -- the permissive answer
    reached by the one path that has verified nothing.
    """
    try:
        model = core.read_model(path)
    except (RuntimeError, OSError) as exc:
        logger.warning("[Probe] Could not read %s (%s); cannot judge the NPU on it.", part, exc)
        return None
    return [f"{p.any_name}{p.get_partial_shape()}" for p in model.inputs if p.get_partial_shape().is_dynamic]


def _open_core():
    """Return an OpenVINO Core, or None when OpenVINO is unavailable or unusable.

    Both failure points are folded together because the caller treats them identically:
    the import fails with OSError when a plugin's shared library cannot be loaded and with
    RuntimeError when the runtime initialises but is unusable, and constructing the Core
    loads every installed plugin, so a broken GPU or NPU plugin throws there instead.
    Left unguarded either one escaped this probe during config import and aborted startup,
    on precisely the hosts the permissive fallback exists to keep serving.

    "Cannot check" is not "is fine". The caller's permissive answer must never be the
    silent one, because a swallowed answer looks exactly like a verified NPU and puts the
    wrong device on the banner -- hence the warning here rather than at the call site.
    """
    try:
        import openvino as ov  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        return ov.Core()
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("[Probe] openvino is unavailable or unusable (%s); cannot judge the NPU.", exc)
        return None


def _inspect_ir_parts(core, model_dir: str) -> list[tuple[str, list[str] | None]]:
    """Every IR part present under ``model_dir``, paired with its dynamic-shape verdict.

    The verdict is None for a part that could not be read, which the caller keeps distinct
    from "read and static": an unreadable IR has verified nothing about the device.
    """
    results = []
    for part in _IR_PARTS:
        path = os.path.join(model_dir, part)
        if os.path.exists(path):
            results.append((part, _dynamic_inputs(core, path, part)))
    return results


def _readable_count(inspections: list[tuple[str, list[str] | None]]) -> int:
    """How many IR parts were actually read, as opposed to merely present."""
    return sum(1 for _part, dynamic in inspections if dynamic is not None)


def _blocking_reason(inspections: list[tuple[str, list[str] | None]], model_dir: str) -> str:
    """Why the NPU cannot run this IR, or empty when nothing inspected blocks it.

    Two distinct blockers, and both belong here rather than folded into a count: a
    dynamically-shaped part, which the NPU plugin rejects outright, and an IR whose every
    part is unreadable, which has proven nothing and must not read as a pass.
    """
    for part, dynamic in inspections:
        if dynamic:
            return f"{part} has dynamic input shapes ({', '.join(dynamic[:3])}) and the NPU plugin requires static upper bounds"
    if inspections and not _readable_count(inspections):
        return f"every OpenVINO IR part under {model_dir} is present but unreadable, so the NPU cannot be shown to execute it"
    return ""


def _scan_ir_shapes(core, model_dir: str) -> tuple[int, str]:
    """Inspect every IR part present, returning (files read, first blocking reason).

    The count is of parts actually *read*, not merely present. A part that could not be read
    used to count as inspected-and-fine, so a wholly unreadable IR reported the NPU as
    usable -- the permissive answer reached by the one path that has verified nothing.
    """
    inspections = _inspect_ir_parts(core, model_dir)
    reason = _blocking_reason(inspections, model_dir)
    if reason:
        return len(inspections), reason
    return _readable_count(inspections), ""


def npu_can_execute(model_dir: str) -> tuple[bool, str]:
    """Return whether the Intel NPU can run the IR in ``model_dir``, and why not if it cannot."""
    core = _open_core()
    if core is None:
        return True, ""

    inspected, reason = _scan_ir_shapes(core, model_dir)
    if reason:
        return False, reason
    if not inspected:
        # The IR is usually absent at this point: weights are provisioned in the background,
        # after the scheduler has already snapshotted its unit pool, so waiting for them is
        # not an option. Defaulting to "usable" here is what let the NPU onto the banner
        # three times over. Every OpenVINO Whisper export produced by optimum is
        # dynamic-shaped and therefore unexecutable on the NPU, so absence of evidence is
        # not evidence the device works -- assume the documented limitation holds. A
        # genuinely static custom export is inspected and honoured on the next start, once
        # its files exist.
        return False, (
            f"no OpenVINO IR to inspect under {model_dir} yet, and the standard Whisper "
            "export is dynamic-shaped, which the NPU plugin cannot execute"
        )
    logger.info("[Probe] NPU: %d IR file(s) inspected, all statically shaped.", inspected)
    return True, ""
