"""Whether the Intel NPU is allowed onto the banner, decided before any request arrives.

This probe is the guard on a failure mode that has bitten this project three times: the NPU
builds a WhisperPipeline in about four seconds, the service reports healthy, the banner
prints `ASR Runtime: OpenVINO (NPU)`, and every request then returns HTTP 500 with
`ZE_RESULT_ERROR_UNKNOWN`. It reads IR metadata rather than compiling, so the whole thing is
pure host-side XML parsing and is fully testable with a fake Core.

The distinction the tests below are really about is three-way, not two-way: "static and
therefore usable", "dynamic and therefore not", and "could not be read at all". Folding the
third into the first is exactly how a device that had verified nothing reached the banner.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modules.core import device_probe


class FakeShape:
    """`is_dynamic` is an attribute, not a method.

    Verified against the shipped runtime rather than assumed: on OpenVINO 2026.3.1
    `PartialShape.is_dynamic` is a bool property and is not callable. A fake exposing it as
    a method would make every shape truthy, so every input would read as dynamic and these
    tests would agree with a probe that had stopped distinguishing anything.
    """

    def __init__(self, dynamic: bool, text: str = "[?,128,3000]"):
        """Record the construction arguments."""
        self.is_dynamic = dynamic
        self._text = text

    def __str__(self) -> str:
        """The shape's textual form, as the probe formats it."""
        return self._text


def _port(name="input_features", dynamic=True):
    """A model input port with the given shape dynamism."""
    return SimpleNamespace(any_name=name, get_partial_shape=lambda: FakeShape(dynamic))


class FakeCore:
    """Reads models from a scripted map of path-suffix -> inputs (or an exception)."""

    def __init__(self, by_part: dict[str, object]):
        """Record the construction arguments."""
        self._by_part = by_part

    def read_model(self, path: str):
        """Return the scripted model, or raise the scripted error."""
        for part, outcome in self._by_part.items():
            if path.endswith(part):
                if isinstance(outcome, Exception):
                    raise outcome
                return SimpleNamespace(inputs=outcome)
        raise RuntimeError(f"unexpected read of {path}")


@pytest.fixture(name="ir")
def _ir(tmp_path):
    """Create the IR part files the probe looks for, and return the directory."""

    def make(*parts: str):
        """Build the object under test with the given settings."""
        for part in parts:
            (tmp_path / part).write_text("<net/>", encoding="utf-8")
        return str(tmp_path)

    return make


ENCODER = "openvino_encoder_model.xml"
DECODER = "openvino_decoder_model.xml"


class TestOpeningTheCore:
    """OpenVINO is absent entirely from the non-Intel images."""

    @pytest.mark.parametrize("error", [ImportError("no openvino"), OSError("libze.so missing"), RuntimeError("plugin broken")])
    def test_an_unusable_runtime_returns_none_and_warns_rather_than_raising(self, monkeypatch, caplog, error):
        """Unguarded, either escaped during config import and aborted startup -- on exactly
        the hosts the permissive fallback exists to keep serving."""
        monkeypatch.setattr(device_probe.importlib, "import_module", lambda _n: (_ for _ in ()).throw(error))
        with caplog.at_level("WARNING"):
            assert device_probe._open_core() is None
        assert "cannot judge the NPU" in caplog.text, "a silent None looks exactly like a verified NPU"

    def test_a_working_runtime_returns_its_core(self, monkeypatch):
        """A working runtime returns its core."""
        core = object()
        monkeypatch.setattr(device_probe.importlib, "import_module", lambda _n: SimpleNamespace(Core=lambda: core))
        assert device_probe._open_core() is core


class TestReadingOneIrPart:
    """Three answers, kept distinct: dynamic inputs, none, or unreadable."""

    def test_dynamic_inputs_are_reported_with_their_shapes(self):
        """Dynamic inputs are reported with their shapes."""
        core = FakeCore({ENCODER: [_port(dynamic=True)]})
        assert device_probe._dynamic_inputs(core, f"/m/{ENCODER}", ENCODER) == ["input_features[?,128,3000]"]

    def test_a_fully_static_part_reports_an_empty_list_not_none(self):
        """A fully static part reports an empty list not none."""
        core = FakeCore({ENCODER: [_port(dynamic=False)]})
        assert device_probe._dynamic_inputs(core, f"/m/{ENCODER}", ENCODER) == []

    @pytest.mark.parametrize("error", [RuntimeError("corrupt xml"), OSError("permission denied")])
    def test_an_unreadable_part_reports_none_not_an_empty_list(self, caplog, error):
        """An empty list meant both "inspected and static" and "could not be read", so a
        wholly unreadable IR counted as inspected-and-fine and the NPU was reported usable.
        """
        core = FakeCore({ENCODER: error})
        with caplog.at_level("WARNING"):
            assert device_probe._dynamic_inputs(core, f"/m/{ENCODER}", ENCODER) is None
        assert "cannot judge the NPU" in caplog.text


class TestScanningTheModelDirectory:
    """Every part Whisper needs is accounted for; a missing one verifies nothing."""

    def test_a_missing_part_is_reported_as_unverified(self, ir):
        """Skipping it let a half-provisioned export clear the NPU on the encoder alone."""
        model_dir = ir(ENCODER)
        core = FakeCore({ENCODER: [_port(dynamic=False)]})
        assert device_probe._inspect_ir_parts(core, model_dir) == [(ENCODER, []), (DECODER, None)]

    def test_a_half_provisioned_export_does_not_read_as_inspected(self, ir):
        """The count that decides the "all statically shaped" log must not include it."""
        model_dir = ir(ENCODER)
        core = FakeCore({ENCODER: [_port(dynamic=False)]})
        assert device_probe._readable_count(device_probe._inspect_ir_parts(core, model_dir)) == 1
        assert device_probe._blocking_reason(device_probe._inspect_ir_parts(core, model_dir), model_dir)

    def test_both_parts_are_inspected_when_both_exist(self, ir):
        """Both parts are inspected when both exist."""
        model_dir = ir(ENCODER, DECODER)
        core = FakeCore({ENCODER: [_port(dynamic=False)], DECODER: [_port(dynamic=False)]})
        assert device_probe._readable_count(device_probe._inspect_ir_parts(core, model_dir)) == 2

    def test_an_empty_directory_reads_nothing(self, ir):
        """Both parts are accounted for, and neither has been read."""
        assert device_probe._readable_count(device_probe._inspect_ir_parts(FakeCore({}), ir())) == 0


class TestTheBlockingReason:
    """Two distinct blockers, and neither may read as a pass."""

    def test_a_dynamic_part_names_itself_and_the_plugin_requirement(self):
        """A dynamic part names itself and the plugin requirement."""
        reason = device_probe._blocking_reason([(ENCODER, ["input_features[?,128,3000]"])], "/m")
        assert ENCODER in reason
        assert "static upper bounds" in reason

    def test_an_entirely_unreadable_ir_blocks_rather_than_passing(self):
        """An entirely unreadable ir blocks rather than passing."""
        reason = device_probe._blocking_reason([(ENCODER, None), (DECODER, None)], "/m")
        assert "unreadable" in reason

    def test_a_partly_readable_ir_blocks_on_the_unreadable_part(self):
        """Whisper needs both halves, so a readable encoder is not evidence about the decoder.

        This previously passed: the check asked whether EVERY part was unreadable, so an IR
        whose encoder read as static and whose decoder could not be read at all reported
        "all statically shaped" -- a device cleared on half the evidence.
        """
        reason = device_probe._blocking_reason([(ENCODER, []), (DECODER, None)], "/m")
        assert DECODER in reason
        assert "unreadable" in reason

    def test_a_fully_readable_static_ir_still_does_not_block(self):
        """The permissive answer survives where it is actually earned."""
        assert device_probe._blocking_reason([(ENCODER, []), (DECODER, [])], "/m") == ""

    def test_nothing_inspected_is_not_itself_a_blocking_reason(self):
        """Absence is the caller's decision, made with its own explanation."""
        assert device_probe._blocking_reason([], "/m") == ""

    def test_only_the_first_three_dynamic_inputs_are_quoted(self):
        """Only the first three dynamic inputs are quoted."""
        reason = device_probe._blocking_reason([(ENCODER, ["a[?]", "b[?]", "c[?]", "d[?]"])], "/m")
        assert "d[?]" not in reason, "the reason is a log line, not a dump"


class TestNpuCanExecute:
    """The answer config.py acts on."""

    def test_no_openvino_is_permissive_because_nothing_can_be_judged(self, monkeypatch):
        """No openvino is permissive because nothing can be judged."""
        monkeypatch.setattr(device_probe, "_open_core", lambda: None)
        assert device_probe.npu_can_execute("/m") == (True, "")

    def test_a_dynamic_ir_is_rejected_with_the_reason(self, monkeypatch, ir):
        """A dynamic ir is rejected with the reason."""
        model_dir = ir(ENCODER)
        monkeypatch.setattr(device_probe, "_open_core", lambda: FakeCore({ENCODER: [_port(dynamic=True)]}))

        ok, reason = device_probe.npu_can_execute(model_dir)

        assert ok is False
        assert "dynamic input shapes" in reason

    def test_a_statically_shaped_ir_is_accepted(self, monkeypatch, ir):
        """A genuinely static custom export is honoured -- the limitation is the IR, not the
        device category."""
        model_dir = ir(ENCODER, DECODER)
        monkeypatch.setattr(
            device_probe, "_open_core", lambda: FakeCore({ENCODER: [_port(dynamic=False)], DECODER: [_port(dynamic=False)]})
        )
        assert device_probe.npu_can_execute(model_dir) == (True, "")

    def test_an_absent_ir_is_rejected_rather_than_assumed_usable(self, monkeypatch, ir):
        """Weights are provisioned in the background, after the pool is snapshotted, so the
        IR is usually missing here. Defaulting to "usable" is what put the NPU on the banner
        three times: every optimum Whisper export is dynamic-shaped, so absence of evidence
        is not evidence the device works.
        """
        monkeypatch.setattr(device_probe, "_open_core", lambda: FakeCore({}))

        ok, reason = device_probe.npu_can_execute(ir())

        assert ok is False
        assert "no OpenVINO IR to inspect" in reason

    def test_a_present_but_unreadable_ir_is_rejected(self, monkeypatch, ir):
        """A present but unreadable ir is rejected."""
        model_dir = ir(ENCODER)
        monkeypatch.setattr(device_probe, "_open_core", lambda: FakeCore({ENCODER: RuntimeError("corrupt")}))

        ok, reason = device_probe.npu_can_execute(model_dir)

        assert ok is False
        assert "unreadable" in reason
