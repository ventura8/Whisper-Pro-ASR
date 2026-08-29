"""Behaviours changed because a review found each of them wrong in a way tests missed.

Every case here is a defect that no existing test could observe: a cached "don't know", a
setting that parses but cannot mean anything, a total that poisons every day after it, an
explicit engine discarded by a mode that was also asked for, and an event pulled off a
handler and then dropped. They are grouped by the failure they prevent rather than by
module, because that is what makes each one worth keeping.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from modules.core import config_paths, device_probe, model_integrity
from modules.inference.engines import worker_runtime
from modules.inference.pipeline.preprocessing import isolation_policy
from modules.monitoring import history_helpers


class TestAnUnreadableGpuArchitectureIsNotRemembered:
    """The answer "could not tell" is a property of the attempt, not of the device.

    The probe was memoised with lru_cache, which stores None as readily as an answer. One
    unlucky read during startup -- a plugin still loading, a driver momentarily busy -- then
    pinned an Arc GPU to the slow in-process UVR path for the life of the process, with no
    way to re-ask.
    """

    @pytest.fixture(autouse=True)
    def _clear(self):
        """Start and finish with an empty cache."""
        isolation_policy._ARCH_CACHE.clear()
        yield
        isolation_policy._ARCH_CACHE.clear()

    def _import_returning(self, architecture):
        """Answer only the OpenVINO import, so unrelated lazy imports still get real modules."""
        core = mock.MagicMock()
        if isinstance(architecture, Exception):
            core.get_property.side_effect = architecture
        else:
            core.get_property.return_value = architecture
        ov = mock.MagicMock()
        ov.Core.return_value = core
        real = __import__("importlib").import_module

        def _import(name, *args, **kwargs):
            return ov if name == "openvino" else real(name, *args, **kwargs)

        return mock.patch("importlib.import_module", side_effect=_import)

    def test_a_failed_read_is_retried_and_can_still_succeed(self):
        """The whole point: the second attempt must reach OpenVINO again."""
        with self._import_returning(RuntimeError("GPU plugin still loading")):
            assert isolation_policy.intel_gpu_arch_for("GPU.0") is None

        with self._import_returning("GPU: vendor=0x8086 arch=v12.74.0"):
            assert isolation_policy.intel_gpu_arch_for("GPU.0") == (12, 74)

    def test_a_successful_read_is_still_answered_from_the_cache(self):
        """The probe costs a full OpenVINO plugin enumeration, so it must not repeat."""
        with self._import_returning("GPU: vendor=0x8086 arch=v12.74.0"):
            assert isolation_policy.intel_gpu_arch_for("GPU.0") == (12, 74)

        # No OpenVINO available at all: only the cache can answer this.
        with mock.patch("importlib.import_module", side_effect=ImportError("no openvino")):
            assert isolation_policy.intel_gpu_arch_for("GPU.0") == (12, 74)


class TestASettingThatParsesButCannotMeanAnything:
    """VAD_THRESHOLD is a probability, and a number outside [0, 1] silences the service."""

    @pytest.mark.parametrize("raw", ["50", "-0.1", "1.5", "nan", "inf"])
    def test_a_value_outside_the_range_falls_back_to_the_default(self, raw):
        """``VAD_THRESHOLD=50`` is a plausible way to write "50%", and rejects every frame."""
        with mock.patch.dict(os.environ, {"VAD_THRESHOLD": raw}):
            assert config_paths.float_env("VAD_THRESHOLD", 0.5, minimum=0.0, maximum=1.0) == 0.5

    @pytest.mark.parametrize("raw,expected", [("0", 0.0), ("1", 1.0), ("0.35", 0.35)])
    def test_the_bounds_are_inclusive_and_valid_values_pass_through(self, raw, expected):
        """The guard must not narrow the range an operator is entitled to use."""
        with mock.patch.dict(os.environ, {"VAD_THRESHOLD": raw}):
            assert config_paths.float_env("VAD_THRESHOLD", 0.5, minimum=0.0, maximum=1.0) == expected

    def test_without_bounds_any_finite_number_is_accepted(self):
        """Bounds are opt-in; the unbounded call sites must be unaffected."""
        with mock.patch.dict(os.environ, {"SOME_SETTING": "1000"}):
            assert config_paths.float_env("SOME_SETTING", 1.0) == 1000.0


class TestNanNeverEntersTheStatisticsFile:
    """NaN is a float, survives isinstance, and then poisons every total it reaches.

    It propagates through max() and ``+=`` into each following day, and serialises as bare
    ``NaN`` -- invalid JSON, so one poisoned day took the whole statistics endpoint down
    rather than being rebuilt from history.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_total_makes_the_day_unusable(self, bad):
        """A non-finite total makes the day unusable."""
        assert history_helpers._is_usable_category({"count": 1, "duration": bad}) is False

    def test_ordinary_totals_are_still_usable(self):
        """The guard must not reject the days it exists to preserve."""
        assert history_helpers._is_usable_category({"count": 4, "duration": 12.5}) is True

    def test_a_bool_is_still_rejected(self):
        """bool is an int subclass; counting True as 1 hides a corrupt file."""
        assert history_helpers._is_usable_category({"count": True, "duration": 1.0}) is False


class TestAnIrThatParsesCanStillFailInspection:
    """Reading a port's name or shape goes back through the plugin and can throw there."""

    def test_a_shape_read_that_raises_is_reported_as_unreadable(self):
        """Unguarded this escaped config import and aborted startup on the affected host."""
        port = mock.MagicMock()
        type(port).any_name = mock.PropertyMock(side_effect=RuntimeError("no unique name"))
        core = mock.MagicMock()
        core.read_model.return_value = mock.MagicMock(inputs=[port])

        assert device_probe._dynamic_inputs(core, "/ir/openvino_encoder_model.xml", "encoder") is None


class TestAModelPathThatCannotBeStatted:
    """Provisioning wants "not valid", not a traceback, from a broken path."""

    def test_a_dangling_symlink_xml_is_invalid_rather_than_an_error(self, tmp_path):
        """A dangling symlink raises OSError from stat(); is_file() alone did not catch it."""
        model_dir = tmp_path / "ir"
        model_dir.mkdir()
        (model_dir / "openvino_encoder_model.xml").symlink_to(tmp_path / "gone.xml")
        # A real BIN of sufficient size, so the assertion below can only be about the XML.
        # With the BIN absent too, the pair was invalid for the ordinary missing-file reason
        # and the dangling symlink -- the thing this regression is named for -- was never
        # actually exercised.
        (model_dir / "openvino_encoder_model.bin").write_bytes(b"x" * 16)

        assert model_integrity._ov_xml_bin_pair_valid(model_dir, "openvino_encoder_model", 1) is False


class TestACancelledStreamKeepsTheEventItAlreadyPulled:
    """The cancel check belongs after the send, not before it.

    Checking first discarded an event that had already been taken off the handler. For a
    transcription that is the ``info`` event carrying detected language and duration, so a
    cancel landing in the first window lost the only metadata the parent would ever get.
    """

    class _Conn:
        """A connection with one cancel queued, pending from the very first check."""

        def __init__(self):
            self.sent = []
            # Drained, not re-served: _cancel_requested loops while poll() is true, so a
            # connection that always reports a pending message never returns.
            self._control = [{"control": "cancel"}]

        def poll(self, _timeout=None):
            """Whether a control message is still pending."""
            return bool(self._control)

        def recv(self):
            """Take the next control message."""
            return self._control.pop(0)

        def send(self, message):
            """Record what the worker forwarded."""
            self.sent.append(message)

    def test_the_first_event_is_forwarded_before_the_cancel_is_honoured(self):
        """The first event is forwarded before the cancel is honoured."""
        conn = self._Conn()
        events = iter([{"event": "info", "language": "en"}, {"event": "data", "text": "never"}])

        cancelled = worker_runtime._pump_events(conn, events, request_id=7, cmd="transcribe")

        assert cancelled is True
        assert [m["event"] for m in conn.sent] == ["info"], "the pulled event must not be dropped"
        assert conn.sent[0]["language"] == "en"
