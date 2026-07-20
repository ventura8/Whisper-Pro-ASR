"""Preprocessing pipeline tests (split from test_preprocessing.py)."""

import logging
from unittest import mock

import pytest

from modules.inference.pipeline import openvino_resolver, preprocessing
from modules.inference.pipeline.preprocessing import CACHE_DIR, PreprocessingManager

logger = logging.getLogger(__name__)


@pytest.fixture
def prep_manager():
    """Fixture to provide a clean PreprocessingManager instance."""
    unit = {"id": "CPU", "type": "CPU", "name": "CPU"}
    return PreprocessingManager(assigned_unit=unit)


@pytest.fixture(autouse=True)
def reset_openvino_family_circuit_breaker():
    openvino_resolver.clear_openvino_disabled_families()
    yield
    openvino_resolver.clear_openvino_disabled_families()


class TestPreprocessAudio:
    """Tests for the main preprocess_audio entry point."""

    def test_preprocess_disabled(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = False
            assert prep_manager.preprocess_audio("test.wav") == "test.wav"

    def test_preprocess_returns_original_when_requested_accelerator_and_cpu_fallback_fail(self, prep_manager):
        """Requested accelerator failures should fall back to CPU and return original if CPU also fails."""
        with (
            mock.patch(
                "modules.inference.pipeline.preprocessing.config.ENABLE_VOCAL_SEPARATION",
                True,
            ),
            mock.patch(
                "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                return_value="audio.wav",
            ),
            mock.patch.object(
                prep_manager,
                "_init_separator",
                side_effect=preprocessing.UVRAcceleratorUnavailableError("OpenVINO unavailable"),
            ),
        ):
            assert prep_manager.preprocess_audio("audio.wav") == "audio.wav"

    def test_preprocess_retries_cpu_for_openvino_session_cpu_fallback_error(self, prep_manager):
        prep_manager._device_type = "NPU"
        prep_manager._device_id = "NPU"
        with (
            mock.patch(
                "modules.inference.pipeline.preprocessing.config.ENABLE_VOCAL_SEPARATION",
                True,
            ),
            mock.patch(
                "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                return_value="audio.wav",
            ),
            mock.patch.object(
                prep_manager,
                "_run_preprocess_pipeline",
                side_effect=RuntimeError(
                    "OpenVINOExecutionProvider did not initialize for device_type=NPU; "
                    "ONNX Runtime fell back to providers=['CPUExecutionProvider']"
                ),
            ),
        ):
            assert prep_manager.preprocess_audio("audio.wav") == "audio.wav"
            assert prep_manager._run_preprocess_pipeline.call_count == 2

    def test_preprocess_success(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
                with mock.patch(
                    "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                    side_effect=lambda path, **_: path,
                ):
                    res = prep_manager.preprocess_audio("test.wav")
                    assert "vocal.wav" in res

    def test_preprocess_stage_order_shows_ffmpeg_before_vocal_separation(self, prep_manager):
        """FFmpeg preparation should occur before Vocal Separation stage is published."""
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            events = []

            def _prep(path, **_kwargs):
                events.append("prepare_for_uvr")
                return path

            def _stage(*args, **kwargs):
                _ = kwargs
                if len(args) >= 2 and args[1] == "Vocal Separation":
                    events.append("vocal_stage")

            with (
                mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort),
                mock.patch(
                    "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                    side_effect=_prep,
                ),
                mock.patch(
                    "modules.inference.pipeline.preprocessing.scheduler.update_task_progress",
                    side_effect=_stage,
                ),
            ):
                prep_manager.preprocess_audio("test.wav")

            assert events.index("prepare_for_uvr") < events.index("vocal_stage")

    def test_preprocess_cleanup_secondary_stems(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav", "instrumental.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with (
                mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort),
                mock.patch("os.path.exists", return_value=True),
                mock.patch("os.remove") as mock_remove,
            ):
                with mock.patch(
                    "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                    side_effect=lambda path, **_: path,
                ):
                    prep_manager.preprocess_audio("test.wav")
                    mock_remove.assert_called()

    def test_preprocess_exception(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            prep_manager._init_separator = mock.MagicMock(side_effect=Exception("Crash"))
            assert prep_manager.preprocess_audio("test.wav") == "test.wav"

    def test_preprocess_force(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = False  # Disabled via config
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            mock_ort = mock.MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with mock.patch("modules.inference.pipeline.preprocessing.ort", mock_ort):
                with mock.patch(
                    "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                    side_effect=lambda path, **_: path,
                ):
                    # Should still run because force=True
                    res = prep_manager.preprocess_audio("test.wav", force=True)
                    assert "vocal.wav" in res

    def test_preprocess_relative_stem(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["relative_vocal.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with mock.patch("modules.inference.pipeline.preprocessing.ort", mock.MagicMock()):
                with mock.patch(
                    "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                    side_effect=lambda path, **_: path,
                ):
                    res = prep_manager.preprocess_audio("test.wav")
                    assert "relative_vocal.wav" in res
                    assert str(CACHE_DIR) in res

    def test_preprocess_resolves_relative_stem_from_effective_separator_output_dir(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["relative_vocal.wav"]
            mock_sep.output_dir = "/alt/output"
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with (
                mock.patch("modules.inference.pipeline.preprocessing.ort", mock.MagicMock()),
                mock.patch(
                    "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                    side_effect=lambda path, **_: path,
                ),
                mock.patch(
                    "os.path.exists",
                    side_effect=lambda p: p.replace("\\", "/") == "/alt/output/relative_vocal.wav",
                ),
            ):
                res = prep_manager.preprocess_audio("test.wav")

            assert res.replace("\\", "/") == "/alt/output/relative_vocal.wav"

    def test_preprocess_relative_stem_falls_back_to_cache_dir_when_not_found_anywhere(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["relative_vocal.wav"]
            mock_sep.output_dir = "/alt/output"
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with (
                mock.patch("modules.inference.pipeline.preprocessing.ort", mock.MagicMock()),
                mock.patch(
                    "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                    side_effect=lambda path, **_: path,
                ),
                mock.patch("os.path.exists", return_value=False),
            ):
                res = prep_manager.preprocess_audio("test.wav")

        assert res == str(CACHE_DIR / "relative_vocal.wav")

    def test_preprocess_cleanup_error(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            mock_sep = mock.MagicMock()
            mock_sep.separate.return_value = ["vocal.wav", "extra.wav"]
            prep_manager._init_separator = mock.MagicMock(return_value=mock_sep)

            with (
                mock.patch("modules.inference.pipeline.preprocessing.ort", mock.MagicMock()),
                mock.patch("os.path.exists", return_value=True),
                mock.patch("os.remove", side_effect=OSError("Busy")),
            ):
                with mock.patch(
                    "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                    side_effect=lambda path, **_: path,
                ):
                    # Should not raise
                    prep_manager.preprocess_audio("test.wav")

    def test_preprocess_prepare_fail(self, prep_manager):
        with mock.patch("modules.inference.pipeline.preprocessing.config") as mock_cfg:
            mock_cfg.ENABLE_VOCAL_SEPARATION = True
            with mock.patch(
                "modules.inference.pipeline.preprocessing.utils.prepare_for_uvr",
                return_value=None,
            ):
                assert prep_manager.preprocess_audio("test.wav") == "test.wav"
