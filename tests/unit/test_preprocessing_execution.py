"""Unit tests for preprocessing execution helpers."""

from unittest import mock

from modules.inference.pipeline.preprocessing import execution


def test_create_separator_passes_expected_defaults():
    """Separator factory should pass project-default UVR settings to constructor."""
    sep_cls = mock.MagicMock()
    lazy_import = mock.MagicMock(return_value=sep_cls)
    execution.create_separator(lazy_import, "out")
    sep_cls.assert_called_once()

    kwargs = sep_cls.call_args.kwargs
    assert kwargs["output_dir"] == "out"
    assert kwargs["output_format"] == "WAV"
    assert kwargs["output_single_stem"] == "Vocals"


def test_create_separator_does_not_attenuate_the_vocal_stem():
    """normalization_threshold is a ceiling, so a low value silently destroys quality.

    audio-separator only attenuates when the stem peak exceeds this value; it never
    amplifies. A small threshold scales every stem down to that peak, and the 16-bit PCM
    Whisper is fed then quantizes the result -- 0.01 left ~9 effective bits and turned
    "the quick brown fox jumps" into "...dumps" on the known-text fixture.
    """
    sep_cls = mock.MagicMock()
    execution.create_separator(mock.MagicMock(return_value=sep_cls), "out")

    threshold = sep_cls.call_args.kwargs["normalization_threshold"]
    assert threshold >= 0.5, f"normalization_threshold={threshold} attenuates the stem into 16-bit quantization noise"
    assert threshold <= 1.0, "normalization_threshold must stay within the library's (0, 1] range"


def test_enable_separator_acceleration_flag_sets_only_for_accelerated_providers():
    """Hardware acceleration flag should be enabled for CUDA/OpenVINO providers only."""
    sep = mock.MagicMock()
    sep.hardware_acceleration_enabled = False

    execution.enable_separator_acceleration_flag(sep, ["CPUExecutionProvider"], "CPU")
    assert sep.hardware_acceleration_enabled is False

    execution.enable_separator_acceleration_flag(sep, ["OpenVINOExecutionProvider", "CPUExecutionProvider"], "NPU")
    assert sep.hardware_acceleration_enabled is True


def test_try_openvino_candidate_load_skips_disabled_families():
    """Candidate loader should immediately skip retries for disabled OpenVINO families."""
    sep = mock.MagicMock()
    first_error = RuntimeError("first")
    with mock.patch(
        "modules.inference.pipeline.preprocessing.execution.openvino_resolver.is_openvino_family_disabled",
        return_value=True,
    ):
        ok, err = execution.try_openvino_candidate_load(sep, "NPU", "NPU.0", first_error)
    assert ok is False
    assert err is first_error


def test_try_openvino_candidate_load_success_and_loader_error_paths():
    """Candidate loader should support successful retries and global disable on loader failures."""
    sep = mock.MagicMock()
    first_error = RuntimeError("first")

    with (
        mock.patch(
            "modules.inference.pipeline.preprocessing.execution.openvino_resolver.is_openvino_family_disabled",
            return_value=False,
        ),
        mock.patch(
            "modules.inference.pipeline.preprocessing.execution.openvino_provider_dispatch.openvino_provider_config",
            return_value=(["OpenVINOExecutionProvider", "CPUExecutionProvider"], [{"device_type": "GPU.0"}]),
        ),
        mock.patch("modules.inference.pipeline.preprocessing.execution.openvino_resolver.set_openvino_context_options") as mock_set,
    ):
        ok, err = execution.try_openvino_candidate_load(sep, "NPU", "GPU.0", first_error)

    assert ok is True
    assert err is first_error
    mock_set.assert_called_once()

    sep = mock.MagicMock()
    retry_error = RuntimeError("loader")
    sep.load_model.side_effect = retry_error
    with (
        mock.patch(
            "modules.inference.pipeline.preprocessing.execution.openvino_resolver.is_openvino_family_disabled",
            return_value=False,
        ),
        mock.patch(
            "modules.inference.pipeline.preprocessing.execution.openvino_provider_dispatch.openvino_provider_config",
            return_value=(["OpenVINOExecutionProvider", "CPUExecutionProvider"], [{"device_type": "GPU.0"}]),
        ),
        mock.patch("modules.inference.pipeline.preprocessing.execution.openvino_resolver.set_openvino_context_options"),
        mock.patch(
            "modules.inference.pipeline.preprocessing.execution.openvino_resolver.is_openvino_runtime_loader_error",
            return_value=True,
        ),
        mock.patch(
            "modules.inference.pipeline.preprocessing.execution.openvino_resolver.disable_all_openvino_families_for_runtime"
        ) as mock_disable,
    ):
        ok, err = execution.try_openvino_candidate_load(sep, "NPU", "GPU.0", first_error)
    assert ok is False
    assert err is retry_error
    mock_disable.assert_called_once()


def test_cleanup_secondary_stems_tracks_and_removes_non_primary():
    """Cleanup should track all non-primary stems and remove only non-primary files."""
    resolver = mock.MagicMock(side_effect=["keep.wav", "stem1.wav", "stem2.wav"])
    with (
        mock.patch("modules.inference.pipeline.preprocessing.execution.utils.track_file") as mock_track,
        mock.patch("modules.inference.pipeline.preprocessing.execution.utils.secure_remove") as mock_remove,
    ):
        execution.cleanup_extra_stems(resolver, ["vocal", "inst", "noise"], mock.MagicMock(), "audio.wav", "keep.wav")

    assert mock_track.call_count == 3
    assert mock_remove.call_count == 2


def test_separate_audio_cpu_lock_and_context_cleanup():
    """Separation helper should set context options and always reset them after execution."""
    lock = mock.MagicMock()
    lock.__enter__ = mock.MagicMock(return_value=lock)
    lock.__exit__ = mock.MagicMock(return_value=False)
    sep = mock.MagicMock()

    fake_separator = mock.MagicMock()
    fake_separator.hardware_acceleration_enabled = True

    def _fallback(_sep, make_sep, _audio, yield_cb=None):
        _ = yield_cb
        made = make_sep("out")
        assert made is fake_separator
        return ["vocals.wav"], fake_separator

    with (
        mock.patch("modules.inference.pipeline.preprocessing.execution.create_separator", return_value=fake_separator),
        mock.patch("modules.inference.pipeline.preprocessing.execution.utils.cpu_lock_ctx") as mock_cpu_ctx,
    ):
        mock_cpu_ctx.return_value.__enter__ = mock.MagicMock()
        mock_cpu_ctx.return_value.__exit__ = mock.MagicMock(return_value=False)
        result = execution.separate_audio(
            lock,
            sep,
            "audio.wav",
            use_cpu_lock=True,
            target_options=[{"device_type": "NPU.0"}],
            active_yield_cb=None,
            lazy_import_separator=mock.MagicMock(),
            separate_with_fallback=_fallback,
        )

    assert result[0] == ["vocals.wav"]
    assert execution.utils.THREAD_CONTEXT.ov_options is None


def test_separate_audio_without_cpu_lock_and_resolve_output_path():
    """Separation should work without CPU lock and output resolver should handle empty and populated stems."""
    lock = mock.MagicMock()
    lock.__enter__ = mock.MagicMock(return_value=lock)
    lock.__exit__ = mock.MagicMock(return_value=False)
    sep = mock.MagicMock()

    with mock.patch("modules.inference.pipeline.preprocessing.execution.utils.cpu_lock_ctx") as mock_cpu_ctx:
        execution.separate_audio(
            lock,
            sep,
            "audio.wav",
            use_cpu_lock=False,
            target_options=[{}],
            active_yield_cb=None,
            lazy_import_separator=mock.MagicMock(),
            separate_with_fallback=lambda *_args, **_kwargs: (["v.wav"], sep),
        )
        mock_cpu_ctx.assert_not_called()

    assert execution.resolve_isolation_output_path(mock.MagicMock(), [], sep, "audio.wav") == "audio.wav"

    resolver = mock.MagicMock(side_effect=["vocals.wav", "inst.wav"])
    with (
        mock.patch("modules.inference.pipeline.preprocessing.execution.utils.track_file") as mock_track,
        mock.patch("modules.inference.pipeline.preprocessing.execution.cleanup_extra_stems") as mock_cleanup,
    ):
        stem_path = execution.resolve_isolation_output_path(resolver, ["vocal", "inst"], sep, "audio.wav")

    assert stem_path == "vocals.wav"
    mock_track.assert_called_once_with("vocals.wav")
    mock_cleanup.assert_called_once()
