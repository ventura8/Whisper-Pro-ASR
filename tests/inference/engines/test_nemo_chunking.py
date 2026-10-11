"""Cutting a request into NeMo-sized chunks (modules/inference/engines/nemo_chunking.py)."""

from unittest import mock

import numpy as np
import pytest

from modules.core import config
from modules.inference.engines import nemo_chunking
from tests.inference.engines.nemo_fixtures import speech


@pytest.mark.parametrize("value", [None, [], "", "0,5"])
def test_no_clips_means_none_never_an_empty_decode(value):
    """An empty list must not read as "decode nothing" -- that would drop the whole request."""
    assert nemo_chunking.clips_from_kwargs(value, 30.0) is None


def test_flat_clip_list_becomes_pairs():
    """The flat seek list becomes (start, end) pairs."""
    assert nemo_chunking.clips_from_kwargs([1.0, 2.5, 4, 6], 30.0) == [(1.0, 2.5), (4.0, 6.0)]


def test_odd_clip_list_runs_its_last_clip_to_the_end_of_the_file():
    """faster-whisper's convention, which the pipeline writes to."""
    assert nemo_chunking.clips_from_kwargs([1.0, 2.0, 5.0], 12.0) == [(1.0, 2.0), (5.0, 12.0)]


def test_quietest_point_finds_the_pause():
    """A cut lands in the silent frame, not through speech."""
    audio = speech(20.0, quiet_at=13.0)
    assert nemo_chunking.quietest_point(audio, 10.0, 20.0) == pytest.approx(13.0, abs=0.01)


def test_quietest_point_falls_back_to_the_limit_with_less_than_a_frame():
    """Too little audio to search: cut at the limit."""
    audio = speech(1.0)
    assert nemo_chunking.quietest_point(audio, 0.98, 1.0) == 1.0


def test_long_spans_are_split_at_quiet_points_until_none_is_too_long():
    """A 50 s span becomes contiguous pieces, none over the limit."""
    audio = speech(50.0, quiet_at=14.0)
    spans = nemo_chunking.bounded([(0.0, 50.0)], audio, 20.0)
    assert spans[0] == (0.0, pytest.approx(14.0, abs=0.01))
    assert all(end - start <= 20.0 for start, end in spans)
    assert spans[-1][1] == 50.0
    # Contiguous: nothing between two chunks is lost.
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_plan_prefers_the_callers_clips():
    """The pipeline's own regions win; VAD is not run again."""
    audio = speech(10.0)
    with mock.patch.object(nemo_chunking.vad, "get_speech_timestamps") as vad:
        spans = nemo_chunking.plan(audio, clip_timestamps=[1.0, 3.0], vad_filter=True, vad_parameters=None, max_sec=20.0)
    assert spans == [(1.0, 3.0)]
    vad.assert_not_called()


def test_plan_runs_vad_with_the_requests_settings():
    """Without clips, Silero runs with the request's threshold and pause length."""
    audio = speech(10.0)
    regions = [{"start": 0.5, "end": 2.0}, {"start": 4.0, "end": 9.0}]
    with mock.patch.object(nemo_chunking.vad, "get_speech_timestamps", return_value=regions) as vad:
        spans = nemo_chunking.plan(
            audio, clip_timestamps=None, vad_filter=True, vad_parameters={"threshold": 0.6, "min_silence_duration_ms": 900}, max_sec=20.0
        )
    assert spans == [(0.5, 2.0), (4.0, 9.0)]
    vad.assert_called_once_with(audio, 0.6, 900, config.VAD_SPEECH_PAD_MS)


def test_plan_without_vad_decodes_every_sample_in_bounded_pieces():
    """vad_filter=False decodes the whole file, in pieces the model accepts."""
    audio = speech(45.0)
    spans = nemo_chunking.plan(audio, clip_timestamps=None, vad_filter=False, vad_parameters=None, max_sec=20.0)
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 45.0
    assert all(end - start <= 20.0 for start, end in spans)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:])), "no sample is skipped"


def test_plan_drops_slivers_and_clamps_to_the_audio():
    """Canary turned 50 ms of silence into a full sentence; nothing that short is decoded."""
    audio = speech(10.0)
    spans = nemo_chunking.plan(audio, clip_timestamps=[0.0, 0.05, 9.0, 14.0], vad_filter=False, vad_parameters=None, max_sec=20.0)
    assert spans == [(9.0, 10.0)]


def test_no_speech_found_is_logged_and_decodes_nothing(caplog):
    """Silence is an empty decode, and the log says why."""
    with mock.patch.object(nemo_chunking.vad, "get_speech_timestamps", return_value=[]):
        spans = nemo_chunking.plan(speech(3.0), clip_timestamps=None, vad_filter=True, vad_parameters=None, max_sec=20.0)
    assert spans == []
    assert "found no speech" in caplog.text


def test_batches_keep_time_order():
    """Batches are consecutive, so segments come back in time order."""
    assert nemo_chunking.batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert nemo_chunking.batches([1, 2], 0) == [[1], [2]]


def test_samples_slice_one_span_as_contiguous_float32():
    """A span's samples are a contiguous float32 copy."""
    audio = np.arange(32000, dtype=np.float64)
    chunk = nemo_chunking.samples(audio, (0.5, 1.0))
    assert chunk.dtype == np.float32
    assert chunk.flags["C_CONTIGUOUS"]
    assert chunk[0] == 8000
    assert len(chunk) == 8000


def test_decode_reads_a_path_and_passes_samples_through():
    """A path is decoded; decoded samples are only converted."""
    with mock.patch.object(nemo_chunking.vad, "decode_audio", return_value=[0.0, 0.5]) as decode:
        assert nemo_chunking.decode("clip.wav").dtype == np.float32
    decode.assert_called_once_with("clip.wav")
    assert nemo_chunking.decode([0.25]).tolist() == [0.25]
