"""Tests for server-side replication of Bazarr's audio stream-selection and
delay-correction logic (modules/core/utils_helpers.py's build_stream_alignment_directives
and its helpers), used only for local_path/video_file-resolved sources -- never for
uploaded audio_file content."""

import json
from unittest import mock

from modules.core import utils_helpers

_probe_streams_and_packets = getattr(utils_helpers, "_probe_streams_and_packets")
_extract_audio_streams = getattr(utils_helpers, "_extract_audio_streams")
_select_audio_stream_index = getattr(utils_helpers, "_select_audio_stream_index")
_first_packet_pts_for_stream = getattr(utils_helpers, "_first_packet_pts_for_stream")
_delay_filter_for_ms = getattr(utils_helpers, "_delay_filter_for_ms")
_build_ffmpeg_standardization_cmd = getattr(utils_helpers, "_build_ffmpeg_standardization_cmd")


def _probe_json(streams=None, packets=None) -> str:
    """Build a fake ffprobe -of json response string for the given streams/packets."""
    return json.dumps({"streams": streams or [], "packets": packets or []})


def test_probe_streams_and_packets_parses_json():
    """A successful ffprobe call should return the parsed JSON dict."""
    fake_output = _probe_json(streams=[{"index": 1, "tags": {"language": "eng"}}])
    with mock.patch("modules.core.process_exec.check_output_text", return_value=fake_output):
        result = _probe_streams_and_packets("movie.mkv")
    assert result["streams"][0]["index"] == 1


def test_probe_streams_and_packets_returns_empty_dict_on_ffprobe_failure():
    """A failing ffprobe invocation must not raise -- callers treat {} as no data."""
    with mock.patch(
        "modules.core.process_exec.check_output_text",
        side_effect=utils_helpers.process_exec.CommandExecutionError(["ffprobe"], 1, "boom"),
    ):
        assert _probe_streams_and_packets("movie.mkv") == {}


def test_probe_streams_and_packets_returns_empty_dict_on_malformed_json():
    """Malformed ffprobe stdout must not raise -- degrade to {} instead."""
    with mock.patch("modules.core.process_exec.check_output_text", return_value="not json"):
        assert _probe_streams_and_packets("movie.mkv") == {}


def test_extract_audio_streams_maps_index_and_language():
    """Each stream's index/language tag should be flattened into a simple dict."""
    probe_result = {"streams": [{"index": 2, "tags": {"language": "spa"}}, {"index": 3, "tags": {}}]}
    streams = _extract_audio_streams(probe_result)
    assert streams == [{"index": 2, "language": "spa"}, {"index": 3, "language": None}]


def test_extract_audio_streams_handles_missing_streams_key():
    """A probe result with no "streams" key should yield an empty list, not raise."""
    assert not _extract_audio_streams({})


def test_select_audio_stream_index_matches_by_substring():
    """Mirrors Bazarr's `target_lang_short in stream_lang` substring matching."""
    streams = [{"index": 1, "language": "eng"}, {"index": 2, "language": "fra"}]
    assert _select_audio_stream_index(streams, "en") == 1
    assert _select_audio_stream_index(streams, "fr") == 2


def test_select_audio_stream_index_no_match_returns_none():
    """No matching stream language should return None (no explicit -map)."""
    streams = [{"index": 1, "language": "eng"}]
    assert _select_audio_stream_index(streams, "fr") is None


def test_select_audio_stream_index_no_target_language_returns_none():
    """No target language at all should return None."""
    streams = [{"index": 1, "language": "eng"}]
    assert _select_audio_stream_index(streams, None) is None


def test_select_audio_stream_index_no_streams_returns_none():
    """An empty stream list should return None."""
    assert _select_audio_stream_index([], "en") is None


def test_select_audio_stream_index_skips_streams_without_language_tag():
    """A stream with no language tag should never match, even if others do."""
    streams = [{"index": 1, "language": None}, {"index": 2, "language": "eng"}]
    assert _select_audio_stream_index(streams, "en") == 2


def test_first_packet_pts_for_stream_filters_by_index():
    """The first packet belonging to the given stream_index should be selected."""
    probe_result = {"packets": [{"stream_index": 0, "pts_time": "1.0"}, {"stream_index": 1, "pts_time": "0.045"}]}
    assert _first_packet_pts_for_stream(probe_result, 1) == 0.045


def test_first_packet_pts_for_stream_no_stream_index_uses_first_packet():
    """With no target stream_index, the first packet overall should be used."""
    probe_result = {"packets": [{"stream_index": 0, "pts_time": "0.5"}]}
    assert _first_packet_pts_for_stream(probe_result, None) == 0.5


def test_first_packet_pts_for_stream_returns_none_on_missing_data():
    """No packets, or a packet with no pts_time, should return None."""
    assert _first_packet_pts_for_stream({"packets": []}, 0) is None
    assert _first_packet_pts_for_stream({"packets": [{"stream_index": 0, "pts_time": None}]}, 0) is None


def test_first_packet_pts_for_stream_returns_none_on_unparseable_pts():
    """An unparseable pts_time value should return None, not raise."""
    probe_result = {"packets": [{"stream_index": 0, "pts_time": "not-a-number"}]}
    assert _first_packet_pts_for_stream(probe_result, 0) is None


def test_delay_filter_for_ms_positive_delay_builds_adelay():
    """A delay past the sync threshold (audio starts late) builds an adelay filter."""
    assert _delay_filter_for_ms(45) == "adelay=45:all=1"


def test_delay_filter_for_ms_negative_delay_builds_atrim():
    """A delay past the negative sync threshold (audio starts early) builds an atrim filter."""
    assert _delay_filter_for_ms(-500) == "atrim=start=0.5,asetpts=PTS-STARTPTS"


def test_delay_filter_for_ms_within_threshold_returns_none():
    """Delays within +/-20ms (Bazarr's own SYNC_THRESHOLD) should not be corrected."""
    assert _delay_filter_for_ms(20) is None
    assert _delay_filter_for_ms(-20) is None
    assert _delay_filter_for_ms(0) is None


def test_build_stream_alignment_directives_positive_delay():
    """End-to-end: matching stream + positive delay yields both directives."""
    fake_output = _probe_json(
        streams=[{"index": 1, "tags": {"language": "eng"}}],
        packets=[{"stream_index": 1, "pts_time": "0.045"}],
    )
    with mock.patch("modules.core.process_exec.check_output_text", return_value=fake_output):
        stream_index, delay_filter = utils_helpers.build_stream_alignment_directives("movie.mkv", "en")
    assert stream_index == 1
    assert delay_filter == "adelay=45:all=1"


def test_build_stream_alignment_directives_negative_delay():
    """End-to-end: no target language (no -map) but a negative delay still corrects."""
    fake_output = _probe_json(
        streams=[{"index": 0, "tags": {"language": "eng"}}],
        packets=[{"stream_index": 0, "pts_time": "-0.5"}],
    )
    with mock.patch("modules.core.process_exec.check_output_text", return_value=fake_output):
        stream_index, delay_filter = utils_helpers.build_stream_alignment_directives("movie.mkv", None)
    assert stream_index is None  # no target language -> no explicit -map
    assert delay_filter == "atrim=start=0.5,asetpts=PTS-STARTPTS"


def test_build_stream_alignment_directives_skips_delay_when_multi_stream_unmatched():
    """Multi-track media with no language match must not invent a delay from an arbitrary packet."""
    fake_output = _probe_json(
        streams=[
            {"index": 0, "tags": {"language": "eng"}},
            {"index": 1, "tags": {"language": "fre"}},
        ],
        packets=[{"stream_index": 0, "pts_time": "0.5"}],
    )
    with mock.patch("modules.core.process_exec.check_output_text", return_value=fake_output):
        stream_index, delay_filter = utils_helpers.build_stream_alignment_directives("movie.mkv", "deu")
    assert (stream_index, delay_filter) == (None, None)


def test_build_stream_alignment_directives_within_threshold_no_filter():
    """A delay within the sync threshold yields a stream index but no filter."""
    fake_output = _probe_json(
        streams=[{"index": 1, "tags": {"language": "eng"}}],
        packets=[{"stream_index": 1, "pts_time": "0.010"}],
    )
    with mock.patch("modules.core.process_exec.check_output_text", return_value=fake_output):
        stream_index, delay_filter = utils_helpers.build_stream_alignment_directives("movie.mkv", "en")
    assert stream_index == 1
    assert delay_filter is None


def test_build_stream_alignment_directives_no_pts_data_returns_stream_index_only():
    """No packet data at all should still allow stream selection, just no delay filter."""
    fake_output = _probe_json(streams=[{"index": 1, "tags": {"language": "eng"}}], packets=[])
    with mock.patch("modules.core.process_exec.check_output_text", return_value=fake_output):
        stream_index, delay_filter = utils_helpers.build_stream_alignment_directives("movie.mkv", "en")
    assert stream_index == 1
    assert delay_filter is None


def test_build_stream_alignment_directives_probe_failure_never_raises():
    """Any failure anywhere in the chain must degrade to (None, None), not raise."""
    with mock.patch("modules.core.process_exec.check_output_text", side_effect=RuntimeError("ffprobe missing")):
        stream_index, delay_filter = utils_helpers.build_stream_alignment_directives("movie.mkv", "en")
    assert (stream_index, delay_filter) == (None, None)


def test_build_stream_alignment_directives_unexpected_internal_error_never_raises():
    """Even an unexpected exception deep in post-processing must not escape."""
    with mock.patch.object(utils_helpers, "_select_audio_stream_index", side_effect=ValueError("boom")):
        with mock.patch("modules.core.process_exec.check_output_text", return_value=_probe_json(streams=[{"index": 1}])):
            stream_index, delay_filter = utils_helpers.build_stream_alignment_directives("movie.mkv", "en")
    assert (stream_index, delay_filter) == (None, None)


def test_build_ffmpeg_standardization_cmd_defaults_unchanged():
    """No stream_index/delay_filter -> byte-for-byte identical to today's command."""
    cmd = _build_ffmpeg_standardization_cmd("in.mkv", "out.wav", ["-vn", "-ar", "16000"], "dynaudnorm=f=150:g=15")
    assert "-map" not in cmd
    assert cmd[cmd.index("-af") + 1] == "dynaudnorm=f=150:g=15"


def test_build_ffmpeg_standardization_cmd_inserts_map_in_output_args():
    """-map must land after -i (output-side), never before it."""
    cmd = _build_ffmpeg_standardization_cmd("in.mkv", "out.wav", ["-vn", "-ar", "16000"], "dynaudnorm=f=150:g=15", stream_index=2)
    i_index = cmd.index("-i")
    map_index = cmd.index("-map")
    assert map_index > i_index
    assert cmd[map_index + 1] == "0:2"


def test_build_ffmpeg_standardization_cmd_no_map_when_stream_index_none():
    """No stream_index means no -map is added at all."""
    cmd = _build_ffmpeg_standardization_cmd("in.mkv", "out.wav", ["-vn", "-ar", "16000"], "dynaudnorm=f=150:g=15", stream_index=None)
    assert "-map" not in cmd


def test_build_ffmpeg_standardization_cmd_prepends_delay_filter_to_af():
    """The delay/trim filter must run before loudness normalization in the -af chain."""
    cmd = _build_ffmpeg_standardization_cmd(
        "in.mkv", "out.wav", ["-vn", "-ar", "16000"], "dynaudnorm=f=150:g=15", delay_filter="adelay=45:all=1"
    )
    assert cmd[cmd.index("-af") + 1] == "adelay=45:all=1,dynaudnorm=f=150:g=15"


def test_build_ffmpeg_standardization_cmd_input_flags_stay_before_dash_i():
    """The unrelated raw-PCM input_flags concept must stay before -i, unaffected by -map."""
    cmd = _build_ffmpeg_standardization_cmd(
        "in.raw",
        "out.wav",
        ["-vn", "-ar", "16000"],
        "dynaudnorm=f=150:g=15",
        input_flags=["-f", "s16le"],
        stream_index=0,
    )
    i_index = cmd.index("-i")
    input_flag_start = i_index - 2
    assert cmd[input_flag_start:i_index] == ["-f", "s16le"]
    assert cmd.index("-map") > i_index
