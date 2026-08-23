"""Bazarr real wire-format and local_path security integration tests.

Split from ``test_bazarr_integration.py`` to stay under the 500-line Python file limit.
"""

from __future__ import annotations

import io
import json
from unittest import mock

import pytest

from tests.conftest import FlaskCompatibleClient

_DEFAULT_TRANSCRIPTION_RESULT = {
    "text": "Hello world from Bazarr",
    "segments": [
        {"timestamp": (0.0, 2.5), "text": "Hello world"},
        {"timestamp": (2.5, 5.0), "text": "from Bazarr"},
    ],
}


class TestRealBazarrWireFormat:
    """Real Bazarr (morpheus65535/bazarr, custom_libs/subliminal_patch/providers/whisperai.py)
    never sends local_path — it always uploads pre-encoded raw s16le/16kHz/mono PCM as a
    multipart `audio_file`, with encode=false, task, language, output=srt, and an optional
    video_file param (caller metadata only, never a path to resolve). This is the actual
    wire shape real Bazarr uses, distinct from the local_path optimization tested in
    ``test_bazarr_integration.py``."""

    def _raw_pcm_bytes(self) -> bytes:
        # Minimal stand-in for ffmpeg-encoded raw s16le/16kHz/mono PCM; the server only
        # needs bytes here since convert_to_wav is mocked.
        return b"\x00\x01" * 1000

    def test_asr_multipart_raw_pcm_encode_false_transcribe(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Bazarr's real wire format for a transcribe request: multipart raw PCM, encode=false."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert:
            resp = bazarr_client.post(
                "/asr?task=transcribe&language=en&output=srt&encode=false&video_file=",
                data={"audio_file": (io.BytesIO(self._raw_pcm_bytes()), "audio.raw")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        assert b"Hello world" in resp.data
        assert b"-->" in resp.data
        mock_convert.assert_called_once()

    def test_asr_multipart_raw_pcm_encode_false_translate(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Bazarr's real wire format for a translate request: multipart raw PCM, encode=false."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert:
            resp = bazarr_client.post(
                "/asr?task=translate&language=fr&output=srt&encode=false",
                data={"audio_file": (io.BytesIO(self._raw_pcm_bytes()), "audio.raw")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        assert b"-->" in resp.data
        mock_convert.assert_called_once()

    def test_asr_video_file_param_present_takes_priority_over_uploaded_audio(
        self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str, tmp_path
    ):
        """video_file is caller metadata Bazarr sends identifying the real source media
        path (populated when pass_video_name is enabled). Now that we replicate
        Bazarr's own audio-track selection + delay correction server-side (see
        utils.get_stream_alignment_directives), a resolvable video_file takes priority
        over uploaded audio -- like the existing local_path zero-copy optimization --
        since it operates on the original, full-quality source and skips the wasted
        upload entirely. Uses a real, readable video_file path whose content is
        distinct from the uploaded audio, then asserts run_transcription actually
        receives the video_file path (not the uploaded audio) as its audio input, and
        that stream alignment was probed against it. Also asserts the request's
        encode=false raw-PCM input_flags (meant for the now-bypassed upload) are
        NOT applied to the video_file container -- that would make FFmpeg misinterpret
        a real media container as headerless PCM."""
        real_video_file = tmp_path / "movies" / "Movie (2024).mkv"
        real_video_file.parent.mkdir(parents=True)
        real_video_file.write_bytes(b"REAL-VIDEO-CONTAINER-BYTES-NOT-THE-UPLOAD")
        resolved_video_path = str(real_video_file.resolve())

        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert,
            mock.patch("modules.core.utils.get_stream_alignment_directives", return_value=(None, None)) as mock_alignment,
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            mock_mm.is_engine_initialized.return_value = True
            mock_mm.increment_active_session.return_value = None
            mock_mm.decrement_active_session.return_value = None
            mock_mm.run_transcription.return_value = _DEFAULT_TRANSCRIPTION_RESULT
            resp = bazarr_client.post(
                f"/asr?task=transcribe&output=srt&encode=false&language=en&video_file={real_video_file}",
                data={"audio_file": (io.BytesIO(self._raw_pcm_bytes()), "audio.raw")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200
        # convert_to_wav (and therefore the alignment probe) must run against the
        # video_file source, not the uploaded audio bytes.
        mock_convert.assert_called_once_with(resolved_video_path, input_flags=[], stream_index=None, delay_filter=None)
        mock_alignment.assert_called_once_with(resolved_video_path, "en")

    def test_asr_video_file_param_absent_default_behavior(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """pass_video_name disabled (Bazarr's default): video_file is simply omitted."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(
                "/asr?task=transcribe&output=srt&encode=false",
                data={"audio_file": (io.BytesIO(self._raw_pcm_bytes()), "audio.raw")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200

    def test_detect_language_multipart_raw_pcm_matches_bazarr_client_expectations(self, bazarr_client: FlaskCompatibleClient):
        """Bazarr's detect_language() parses language_code + detected_language from the
        JSON body and treats 'und'/empty language_code as 'no detection', not an error."""
        with mock.patch("modules.core.utils.get_audio_duration", return_value=90.0):
            resp = bazarr_client.post(
                "/detect-language?encode=false&video_file=",
                data={"audio_file": (io.BytesIO(self._raw_pcm_bytes()), "audio.raw")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "language_code" in data
        assert "detected_language" in data

    def test_detect_language_und_or_empty_code_is_not_an_error(self, bazarr_client: FlaskCompatibleClient):
        """When Whisper can't determine a language, Bazarr expects a normal 200 response
        with an empty/'und' language_code, not an HTTP error — it treats that as a
        detection-failed sentinel client-side."""
        with (
            mock.patch("modules.core.utils.get_audio_duration", return_value=60.0),
            mock.patch("modules.inference.pipeline.language_detection.run_voting_detection") as mock_ld,
        ):
            mock_ld.return_value = {
                "confidence": 0.0,
                "detected_language": "",
                "language": "und",
                "language_code": "und",
            }
            resp = bazarr_client.post(
                "/detect-language?encode=false",
                data={"audio_file": (io.BytesIO(self._raw_pcm_bytes()), "audio.raw")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["language_code"] == "und"


def test_local_path_outside_approved_roots_is_rejected(bazarr_client: FlaskCompatibleClient):
    """local_path is access-controlled (see get_approved_roots/is_path_approved in
    modules/api/support/local_path.py): a path-traversal-style local_path pointing at
    an unmounted/unapproved location (e.g. outside TEMP_DIR/PERSISTENT_DIR/cwd/
    APPROVED_ROOTS) must be rejected, not read, even if the file exists on disk."""
    resp = bazarr_client.post("/asr?local_path=/etc/passwd&output=json")
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "not accessible" in data["error"].lower() or "unmapped" in data["error"].lower()


def test_detect_language_video_file_only_resolves_as_last_resort_path(bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
    """Regression: a request supplying ONLY `video_file` (no local_path, no uploaded
    audio -- e.g. manual testing/debugging against a volume-mapped media library) must
    still resolve and succeed, exactly like a real local_path request, rather than
    erroring with 'No audio source provided'. This is a last-resort fallback only: it
    goes through the same approved-roots gate as local_path (see resolve_local_path).
    See test_asr_video_file_param_present_takes_priority_over_uploaded_audio above,
    which asserts a resolvable video_file now takes priority over uploaded audio."""
    with mock.patch("modules.core.utils.get_audio_duration", return_value=90.0):
        resp = bazarr_client.post(f"/detect-language?video_file={bazarr_wav}")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["detected_language"] == "en"


def test_asr_local_path_triggers_stream_alignment_probing(bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
    """A local_path request (no uploaded audio_file) must be probed for stream
    selection/delay correction, with the request's target language forwarded, and the
    resulting directives threaded into convert_to_wav."""
    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert,
        mock.patch("modules.core.utils.get_stream_alignment_directives", return_value=(1, "adelay=45:all=1")) as mock_alignment,
    ):
        resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&language=en&output=json")
    assert resp.status_code == 200
    mock_alignment.assert_called_once_with(bazarr_wav, "en")
    mock_convert.assert_called_once_with(bazarr_wav, input_flags=[], stream_index=1, delay_filter="adelay=45:all=1")


def test_asr_video_file_only_triggers_stream_alignment_probing(bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
    """The video_file-only fallback (no local_path, no uploaded audio) must also be
    probed for stream alignment, exactly like local_path."""
    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert,
        mock.patch("modules.core.utils.get_stream_alignment_directives", return_value=(None, None)) as mock_alignment,
    ):
        resp = bazarr_client.post(f"/asr?video_file={bazarr_wav}&output=json")
    assert resp.status_code == 200
    mock_alignment.assert_called_once_with(bazarr_wav, None)
    mock_convert.assert_called_once_with(bazarr_wav, input_flags=[], stream_index=None, delay_filter=None)


def test_asr_uploaded_audio_file_never_triggers_stream_alignment_probing(bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
    """A genuine multipart audio_file upload must never be probed -- Bazarr's own
    client already applied stream selection and delay correction before uploading."""
    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert,
        mock.patch("modules.core.utils.get_stream_alignment_directives") as mock_alignment,
    ):
        resp = bazarr_client.post(
            "/asr?task=transcribe&language=en&output=srt&encode=false",
            data={"audio_file": (io.BytesIO(b"\x00\x01" * 1000), "audio.raw")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    mock_alignment.assert_not_called()
    _, call_kwargs = mock_convert.call_args
    assert call_kwargs["stream_index"] is None
    assert call_kwargs["delay_filter"] is None


def test_asr_local_path_stream_alignment_probe_failure_falls_back_silently(bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
    """A real probing failure (e.g. ffprobe missing/erroring) must not block
    transcription -- it must fall back to today's unmodified behavior."""
    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert,
        mock.patch("modules.core.utils_helpers._probe_streams_and_packets", side_effect=RuntimeError("ffprobe missing")),
    ):
        resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&language=en&output=json")
    assert resp.status_code == 200
    mock_convert.assert_called_once_with(bazarr_wav, input_flags=[], stream_index=None, delay_filter=None)


def test_asr_unresolvable_video_file_falls_back_to_uploaded_audio(bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
    """A video_file that does not resolve (outside approved roots / does not exist)
    must fall back to the uploaded audio_file -- exactly like an unresolvable
    local_path does -- and must not trigger stream-alignment probing, since the
    upload (not video_file) ends up as the actual transcription source."""
    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert,
        mock.patch("modules.core.utils.get_stream_alignment_directives") as mock_alignment,
    ):
        resp = bazarr_client.post(
            "/asr?task=transcribe&output=srt&encode=false&video_file=/etc/passwd",
            data={"audio_file": (io.BytesIO(b"\x00\x01" * 1000), "audio.raw")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    mock_alignment.assert_not_called()
    _, call_kwargs = mock_convert.call_args
    assert call_kwargs["stream_index"] is None
    assert call_kwargs["delay_filter"] is None


@pytest.mark.parametrize("endpoint", ["/asr", "/v1/audio/transcriptions", "/v1/audio/translations"])
def test_stream_alignment_probing_applies_identically_across_all_transcription_aliases(
    bazarr_client: FlaskCompatibleClient, bazarr_wav: str, endpoint: str
):
    """/asr, /v1/audio/transcriptions, and /v1/audio/translations share the same
    handler -- stream-alignment probing must trigger identically for a local_path
    request through any of the three aliases."""
    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav) as mock_convert,
        mock.patch("modules.core.utils.get_stream_alignment_directives", return_value=(1, "adelay=45:all=1")) as mock_alignment,
    ):
        resp = bazarr_client.post(f"{endpoint}?local_path={bazarr_wav}&language=en&output=json")
    assert resp.status_code == 200
    mock_alignment.assert_called_once_with(bazarr_wav, "en")
    mock_convert.assert_called_once_with(bazarr_wav, input_flags=[], stream_index=1, delay_filter="adelay=45:all=1")


@pytest.mark.parametrize("language", ["EN", "en-US", "eng"])
def test_asr_local_path_stream_alignment_forwards_language_as_given(bazarr_client: FlaskCompatibleClient, bazarr_wav: str, language: str):
    """Whatever language format the caller sends (uppercase, locale-style, alpha-3)
    must be forwarded to the alignment probe verbatim -- stream-matching's own
    substring logic (see utils_helpers._select_audio_stream_index) is responsible
    for normalizing/truncating it, not this call site."""
    with (
        mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
        mock.patch("modules.core.utils.get_stream_alignment_directives", return_value=(None, None)) as mock_alignment,
    ):
        resp = bazarr_client.post(f"/asr?local_path={bazarr_wav}&language={language}&output=json")
    assert resp.status_code == 200
    mock_alignment.assert_called_once_with(bazarr_wav, language)
