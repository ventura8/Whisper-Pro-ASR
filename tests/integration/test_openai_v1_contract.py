"""OpenAI-compatible /v1/audio/* contract tests for the ASR routes.

Split out from test_bazarr_integration.py: this file covers the OpenAI API
surface (response_format/prompt aliasing, task inference from path, alias
equivalence, file-field precedence, full parameter matrix) as its own
concern, distinct from Bazarr-specific wire-format/local_path/schema tests.
"""

from __future__ import annotations

import io
import json
from unittest import mock

import pytest

from tests.conftest import FlaskCompatibleClient
from tests.integration.conftest import _BAZARR_TRANSCRIPTION_RESULT, mock_asr_manager_for_transcription


class TestOpenAIV1ContractMatrix:
    """Full v1/audio parameter matrix, cross-checked against OpenAI's documented
    /v1/audio/transcriptions and /v1/audio/translations contract (not just our own
    prior assumptions about what "OpenAI compatible" means)."""

    @pytest.mark.parametrize(
        "response_format,expect_in_body",
        [
            ("json", '"text"'),
            ("text", "Hello world"),
            ("srt", "-->"),
            ("verbose_json", '"text"'),
            ("vtt", "WEBVTT"),
        ],
    )
    def test_response_format_spec_values(
        self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str, response_format: str, expect_in_body: str
    ):
        """Every OpenAI-documented response_format value must be honored, not silently
        downgraded to srt (this was a real bug: 'text' and 'verbose_json' fell through
        to the srt branch before response_format aliasing was added)."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&response_format={response_format}")
        assert resp.status_code == 200
        assert expect_in_body in resp.data.decode()

    def test_response_format_text_matches_txt_plain_text_output(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """response_format=text (OpenAI's field name) must produce the same plain-text
        body as our internal 'txt' format, not SRT-formatted content."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            text_resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&response_format=text")
            txt_resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&output=txt")
        assert text_resp.data == txt_resp.data

    def test_response_format_verbose_json_matches_json_output(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """response_format=verbose_json must produce the same structured JSON body as
        our internal 'json' format."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            verbose_resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&response_format=verbose_json")
            json_resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&output=json")
        assert json.loads(verbose_resp.data) == json.loads(json_resp.data)

    def test_multipart_response_format_text_produces_plain_text_output(self, bazarr_client: FlaskCompatibleClient):
        """response_format=text sent as a multipart form field (alongside the audio upload,
        as OpenAI SDKs do) must produce plain-text output, not silently fall through to srt."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value="/tmp/converted.wav"):
            resp = bazarr_client.post(
                "/v1/audio/transcriptions",
                data={
                    "file": (io.BytesIO(b"RIFF" + b"\x00" * 40), "movie.wav"),
                    "response_format": "text",
                },
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        body = resp.data.decode()
        assert "Hello world" in body
        assert "-->" not in body

    def test_multipart_response_format_verbose_json_produces_json_output(self, bazarr_client: FlaskCompatibleClient):
        """response_format=verbose_json sent as a multipart form field must produce
        structured JSON output, matching the query-param path."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value="/tmp/converted.wav"):
            resp = bazarr_client.post(
                "/v1/audio/transcriptions",
                data={
                    "file": (io.BytesIO(b"RIFF" + b"\x00" * 40), "movie.wav"),
                    "response_format": "verbose_json",
                },
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["text"] == _BAZARR_TRANSCRIPTION_RESULT["text"]

    def test_unrecognized_response_format_falls_back_to_srt(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """An unrecognized response_format value must fall back to srt, not error."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&response_format=bogus")
        assert resp.status_code == 200
        assert "-->" in resp.data.decode()

    def test_prompt_field_reaches_transcription_same_as_initial_prompt(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """OpenAI SDKs send `prompt`, not `initial_prompt` — this was a real bug where
        `prompt` was silently dropped. Verify it now reaches model_manager.run_transcription
        as initial_prompt."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            mock_asr_manager_for_transcription(mock_mm)
            resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&prompt=proper+noun+glossary&output=json")
        assert resp.status_code == 200
        assert mock_mm.run_transcription.call_args.kwargs["initial_prompt"] == "proper noun glossary"

    def test_multipart_prompt_field_reaches_transcription_as_initial_prompt(self, bazarr_client: FlaskCompatibleClient):
        """OpenAI SDKs send `prompt` as a multipart form field alongside the audio upload,
        not as a query param -- verify that path also maps to initial_prompt."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value="/tmp/converted.wav"),
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            mock_asr_manager_for_transcription(mock_mm)
            resp = bazarr_client.post(
                "/v1/audio/transcriptions?output=json",
                data={
                    "file": (io.BytesIO(b"RIFF" + b"\x00" * 40), "movie.wav"),
                    "prompt": "multipart proper noun glossary",
                },
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        assert mock_mm.run_transcription.call_args.kwargs["initial_prompt"] == "multipart proper noun glossary"

    def test_initial_prompt_still_takes_priority_over_prompt(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Back-compat: our own `initial_prompt` param must still win if a caller sends both."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            mock_asr_manager_for_transcription(mock_mm)
            resp = bazarr_client.post(
                f"/v1/audio/transcriptions?local_path={bazarr_wav}&initial_prompt=explicit&prompt=fallback&output=json"
            )
        assert resp.status_code == 200
        assert mock_mm.run_transcription.call_args.kwargs["initial_prompt"] == "explicit"

    @pytest.mark.parametrize("model_value", ["whisper-1", "gpt-4o-transcribe", "some-unknown-model-string"])
    def test_model_field_is_accepted_and_ignored(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str, model_value: str):
        """Every OpenAI SDK call includes `model`; the server has one model and must not
        4xx just because a client sent this required-by-spec field."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&model={model_value}&output=json")
        assert resp.status_code == 200

    def test_temperature_timestamp_granularities_and_stream_are_gracefully_ignored(
        self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str
    ):
        """These OpenAI fields aren't implemented server-side; a spec-following client
        that sends them by default must still get a 200, not a rejection."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(
                f"/v1/audio/transcriptions?local_path={bazarr_wav}&output=json"
                "&temperature=0.2&timestamp_granularities[]=word&timestamp_granularities[]=segment&stream=false"
            )
        assert resp.status_code == 200

    @pytest.mark.parametrize("language_value", ["fr", "not-a-real-language-code", ""])
    def test_language_param_accepted_for_valid_unsupported_and_empty_values(
        self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str, language_value: str
    ):
        """language accepts a valid code, an unsupported code, and an empty value without erroring."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&language={language_value}&output=json")
        assert resp.status_code == 200

    def test_language_omitted_uses_auto_detect_path(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Omitting language entirely must route through auto-detection, not error."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&output=json")
        assert resp.status_code == 200


class TestTaskInferenceFromPath:
    """The /v1/audio/translations endpoint has no `task` field in the OpenAI spec — it
    always translates. Verify the path-based override at asr.py's request-parsing layer
    wins over any explicit `task` query param, in all four combinations."""

    def test_translations_endpoint_with_no_task_param_translates(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """/v1/audio/translations with no task param must still translate (path wins)."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            mock_asr_manager_for_transcription(mock_mm)
            resp = bazarr_client.post(f"/v1/audio/translations?local_path={bazarr_wav}&output=json")
        assert resp.status_code == 200
        assert mock_mm.run_transcription.call_args.args[2] == "translate"

    def test_translations_endpoint_ignores_explicit_task_transcribe(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """Path must win: even an explicit task=transcribe sent to /translations must still translate."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            mock_asr_manager_for_transcription(mock_mm)
            resp = bazarr_client.post(f"/v1/audio/translations?local_path={bazarr_wav}&task=transcribe&output=json")
        assert resp.status_code == 200
        assert mock_mm.run_transcription.call_args.args[2] == "translate"

    def test_transcriptions_endpoint_with_explicit_task_translate(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """/v1/audio/transcriptions with an explicit task=translate must translate."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            mock_asr_manager_for_transcription(mock_mm)
            resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&task=translate&output=json")
        assert resp.status_code == 200
        assert mock_mm.run_transcription.call_args.args[2] == "translate"

    def test_transcriptions_endpoint_with_task_omitted_transcribes(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """/v1/audio/transcriptions with task omitted must default to transcribe."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            mock_asr_manager_for_transcription(mock_mm)
            resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&output=json")
        assert resp.status_code == 200
        assert mock_mm.run_transcription.call_args.args[2] == "transcribe"


class TestAliasEquivalence:
    """/asr, /v1/audio/transcriptions, and /v1/audio/translations (with task=translate)
    must behave identically for a fixed param set, so an edit to one alias can't silently
    diverge from the others."""

    @pytest.mark.parametrize("path", ["/asr", "/v1/audio/transcriptions"])
    def test_transcription_aliases_produce_identical_output(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str, path: str):
        """/asr and /v1/audio/transcriptions must produce identical output for the same params."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"{path}?local_path={bazarr_wav}&output=json&prompt=glossary")
        assert resp.status_code == 200
        assert json.loads(resp.data)["text"] == _BAZARR_TRANSCRIPTION_RESULT["text"]

    def test_translation_alias_forces_translate_task(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """/v1/audio/translations must force task=translate regardless of other params."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            mock_asr_manager_for_transcription(mock_mm)
            resp = bazarr_client.post(f"/v1/audio/translations?local_path={bazarr_wav}&output=json")
        assert resp.status_code == 200
        assert mock_mm.run_transcription.call_args.args[2] == "translate"


class TestFileFieldNamePrecedence:
    """OpenAI SDKs always name the multipart field `file`; Bazarr's own client (see
    class docstring in TestBazarrUploadFallback) uses `audio_file`. Both must work, and
    sending both at once must have defined, tested precedence."""

    def _wav_bytes(self) -> bytes:
        return b"RIFF" + b"\x00" * 40

    def test_file_field_name_accepted(self, bazarr_client: FlaskCompatibleClient):
        """OpenAI's multipart field name `file` must be accepted."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value="/tmp/converted.wav"):
            resp = bazarr_client.post(
                "/v1/audio/transcriptions?output=json",
                data={"file": (io.BytesIO(self._wav_bytes()), "movie.wav")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200

    def test_audio_file_field_name_accepted(self, bazarr_client: FlaskCompatibleClient):
        """Bazarr's multipart field name `audio_file` must be accepted."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value="/tmp/converted.wav"):
            resp = bazarr_client.post(
                "/v1/audio/transcriptions?output=json",
                data={"audio_file": (io.BytesIO(self._wav_bytes()), "movie.wav")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200

    def test_audio_file_takes_precedence_when_both_fields_sent(self, bazarr_client: FlaskCompatibleClient):
        """Document current, intentional precedence: audio_file wins over file when both
        are present, matching the route signature's parameter order. Verify this by giving
        each field distinct content and checking the bytes actually written to the
        temp file convert_to_wav receives came from audio_file, not file."""
        audio_file_bytes = self._wav_bytes() + b"PRIORITY_MARKER"
        file_bytes = self._wav_bytes() + b"IGNORED_MARKER"
        captured_contents: list[bytes] = []

        def _capture_convert(path: str, **_kwargs: object) -> str:
            with open(path, "rb") as f:
                captured_contents.append(f.read())
            return "/tmp/converted.wav"

        with mock.patch("modules.core.utils.convert_to_wav", side_effect=_capture_convert):
            resp = bazarr_client.post(
                "/v1/audio/transcriptions?output=json",
                data={
                    "audio_file": (io.BytesIO(audio_file_bytes), "priority.wav"),
                    "file": (io.BytesIO(file_bytes), "ignored.wav"),
                },
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        assert len(captured_contents) == 1
        assert b"PRIORITY_MARKER" in captured_contents[0]
        assert b"IGNORED_MARKER" not in captured_contents[0]


class TestV1FullParamSurface:
    """The v1 aliases must honor the full RequestParams surface, not just the narrow
    param set used by the local_path/upload happy-path tests above."""

    FULL_PARAM_QUERY = (
        "diarize=true&min_speakers=1&max_speakers=3"
        "&word_timestamps=true&vad_filter=false"
        "&max_line_width=40&max_line_count=2"
        "&initial_prompt=proper+noun+glossary"
    )

    def _mock_manager(self, mock_mm: mock.MagicMock) -> None:
        mock_asr_manager_for_transcription(mock_mm)

    @pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/v1/audio/translations"])
    def test_full_param_surface_individually_and_combined(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str, path: str):
        """Every RequestParams field must be honored, individually and combined, through both v1 aliases."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            self._mock_manager(mock_mm)
            resp = bazarr_client.post(f"{path}?local_path={bazarr_wav}&output=json&{self.FULL_PARAM_QUERY}")
        assert resp.status_code == 200
        kwargs = mock_mm.run_transcription.call_args.kwargs
        expected = {
            "diarize": True,
            "min_speakers": 1,
            "max_speakers": 3,
            "word_timestamps": True,
            "vad_filter": False,
            "initial_prompt": "proper noun glossary",
        }
        assert {key: kwargs[key] for key in expected} == expected

    @pytest.mark.parametrize(
        "path,output,generate_target",
        [("/v1/audio/transcriptions", "srt", "generate_srt"), ("/v1/audio/translations", "vtt", "generate_vtt")],
    )
    def test_subtitle_line_wrap_params_reach_generate_srt_or_vtt(
        self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str, path: str, output: str, generate_target: str
    ):
        """max_line_width/max_line_count must reach the actual subtitle-formatting call
        (utils.generate_srt/generate_vtt), not just be accepted without erroring."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
            mock.patch(f"modules.api.routes.asr.utils.{generate_target}", return_value="mocked-subtitle-body") as mock_generate,
        ):
            resp = bazarr_client.post(f"{path}?local_path={bazarr_wav}&output={output}&max_line_width=40&max_line_count=2")
        assert resp.status_code == 200
        mock_generate.assert_called_once()
        assert mock_generate.call_args.kwargs["max_line_width"] == 40
        assert mock_generate.call_args.kwargs["max_line_count"] == 2

    @pytest.mark.parametrize(
        "param,value",
        [
            ("diarize", "true"),
            ("min_speakers", "2"),
            ("max_speakers", "4"),
            ("word_timestamps", "true"),
            ("vad_filter", "false"),
            ("max_line_width", "35"),
            ("max_line_count", "1"),
            ("initial_prompt", "context+hint"),
            ("hf_token", "hf_dummy_token"),
        ],
    )
    def test_each_param_individually_through_v1_transcriptions(
        self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str, param: str, value: str
    ):
        """Each optional param must be accepted on its own through /v1/audio/transcriptions."""
        with mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav):
            resp = bazarr_client.post(f"/v1/audio/transcriptions?local_path={bazarr_wav}&output=json&{param}={value}")
        assert resp.status_code == 200

    def test_hf_token_forwarded_via_header(self, bazarr_client: FlaskCompatibleClient, bazarr_wav: str):
        """hf_token supplied via the X-HF-Token header must reach the transcription call."""
        with (
            mock.patch("modules.core.utils.convert_to_wav", return_value=bazarr_wav),
            mock.patch("modules.api.routes.asr.model_manager") as mock_mm,
        ):
            self._mock_manager(mock_mm)
            resp = bazarr_client.post(
                f"/v1/audio/transcriptions?local_path={bazarr_wav}&output=json&diarize=true",
                headers={"X-HF-Token": "hf_via_header"},
            )
        assert resp.status_code == 200
        assert mock_mm.run_transcription.call_args.kwargs["hf_token"] == "hf_via_header"
