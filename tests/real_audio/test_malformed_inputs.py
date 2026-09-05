"""Files that are not really audio.

A truncated header, an empty upload, an MP3 wearing a .wav name, a file containing only
noise or tones. The service is not expected to transcribe these -- most have nothing to
transcribe. It is expected to answer *promptly* with a documented status rather than
returning a 500, hallucinating sentences into silence, or pinning a worker indefinitely.

Every request here runs under ``REAL_ASR_ADVERSARIAL_TIMEOUT`` (120s by default) and a
timeout is a test failure, because a hang is the most expensive of these failure modes and
the easiest to mistake for success.
"""

from __future__ import annotations

import pytest

from tests.real_audio import matrix_support, service_client

pytestmark = matrix_support.REAL_ASR_MARKS


MALFORMED_IDS = {
    "silence_only",
    "noise_only",
    "tones_only",
    "sub_second_clip",
    "truncated_wav_header",
    "zero_byte_upload",
    "mp3_named_wav",
}


def _malformed_params():
    """Return the manifest entries covering non-speech and malformed uploads."""
    params = [param for param in matrix_support.clip_params("adversarial") if param.id in MALFORMED_IDS]
    assert {param.id for param in params} == MALFORMED_IDS
    return params


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("case", _malformed_params())
def test_malformed_input_fails_gracefully_and_promptly(case, clip_path):
    """The service must answer within the adversarial budget with a declared status."""
    matrix_support.assert_declared_response(case, service_client.post_promptly(clip_path(case["id"])))


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("case", [param for param in _malformed_params() if param.id in {"silence_only", "sub_second_clip"}])
def test_silence_does_not_hallucinate_text(case, clip_path):
    """Silence must not become confident sentences -- the classic long-tail ASR failure."""
    response = service_client.post_promptly(clip_path(case["id"]))
    if response.status_code != 200:
        pytest.skip(f"{case['id']}: rejected with {response.status_code}, which is also acceptable")
    spoken = " ".join(matrix_support.spoken_words(response.json())).strip()
    assert len(spoken) < 40, f"{case['id']}: silence produced {len(spoken)} characters of text: {spoken!r}"
