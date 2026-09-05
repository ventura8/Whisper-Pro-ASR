"""Audio that is valid but hostile.

Nothing here is malformed -- every file decodes cleanly. They are the recordings the
service will actually meet in the wild and handle badly if a preprocessing assumption is
wrong: overdriven audio from a hot microphone, near-silent audio from a distant one, an
8 kHz telephone call, a stereo file where the pipeline expects mono, sample rates that need
resampling, and a speaker who says nothing for the first thirty seconds.

The cases and what counts as acceptable for each live in the manifest's ``adversarial``
section, so tightening or relaxing an expectation is a data change.
"""

from __future__ import annotations

import pytest

from tests.real_audio import matrix_support, service_client

pytestmark = matrix_support.REAL_ASR_MARKS


DEGRADED_IDS = {
    "clipped_speech",
    "very_quiet_speech",
    "telephone_band",
    "stereo_speech",
    "rate_44100",
    "rate_48000",
    "rate_8000",
    "speech_after_30s_silence",
}


def _degraded_params():
    """Return the manifest entries covering valid-but-difficult audio.

    A DEGRADED_ID that no longer names a manifest entry is a collection error, not a
    quietly smaller run: renaming or dropping an adversarial entry used to shrink this set
    silently, so the case simply stopped being tested and the suite still reported all
    green.
    """
    params = [param for param in matrix_support.clip_params("adversarial") if param.id in DEGRADED_IDS]
    missing = sorted(DEGRADED_IDS - {param.id for param in params})
    if missing:
        raise AssertionError(f"DEGRADED_IDS names entries absent from the adversarial manifest section: {missing}")
    return params


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("case", _degraded_params())
def test_degraded_audio_is_handled_as_declared(case, clip_path):
    """The response status and transcript must match the contract the manifest declares."""
    matrix_support.assert_declared_response(case, service_client.post_promptly(clip_path(case["id"])))


@pytest.mark.usefixtures("service_ready")
@pytest.mark.parametrize("case", [param for param in _degraded_params() if param.id == "speech_after_30s_silence"])
def test_speech_after_long_silence_is_not_missed(case, clip_path):
    """A long silent lead-in must not make the engine stop before the speech starts."""
    payload = service_client.transcribe(clip_path(case["id"]))
    segments = [seg for seg in (payload.get("segments") or []) if str(seg.get("text", "")).strip()]
    assert segments, "no speech found after a 30s silent lead-in"
    assert max(float(seg["end"]) for seg in segments) > 25.0, "segments never reach the speech, which starts at 30s"
