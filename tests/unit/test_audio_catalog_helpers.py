"""Tests for the cell-level helpers in scripts/audio_catalog.py.

The catalog is written for agents choosing a fixture, and `--check` compares it byte for
byte against the committed copy -- so a helper that formats a cell differently on one
machine makes the gate fail with "stale" while naming nothing. These tests pin the
formatting and the degradation paths; the document assembly is in
test_audio_catalog_document.py.
"""

import subprocess
from unittest import mock

import pytest

from scripts import audio_catalog


def _probe_payload(duration="12.5", rate="16000", channels=1):
    """An ffprobe JSON document shaped like the real one."""
    return f'{{"streams": [{{"sample_rate": "{rate}", "channels": {channels}}}], "format": {{"duration": "{duration}"}}}}'


def test_probe_returns_none_without_ffprobe(tmp_path):
    """No ffprobe means no measurements rather than a crash."""
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"RIFF")

    with mock.patch.object(audio_catalog.shutil, "which", return_value=None):
        assert audio_catalog._probe(clip) is None


def test_probe_returns_none_for_a_missing_file(tmp_path):
    """On-demand clips are absent on a fresh checkout."""
    with mock.patch.object(audio_catalog.shutil, "which", return_value="/usr/bin/ffprobe"):
        assert audio_catalog._probe(tmp_path / "absent.wav") is None


def test_probe_reports_duration_rate_channels_and_size(tmp_path):
    """The four values the Measured column shows."""
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"x" * 2048)
    completed = subprocess.CompletedProcess([], 0, stdout=_probe_payload(), stderr="")

    with mock.patch.object(audio_catalog.shutil, "which", return_value="/usr/bin/ffprobe"):
        with mock.patch.object(audio_catalog.subprocess, "run", return_value=completed):
            probed = audio_catalog._probe(clip)

    assert probed == {"duration": "12.5s", "rate": "16000", "channels": 1, "size": "2 KB"}


def test_ffprobe_json_is_bounded_by_the_shared_timeout(tmp_path):
    """An unbounded ffprobe on a truncated fixture hangs the whole catalog run."""
    completed = subprocess.CompletedProcess([], 0, stdout=_probe_payload(), stderr="")

    with mock.patch.object(audio_catalog.subprocess, "run", return_value=completed) as run:
        audio_catalog._ffprobe_json(tmp_path / "a.wav")

    assert run.call_args.kwargs["timeout"] == audio_catalog.FFMPEG_TIMEOUT_SEC


def test_ffprobe_json_returns_none_on_failure(tmp_path):
    """An unreadable file becomes a dash in the table, not a traceback."""
    with mock.patch.object(audio_catalog.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ffprobe")):
        assert audio_catalog._ffprobe_json(tmp_path / "a.wav") is None


def test_ffprobe_json_returns_none_on_timeout(tmp_path):
    """The timeout has to be handled, or bounding it only moves the failure."""
    with mock.patch.object(audio_catalog.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffprobe", 600)):
        assert audio_catalog._ffprobe_json(tmp_path / "a.wav") is None


def test_ffprobe_json_treats_empty_output_as_failure(tmp_path):
    """json.loads would raise on it, turning an unreadable file into a traceback."""
    completed = subprocess.CompletedProcess([], 0, stdout="   \n", stderr="")

    with mock.patch.object(audio_catalog.subprocess, "run", return_value=completed):
        assert audio_catalog._ffprobe_json(tmp_path / "a.wav") is None


@pytest.mark.parametrize(("value", "expected"), [("12.5", "12.5s"), ("0.05", "0.1s"), (None, "-"), ("", "-")])
def test_format_seconds(value, expected):
    """A duration ffprobe did not report renders as a dash."""
    assert audio_catalog._format_seconds(value) == expected


def test_locate_distinguishes_committed_from_generated():
    """Committed clips are FLAC in the core directory; generated ones are WAV on demand."""
    committed_path, committed_state = audio_catalog._locate("en_core", True)
    generated_path, generated_state = audio_catalog._locate("am_tail", False)

    assert committed_path.suffix == ".flac" and committed_state == "committed"
    assert generated_path.suffix == ".wav" and generated_state == "on demand"


def test_facts_does_not_probe_a_generated_clip(tmp_path):
    """Generated clips are absent on a fresh checkout, so measuring them would make the
    catalog differ per machine and --check meaningless."""
    with mock.patch.object(audio_catalog, "_probe") as probe:
        assert audio_catalog._facts(tmp_path / "a.wav", committed=False) == "-"

    probe.assert_not_called()


def test_facts_renders_a_measured_cell(tmp_path):
    """The committed path, where the numbers are stable across machines."""
    measured = {"duration": "12.5s", "rate": "16000", "channels": 1, "size": "2 KB"}

    with mock.patch.object(audio_catalog, "_probe", return_value=measured):
        assert audio_catalog._facts(tmp_path / "a.flac") == "12.5s, 16000 Hz, 1 ch, 2 KB"


def test_facts_is_a_dash_when_probing_fails(tmp_path):
    """One unreadable file must not break the row."""
    with mock.patch.object(audio_catalog, "_probe", return_value=None):
        assert audio_catalog._facts(tmp_path / "a.flac") == "-"


def test_describe_adversarial_fills_the_template():
    """The description exists so an agent need not read adversarial.py first."""
    entry = {"builder": "clipped", "params": {"source": "en_core", "gain": 12.0}}

    assert audio_catalog._describe_adversarial(entry) == "`en_core` amplified 12.0x into hard clipping -- distorted speech."


def test_describe_adversarial_falls_back_when_a_param_is_absent():
    """A template naming a param the entry omits renders unfilled rather than raising."""
    entry = {"builder": "clipped", "params": {}}

    assert "{gain}" in audio_catalog._describe_adversarial(entry)


def test_describe_adversarial_falls_back_to_the_builder_name():
    """A builder with no description still names itself."""
    assert audio_catalog._describe_adversarial({"builder": "brand_new", "params": {}}) == "brand_new"


def test_flag_notes_is_a_dash_when_there_is_nothing_to_say():
    """An unremarkable entry gets a dash, not an empty cell."""
    assert audio_catalog._flag_notes({}) == "-"


def test_flag_notes_marks_the_smoke_subset():
    """`smoke` is the representative subset the pre-merge check runs."""
    assert audio_catalog._flag_notes({"smoke": True}) == "smoke"


def test_flag_notes_records_a_measured_bar_with_its_provenance():
    """The near misses hold a strict 0.3 instead of an xfail, and the catalog says why."""
    note = audio_catalog._flag_notes({"bar_note": "measured 0.31 on the 3080"})

    assert note == "bar: measured 0.31 on the 3080"


def test_xfail_note_names_the_engines_a_defect_is_limited_to():
    """A bare "xfail:" reads as "this clip is broken", which is wrong for the code-switched
    entries: they are defects on WHISPERX and held strictly on FASTER-WHISPER."""
    entry = {"xfail_reason": "drops the second leg", "xfail_engines": ["WHISPERX"]}

    assert audio_catalog._flag_notes(entry) == "xfail (WHISPERX): drops the second leg"


def test_xfail_note_without_engines_applies_everywhere():
    """An unscoped defect is genuinely unscoped."""
    assert audio_catalog._flag_notes({"xfail_reason": "hangs"}) == "xfail: hangs"


def test_flag_notes_joins_several_notes():
    """An entry can be in the smoke set and carry a defect."""
    entry = {"smoke": True, "xfail_reason": "hangs"}

    assert audio_catalog._flag_notes(entry) == "smoke; xfail: hangs"


def test_cell_escapes_a_pipe():
    """An unescaped pipe would break out of the Markdown row."""
    assert audio_catalog._cell("a|b") == "a\\|b"


def test_cell_flattens_newlines():
    """A newline would end the table row early."""
    assert audio_catalog._cell("a\nb") == "a b"


def test_cell_stringifies_non_strings():
    """Channel counts arrive as integers."""
    assert audio_catalog._cell(2) == "2"


def test_table_pads_the_separator_row_like_the_header():
    """markdownlint's MD060 enforces one table style per document.

    The unpadded "|---|---|" form contradicts the padded header this same function emits,
    so the generated catalog failed the repo's own markdown gate.
    """
    lines = audio_catalog._table(["A", "B"], [["1", "2"]])

    assert lines[0] == "| A | B |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 2 |"
    assert lines[-1] == ""


def test_split_by_tier_separates_the_committed_core():
    """Tier A is committed; everything else renders on demand."""
    clips = [{"tier": "A", "id": "a"}, {"tier": "B", "id": "b"}, {"id": "c"}]

    core, tail = audio_catalog._split_by_tier(clips)

    assert [c["id"] for c in core] == ["a"]
    assert [c["id"] for c in tail] == ["b", "c"]


@pytest.mark.parametrize(
    ("languages", "expected"),
    [
        ([], "no languages"),
        (["en"], "one language (en)"),
        (["en", "fr"], "2 languages (en, fr)"),
    ],
)
def test_languages_label(languages, expected):
    """An explicit empty list is a spec the generator refuses to build, and the catalog has
    to say so rather than fail on `languages[0]`."""
    assert expected in audio_catalog._languages_label(languages)


def test_known_defect_lines_says_so_when_there_are_none():
    """The manifest records a fix by deleting the entry, so empty is the normal state.

    Emitting the heading with nothing under it left two consecutive blank lines that
    markdownlint rejected the moment the last defect was cleared.
    """
    lines = audio_catalog._known_defect_lines({})

    assert any("No known defects recorded" in line for line in lines)
    assert not any("Known defects it currently reproduces" in line for line in lines)


def test_known_defect_lines_lists_each_defect():
    """The ordinary case, one bullet per defect."""
    lines = audio_catalog._known_defect_lines({"windows": "67 of 118 wrong", "quiet": "11 invented"})

    assert "Known defects it currently reproduces:" in lines
    assert "- **windows** -- 67 of 118 wrong" in lines
    assert "- **quiet** -- 11 invented" in lines
