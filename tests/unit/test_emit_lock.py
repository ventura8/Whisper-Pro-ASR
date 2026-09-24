"""Tests for scripts/audio_matrix/_emit_lock.py.

The generated file pins the fixture toolchain by hash, so the properties that matter are
that every resolved package reaches the output with a hash, that a hash the index did not
publish is downloaded and checked against PyPI rather than trusted silently, and that a
package PyPI cannot corroborate is reported loudly instead of blending in.
"""

import hashlib
import io
import json
from unittest import mock

from scripts.audio_matrix import _emit_lock


def _package(name, version, published_hash=None, url="https://example.invalid/pkg.whl"):
    """One entry of pip's --report install list."""
    archive_info = {"hashes": {"sha256": published_hash}} if published_hash else {}
    return {
        "metadata": {"name": name, "version": version},
        "download_info": {"url": url, "archive_info": archive_info},
    }


def _write_report(tmp_path, packages):
    """Write a pip --report document and return its path."""
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"install": packages}), encoding="utf-8")
    return str(report)


def test_uses_the_hash_the_index_published(tmp_path, capsys):
    """When the index publishes a sha256, it is used verbatim and nothing is downloaded."""
    report = _write_report(tmp_path, [_package("piper-tts", "1.8.0", published_hash="a" * 64)])
    out = tmp_path / "out.txt"

    with mock.patch.object(_emit_lock.urllib.request, "urlopen") as urlopen:
        assert _emit_lock.main(report, str(out)) == 0

    urlopen.assert_not_called()
    body = out.read_text(encoding="utf-8")
    assert "piper-tts==1.8.0 \\" in body
    assert f"--hash=sha256:{'a' * 64}" in body
    assert "1 packages pinned" in capsys.readouterr().out


def test_downloads_and_corroborates_a_hash_the_index_withheld(tmp_path):
    """A package with no published hash is downloaded, hashed, and checked against PyPI."""
    payload = b"wheel-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    report = _write_report(tmp_path, [_package("uroman", "1.3.1.1")])
    out = tmp_path / "out.txt"

    def _urlopen(url, timeout=None):  # signature mirrors urllib.request.urlopen
        if "pypi.org" in str(url):
            return _ctx(io.BytesIO(json.dumps({"urls": [{"digests": {"sha256": digest}}]}).encode()))
        return _ctx(io.BytesIO(payload))

    with mock.patch.object(_emit_lock.urllib.request, "urlopen", side_effect=_urlopen):
        assert _emit_lock.main(report, str(out)) == 0

    body = out.read_text(encoding="utf-8")
    assert f"--hash=sha256:{digest}" in body
    assert "verified identical to PyPI" in body
    assert "NOT corroborated" not in body


def test_flags_a_hash_pypi_cannot_corroborate(tmp_path, capsys):
    """A mirror-only artefact PyPI does not match is written, but reported on stderr.

    The point of the LOCAL note is that the digest is corroborated rather than merely
    trusted, so the uncorroborated case has to be visible to whoever regenerates the lock.
    """
    report = _write_report(tmp_path, [_package("torch", "2.13.0+cpu")])
    out = tmp_path / "out.txt"

    def _urlopen(url, timeout=None):  # signature mirrors urllib.request.urlopen
        if "pypi.org" in str(url):
            return _ctx(io.BytesIO(json.dumps({"urls": [{"digests": {"sha256": "b" * 64}}]}).encode()))
        return _ctx(io.BytesIO(b"different-bytes"))

    with mock.patch.object(_emit_lock.urllib.request, "urlopen", side_effect=_urlopen):
        assert _emit_lock.main(report, str(out)) == 0

    assert "NOT corroborated by PyPI" in out.read_text(encoding="utf-8")
    assert "torch" in capsys.readouterr().err


def test_treats_an_unreachable_pypi_as_no_corroboration(tmp_path):
    """If PyPI cannot be asked, the digest is kept but explicitly not claimed as verified."""
    report = _write_report(tmp_path, [_package("uroman", "1.3.1.1")])
    out = tmp_path / "out.txt"

    def _urlopen(url, timeout=None):  # signature mirrors urllib.request.urlopen
        if "pypi.org" in str(url):
            raise OSError("name resolution failed")
        return _ctx(io.BytesIO(b"wheel-bytes"))

    with mock.patch.object(_emit_lock.urllib.request, "urlopen", side_effect=_urlopen):
        assert _emit_lock.main(report, str(out)) == 0

    assert "NOT corroborated by PyPI" in out.read_text(encoding="utf-8")


def test_pypi_digests_ignores_a_malformed_response():
    """A 200 that is not JSON yields no digests rather than propagating a parse error."""
    with mock.patch.object(_emit_lock.urllib.request, "urlopen", return_value=_ctx(io.BytesIO(b"<html>"))):
        assert _emit_lock._pypi_digests("anything", "1.0") == set()


def test_packages_are_written_in_case_insensitive_name_order(tmp_path):
    """Ordering is stable and case-insensitive, so regenerating produces no spurious diff."""
    packages = [
        _package("Zstandard", "0.1", published_hash="c" * 64),
        _package("absl-py", "0.2", published_hash="d" * 64),
    ]
    out = tmp_path / "out.txt"

    assert _emit_lock.main(_write_report(tmp_path, packages), str(out)) == 0

    body = out.read_text(encoding="utf-8")
    assert body.index("absl-py==") < body.index("Zstandard==")


def test_the_local_note_is_its_own_line(tmp_path):
    """The note must not trail the requirement: a comment after `\\` is a pip syntax error.

    pip reports that failure without naming a package, so it is worth pinning here.
    """
    report = _write_report(tmp_path, [_package("uroman", "1.3.1.1")])
    out = tmp_path / "out.txt"

    def _urlopen(url, timeout=None):  # signature mirrors urllib.request.urlopen
        if "pypi.org" in str(url):
            raise OSError("offline")
        return _ctx(io.BytesIO(b"bytes"))

    with mock.patch.object(_emit_lock.urllib.request, "urlopen", side_effect=_urlopen):
        _emit_lock.main(report, str(out))

    lines = out.read_text(encoding="utf-8").splitlines()
    note_index = next(i for i, line in enumerate(lines) if "LOCAL:" in line)
    assert lines[note_index].lstrip().startswith("#")
    assert lines[note_index + 1].startswith("uroman==")


def _ctx(stream):
    """Wrap a stream so it behaves like urlopen's context manager."""
    manager = mock.MagicMock()
    manager.__enter__.return_value = stream
    manager.__exit__.return_value = False
    return manager
