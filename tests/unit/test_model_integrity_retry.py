"""Coverage for the download/validate/purge retry loop and the purge it depends on.

A corrupt model that is not purged is worse than a failed download: the next attempt sees
a file already present, and without the purge the retry loop would keep validating the same
bad bytes instead of fetching new ones.
"""

from unittest import mock

from modules.core import model_integrity


class TestPurgingACorruptedPath:
    """Removing a bad asset is what makes the retry a retry rather than a re-check."""

    def test_an_absent_path_is_already_purged(self, tmp_path):
        """Nothing to remove is success, not a failure to remove something."""
        assert model_integrity.purge_corrupted_path(tmp_path / "missing.bin") is True

    def test_a_file_is_removed(self, tmp_path):
        """A corrupt single-file asset is deleted so the retry refetches it."""
        target = tmp_path / "model.bin"
        target.write_bytes(b"corrupt")
        assert model_integrity.purge_corrupted_path(target) is True
        assert not target.exists()

    def test_a_directory_is_removed_recursively(self, tmp_path):
        """OpenVINO models are directories; a partial tree must go whole."""
        target = tmp_path / "ov_model"
        (target / "nested").mkdir(parents=True)
        (target / "nested" / "weights.bin").write_bytes(b"x")
        assert model_integrity.purge_corrupted_path(target) is True
        assert not target.exists()

    def test_a_removal_failure_is_reported_rather_than_raised(self, tmp_path):
        """The caller decides what to do about it; an exception here would abort provisioning."""
        target = tmp_path / "model.bin"
        target.write_bytes(b"corrupt")
        with mock.patch.object(model_integrity.Path, "unlink", side_effect=PermissionError("read-only")):
            assert model_integrity.purge_corrupted_path(target, "UVR model") is False


class TestTheDownloadRetryLoop:
    """Download, validate, purge, repeat -- bounded by a total attempt count."""

    def _valid_after(self, path, successes_from):
        calls = {"n": 0}

        def validator(_p):
            calls["n"] += 1
            return calls["n"] >= successes_from

        return validator, calls

    def test_an_existing_valid_asset_skips_the_download(self, tmp_path):
        """Provisioning is a boot-time cost, so a good asset is never refetched."""
        target = tmp_path / "model.bin"
        target.write_bytes(b"good")
        download = mock.Mock()
        assert model_integrity.download_with_integrity_retry(download_fn=download, validator_fn=lambda _p: True, target_path=target) is True
        download.assert_not_called()

    def test_an_existing_invalid_asset_is_purged_and_refetched(self, tmp_path):
        """Without the purge the retry would revalidate the same bad bytes."""
        target = tmp_path / "model.bin"
        target.write_bytes(b"corrupt")
        validator, _calls = self._valid_after(target, successes_from=2)
        with mock.patch.object(model_integrity, "purge_corrupted_path", wraps=model_integrity.purge_corrupted_path) as purge:
            result = model_integrity.download_with_integrity_retry(
                download_fn=lambda: target.write_bytes(b"good"), validator_fn=validator, target_path=target
            )
        assert result is True
        purge.assert_called()

    def test_a_second_attempt_recovers_a_first_failed_validation(self, tmp_path):
        """The point of the loop: a truncated first download is recoverable."""
        target = tmp_path / "model.bin"
        validator, calls = self._valid_after(target, successes_from=2)
        result = model_integrity.download_with_integrity_retry(
            download_fn=lambda: target.write_bytes(b"bytes"), validator_fn=validator, target_path=target, max_retries=2
        )
        assert result is True
        assert calls["n"] == 2

    def test_exhausting_every_attempt_reports_failure(self, tmp_path):
        """max_retries is the total attempt count, not a count of extra retries."""
        target = tmp_path / "model.bin"
        download = mock.Mock(side_effect=lambda: target.write_bytes(b"bad"))
        result = model_integrity.download_with_integrity_retry(
            download_fn=download, validator_fn=lambda _p: False, target_path=target, max_retries=2
        )
        assert result is False
        assert download.call_count == 2, "max_retries is the total number of attempts, not extra ones"
