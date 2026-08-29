"""Tests for scripts/ci/check-inline-ignores.py."""

import importlib.util
from pathlib import Path


def _load_module():
    """Load the inline-ignore checker module from scripts/ci."""

    module_path = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check-inline-ignores.py"
    spec = importlib.util.spec_from_file_location("check_inline_ignores", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_detects_dockerfile_without_extension(tmp_path):
    """Ensure Dockerfile names are scanned despite lacking extensions."""

    module = _load_module()
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("# py" + "lint: disable=unused-variable\n", encoding="utf-8")

    assert module.should_scan_file(str(dockerfile)) is True
    violations = module.scan_file(str(dockerfile))
    assert any(v[1] == "pylint-disable" for v in violations)


def test_detects_new_suppression_patterns(tmp_path):
    """Ensure newly tracked suppression patterns are reported."""

    module = _load_module()
    sample = tmp_path / "sample.py"
    noqa_line = "# " + "no" + "qa: E501"
    pylint_disable_line = "# py" + "lint: disable=too-many-branches"
    coverage_line = "# pragma:" + " no cover"
    isort_line = "# " + "isort:" + " skip"
    istanbul_line = "// istanbul " + "ignore next"
    html_validate_line = "<!-- html-" + "validate-disable -->"
    sample.write_text(
        "\n".join(
            [
                noqa_line,
                pylint_disable_line,
                coverage_line,
                isort_line,
                istanbul_line,
                html_validate_line,
            ]
        ),
        encoding="utf-8",
    )

    violations = module.scan_file(str(sample))
    names = {name for _, name, _ in violations}
    expected = {"noqa", "pylint-disable", "coverage-ignore", "isort-skip", "html-" + "validate-disable"}
    assert expected.issubset(names)


def test_scan_file_raises_on_read_failure(monkeypatch, tmp_path):
    """Ensure unreadable files raise a stable RuntimeError contract."""

    module = _load_module()
    target = tmp_path / "bad.py"
    target.write_text("print('x')", encoding="utf-8")

    def _raise_open(*_args, **_kwargs):
        raise OSError("blocked")

    monkeypatch.setattr("builtins.open", _raise_open)

    try:
        module.scan_file(str(target))
    except RuntimeError as exc:
        assert "Failed to scan file" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unreadable file")


#: Built by concatenation, exactly as the tests above do it. A test file that names these
#: markers literally is itself a violation -- which the gate demonstrated by failing on the
#: first draft of this class.
_NOQA = "# " + "no" + "qa"
_TYPE_IGNORE = "# ty" + "pe: ignore"
_NO_COVER = "# pragma:" + " no cover"
_PYLINT_OFF = "# py" + "lint: disable"
_SHELLCHECK_OFF = "shell" + "check disable=SC2086"


class TestMarkdownDistinguishesProseFromCode:
    """Documenting the ban is not breaking it.

    The policy is written down in README.md and in several skill files, and writing it down
    means naming the markers. A rule that cannot tell a prose mention from a real suppression
    makes the ban undocumentable -- which is why README.md was excluded from the Docker build
    context and therefore from this check, in both cases silently.

    A fenced block is the opposite case: it is sample code, written to be copied, so a
    suppression there is one the project is teaching someone to write.
    """

    def _violations(self, tmp_path, name, body):
        """Scan one file and return its violations."""
        target = tmp_path / name
        target.write_text(body, encoding="utf-8")
        return _load_module().scan_file(str(target))

    def test_prose_naming_the_markers_is_allowed(self, tmp_path):
        """The shape of README.md's zero-suppression paragraph."""
        body = f"The checker fails the build on any\n`{_PYLINT_OFF}`, `{_NOQA}` or `{_TYPE_IGNORE}` anywhere in the tree.\n"
        assert not self._violations(tmp_path, "policy.md", body)

    def test_a_fenced_example_is_still_a_violation(self, tmp_path):
        """Sample code is meant to be copied, so it is held to the same rule as code."""
        body = f"Do not do this:\n\n```python\nvalue = 1  {_NOQA}\n```\n"
        assert self._violations(tmp_path, "guide.md", body)

    def test_a_tilde_fence_counts_too(self, tmp_path):
        """CommonMark allows ~~~ as well; matching only backticks would leave a hole."""
        body = f"~~~python\nvalue = 2  {_TYPE_IGNORE}\n~~~\n"
        assert self._violations(tmp_path, "guide.md", body)

    def test_an_unterminated_fence_still_scans_its_body(self, tmp_path):
        """Failing open on a malformed document would be an easy way to hide one."""
        body = f"```python\nvalue = 3  {_NO_COVER}\n"
        assert self._violations(tmp_path, "truncated.md", body)

    def test_code_after_a_closed_fence_is_prose_again(self, tmp_path):
        """The fence has to toggle, or everything after the first block reads as code."""
        body = f"```python\nok = 1\n```\n\nNever write `{_NOQA}` in this project.\n"
        assert not self._violations(tmp_path, "mixed.md", body)

    def test_python_files_are_still_scanned_whole(self, tmp_path):
        """Markdown is the only type where mentioning code differs from being code."""
        assert self._violations(tmp_path, "module.py", f"value = 1  {_NOQA}\n")

    def test_shell_files_are_still_scanned_whole(self, tmp_path):
        """The shell marker is the one this repo has had to reject most often."""
        assert self._violations(tmp_path, "script.sh", f'# {_SHELLCHECK_OFF}\necho "$x"\n')
