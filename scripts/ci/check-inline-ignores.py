#!/usr/bin/env python3
import logging
import os
import re
import sys
from collections.abc import Iterable, Iterator

type Violation = tuple[int, str, str]

# Directory exclusions
EXCLUDE_DIRS = {
    "node_modules",
    ".venv",
    "venv",
    ".git",
    ".ruff_cache",
    "state",
    # Gitignored local caches of third-party source. They are absent on a CI runner, so
    # scanning them found upstream's `# pragma: no cover` markers on a developer machine
    # only -- a zero-suppression policy is about THIS repository's code.
    "model_cache",
    ".fixture-tooling",
    "test_data",
    "reports",
    "coverage-js",
    "test-results",
    "__pycache__",
}

# File extensions to scan
EXTENSIONS = {".py", ".js", ".cjs", ".mjs", ".css", ".html", ".md", ".sh", ".ps1", ".yml", ".yaml", ".toml"}
SPECIAL_FILENAMES = {"Dockerfile", "Dockerfile.test"}

# Regex patterns for suppression comments
PATTERNS = {
    "eslint-disable": re.compile(r"eslint-disable"),
    "stylelint-disable": re.compile(r"stylelint-disable"),
    "markdownlint-disable": re.compile(r"markdownlint-disable"),
    "shellcheck-disable": re.compile(r"shellcheck\s+disable"),
    "type-ignore": re.compile(r"#\s*type:\s*ignore"),
    "nosec": re.compile(r"#\s*nosec\b"),
    "noqa": re.compile(r"#\s*noqa\b"),
    "pylint-disable": re.compile(r"pylint:\s*disable"),
    "coverage-ignore": re.compile(r"pragma:\s*no\s*cover|istanbul\s+ignore|c8\s+ignore"),
    "isort-skip": re.compile(r"#\s*isort:\s*(skip|off)"),
    "formatter-ignore": re.compile(r"prettier-ignore|ruff:\s*noqa"),
    "html-validate-disable": re.compile(r"html-validate-disable"),
    # PSScriptAnalyzer's attribute form. Absent from this list, four of them lived in
    # scripts/*.ps1 for months under a policy that bans exactly this -- invisible rather
    # than allowed. The exemption now lives in PSScriptAnalyzerSettings.psd1.
    # The "Attribute" suffix is optional in PowerShell attribute syntax, so
    # [Diagnostics.CodeAnalysis.SuppressMessage(...)] is the same suppression written in
    # the shorter spelling and slipped past a pattern anchored on the long one.
    # IGNORECASE: PowerShell resolves attribute names case-insensitively, so
    # [diagnostics.codeanalysis.suppressmessage(...)] is the same suppression in a spelling
    # the case-sensitive pattern did not see.
    "psscriptanalyzer-suppress": re.compile(r"SuppressMessage(Attribute)?\s*\(", re.IGNORECASE),
}

logger = logging.getLogger(__name__)


def should_scan_file(filepath: str) -> bool:
    if os.path.basename(filepath) == "check-inline-ignores.py":
        return False
    if os.path.basename(filepath) in SPECIAL_FILENAMES:
        return True
    return os.path.splitext(filepath)[1] in EXTENSIONS


def _should_scan_file(filepath: str) -> bool:
    """Backward-compatible alias for internal callers."""
    return should_scan_file(filepath)


def _scan_files_in_dir(root: str, files: list[str]) -> list[str]:
    return [os.path.join(root, filename) for filename in files if should_scan_file(os.path.join(root, filename))]


def _iter_scan_targets(root_dir: str) -> Iterator[str]:
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        yield from _scan_files_in_dir(root, files)


def _report_violations(root_dir: str, filepath: str, violations: list[Violation]) -> None:
    rel_path = os.path.relpath(filepath, root_dir)
    logger.error("Violations in %s:", rel_path)
    for line_num, name, line in violations:
        logger.error("  Line %d: [%s] %s", line_num, name, line)


def _is_fence(line: str) -> bool:
    """Whether a Markdown line opens or closes a fenced code block."""
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _fenced_code_lines(handle: Iterable[str]) -> Iterator[tuple[int, str]]:
    """Only the lines inside fenced code blocks, with their real line numbers."""
    in_fence = False
    for line_num, line in enumerate(handle, 1):
        if _is_fence(line):
            in_fence = not in_fence
            continue
        if in_fence:
            yield line_num, line


def _scannable_lines(filepath: str, handle: Iterable[str]) -> Iterator[tuple[int, str]]:
    """The lines of a file where a suppression marker would be a real suppression.

    In Markdown that is fenced code only. Prose has to be able to *name* these markers --
    the policy is documented in README.md and in several skill files, and a rule that
    cannot tell "never write `# noqa`" from an actual `# noqa` makes the ban undocumentable.
    A fence is different: it is sample code, written to be copied, so a suppression there is
    one this project is teaching someone to write.

    Every other file type is scanned whole. Markdown is the only one where the difference
    between mentioning code and being code is expressible.
    """
    if os.path.splitext(filepath)[1].lower() != ".md":
        yield from enumerate(handle, 1)
        return
    yield from _fenced_code_lines(handle)


def scan_file(filepath: str) -> list[Violation]:
    violations: list[Violation] = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in _scannable_lines(filepath, f):
                for name, pattern in PATTERNS.items():
                    if pattern.search(line):
                        violations.append((line_num, name, line.strip()))
    except Exception as exc:
        logger.error("Error reading %s: %s", filepath, exc)
        raise RuntimeError(f"Failed to scan file: {filepath}") from exc
    return violations


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    has_violations = False

    for filepath in _iter_scan_targets(root_dir):
        violations = scan_file(filepath)
        if violations:
            has_violations = True
            _report_violations(root_dir, filepath, violations)

    if has_violations:
        logger.error("Error: Inline suppressions / ignores are disallowed under zero-suppression policy.")
        sys.exit(1)
    logger.info("Success: No inline suppressions found.")
    sys.exit(0)


if __name__ == "__main__":
    main()
