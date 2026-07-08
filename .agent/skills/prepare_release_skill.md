# Prepare Release Skill

This skill outlines the structured workflow to prepare a new release of Whisper Pro ASR from a feature branch.

## Objective
Extract the target version from the current feature branch name (e.g., `feature/v1.1.4` -> version `v1.1.4`), verify code quality and performance gates, review the whole current commit for documentation and release-note drift, update project documentation, generate release notes, and consolidate all modifications into a unified amended commit.

---

## Procedure

### 1. Identify Release Version
Extract the version identifier from the active git branch name:
```bash
# Display active branch name
git branch --show-current
```
*Example*: If on branch `feature/v1.1.4`, the release tag and documentation target version is `v1.1.4`.

### 2. Verify Pipeline Quality Gates
Ensure all tests are passing and the linter score is perfect before locking the release:
- **Pytest**: Run unit and integration suites to ensure a 100% pass rate.
  ```bash
  python3 -m pytest tests/
  ```
- **Code Coverage**: Verify total test coverage is above the build-gate threshold (**>= 90%**).
- **Pylint**: Run static analysis across all python files and verify it receives a perfect **10.0/10.0** rating with no suppressions.
  ```bash
  pylint modules/ tests/ whisper_pro_asr.py check_coverage.py
  ```

### 3. Update Project Documentation
Ensure all documentation files are synchronized with codebase features:
- Review the full current commit diff first so docs and release notes are checked against every changed file, not only the most obvious feature file.
- **README.md**: Document new parameters, environmental flags, API routes, and update directory tree structures.
- **docs/ARCHITECTURE.md**: Document deep-dives on concurrency mechanisms, lifecycle threads, caching, and model offloading patterns.
- **docs/API.md**: Update route schemas, query parameters, returned formats, and example JSON payloads.
- **docs/DOCKERHUB_DESCRIPTION.md**: Update features and sync with README changes.

### 4. Generate GitHub Release Notes
Create a new version-specific markdown release file at `docs/releases/GITHUB_RELEASE_v<VERSION>.md`:
*   Highlight key features and structural improvements.
*   Document optimizations, bug fixes, and security enhancements.
*   Include verification results (test passes, pylint rating, and coverage totals).

### 5. Consolidate & Amend Git Commit
A new release must reside in a single, well-documented commit at the head of the feature branch:
- Stage all modified files and the new release markdown file:
  ```bash
  git add -A
  ```
- Amend the commit to include all changes under a descriptive title and detailed summary of changes:
  ```bash
  git commit --amend -m "v1.x.y: Short Summary of Main Features

  - Detailed bullet point 1
  - Detailed bullet point 2"
  ```
- Confirm the git tree is completely clean (fail-fast check):
  ```bash
  # Check if there are any untracked, unstaged, or staged changes remaining
  if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory is not clean. Stage and commit all changes first."
    exit 1
  fi
  ```
