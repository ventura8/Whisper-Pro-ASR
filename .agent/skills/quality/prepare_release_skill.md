# Prepare Release Skill

This skill defines the release preparation workflow for Whisper Pro ASR from an active working branch.

## Objective

Determine the target release version, verify quality gates, review documentation/release-note drift across the full change set, update docs, generate release notes, and produce a clean release-ready commit.

---

## Procedure

### 1. Identify Release Version

Determine the release version in this order:

1. If branch name matches `feature/vX.Y.Z`, use `X.Y.Z`.
2. Otherwise, if branch name matches `release/vX.Y.Z`, `feature/X.Y.Z`, or `release/X.Y.Z`, use `X.Y.Z`.
3. Otherwise, use an explicitly provided release version.
4. If neither is available, stop and request the target version before publishing release notes.

```bash
# Display active branch name
git branch --show-current
```

*Example*: On branch `feature/v1.1.6`, release tag/docs target version is `v1.1.6`.

### 2. Verify Pipeline Quality Gates

Ensure tests and lint gates are passing before release finalization:

- Run Docker-backed parity wrapper (preferred):

  ```bash
  ./scripts/ci/build-and-test.sh
  ```

  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\ci\build-and-test.ps1
  ```

- Or run the explicit Docker test image/entrypoint (still Docker-only):

  ```bash
  docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
  docker run --rm -e CI=true -v "$PWD/assets:/app/assets" -v "$PWD/reports:/reports" whisper-pro-asr-test /bin/bash -lc "tests/run_suite.sh"
  ```

- Required gates remain unchanged: backend tests, coverage >= 90%, flake8/pylint quality, frontend quality + npm audit, markdown lint (when markdown changed).

### 3. Update Project Documentation

Ensure all documentation files are synchronized with the actual shipped behavior:

- Review the full current commit diff first so docs and release notes are checked against every changed file, not only the most obvious feature file.
- **README.md**: Update user-visible capabilities, command snippets, and template/module tree references.
- **docs/ARCHITECTURE.md**: Keep pipeline, locking, lifecycle, and monitoring structure details accurate.
- **docs/API.md**: Keep endpoint classes, parameters, and outputs accurate.
- **docs/DOCKERHUB_DESCRIPTION.md**: Keep feature summary aligned with README.
- **.agent/** skills/workflows: Update when behavior, quality gates, or execution policy changed.

### 4. Generate GitHub Release Description

Create or update the curated GitHub Release body at
`docs/releases/v<VERSION>_github_description.md` (example:
`docs/releases/v1.2.2_github_description.md`):

- Start with a single H1 that becomes the GitHub Release title in CI
  (e.g. `# Release v1.2.2 - Short Theme`).
- Highlight key features and structural improvements.
- Document optimizations, bug fixes, and security enhancements.
- Include verification results (backend/frontend tests, Playwright E2E, coverage, lint, audit status).

On a tag push matching `vMAJOR.MINOR.PATCH` (e.g. `v1.2.2`), `.github/workflows/ci.yml`
validates the semantic-version format, verifies this file exists, creates the
GitHub Release automatically via the preinstalled `gh` CLI
(`gh release create --title "$RELEASE_NAME" --notes-file <path>`), and uses the
file as release notes (not auto-generated notes). Tags that don't match
`vMAJOR.MINOR.PATCH` fail validation before release creation.

### 5. Consolidate Git Commit

Before staging anything, record the session baseline so the release commit's
scope is verifiable: run `git rev-parse HEAD` at the very start of this
procedure and keep that exact commit hash. This is also the
`<release-parent>` referenced in the squash path below.

A release change set should be staged and committed with a descriptive message.
Clear the index first so only the deliberately reviewed paths below get staged —
any unrelated file already staged from earlier in the session would otherwise be
swept into this commit unnoticed, the same risk the amend and squash paths below
guard against. Then review `git status --porcelain` and stage the files actually
touched by this release (source, tests, docs, the new release-notes file)
deliberately — avoid `git add -A`, which would also sweep in any
unrelated/incidental files that happen to be sitting in the working tree:

  ```bash
  git reset
  git status --porcelain
  git add <the reviewed release paths>
  ```

- Create a release commit with a descriptive title and concise summary bullets:

  ```bash
  git commit -m "v1.x.y: Short Summary of Main Features

  - Detailed bullet point 1
  - Detailed bullet point 2"
  ```

- **Amend path** — the branch's current HEAD commit is already the release
  commit for this version (e.g. it was created earlier in the same session)
  and there are only new uncommitted changes to fold in (later fixes, a final
  release-notes/doc pass). Stage them and refresh the commit's title/description
  via `git commit --amend`:

  ```bash
  # Clear the index first so only the deliberately reviewed paths below get
  # staged -- any unrelated file left staged from earlier in the session
  # would otherwise be swept into the amended commit unnoticed.
  git reset
  git status --porcelain
  git add <the reviewed release paths>
  git commit --amend -m "v1.x.y: Short Summary of Main Features

  - Detailed bullet point 1
  - Detailed bullet point 2"
  ```

  Only amend the release commit itself (the single commit this skill run is
  producing) — never amend a commit that predates this release-prep session or
  that has already been pushed/shared.

  **Recovering from a disallowed path already baked into the commit** — the
  `git reset` above only stops a *new* unrelated file from being staged; it does
  not remove a disallowed path that a prior `git add`/amend in this same session
  already committed (that path is already part of HEAD's tree, so `git reset`
  restores the index right back to including it). If the post-commit check
  below (step 2) reports such a path, recover with one of:

  - Reconstruct the commit from `<release-parent>` instead: follow the squash
    path below (`git reset --soft <release-parent>`, then `git reset`, then
    stage only the reviewed paths) — this rebuilds the tree from scratch so the
    disallowed path is never carried forward.
  - Or restore just the disallowed path(s) in the index back to their
    `<release-parent>` state (not `git rm --cached`, which would delete the
    path from the resulting commit entirely if it's a legitimate pre-existing
    tracked file, not a brand-new one):
    `git restore --source=<release-parent> --staged <disallowed-path>` for
    each one reported. This only touches the index, leaving the working tree
    untouched — it does not by itself change `HEAD`, so run
    `git commit --amend --no-edit` afterward to fold the restored index into
    the commit before re-running the post-commit check (which compares
    against `HEAD`, not the index).

- **Squash path** — multiple *unshared* commits on this branch need to be
  consolidated into one clean release commit (e.g. several fix commits were
  created during this session instead of one). Soft-reset to the parent of the
  release commit range, stage only the deliberately reviewed release paths, and
  create one new commit:

  ```bash
  # <release-parent> is the exact commit hash recorded via `git rev-parse HEAD`
  # at the start of this procedure (step 5's baseline capture, above).
  git reset --soft <release-parent>
  # Clear the index (mixed reset, no path) so only the deliberately reviewed
  # paths below get staged -- a soft reset alone leaves the old index intact,
  # which would re-stage anything from the squashed commits, including any
  # unrelated/incidental files.
  git reset
  git status --porcelain
  git add <the reviewed release paths>
  git commit -m "v1.x.y: Short Summary of Main Features

  - Detailed bullet point 1
  - Detailed bullet point 2"
  ```

  As with the amend path, only ever do this across commits that are unshared
  (not pushed, not merged, not referenced by anyone else) — a soft reset past
  a shared or older commit rewrites history others may depend on.

Do not amend or reset commits unless explicitly requested, or unless one of
the two paths above applies as described.

- After the initial commit (and after either the amend or squash path, if
  used), run two fail-fast checks:

  1. Confirm the reviewed release paths are fully committed. This intentionally
     checks only the paths that were staged for the release, not the whole
     working tree — pre-existing or unrelated changes that were deliberately
     left unstaged (per the `git add -A` warning above) must not be rejected
     here:

     ```bash
     # Check if any of the reviewed release paths still have uncommitted changes
     if [ -n "$(git status --porcelain -- <the reviewed release paths>)" ]; then
       echo "Error: reviewed release paths are not fully committed. Stage and commit them first."
       exit 1
     fi
     ```

  2. Confirm nothing *outside* the reviewed paths made it **into** the commit
     itself — the check above only catches leftovers still uncommitted, not an
     unrelated file that got `git add`-ed alongside the reviewed paths and is
     now baked into the commit. Compare the actual committed diff against the
     same reviewed-path allowlist used for staging. Select the comparison
     command by which workflow path was actually taken — do not rely on one
     command silently failing over to the other, since `git diff A..B` rarely
     errors even when it is the wrong comparison for the situation:

     - **Squash path**: compare against `<release-parent>`, since that commit
       range is exactly what got folded into the new commit:

       ```bash
       committed_paths=$(git diff --name-only "<release-parent>"..HEAD)
       ```

     - **Initial-commit or amend path**: compare against the single commit
       itself. For amend specifically, `git show` reflects the complete,
       final contents of the amended commit (not just what changed since the
       last amend), so a stray path from an earlier amend in the same session
       is still caught:

       ```bash
       committed_paths=$(git show --name-only --pretty='' HEAD)
       ```

     Then, regardless of which command was used:

     ```bash
     unexpected=$(comm -23 <(echo "$committed_paths" | sort -u) <(printf '%s\n' <the reviewed release paths> | sort -u))
     if [ -n "$unexpected" ]; then
       echo "Error: commit includes paths outside the reviewed release allowlist:"
       echo "$unexpected"
       exit 1
     fi
     ```
