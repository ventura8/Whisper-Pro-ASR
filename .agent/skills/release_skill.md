# Release Skill

**Purpose**: Prepare the curated GitHub Release description for the current version.

When invoked, this skill will:

1. Detect the current Git branch name.
2. Extract the version number from a branch named like `feature/v1.1.6` (preferred), with compatibility for `release/v1.1.6`, `feature/1.1.6`, or `release/1.1.6`.
3. Write or update `docs/releases/vX.Y.Z_github_description.md` with:
   - An H1 that becomes the GitHub Release title in CI.
   - Highlights, fixes, verification results, and any extra notes supplied by the user.
4. Remind the operator that pushing tag `vX.Y.Z` triggers `.github/workflows/ci.yml`, which creates the GitHub Release using that file as the release body.

**Usage**: Prefer `.agent/skills/quality/prepare_release_skill.md` for the full release prep flow (version sync, docs, quality gates, commit). Use this skill when only the GitHub description file needs drafting.

**Note**: Do **not** run `gh release create` unless the user explicitly asks. Tag-push CI owns automatic release creation.
