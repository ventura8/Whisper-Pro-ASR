# Markdown Quality Gates Skill

Use this skill whenever repository Markdown files are added or changed.

## Mandatory Rule

Every time you work on this project, you MUST update all relevant markdown files in the same task. Never ship code-only changes. After those doc updates, run this skill's lint/fix flow.

## Objective

Keep Markdown documentation lint-clean and auto-fixable in the same Docker-backed quality flow used by the rest of the repository.

## Tooling

- Linter: `markdownlint-cli2`
- Auto-fixer: `markdownlint-cli2 --fix`
- Repo config: `.markdownlint-cli2.jsonc`

## Required Commands

1. Auto-fix Markdown formatting before manual cleanup:

   ```bash
   npm run fix:md
   ```

2. Verify Markdown lint:

   ```bash
   npm run lint:md
   ```

## CI / Local Pipeline Integration

- The `Dockerfile.test` test image pipeline must run `npm run lint:md`.
- `tests/run_suite.sh` must run `npm run lint:md` in the Docker test-image path before the frontend quality gates.
- Local parity scripts must continue using the Dockerfile-based pipeline so Markdown lint runs automatically through the existing build-and-test flow.

## Config Guidance

- Keep `.markdownlint-cli2.jsonc` as the single source of truth for Markdown lint rules and exclusions.
- Prefer rule settings that work with existing project conventions such as embedded HTML and long release-note lines.
- Do not bypass persistent lint failures with broad ignore globs when a narrower config or document fix is possible.

## Done Criteria

- `npm run fix:md` has been applied when formatting drift exists.
- `npm run lint:md` passes.
- Relevant docs and `.agent` skill references are updated when commands or lint expectations change.
