# Agent Instructions

This file is the canonical pre-task instruction entrypoint for `.agent` assets.

## Mandatory Pre-Task Review

Before implementation:

1. Read `.agent/instructions.md`.
2. Read `.agent/skills/SKILLS_CATALOG.md`.
3. Read all directly relevant skill/workflow files for the task domain.

## Mandatory Markdown Update (Every Task, No Exceptions)

Every time you work on this project, you MUST update all relevant markdown files in the same task. Never ship code-only changes.

- This applies to every kind of change, including CI/CD, Docker, and other infrastructure/tooling work, not just application code.
- Sync every impacted file among `README.md`, `docs/*.md`, and `.agent/*.md` so they describe the current implementation.
- Treat documentation as part of the deliverable, not a follow-up.
- Do not close a task until relevant docs and agent assets are updated.

## Global Execution Rules

- Concurrency correctness (deadlock/livelock safety and bounded progress) takes priority over throughput optimizations.
- Enforce defense-in-depth security: wildcard CORS disabled by default, API/Admin authentication contracts, dynamic model supply chain allowlisting, CSRF origin verification on administrative endpoints, and localhost-only Compose host publish by default (`127.0.0.1:9000:9000`) so unauthenticated management data is not reachable on the LAN.
- Keep endpoint taxonomy contract aligned everywhere:
  - Standard ASR class: `/asr`, `/v1/audio/transcriptions`, `/v1/audio/translations`
  - Priority language-ID class: `/detect-language`, `/detectlang`
- For dashboard/frontend changes, keep frontend quality-gate docs and commands synchronized, including Playwright browser prerequisites and npm audit enforcement.
- **No Inline Suppressions or Disables (Hard Rule)**: Do not use inline lint suppressions, excludes, ignores, or disables anywhere in code or tests (such as linter disable comments, type-ignore annotations, or warning bypasses). All code and tests must be written cleanly to pass quality gates without inline suppressions.

## Agent Asset Maintenance

When code/process changes impact agent guidance:

- Update affected files in `.agent/skills/`.
- Update `.agent/workflows/` when flow/commands change.
- Update `.agent/skills/SKILLS_CATALOG.md` for add/remove/rename changes.
- Keep redirects valid and workspace-relative for moved skills.

## Documentation Completion Rule

Do not close any task until all impacted documentation and agent assets are synchronized with the work just completed.
