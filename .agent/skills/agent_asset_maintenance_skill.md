# Agent Asset Maintenance Skill

This skill ensures project agent assets remain accurate whenever behavior, architecture, or process changes.

## Objective
Keep all `.agent` files aligned with current repository reality so agent behavior does not drift from the codebase.

## Trigger Conditions
Run this skill whenever any of the following changes:
- Scheduling, concurrency, or inference behavior.
- API contract, request/response shape, or endpoint semantics.
- CI/lint/test/coverage/release workflow.
- Docker/runtime operational guidance.
- Monitoring, telemetry, dashboard ordering, or diagnostics workflow.

## Required Updates
1. Update `.agent/instructions.md` if global policies or non-negotiable rules changed.
2. Update affected skill files in `.agent/skills/`.
3. Create new skill files in `.agent/skills/` when new behavior, workflows, or operational domains are introduced and existing skills do not fully cover them.
4. Update `.agent/workflows/` when execution flow, gate order, or command steps changed.
5. Update `.agent/skills/SKILLS_CATALOG.md` when adding/removing/renaming skills.
6. Update `README.md` and all affected `.md` files in the repository (`docs/`, release notes, operational guides) when behavior, APIs, workflows, or architecture descriptions changed.

## Completion Checklist
- All touched agent assets reflect current code and docs.
- No stale references to removed files, symbols, or behaviors.
- New behavior has at least one skill or instruction source describing how to validate it.
- New domains introduced by recent changes have dedicated new skills when existing skill coverage is insufficient.
- `README.md` and all impacted Markdown docs are synchronized with the implemented behavior.
- Changes pass local quality gates (`build-and-test.sh`) when relevant to code flow updates.

## Enforcement Rule
Do not close a task that changes system behavior until required `.agent` asset updates are included in the same change set.
