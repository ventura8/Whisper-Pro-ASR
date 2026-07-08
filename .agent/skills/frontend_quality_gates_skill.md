# Frontend Quality Gates Skill

Use this skill whenever dashboard HTML, JavaScript, or CSS files are added/changed.

## Objective
Keep frontend quality gates deterministic and enforceable in local runs and CI.

## Scope
- HTML in `modules/monitoring/templates/*.html`
- JavaScript in `modules/monitoring/templates/*.js`
- CSS in `static/**/*.css` and `modules/monitoring/templates/**/*.css`
- JS tests in `tests/js/**/*.test.js`

## Required Gates
1. HTML lint: `npm run lint:html`
2. JavaScript lint: `npm run lint:js`
3. CSS lint: `npm run lint:css`
4. JS tests + coverage: `npm run test:js`
5. Frontend security audit: `npm audit --audit-level=low`
6. Aggregate gate: `npm run quality:frontend`

## Coverage Policy
- Enforce per-file coverage for monitored JS files.
- Minimum threshold: 90% for `lines` and `statements` per file.
- CI must fail if any monitored JS file drops below the threshold.

## CI Integration
- Ensure `.github/workflows/ci.yml` runs `npm ci`, `npm audit --audit-level=low`, and `npm run quality:frontend`.
- Keep local parity scripts (`build-and-test.sh`, `build-and-test.ps1`) aligned with CI frontend gates.
- Local parity scripts may bootstrap missing `npm`/`docker` dependencies on Linux, but must still fail hard if bootstrap cannot complete.

## Test Strategy Guidance
- Favor deterministic unit tests with mocked DOM, fetch, timers, and charting APIs.
- Keep template HTML structurally valid and lint-clean alongside JS/CSS changes.
- Add branch-targeted tests for queue/task rendering, telemetry chart updates, and export/download paths.
- Avoid disabling lint rules or lowering thresholds to bypass regressions.

## Done Criteria
- `npm run quality:frontend` passes locally.
- `npm audit --audit-level=low` passes locally.
- CI job executes same frontend gates.
- README and relevant `.agent` docs reflect any changes to commands/policies.
