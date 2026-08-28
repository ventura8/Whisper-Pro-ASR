# Frontend Quality Gates Skill

Use this skill whenever dashboard HTML, JavaScript, or CSS files are added/changed.

## Objective

Keep frontend quality gates deterministic and enforceable in local runs and CI.

## Scope

- HTML in `modules/monitoring/templates/*.html`
- JavaScript in `modules/monitoring/templates/dashboard/**/*.js` and `modules/monitoring/templates/analytics/**/*.js`
- CSS in `static/**/*.css` and `modules/monitoring/templates/**/*.css`
- JS tests in `tests/js/**/*.test.js`

Load-order contract note:

- Dashboard and analytics scripts are concatenated via manifest order (`dashboard_js_files.txt` and `analytics_js_files.txt`), not ESM imports. Test fixtures and script loaders must preserve the same ordering.

## Required Gates

All gates below run exclusively inside the Docker test image. Use
`scripts/ci/build-and-test.sh` / `scripts/ci/build-and-test.ps1` (they build
`Dockerfile.test` and run `tests/run_suite.sh` inside the container). Do not run
these npm commands on the host.

Inside `tests/run_suite.sh`, the gates are:

1. HTML lint: `npm run lint:html`
2. JavaScript lint: `npm run lint:js`
3. JavaScript complexity lint: `npm run lint:js:complexity`
4. CSS lint: `npm run lint:css`
5. TOML lint: `npm run lint:toml`
6. JS tests + coverage: `npm run test:js`
7. Playwright E2E (fixture-mock backend): `npm run test:e2e`
8. Playwright E2E (real backend — `tests/e2e/real/`, runs against the actual FastAPI app via `tests/e2e/real_backend/serve_real_app.py` with only ASR inference/language-detection patched): `npm run test:e2e:real`. `playwright.real.config.cjs` single-sources `ADMIN_API_KEY` from `tests/e2e/real/helpers.js`, pins `WHISPER_E2E_PORT` via one shared port constant, and sets `WHISPER_E2E_FAKE_DELAY_SEC=3.0` so lifecycle UI assertions can observe in-flight tasks.
9. Frontend security audit: `npm audit --audit-level=low`
10. Aggregate gate: `npm run quality:frontend`

Note: `npm run quality:frontend` runs both gate 7 (fixture-mock) and gate 8 (real backend). `tests/run_suite.sh` runs each gate individually rather than invoking the aggregate script directly, but covers the same steps; gate 8 requires `WHISPER_PRO_ASR_TEST_IMAGE=1` (from `Dockerfile.test` target `test`), with `SKIP_REAL_E2E=1` as the opt-out.

In CI, gates 1-5 run in the `lint-and-security` job (`PIPELINE_STAGE=lint`) which must complete before any test job starts. Gate 6 runs in `js-unit-tests`, gate 7 in `e2e-fixture`, and gate 8 in `e2e-real` -- each `needs: [build-image, lint-and-security]`. Locally, `build-and-test.sh`/`.ps1` still run every gate in one invocation (`PIPELINE_STAGE` unset/`all`) in lint-then-tests order: lint → js-unit-tests → python-tests → e2e-fixture → e2e-real. ESLint (gates 2-3) and Stylelint (gate 4) run with `--cache --cache-location` pointed at the `whisper-pro-asr-tool-cache` volume so repeat local runs re-lint only changed files.

## Tooling Policy

- Use Playwright via CLI (`npx playwright ...`) or npm scripts that wrap the Playwright CLI.
- Use MCP browser tooling to inspect DOM state, selectors, and runtime page data when diagnosing flaky or unexpected frontend behavior.
- Do not treat manual browser clicks/visual checks as a substitute for Playwright CLI and MCP-backed validation.

## Coverage Policy

- Enforce per-file coverage for monitored JS files.
- Minimum threshold: 90% for `lines` and `statements` per file.
- CI must fail if any monitored JS file drops below the threshold.

## CI Integration

- Ensure `.github/workflows/ci.yml` executes frontend gates through the `Dockerfile.test` test image (`tests/run_suite.sh`), not host Node steps.
- Keep local parity scripts (`scripts/ci/build-and-test.sh`, `scripts/ci/build-and-test.ps1`) aligned with the same Docker test-image frontend gate path.
- The Docker test image must include Node/npm dependencies and Playwright Chromium required for frontend checks.

## Test Strategy Guidance

- Favor deterministic unit tests with mocked DOM, fetch, timers, and charting APIs.
- Keep template HTML structurally valid and lint-clean alongside JS/CSS changes.
- Add branch-targeted tests for queue/task rendering, telemetry chart updates, and export/download paths.
- Avoid disabling lint rules or lowering thresholds to bypass regressions.

## Done Criteria

- The Docker-based lint build stage passes.
- `tests/run_suite.sh` runs and passes all frontend/E2E test suites inside the Docker test image.
- Host-based execution of frontend lints and audits is forbidden; all validation must happen inside Docker.
- Playwright browser binaries are installed automatically inside the Docker test image before E2E execution.
- README and relevant `.agent` docs reflect the Docker-only execution policy.
