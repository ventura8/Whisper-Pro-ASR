# CI Quality Gates Skill

Use this skill before merge/release to enforce repository standards.

## Objective

Maintain a zero-regression quality baseline.

## Required Gates

1. Pylint score: `10.00/10` for project command scope (executed inside the Docker test image via `tests/run_suite.sh`).
2. Flake8 gate pass for Python sources (`modules`, `whisper_pro_asr.py`, `tests`, `tests/check_coverage.py`) with zero Flake8 ignore directives (executed inside the Docker test image).
3. Markdown lint pass (`npm run lint:md`) via the repo-configured `markdownlint-cli2` gate (executed inside the Docker test image).
4. Test coverage: `>= 90%` (current baseline higher, verified inside the Docker test image).
5. Full test suite pass (executed inside the Docker test image).
6. No lint suppressions added as workaround.
7. Frontend quality gate pass (`npm run quality:frontend`), including HTML lint, executed exclusively inside the Docker image.
8. Frontend security audit pass: `npm audit --audit-level=low`, executed inside the Docker image.
9. Frontend Playwright E2E pass (`npm run test:e2e`), executed inside the Docker image.
10. JS per-file coverage threshold enforced at `>= 90%` for lines/statements on monitored dashboard files.
11. Concurrency-affecting changes must include liveness tests (pause/resume, queued waiting behavior, acquisition behavior) and pass related scheduler suites.
12. Concurrency-affecting changes must include synchronized documentation updates (`README.md`, `docs/CONCURRENCY.md`, and relevant `.agent/skills` files).
13. **Docker-Only execution policy**: Local parity scripts (`scripts/ci/build-and-test.sh`, `scripts/ci/build-and-test.ps1`) must not run eslint, stylelint, hadolint, shellcheck, taplo, bandit, or pip-audit on the host. They must build the Docker test image and run all tests/check output exclusively via that image.
14. Frontend verification and debugging must use Playwright CLI and MCP tooling when available; do not rely on manual-only browser validation.
15. Node.js dependencies and Playwright Chromium must be bootstrapped inside the Docker image for testing.
16. Python source code cyclomatic complexity must have 100% A-grade ranks (Radon cc rank A, score <= 5) for all functions, methods, and blocks.
17. The Docker test image pipeline must fail when any rank-B-or-worse function/block is detected. `run_radon_complexity_gate` uses a **single** `radon cc -s` invocation: output is piped to `tee complexity_output.txt` (the report artifact), then `grep -E ' [B-F] '` on that file detects violations without re-running radon over the source list.
18. `tests/run_suite.sh` inside the Docker test image must execute Radon complexity summary and rank-A enforcement in the `lint` stage, before any test stage (`js-unit-tests`, `python-tests`, `e2e-fixture`, `e2e-real`).
19. In Docker test images, Radon source enumeration must be filesystem-based (e.g., `find ... -name '*.py'`) and must not depend on `.git` metadata.
20. Dockerfile lint gate must pass with Hadolint (`hadolint --failure-threshold warning --disable-ignore-pragma Dockerfile Dockerfile.test`) inside the Docker test image.
21. PowerShell script lint gate must pass with PSScriptAnalyzer inside the Docker test image.
22. Shell script lint gate must pass with ShellCheck inside the Docker test image.
23. CSS lint gate must run explicitly (`npm run lint:css`) inside the Docker test image.
24. HTML lint gate must run explicitly (`npm run lint:html`) inside the Docker test image.
25. Python formatter checks must run in the Docker test image using `black --check .` and `isort --check-only .`.
26. `tests/run_suite.sh` is stage-selectable via the `PIPELINE_STAGE` env var (`all` [default], `lint`, `python-tests`, `js-unit-tests`, `e2e-fixture`, `e2e-real`) so CI can run each stage as its own job while local wrappers (`build-and-test.sh`/`.ps1`) continue to run every stage in one invocation (`PIPELINE_STAGE` unset). For `all`, order is lint → js-unit-tests → python-tests → e2e-fixture → e2e-real. Each stage gate wraps the same code executed by `all` -- no duplicated logic between "run everything" and "run one stage".
27. The `lint` stage's independent tools (including Radon) run concurrently (background jobs + `wait`, aggregating every failure before exiting non-zero) rather than sequentially -- a tool failure must still be reported even if other backgrounded tools are still running or already failed.
28. `python-tests` runs pytest twice: a parallel bulk invocation (`pytest-xdist`, `-n auto --dist=loadscope`) covering everything except the timing-sensitive concurrency/preemption test files, and a separate serial invocation for exactly those files (real-thread + `sleep()`-based synchronization tests that must not compete for CPU with other workers). `--dist=loadscope` keeps each test class on one worker; `tests/class_progress.py` prints each `module.py::TestClass` line when that class's expected results are complete on the controller (not when a worker finishes). Coverage from both is merged via `coverage combine`/`coverage xml` before `tests/check_coverage.py` and `genbadge` run; the 90% overall gate is enforced exactly once, post-combine, via `coverage report --fail-under=90` -- neither `pytest.ini`'s `addopts` nor `.coveragerc`'s `[report]` section may reintroduce a per-invocation `fail_under`/`--cov-fail-under`, since that would false-fail each partial invocation. The final `coverage_output.txt` must start with a pytest-cov-style `----------- coverage: platform ... -----------` header so CI's `MishaKav/pytest-coverage-comment` can parse it (plain `coverage report` omits that line).
29. A named Docker volume (`whisper-pro-asr-tool-cache`, mounted at `/var/cache/whisper-pro-asr-tools`) persists ESLint/Stylelint/ruff/pytest run-time caches across separate `docker run` invocations on the same machine; build-time BuildKit cache mounts alone never reach the running container. This volume is mounted by both local wrappers and every CI job's `docker run`, but only meaningfully speeds up local/repeat runs -- each CI job is a fresh runner VM, so the volume does not persist between CI jobs.
30. The Docker build itself must maximize BuildKit cache-mount reuse: every cache mount (apt, the from-source ffmpeg compile via `ccache`, poetry, npm, Playwright browsers) carries an explicit `id=`. Local wrappers build via `docker buildx build --cache-from=type=local --cache-to=type=local,mode=max` (requires a `docker-container`-driver buildx builder, since the default driver does not support `--cache-to=type=local`) so local iterative builds get the same fast-rebuild property CI already gets from `cache-from/cache-to: type=gha`.
31. Pip must be the current pinned latest everywhere Python tooling is bootstrapped: `ARG PIP_VERSION` in `Dockerfile` / `Dockerfile.test`, `PIP_VERSION` in `scripts/ci/dependencies.env`, and the poetry-lock enforcement steps in `scripts/ci/build-and-test.sh`, `scripts/ci/build-and-test.ps1`, and `.github/workflows/ci.yml` all install/upgrade `pip==${PIP_VERSION}` before Poetry. Do not leave lock-bootstrap containers on the base image's older pip.
32. Local buildx cache export must use a fresh `.docker-build-cache.new` destination and atomically replace `.docker-build-cache` afterward (same pattern in `.sh` and `.ps1`). Exporting `cache-to` into the same directory used for `cache-from` can fail with `mkdir: cannot create directory ''` during local OCI cache finalization.

## Verification Commands

```bash
./scripts/ci/build-and-test.sh

# Or on Windows
powershell -ExecutionPolicy Bypass -File .\scripts\ci\build-and-test.ps1

# Optional explicit image invocation (still Docker-only)
docker build -f Dockerfile.test --target test -t whisper-pro-asr-test .
mkdir -p reports assets
docker run --rm -e CI=true -v "$PWD/assets:/app/assets" -v "$PWD/reports:/reports" whisper-pro-asr-test /bin/bash -lc "tests/run_suite.sh"
```

Lock-file contract: CI verifies `poetry.lock` before Docker builds and fails when the lockfile is missing or out of sync. Local parity scripts may still regenerate when missing or stale.

Frontend tooling contract: local parity scripts must verify Playwright CLI availability, perform idempotent browser bootstrap (`npx playwright install chromium`), and verify MCP CLI availability via `npx @playwright/mcp --help`.

GitHub Actions cache contract: the Docker test image build must use a dedicated GHA cache scope so its environment layers remain reusable across CI runs, and the production image build may read from that test-image scope without overwriting it. In CI, a dedicated `build-image` job is the only writer of that cache scope (`cache-to: type=gha,mode=max,scope=whisper-pro-asr-test`); `lint-and-security` runs next (`needs: build-image`); every test job (`python-tests`, `js-unit-tests`, `e2e-fixture`, `e2e-real`) `needs:` both `build-image` and `lint-and-security` so lint always finishes before tests start (test jobs may still run in parallel with each other). Each test job only reads the image cache (`cache-from`, no `cache-to`) via a fast cache-hit rebuild, and a final `publish` job (`needs:` all of the above) gates release/production-image steps on every stage job succeeding. Runner disk space is maximized prior to large Docker image builds (`build-image` and `publish`) by disabling swap, removing unused pre-installed runner tooling (e.g. .NET, Android SDK, GHC, GHCup, JVM, Swift, Boost, PowerShell, vcpkg, Julia, AWS CLI, `/opt/hostedtoolcache`), clearing apt caches, and pruning Docker/BuildKit assets.

Least-privilege job permissions: `build-image` uses `contents: read` + `actions: write` (required for `cache-to: type=gha`); `lint-and-security`, `js-unit-tests`, `e2e-fixture`, and `e2e-real` use `contents: read`; `python-tests` uses `contents: read` plus `pull-requests`/`issues`/`checks` write for the PR coverage comment (that comment step runs only on `pull_request` -- push events skip it so CI does not need `contents: write` for commit comments); `publish` uses `contents: write` for release creation. Tag-push GitHub Releases are created/updated with the runner's preinstalled `gh` CLI: the step is **idempotent** -- `gh release view` checks for an existing release first; if found, `gh release edit` updates the title and notes; if absent, `gh release create` creates it. This prevents failure on tag re-pushes. JUnit XML merge in `run_suite.sh` checks file existence before calling `ET.parse` so the script continues (and reports pytest/coverage exit codes) when one of the partial XML files is absent.

Local BuildKit cache dirs (`.docker-build-cache`, `.buildx-cache`, `.docker-build-cache.new`, `.docker-build-cache.old`) MUST be listed in `.dockerignore`. They are written under the project root by the parity wrappers / compose builds; omitting them from `.dockerignore` makes `docker buildx build ... .` re-upload the entire cache into the build context (multi-tens-of-GB transfers).

## Done Criteria

- All listed verification commands pass in local environment.
- Any changed behavior has matching tests.
