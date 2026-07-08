# CI Quality Gates Skill

Use this skill before merge/release to enforce repository standards.

## Objective
Maintain a zero-regression quality baseline.

## Required Gates
1. Pylint score: `10.00/10` for project command scope.
2. Test coverage: `>= 90%` (current baseline higher).
3. Full test suite pass.
4. No lint suppressions added as workaround.
5. Frontend quality gate pass (`npm run quality:frontend`), including HTML lint.
6. Frontend security audit pass: `npm audit --audit-level=low`.
7. JS per-file coverage threshold enforced at `>= 90%` for lines/statements on monitored dashboard files.
8. Concurrency-affecting changes must include liveness tests (pause/resume, queued waiting behavior, acquisition behavior) and pass related scheduler suites.
9. Concurrency-affecting changes must include synchronized documentation updates (`README.md`, `docs/CONCURRENCY.md`, and relevant `.agent/skills` files).
10. Local parity scripts (`build-and-test.sh`, `build-and-test.ps1`) must stay aligned with CI gates and dependency bootstrap expectations.

## Verification Commands
```bash
pylint modules/ tests/ whisper_pro_asr.py check_coverage.py
.venv/bin/python -m pytest tests/
npm audit --audit-level=low
npm run quality:frontend
```

## Done Criteria
- Both commands pass in local environment.
- Any changed behavior has matching tests.