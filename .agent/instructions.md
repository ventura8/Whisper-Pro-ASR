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
- Enforce defense-in-depth security: wildcard CORS disabled by default, API/Admin authentication contracts, dynamic model supply chain allowlisting, and CSRF origin verification on administrative endpoints.
- Keep endpoint taxonomy contract aligned everywhere:
  - Standard ASR class: `/asr`, `/v1/audio/transcriptions`, `/v1/audio/translations`
  - Priority language-ID class: `/detect-language`, `/detectlang`
- For dashboard/frontend changes, keep frontend quality-gate docs and commands synchronized, including Playwright browser prerequisites and npm audit enforcement.
- For multilingual or language-detection changes, extend the real-audio matrix (`tests/real_audio/`) rather than relying on mocked engine tests. Its expectations live as data in `tests/e2e/fixtures/audio_matrix/manifest.json`; tune a language by editing that manifest, never by weakening an assertion in a test. Run the smoke set (`-m "real_audio and smoke"`, under 20 minutes) for routine validation; the full matrix and the 20-minute long-form clip are opt-in stress runs. See `.agent/skills/quality/testing_strategy_skill.md`.
- **No Inline Suppressions or Disables (Hard Rule)**: Do not use inline lint suppressions, excludes, ignores, or disables anywhere in code or tests (such as linter disable comments, type-ignore annotations, or warning bypasses). All code and tests must be written cleanly to pass quality gates without inline suppressions.

## Agent Asset Maintenance

When code/process changes impact agent guidance:

- Update affected files in `.agent/skills/`.
- Update `.agent/workflows/` when flow/commands change.
- Update `.agent/skills/SKILLS_CATALOG.md` for add/remove/rename changes.
- Keep redirects valid and workspace-relative for moved skills.

## Documentation Completion Rule

Do not close any task until all impacted documentation and agent assets are synchronized with the work just completed.

## Hardware Validation After Every Review Wave (Mandatory)

A review wave (CodeRabbit or otherwise) is **not finished when the mocked suite is green**.
Every test in this repository except `tests/real_audio` and
`tests/integration/test_transcription_accuracy.py` mocks the ASR engine, so a broken
accelerator path passes all of them.

Before reporting a wave complete:

1. Work out which accelerator paths the wave touched, and pick the host that can actually
   exercise each one. `scripts/audit_hardware.sh` on that host says what it really has.
2. Run `scripts/remote_validate.sh <user>@<host> --target <t> --device <d> --full --suite smoke`
   (or `scripts/validation_matrix.sh` for several machines), and read the startup banner
   rather than the HTTP status -- a 200 is entirely compatible with a silent CPU fallback.
   `--full` selects the whole sync/build/run pipeline; `--suite` selects the test depth, and
   the two are independent. Use `smoke` for a wave -- `full` and `stress` are release depth,
   and a wave that spends two hours per host is one that gets skipped next time.
3. Watch that it is actually progressing, not merely running. A wedged service looks
   identical to a slow one from the outside: check the task list rather than the elapsed
   time. Queued tasks spaced at exactly `REAL_ASR_TIMEOUT` apart mean nothing is completing
   at all -- that pattern was a real worker-channel deadlock, not a slow machine.
4. Report the result. If a host is unreachable, name the change that is therefore
   unvalidated. Never present the mocked run as if it were hardware validation.

| Change area | Host that can validate it |
| :--- | :--- |
| `modules/core/config*.py` engine/device/pool resolution | the box owning that accelerator (NPU logic needs the Intel NPU host) |
| `preprocessor_pool`, `isolation_policy`, UVR routing | a hybrid NVIDIA+Intel host, with vocal separation enabled |
| CUDA engine/decode paths | an NVIDIA host |
| manifest expectations, tokenizers, scoring | the real-audio matrix (`--suite smoke` or above, requires synchronized fixtures) |

**Why this rule exists.** It was skipped once. The 2026-09-06 wave changed `config.py` to
drop a rejected NPU from the scheduler pool, passed 1826 mocked tests, and then failed 8 of
9 accuracy tests on the NUC: with `ASR_DEVICE=NPU` the pool is already narrowed to that one
unit, so retyping it to `CPU` made `engine_factory` degrade `INTEL-WHISPER` to
Faster-Whisper and hand the OpenVINO IR directory to CTranslate2
(`Unable to open file 'model.bin'`). No mocked test could have caught it, and the CT2
corruption handler could have purged the OpenVINO weights.
