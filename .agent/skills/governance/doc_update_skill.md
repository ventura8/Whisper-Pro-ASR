# Document Update Skill

This skill automates synchronization of repository documentation with the current codebase and release state.

## Mandatory Rule

Every time you work on this project, you MUST update all relevant markdown files in the same task. Never ship code-only changes. A task is incomplete until `README.md`, `docs/*.md`, and affected `.agent/*.md` files describe the current implementation.

## Objective

Review the full current change set for documentation drift, then update `README.md`, affected files in `docs/`, and impacted Mermaid diagrams in `.md` files so they match implemented behavior.

When frontend dashboard/analytics assets change, documentation must describe:

- Manifest-driven script assembly (`dashboard_js_files.txt`, `analytics_js_files.txt`).
- Current folder structure under `modules/monitoring/templates/dashboard/` and `modules/monitoring/templates/analytics/`.
- Any updated quality-gate commands and prerequisites (for example Playwright browser install requirements).

When release-impacting behavior is changed, add or update a file in `docs/releases/` describing the delta and verification outcomes.

Also enforce the Concurrency-First policy: concurrency correctness must be reflected consistently across user docs and agent governance docs.
Also enforce endpoint taxonomy consistency: `/asr` and `/v1/audio/...` are one standard ASR class; `/detect-language` and `/detectlang` are one priority language-ID class.
When Markdown files are touched, run the repo Markdown auto-fix/lint flow (`npm run fix:md`, then `npm run lint:md`) before closing the task.

## Required Concurrency-First Checklist

- `README.md`: includes project-level concurrency priority statement.
- `docs/Instructions.md`: includes mandatory concurrency checklist.
- `docs/CONCURRENCY.md`: canonical lock order and bounded-wait policy.
- `docs/ARCHITECTURE.md`: concurrency safety/liveness boundaries.
- `docs/API.md`: endpoint concurrency semantics.
- `docs/TUNING.md`: liveness-safe tuning guidance.
- `docs/SETUP.md`: concurrency verification commands.
- `docs/DOCKERHUB_DESCRIPTION.md`: concise reliability wording.
- `.agent/instructions.md` and relevant `.agent/skills/*.md`: concurrency-first governance alignment.
- `docs/API.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/CONCURRENCY.md`, `docs/Instructions.md`: endpoint taxonomy and alias policy alignment.

## Procedure

### 0. Review the Full Current Commit

Before editing docs, inspect the full current commit diff so documentation and release notes are checked against every changed file, not just the obvious feature surface.

### 1. Update Mermaid Diagrams

Locate and update Mermaid diagrams in all project `.md` files to match the current architecture, scheduler semantics, and monitoring pipelines.

### 2. Update `README.md`

- Keep feature and architecture sections aligned with current behavior.
- Keep dashboard/analytics template tree and file references aligned with current JS/CSS/HTML layout.
- Keep quality-gate command snippets current with actual scripts and prerequisites.

### 3. Update `docs/ARCHITECTURE.md`

- Keep concurrency, pipeline, and model lifecycle details synchronized with implementation.
- Keep monitoring/dashboard architecture details synchronized, including modular template and manifest loading behavior.

### 4. Update `docs/CONCURRENCY.md`

- Ensure lock ordering, preemption, and bounded progress policies match implementation and tests.

### 5. Update `docs/API.md`

- Ensure endpoint classes, request/response semantics, and parameter tables remain aligned with implementation.

### 6. Update `docs/DOCKERHUB_DESCRIPTION.md`

- Synchronize feature and operation wording with README and current runtime behavior.

### 7. Update `docs/SETUP.md`

- Keep setup and prerequisite steps aligned with current runtime and frontend tooling needs.

### 8. Update `docs/TUNING.md`

- Keep tuning guidance aligned with current knobs and operational recommendations.

### 9. Release Notes Synchronization

Create or update the corresponding file under `docs/releases/` whenever behavior, architecture, testing strategy, or quality-gate outcomes materially change.

### 10. Hardware Compatibility Matrix

Ensure a **Hardware Compatibility Matrix** is present and updated in `README.md`, `docs/ARCHITECTURE.md`, and `docs/DOCKERHUB_DESCRIPTION.md`. The matrix must accurately reflect current backend support for Vocal Separation, ASR Inference, and Speaker Diarization across CPU, NVIDIA, and Intel architectures.
