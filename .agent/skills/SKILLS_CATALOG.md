# Whisper Pro ASR Skills Catalog

This catalog lists all available project skills and when to use them.

## Mandatory Execution Rule

1. Before any task, read `.agent/instructions.md`, this catalog, and all relevant skill/workflow markdown files.
2. Every time you work on this project, update all relevant markdown files in the same task (`README.md`, `docs/*.md`, `.agent/*.md`). Never ship code-only changes. A task is incomplete until impacted docs describe the current implementation.
3. For frontend/dashboard validation and debugging, prefer Playwright CLI-driven checks and MCP tooling over manual browser-only verification.

## Folder Layout

- `governance/`: Agent maintenance and repository-wide documentation sync.
- `quality/`: Lint, test, pipeline, coverage, and release verification.
- `runtime/`: Scheduler, lifecycle, hardware/runtime, storage, and operational troubleshooting.
- `monitoring/`: Dashboard, telemetry, task status, and observability guidance.
- `integrations/`: External API surface and ecosystem integrations.
- `workflow/`: Human-in-the-loop or PR workflow helpers.

## Governance Skills

- `governance/agent_asset_maintenance_skill.md`: Keep `.agent` instructions, skills, and workflows synchronized with code/process changes.
- `governance/doc_update_skill.md`: Sync README/docs/diagrams with architecture changes.

## Quality Skills

- `quality/ci_quality_gates_skill.md`: Repository quality gates, enforcement, and pre-merge checklist.
- `quality/frontend_quality_gates_skill.md`: Dashboard JS/CSS lint + per-file coverage gates and CI parity workflow. All 10 gates (lint, coverage, Playwright E2E, security audit, aggregate) run exclusively via `scripts/ci/build-and-test.sh` / `.ps1` → Docker `tests/run_suite.sh` (not host npm).
- `quality/markdown_quality_gates_skill.md`: Markdown lint/fix workflow and Docker pipeline integration.
- `quality/pipeline_skill.md`: Run local/CI-equivalent lint + tests + coverage.
- `quality/prepare_release_skill.md`: Release prep workflow and verification gates. Writes curated GitHub Release body to `docs/releases/vX.Y.Z_github_description.md`; tag-push CI creates the GitHub Release from that file.
- `quality/testing_strategy_skill.md`: Test planning and deterministic concurrency regression workflow.

## Runtime Skills

- `runtime/concurrency_orchestration_skill.md`: Validate preemption, yielding, and resource orchestration.
- `runtime/docker_runtime_ops_skill.md`: Docker runtime setup, hardware mapping, and operational checks.
- `runtime/intel_hardware_inference_skill.md`: Intel-specific runtime and inference behavior.
- `runtime/amd_hardware_inference_skill.md`: AMD-specific runtime, DirectML/ROCm pre-processing, and CTranslate2 CPU fallbacks.
- `runtime/language_detection_priority_skill.md`: Priority orchestration rules and expected ASR/detect interaction.
- `runtime/model_lifecycle_management_skill.md`: Idle timeout/offload behavior and memory lifecycle.
- `runtime/storage_persistence_hygiene_skill.md`: Temp file cleanup, persistent volumes, and retention behavior.
- `runtime/troubleshooting_playbook_skill.md`: Reproduce and isolate deadlocks, stalls, and queue anomalies.

## Monitoring Skills

- `monitoring/monitoring_telemetry_dashboard_skill.md`: Telemetry payloads, dashboard ordering, and observability checks.
- `monitoring/task_status_display_specification_skill.md`: Specifications for dashboard and task status displays.

## Integration Skills

- `integrations/api_contract_validation_skill.md`: Keep `/asr`, `/detect-language`, and OpenAI-compatible routes stable.
- `integrations/bazarr_integration_skill.md`: Bazarr endpoint/path-mapping and practical integration checks.

## Workflow Skills

- `resolve-pr-comments/SKILL.md`: Guidelines for resolving PR review comments with `gh` (canonical skill entry).
- `workflow/resolve-pr-comments-run.sh`: Shell helper for the PR comment resolution workflow. Auto-resolve requires explicit operator argv overrides (`EVIDENCE_TEST_PATH_OVERRIDE`, `LINKED_COMMIT_SHA_OVERRIDE`); the linked commit must be an ancestor of the PR head ref (not merely a local or foreign-branch commit). Bot comment bodies are never parsed for verification evidence.

## Recommended Usage Order

1. `runtime/concurrency_orchestration_skill.md` and `quality/testing_strategy_skill.md` for any scheduler/resource/lifecycle change.
2. `quality/ci_quality_gates_skill.md` for required liveness and documentation gates.
3. `runtime/troubleshooting_playbook_skill.md` and `runtime/language_detection_priority_skill.md` for issue triage and priority behavior checks.
4. `governance/doc_update_skill.md` and `quality/prepare_release_skill.md` for release finalization.
