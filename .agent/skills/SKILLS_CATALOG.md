# Whisper Pro ASR Skills Catalog

This catalog lists all available project skills and when to use them.

## Mandatory Execution Rule
1. Before any task, read `.agent/instructions.md`, this catalog, and all relevant skill/workflow markdown files.
2. If implementation changes behavior, contracts, architecture, tests, or operations, update all impacted markdown documentation in the same task (`README.md`, `docs/*.md`, `.agent/*.md`).

## Core Existing Skills
- `pipeline_skill.md`: Run local/CI-equivalent lint + tests + coverage.
- `concurrency_orchestration_skill.md`: Validate preemption, yielding, and resource orchestration.
- `intel_hardware_inference_skill.md`: Intel-specific runtime and inference behavior.
- `model_lifecycle_management_skill.md`: Idle timeout/offload behavior and memory lifecycle.
- `doc_update_skill.md`: Sync README/docs/diagrams with architecture changes.
- `prepare_release_skill.md`: Release prep workflow and verification gates.

## New Comprehensive Skills
- `api_contract_validation_skill.md`: Keep `/asr`, `/detect-language`, and OpenAI-compatible routes stable.
- `testing_strategy_skill.md`: Test planning and deterministic concurrency regression workflow.
- `monitoring_telemetry_dashboard_skill.md`: Telemetry payloads, dashboard ordering, and observability checks.
- `docker_runtime_ops_skill.md`: Docker runtime setup, hardware mapping, and operational checks.
- `language_detection_priority_skill.md`: Priority orchestration rules and expected ASR/detect interaction.
- `storage_persistence_hygiene_skill.md`: Temp file cleanup, persistent volumes, and retention behavior.
- `troubleshooting_playbook_skill.md`: Reproduce and isolate deadlocks, stalls, and queue anomalies.
- `ci_quality_gates_skill.md`: Repository quality gates, enforcement, and pre-merge checklist.
- `frontend_quality_gates_skill.md`: Dashboard JS/CSS lint + per-file coverage gates and CI parity workflow.
- `bazarr_integration_skill.md`: Bazarr endpoint/path-mapping and practical integration checks.
- `agent_asset_maintenance_skill.md`: Keep `.agent` instructions, skills, and workflows synchronized with code/process changes.
- `task_status_display_specification_skill.md`: Specifications for dashboard and task status displays.
- `resolve-pr-comments.md`: Guidelines for resolving PR comments and feedback.

## Recommended Usage Order
1. `concurrency_orchestration_skill.md` and `testing_strategy_skill.md` for any scheduler/resource/lifecycle change.
2. `ci_quality_gates_skill.md` for required liveness and documentation gates.
3. `troubleshooting_playbook_skill.md` and `language_detection_priority_skill.md` for issue triage and priority behavior checks.
4. `doc_update_skill.md` and `prepare_release_skill.md` for release finalization.