# Monitoring, Telemetry, and Dashboard Skill

Use this skill for any changes that impact task visibility, ordering, progress, or analytics.

## Objective
Keep observability trustworthy across:
- `/status` payload (task list, ordering, status values)
- Dashboard active/queued/history rendering
- Telemetry retention/downsampling
- Analytics aggregation
- Frontend status rendering (icons, badges, hints)

## Non-Negotiable Display Rule

No placeholder of any kind is allowed ever in dashboard-visible status/stage output.
Forbidden display values include (non-exhaustive): `unknown`, `none`, `null`, `undefined`, `(0/0)`, `resuming`, and sentinel placeholder labels.
If runtime state is undefined, normalize it to a concrete canonical runtime label before payload assembly and frontend rendering.

## Key Invariants & Display Rules

### 1. Task Ordering Determinism

Dashboard task list MUST always render in this **exact** order:

1. **Active Tasks First** (all `status='active'`)
   - Sort by `start_time` ascending (oldest first)
   - Tie-break by `task_id` lexicographic order
   - No priority override inside active tier (pure time order)

2. **Then Priority Queued Tasks** (all `status='queued'` with `is_priority=true`)
   - Sort by `start_time` ascending
   - Tie-break by `task_id` lexicographic order

3. **Then Standard Queued Tasks** (all `status='queued'` with `is_priority=false`)
   - Sort by `start_time` ascending
   - Tie-break by `task_id` lexicographic order

**Verification**: Calling `/status` multiple times on a stable (no new arrivals) system MUST return tasks in identical order.

### 2. Status + Stage Display Combinations

| Status | Stage | UI Display | Icon | Color | Pulse |
|--------|-------|------------|------|-------|-------|
| `initializing` | (any) | "Initializing" | ⏳ hourglass_top | Gray | ✗ |
| `queued` | contains "Paused for Priority Task" | "Paused for priority task" | ⏳ hourglass_empty | Orange | ✗ |
| `queued` | (other or null) | "Waiting for hardware" | ⏳ hourglass_empty | Orange | ✗ |
| `active` | (any) | <stage_name> | ♻️ sync | Blue | ✓ |
| `post-processing` | (any) | "Finalizing..." | ♻️ sync | Blue | ✓ |
| `completed` | (any) | (hidden; history only) | ✅ check | Green | ✗ |
| `failed` | (any) | (hidden; history only) | ❌ error | Red | ✗ |

### 3. Priority Task Visibility & Preemption

- Priority tasks (`/detect-language` and `/detectlang`, language identification) are registered with `status='queued'` (never `initializing`).
- Priority tasks appear immediately in `/status` upon registration.
- When priority task preemption triggers, concurrent standard tasks transition to `queued` with stage "Paused for Priority Task".
- Upon priority completion, standard tasks resume: briefly may appear `queued`, then immediately transition back to `active`.
- **Expected behavior**: No stuck queued tasks after priority preemption completes.

### Endpoint Surface Taxonomy

- `/asr` and `/v1/audio/...` are equivalent standard-priority ASR surfaces.
- `/detect-language` and `/detectlang` are equivalent high-priority language-ID surfaces.
- Monitoring and analytics may display these surfaces separately for observability, but scheduler class semantics must remain equivalent within each class.

### 4. Hardware Utilization Telemetry

- Active tasks with stage containing "inference"/"transcrib" count toward ASR utilization.
- Active tasks with stage containing "isolation"/"uvr"/"separation" count toward UVR utilization.
- Utilization % calculated as: `(actual_active_count) / hardware_unit_count * 100`.
- Queued tasks do NOT count toward utilization (they are not consuming hardware).

### 5. Live Progress Updates

- Dashboard polls `/status` every 2 seconds.
- Task progress fields (`current_position`, `progress`, `stage`) update in-place without removing/re-adding task card.
- Log buffers auto-scroll to bottom on update.
- Chart downsampling: server limits to 300 points; client applies additional downsampling for browser performance.

## Status Payload Contract

### `/status` Response Fields (per task object)

```json
{
  "task_id": "uuid-string",
  "type": "Transcription|Language Detection|Vocal Separation",
   "status": "initializing|queued|active|post-processing|completed|failed",
  "stage": "Pipeline stage name (e.g., 'Inference', 'Paused for Priority Task')",
  "start_time": 1234567890 (Unix timestamp),
  "progress": 0-100 (integer percent),
  "current_position": "HH:MM:SS or media timestamp",
  "is_priority": true|false
}
```

**Validation**:
- Dashboard-visible status values must be concrete and non-placeholder.
- Stage field MUST contain "Paused for Priority Task" during preemption (not optional).
- `start_time` MUST be populated for deterministic ordering.
- No extra fields that leak internal state.
- No placeholder-like status/stage values in dashboard-visible payload fields.

## Validation Commands

```bash
# Run full monitoring + concurrency test suite
.venv/bin/python -m pytest tests/monitoring/ tests/inference/test_scheduler.py tests/inference/priority/ -v -k "status or order or preemption"

# Run telemetry-specific tests
.venv/bin/python -m pytest tests/monitoring/test_telemetry_loop.py tests/monitoring/test_history_manager.py -q

# Manual endpoint validation
curl -s http://localhost:9000/status | jq '.tasks[] | {task_id, status, stage, start_time, is_priority}'

# Verify:
# - All statuses are one of 7 canonical values
# - Task ordering matches deterministic rules (active first, then time-ordered)
# - No duplicate task_ids
# - Queued tasks show stage with 'Paused' info if preempted
# - No 'unknown' status unless bug indicator
```

## Frontend Rendering Alignment

The frontend dashboard (dashboard_main.js) MUST implement status rendering per these rules:

1. **Badge Colors & Icons**:
   - active → badge-active, sync icon with pulse
   - queued → badge-queued, hourglass icon
   - initializing → badge-initializing, hourglass_top icon
   - completed → badge-completed (history only), check icon
   - failed → badge-failed (history only), error icon
   - unknown is internal-only and must never be rendered in dashboard-visible task cards

2. **Hint Text for Queued Tasks**:
   ```javascript
   const isPausedForPriority = t.status === 'queued' && 
     (t.stage || '').toLowerCase().includes('paused for priority task');
   const hint = isPausedForPriority 
     ? 'Paused for priority task' 
     : 'Waiting for hardware';
   ```

3. **Task Ordering in Rendered List**:
   - Must match backend ordering rules (active first, then priority queued, then standard queued)
   - Sort function must be deterministic and repeated calls produce same order

## Change Impact & Validation

Any code change touching the following MUST be validated against these rules:
- Scheduler status update or transition logic
- Preemption triggering or resumption
- Dashboard status rendering
- Status payload assembly in `/status` endpoint

### Mandatory Checks After Changes

1. **Order consistency test**: Call `/status` 5 times in quick succession; assert identical ordering all 5 times.
2. **Status rendering test**: For all 7 statuses, verify badge class, icon, and pulse animation match table above.
3. **Preemption test**: Verify queued tasks show "Paused for Priority Task" stage during priority preemption.
4. **Utilization telemetry test**: Verify active task counts correctly toward hardware utilization %.

## Done Criteria
- Dashboard ordering and stage transitions are correct per rules above.
- Task list deterministically orders active-first, then time-ordered within each status tier.
- Queued tasks correctly display "Paused for priority" vs "Waiting for hardware" hints based on stage.
- Telemetry and history tests pass: all 7 status values appear correctly in aggregations.
- No broken status payload fields.
- Frontend tests pass with ≥90% lines/statements coverage (branches/functions pragmatic).