# Task Status & Display Specification Skill

Use this skill whenever making changes to scheduling, preemption, status transitions, or task visibility on the dashboard.

## Objective
Define correct task status display as **Priority 1** for dashboard observability. Ensure all code changes to scheduler, concurrency control, or monitoring preserve status display correctness and require synchronized agent asset updates.

## Non-Negotiable Display Rule

No placeholder of any kind is allowed ever in dashboard-visible status/stage output.
Forbidden display values include (non-exhaustive): `unknown`, `none`, `null`, `undefined`, `(0/0)`, `resuming`, and generic placeholder/sentinel text.
If internal state is undefined or stale, normalize it to a concrete canonical runtime stage before exposure.

## Canonical Task Status Lifecycle

| Status | Meaning | Duration | UI Icon | Color |
|--------|---------|----------|---------|-------|
| `initializing` | Registered, awaiting hardware claim | Seconds | ⏳ hourglass_top | Gray |
| `queued` | Waiting (hardware OR paused for priority) | Variable | ⏳ hourglass_empty | Orange |
| `active` | Running (hardware lock held) | Task length | ♻️ sync (pulse) | Blue |
| `post-processing` | Completion cleanup | Seconds | ♻️ sync | Blue |
| `completed` | Success; archived to history | 0 (archived) | ✅ check | Green |
| `failed` | Error; archived to history | 0 (archived) | ❌ error | Red |
| `unknown` | Internal-only guardrail for invalid state (must be normalized before dashboard output) | 0 | ❓ help | Gray |

## Status + Stage Display Combinations

Distinguish between "Paused for Priority" and "Waiting for Hardware" using **status + stage** tuples:

- `status='queued'` + `stage contains 'Paused for Priority Task'` → **Paused for priority** 
  - UI Display: "Paused for priority task"
  - Icon: hourglass_empty
  - Color: Orange

- `status='queued'` + `stage does NOT contain 'Paused for Priority Task'` → **Waiting for hardware**
  - UI Display: "Waiting for hardware"
  - Icon: hourglass_empty
  - Color: Orange

- `status='active'` → **Running**
  - UI Display: stage name (e.g., "Inference", "Language Detection")
  - Icon: sync with pulse animation
  - Color: Blue

Whisper hardware-unit busy state should follow the same active-ASR rule: while a unit is occupied by an active transcription, translation, or inference sub-stage, its `whisper_status` should be `busy`. UVR/vocal-separation remains tracked separately through `uvr_status`.

- All other statuses: display as-is with their corresponding icon/color.

## Task Ordering Rules (Deterministic)

Dashboard task list MUST always render in this order:

1. **Active tasks first** (all `status='active'`)
   - Sort by `start_time` ascending (oldest first)
   - Tie-break by `task_id` lexicographic order
   
2. **Then priority queued tasks** (all `status='queued'` with `is_priority=true`)
   - Sort by `start_time` ascending
   - Tie-break by `task_id` lexicographic order
   
3. **Then standard queued tasks** (all `status='queued'` with `is_priority=false`)
   - Sort by `start_time` ascending
   - Tie-break by `task_id` lexicographic order

**Verification**: Calling `/status` multiple times on a stable (no new arrivals) system MUST return tasks in identical order.

## Frontend Rendering Checklist

When rendering task cards in dashboard, validate:

- [ ] **Status normalization before render**: Per Non-Negotiable Display Rule and task-card status/stage rules, `unknown` is converted to a concrete canonical runtime state before dashboard output (never rendered as a normal state)
- [ ] **Badge colors**: active=blue, queued=orange, completed=green, failed=red, initializing=gray
- [ ] **Icons**: active=sync, queued=hourglass, completed=check, failed=error, initializing=hourglass_top
- [ ] **Pulse animation**: ONLY on active status
- [ ] **Hardware-wait hint**: Shows "Waiting for hardware" ONLY for `queued` without paused-for-priority in stage
- [ ] **Priority-paused hint**: Shows "Paused for priority task" ONLY for `queued` WITH paused-for-priority in stage
- [ ] **Task ordering**: Rendered order matches deterministic rules above
- [ ] **Active-first policy**: No queued tasks appear above active tasks in list

## Code Change Impact & Validation

Any code change touching the following MUST update this skill if behavior changes:
- Scheduler status update or transition logic (`modules/inference/scheduler.py`)
- Preemption triggering or resumption logic (`modules/inference/concurrency.py`)
- Dashboard status rendering (`modules/monitoring/templates/dashboard_main.js`)
- Status payload assembly (`modules/api/routes_system.py`, `/status` endpoint)

### Mandatory Validation After Changes

1. **Run backend tests**:
   ```bash
   .venv/bin/python -m pytest tests/monitoring/ tests/inference/test_scheduler.py tests/inference/priority/test_priority_fifo_ordering.py -v -k "status or order or preemption"
   ```

2. **Run frontend tests**:
   ```bash
   npm run test:js -- tests/js/dashboard_main.test.js --coverage
   ```

3. **Manual verification**:
   ```bash
   # Start service or connect to running instance
   curl -s http://localhost:9000/status | jq '.tasks[] | {task_id, status, stage, start_time, is_priority}'
   
   # Verify output:
   # - All statuses are one of the 7 canonical values
   # - Task ordering matches deterministic rules (active first, then time-ordered)
   # - No duplicate task_ids
   # - No 'unknown' status unless indicating a bug
   ```

4. **Mixed-load stress test** (ASR + priority tasks):
   ```bash
   # Trigger concurrent standard + priority requests
   for i in {1..3}; do
     curl -X POST -F "audio_file=@sample.mp3" http://localhost:9000/asr &
     curl -X POST -F "audio_file=@sample.mp3" http://localhost:9000/detect-language &
   done
   wait
   
   # Verify /status shows:
   # - Priority tasks visible immediately upon registration
   # - ASR tasks briefly transition to queued with "Paused for Priority Task" stage
   # - Upon priority completion, ASR tasks resume to active (not stuck queued)
   # - No orphaned tasks in unknown state
   ```

## Status Transition State Machine

```
initializing → active          (hardware claim acquired)
active → post-processing       (inference complete, cleanup phase)
post-processing → completed    (success cleanup done)
active → queued                (preemption: transition to paused state)
queued → active                (preemption ended: resume)
queued → completed             (rare: task succeeded while queued)
queued → failed                (rare: error detected in queue)
active → failed                (error during inference)
any → failed                   (error at any stage)
completed/failed → (archived)  (moved to history; removed from active list)
```

## Common Pitfalls & Prevention

| Pitfall | Symptom | Prevention |
|---------|---------|-----------|
| Queued tasks not distinguished by paused vs waiting | Dashboard shows same "Waiting" hint for both priority-paused and hardware-waiting tasks | Always set stage to "Paused for Priority Task" during preemption; validate in dashboard_main.js |
| Non-deterministic ordering | Same system state produces different task order on repeated /status calls | Use start_time + task_id tie-break; never rely on dict/map iteration order |
| Ordering policy inconsistent between backend and frontend | Backend sorts one way, frontend renders differently | Validate ordering rules in both scheduler.py (payload assembly) and dashboard_main.js (rendering) |
| Unknown status leaks to dashboard | "Unknown" status appears in UI, confusing operators | Use defensive assertions in status-setting code; log all unknown-status cases as errors |
| Post-processing status never visible | Task seems to jump from active to completed without transition | Accept that post-processing is typically transient; include in status payload correctly but don't require visible UI presence |

## Done Criteria

 - [ ] Dashboard-visible statuses are concrete and non-placeholder (`initializing`, `queued`, `active`, `post-processing`, `completed`, `failed`)
- [ ] Task ordering in `/status` response is deterministic and matches rules (active first, then time-ordered)
- [ ] Queued tasks distinguish paused-for-priority vs waiting-for-hardware via stage field
- [ ] Dashboard renders all statuses with correct badge colors, icons, and hints
- [ ] Pulse animation only appears on active tasks
- [ ] Backend monitoring tests pass: `pytest tests/monitoring/` includes status transition and ordering checks
- [ ] Frontend dashboard tests pass: coverage ≥90% lines/statements for dashboard_main.js
- [ ] No unknown status leaks to dashboard under normal or stress-test conditions
- [ ] Preemption/resumption cycle preserves deterministic ordering during transitions
- [ ] No placeholder-like status/stage values appear in dashboard payload or rendered UI under normal or stress conditions
