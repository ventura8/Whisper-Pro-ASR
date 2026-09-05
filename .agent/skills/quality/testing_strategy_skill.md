# Testing Strategy Skill

Use this skill to design and execute reliable tests, especially for concurrency and preemption behavior.

## Objective

Maintain high-confidence correctness with deterministic tests and >=90% coverage.

For concurrency-impacting changes, liveness regressions are blocking failures.

## Strategy

1. Start with targeted tests for the changed subsystem.
2. Add a regression test that reproduces the exact user-reported sequence.
3. Use deterministic synchronization (`Event`, controlled sleeps, explicit joins).
4. Assert behavior with invariant checks rather than fragile timing-only assertions.
5. Run full suite after targeted passes are green.
6. **Preserve helpful comments** during optimization—code clarity (especially around synchronization and lock ordering) takes precedence over aggressive line count reduction.

## Test Hygiene Patterns

The following patterns are language-neutral and mandatory for all new tests (Python and JS,
e.g. `tests/js/*.test.js`), unless marked Python-only below:

- **Global state cleanup**: Any test that mutates global state (e.g. `scheduler.STATE.task_registry`) MUST restore it in a `try/finally` block so a mid-test assertion failure does not leak dirty state into subsequent tests.
- **Bounded thread joins**: Prefer the shared helpers in `tests/thread_join_helpers.py` (`join_scenario_threads`, `join_if_started`, …) for preemption scenarios that start worker threads, so unstarted/straggler joins stay consistent across priority-stage and e2e preemption modules.
- **Workload-relative timing thresholds**: Use `baseline_duration * N + fixed_overhead` bounds instead of fixed absolute second values for elapsed-time assertions. Fixed bounds are fragile on slow CI runners; relative bounds absorb runner variance while still catching real regressions.

The following patterns are Python-only (they rely on Python-specific language semantics with
no direct JS equivalent) and apply to `tests/**/*.py`, not to `tests/js/*.test.js`:

- **Zip length validation** (Python-only): Use `zip(..., strict=True)` (Python 3.10+) when iterating paired actual/expected lists. Silently truncated mismatches hide bugs.
- **Numeric type assertion** (Python-only): When asserting `isinstance(value, (int, float))`, add an explicit `and not isinstance(value, bool)` guard. `bool` is a subclass of `int` in Python and must be explicitly excluded when numeric intent is required.
- **Shallow copy snapshots** (Python-only): When capturing a dict from a shared registry (e.g. `task_registry`) for later comparison, return `dict(live_entry)` while still holding the registry lock. A live reference can be mutated after the lock releases, causing false assertions.
- **Config `importlib.reload` restore** (Python-only): Any test that mutates `os.environ` and calls `importlib.reload(modules.core.config)` MUST restore the original environment and reload again afterward. Leaving reloaded module globals (e.g. `SUBTITLE_PROMO_ENABLED=True`) pollutes later tests on the same pytest-xdist worker. Use the shared `restore_config_after_reload` fixture from `tests/config_reload_helpers.py` (via `@pytest.mark.usefixtures("restore_config_after_reload")` or `pytestmark`). Apply this on every module that reloads config (including `test_config.py`, AMD/SSD/logging/robustness suites), and keep formatter assertions that care about cue indices patched with `SUBTITLE_PROMO_ENABLED=False`.
- **Preemption pause confirmation** (Python-only): Tests that assert `pause_confirmed` after `check_preemption()` MUST leave at least one `is_priority=True` task in `task_registry`. With no priority work, the runtime self-heals the pause and clears `pause_confirmed` before the main thread can observe it.

## Priority/Concurrency Test Guidance

1. Explicitly control task arrival ordering.
2. Track state transitions (`queued`, `active`, `paused`) through events.
3. Validate liveness (`thread.join(timeout=...)` then `is_alive()` assertions).
4. Validate fairness and throughput separately:
   - fairness: ordering invariants
   - throughput: max concurrent workers
5. Validate wait policy explicitly: priority/preemption paths do not fail on scheduler timeout, and queued waits unblock correctly when resources/events are released.
6. Add at least one stress-style regression for queue contention when scheduler behavior changes.

## Task Status Display Regression Testing

For any change affecting scheduler status updates, preemption, or task ordering:

### Dashboard Concurrency UI E2E Interaction Coverage

Add Playwright E2E tests that exercise real user interactions (tabs + filter buttons) against the dashboard while the system is in an active concurrency burst scenario.

Minimum checks that we consider "full interaction" for this UI layer:

- task filter buttons correctly narrow visible task cards (active vs queued vs paused-for-priority vs v1 categories)
- switching to the history tab shows the correct empty-state (no placeholder/sentinel strings)
- placeholder-like tokens (`unknown`, `null`, `undefined`, `none`, `(0/0)`) never appear in DOM text during concurrency

### 1. Status Transition Test

Add test to validate correct status/stage semantics during preemption:

```python
def test_task_status_transitions_during_preemption():
    """Verify queued+paused-stage distinguishes from true hardware-wait during preemption."""
    # Start ASR task on hardware
    asr_task_id = enqueue_asr_task()
    wait_for_status(asr_task_id, "active", timeout=5)

    # Trigger priority detection → should preempt ASR
    priority_task_id = enqueue_priority_task()

    # Wait on the ASR task itself reaching queued+paused, not on the priority
    # task reaching queued -- the latter can observably happen a moment before
    # the scheduler has actually flipped ASR's stage to the paused marker,
    # making an assertion right after it a race. Poll the actual condition
    # this test cares about instead of a proxy for it.
    def _asr_is_queued_and_paused():
        status_data = get_status()
        asr_in_queue = next((t for t in status_data["tasks"] if t["task_id"] == asr_task_id), None)
        return asr_in_queue is not None and asr_in_queue["status"] == "queued" and "Paused for Priority Task" in asr_in_queue["stage"]

    wait_until(_asr_is_queued_and_paused, timeout=3)

    # Verify ASR now shows paused stage (not stuck queued)
    status_data = get_status()
    asr_in_queue = next((t for t in status_data["tasks"] if t["task_id"] == asr_task_id), None)
    assert asr_in_queue is not None, "ASR task missing from queue"
    assert asr_in_queue["status"] == "queued", "ASR should be queued during preemption"
    assert "Paused for Priority Task" in asr_in_queue["stage"], f"Stage should contain paused marker, got: {asr_in_queue['stage']}"

    # Wait for priority completion
    wait_for_status(priority_task_id, "completed", timeout=10)

    # Verify ASR resumes to active
    wait_for_status(asr_task_id, "active", timeout=5)
    asr_resumed = get_task_status(asr_task_id)
    assert asr_resumed["status"] == "active", "ASR should resume to active after priority completion"
```

### 2. Ordering Determinism Test

Ensure `/status` ordering is identical across repeated calls:

```python
def test_task_ordering_deterministic_across_calls():
    """Verify task ordering is stable across multiple /status calls."""
    # Trigger concurrent mixed-priority arrivals
    tasks = []
    for i in range(5):
        tasks.append(('asr', enqueue_asr_task()))
        tasks.append(('priority', enqueue_priority_task()))
    
    # Wait for all to stabilize
    time.sleep(2)
    
    # Call /status multiple times and collect orderings
    orderings = []
    for call_num in range(5):
        status_data = get_status()
        order = [t['task_id'] for t in status_data['tasks']]
        orderings.append(order)
    
    # Verify all orderings identical
    for call_num in range(1, 5):
        assert orderings[call_num] == orderings[0], f"Ordering changed between call 0 and call {call_num}"

def test_task_ordering_active_first_then_non_active_by_start_time():
  """Verify active tasks render first, then all non-active tasks by start_time."""
  # Create mixed tasks: 2 active, 1 queued-priority, 2 queued-standard
    active_tasks = [enqueue_asr_task() for _ in range(2)]
    for task_id in active_tasks:
        wait_for_status(task_id, 'active', timeout=3)
    
    # Pause with priority
    priority_task = enqueue_priority_task()
    wait_for_status(priority_task, 'queued', timeout=2)
    
    # Now issue more standard tasks (will be queued)
    standard_tasks = [enqueue_asr_task() for _ in range(2)]
    time.sleep(1)
    
    # Get ordered list
    status_data = get_status()
    task_ids = [t['task_id'] for t in status_data['tasks']]
    
    # Find indices
    active_indices = [task_ids.index(tid) for tid in active_tasks]
    priority_index = task_ids.index(priority_task)
    standard_indices = [task_ids.index(tid) for tid in standard_tasks]
    task_by_id = {t['task_id']: t for t in status_data['tasks']}
    
    # Verify ordering: all active indices < all non-active indices, with time ordering inside non-active.
    max_active = max(active_indices) if active_indices else -1
    min_non_active = min([priority_index] + standard_indices) if standard_indices else priority_index
    non_active_ids = [tid for tid in task_ids if task_by_id[tid].get('status') != 'active']
    expected_non_active = sorted(
      non_active_ids,
      key=lambda tid: (task_by_id[tid].get('start_time', 0), tid),
    )
    expected_active = sorted(
      active_tasks,
      key=lambda tid: (task_by_id[tid].get('start_time', 0), tid),
    )
    
    assert max_active < min_non_active, f"Active tasks should appear before all non-active tasks"
    assert non_active_ids == expected_non_active, "Non-active tasks must be sorted by start_time then task_id"
    assert [tid for tid in task_ids if tid in active_tasks] == expected_active, "Active tasks must be sorted by start_time then task_id"
    assert priority_task in non_active_ids, "Priority task should remain in the non-active group ordering"
```

### 3. Frontend Status Rendering Matrix Test

Add to `tests/js/dashboard_main.test.js` to validate all 7 statuses render correctly:

```javascript
describe('Task Status Rendering (All 7 Statuses)', () => {
  
  test('All status values render with correct badge colors and icons', () => {
    const statusExpectations = {
      'initializing': { badgeClass: 'badge-initializing', icon: 'hourglass_top', pulse: false },
      'queued': { badgeClass: 'badge-queued', icon: 'hourglass_empty', pulse: false },
      'active': { badgeClass: 'badge-active', icon: 'sync', pulse: true },
      'post-processing': { badgeClass: 'badge-active', icon: 'sync', pulse: true },
      'completed': { badgeClass: 'badge-completed', icon: 'check_circle', pulse: false },
      'failed': { badgeClass: 'badge-failed', icon: 'error', pulse: false },
      'unknown': { badgeClass: 'badge-unknown', icon: 'help', pulse: false }
    };
    
    Object.entries(statusExpectations).forEach(([status, expected]) => {
      const mockTask = {
        task_id: `test-${status}`,
        status: status,
        stage: 'Test Stage',
        type: 'Transcription',
        progress: 50,
        start_time: Math.floor(Date.now() / 1000)
      };
      
      const rendered = renderTaskCard(mockTask);
      const badge = rendered.querySelector('[class*="badge-"]');
      const icon = rendered.querySelector('.material-icons-sharp');
      
      expect(badge.className).toContain(expected.badgeClass);
      expect(icon.textContent).toBe(expected.icon);
      
      if (expected.pulse) {
        expect(icon.classList.contains('pulse')).toBe(true);
      } else {
        expect(icon.classList.contains('pulse')).toBe(false);
      }
    });
  });

  test('Paused-for-priority stage shows distinct hint from hardware-wait queued', () => {
    const pausedTask = {
      task_id: 'paused',
      status: 'queued',
      stage: 'Paused for Priority Task',
      type: 'Transcription',
      start_time: Math.floor(Date.now() / 1000),
      is_priority: false
    };
    
    const waitingTask = {
      task_id: 'waiting',
      status: 'queued',
      stage: 'Initializing',
      type: 'Transcription',
      start_time: Math.floor(Date.now() / 1000),
      is_priority: false
    };
    
    const pausedHint = extractWaitHint(pausedTask);
    const waitingHint = extractWaitHint(waitingTask);
    
    expect(pausedHint.toLowerCase()).toContain('paused for priority');
    expect(waitingHint.toLowerCase()).toContain('waiting for hardware');
    expect(pausedHint).not.toEqual(waitingHint);
  });

  test('Task ordering is deterministic (active before queued before completed)', () => {
    const tasks = [
      { task_id: 'q1', status: 'queued', start_time: 100, is_priority: false },
      { task_id: 'a1', status: 'active', start_time: 50, is_priority: false },
      { task_id: 'q2', status: 'queued', start_time: 90, is_priority: false },
      { task_id: 'a2', status: 'active', start_time: 60, is_priority: false },
      { task_id: 'c1', status: 'completed', start_time: 10, is_priority: false }
    ];
    
    const sorted = sortTasksForDisplay(tasks);
    const order = sorted.map(t => t.task_id);
    
    // Active (by start_time), then queued (by start_time), then completed
    // a1 (50), a2 (60), q2 (90), q1 (100), c1 (10 but completed, hidden)
    expect(order[0]).toBe('a1');
    expect(order[1]).toBe('a2');
    expect(order[2]).toBe('q2');
    expect(order[3]).toBe('q1');
  });
});
```

## Validation Commands

```bash
# End-to-end concurrency suite in Docker (HW matrix, volume tiers, yielding stages, idle timeout, edge cases, telemetry, UI HTML)
docker run --rm -v "$(pwd):/app" -w /app whisper-pro-asr-test python3 -m pytest tests/integration/concurrency/ -v

# Playwright E2E UI concurrency & preemption tests
docker run --rm -v "$(pwd):/app" -w /app whisper-pro-asr-test npm run test:e2e

# Backend status/ordering/priority unit & integration tests
docker run --rm -v "$(pwd):/app" -w /app whisper-pro-asr-test python3 -m pytest tests/monitoring/ tests/inference/scheduler/ tests/integration/concurrency/ -v -k "status or order or preemption"

# Frontend status rendering tests
docker run --rm -v "$(pwd):/app" -w /app whisper-pro-asr-test npm run test:js -- tests/js/dashboard_main.test.js --coverage

# Full suite in Docker
scripts/ci/build-and-test.sh
```

## Real-Audio Matrix (`tests/real_audio/`)

Everything else in the suite mocks the ASR engine, so multilingual regressions are
invisible to it. `tests/real_audio/` drives a **live** service over HTTP with real neural
speech per language, code-switched clips, degraded/malformed audio, and a 20-minute
long-form stress clip.

Rules when working in this area:

- **Expectations are data.** `tests/e2e/fixtures/audio_matrix/manifest.json` carries each
  clip's tier, expected words, acceptable detection codes, tokenizer and optional
  `xfail_reason`. Tune a language by editing the manifest, **never** by editing a test or
  loosening an assertion in code.
- **Tier A** asserts transcript content *and* detected language. **Tier B** (the long tail)
  asserts "detected correctly, or at least transcribed to something". Promote a language
  between tiers only on evidence from real runs.
- **Fixtures are generated, not mystery binaries.** `scripts/generate_audio_matrix.py`
  renders every clip from the manifest with Piper. Only the ~10-language core tier is
  committed (`core/*.flac`, ~1.3 MB); everything else is cached in the gitignored
  `test_data/audio_matrix/`. Generation is content-addressed and idempotent -- re-running
  it must leave `git status` clean.
- **Determinism is pinned deliberately.** Piper has no seed and samples noise per run; the
  manifest pins `noise_scale`/`noise_w_scale` to zero to make rendering bit-identical, and
  voice models are verified against upstream MD5 digests. Do not "improve" prosody by
  raising those pins without accepting that committed fixtures then churn on every rebuild.
- **The generator lives in `scripts/`** because that path is Radon rank-A gated
  (complexity <= 5) but coverage-exempt. Keep functions decomposed; use dispatch tables
  rather than `if`/`elif` chains.
- **Markers**: `real_asr`, `real_audio`, `smoke`, `gpu`, `slow` (registered in
  `pytest.ini`). The long-form test carries `gpu` + `slow` and requires
  `RUN_GPU_LONG_ASR=1` plus `nvidia-smi`.
- **Run the smoke set, not the matrix, by default.** `smoke` selects a representative
  subset (24 of 156 tests: 4 languages spanning Latin/Cyrillic/CJK, one code-switched
  clip, five degraded and malformed cases, the request-contract checks) budgeted under 20
  minutes. The full matrix is ~2 hours because each request runs UVR Vocal Separation, so
  it is stress testing, not routine validation. Which entries are in the smoke set is
  manifest data (`"smoke": true`), not a hard-coded list in a test.
- **Known engine defects are recorded, never hidden.** `xfail_reason` plus `xfail_scope`
  on a manifest entry relaxes only the assertion the defect actually breaks; the long-form
  spec carries `known_defects` keyed by property. A fix is applied by deleting that data,
  and the test flips to XPASS on its own. Lowering a threshold to make a defect disappear
  is not an acceptable alternative.

Both real-audio stages run through the Docker test image against a **live** stack, like
every other gate in this repository -- never as host pytest:

```bash
RUN_REAL_ASR=1 PIPELINE_STAGE=real-audio scripts/ci/build-and-test.sh          # smoke, <20 min
RUN_REAL_ASR=1 PIPELINE_STAGE=real-audio-stress scripts/ci/build-and-test.sh   # full matrix ~2h + long-form
python3 scripts/generate_audio_matrix.py verify                                # coverage report
```

The stack itself must already be up on the override matching `BUILD_TARGET`; neither stage
starts it. `BUILD_TARGET` lives in `.env` (written by `scripts/audit_hardware.sh --env`), so
source it before expanding the filename -- an unset variable yields
`docker-compose..yml`, which does not exist:

```bash
set -a; . ./.env; set +a
docker compose -f docker-compose.yml -f "docker-compose.${BUILD_TARGET}.yml" up -d
```

## Done Criteria

- New regression tests exist and pass (backend + frontend).
- Existing related tests pass.
- Full suite passes with coverage gate.
- Status transition test validates pause/resume correctness.
- Ordering determinism test confirms stable sort across calls.
- Frontend rendering test validates all 7 statuses render correctly.
- Paused-vs-waiting hint distinction validated in frontend test.
- Playwright E2E UI concurrency test validates multi-hardware cards, active session counters, preemption hint banners, and zero DOM placeholders.
