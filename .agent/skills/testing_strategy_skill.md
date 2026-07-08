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

## Priority/Concurrency Test Guidance
1. Explicitly control task arrival ordering.
2. Track state transitions (`queued`, `active`, `paused`) through events.
3. Validate liveness (`thread.join(timeout=...)` then `is_alive()` assertions).
4. Validate fairness and throughput separately:
   - fairness: ordering invariants
   - throughput: max concurrent workers
5. Validate wait policy explicitly:
  - priority/preemption paths do not fail on scheduler timeout
  - queued waits unblock correctly when resources/events are released
6. Add at least one stress-style regression for queue contention when scheduler behavior changes.

## Task Status Display Regression Testing

For any change affecting scheduler status updates, preemption, or task ordering:

### 1. Status Transition Test

Add test to validate correct status/stage semantics during preemption:

```python
def test_task_status_transitions_during_preemption():
    """Verify queued+paused-stage distinguishes from true hardware-wait during preemption."""
    # Start ASR task on hardware
    asr_task_id = enqueue_asr_task()
    wait_for_status(asr_task_id, 'active', timeout=5)
    
    # Trigger priority detection → should preempt ASR
    priority_task_id = enqueue_priority_task()
    wait_for_status(priority_task_id, 'queued', timeout=3)
    
    # Verify ASR now shows paused stage (not stuck queued)
    status_data = get_status()
    asr_in_queue = next((t for t in status_data['tasks'] if t['task_id'] == asr_task_id), None)
    assert asr_in_queue is not None, "ASR task missing from queue"
    assert asr_in_queue['status'] == 'queued', "ASR should be queued during preemption"
    assert 'Paused for Priority Task' in asr_in_queue['stage'], f"Stage should contain paused marker, got: {asr_in_queue['stage']}"
    
    # Wait for priority completion
    wait_for_status(priority_task_id, 'completed', timeout=10)
    
    # Verify ASR resumes to active
    wait_for_status(asr_task_id, 'active', timeout=5)
    asr_resumed = get_task_status(asr_task_id)
    assert asr_resumed['status'] == 'active', "ASR should resume to active after priority completion"
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

def test_task_ordering_active_first_priority_then_standard():
    """Verify active tasks render first, then priority-queued, then standard-queued."""
    # Create mixed tasks: 2 active, 2 priority-queued, 2 standard-queued
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
    priority_indices = [task_ids.index(priority_task)]
    standard_indices = [task_ids.index(tid) for tid in standard_tasks]
    
    # Verify ordering: all active indices < all priority indices < all standard indices
    max_active = max(active_indices) if active_indices else -1
    min_priority = min(priority_indices) if priority_indices else float('inf')
    max_priority = max(priority_indices) if priority_indices else -1
    min_standard = min(standard_indices) if standard_indices else float('inf')
    
    assert max_active < min_priority, f"Active tasks should appear before priority queued"
    assert max_priority < min_standard, f"Priority queued should appear before standard queued"
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
# Backend status/ordering tests
.venv/bin/python -m pytest tests/monitoring/ tests/inference/test_scheduler.py tests/inference/priority/test_priority_concurrency.py -v -k "status or order or preemption"

# Frontend status rendering tests
npm run test:js -- tests/dashboard_main.test.js --coverage

# Full suite
.venv/bin/python -m pytest tests/
```

## Done Criteria
- New regression tests exist and pass (backend + frontend).
- Existing related tests pass.
- Full suite passes with coverage gate.
- Status transition test validates pause/resume correctness.
- Ordering determinism test confirms stable sort across calls.
- Frontend rendering test validates all 7 statuses render correctly.
- Paused-vs-waiting hint distinction validated in frontend test.