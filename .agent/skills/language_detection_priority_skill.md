# Language Detection Priority Skill

Use this skill for scheduler/preemption changes affecting `/detect-language` (`/detectlang` alias) vs `/asr` (including `/v1/audio/...`) behavior.

## Objective
Preserve the intended policy:
1. Detect-language (`/detect-language` and `/detectlang`) is high priority and can preempt ASR.
2. FIFO is preserved within each priority tier where applicable.
3. Paused ASR does not resume while queued priority work still exists.
4. Priority execution can use available parallel unit capacity.
5. Detect-language tasks are not globally serialized for their full lifetime; only hardware acquisition order remains FIFO-aware.

## Behavioral Scenarios to Validate
1. 1 ASR + 1 detect-language: ASR yields, detect completes, ASR resumes.
2. 2 ASR on 2 units + multiple detect-language: detect uses both units in parallel when available.
3. 3 busy accelerators (NPU + GPU + CUDA) + multiple detect-language: at least two detect tasks may overlap on borrowed units while FIFO is preserved for queued same-priority tasks.
4. Mixed bursts (ASR, many detect, ASR): no stuck waiting-hardware states.
5. Priority backlog drain: resume only after queued priority tasks are exhausted.
6. ASR stage checkpoints: ASR invokes cooperative yield before vocal separation and again immediately before inference starts.

## Validation Commands
```bash
.venv/bin/python -m pytest tests/inference/priority/test_priority_concurrency.py tests/inference/priority/test_priority_fifo_ordering.py -q
.venv/bin/python -m pytest tests/inference/test_scheduler.py -q
```

## Done Criteria
- Priority tests pass for liveness, fairness, and throughput.
- No deadlocks/timeouts under bursty mixed workloads.
- Multi-accelerator detect-language bursts prove the scheduler does not collapse to one priority task at a time.
- Stage-aware preemption tests confirm pre-vocal and pre-inference ASR yield checkpoints.