const path = require("path");
const { loadScriptInContext } = require("./helpers");

// The speed multiplier and ETA math for the industrial telemetry cards
// (Section 7 of docs/E2E_TEST_PLAN_ORCHESTRATION.md) is computed entirely
// client-side in speed_status.js -- there is no Python equivalent to unit
// test. These tests load the real script into a sandboxed VM context and
// drive its calculation functions directly against known fixtures.
describe("speed_status.js", () => {
  let context;

  beforeEach(() => {
    context = loadScriptInContext(
      path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/speed_status.js"),
      {
        // No activeTaskTimeline provided for the base fixtures: this forces
        // every calculation through the stateless "processedDuration / elapsedActive"
        // path (no exponential smoothing, no timeline-based live speed), which is
        // exactly the audio_duration / processing_time formula under test.
      }
    );
  });

  describe("gap 7.1: speed multiplier matches audio_duration / processing_time", () => {
    it("computes the calculated speed from processed duration over elapsed active time (steady rate, full run)", () => {
      // Fixture: 100s of audio, processed steadily at 2x, elapsed 50s total ASR time.
      const task = {
        video_duration: 100,
        start_active: 1000,
        start_inference: 1000,
        current_position: 100, // fully processed
        stage: "Inference",
      };
      const now = 1050; // 50s elapsed

      const estimate = context.calculateTaskSpeedAndEta(task, now, null, false);

      // 100s of audio processed in 50s of wall time => speed multiplier is exactly 2.0x,
      // matching audio_duration / processing_time for this known fixture.
      expect(estimate.calculatedSpeed).toBeCloseTo(2.0, 5);
    });

    it("computes the calculated speed correctly mid-run (partial computation, task not yet finished)", () => {
      // Fixture: 100s of audio, only 40s processed so far, 20s of ASR elapsed.
      const task = {
        video_duration: 100,
        start_active: 1000,
        start_inference: 1000,
        current_position: 40,
        stage: "Inference",
      };
      const now = 1020; // 20s elapsed

      const estimate = context.calculateTaskSpeedAndEta(task, now, null, false);

      // Instantaneous throughput: 40s processed / 20s elapsed = 2.0x.
      // Because throughput is steady, the full-run projected speed (video_duration /
      // (elapsed + projected remaining)) converges to the same 2.0x for this fixture:
      //   remaining = (100 - 40) / 2.0 = 30s -> total = 20 + 30 = 50s -> 100 / 50 = 2.0x
      expect(estimate.calculatedSpeed).toBeCloseTo(2.0, 5);
      // Partial computation must be strictly positive and bounded (not zero, not runaway).
      expect(estimate.calculatedSpeed).toBeGreaterThan(0);
    });

    it("falls back to the historical expected speed when there is no progress signal yet", () => {
      const task = {
        video_duration: 100,
        start_active: 1000,
        start_inference: 1000,
        current_position: 0,
        stage: "Inference",
      };
      const now = 1000; // no elapsed time yet -> no progress signal

      const estimate = context.calculateTaskSpeedAndEta(task, now, { expectedAsrSpeed: 4.0, expectedUvrSpeed: 2.0 }, false);

      // With zero elapsed time and zero processed duration, _pickBaseSpeed falls back
      // to the historical expected ASR speed fixture value (4.0x) rather than 0 or NaN.
      expect(estimate.calculatedSpeed).toBeCloseTo(4.0, 5);
    });

    it("derives expected historical speeds from completed-task history as (video_duration / inference_sec) averages", () => {
      const history = [
        {
          status: "completed",
          video_duration: 100,
          result: { performance: { inference_sec: 25 } }, // 4.0x
        },
        {
          status: "completed",
          video_duration: 60,
          result: { performance: { inference_sec: 30 } }, // 2.0x
        },
        // Non-completed entries must be excluded from the average.
        {
          status: "active",
          video_duration: 100,
          result: { performance: { inference_sec: 1 } },
        },
      ];

      const expected = context.calculateHistoricalSpeeds(history);

      expect(expected.expectedAsrSpeed).toBeCloseTo((4.0 + 2.0) / 2, 5);
    });
  });

  describe("gap 7.2: ETA recalculates as speed fluctuates, including preemption pauses", () => {
    it("computes remaining time consistent with (video_duration - processed) / speed for a known fixture", () => {
      const task = {
        video_duration: 100,
        start_active: 1000,
        start_inference: 1000,
        current_position: 40,
        stage: "Inference",
      };
      const now = 1020;

      const estimate = context.calculateTaskSpeedAndEta(task, now, null, false);

      // remaining = (100 - 40) / 2.0 = 30s, per the same fixture as the 7.1 partial test.
      expect(estimate.remainingSeconds).toBeCloseTo(30, 5);
      expect(estimate.remainingSeconds).toBeGreaterThanOrEqual(0);
    });

    it("never reports a negative ETA even when processed duration already exceeds video_duration (clock skew / rounding)", () => {
      const task = {
        video_duration: 100,
        start_active: 1000,
        start_inference: 1000,
        current_position: 120, // overshoot
        stage: "Inference",
      };
      const now = 1010;

      const estimate = context.calculateTaskSpeedAndEta(task, now, null, false);

      expect(estimate.remainingSeconds).toBe(0);
    });

    it("extends the ETA (does not freeze or shrink) once a preemption pause slows real progress, using a live timeline", () => {
      // Use a live timeline so the engine's smoothing/live-speed machinery is exercised,
      // matching how the real dashboard polls calculateTaskSpeedAndEta repeatedly.
      const timelineContext = loadScriptInContext(
        path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/speed_status.js"),
        { activeTaskTimeline: {} }
      );

      const taskId = "preempt-task-1";
      const videoDuration = 100;
      const startActive = 1000;

      // Phase 1: task runs at a steady 2x for 20s (0 -> 40s processed).
      const runningTask = {
        task_id: taskId,
        video_duration: videoDuration,
        start_active: startActive,
        start_inference: startActive,
        current_position: 40,
        stage: "Inference",
      };
      const beforePause = timelineContext.calculateTaskSpeedAndEta(runningTask, startActive + 20, null, false);
      expect(beforePause.remainingSeconds).toBeCloseTo(30, 5); // (100-40)/2.0

      // Phase 2: scheduler preempts the task for a priority job. Real progress halts
      // (current_position unchanged) while wall-clock time keeps advancing, and the
      // scheduler-reported stage changes to reflect the pause (as concurrency.py's
      // preemption flow does via scheduler.update_task_progress/"Paused for Priority Task").
      const pausedTask = {
        ...runningTask,
        stage: "Paused for Priority Task",
      };
      // Sample repeatedly through a 25s pause window, as the dashboard poll loop would.
      const samplesDuringPause = [10, 15, 20, 25].map((pauseElapsed) =>
        timelineContext.calculateTaskSpeedAndEta(pausedTask, startActive + 20 + pauseElapsed, null, false)
      );

      for (const sample of samplesDuringPause) {
        // ETA must never go negative during a pause.
        expect(sample.remainingSeconds).toBeGreaterThanOrEqual(0);
      }

      // Because no real progress was made while wall-clock time advanced, the projected
      // remaining time must visibly extend relative to the pre-pause estimate -- it must
      // not freeze at the pre-pause value and must not shrink all the way back down as if
      // work were continuing throughout the whole pause.
      const lastSample = samplesDuringPause[samplesDuringPause.length - 1];
      expect(lastSample.remainingSeconds).toBeGreaterThan(beforePause.remainingSeconds);

      // The ETA must never collapse back to (or below) the pre-pause estimate at any point
      // during the pause -- a brief poll-to-poll dip from interpolation is tolerated, but it
      // must never erase the extension entirely.
      for (const sample of samplesDuringPause) {
        expect(sample.remainingSeconds).toBeGreaterThan(beforePause.remainingSeconds * 0.9);
      }
    });

    it("rejects cache reuse when position advances, even within the 5-second stall window", () => {
      // _isSameTimelinePositionAndStage checks timeline.lastPosition !== ctx.processedDuration.
      // Even if only 2 seconds have elapsed (well within the 5-second stall limit),
      // advancing current_position must cause _canReuseTimelineEstimate to return false,
      // so the engine performs a fresh calculation with an updated ETA reflecting real progress.
      const timelineCtx = loadScriptInContext(
        path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/speed_status.js"),
        { activeTaskTimeline: {} }
      );

      const task = {
        task_id: "position-advance-test",
        video_duration: 100,
        start_active: 1000,
        start_inference: 1000,
        current_position: 30,
        stage: "Inference",
      };

      // First call: prime the timeline at t=1020, position=30 -> 10s elapsed, 30s processed.
      const firstNow = 1020;
      const first = timelineCtx.calculateTaskSpeedAndEta(task, firstNow, null, false);
      expect(first.calculatedSpeed).toBeGreaterThan(0);
      expect(first.remainingSeconds).toBeGreaterThanOrEqual(0);

      // Verify the timeline was primed with position=30.
      const timeline = timelineCtx.activeTaskTimeline[task.task_id];
      expect(timeline).toBeDefined();
      expect(timeline.lastPosition).toBe(30);

      // Second call: only 2s later (within 5-second reuse window), but position has ADVANCED to 40.
      const secondNow = firstNow + 2; // sinceLastSample = 2s <= 5s limit
      const advancedTask = { ...task, current_position: 40 };

      // Directly verify _canReuseTimelineEstimate returns false because processedDuration differs
      // from timeline.lastPosition (30 vs 40), regardless of sinceLastSample being within the limit.
      const advancedCtx = { processedDuration: 40, currentStage: task.stage };
      expect(timelineCtx._canReuseTimelineEstimate(timeline, advancedCtx, secondNow)).toBe(false);

      // calculateTaskSpeedAndEta must perform a fresh calculation (not reuse the cached estimate).
      const fresh = timelineCtx.calculateTaskSpeedAndEta(advancedTask, secondNow, null, false);

      // A fresh calculation must produce a valid, non-negative result.
      expect(fresh).toBeDefined();
      expect(fresh.calculatedSpeed).toBeGreaterThan(0);
      expect(fresh.remainingSeconds).toBeGreaterThanOrEqual(0);

      // The timeline's lastPosition must now reflect the new position (40), not the cached 30.
      const updatedTimeline = timelineCtx.activeTaskTimeline[task.task_id];
      expect(updatedTimeline.lastPosition).toBe(40);
    });

    it("reuses the cached timeline estimate when sinceLastSample is under the 5-second stall limit", () => {
      // The 5-second stall limit (_TIMELINE_REUSE_STALL_LIMIT_SEC) prevents the ETA from
      // shrinking toward zero while the task position hasn't moved (e.g. scheduler preemption).
      // Below the limit, the engine should reuse the last stored estimate rather than
      // recalculating, so position/stage/speed are unchanged between the two calls.
      const timelineCtx = loadScriptInContext(
        path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/speed_status.js"),
        { activeTaskTimeline: {} }
      );

      const task = {
        task_id: "reuse-test",
        video_duration: 100,
        start_active: 1000,
        start_inference: 1000,
        current_position: 50,
        stage: "Inference",
      };

      // First call: prime the timeline with a known speed/ETA estimate.
      const firstNow = 1025; // 25s elapsed, 50s processed -> 2x speed, 25s remaining
      const first = timelineCtx.calculateTaskSpeedAndEta(task, firstNow, null, false);
      expect(first.calculatedSpeed).toBeGreaterThan(0);
      expect(first.remainingSeconds).toBeGreaterThanOrEqual(0);

      // Second call: 4 seconds later, same position and stage -> under the 5s limit -> reuse.
      const secondNow = firstNow + 4; // sinceLastSample = 4s <= 5s limit
      const reused = timelineCtx.calculateTaskSpeedAndEta(task, secondNow, null, false);

      // Speed must be the same cached value (no recalculation).
      expect(reused.calculatedSpeed).toBeCloseTo(first.calculatedSpeed, 5);
      // Remaining time should tick down by ~4s (elapsed since last sample), not stay frozen.
      expect(reused.remainingSeconds).toBeCloseTo(
        Math.max(0, first.remainingSeconds - 4),
        0 // precision 0 -> 0.5s tolerance for floating-point rounding in the reuse path
      );
    });

    it("reuses the cached timeline estimate at the exact 5-second stall limit (inclusive boundary)", () => {
      // _canReuseTimelineEstimate compares with `sinceLastSample <= _TIMELINE_REUSE_STALL_LIMIT_SEC`,
      // an inclusive comparison -- exactly 5s must still reuse the cached estimate, not
      // trigger a fresh calculation. The existing tests cover 4s (reuse) and 6s (fresh);
      // this closes the untested exact-boundary case.
      const timelineCtx = loadScriptInContext(
        path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/speed_status.js"),
        { activeTaskTimeline: {} }
      );

      const task = {
        task_id: "boundary-test",
        video_duration: 100,
        start_active: 1000,
        start_inference: 1000,
        current_position: 50,
        stage: "Inference",
      };

      const firstNow = 1025;
      const first = timelineCtx.calculateTaskSpeedAndEta(task, firstNow, null, false);
      expect(first.calculatedSpeed).toBeGreaterThan(0);

      const secondNow = firstNow + 5; // sinceLastSample = 5s == limit -> still reuse
      const reused = timelineCtx.calculateTaskSpeedAndEta(task, secondNow, null, false);

      expect(reused.calculatedSpeed).toBeCloseTo(first.calculatedSpeed, 5);
      expect(reused.remainingSeconds).toBeCloseTo(
        Math.max(0, first.remainingSeconds - 5),
        0 // precision 0 -> 0.5s tolerance for floating-point rounding in the reuse path
      );
    });

    it("rejects a future-dated cached sample (now earlier than the last recorded timestamp) instead of reusing it", () => {
      // Guards _canReuseTimelineEstimate's sinceLastSample >= 0 check: if the timeline's
      // lastSmoothedTimestamp is somehow ahead of `now` (clock skew), sinceLastSample goes
      // negative and would otherwise satisfy "<= 5s" trivially, incorrectly reusing a
      // stale/future estimate. This must force a fresh calculation instead.
      const timelineCtx = loadScriptInContext(
        path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/speed_status.js"),
        { activeTaskTimeline: {} }
      );

      const task = {
        task_id: "clock-skew-test",
        video_duration: 100,
        start_active: 1000,
        start_inference: 1000,
        current_position: 50,
        stage: "Inference",
      };

      // First call primes the timeline at t=1025.
      const firstNow = 1025;
      const first = timelineCtx.calculateTaskSpeedAndEta(task, firstNow, null, false);
      expect(first.calculatedSpeed).toBeGreaterThan(0);

      // Second call's `now` is BEFORE the primed timestamp (simulated clock skew) --
      // sinceLastSample would be negative, which must NOT be treated as "within the stall limit".
      const skewedNow = firstNow - 10;

      // Assert the reuse gate itself rejects this sample directly, rather than only
      // inferring it from downstream values: a wrongly-reused cached estimate would
      // also satisfy "remainingSeconds/calculatedSpeed >= 0" (the primed values were
      // themselves non-negative), so those checks alone can't distinguish a fresh
      // calculation from an incorrect cache reuse.
      const timeline = timelineCtx.activeTaskTimeline[task.task_id];
      const skewCtx = { processedDuration: task.current_position, currentStage: task.stage };
      expect(timelineCtx._canReuseTimelineEstimate(timeline, skewCtx, skewedNow)).toBe(false);

      const fresh = timelineCtx.calculateTaskSpeedAndEta(task, skewedNow, null, false);

      // A fresh calculation must still produce a sane, non-negative result rather than
      // reusing (and ticking down from) the cached estimate using a negative elapsed time.
      expect(fresh.remainingSeconds).toBeGreaterThanOrEqual(0);
      expect(fresh.calculatedSpeed).toBeGreaterThanOrEqual(0);
    });

    it("triggers a fresh calculation when sinceLastSample exceeds the 5-second stall limit", () => {
      // Past the 5-second stall limit, the engine performs a fresh calculation rather than
      // reusing the cached estimate. This correctly factors a prolonged position stall (e.g.
      // a preemption pause) into a lower speed and a growing remaining-time estimate rather
      // than letting the ETA count down as if work continued.
      const timelineCtx = loadScriptInContext(
        path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/speed_status.js"),
        { activeTaskTimeline: {} }
      );

      const task = {
        task_id: "fresh-calc-test",
        video_duration: 100,
        start_active: 1000,
        start_inference: 1000,
        current_position: 50,
        stage: "Inference",
      };

      // First call: prime the timeline at t=1025 -> speed=2x, remaining~25s.
      const firstNow = 1025;
      const first = timelineCtx.calculateTaskSpeedAndEta(task, firstNow, null, false);
      expect(first.calculatedSpeed).toBeGreaterThan(0);

      // Second call: 6 seconds later, same position and stage -> over the 5s limit -> fresh calc.
      const secondNow = firstNow + 6; // sinceLastSample = 6s > 5s limit
      const fresh = timelineCtx.calculateTaskSpeedAndEta(task, secondNow, null, false);

      // A fresh calculation must produce a result (not null/undefined).
      expect(fresh).toBeDefined();
      expect(fresh.calculatedSpeed).toBeGreaterThan(0);
      expect(fresh.remainingSeconds).toBeGreaterThanOrEqual(0);

      // The fresh-calculated speed will be lower than the primed estimate because
      // wall-clock time advanced (31s elapsed) while position stayed at 50s -> ~1.6x,
      // demonstrably different from the primed 2x value.
      // We only assert it differs from the cached speed by more than float rounding
      // (i.e. a real recalculation happened, not a reuse).
      expect(Math.abs(fresh.calculatedSpeed - first.calculatedSpeed)).toBeGreaterThan(0.01);
      // A lower recalculated speed against the same remaining audio must also project a
      // longer remaining time -- i.e. the ETA grows rather than freezing/shrinking, the
      // same "stall must extend the ETA" guarantee the gap 7.2 pause test asserts directly.
      expect(fresh.remainingSeconds).toBeGreaterThan(first.remainingSeconds);
    });
  });
});
