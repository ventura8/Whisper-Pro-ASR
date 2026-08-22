function calculateHistoricalSpeeds(history) {
    const agg = { sumAsrSpeed: 0, countAsr: 0, sumUvrSpeed: 0, countUvr: 0 };
    (history || []).forEach((item) => _accumulateHistoricalSpeed(agg, item));
    return {
        expectedAsrSpeed: agg.countAsr > 0 ? agg.sumAsrSpeed / agg.countAsr : 0,
        expectedUvrSpeed: agg.countUvr > 0 ? agg.sumUvrSpeed / agg.countUvr : 0
    };
}

function _accumulateHistoricalSpeed(agg, item) {
    if (normalizeStatus(item.status) !== 'completed') {
        return;
    }
    if (!(item.video_duration > 0)) {
        return;
    }
    const perf = _historyPerformance(item);
    if (!perf) {
        return;
    }
    _addPerfSpeed(agg, item.video_duration, perf.inference_sec, 'sumAsrSpeed', 'countAsr');
    _addPerfSpeed(agg, item.video_duration, perf.isolation_sec, 'sumUvrSpeed', 'countUvr');
}

function _historyPerformance(item) {
    if (item.result && item.result.performance) {
        return item.result.performance;
    }
    if (item.response_json && item.response_json.performance) {
        return item.response_json.performance;
    }
    return null;
}

function _addPerfSpeed(agg, videoDuration, perfSeconds, sumField, countField) {
    if (!(perfSeconds > 0)) {
        return;
    }
    agg[sumField] += videoDuration / perfSeconds;
    agg[countField] += 1;
}

function calculateTaskSpeedAndEta(t, now, historicalSpeeds, isUvr) {
    const taskModeIsUvr = _resolveTaskModeIsUvr(t, isUvr);
    const ctx = _buildTaskSpeedContext(t, now);
    const timeline = _prepareTaskTimeline(ctx, now);
    const reusedEstimate = _reuseTimelineEstimate(timeline, ctx, now);
    if (reusedEstimate) {
        return reusedEstimate;
    }
    const liveSpeed = _computeLiveSpeed(ctx.taskId, ctx.processedDuration, now);
    const expected = _resolveExpectedSpeeds(historicalSpeeds);
    const estimate = taskModeIsUvr
        ? _estimateUvrSpeedAndEta(t, ctx, liveSpeed, expected, timeline, now)
        : _estimateAsrSpeedAndEta(t, ctx, liveSpeed, expected, timeline, now);
    _persistTimelineEstimate(timeline, ctx, estimate, now);
    return estimate;
}

function _resolveTaskModeIsUvr(task, isUvr) {
    if (isUvr !== undefined) {
        return isUvr;
    }
    const stage = _taskFieldLower(task, 'stage');
    return task.type === 'Isolation' || _containsAnyKeyword(stage, ['vocal', 'separation', 'uvr']);
}

function _buildTaskSpeedContext(task, now) {
    const startActive = task.start_active ? task.start_active : task.start_time;
    return {
        startActive,
        elapsedActive: now - startActive,
        processedDuration: _processedTaskDuration(task),
        taskId: task.task_id ? task.task_id : task.filename,
        currentStage: task.stage ? task.stage : ''
    };
}

function _processedTaskDuration(task) {
    if (task.current_position) {
        return task.current_position;
    }
    if (task.progress && task.video_duration) {
        return (task.progress / 100) * task.video_duration;
    }
    return 0;
}

function _prepareTaskTimeline(ctx, now) {
    if (typeof activeTaskTimeline === 'undefined' || !ctx.taskId) {
        return null;
    }
    if (!activeTaskTimeline[ctx.taskId]) {
        activeTaskTimeline[ctx.taskId] = [];
    }
    const timeline = activeTaskTimeline[ctx.taskId];
    _resetTimelineOnStageChange(timeline, ctx.currentStage);
    _appendTimelinePoint(timeline, now, ctx.processedDuration);
    _trimTimeline(timeline, now);
    return timeline;
}

function _resetTimelineOnStageChange(timeline, currentStage) {
    if (timeline.lastStage !== undefined && timeline.lastStage !== currentStage) {
        timeline.length = 0;
        _clearTimelineEstimateCache(timeline);
    }
    timeline.lastStage = currentStage;
}

function _clearTimelineEstimateCache(timeline) {
    delete timeline.lastPosition;
    delete timeline.lastCalculatedSpeed;
    delete timeline.lastRemainingSeconds;
    delete timeline.lastSmoothedTimestamp;
    delete timeline.lastSmoothedUvrSpeed;
    delete timeline.lastSmoothedUvrTimestamp;
    delete timeline.lastSmoothedAsrSpeed;
    delete timeline.lastSmoothedAsrTimestamp;
}

function _appendTimelinePoint(timeline, now, processedDuration) {
    if (timeline.length === 0 || timeline[timeline.length - 1].timestamp !== now) {
        timeline.push({ timestamp: now, position: processedDuration });
    }
}

function _trimTimeline(timeline, now) {
    const cutoff = now - 60;
    while (timeline.length > 0 && timeline[0].timestamp < cutoff) {
        timeline.shift();
    }
}

// Reuse is only safe to bridge small gaps between polls where the processed
// position simply hasn't ticked yet (the task is still actively working).
// Beyond this window an unchanged position means real progress has actually
// stalled (e.g. a scheduler preemption pause), and blindly counting the ETA
// down by wall-clock elapsed time would make it shrink toward zero exactly
// while no work is happening. Past the limit, fall through to a fresh
// calculation instead, which correctly factors the stall into a lower speed
// and a growing remaining-time estimate.
const _TIMELINE_REUSE_STALL_LIMIT_SEC = 5;

function _reuseTimelineEstimate(timeline, ctx, now) {
    if (!_canReuseTimelineEstimate(timeline, ctx, now)) {
        return null;
    }
    const calculatedSpeed = timeline.lastCalculatedSpeed || 0;
    const elapsedSinceSample = now - (timeline.lastSmoothedTimestamp || now);
    const remainingSeconds = Math.max(0, (timeline.lastRemainingSeconds || 0) - elapsedSinceSample);
    return { calculatedSpeed, remainingSeconds };
}

function _canReuseTimelineEstimate(timeline, ctx, now) {
    if (!_isSameTimelinePositionAndStage(timeline, ctx)) {
        return false;
    }
    const sinceLastSample = now - (timeline.lastSmoothedTimestamp || now);
    return sinceLastSample >= 0 && sinceLastSample <= _TIMELINE_REUSE_STALL_LIMIT_SEC;
}

function _isSameTimelinePositionAndStage(timeline, ctx) {
    // NOTE: call-order dependency -- this function is only called from
    // _reuseTimelineEstimate, which is called AFTER _prepareTaskTimeline in
    // calculateTaskSpeedAndEta. _prepareTaskTimeline calls _resetTimelineOnStageChange,
    // which updates timeline.lastStage to ctx.currentStage (and clears the cache on a
    // stage change). The `timeline.lastStage === ctx.currentStage` check below is
    // therefore guaranteed to be trivially true for any same-stage call, but it serves
    // as a defensive guard: reordering _prepareTaskTimeline and _reuseTimelineEstimate
    // in calculateTaskSpeedAndEta would break that guarantee and allow stale cached
    // estimates from a prior stage to be reused incorrectly.
    if (!timeline) {
        return false;
    }
    if (timeline.lastPosition === undefined || timeline.lastPosition !== ctx.processedDuration) {
        return false;
    }
    return timeline.lastStage === ctx.currentStage;
}

function _computeLiveSpeed(taskId, processedDuration, now) {
    if (typeof activeTaskTimeline === 'undefined' || !taskId) {
        return 0;
    }
    const timeline = activeTaskTimeline[taskId] || [];
    if (timeline.length < 2) {
        return 0;
    }
    const referencePoint = _timelineReferencePoint(timeline, now - 15);
    return _deltaSpeed(processedDuration, referencePoint.position, now, referencePoint.timestamp);
}

function _timelineReferencePoint(timeline, targetTime) {
    for (let i = timeline.length - 1; i >= 0; i--) {
        if (timeline[i].timestamp <= targetTime) {
            return timeline[i];
        }
    }
    return timeline[0];
}

function _deltaSpeed(currentPosition, previousPosition, currentTime, previousTime) {
    const deltaPos = currentPosition - previousPosition;
    const deltaTime = currentTime - previousTime;
    if (deltaTime <= 0 || deltaPos < 0) {
        return 0;
    }
    return deltaPos / deltaTime;
}

function _resolveExpectedSpeeds(historicalSpeeds) {
    if (!historicalSpeeds) {
        return { expectedAsrSpeed: 0, expectedUvrSpeed: 0 };
    }
    if (Array.isArray(historicalSpeeds)) {
        return calculateHistoricalSpeeds(historicalSpeeds);
    }
    return {
        expectedAsrSpeed: historicalSpeeds.expectedAsrSpeed || 0,
        expectedUvrSpeed: historicalSpeeds.expectedUvrSpeed || 0
    };
}

function _estimateUvrSpeedAndEta(task, ctx, liveSpeed, expected, timeline, now) {
    const baseSpeed = _pickBaseSpeed(liveSpeed, ctx.elapsedActive, ctx.processedDuration, expected.expectedUvrSpeed || 2.0);
    const uvrSpeed = _smoothSpeed(timeline, now, baseSpeed, 'lastSmoothedUvrSpeed', 'lastSmoothedUvrTimestamp');
    const remainingUvrSec = Math.max(0, (task.video_duration - ctx.processedDuration) / uvrSpeed);
    const expectedAsrSec = _expectedAsrSeconds(task.video_duration, expected.expectedAsrSpeed);
    const remainingSeconds = remainingUvrSec + expectedAsrSec;
    const totalEstimatedSec = ctx.elapsedActive + remainingSeconds;
    const calculatedSpeed = totalEstimatedSec > 0 ? (task.video_duration / totalEstimatedSec) : 0;
    return { calculatedSpeed, remainingSeconds };
}

function _estimateAsrSpeedAndEta(task, ctx, liveSpeed, expected, timeline, now) {
    const startInference = task.start_inference ? task.start_inference : ctx.startActive;
    const elapsedAsr = now - startInference;
    const uvrElapsed = task.start_inference ? (task.start_inference - ctx.startActive) : 0;
    const baseSpeed = _pickBaseSpeed(liveSpeed, elapsedAsr, ctx.processedDuration, expected.expectedAsrSpeed || 5.0);
    const asrSpeed = _smoothSpeed(timeline, now, baseSpeed, 'lastSmoothedAsrSpeed', 'lastSmoothedAsrTimestamp');
    const remainingSeconds = Math.max(0, (task.video_duration - ctx.processedDuration) / asrSpeed);
    const totalEstimatedSec = uvrElapsed + elapsedAsr + remainingSeconds;
    const calculatedSpeed = totalEstimatedSec > 0 ? (task.video_duration / totalEstimatedSec) : 0;
    return { calculatedSpeed, remainingSeconds };
}

function _pickBaseSpeed(liveSpeed, elapsedSeconds, processedDuration, fallbackSpeed) {
    if (elapsedSeconds > 5 && liveSpeed > 0) {
        return liveSpeed;
    }
    if (elapsedSeconds > 0 && processedDuration > 0) {
        return processedDuration / elapsedSeconds;
    }
    return fallbackSpeed;
}

function _smoothSpeed(timeline, now, currentSpeed, speedField, timeField) {
    if (!timeline) {
        return currentSpeed;
    }
    let smoothed = currentSpeed;
    if (timeline[speedField] !== undefined && timeline[timeField] !== undefined) {
        const dt = Math.max(0.1, now - timeline[timeField]);
        const alpha = 1 - Math.exp(-dt / 12.3);
        smoothed = alpha * currentSpeed + (1 - alpha) * timeline[speedField];
    }
    timeline[speedField] = smoothed;
    timeline[timeField] = now;
    return smoothed;
}

function _expectedAsrSeconds(videoDuration, expectedAsrSpeed) {
    if (expectedAsrSpeed > 0) {
        return videoDuration / expectedAsrSpeed;
    }
    return videoDuration / 5.0;
}

function _persistTimelineEstimate(timeline, ctx, estimate, now) {
    if (!timeline) {
        return;
    }
    timeline.lastCalculatedSpeed = estimate.calculatedSpeed;
    timeline.lastRemainingSeconds = estimate.remainingSeconds;
    timeline.lastPosition = ctx.processedDuration;
    timeline.lastStage = ctx.currentStage;
    timeline.lastSmoothedTimestamp = now;
}

function getDefaultStageFromStatus(status) {
    const key = normalizeStatus(status);
    const defaultStages = {
        active: 'Active',
        queued: 'Queued',
        'post-processing': 'Post-Processing',
        completed: 'Completed',
        failed: 'Failed',
        initializing: 'Initializing'
    };
    return defaultStages[key] || 'Initializing';
}

const DISPLAYABLE_STATUSES = new Set([
    'initializing',
    'queued',
    'active',
    'post-processing',
    'completed',
    'failed',
]);

function normalizeStatus(status) {
    const key = String(status || '').trim().toLowerCase();
    return DISPLAYABLE_STATUSES.has(key) ? key : 'initializing';
}

function isPlaceholderStage(value) {
    const normalized = String(value || '').trim().toLowerCase();
    return _placeholderChecks(normalized).some((checkFn) => checkFn());
}

function _placeholderChecks(normalized) {
    return [
        () => !normalized,
        () => _isNullLikePlaceholder(normalized),
        () => _isZeroRatioPlaceholder(normalized),
        () => normalized.includes('placeholder'),
        () => _isResumePlaceholder(normalized)
    ];
}

function _isNullLikePlaceholder(normalized) {
    return ['none', 'null', 'undefined', 'unknown', 'na', 'n/a'].includes(normalized);
}

function _isZeroRatioPlaceholder(normalized) {
    return normalized.replace(/[()\s]/g, '') === '0/0';
}

function _isResumePlaceholder(normalized) {
    return normalized === 'resume' || normalized === 'resuming';
}

function normalizeStage(task) {
    const str = _taskStageText(task);
    if (!isPlaceholderStage(str)) {
        return str;
    }
    return getDefaultStageFromStatus(task ? task.status : '');
}

function _taskStageText(task) {
    const raw = task ? task.stage : '';
    if (raw === null || raw === undefined) {
        return '';
    }
    return String(raw).trim();
}