function _renderActiveTaskList(data, now, historicalSpeeds) {
    const tList = document.getElementById('task-list');
    const tasks = _filteredActiveTasks(data.tasks || []);
    if (tasks.length === 0) {
        tList.innerHTML = `<div class="empty-state"><span class="material-icons-sharp empty-icon">auto_awesome</span><div><strong>Service is idle</strong></div></div>`;
        return;
    }
    const ordered = tasks.slice().sort(_compareTaskOrder);
    tList.innerHTML = ordered.map((task, index) => _renderActiveTaskCard(task, now, historicalSpeeds, _makeOpaqueDomKey(index))).join('');
}

function _makeOpaqueDomKey(index) {
    return `task-${index}`;
}

function _filteredActiveTasks(tasks) {
    return tasks.filter((task) => matchesCategoryFilter(task, globalThis.activeTaskFilter));
}

function _compareTaskOrder(a, b) {
    const keyA = _taskOrderKey(a);
    const keyB = _taskOrderKey(b);
    if (keyA[0] !== keyB[0]) return keyA[0] - keyB[0];
    if (keyA[1] !== keyB[1]) return keyA[1] - keyB[1];
    return keyA[2].localeCompare(keyB[2]);
}

function _taskOrderKey(task) {
    const status = normalizeStatus(task.status);
    const startTime = Number(task.start_time || 0);
    const taskId = String(task.task_id || '');
    const tier = _taskOrderTier(status);
    return [tier, startTime, taskId];
}

function _taskOrderTier(status) {
    if (status === 'active') {
        return 0;
    }
    return 1;
}

function _renderActiveTaskCard(task, now, historicalSpeeds, domKey) {
    const vm = _activeTaskViewModel(task, now, historicalSpeeds, domKey);
    const safeTaskId = escapeHtml(vm.id);
    return `
        <div class="task-card ${vm.normalizedStatus === 'queued' ? 'queued' : ''}" data-task-id="${safeTaskId}">
            <div class="task-header">
                <div class="task-icon-container" style="background:var(--md-sys-color-primary-container); color:var(--md-sys-color-primary);">
                    <span class="material-icons-sharp">${vm.typeIcon}</span>
                </div>
                <div class="item-info">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                        <span class="item-primary">${vm.safeFilename}</span>
                        <span class="meta-tag" style="background:var(--md-sys-color-secondary-container); color:var(--md-sys-color-on-secondary-container); border:none; padding:1px 8px; border-radius:100px; font-weight:700;">
                            <span class="material-icons-sharp ${vm.statusPulseClass}" style="font-size:13px">${vm.statusIcon}</span> ${vm.taskTypeLabel}
                        </span>
                    </div>
                    <div class="item-secondary">
                        <span class="meta-tag" style="color:var(--md-sys-color-primary); font-weight:600;"><span class="material-icons-sharp" style="font-size:12px">layers</span><span class="stage-text">${vm.stageText}</span></span>
                        <span class="meta-tag"><span class="material-icons-sharp" style="font-size:12px">movie</span>${formatDur(task.video_duration)}</span>
                        <span class="meta-tag"><span class="material-icons-sharp" style="font-size:12px">timer</span><span class="timer-text">${vm.timerText}</span></span>
                        <span class="meta-tag speed-tag" style="display:${vm.speedDisplay};"><span class="material-icons-sharp" style="font-size:12px">speed</span>Speed: <span class="speed-text">${vm.speedText}</span></span>
                        <span class="meta-tag eta-tag" style="display:${vm.etaDisplay};"><span class="material-icons-sharp" style="font-size:12px">schedule</span>ETA: <span class="eta-text">${vm.etaText}</span></span>
                        <span class="meta-tag hw-tag"><span class="material-icons-sharp hw-icon" style="font-size:12px;color:var(--md-sys-color-primary)">${vm.hw.icon}</span><span class="hw-text">${vm.hw.label}</span></span>
                    </div>
                </div>
                <span class="badge badge-${vm.normalizedStatus}">${vm.statusBadgeLabel}</span>
            </div>
            <div style="display:flex;flex-direction:column;gap:4px;">
                <div class="progress-container">
                    <div class="progress-bar ${vm.progressPulseClass}" style="width:${vm.progressPct}%"></div>
                </div>
                <div style="font-size:10px; color:var(--md-sys-color-secondary); text-align:right; font-weight:600;"><span class="progress-text">${vm.progressPct}</span>%</div>
            </div>
            ${renderAuditDetails(task)}
            ${vm.liveSectionHtml}
            <details class="js-toggle" data-toggle-id="${domKey}_logs" ${expandedElements.has(`${domKey}_logs`) ? 'open' : ''}>
                <summary><span class="material-icons-sharp">terminal</span> Real-time Logs</summary>
                <div class="log-buffer">${escapeHtml(vm.logContent)}</div>
            </details>
            <div class="hw-wait-msg" style="font-size:11px; color:var(--md-sys-color-warning); font-style:italic; margin-top:8px; display:${vm.queueHintDisplay}; align-items:center; gap:4px;">
                <span class="material-icons-sharp" style="font-size:14px">hourglass_empty</span> ${vm.queueHint}
            </div>
        </div>
    `;
}

function _activeTaskViewModel(task, now, historicalSpeeds, domKey) {
    const normalizedStatus = normalizeStatus(task.status);
    const stageText = normalizeStage({ ...task, status: normalizedStatus });
    const identity = _activeTaskIdentity(task, normalizedStatus, stageText, now, domKey);
    const speedEta = _activeTaskSpeedEta(task, now, historicalSpeeds, stageText);
    const visuals = _activeTaskVisuals(task, normalizedStatus);
    return {
        ...identity,
        ...visuals,
        speedDisplay: speedEta.showSpeed ? 'inline-flex' : 'none',
        speedText: speedEta.speedText,
        etaDisplay: speedEta.showEta ? 'inline-flex' : 'none',
        etaText: speedEta.etaText
    };
}

function _activeTaskQueueHintDisplay(normalizedStatus) {
    return normalizedStatus === 'queued' ? 'flex' : 'none';
}

function _activeTaskIdentity(task, normalizedStatus, stageText, now, domKey) {
    const id = task.task_id || task.filename;
    const isAsr = isAsrLikeCategory(getTaskFilterCategory(task));
    return {
        id,
        safeFilename: escapeHtml(task.filename || 'Unknown Media'),
        normalizedStatus,
        stageText,
        progressPct: task.progress || 0,
        timerText: getTimerText(task, now),
        hw: getHwIconAndLabel(task.unit_id),
        queueHint: _queueHint(normalizedStatus, stageText),
        queueHintDisplay: _activeTaskQueueHintDisplay(normalizedStatus),
        liveSectionHtml: _activeTaskLiveSection(isAsr, domKey, id, task.live_text),
        logContent: (task.logs || []).join('\n')
    };
}

function _activeTaskVisuals(task, normalizedStatus) {
    const statusVisuals = _activeTaskStatusVisuals(normalizedStatus);
    return {
        ...statusVisuals,
        taskTypeLabel: escapeHtml(_taskTypeLabel(task)),
        typeIcon: _taskTypeIcon(task)
    };
}

function _activeTaskStatusVisuals(normalizedStatus) {
    const isActive = normalizedStatus === 'active';
    return {
        statusIcon: _statusIcon(normalizedStatus),
        statusPulseClass: isActive ? 'pulse' : '',
        statusBadgeLabel: normalizedStatus === 'queued' ? 'queue' : normalizedStatus,
        progressPulseClass: isActive ? 'pulse' : ''
    };
}

function _taskTypeLabel(task) {
    return task.type || 'Task';
}

function _taskTypeIcon(task) {
    return isAsrLikeCategory(getTaskFilterCategory(task)) ? 'record_voice_over' : 'translate';
}

function _statusIcon(status) {
    if (status === 'active') return 'sync';
    if (status === 'initializing') return 'hourglass_top';
    return 'hourglass_empty';
}

function _queueHint(status, stageText) {
    const pausedForPriority = status === 'queued' && String(stageText || '').toLowerCase().includes('paused for priority task');
    return pausedForPriority ? 'Paused for priority detect-language tasks...' : 'Waiting for available hardware unit...';
}

function _activeTaskLiveSection(isAsr, domKey, id, liveText) {
    if (!isAsr) {
        return '';
    }
    const openAttr = expandedElements.has(`${domKey}_live`) ? 'open' : '';
    const safeTaskId = escapeHtml(id);
    const text = liveText || 'Waiting for first transcription segment...';
    return `
                            <div style="margin-top:8px;">
                                <details class="js-toggle" data-toggle-id="${domKey}_live" ${openAttr}>
                                    <summary style="font-size:11px; font-weight:700; color:var(--md-sys-color-primary); display:flex; align-items:center; gap:4px;">
                                        <span class="material-icons-sharp pulse" style="font-size:14px; color:var(--md-sys-color-error)">radio_button_checked</span> LIVE SRT STREAM
                                    </summary>
                                    <div class="result-box live-text-box" data-task-id="${safeTaskId}" style="margin-top:4px; border: 1px dashed var(--md-sys-color-primary);">${escapeHtml(text)}</div>
                                </details>
                            </div>`;
}

function _activeTaskSpeedEta(task, now, historicalSpeeds, stageText) {
    const elapsedActive = _uiElapsedActiveSeconds(task, now);
    if (!(task.video_duration > 0)) {
        return { showSpeed: false, speedText: '0.0x', showEta: false, etaText: '00:00:00' };
    }
    const estimate = _taskSpeedEtaEstimate(task, now, historicalSpeeds, stageText);
    return _taskSpeedEtaView(elapsedActive, estimate);
}

function _uiElapsedActiveSeconds(task, now) {
    const startActive = _taskUiStartActive(task, now);
    if (!_areComparableTimestamps(startActive, now)) {
        return 0;
    }
    return Math.max(0, now - startActive);
}

function _taskUiStartActive(task, fallbackNow) {
    if (task.start_active !== undefined && task.start_active !== null) {
        return task.start_active;
    }
    if (task.start_time !== undefined && task.start_time !== null) {
        return task.start_time;
    }
    return fallbackNow;
}

function _areComparableTimestamps(a, b) {
    const epochThreshold = 100000000;
    const bothEpoch = a >= epochThreshold && b >= epochThreshold;
    const bothRelative = a < epochThreshold && b < epochThreshold;
    return bothEpoch || bothRelative;
}

function _taskSpeedEtaEstimate(task, now, historicalSpeeds, stageText) {
    const isUvr = task.type === 'Isolation' || _containsAnyKeyword(String(stageText || '').toLowerCase(), ['vocal', 'separation', 'uvr']);
    return calculateTaskSpeedAndEta(task, now, historicalSpeeds, isUvr);
}

function _taskSpeedEtaView(elapsedActive, estimate) {
    return {
        showSpeed: elapsedActive > 5 && estimate.calculatedSpeed > 0,
        speedText: estimate.calculatedSpeed.toFixed(1) + 'x',
        showEta: elapsedActive > 5 && estimate.remainingSeconds > 0,
        etaText: formatDur(estimate.remainingSeconds)
    };
}

function _cleanupTimelineForTasks(tasks) {
    if (typeof activeTaskTimeline === 'undefined') {
        return;
    }
    const activeIds = new Set(tasks.map((task) => task.task_id || task.filename));
    for (const id in activeTaskTimeline) {
        if (!activeIds.has(id)) {
            delete activeTaskTimeline[id];
        }
    }
}