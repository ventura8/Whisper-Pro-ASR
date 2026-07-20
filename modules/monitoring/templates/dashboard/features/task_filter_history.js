function normalizeTaskFilterType(type) {
    if (type === 'isolation' || type === 'isolations') return 'detectlang';
    if (type === 'detect-language') return 'detectlang';
    return type;
}

function isAsrLikeCategory(category) {
    return category === 'asr' || category === 'v1';
}

function matchesCategoryFilter(item, rawFilter) {
    const selected = normalizeTaskFilterType(rawFilter);
    const category = getTaskFilterCategory(item);
    if (selected === 'asr') return category === 'asr';
    if (selected === 'detectlang') return category === 'detectlang';
    if (selected === 'v1') return category === 'v1';
    return true;
}

function getTaskFilterCategory(task) {
    const typeLower = _taskFieldLower(task, 'type');
    const stageLower = _taskFieldLower(task, 'stage');

    if (_isV1TaskType(typeLower)) return 'v1';
    if (_isDetectLanguageTaskType(typeLower, stageLower)) return 'detectlang';
    if (_isAsrTaskType(typeLower)) return 'asr';
    return 'other';
}

function _taskFieldLower(task, field) {
    if (!task || task[field] === null || task[field] === undefined) {
        return '';
    }
    return String(task[field]).toLowerCase();
}

function _containsAnyKeyword(text, keywords) {
    return keywords.some((keyword) => text.includes(keyword));
}

function _isV1TaskType(typeLower) {
    return _containsAnyKeyword(typeLower, ['/v1/audio/', 'v1/audio']);
}

function _isDetectLanguageTaskType(typeLower, stageLower) {
    const detectTypeKeywords = [
        '/detect-language',
        '/detectlang',
        'detect-language',
        'detectlang',
        'language detection',
        'isolation',
        'uvr'
    ];
    const detectStageKeywords = ['vocal', 'separation', 'uvr'];
    return _containsAnyKeyword(typeLower, detectTypeKeywords) || _containsAnyKeyword(stageLower, detectStageKeywords);
}

function _isAsrTaskType(typeLower) {
    if (_isV1TaskType(typeLower)) {
        return false;
    }
    return _containsAnyKeyword(typeLower, ['/asr', 'asr', 'transcription', 'trans']);
}

function filterTasks(type) {
    const normalizedType = normalizeTaskFilterType(type);
    globalThis.activeTaskFilter = normalizedType;
    document.querySelectorAll('#active-section .filter-row button').forEach(btn => {
        btn.classList.remove('active-filter');
    });
    const selectedBtn = document.getElementById(`filter-${normalizedType}`);
    if (selectedBtn) selectedBtn.classList.add('active-filter');
    updateStats();
}

function filterHistory(type) {
    const normalizedType = normalizeTaskFilterType(type);
    globalThis.historyTaskFilter = normalizedType;
    document.querySelectorAll('#history-section .filter-row button').forEach(btn => {
        btn.classList.remove('active-filter');
    });
    const selectedBtn = document.getElementById(`hist-filter-${normalizedType}`);
    if (selectedBtn) selectedBtn.classList.add('active-filter');
    renderHistory();
}

function toggleRefresh() {
    refreshEnabled = !refreshEnabled;
    const icon = document.getElementById('refresh-icon');
    const text = document.getElementById('refresh-text');
    const btn = document.getElementById('toggle-refresh');
    if (refreshEnabled) {
        icon.innerText = 'sync';
        icon.classList.add('pulse');
        text.innerText = 'Live Refresh';
        btn.style.background = 'var(--md-sys-color-primary-container)';
        btn.style.color = 'var(--md-sys-color-primary)';
        updateStats();
        startRefreshInterval();
    } else {
        icon.innerText = 'sync_disabled';
        icon.classList.remove('pulse');
        text.innerText = 'Refresh Paused';
        btn.style.background = 'var(--meta-bg)';
        btn.style.color = 'var(--md-sys-color-secondary)';
        startRefreshInterval();
    }
}

function renderHistory() {
    const hList = document.getElementById('history-list');
    if (!fullTaskHistory || fullTaskHistory.length === 0) {
        hList.innerHTML = `<div class="empty-state"><span class="material-icons-sharp empty-icon">history</span><div><strong>No history yet</strong></div></div>`;
        return;
    }

    const filteredHistory = [...fullTaskHistory].filter(_historyMatchesSelectedFilter);

    if (filteredHistory.length === 0) {
        hList.innerHTML = `<div class="empty-state"><span class="material-icons-sharp empty-icon">history</span><div><strong>No history matches filter</strong></div></div>`;
        return;
    }

    const orderedHistory = filteredHistory.sort(_compareHistoryItems);

    hList.innerHTML = orderedHistory.map(_renderHistoryCardByIndex).join('');
}

function _historyMatchesSelectedFilter(item) {
    return matchesCategoryFilter(item, globalThis.historyTaskFilter);
}

function _compareHistoryItems(a, b) {
    const startDiff = _historyStartTime(b) - _historyStartTime(a);
    if (startDiff !== 0) {
        return startDiff;
    }
    return _historyTaskId(b).localeCompare(_historyTaskId(a));
}

function _historyStartTime(item) {
    if (!item || item.start_time === null || item.start_time === undefined) {
        return 0;
    }
    return Number(item.start_time);
}

function _historyTaskId(item) {
    if (!item || item.task_id === null || item.task_id === undefined) {
        return '';
    }
    return String(item.task_id);
}

function _renderHistoryCardByIndex(item, index) {
    const card = _buildHistoryCardData(item, index);
    return _renderHistoryCardHtml(card);
}

function _buildHistoryCardData(h, i) {
    const id = h.task_id || `hist-${i}`;
    const result = _historyResultPayload(h);
    const category = getTaskFilterCategory(h);
    const isAsrLike = isAsrLikeCategory(category);
    const finalSrt = _historyFinalSrt(h, result);
    _warnIfMissingHistoryText(result, id, h);
    const statusMeta = _historyStatusMeta(h.status);
    return {
        id,
        h,
        result,
        safeFilename: escapeHtml(h.filename || 'Unknown Media'),
        isAsrLike,
        finalSrt,
        speed: _historySpeedText(h),
        statusMeta,
        langBadge: _historyLanguageBadge(result),
        typeIcon: isAsrLike ? 'record_voice_over' : 'translate',
        hwTag: _historyHardwareTag(h),
        resText: _historyResultSection(id, result, finalSrt, isAsrLike),
        segmentsCount: _historySegmentsCount(h, result),
        taskTypeLabel: escapeHtml(h.type ? h.type : 'Task'),
        processingDuration: _historyProcessingDuration(h),
        queueDuration: _historyQueueDuration(h),
        logsText: _historyLogsText(h)
    };
}

function _historyProcessingDuration(h) {
    return h.active_elapsed_sec ? h.active_elapsed_sec : h.total_elapsed_sec;
}

function _historyQueueDuration(h) {
    return h.queue_elapsed_sec ? h.queue_elapsed_sec : 0;
}

function _historyLogsText(h) {
    return escapeHtml((h.logs || []).join('\n'));
}

function _historyResultPayload(h) {
    return h.result ? h.result : (h.response_json ? h.response_json : {});
}

function _historyFinalSrt(h, result) {
    return result.text ? result.text : h.live_text;
}

function _warnIfMissingHistoryText(result, id, item) {
    if (!result.text) {
        console.warn('[Dashboard] History item missing result.text:', id, item);
    }
}

function _historySpeedText(h) {
    const elapsed = h.active_elapsed_sec || h.total_elapsed_sec;
    if (!(h.video_duration > 0) || !(elapsed > 0)) {
        return '0.0x';
    }
    return (h.video_duration / elapsed).toFixed(1) + 'x';
}

function _historyStatusMeta(status) {
    const isFailed = normalizeStatus(status) === 'failed';
    if (isFailed) {
        return {
            statusClass: 'badge-failed',
            statusLabel: 'Failed',
            metaBg: 'rgba(186, 26, 26, 0.15)',
            metaColor: 'var(--md-sys-color-error)',
            metaIcon: 'close'
        };
    }
    return {
        statusClass: 'badge-active',
        statusLabel: 'Finished',
        metaBg: 'rgba(46, 125, 50, 0.15)',
        metaColor: 'var(--md-sys-color-success)',
        metaIcon: 'check'
    };
}

function _historyLanguageBadge(result) {
    const code = result.language || result.detected_language;
    if (!code) {
        return '';
    }
    return `<span class="badge badge-lang" style="margin-left:auto;">${escapeHtml(code.toUpperCase())}</span>`;
}

function _historyHardwareTag(h) {
    const historyUnitId = h.history_unit_id || h.unit_id;
    if (historyUnitId) {
        const hw = getHwIconAndLabel(historyUnitId);
        if (hw.label !== 'Queued') {
            return `<span class="meta-tag" title="Hardware"><span class="material-icons-sharp" style="font-size:12px;color:var(--md-sys-color-primary)">${hw.icon}</span>${hw.label}</span>`;
        }
    }

    const fallbackType = h.history_unit_type || h.unit_type;
    const fallbackName = h.history_unit_name || h.unit_name;
    if (!fallbackType && !fallbackName) {
        return '';
    }

    const icon = _historyHardwareIconForType(fallbackType);
    const label = _historyHardwareLabelFromMeta(fallbackType, fallbackName);
    return `<span class="meta-tag" title="Hardware"><span class="material-icons-sharp" style="font-size:12px;color:var(--md-sys-color-primary)">${icon}</span>${escapeHtml(label)}</span>`;
}

function _historyHardwareIconForType(unitType) {
    const type = String(unitType || '').toUpperCase();
    if (type.startsWith('CUDA')) return 'rocket_launch';
    if (type.startsWith('NPU')) return 'psychology_alt';
    if (type.startsWith('GPU')) return 'developer_board';
    if (type.startsWith('AMD')) return 'bolt';
    if (type === 'CPU') return 'settings_input_component';
    return 'memory';
}

function _historyHardwareLabelFromMeta(unitType, unitName) {
    if (unitName) {
        return String(unitName);
    }
    if (unitType) {
        return String(unitType);
    }
    return 'Unknown Hardware';
}

function _historyResultSection(id, result, finalSrt, isAsrLike) {
    if (!(result.error || finalSrt || isAsrLike)) {
        return '';
    }
    const contentHtml = _historyResultContent(result, finalSrt, isAsrLike);
    const downloadButton = _historyResultDownloadButton(id, finalSrt);
    const openAttr = expandedElements.has(`${id}_trans`) ? 'open' : '';
    return `
            <div style="margin-top:12px; width:100%;">
                <details ${openAttr} style="width:100%;" ontoggle="handleToggle('${id}_trans', this.open)">
                    <summary style="display:flex; align-items:center; justify-content:space-between; width:100%;">
                        <span style="display:flex; align-items:center; gap:4px;">
                            <span class="material-icons-sharp">subtitles</span> View Transcription Result
                        </span>
                        ${downloadButton}
                    </summary>
                    <div style="margin-top:8px; width:100%;">
                        ${contentHtml}
                    </div>
                </details>
            </div>`;
}

function _historyResultDownloadButton(id, finalSrt) {
    if (!finalSrt) {
        return '';
    }
    return `
                        <button class="btn-time" style="display:flex; align-items:center; gap:4px; padding:6px 12px;" onclick="event.stopPropagation(); downloadSrtById('${id}')">
                            <span class="material-icons-sharp" style="font-size:16px">download</span> SRT
                        </button>`;
}

function _historyResultContent(result, finalSrt, isAsrLike) {
    if (result.error) {
        return `<div class="result-box" style="color:var(--md-sys-color-error); border-color:var(--md-sys-color-error)">
                <span class="material-icons-sharp" style="font-size:14px; vertical-align:middle">error</span> 
                <strong>Error:</strong> ${escapeHtml(result.error)}
            </div>`;
    }
    if (finalSrt) {
        return `<div class="result-box">${escapeHtml(finalSrt)}</div>`;
    }
    if (isAsrLike) {
        return `<div class="result-box" style="font-style:italic; color:var(--md-sys-color-secondary)">
                <span class="material-icons-sharp" style="font-size:14px; vertical-align:middle">info</span> 
                No speech detected or transcription failed.
            </div>`;
    }
    return '';
}

function _historySegmentsCount(h, result) {
    if (h.segments_processed !== undefined) {
        return h.segments_processed;
    }
    if (result.segments) {
        return result.segments.length;
    }
    if (result.segments_processed) {
        return result.segments_processed;
    }
    return 0;
}

function _renderHistoryCardHtml(card) {
    return `<div class="history-card">
            <div class="task-header">
                <div class="task-icon-container" style="background:var(--md-sys-color-primary-container); color:var(--md-sys-color-primary);">
                    <span class="material-icons-sharp">${card.typeIcon}</span>
                </div>
                <div class="item-info">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                        <span class="item-primary">${card.safeFilename}</span>
                        <span class="meta-tag" style="background:${card.statusMeta.metaBg}; color:${card.statusMeta.metaColor}; border:none; padding:1px 8px; border-radius:100px; font-weight:700;">
                            <span class="material-icons-sharp" style="font-size:13px">${card.statusMeta.metaIcon}</span> ${card.taskTypeLabel}
                        </span>
                    </div>
                    <div class="item-secondary">
                        <span class="meta-tag" title="Completed At"><span class="material-icons-sharp" style="font-size:12px;color:var(--md-sys-color-secondary)">schedule</span>${card.h.completed_at}</span>
                        <span class="meta-tag"><span class="material-icons-sharp" style="font-size:12px">movie</span>${formatDur(card.h.video_duration)}</span>
                        <span class="meta-tag" title="Processing Time"><span class="material-icons-sharp" style="font-size:12px">timer</span>Took: ${formatDur(card.processingDuration)}</span>
                        <span class="meta-tag" title="Queue Time"><span class="material-icons-sharp" style="font-size:12px">hourglass_empty</span>Queue: ${formatDur(card.queueDuration)}</span>
                        <span class="meta-tag" title="Transcription Speed"><span class="material-icons-sharp" style="font-size:12px">speed</span>Speed: ${card.speed}</span>
                        ${card.hwTag}
                        <span class="meta-tag" title="Processed Segments"><span class="material-icons-sharp" style="font-size:12px">segment</span>Segments: ${card.segmentsCount}</span>
                    </div>
                </div>
                ${card.langBadge}
                <span class="badge ${card.statusMeta.statusClass}" style="margin-left:8px;">${card.statusMeta.statusLabel}</span>
            </div>
            ${card.resText}
            ${renderAuditDetails(card.h)}
            <details ${expandedElements.has(`${card.id}_logs`) ? 'open' : ''} ontoggle="handleToggle('${card.id}_logs', this.open)">
                <summary><span class="material-icons-sharp">terminal</span> View Execution Logs</summary>
                <div class="log-buffer">${card.logsText}</div>
            </details>
        </div>`;
}