function showTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');
    ['active', 'history', 'analytics', 'charts', 'settings'].forEach(s => {
        document.getElementById(`${s}-section`).style.display = (s === tab ? 'block' : 'none');
    });

    if (tab === 'charts') {
        // Clear the cached range/theme states to force a full updateOptions redraw.
        // This resolves ApexCharts collapsing/failing to render when transitioning
        // from display: none to display: block.
        for (let id in lastChartStates) {
            lastChartStates[id].rangeMs = null;
            lastChartStates[id].theme = null;
        }
        renderCharts();
        // Trigger a resize event to ensure ApexCharts properly recalculates visual dimensions
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 50);
    }
    if (tab === 'history') {
        fetch('/history').then(res => res.json()).then(data => {
            fullTaskHistory = data || [];
            renderHistory();
        });
    }
}

function handleToggle(id, isOpen) {
    if (isOpen) {
        expandedElements.add(id);
        // Force scroll to bottom on open
        setTimeout(() => {
            const el = document.querySelector(`[data-task-id="${id.split('_')[0]}"] .log-buffer, [data-task-id="${id.split('_')[0]}"] .live-text-box`);
            if (el) el.scrollTop = el.scrollHeight;
        }, 50);
    } else {
        expandedElements.delete(id);
    }
}

async function saveSettings() {
    const telemetryHours = document.getElementById('retention-range').value;
    const logDays = document.getElementById('log-retention-range').value;
    try {
        await fetch('/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                telemetry_retention_hours: parseInt(telemetryHours),
                log_retention_days: parseInt(logDays)
            })
        });
        alert("Configuration saved!");
    } catch (e) { alert("Failed to save settings: " + e); }
}

function renderAuditDetails(item, isOpen) {
    const id = item.task_id || item.filename;
    const caller = item.caller_info || {};
    const reqJson = JSON.stringify(item.request_json || {}, null, 2);
    const resJson = JSON.stringify(item.result || item.response_json || {}, null, 2);

    const auditOpen = expandedElements.has(`${id}_audit`) ? 'open' : '';
    const reqOpen = expandedElements.has(`${id}_req`) ? 'open' : '';
    const resOpen = expandedElements.has(`${id}_res`) ? 'open' : '';

    return `
        <div style="margin-top:12px; border-top:1px solid var(--border-color); padding-top:12px;">
            <details ${auditOpen} ontoggle="handleToggle('${id}_audit', this.open)">
                <summary style="font-size:13px; color:var(--md-sys-color-secondary); margin-bottom:8px;">
                    <span class="material-icons-sharp" style="font-size:14px">policy</span> Audit & Caller Info
                </summary>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; padding-left:16px;">
                    <div class="item-secondary"><span class="material-icons-sharp" style="font-size:12px">language</span> IP: ${escapeHtml(caller.ip || 'Local')}</div>
                    <div class="item-secondary" title="${escapeHtml(caller.user_agent || '')}"><span class="material-icons-sharp" style="font-size:12px">devices</span> UA: ${caller.user_agent ? escapeHtml(caller.user_agent.length > 30 ? caller.user_agent.substring(0, 30) + '...' : caller.user_agent) : 'Not provided by client'}</div>
                </div>
            </details>
            <details ${reqOpen} ontoggle="handleToggle('${id}_req', this.open)">
                <summary><span class="material-icons-sharp">code</span> Request Payload</summary>
                <div class="json-buffer">${reqJson}</div>
            </details>
            <details ${resOpen} style="margin-top:8px;" ontoggle="handleToggle('${id}_res', this.open)">
                <summary><span class="material-icons-sharp">data_object</span> Response Payload</summary>
                <div class="json-buffer">${resJson}</div>
            </details>
        </div>
    `;
}

function renderHistory() {
    const hList = document.getElementById('history-list');
    if (!fullTaskHistory || fullTaskHistory.length === 0) {
        hList.innerHTML = `<div class="empty-state"><span class="material-icons-sharp empty-icon">history</span><div><strong>No history yet</strong></div></div>`;
        return;
    }
    const orderedHistory = [...fullTaskHistory].sort((a, b) => {
        const aStart = Number(a.start_time || 0);
        const bStart = Number(b.start_time || 0);
        if (aStart !== bStart) return bStart - aStart;
        return String(b.task_id || '').localeCompare(String(a.task_id || ''));
    });

    hList.innerHTML = orderedHistory.map((h, i) => {
        const id = h.task_id || `hist-${i}`;
        const result = h.result || {};
        
        if (!result.text) {
            console.warn("[Dashboard] History item missing result.text:", id, h);
        }
        const speed = (h.video_duration > 0 && (h.active_elapsed_sec || h.total_elapsed_sec) > 0)
            ? (h.video_duration / (h.active_elapsed_sec || h.total_elapsed_sec)).toFixed(1) + 'x'
            : '0.0x';
        
        const typeLower = (h.type || "").toLowerCase();
        const isAsr = typeLower.includes('asr') || typeLower.includes('trans') || typeLower.includes('audio');
        const finalSrt = result.text || h.live_text;
        
        let contentHtml = '';
        if (result.error) {
            contentHtml = `<div class="result-box" style="color:var(--md-sys-color-error); border-color:var(--md-sys-color-error)">
                <span class="material-icons-sharp" style="font-size:14px; vertical-align:middle">error</span> 
                <strong>Error:</strong> ${escapeHtml(result.error)}
            </div>`;
        } else if (finalSrt) {
            contentHtml = `<div class="result-box">${escapeHtml(finalSrt)}</div>`;
        } else if (isAsr) {
            contentHtml = `<div class="result-box" style="font-style:italic; color:var(--md-sys-color-secondary)">
                <span class="material-icons-sharp" style="font-size:14px; vertical-align:middle">info</span> 
                No speech detected or transcription failed.
            </div>`;
        }

        const resText = (result.error || finalSrt || isAsr) ? `
            <div style="margin-top:12px; width:100%;">
                <details ${expandedElements.has(`${id}_trans`) ? 'open' : ''} style="width:100%;" ontoggle="handleToggle('${id}_trans', this.open)">
                    <summary style="display:flex; align-items:center; justify-content:space-between; width:100%;">
                        <span style="display:flex; align-items:center; gap:4px;">
                            <span class="material-icons-sharp">subtitles</span> View Transcription Result
                        </span>
                        ${finalSrt ? `
                        <button class="btn-time" style="display:flex; align-items:center; gap:4px; padding:6px 12px;" onclick="event.stopPropagation(); downloadSrtById('${id}')">
                            <span class="material-icons-sharp" style="font-size:16px">download</span> SRT
                        </button>` : ''}
                    </summary>
                    <div style="margin-top:8px; width:100%;">
                        ${contentHtml}
                    </div>
                </details>
            </div>` : '';

        const isFailed = normalizeStatus(h.status) === 'failed';
        const statusClass = isFailed ? 'badge-failed' : 'badge-active';
        const statusLabel = isFailed ? 'Failed' : 'Finished';
        const metaBg = isFailed ? 'rgba(186, 26, 26, 0.15)' : 'rgba(46, 125, 50, 0.15)';
        const metaColor = isFailed ? 'var(--md-sys-color-error)' : 'var(--md-sys-color-success)';
        const metaIcon = isFailed ? 'close' : 'check';

        const langCode = result.language || result.detected_language;
        const langBadge = langCode ? `<span class="badge badge-lang" style="margin-left:auto;">${langCode.toUpperCase()}</span>` : '';

        const typeIcon = (h.type === '/asr' || h.type === 'Transcription') ? 'record_voice_over' : 'translate';
        const hw = getHwIconAndLabel(h.unit_id);

        return `<div class="history-card">
            <div class="task-header">
                <div class="task-icon-container" style="background:var(--md-sys-color-primary-container); color:var(--md-sys-color-primary);">
                    <span class="material-icons-sharp">${typeIcon}</span>
                </div>
                <div class="item-info">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                        <span class="item-primary">${h.filename}</span>
                        <span class="meta-tag" style="background:${metaBg}; color:${metaColor}; border:none; padding:1px 8px; border-radius:100px; font-weight:700;">
                            <span class="material-icons-sharp" style="font-size:13px">${metaIcon}</span> ${h.type || 'Task'}
                        </span>
                    </div>
                    <div class="item-secondary">
                        <span class="meta-tag" title="Completed At"><span class="material-icons-sharp" style="font-size:12px;color:var(--md-sys-color-secondary)">schedule</span>${h.completed_at}</span>
                        <span class="meta-tag"><span class="material-icons-sharp" style="font-size:12px">movie</span>${formatDur(h.video_duration)}</span>
                        <span class="meta-tag" title="Processing Time"><span class="material-icons-sharp" style="font-size:12px">timer</span>Took: ${formatDur(h.active_elapsed_sec || h.total_elapsed_sec)}</span>
                        <span class="meta-tag" title="Queue Time"><span class="material-icons-sharp" style="font-size:12px">hourglass_empty</span>Queue: ${formatDur(h.queue_elapsed_sec || 0)}</span>
                        <span class="meta-tag" title="Transcription Speed"><span class="material-icons-sharp" style="font-size:12px">speed</span>Speed: ${speed}</span>
                        <span class="meta-tag" title="Hardware"><span class="material-icons-sharp" style="font-size:12px;color:var(--md-sys-color-primary)">${hw.icon}</span>${hw.label}</span>
                        <span class="meta-tag" title="Processed Segments"><span class="material-icons-sharp" style="font-size:12px">segment</span>Segments: ${h.segments_processed !== undefined ? h.segments_processed : ((result.segments) ? result.segments.length : ((result.segments_processed) ? result.segments_processed : 0))}</span>
                    </div>
                </div>
                ${langBadge}
                <span class="badge ${statusClass}" style="margin-left:8px;">${statusLabel}</span>
            </div>
            ${resText}
            ${renderAuditDetails(h)}
            <details ${expandedElements.has(`${id}_logs`) ? 'open' : ''} ontoggle="handleToggle('${id}_logs', this.open)">
                <summary><span class="material-icons-sharp">terminal</span> View Execution Logs</summary>
                <div class="log-buffer">${(h.logs || []).join('\n')}</div>
            </details>
        </div>`;
    }).join('');
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
    } else {
        icon.innerText = 'sync_disabled';
        icon.classList.remove('pulse');
        text.innerText = 'Refresh Paused';
        btn.style.background = 'var(--meta-bg)';
        btn.style.color = 'var(--md-sys-color-secondary)';
    }
}

// Helper function to calculate expected speeds from history
function calculateHistoricalSpeeds(history) {
    let sumAsrSpeed = 0, countAsr = 0;
    let sumUvrSpeed = 0, countUvr = 0;
    if (history) {
        history.forEach(h => {
            if (normalizeStatus(h.status) === 'completed' && h.video_duration > 0) {
                const perf = (h.result && h.result.performance) ? h.result.performance : (h.response_json && h.response_json.performance ? h.response_json.performance : null);
                if (perf) {
                    if (perf.inference_sec > 0) {
                        sumAsrSpeed += h.video_duration / perf.inference_sec;
                        countAsr++;
                    }
                    if (perf.isolation_sec > 0) {
                        sumUvrSpeed += h.video_duration / perf.isolation_sec;
                        countUvr++;
                    }
                }
            }
        });
    }
    return {
        expectedAsrSpeed: countAsr > 0 ? sumAsrSpeed / countAsr : 0,
        expectedUvrSpeed: countUvr > 0 ? sumUvrSpeed / countUvr : 0
    };
}

// Helper function to calculate speed and ETA for a task
function calculateTaskSpeedAndEta(t, now, history, isUvr) {
    let calculatedSpeed = 0;
    let remainingSeconds = 0;
    const startActive = t.start_active || t.start_time;
    const elapsedActive = now - startActive;
    const processedDuration = t.current_position || 0;

    const { expectedAsrSpeed, expectedUvrSpeed } = calculateHistoricalSpeeds(history);

    if (isUvr) {
        const uvrSpeed = (elapsedActive > 0 && processedDuration > 0) ? (processedDuration / elapsedActive) : 0;

        if (uvrSpeed > 0) {
            const remainingUvrSec = (t.video_duration - processedDuration) / uvrSpeed;
            const expectedAsrSec = (expectedAsrSpeed > 0) ? (t.video_duration / expectedAsrSpeed) : 0;
            const totalEstimatedSec = elapsedActive + remainingUvrSec + expectedAsrSec;
            if (totalEstimatedSec > 0) {
                calculatedSpeed = t.video_duration / totalEstimatedSec;
                remainingSeconds = remainingUvrSec + expectedAsrSec;
            }
        } else if (expectedUvrSpeed > 0) {
            const expectedUvrSec = t.video_duration / expectedUvrSpeed;
            const expectedAsrSec = (expectedAsrSpeed > 0) ? (t.video_duration / expectedAsrSpeed) : 0;
            const totalEstimatedSec = expectedUvrSec + expectedAsrSec;
            if (totalEstimatedSec > 0) {
                calculatedSpeed = t.video_duration / totalEstimatedSec;
                remainingSeconds = Math.max(0, totalEstimatedSec - elapsedActive);
            }
        }
    } else {
        const startInference = t.start_inference || startActive;
        const elapsedAsr = now - startInference;
        const uvrElapsed = t.start_inference ? (t.start_inference - startActive) : 0;

        if (elapsedAsr > 5 && processedDuration > 0) {
            const asrSpeed = processedDuration / elapsedAsr;
            if (asrSpeed > 0) {
                remainingSeconds = (t.video_duration - processedDuration) / asrSpeed;
                const totalEstimatedSec = uvrElapsed + elapsedAsr + remainingSeconds;
                if (totalEstimatedSec > 0) {
                    calculatedSpeed = t.video_duration / totalEstimatedSec;
                }
            }
        } else if (expectedAsrSpeed > 0 && elapsedAsr > 0) {
            const expectedAsrSec = t.video_duration / expectedAsrSpeed;
            const totalEstimatedSec = uvrElapsed + expectedAsrSec;
            if (totalEstimatedSec > 0) {
                calculatedSpeed = t.video_duration / totalEstimatedSec;
                remainingSeconds = Math.max(0, totalEstimatedSec - elapsedActive);
            }
        }
    }

    return { calculatedSpeed, remainingSeconds };
}

function getDefaultStageFromStatus(status) {
    const key = normalizeStatus(status);
    if (key === 'active') return 'Active';
    if (key === 'queued') return 'Queued';
    if (key === 'post-processing') return 'Post-Processing';
    if (key === 'completed') return 'Completed';
    if (key === 'failed') return 'Failed';
    return 'Initializing';
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
    if (!normalized) {
        return true;
    }

    if (['none', 'null', 'undefined', 'unknown', 'na', 'n/a'].includes(normalized)) {
        return true;
    }

    const ratioCandidate = normalized.replace(/[()\s]/g, '');
    if (ratioCandidate === '0/0') {
        return true;
    }

    if (normalized.includes('placeholder')) {
        return true;
    }

    if (normalized === 'resume' || normalized === 'resuming') {
        return true;
    }

    return false;
}

function normalizeStage(task) {
    const raw = task ? task.stage : '';
    const str = (raw === null || raw === undefined) ? '' : String(raw).trim();
    if (!isPlaceholderStage(str)) {
        return str;
    }
    return getDefaultStageFromStatus(task ? task.status : '');
}

async function updateStats() {
    if (!refreshEnabled) return;
    try {
        const res = await fetch('/status');
        const data = await res.json();
        const now = Date.now() / 1000;
        document.getElementById('app-version').innerText = `Version ${data.version}`;
        document.getElementById('app-cpu-val').innerText = `${data.system.app_cpu_percent}%`;
        document.getElementById('app-cpu-bar').style.width = `${Math.min(100, data.system.app_cpu_percent)}%`;
        document.getElementById('sys-cpu-val').innerText = `${Math.round(data.system.cpu_percent)}%`;
        document.getElementById('sys-cpu-bar').style.width = `${data.system.cpu_percent}%`;
        document.getElementById('app-mem-val').innerText = `${data.system.app_memory_gb.toFixed(2)} GB`;
        document.getElementById('app-mem-bar').style.width = `${(data.system.app_memory_gb / data.system.memory_total_gb) * 100}%`;
        document.getElementById('sys-mem-val').innerText = `${data.system.memory_used_gb.toFixed(2)} / ${data.system.memory_total_gb.toFixed(2)} GB`;
        document.getElementById('sys-mem-bar').style.width = `${data.system.memory_percent}%`;
        if (!data || !data.system) {
            console.warn("Incomplete status data received");
            return;
        }
        document.getElementById('active-val').innerText = data.active_sessions || 0;
        document.getElementById('queued-val').innerText = data.queued_sessions || 0;

        lastStatusData = data;
        currentTelemetry = data.telemetry_history || [];

        // Update rolling telemetry buffer (up to 10 minutes retention)
        const nowSec = Date.now() / 1000;
        const newPoint = {
            timestamp: nowSec,
            system: data.system,
            telemetry: data.telemetry
        };
        rollingTelemetryBuffer.push(newPoint);
        // Keep last 10 minutes (600 seconds)
        rollingTelemetryBuffer = rollingTelemetryBuffer.filter(h => h.timestamp >= nowSec - 600);

        // Initial pre-population from server history on first load
        if (rollingTelemetryBuffer.length === 1 && data.telemetry_history && data.telemetry_history.length > 0) {
            const serverHist = data.telemetry_history.filter(h => h.timestamp >= nowSec - 600 && h.timestamp < nowSec);
            rollingTelemetryBuffer = [...serverHist, ...rollingTelemetryBuffer];
        }

        fullTaskHistory = data.history || [];

        if (currentTab === 'charts') renderCharts();
        if (currentTab === 'history') renderHistory();

        // Auto-scroll active live transcription and logs only
        setTimeout(() => {
            document.querySelectorAll('#task-list .live-text-box, #task-list .log-buffer').forEach(box => {
                box.scrollTop = box.scrollHeight;
            });
        }, 100);


        if (data.history_stats) {
            document.getElementById('analytics-grid').innerHTML = `
                <div class="stat-box"><div class="stat-label">Today</div><div class="stat-value">${formatDDHHMMSS(data.history_stats.today)}</div></div>
                <div class="stat-box"><div class="stat-label">Tasks Today</div><div class="stat-value">${data.history_stats.count_today}</div></div>
                <div class="stat-box"><div class="stat-label">This Month</div><div class="stat-value">${formatDDHHMMSS(data.history_stats.this_month)}</div></div>
                <div class="stat-box"><div class="stat-label">All Time</div><div class="stat-value">${formatDDHHMMSS(data.history_stats.all_time)}</div></div>
                <div class="stat-box"><div class="stat-label">Total Tasks</div><div class="stat-value">${data.history_stats.count_all_time}</div></div>
                <div class="stat-box"><div class="stat-label">Service Uptime</div><div class="stat-value">${formatDDHHMMSS(data.uptime_sec)}</div></div>
            `;
        }
        const tel = data.telemetry || {};
        document.getElementById('hw-pool').innerHTML = (data.hardware_units || []).map(u => {
            const unitId = u.id;
            const tasks = data.tasks || [];
            const tel = data.telemetry || {};
            let isUsed = tasks.some(t => String(t.unit_id) === String(unitId) && t.status === 'active');
            
            let icon = 'memory';

            if (tel.hardware_util && tel.hardware_util[unitId] !== undefined) {
                if (!isUsed) isUsed = tel.hardware_util[unitId] > 0;
            }

            if (u.type === 'GPU' && !u.name.includes('NVIDIA')) {
                if (!isUsed && tel.hardware_util === undefined) isUsed = (tel.intel_gpu_load || 0) > 0;
                icon = 'developer_board';
            } else if (u.type === 'NPU') {
                if (!isUsed && tel.hardware_util === undefined) isUsed = (tel.npu_load || 0) > 0;
                icon = 'psychology_alt';
            } else if (u.type === 'CUDA' || (u.type === 'GPU' && u.name.includes('NVIDIA'))) {
                const idx = parseInt(unitId.split(':')[1] || 0);
                if (!isUsed && tel.hardware_util === undefined && tel.nvidia && tel.nvidia[idx]) {
                    isUsed = tel.nvidia[idx].util > 0;
                }
                icon = 'rocket_launch';
            } else if (u.type === 'CPU') {
                if (!isUsed) isUsed = tasks.some(t => t.unit_id === 'CPU' && t.status === 'active');
                icon = 'settings_input_component';
            }

            const statusText = isUsed ? "Used" : "Not used";
            const statusClass = isUsed ? "status-used" : "status-idle";

            return `
                <div class="hw-card">
                    <div class="hw-card-title"><span class="material-icons-sharp" style="font-size:12px">${icon}</span> ${u.type}</div>
                    <div style="font-size: 11px; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 2px;">${u.name}</div>
                    <div class="hw-card-status ${statusClass}" style="margin-bottom: 6px;">${statusText}</div>
                    <div style="display: flex; flex-direction: column; gap: 4px; align-items: flex-start;">
                        <span class="badge badge-${u.uvr_status || 'ready'}" style="padding: 2px 6px; font-size: 9px; border-radius: 4px; line-height: 1.2;">UVR: ${u.uvr_status || 'ready'}</span>
                        <span class="badge badge-${u.whisper_status || 'ready'}" style="padding: 2px 6px; font-size: 9px; border-radius: 4px; line-height: 1.2;">Whisper: ${u.whisper_status || 'ready'}</span>
                    </div>
                </div>
            `;
        }).join('');

        const tasks = data.tasks || [];
        const tList = document.getElementById('task-list');
        const existingTaskIds = new Set(tasks.map(t => t.task_id || t.filename));

        Array.from(tList.children).forEach(card => {
            if (card.dataset.taskId && !existingTaskIds.has(card.dataset.taskId)) {
                card.remove();
            }
        });

        if (tasks.length === 0) {
            if (!tList.querySelector('.empty-state')) {
                tList.innerHTML = `<div class="empty-state"><span class="material-icons-sharp empty-icon">auto_awesome</span><div><strong>Service is idle</strong></div></div>`;
            }
        } else {
            const empty = tList.querySelector('.empty-state');
            if (empty) empty.remove();

            tasks
                .slice()
                .sort((a, b) => {
                    // Enforce deterministic three-tier ordering per task_status_display_specification_skill:
                    // 1. Active tasks first (status='active'), sorted by start_time ascending
                    // 2. Priority queued tasks (status='queued' with is_priority=true), sorted by start_time ascending
                    // 3. Standard queued tasks (status='queued' with is_priority=false), sorted by start_time ascending
                    const getOrderKey = (t) => {
                        const status = normalizeStatus(t.status);
                        const startTime = Number(t.start_time || 0);
                        const taskId = String(t.task_id || '');
                        const isPriority = !!t.is_priority;
                        
                        if (status === 'active') {
                            return [0, startTime, taskId];
                        } else if (status === 'queued' && isPriority) {
                            return [1, startTime, taskId];
                        } else if (status === 'queued' && !isPriority) {
                            return [2, startTime, taskId];
                        } else {
                            return [3, startTime, taskId];
                        }
                    };
                    
                    const keyA = getOrderKey(a);
                    const keyB = getOrderKey(b);
                    
                    if (keyA[0] !== keyB[0]) return keyA[0] - keyB[0];
                    if (keyA[1] !== keyB[1]) return keyA[1] - keyB[1];
                    return keyA[2].localeCompare(keyB[2]);
                })
                .forEach(t => {
                const id = t.task_id || t.filename;
                let card = tList.querySelector(`[data-task-id="${id}"]`);
                const normalizedStatus = normalizeStatus(t.status);

                if (!card) {
                    card = document.createElement('div');
                    card.className = 'task-card';
                    card.dataset.taskId = id;
                    tList.appendChild(card);
                }

                if (normalizedStatus === 'queued') {
                    card.classList.add('queued');
                } else {
                    card.classList.remove('queued');
                }

                const icon = (t.type === '/asr' || t.type === 'Transcription' || t.type === 'ASR') ? 'record_voice_over' : 'translate';
                const pulseClass = normalizedStatus === 'active' ? 'pulse' : '';

                const typeLower = (t.type || "").toLowerCase();
                const isAsr = typeLower.includes('asr') || typeLower.includes('trans');

                const progressPct = t.progress || 0;
                const stageText = normalizeStage({ ...t, status: normalizedStatus });
                const logContent = (t.logs || []).join('\n');
                const liveText = t.live_text || "Waiting for first transcription segment...";
                const hw = getHwIconAndLabel(t.unit_id);
                const isPausedForPriority = normalizedStatus === 'queued' && stageText.toLowerCase().includes('paused for priority task');
                const queueHint = isPausedForPriority
                    ? 'Paused for priority detect-language tasks...'
                    : 'Waiting for available hardware unit...';

                if (card.innerHTML.trim() === '') {
                    card.innerHTML = `
                        <div class="task-header">
                            <div class="task-icon-container" style="background:var(--md-sys-color-primary-container); color:var(--md-sys-color-primary);">
                                <span class="material-icons-sharp">${icon}</span>
                            </div>
                            <div class="item-info">
                                <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                                    <span class="item-primary">${t.filename}</span>
                                    <span class="meta-tag" style="background:var(--md-sys-color-secondary-container); color:var(--md-sys-color-on-secondary-container); border:none; padding:1px 8px; border-radius:100px; font-weight:700;">
                                        <span class="material-icons-sharp ${normalizedStatus==='active'?'pulse':''}" style="font-size:13px">${normalizedStatus==='active'?'sync':(normalizedStatus==='initializing'?'hourglass_top':'hourglass_empty')}</span> ${t.type || 'Task'}
                                    </span>
                                </div>
                                <div class="item-secondary">
                                    <span class="meta-tag" style="color:var(--md-sys-color-primary); font-weight:600;"><span class="material-icons-sharp" style="font-size:12px">layers</span><span class="stage-text">${stageText}</span></span>
                                    <span class="meta-tag"><span class="material-icons-sharp" style="font-size:12px">movie</span>${formatDur(t.video_duration)}</span>
                                    <span class="meta-tag"><span class="material-icons-sharp" style="font-size:12px">timer</span><span class="timer-text">${getTimerText(t, now)}</span></span>
                                    <span class="meta-tag speed-tag" style="display: none;"><span class="material-icons-sharp" style="font-size:12px">speed</span>Speed: <span class="speed-text">0.0x</span></span>
                                    <span class="meta-tag eta-tag" style="display: none;"><span class="material-icons-sharp" style="font-size:12px">schedule</span>ETA: <span class="eta-text">00:00:00</span></span>
                                    <span class="meta-tag hw-tag"><span class="material-icons-sharp hw-icon" style="font-size:12px;color:var(--md-sys-color-primary)">${hw.icon}</span><span class="hw-text">${hw.label}</span></span>
                                </div>
                            </div>
                            <span class="badge badge-${normalizedStatus}">${normalizedStatus === 'queued' ? 'queue' : normalizedStatus}</span>
                        </div>
                        <div style="display:flex;flex-direction:column;gap:4px;">
                            <div class="progress-container">
                                <div class="progress-bar ${pulseClass}" style="width:${progressPct}%"></div>
                            </div>
                            <div style="font-size:10px; color:var(--md-sys-color-secondary); text-align:right; font-weight:600;"><span class="progress-text">${progressPct}</span>%</div>
                        </div>
                        ${renderAuditDetails(t)}
                        ${isAsr ? `
                            <div style="margin-top:8px;">
                                <details ${expandedElements.has(`${id}_live`) ? 'open' : ''} ontoggle="handleToggle('${id}_live', this.open)">
                                    <summary style="font-size:11px; font-weight:700; color:var(--md-sys-color-primary); display:flex; align-items:center; gap:4px;">
                                        <span class="material-icons-sharp pulse" style="font-size:14px; color:var(--md-sys-color-error)">radio_button_checked</span> LIVE SRT STREAM
                                    </summary>
                                    <div class="result-box live-text-box" data-task-id="${id}" style="margin-top:4px; border: 1px dashed var(--md-sys-color-primary);">${escapeHtml(liveText)}</div>
                                </details>
                            </div>` : ''}
                        <details ${expandedElements.has(`${id}_logs`) ? 'open' : ''} ontoggle="handleToggle('${id}_logs', this.open)">
                            <summary><span class="material-icons-sharp">terminal</span> Real-time Logs</summary>
                            <div class="log-buffer">${escapeHtml(logContent)}</div>
                        </details>
                        <div class="hw-wait-msg" style="font-size:11px; color:var(--md-sys-color-warning); font-style:italic; margin-top:8px; display:${normalizedStatus==='queued'?'flex':'none'}; align-items:center; gap:4px;">
                            <span class="material-icons-sharp" style="font-size:14px">hourglass_empty</span> ${queueHint}
                        </div>
                    `;
                } else {
                    // Update Status Icon and Badge
                    const statusIcon = card.querySelector('.item-info .material-icons-sharp');
                    if (statusIcon) {
                        statusIcon.innerText = normalizedStatus==='active'?'sync':(normalizedStatus==='initializing'?'hourglass_top':'hourglass_empty');
                        statusIcon.className = `material-icons-sharp ${normalizedStatus==='active'?'pulse':''}`;
                    }
                    const statusBadge = card.querySelector('.badge');
                    if (statusBadge) {
                        statusBadge.innerText = normalizedStatus === 'queued' ? 'queue' : normalizedStatus;
                        statusBadge.className = `badge badge-${normalizedStatus}`;
                    }

                    const hwWait = card.querySelector('.hw-wait-msg');
                    if (hwWait) {
                        hwWait.style.display = normalizedStatus === 'queued' ? 'flex' : 'none';
                        const waitHint = normalizedStatus === 'queued' && stageText.toLowerCase().includes('paused for priority task')
                            ? 'Paused for priority detect-language tasks...'
                            : 'Waiting for available hardware unit...';
                        hwWait.innerHTML = `<span class="material-icons-sharp" style="font-size:14px">hourglass_empty</span> ${waitHint}`;
                    }

                     card.querySelector('.stage-text').innerText = stageText;
                     card.querySelector('.timer-text').innerText = getTimerText(t, now);
                     card.querySelector('.progress-bar').style.width = progressPct + '%';
                     card.querySelector('.progress-text').innerText = progressPct;
                     
                      const hwIconEl = card.querySelector('.hw-icon');
                      const hwTextEl = card.querySelector('.hw-text');
                      if (hwIconEl && hwTextEl) {
                          hwIconEl.innerText = hw.icon;
                          hwTextEl.innerText = hw.label;
                      }

                      const speedTag = card.querySelector('.speed-tag');
                      const speedText = card.querySelector('.speed-text');
                      const etaTag = card.querySelector('.eta-tag');
                      const etaText = card.querySelector('.eta-text');

                      const startActive = t.start_active || t.start_time || now;
                      const elapsedActive = now - startActive;
                      const processedDuration = t.current_position || ((progressPct / 100) * t.video_duration);

                      let calculatedSpeed = 0;
                      let remainingSeconds = 0;

                      if (t.video_duration > 0) {
                          const isUvr = t.type === 'Isolation' || (stageText.toLowerCase().includes('vocal') || stageText.toLowerCase().includes('separation') || stageText.toLowerCase().includes('uvr'));
                          
                          // Use helper function for speed/ETA calculation
                          const result = calculateTaskSpeedAndEta(t, now, data.history, isUvr);
                          calculatedSpeed = result.calculatedSpeed;
                          remainingSeconds = result.remainingSeconds;
                      }

                      if (speedTag && speedText) {
                          if (elapsedActive > 5 && calculatedSpeed > 0) {
                              speedText.textContent = calculatedSpeed.toFixed(1) + 'x';
                              speedTag.style.display = 'inline-flex';
                          } else {
                              speedTag.style.display = 'none';
                          }
                      } else if (speedTag) {
                          speedTag.style.display = 'none';
                      }

                      if (etaTag && etaText) {
                          if (elapsedActive > 5 && remainingSeconds > 0) {
                              etaText.textContent = formatDur(remainingSeconds);
                              etaTag.style.display = 'inline-flex';
                          } else {
                              etaTag.style.display = 'none';
                          }
                      } else if (etaTag) {
                          etaTag.style.display = 'none';
                      }

                     const lb = card.querySelector('.log-buffer');
                    if (lb && lb.textContent !== logContent) {
                        lb.textContent = logContent;
                        lb.scrollTop = lb.scrollHeight;
                    }
                    const ltb = card.querySelector('.live-text-box');
                    if (ltb && ltb.textContent !== liveText) {
                        ltb.textContent = liveText;
                        ltb.scrollTop = ltb.scrollHeight;
                    }
                }
            });

            const activeIds = new Set(tasks.map(t => t.task_id || t.filename));
            tList.querySelectorAll('.task-card').forEach(card => {
                if (!activeIds.has(card.dataset.taskId)) {
                    card.remove();
                }
            });
        }


        document.getElementById('last-update').innerText = `Updated: ${new Date().toLocaleTimeString()}`;
    } catch (e) { console.error(e); }
}
window.onload = () => {
    updateStats();
    setInterval(updateStats, 2000);
    showTab('active');
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
            if (currentTab === 'charts') {
                renderCharts();
            }
        });
    }
};
