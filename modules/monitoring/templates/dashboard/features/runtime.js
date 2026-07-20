let latestStatusRequestSeq = 0;

async function updateStats() {
    if (!refreshEnabled) return;
    const requestSeq = ++latestStatusRequestSeq;
    try {
        const data = await _fetchStatusData();
        if (requestSeq !== latestStatusRequestSeq) {
            return;
        }
        if (!_hasValidStatusData(data)) {
            return;
        }
        const now = Date.now() / 1000;
        _renderTopStats(data);
        _renderQueueCounters(data);
        _updateTelemetryState(data, now);
        const historicalSpeeds = _refreshHistoryAndTabs(data);
        _autoScrollTaskBuffers();
        _renderAnalyticsGrid(data);
        _renderHardwarePool(data);
        _renderActiveTaskList(data, now, historicalSpeeds);
        bindToggleHandlers();
        _cleanupTimelineForTasks(data.tasks || []);
        _renderLastUpdate();
    } catch (e) { console.error(e); }
}

async function _fetchStatusData() {
    const response = await fetch('/status');
    return response.json();
}

function _hasValidStatusData(data) {
    if (data && data.system) {
        return true;
    }
    console.warn('Incomplete status data received');
    return false;
}

function _renderTopStats(data) {
    document.getElementById('app-version').innerText = `Version ${data.version}`;
    document.getElementById('app-cpu-val').innerText = `${data.system.app_cpu_percent}%`;
    document.getElementById('app-cpu-bar').style.width = `${Math.min(100, data.system.app_cpu_percent)}%`;
    document.getElementById('sys-cpu-val').innerText = `${Math.round(data.system.cpu_percent)}%`;
    document.getElementById('sys-cpu-bar').style.width = `${data.system.cpu_percent}%`;
    document.getElementById('app-mem-val').innerText = `${data.system.app_memory_gb.toFixed(2)} GB`;
    document.getElementById('app-mem-bar').style.width = `${(data.system.app_memory_gb / data.system.memory_total_gb) * 100}%`;
    document.getElementById('sys-mem-val').innerText = `${data.system.memory_used_gb.toFixed(2)} / ${data.system.memory_total_gb.toFixed(2)} GB`;
    document.getElementById('sys-mem-bar').style.width = `${data.system.memory_percent}%`;
}

function _renderQueueCounters(data) {
    document.getElementById('active-val').innerText = data.active_sessions || 0;
    document.getElementById('queued-val').innerText = data.queued_sessions || 0;
}

function _updateTelemetryState(data, nowSec) {
    lastStatusData = data;
    currentTelemetry = data.telemetry_history || [];
    rollingTelemetryBuffer.push({
        timestamp: nowSec,
        system: data.system,
        telemetry: data.telemetry
    });
    rollingTelemetryBuffer = rollingTelemetryBuffer.filter((h) => h.timestamp >= nowSec - 600);
    _prepopulateTelemetryHistory(data, nowSec);
}

function _prepopulateTelemetryHistory(data, nowSec) {
    if (rollingTelemetryBuffer.length !== 1) {
        return;
    }
    if (!data.telemetry_history || data.telemetry_history.length === 0) {
        return;
    }
    const serverHist = data.telemetry_history.filter((h) => h.timestamp >= nowSec - 600 && h.timestamp < nowSec);
    rollingTelemetryBuffer = [...serverHist, ...rollingTelemetryBuffer];
}

function _refreshHistoryAndTabs(data) {
    fullTaskHistory = data.history || [];
    const historicalSpeeds = calculateHistoricalSpeeds(data.history);
    if (currentTab === 'charts') {
        renderCharts();
    }
    if (currentTab === 'history') {
        renderHistory();
    }
    return historicalSpeeds;
}

function _autoScrollTaskBuffers() {
    setTimeout(() => {
        document.querySelectorAll('#task-list .live-text-box, #task-list .log-buffer').forEach((box) => {
            box.scrollTop = box.scrollHeight;
        });
    }, 100);
}

function _renderAnalyticsGrid(data) {
    if (!data.history_stats) {
        return;
    }
    document.getElementById('analytics-grid').innerHTML = `
                <div class="stat-box"><div class="stat-label">Today</div><div class="stat-value">${formatDDHHMMSS(data.history_stats.today)}</div></div>
                <div class="stat-box"><div class="stat-label">Tasks Today</div><div class="stat-value">${data.history_stats.count_today}</div></div>
                <div class="stat-box"><div class="stat-label">This Month</div><div class="stat-value">${formatDDHHMMSS(data.history_stats.this_month)}</div></div>
                <div class="stat-box"><div class="stat-label">All Time</div><div class="stat-value">${formatDDHHMMSS(data.history_stats.all_time)}</div></div>
                <div class="stat-box"><div class="stat-label">Total Tasks</div><div class="stat-value">${data.history_stats.count_all_time}</div></div>
                <div class="stat-box"><div class="stat-label">Service Uptime</div><div class="stat-value">${formatDDHHMMSS(data.uptime_sec)}</div></div>
            `;
}

function _renderHardwarePool(data) {
    const units = data.hardware_units || [];
    document.getElementById('hw-pool').innerHTML = units.map((unit) => _renderHardwareCard(unit, data)).join('');
}

function _renderHardwareCard(unit, data) {
    const usage = _resolveHardwareUsage(unit, data);
    const statusText = usage.isUsed ? 'Used' : 'Not used';
    const statusClass = usage.isUsed ? 'status-used' : 'status-idle';
    const safeType = escapeHtml(unit.type ?? 'Unknown');
    const safeName = escapeHtml(unit.name ?? 'Unnamed Unit');
    const uvrStatusRaw = String(unit.uvr_status ?? 'ready');
    const whisperStatusRaw = String(unit.whisper_status ?? 'ready');
    const uvrStatusClass = uvrStatusRaw.replace(/[^a-zA-Z0-9_-]/g, '').toLowerCase() || 'ready';
    const whisperStatusClass = whisperStatusRaw.replace(/[^a-zA-Z0-9_-]/g, '').toLowerCase() || 'ready';
    const uvrStatus = escapeHtml(uvrStatusRaw);
    const whisperStatus = escapeHtml(whisperStatusRaw);
    return `
                <div class="hw-card">
                    <div class="hw-card-title"><span class="material-icons-sharp" style="font-size:12px">${usage.icon}</span> ${safeType}</div>
                    <div style="font-size: 11px; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 2px;">${safeName}</div>
                    <div class="hw-card-status ${statusClass}" style="margin-bottom: 6px;">${statusText}</div>
                    <div style="display: flex; flex-direction: column; gap: 4px; align-items: flex-start;">
                        <span class="badge badge-${uvrStatusClass}" style="padding: 2px 6px; font-size: 9px; border-radius: 4px; line-height: 1.2;">UVR: ${uvrStatus}</span>
                        <span class="badge badge-${whisperStatusClass}" style="padding: 2px 6px; font-size: 9px; border-radius: 4px; line-height: 1.2;">Whisper: ${whisperStatus}</span>
                    </div>
                </div>
            `;
}

function _resolveHardwareUsage(unit, data) {
    const telemetry = data.telemetry || {};
    const tasks = data.tasks || [];
    const usedFromTask = _isUnitUsedByActiveTask(unit.id, tasks);
    const usedWithUtil = _applyHardwareUtil(usedFromTask, unit.id, telemetry);
    const kind = _hardwareKind(unit);
    return _hardwareKindVisual(kind, unit, usedWithUtil, telemetry, tasks);
}

function _isUnitUsedByActiveTask(unitId, tasks) {
    return tasks.some((task) => String(task.unit_id) === String(unitId) && task.status === 'active');
}

function _applyHardwareUtil(isUsed, unitId, telemetry) {
    if (telemetry.hardware_util && telemetry.hardware_util[unitId] !== undefined && !isUsed) {
        return telemetry.hardware_util[unitId] > 0;
    }
    return isUsed;
}

function _hardwareKind(unit) {
    const fixedKinds = {
        NPU: 'npu',
        CPU: 'cpu',
        CUDA: 'cuda',
        AMD: 'amd'
    };
    if (fixedKinds[unit.type]) {
        return fixedKinds[unit.type];
    }
    return _gpuOrOtherHardwareKind(unit);
}

function _gpuOrOtherHardwareKind(unit) {
    if (unit.type !== 'GPU') {
        return 'other';
    }
    return _isNvidiaGpu(unit) ? 'cuda' : 'intel-gpu';
}

function _isNvidiaGpu(unit) {
    return String(unit.name || '').includes('NVIDIA');
}

function _hardwareKindVisual(kind, unit, isUsed, telemetry, tasks) {
    if (kind === 'intel-gpu') return _intelGpuVisual(isUsed, telemetry);
    if (kind === 'npu') return _npuVisual(isUsed, telemetry);
    if (kind === 'cuda') return _cudaVisual(unit, isUsed, telemetry);
    if (kind === 'amd') return _amdVisual(isUsed, telemetry);
    if (kind === 'cpu') return _cpuVisual(isUsed, tasks);
    return { icon: 'memory', isUsed };
}

function _amdVisual(isUsed, telemetry) {
    return { icon: 'bolt', isUsed };
}

function _intelGpuVisual(isUsed, telemetry) {
    const inferredUsed = !isUsed && telemetry.hardware_util === undefined ? (telemetry.intel_gpu_load || 0) > 0 : isUsed;
    return { icon: 'developer_board', isUsed: inferredUsed };
}

function _npuVisual(isUsed, telemetry) {
    const inferredUsed = !isUsed && telemetry.hardware_util === undefined ? (telemetry.npu_load || 0) > 0 : isUsed;
    return { icon: 'psychology_alt', isUsed: inferredUsed };
}

function _cudaVisual(unit, isUsed, telemetry) {
    const idx = parseInt(String(unit.id || '').split(':')[1] || 0, 10);
    const inferredUsed = !isUsed && telemetry.hardware_util === undefined ? _isCudaUtilUsed(telemetry, idx) : isUsed;
    return { icon: 'rocket_launch', isUsed: inferredUsed };
}

function _isCudaUtilUsed(telemetry, idx) {
    if (!telemetry.nvidia || !telemetry.nvidia[idx]) {
        return false;
    }
    return telemetry.nvidia[idx].util > 0;
}

function _cpuVisual(isUsed, tasks) {
    const inferredUsed = isUsed ? true : tasks.some((task) => task.unit_id === 'CPU' && task.status === 'active');
    return { icon: 'settings_input_component', isUsed: inferredUsed };
}

function _renderLastUpdate() {
    document.getElementById('last-update').innerText = `Updated: ${new Date().toLocaleTimeString()}`;
}