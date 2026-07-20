function getHwIconAndLabel(unitId) {
    if (!unitId) return { icon: 'hourglass_empty', label: 'Queued' };
    const uid = unitId.toString().toLowerCase();
    const prefixMatch = _resolveHwPrefixMatch(uid, unitId);
    if (prefixMatch) {
        return prefixMatch;
    }
    if (uid === 'cpu') {
        return { icon: 'settings_input_component', label: 'Host CPU' };
    }
    return { icon: 'memory', label: unitId };
}

function _resolveHwPrefixMatch(uid, unitId) {
    const mapping = [
        ['cuda', 'rocket_launch', 'NVIDIA GPU'],
        ['npu', 'psychology_alt', 'Intel NPU'],
        ['gpu', 'developer_board', 'Intel GPU'],
        ['amd', 'bolt', 'AMD GPU']
    ];
    const match = mapping.find(([prefix]) => uid.startsWith(prefix));
    if (!match) {
        return null;
    }
    const [family, icon, label] = match;
    if (_shouldAppendHardwareSlot(family) && _isConcreteHardwareSlotId(unitId)) {
        return { icon: icon, label: `${label} (${unitId})` };
    }
    return { icon: icon, label: label };
}

function _isConcreteHardwareSlotId(unitId) {
    const token = String(unitId || '').toUpperCase();
    return /[.:]\d+$/.test(token);
}

function _shouldAppendHardwareSlot(family) {
    return _countHardwareUnitsByFamily(family) > 1;
}

function _countHardwareUnitsByFamily(family) {
    const data = _statusDataForHardwareLabeling();
    const units = (data && Array.isArray(data.hardware_units)) ? data.hardware_units : [];
    return units.filter((unit) => _normalizeHardwareFamily(unit) === family).length;
}

function _statusDataForHardwareLabeling() {
    if (typeof lastStatusData !== 'undefined' && lastStatusData) {
        return lastStatusData;
    }
    if (globalThis && globalThis.lastStatusData) {
        return globalThis.lastStatusData;
    }
    return null;
}

function _normalizeHardwareFamily(unit) {
    const id = String((unit && unit.id) || '').toLowerCase();
    const type = String((unit && unit.type) || '').toLowerCase();
    const source = id || type;
    if (source.startsWith('cuda')) return 'cuda';
    if (source.startsWith('npu')) return 'npu';
    if (source.startsWith('gpu')) return 'gpu';
    if (source.startsWith('amd')) return 'amd';
    return source;
}

function escapeHtml(text) {
    if (!text) return "";
    return text.toString()
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function downloadSrt(filename, content) {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename.replace(/\.[^/.]+$/, "") + ".srt";
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function downloadSrtById(id) {
    const task = fullTaskHistory.find(t => (t.task_id || t.filename) === id);
    if (task && task.result && task.result.text) {
        downloadSrt(task.filename, task.result.text);
    } else {
        alert("Transcription content not found in history.");
    }
}

function formatDur(sec) {
    if (!_isValidDuration(sec)) return "00:00:00";
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return `${_pad2(h)}:${_pad2(m)}:${_pad2(s)}`;
}

function _isValidDuration(sec) {
    return sec !== undefined && sec !== null && sec >= 0;
}

function _pad2(value) {
    return String(value).padStart(2, '0');
}

function formatDDHHMMSS(sec) {
    if (sec === undefined || sec === null || sec < 0) return "0d 0h 0m";
    const d = Math.floor(sec / 86400);
    const h = Math.floor((sec % 86400) / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${d}d ${h}h ${m}m`;
}

function getTimerText(t, now) {
    const startTime = t.start_time ?? now;
    const startActive = t.start_active ?? startTime;
    if (t.status === 'queued') {
        return 'Queued for: ' + formatDur(now - startTime);
    }
    const activeDur = formatDur(now - startActive);
    const queueDur = formatDur(startActive - startTime);
    return 'Running: ' + activeDur + ' (Queue: ' + queueDur + ')';
}
