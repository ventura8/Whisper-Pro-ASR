function getHwIconAndLabel(unitId) {
    if (!unitId) return { icon: 'hourglass_empty', label: 'Queued' };
    const uid = unitId.toString().toLowerCase();
    if (uid.startsWith('cuda')) {
        return { icon: 'rocket_launch', label: 'NVIDIA GPU (' + unitId + ')' };
    } else if (uid.startsWith('npu')) {
        return { icon: 'psychology_alt', label: 'Intel NPU (' + unitId + ')' };
    } else if (uid.startsWith('gpu')) {
        return { icon: 'developer_board', label: 'Intel GPU (' + unitId + ')' };
    } else if (uid === 'cpu') {
        return { icon: 'settings_input_component', label: 'Host CPU' };
    }
    return { icon: 'memory', label: unitId };
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
    if (sec === undefined || sec === null || sec < 0) return "00:00:00";
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return (h < 10 ? "0" + h : h) + ":" + (m < 10 ? "0" + m : m) + ":" + (s < 10 ? "0" + s : s);
}

function formatDDHHMMSS(sec) {
    if (sec === undefined || sec === null || sec < 0) return "0d 0h 0m";
    const d = Math.floor(sec / 86400);
    const h = Math.floor((sec % 86400) / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${d}d ${h}h ${m}m`;
}

function getTimerText(t, now) {
    const startTime = t.start_time || now;
    const startActive = t.start_active || startTime;
    if (t.status === 'queued') {
        return 'Queued for: ' + formatDur(now - startTime);
    }
    const activeDur = formatDur(now - startActive);
    const queueDur = formatDur(startActive - startTime);
    return 'Running: ' + activeDur + ' (Queue: ' + queueDur + ')';
}
