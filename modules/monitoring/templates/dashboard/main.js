function showTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');
    ['active', 'history', 'analytics', 'charts', 'settings'].forEach(s => {
        document.getElementById(`${s}-section`).style.display = (s === tab ? 'block' : 'none');
    });

    if (tab === 'charts') {
        // Clear cached chart state to force redraw when section becomes visible.
        for (let id in lastChartStates) {
            lastChartStates[id].rangeMs = null;
            lastChartStates[id].theme = null;
        }
        renderCharts();
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 50);
    }
    if (tab === 'history') {
        fetch('/history', { headers: getAuthHeaders() }).then(res => res.json()).then(data => {
            fullTaskHistory = data || [];
            renderHistory();
        });
    }
    if (tab === 'settings') {
        loadDashboardApiKeys();
    }
}

function handleToggle(id, isOpen, detailsEl) {
    if (isOpen) {
        expandedElements.add(id);
        setTimeout(() => {
            const taskCard = detailsEl ? detailsEl.closest('.task-card') : null;
            if (!taskCard) {
                return;
            }
            const el = taskCard.querySelector('.log-buffer, .live-text-box');
            if (el) {
                el.scrollTop = el.scrollHeight;
            }
        }, 50);
    } else {
        expandedElements.delete(id);
    }
}

function bindToggleHandlers(root = document) {
    root.querySelectorAll('details.js-toggle').forEach((detailsEl) => {
        if (detailsEl.dataset.toggleBound === '1') {
            return;
        }
        detailsEl.dataset.toggleBound = '1';
        detailsEl.addEventListener('toggle', () => {
            const toggleId = detailsEl.dataset.toggleId;
            if (!toggleId) {
                return;
            }
            handleToggle(toggleId, detailsEl.open, detailsEl);
        });
    });
}

function startRefreshInterval() {
    if (typeof refreshTimer !== 'undefined' && refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
    }
    if (refreshEnabled && typeof currentRefreshInterval !== 'undefined') {
        refreshTimer = setInterval(updateStats, currentRefreshInterval);
    }
}

function changeRefreshInterval(val) {
    currentRefreshInterval = parseInt(val, 10);
    startRefreshInterval();
    renderCharts();
}

function handleTelemetryPurgeSuccess() {
    alert('Telemetry history purged successfully.');
    rollingTelemetryBuffer = [];
    if (typeof globalThis.resetTelemetryChartsAndStats === 'function') {
        globalThis.resetTelemetryChartsAndStats();
    }
    renderCharts();
}

async function saveSettings() {
    persistDashboardApiKeys();
    const telemetryHours = document.getElementById('retention-range').value;
    const logDays = document.getElementById('log-retention-range').value;
    try {
        const response = await fetch('/settings', {
            method: 'POST',
            headers: getAuthHeaders('application/json', true),
            body: JSON.stringify({
                telemetry_retention_hours: parseInt(telemetryHours, 10),
                log_retention_days: parseInt(logDays, 10)
            })
        });
        if (!response.ok) {
            throw new Error(`Failed to save settings (${response.status})`);
        }
        alert('Configuration saved!');
    } catch (e) { alert('Failed to save settings: ' + e); }
}

async function clearTaskHistory() {
    if (!confirm('Are you sure you want to permanently clear all task history? This cannot be undone.')) {
        return;
    }
    try {
        const res = await fetch('/system/history/clear', {
            method: 'POST',
            headers: getAuthHeaders(null, true)
        });
        if (res.ok) {
            alert('Task history purged successfully.');
            fullTaskHistory = [];
            renderHistory();
        } else {
            alert('Failed to clear task history.');
        }
    } catch (e) { alert('Error: ' + e); }
}

async function clearTelemetryMetrics() {
    if (!confirm('Are you sure you want to permanently clear all telemetry metrics? This cannot be undone.')) {
        return;
    }
    try {
        const res = await fetch('/system/telemetry/clear', {
            method: 'POST',
            headers: getAuthHeaders(null, true)
        });
        if (res.ok) {
            handleTelemetryPurgeSuccess();
        } else {
            alert('Failed to clear telemetry metrics.');
        }
    } catch (e) { alert('Error: ' + e); }
}

window.onload = () => {
    loadDashboardApiKeys();
    updateStats();
    startRefreshInterval();
    showTab('active');
    bindToggleHandlers();
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
            if (currentTab === 'charts') {
                renderCharts();
            }
        });
    }
};