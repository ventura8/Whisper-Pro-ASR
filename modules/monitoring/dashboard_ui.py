# pylint: disable=too-many-lines
"""
Dashboard UI Components
"""


def get_dashboard_html():
    """Returns the rendered HTML for the monitoring dashboard."""
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Pro Dashboard</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><defs><linearGradient id='g' x1='0%25' y1='0%25' x2='100%25' y2='100%25'><stop offset='0%25' style='stop-color:%2300d2ff;stop-opacity:1' /><stop offset='100%25' style='stop-color:%233a7bd5;stop-opacity:1' /></linearGradient></defs><path d='M6 10 L11 22 L16 12 L21 22 L26 10' fill='none' stroke='url(%23g)' stroke-width='3.5' stroke-linecap='round' stroke-linejoin='round'/><path d='M4 16 L6 16 M26 16 L28 16' fill='none' stroke='url(%23g)' stroke-width='2' stroke-linecap='round'/></svg>">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Roboto:wght@300;400;500;700&family=Roboto+Mono&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons+Sharp" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        :root {
            --md-sys-color-primary: #006495;
            --md-sys-color-on-primary: #ffffff;
            --md-sys-color-primary-container: #cbe6ff;
            --md-sys-color-secondary: #50606e;
            --md-sys-color-surface: #fdfcff;
            --md-sys-color-surface-variant: #dee3eb;
            --md-sys-color-outline: #72777f;
            --md-sys-color-background: #fdfcff;
            --md-sys-color-on-surface: #191c1e;
            --md-sys-color-success: #2e7d32;
            --md-sys-color-error: #ba1a1a;
            --md-sys-color-warning: #e65100;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Roboto', sans-serif;
            background-color: var(--md-sys-color-background);
            color: var(--md-sys-color-on-surface);
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            max-width: 1600px;
            margin: 0 auto;
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .header-title { display: flex; align-items: center; gap: 16px; }
        .header-actions { display: flex; align-items: center; gap: 16px; }

        h1 { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 28px; }

        .btn-swagger {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--md-sys-color-primary);
            color: white;
            text-decoration: none;
            border-radius: 100px;
            font-size: 14px;
            font-weight: 500;
            transition: opacity 0.2s;
            border: none;
            cursor: pointer;
        }
        .btn-swagger:hover { opacity: 0.9; }

        .btn-download {
            background: #f8fafc;
            color: var(--md-sys-color-primary);
            border: 1px solid var(--md-sys-color-primary);
        }
        .btn-download:hover { background: #e0f2fe; }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }

        .card {
            background: #ffffff;
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--md-sys-color-surface-variant);
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .card-title {
            font-size: 11px;
            color: var(--md-sys-color-secondary);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .card-value { font-size: 20px; font-family: 'Outfit', sans-serif; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

        .progress-container {
            height: 6px;
            background: var(--md-sys-color-surface-variant);
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--md-sys-color-primary), #00a8e8);
            transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .progress-bar.pulse {
            animation: pulse 2s infinite linear;
            background-size: 200% 100%;
            background-image: linear-gradient(90deg, var(--md-sys-color-primary) 0%, #00a8e8 50%, var(--md-sys-color-primary) 100%);
        }

        @keyframes pulse {
            0% { background-position: 100% 0%; }
            100% { background-position: -100% 0%; }
        }

        .dashboard-main {
            display: grid;
            grid-template-columns: 3fr 1fr;
            gap: 24px;
        }

        /* Tabs Styling */
        .tabs {
            display: flex;
            gap: 16px;
            border-bottom: 1px solid var(--md-sys-color-surface-variant);
            margin-bottom: 16px;
        }

        .tab {
            padding: 12px 24px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            color: var(--md-sys-color-secondary);
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
        }

        .tab.active {
            color: var(--md-sys-color-primary);
            border-bottom-color: var(--md-sys-color-primary);
        }

        .section-title {
            font-size: 18px;
            font-family: 'Outfit', sans-serif;
            font-weight: 500;
            margin-bottom: 12px;
            display: flex; align-items: center; gap: 8px;
        }

        .task-list, .engine-list, .info-list, .history-list, .stats-list, .charts-list {
            display: flex; flex-direction: column; gap: 16px;
        }

        .task-card, .history-card, .chart-card {
            padding: 20px;
            background: white;
            border: 1px solid var(--md-sys-color-surface-variant);
            border-radius: 20px;
            display: flex; flex-direction: column; gap: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            transition: transform 0.2s;
        }

        .task-card.queued {
            border-left: 4px solid var(--md-sys-color-warning);
            opacity: 0.85;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .time-controls {
            display: flex;
            gap: 8px;
        }

        .btn-time {
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 700;
            border-radius: 8px;
            border: 1px solid var(--md-sys-color-outline);
            background: white;
            color: var(--md-sys-color-secondary);
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-time.active {
            background: var(--md-sys-color-primary);
            color: white;
            border-color: var(--md-sys-color-primary);
        }

        .empty-state {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 60px 40px; background: white; border: 1px dashed var(--md-sys-color-outline);
            border-radius: 24px; color: var(--md-sys-color-secondary); text-align: center; gap: 16px;
        }

        .empty-icon { font-size: 48px; color: var(--md-sys-color-surface-variant); }

        .task-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
        .task-icon-container {
            width: 44px; height: 44px; border-radius: 12px;
            background: var(--md-sys-color-primary-container); color: var(--md-sys-color-primary);
            display: flex; align-items: center; justify-content: center;
        }

        .item-info { display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 0; }
        .item-primary { font-weight: 600; font-size: 15px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .item-secondary { font-size: 12px; color: var(--md-sys-color-secondary); display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

        .meta-tag {
            background: #f1f3f4; padding: 2px 8px; border-radius: 4px; font-weight: 500;
            color: var(--md-sys-color-secondary); font-size: 11px; display: flex; align-items: center; gap: 4px;
        }

        .badge { padding: 4px 10px; border-radius: 100px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
        .badge-active { background: #e8f5e9; color: #1b5e20; }
        .badge-busy { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
        .badge-queued { background: #fff3e0; color: #e65100; border: 1px solid #ffccbc; }
        .badge-ready { background: #e3f2fd; color: #0d47a1; }
        .badge-initializing { background: rgba(0, 100, 149, 0.1); color: #006495; }
        .badge-initialized { background: #f0f9ff; color: #0369a1; }
        .badge-loaded { background: #ecfdf5; color: #047857; }
        .badge-failed { background: #fef2f2; color: #b91c1c; }

        .badge-lang {
            background: var(--md-sys-color-primary-container);
            color: var(--md-sys-color-primary);
            border: 1px solid var(--md-sys-color-primary);
        }

        .log-buffer, .json-buffer {
            padding: 14px; font-family: 'Roboto Mono', monospace; font-size: 11px; line-height: 1.6;
            color: #2d3748; white-space: pre-wrap; max-height: 400px; overflow-y: auto; background: #f8fafc; border-top: 1px solid #edf2f7;
        }

        .json-buffer { background: #1e293b; color: #f1f5f9; border: none; border-radius: 8px; margin-top: 8px; }

        .result-box {
            padding: 16px; background: #1e293b; border-radius: 12px; border: 1px solid #334155;
            font-size: 12px; line-height: 1.6; color: #f1f5f9; margin-top: 8px;
            font-family: 'Roboto Mono', monospace; white-space: pre-wrap;
            max-height: 500px; overflow-y: auto;
        }

        .list-item {
            padding: 12px 16px; background: white; border: 1px solid var(--md-sys-color-surface-variant);
            border-radius: 12px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
        }

        .hw-card {
            padding: 12px; background: #f8fafc; border-radius: 12px;
            border: 1px solid var(--md-sys-color-surface-variant);
            display: flex; flex-direction: column; gap: 4px;
        }
        .hw-card-title { font-size: 10px; font-weight: 700; color: var(--md-sys-color-secondary); text-transform: uppercase; display: flex; align-items: center; gap: 4px; }
        .hw-card-status { font-size: 13px; font-weight: 600; font-family: 'Outfit', sans-serif; }
        .status-used { color: #1b5e20; }
        .status-idle { color: #5f6368; }

        .refresh-indicator { font-size: 12px; color: var(--md-sys-color-outline); text-align: right; padding-top: 12px; }
        #history-section, #analytics-section, #charts-section, #settings-section { display: none; }

        .stat-grid-inner {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px;
        }
        .stat-box {
            padding: 16px; background: #f8fafc; border-radius: 16px; border: 1px solid #e2e8f0;
        }
        .stat-label { font-size: 11px; color: var(--md-sys-color-secondary); margin-bottom: 6px; font-weight: 600; text-transform: uppercase; }
        .stat-value { font-family: 'Outfit', sans-serif; font-size: 20px; font-weight: 500; }

        .chart-container { height: 350px; width: 100%; }

        summary { cursor: pointer; padding: 4px 0; font-size: 12px; font-weight: 600; color: var(--md-sys-color-primary); display: flex; align-items: center; gap: 4px; }
        summary:hover { opacity: 0.8; }

        @media (max-width: 1200px) {
            .dashboard-main { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <span class="material-icons-sharp" style="font-size: 32px; color: var(--md-sys-color-primary)">analytics</span>
            <h1 id="app-name">Whisper Pro Monitor</h1>
        </div>
        <div class="header-actions">
            <a href="/logs/download" class="btn-swagger btn-download">
                <span class="material-icons-sharp" style="font-size: 18px">download</span>
                Download Logs
            </a>
            <a href="/docs" class="btn-swagger">
                <span class="material-icons-sharp" style="font-size: 18px">api</span>
                Swagger UI
            </a>
            <button id="toggle-refresh" class="btn-swagger" onclick="toggleRefresh()" style="border:none; cursor:pointer; min-width: 130px;">
                <span class="material-icons-sharp" style="font-size: 18px" id="refresh-icon">sync</span>
                <span id="refresh-text">Live Refresh</span>
            </button>
            <div id="app-version" style="font-size: 14px; font-weight: 500; color: var(--md-sys-color-primary)">v--</div>
        </div>
    </header>

    <div class="stats-grid" id="top-grid">
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">terminal</span> App CPU</div>
            <div class="card-value" id="app-cpu-val">--%</div>
            <div class="progress-container"><div class="progress-bar" id="app-cpu-bar" style="width: 0%"></div></div>
        </div>
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">speed</span> System CPU</div>
            <div class="card-value" id="sys-cpu-val">--%</div>
            <div class="progress-container"><div class="progress-bar" id="sys-cpu-bar" style="width: 0%"></div></div>
        </div>
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">layers</span> App Memory</div>
            <div class="card-value" id="app-mem-val">-- GB</div>
            <div class="progress-container"><div class="progress-bar" id="app-mem-bar" style="width: 0%"></div></div>
        </div>
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">memory</span> System Memory</div>
            <div class="card-value" id="sys-mem-val">-- GB</div>
            <div class="progress-container"><div class="progress-bar" id="sys-mem-bar" style="width: 0%"></div></div>
        </div>
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">play_arrow</span> Active</div>
            <div class="card-value" id="active-val">0</div>
        </div>
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">pause</span> Queued</div>
            <div class="card-value" id="queued-val">0</div>
        </div>
    </div>

    <div class="dashboard-main">
        <div class="section">
            <div class="tabs">
                <div class="tab active" id="tab-active" onclick="showTab('active')">Active</div>
                <div class="tab" id="tab-history" onclick="showTab('history')">History</div>
                <div class="tab" id="tab-charts" onclick="showTab('charts')">Charts</div>
                <div class="tab" id="tab-analytics" onclick="showTab('analytics')">Analytics</div>
                <div class="tab" id="tab-settings" onclick="showTab('settings')">Settings</div>
            </div>

            <div id="active-section">
                <div class="task-list" id="task-list"></div>
            </div>

            <div id="history-section">
                <div class="history-list" id="history-list"></div>
            </div>

            <div id="charts-section">
                <div class="charts-list">
                    <div class="chart-header" style="margin-bottom: 8px;">
                        <span class="section-title" style="margin:0"><span class="material-icons-sharp">show_chart</span> Device Telemetry History</span>
                    </div>
                    <div class="chart-card">
                        <div class="section-title"><span class="material-icons-sharp">speed</span> CPU Utilization (App vs System)</div>
                        <div class="chart-container" id="cpuChart"></div>
                    </div>
                    <div class="chart-card">
                        <div class="section-title"><span class="material-icons-sharp">memory</span> Memory Allocation (GB)</div>
                        <div class="chart-container" id="memChart"></div>
                    </div>
                    <div class="chart-header">
                        <div class="card-title">Hardware Acceleration</div>
                    </div>
                    <div id="hwChart" class="chart-container"></div>
                </div>
            </div>

            <div id="analytics-section">
                <div class="stats-list">
                    <div class="card">
                        <div class="section-title"><span class="material-icons-sharp">bar_chart</span> Output Analytics</div>
                        <div class="stat-grid-inner" id="analytics-grid"></div>
                    </div>
                </div>
            </div>

            <div id="settings-section">
                <div class="card">
                    <div class="section-title"><span class="material-icons-sharp">settings</span> Service Configuration</div>
                    <div style="display:flex; flex-direction:column; gap:24px; padding: 10px;">
                        <div>
                            <label style="font-size:14px; font-weight:600; display:block; margin-bottom:8px;">Telemetry History Retention (Hours)</label>
                            <div style="display:flex; align-items:center; gap:16px;">
                                <input type="range" id="retention-range" min="1" max="168" value="24" style="flex:1;" oninput="document.getElementById('retention-label').innerText = this.value + 'h'">
                                <span id="retention-label" style="font-family:'Roboto Mono'; font-weight:700; color:var(--md-sys-color-primary); width:40px;">24h</span>
                            </div>
                        </div>
                        <div>
                            <label style="font-size:14px; font-weight:600; display:block; margin-bottom:8px;">System Log Retention (Days)</label>
                            <div style="display:flex; align-items:center; gap:16px;">
                                <input type="range" id="log-retention-range" min="1" max="30" value="7" style="flex:1;" oninput="document.getElementById('log-retention-label').innerText = this.value + 'd'">
                                <span id="log-retention-label" style="font-family:'Roboto Mono'; font-weight:700; color:var(--md-sys-color-primary); width:40px;">7d</span>
                            </div>
                        </div>
                        <div style="display:flex; justify-content:flex-end;">
                            <button onclick="saveSettings()" class="btn-swagger" style="border:none; cursor:pointer; padding: 10px 40px; font-size:15px;">Save Configuration</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title"><span class="material-icons-sharp">info</span> System Information</div>
            <div class="info-list">
                <div class="card" style="padding: 16px; gap: 12px;">
                    <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--md-sys-color-secondary);">Hardware Resource Pool</div>
                    <div id="hw-pool" style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;"></div>
                </div>
                <div id="hw-widgets" class="stats-list" style="grid-template-columns: 1fr; margin-top: 16px;"></div>
                <div class="engine-list" id="engine-list" style="margin-top: 16px;"></div>
            </div>
        </div>
    </div>

    <div class="refresh-indicator" id="last-update">Updating...</div>

    <script>
        const expandedElements = new Set();
        let currentTab = 'active';
        let charts = {};
        let currentTelemetry = [];
        let fullTaskHistory = [];

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

        const COLORS = [
            '#006495', '#2e7d32', '#e65100', '#d81b60', '#5e35b1',
            '#00acc1', '#fb8c00', '#43a047', '#3949ab', '#8e24aa'
        ];

        function showTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(`tab-${tab}`).classList.add('active');
            ['active', 'history', 'analytics', 'charts', 'settings'].forEach(s => {
                document.getElementById(`${s}-section`).style.display = (s === tab ? 'block' : 'none');
            });

            if (tab === 'charts') renderCharts();
            if (tab === 'history') {
                fetch('/history').then(res => res.json()).then(data => {
                    fullTaskHistory = data || [];
                    renderHistory();
                });
            }
        }


        function formatDur(sec) {
            if (sec === undefined || sec === null || sec < 0) return "00:00:00";
            const h = Math.floor(sec / 3600);
            const m = Math.floor((sec % 3600) / 60);
            const s = Math.floor(sec % 60);
            return (h < 10 ? "0" + h : h) + ":" + (m < 10 ? "0" + m : m) + ":" + (s < 10 ? "0" + s : s);
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

        function renderCharts(data) {
            if (!currentTelemetry || currentTelemetry.length === 0) return;
            const sorted = [...currentTelemetry].sort((a,b) => a.timestamp - b.timestamp);
            if (sorted.length === 0) return;

            const filtered = sorted;

            const labels = filtered.map(h => {
                const d = new Date(h.timestamp * 1000);
                return d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
            });

            createOrUpdateLineChart('cpuChart', labels, [
                { label: 'System CPU %', data: filtered.map(h => h.system ? h.system.cpu_percent : h.cpu_sys), color: COLORS[0] },
                { label: 'App CPU %', data: filtered.map(h => h.system ? h.system.app_cpu_percent : h.cpu_app), color: COLORS[3] }
            ], false);

            createOrUpdateLineChart('memChart', labels, [
                { label: 'App Memory (GB)', data: filtered.map(h => h.system ? h.system.app_memory_gb : h.mem_app_gb), color: COLORS[1] }
            ], false);

            const hwDatasets = [];
            const first = filtered[0];
            if (!first) return;
            const nvidiaData = (first.telemetry && first.telemetry.nvidia) ? first.telemetry.nvidia : first.nvidia_util;

            if (nvidiaData && Array.isArray(nvidiaData)) {
                nvidiaData.forEach((_, i) => {
                    hwDatasets.push({
                        label: `NVIDIA GPU ${i} %`,
                        data: filtered.map(h => {
                            const nv = (h.telemetry && h.telemetry.nvidia) ? h.telemetry.nvidia : h.nvidia_util;
                            return nv ? (nv[i]?.util !== undefined ? nv[i].util : nv[i]) : 0;
                        }),
                        color: COLORS[i + 2]
                    });
                });
            }

            const hasIntel = (data.hardware_units || []).some(u => u.type === 'GPU' && !u.name.includes('NVIDIA'));
            if (hasIntel) { hwDatasets.push({ label: 'Intel GPU %', data: filtered.map(h => h.telemetry?.intel_gpu_load || h.intel_util || 0), color: '#3949ab' }); }
            
            const hasNpu = (data.hardware_units || []).some(u => u.type === 'NPU');
            if (hasNpu) { hwDatasets.push({ label: 'NPU Load %', data: filtered.map(h => h.telemetry?.npu_load || h.npu_util || 0), color: '#8e24aa' }); }
            
            createOrUpdateLineChart('hwChart', labels, hwDatasets, true);
        }

        function createOrUpdateLineChart(id, labels, datasets, percent) {
            const el = document.getElementById(id);
            if (!el) return;
            
            // Re-map datasets to ApexCharts format
            let series = datasets.map(d => ({
                name: d.label,
                data: d.data
            }));
            
            // Ensure at least one series exists to prevent ApexCharts from failing to render
            if (series.length === 0) {
                series = [{ name: 'No Acceleration Detected', data: new Array(labels.length).fill(0) }];
            }

            const options = {
                series: series,
                chart: {
                    type: 'line',
                    height: 350,
                    toolbar: { show: false },
                    zoom: { enabled: false },
                    animations: { enabled: true, easing: 'easeinout', speed: 800 },
                    background: '#fff'
                },
                colors: datasets.map(d => d.color || '#006495'),
                dataLabels: { enabled: false },
                stroke: { curve: 'smooth', width: 3 },
                xaxis: {
                    categories: labels,
                    labels: {
                        style: { colors: '#72777f', fontSize: '10px' },
                        rotate: 0,
                        hideOverlappingLabels: true
                    },
                    axisBorder: { show: false },
                    axisTicks: { show: false }
                },
                yaxis: {
                    min: 0,
                    max: percent ? 100 : undefined,
                    labels: {
                        style: { colors: '#72777f', fontSize: '10px' },
                        formatter: (val) => percent ? val.toFixed(0) + '%' : val.toFixed(1)
                    }
                },
                grid: {
                    borderColor: '#f0f0f0',
                    strokeDashArray: 4,
                    xaxis: { lines: { show: false } }
                },
                legend: {
                    position: 'top',
                    horizontalAlign: 'right',
                    fontFamily: 'Outfit',
                    fontSize: '12px',
                    labels: { colors: '#50606e' },
                    markers: { radius: 12 }
                },
                tooltip: {
                    theme: 'light',
                    x: { show: true },
                    y: { formatter: (val) => percent ? val.toFixed(1) + '%' : val.toFixed(2) + ' GB' }
                }
            };

            if (charts[id]) {
                charts[id].updateOptions({
                    xaxis: { categories: labels },
                    series: series
                });
            } else {
                charts[id] = new ApexCharts(el, options);
                charts[id].render();
            }
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
                <div style="margin-top:12px; border-top:1px solid #eee; padding-top:12px;">
                    <details ${auditOpen} ontoggle="handleToggle('${id}_audit', this.open)">
                        <summary style="font-size:13px; color:var(--md-sys-color-secondary); margin-bottom:8px;">
                            <span class="material-icons-sharp" style="font-size:14px">policy</span> Audit & Caller Info
                        </summary>
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; padding-left:16px;">
                            <div class="item-secondary"><span class="material-icons-sharp" style="font-size:12px">language</span> IP: ${caller.ip || 'Local'}</div>
                            <div class="item-secondary" title="${caller.user_agent}"><span class="material-icons-sharp" style="font-size:12px">devices</span> UA: ${caller.user_agent ? (caller.user_agent.length > 30 ? caller.user_agent.substring(0, 30) + '...' : caller.user_agent) : 'Unknown'}</div>
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
            hList.innerHTML = fullTaskHistory.map((h, i) => {
                const id = h.task_id || `hist-${i}`;
                if (h.live_text) console.debug("[Dashboard] Live text received for:", id);
                const result = h.result || {};
                
                if (!result.text) {
                    console.warn("[Dashboard] History item missing result.text:", id, h);
                }
                const speed = (h.video_duration > 0 && h.total_elapsed_sec > 0) 
                    ? (h.video_duration / h.total_elapsed_sec).toFixed(1) + 'x'
                    : 'N/A';
                
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
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-top:12px;">
                        <details ${expandedElements.has(`${id}_trans`) ? 'open' : ''} style="flex:1;" ontoggle="handleToggle('${id}_trans', this.open)">
                            <summary><span class="material-icons-sharp">subtitles</span> View Transcription Result</summary>
                            ${contentHtml}
                        </details>
                        ${finalSrt ? `
                        <button class="btn-time" style="margin-left:8px; display:flex; align-items:center; gap:4px; padding:6px 12px;" onclick="downloadSrtById('${id}')">
                            <span class="material-icons-sharp" style="font-size:16px">download</span> SRT
                        </button>` : ''}
                    </div>` : '';

                const langCode = result.language || result.detected_language;
                const langBadge = langCode ? `<span class="badge badge-lang" style="margin-left:auto;">${langCode.toUpperCase()}</span>` : '';

                const typeIcon = (h.type === '/asr' || h.type === 'Transcription') ? 'record_voice_over' : 'translate';

                return `<div class="history-card">
                    <div class="task-header">
                        <div class="task-icon-container" style="background:var(--md-sys-color-primary-container); color:var(--md-sys-color-primary);">
                            <span class="material-icons-sharp">${typeIcon}</span>
                        </div>
                        <div class="item-info">
                            <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                                <span class="item-primary">${h.filename}</span>
                                <span class="meta-tag" style="background:#e8f5e9; color:#1b5e20; border:none; padding:1px 8px; border-radius:100px; font-weight:700;">
                                    <span class="material-icons-sharp" style="font-size:13px">check</span> ${h.type || 'Task'}
                                </span>
                            </div>
                            <div class="item-secondary">
                                <span class="meta-tag" title="Completed At"><span class="material-icons-sharp" style="font-size:12px;color:var(--md-sys-color-secondary)">schedule</span>${h.completed_at}</span>
                                <span class="meta-tag"><span class="material-icons-sharp" style="font-size:12px">movie</span>${formatDur(h.video_duration)}</span>
                                <span class="meta-tag" title="Processing Time"><span class="material-icons-sharp" style="font-size:12px">timer</span>Took: ${formatDur(h.total_elapsed_sec)}</span>
                                <span class="meta-tag" title="Transcription Speed"><span class="material-icons-sharp" style="font-size:12px">speed</span>Speed: ${speed}</span>
                            </div>
                        </div>
                        ${langBadge}
                        <span class="badge badge-active" style="margin-left:8px;">Finished</span>
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

        let refreshEnabled = true;
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
                btn.style.background = '#f1f3f4';
                btn.style.color = '#5f6368';
            }
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

                currentTelemetry = data.telemetry_history || [];
                fullTaskHistory = data.history || [];
                console.log("[Dashboard] Update: Active Tasks:", (data.tasks||[]).length, "History:", fullTaskHistory.length);

                if (currentTab === 'charts') renderCharts(data);
                if (currentTab === 'history') renderHistory();

                // Auto-scroll active live transcription and logs only
                setTimeout(() => {
                    document.querySelectorAll('#task-list .live-text-box, #task-list .log-buffer').forEach(box => {
                        box.scrollTop = box.scrollHeight;
                    });
                }, 100);

                const hwWidgets = document.getElementById('hw-widgets');
                hwWidgets.innerHTML = '';

                if (data.telemetry && data.telemetry.nvidia) {
                    data.telemetry.nvidia.forEach((gpu, i) => {
                        const isUsed = gpu.util > 0;
                        const card = document.createElement('div'); card.className = 'card';
                        card.innerHTML = `<div class="card-title"><span class="material-icons-sharp">rocket_launch</span> NV-GPU ${i}</div>
                                          <div class="card-value" style="color: ${isUsed ? '#1b5e20' : 'inherit'}">${isUsed ? 'Used' : 'Not used'}</div>
                                          <div class="progress-container"><div class="progress-bar" style="width: ${gpu.util}%"></div></div>
                                          <div class="item-secondary" style="font-size: 10px;">${gpu.mem_used}MB / ${gpu.mem_total}MB</div>`;
                        hwWidgets.appendChild(card);
                    });
                }
                if (data.history_stats) {
                    document.getElementById('analytics-grid').innerHTML = `
                        <div class="stat-box"><div class="stat-label">Today</div><div class="stat-value">${formatDur(data.history_stats.today)}</div></div>
                        <div class="stat-box"><div class="stat-label">Tasks Today</div><div class="stat-value">${data.history_stats.count_today}</div></div>
                        <div class="stat-box"><div class="stat-label">This Month</div><div class="stat-value">${formatDur(data.history_stats.this_month)}</div></div>
                        <div class="stat-box"><div class="stat-label">All Time</div><div class="stat-value">${formatDur(data.history_stats.all_time)}</div></div>
                        <div class="stat-box"><div class="stat-label">Total Tasks</div><div class="stat-value">${data.history_stats.count_all_time}</div></div>
                        <div class="stat-box"><div class="stat-label">Service Uptime</div><div class="stat-value">${formatDur(data.uptime_sec)}</div></div>
                    `;
                }
                const tel = data.telemetry || {};
                document.getElementById('hw-pool').innerHTML = (data.hardware_units || []).map(u => {
                    const unitId = u.id;
                    const tasks = data.tasks || [];
                    const tel = data.telemetry || {};
                    let isUsed = tasks.some(t => String(t.unit_id) === String(unitId) && t.status === 'active');
                    
                    if (isUsed) console.debug(`[Dashboard] Unit ${unitId} is BUSY with task`);
                    let icon = 'memory';

                    if (u.type === 'GPU' && !u.name.includes('NVIDIA')) {
                        if (!isUsed) isUsed = (tel.intel_gpu_load || 0) > 0;
                        icon = 'developer_board';
                    } else if (u.type === 'NPU') {
                        if (!isUsed) isUsed = (tel.npu_load || 0) > 0;
                        icon = 'psychology_alt';
                    } else if (u.type === 'CUDA' || (u.type === 'GPU' && u.name.includes('NVIDIA'))) {
                        const idx = parseInt(unitId.split(':')[1] || 0);
                        if (!isUsed && tel.nvidia && tel.nvidia[idx]) {
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
                            <div style="font-size: 11px; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${u.name}</div>
                            <div class="hw-card-status ${statusClass}">${statusText}</div>
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

                    tasks.sort((a,b) => (a.status==='active'?-1:1)).forEach(t => {
                        const id = t.task_id || t.filename;
                        let card = tList.querySelector(`[data-task-id="${id}"]`);

                        if (!card) {
                            card = document.createElement('div');
                            card.className = 'task-card';
                            card.dataset.taskId = id;
                            tList.appendChild(card);
                        }

                        if (t.status === 'queued') {
                            card.classList.add('queued');
                        } else {
                            card.classList.remove('queued');
                        }

                        const icon = (t.type === '/asr' || t.type === 'Transcription' || t.type === 'ASR') ? 'record_voice_over' : 'translate';
                        const pulseClass = t.status === 'active' ? 'pulse' : '';

                        const typeLower = (t.type || "").toLowerCase();
                        const isAsr = typeLower.includes('asr') || typeLower.includes('trans');

                        const progressPct = t.progress || 0;
                        const logContent = (t.logs || []).join('\n');
                        const liveText = t.live_text || "Waiting for first transcription segment...";

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
                                                <span class="material-icons-sharp ${t.status==='active'?'pulse':''}" style="font-size:13px">${t.status==='active'?'sync':(t.status==='initializing'?'hourglass_top':'hourglass_empty')}</span> ${t.type || 'Task'}
                                            </span>
                                        </div>
                                        <div class="item-secondary">
                                            <span class="meta-tag" style="color:var(--md-sys-color-primary); font-weight:600;"><span class="material-icons-sharp" style="font-size:12px">layers</span><span class="stage-text">${t.stage || 'Initializing'}</span></span>
                                            <span class="meta-tag"><span class="material-icons-sharp" style="font-size:12px">movie</span>${formatDur(t.video_duration)}</span>
                                            <span class="meta-tag"><span class="material-icons-sharp" style="font-size:12px">timer</span><span class="timer-text">${t.status==='queued'?'Queued for: ':'Running: '}${formatDur(now-(t.start_time || now))}</span></span>
                                        </div>
                                    </div>
                                    <span class="badge badge-${t.status || 'unknown'}">${t.status === 'queued' ? 'queue' : (t.status || 'unknown')}</span>
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
                                <div class="hw-wait-msg" style="font-size:11px; color:var(--md-sys-color-warning); font-style:italic; margin-top:8px; display:${t.status==='queued'?'flex':'none'}; align-items:center; gap:4px;">
                                    <span class="material-icons-sharp" style="font-size:14px">hourglass_empty</span> Waiting for available hardware unit...
                                </div>
                            `;
                        } else {
                            // Update Status Icon and Badge
                            const statusIcon = card.querySelector('.item-info .material-icons-sharp');
                            if (statusIcon) {
                                statusIcon.innerText = t.status==='active'?'sync':(t.status==='initializing'?'hourglass_top':'hourglass_empty');
                                statusIcon.className = `material-icons-sharp ${t.status==='active'?'pulse':''}`;
                            }
                            const statusBadge = card.querySelector('.badge');
                            if (statusBadge) {
                                statusBadge.innerText = t.status === 'queued' ? 'queue' : (t.status || 'unknown');
                                statusBadge.className = `badge badge-${t.status || 'unknown'}`;
                            }

                            const hwWait = card.querySelector('.hw-wait-msg');
                            if (hwWait) {
                                hwWait.style.display = t.status === 'queued' ? 'flex' : 'none';
                            }

                            card.querySelector('.stage-text').innerText = t.stage || 'Initializing';
                            card.querySelector('.timer-text').innerText = (t.status==='queued'?'Queued for: ':'Running: ') + formatDur(now-(t.start_time || now));
                            card.querySelector('.progress-bar').style.width = progressPct + '%';
                            card.querySelector('.progress-text').innerText = progressPct;
                            const lb = card.querySelector('.log-buffer');
                            if (lb && lb.innerText !== logContent) {
                                lb.innerText = logContent;
                                lb.scrollTop = lb.scrollHeight;
                            }
                            const ltb = card.querySelector('.live-text-box');
                            if (ltb && ltb.innerText !== liveText) {
                                ltb.innerText = liveText;
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

                const engines = data.engines || {};
                document.getElementById('engine-list').innerHTML = Object.entries(engines).map(([k,v]) => `
                    <div class="list-item">
                        <div class="item-info">
                            <span class="item-primary">${k.toUpperCase()}</span>
                            <span class="item-secondary">${v.model || 'Unknown'}</span>
                        </div>
                        <span class="badge badge-${v.status || 'unknown'}">${v.status || 'unknown'}</span>
                    </div>`).join('');
                document.getElementById('last-update').innerText = `Updated: ${new Date().toLocaleTimeString()}`;
            } catch (e) { console.error(e); }
        }
        window.onload = () => {
            updateStats();
            setInterval(updateStats, 2000);
            showTab('active');
        };
    </script>
</body>
</html>
"""
