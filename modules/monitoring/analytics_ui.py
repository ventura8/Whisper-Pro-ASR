# pylint: disable=duplicate-code
"""
HTML template for the Whisper Pro Analytics Dashboard.
"""


def get_analytics_html() -> str:
    """Returns the rendered HTML for the analytics page."""
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Pro Analytics</title>
    <meta name="description" content="Detailed historical performance and usage analytics for Whisper Pro ASR service.">
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
            --md-sys-color-surface: #ffffff;
            --md-sys-color-surface-variant: #dee3eb;
            --md-sys-color-outline: #72777f;
            --md-sys-color-background: #fdfcff;
            --md-sys-color-on-surface: #191c1e;
            --md-sys-color-success: #2e7d32;
            --md-sys-color-error: #ba1a1a;
            --md-sys-color-warning: #e65100;

            --card-bg: #ffffff;
            --item-bg: #f8fafc;
            --border-color: var(--md-sys-color-surface-variant);
            --meta-bg: #f1f3f4;
            --btn-hover-bg: #e0f2fe;
            --table-header-bg: #f1f5f9;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --md-sys-color-primary: #90caf9;
                --md-sys-color-on-primary: #00325b;
                --md-sys-color-primary-container: #004b77;
                --md-sys-color-secondary: #a2aab3;
                --md-sys-color-surface: #1c1c1e;
                --md-sys-color-surface-variant: #43474e;
                --md-sys-color-outline: #8c9199;
                --md-sys-color-background: #121214;
                --md-sys-color-on-surface: #e2e2e6;
                --md-sys-color-success: #81c784;
                --md-sys-color-error: #e57373;
                --md-sys-color-warning: #ffb74d;

                --card-bg: #1c1c1e;
                --item-bg: #2a2a2e;
                --border-color: #2e3035;
                --meta-bg: #2e3035;
                --btn-hover-bg: #374151;
                --table-header-bg: #2a2a2e;
            }
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
            transition: background-color 0.3s, color 0.3s;
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

        .btn-action {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 20px;
            background: var(--md-sys-color-primary);
            color: var(--md-sys-color-on-primary);
            text-decoration: none;
            border-radius: 100px;
            font-size: 14px;
            font-weight: 500;
            transition: opacity 0.2s, background-color 0.3s, color 0.3s;
            border: 1px solid var(--md-sys-color-primary);
            cursor: pointer;
        }
        .btn-action:hover { opacity: 0.9; }

        .btn-secondary {
            background: var(--card-bg);
            color: var(--md-sys-color-primary);
            border: 1px solid var(--md-sys-color-primary);
        }
        .btn-secondary:hover { background: var(--btn-hover-bg); }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }

        .card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
            border: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 8px;
            transition: background-color 0.3s, border-color 0.3s, transform 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
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

        .card-value { font-size: 24px; font-family: 'Outfit', sans-serif; font-weight: 600; }

        .layout-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 24px;
        }

        @media (max-width: 1200px) {
            .layout-grid { grid-template-columns: 1fr; }
        }

        .charts-container {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .chart-card {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }

        .section-title {
            font-size: 18px;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .chart-element {
            height: 350px;
            width: 100%;
        }

        .table-card {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 24px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .table-container {
            overflow-x: auto;
            max-height: 750px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 14px;
        }

        th, td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }

        th {
            background: var(--table-header-bg);
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            color: var(--md-sys-color-secondary);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        tr:hover td {
            background: var(--item-bg);
        }

        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 80px 40px;
            color: var(--md-sys-color-secondary);
            text-align: center;
            gap: 16px;
        }
        .empty-icon { font-size: 56px; color: var(--md-sys-color-surface-variant); }

        .refresh-indicator {
            font-size: 12px;
            color: var(--md-sys-color-outline);
            text-align: right;
            margin-top: 12px;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <span class="material-icons-sharp" style="font-size: 32px; color: var(--md-sys-color-primary)">bar_chart</span>
            <h1>Whisper Pro Analytics</h1>
        </div>
        <div class="header-actions">
            <a href="/dashboard" class="btn-action btn-secondary" id="btn-back">
                <span class="material-icons-sharp" style="font-size: 18px">dashboard</span>
                Dashboard
            </a>
            <button onclick="exportJson()" class="btn-action" id="btn-export">
                <span class="material-icons-sharp" style="font-size: 18px">download</span>
                Export Data
            </button>
        </div>
    </header>

    <div class="stats-grid">
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">all_inclusive</span> All-Time Processed</div>
            <div class="card-value" id="val-all-time">--</div>
        </div>
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">task_alt</span> Total Tasks</div>
            <div class="card-value" id="val-total-tasks">--</div>
        </div>
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">today</span> Today's Processed</div>
            <div class="card-value" id="val-today">--</div>
        </div>
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">playlist_add_check</span> Today's Tasks</div>
            <div class="card-value" id="val-today-tasks">--</div>
        </div>
        <div class="card">
            <div class="card-title"><span class="material-icons-sharp">hourglass_empty</span> Avg Task Duration</div>
            <div class="card-value" id="val-avg-duration">--</div>
        </div>
    </div>

    <div class="layout-grid">
        <div class="charts-container">
            <div class="chart-card">
                <div class="section-title"><span class="material-icons-sharp">show_chart</span> Daily Tasks Completed</div>
                <div id="tasksChart" class="chart-element"></div>
            </div>
            <div class="chart-card">
                <div class="section-title"><span class="material-icons-sharp">hourglass_full</span> Daily Transcription Volume (Minutes)</div>
                <div id="durationChart" class="chart-element"></div>
            </div>
        </div>

        <div class="table-card">
            <div class="section-title"><span class="material-icons-sharp">list_alt</span> Daily Breakdown History</div>
            <div class="table-container">
                <table id="daily-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Tasks</th>
                            <th>Duration</th>
                        </tr>
                    </thead>
                    <tbody id="table-body">
                        <tr>
                            <td colspan="3" style="text-align: center; color: var(--md-sys-color-secondary)">Loading...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="refresh-indicator" id="last-update">Updating...</div>

    <script>
        let rawData = null;
        let charts = {};

        function formatDuration(sec) {
            if (sec === undefined || sec === null || sec < 0) return "0s";
            if (sec < 60) return sec.toFixed(1) + "s";
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60);
            if (m < 60) return m + "m " + s + "s";
            const h = Math.floor(m / 60);
            const rm = m % 60;
            return h + "h " + rm + "m";
        }

        async function fetchAnalytics() {
            try {
                const res = await fetch('/analytics', {
                    headers: { 'Accept': 'application/json' }
                });
                rawData = await res.json();
                renderAnalytics(rawData);
            } catch (e) {
                console.error("Failed to fetch analytics:", e);
            }
        }

        function renderAnalytics(data) {
            const cumulative = data.cumulative || {};
            const daily = data.daily || {};

            // Update Cards
            document.getElementById('val-all-time').innerText = formatDuration(cumulative.all_time);
            document.getElementById('val-total-tasks').innerText = cumulative.count_all_time || 0;
            document.getElementById('val-today').innerText = formatDuration(cumulative.today);
            document.getElementById('val-today-tasks').innerText = cumulative.count_today || 0;

            const avgDur = cumulative.count_all_time > 0 ? (cumulative.all_time / cumulative.count_all_time) : 0;
            document.getElementById('val-avg-duration').innerText = formatDuration(avgDur);

            // Sort Daily Breakdown by Date (oldest to newest for charts)
            const sortedDates = Object.keys(daily).sort();
            
            // Render Table (newest to oldest)
            const tbody = document.getElementById('table-body');
            if (sortedDates.length === 0) {
                tbody.innerHTML = `<tr><td colspan="3" style="text-align: center; color: var(--md-sys-color-secondary)">No analytics data recorded yet.</td></tr>`;
            } else {
                tbody.innerHTML = [...sortedDates].reverse().map(date => {
                    const info = daily[date] || {};
                    return `
                        <tr>
                            <td style="font-family: 'Roboto Mono', monospace; font-weight: 500;">${date}</td>
                            <td>${info.count || 0}</td>
                            <td>${formatDuration(info.duration)}</td>
                        </tr>
                    `;
                }).join('');
            }

            // Render Charts
            renderCharts(sortedDates, daily);

            document.getElementById('last-update').innerText = `Updated: ${new Date().toLocaleTimeString()}`;
        }

        function renderCharts(dates, daily) {
            const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            
            const taskCounts = dates.map(d => daily[d].count || 0);
            const durationsMin = dates.map(d => parseFloat(((daily[d].duration || 0) / 60).toFixed(2)));

            // 1. Tasks Chart
            const tasksOptions = {
                series: [{
                    name: 'Tasks Completed',
                    data: taskCounts
                }],
                chart: {
                    type: 'bar',
                    height: 350,
                    toolbar: { show: false },
                    background: 'transparent'
                },
                theme: { mode: isDark ? 'dark' : 'light' },
                colors: ['#006495'],
                plotOptions: {
                    bar: {
                        borderRadius: 6,
                        columnWidth: '50%',
                    }
                },
                dataLabels: { enabled: false },
                xaxis: {
                    categories: dates,
                    labels: { style: { fontSize: '10px' } }
                },
                yaxis: {
                    title: { text: 'Count', style: { fontFamily: 'Outfit' } },
                    labels: { formatter: (val) => val.toFixed(0) }
                },
                grid: {
                    borderColor: isDark ? 'rgba(255, 255, 255, 0.08)' : '#f0f0f0',
                    strokeDashArray: 4
                }
            };

            // 2. Duration Chart
            const durationOptions = {
                series: [{
                    name: 'Audio Processed',
                    data: durationsMin
                }],
                chart: {
                    type: 'area',
                    height: 350,
                    toolbar: { show: false },
                    background: 'transparent'
                },
                theme: { mode: isDark ? 'dark' : 'light' },
                colors: ['#2e7d32'],
                stroke: { curve: 'smooth', width: 3 },
                dataLabels: { enabled: false },
                xaxis: {
                    categories: dates,
                    labels: { style: { fontSize: '10px' } }
                },
                yaxis: {
                    title: { text: 'Minutes', style: { fontFamily: 'Outfit' } },
                    labels: { formatter: (val) => val.toFixed(1) + ' m' }
                },
                grid: {
                    borderColor: isDark ? 'rgba(255, 255, 255, 0.08)' : '#f0f0f0',
                    strokeDashArray: 4
                },
                fill: {
                    type: 'gradient',
                    gradient: {
                        shadeIntensity: 1,
                        opacityFrom: 0.45,
                        opacityTo: 0.05,
                        stops: [0, 100]
                    }
                }
            };

            updateOrCreateChart('tasksChart', tasksOptions);
            updateOrCreateChart('durationChart', durationOptions);
        }

        function updateOrCreateChart(id, options) {
            const el = document.getElementById(id);
            if (!el) return;

            if (charts[id]) {
                charts[id].updateOptions({
                    xaxis: { categories: options.xaxis.categories },
                    series: options.series,
                    theme: { mode: options.theme.mode },
                    grid: { borderColor: options.grid.borderColor }
                });
            } else {
                charts[id] = new ApexCharts(el, options);
                charts[id].render();
            }
        }

        function exportJson() {
            if (!rawData) return;
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(rawData, null, 2));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href",     dataStr);
            downloadAnchor.setAttribute("download", "whisper_pro_analytics.json");
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        }

        window.onload = () => {
            fetchAnalytics();
            setInterval(fetchAnalytics, 10000); // refresh every 10 seconds

            if (window.matchMedia) {
                window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
                    if (rawData) {
                        renderAnalytics(rawData);
                    }
                });
            }
        };
    </script>
</body>
</html>"""
