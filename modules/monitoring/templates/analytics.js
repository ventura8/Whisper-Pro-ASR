let rawData = null;
let charts = {};

function escapeHtml(value) {
    const str = String(value ?? "");
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function formatDuration(sec) {
    if (sec === undefined || sec === null || sec < 0) return "0s";
    if (sec < 60) return sec.toFixed(1) + "s";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    if (m < 60) return m + "m " + s + "s";
    const h = Math.floor(m / 60);
    const rm = m % 60;
    if (h < 24) return h + "h " + rm + "m";
    const d = Math.floor(h / 24);
    const rh = h % 24;
    return d + "d " + rh + "h " + rm + "m";
}

function formatDDHHMMSS(sec) {
    if (sec === undefined || sec === null || sec < 0) return "0d 0h 0m";
    const d = Math.floor(sec / 86400);
    const h = Math.floor((sec % 86400) / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${d}d ${h}h ${m}m`;
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
    document.getElementById('val-all-time').innerText = formatDDHHMMSS(cumulative.all_time);
    document.getElementById('val-this-month').innerText = formatDDHHMMSS(cumulative.this_month || 0);
    document.getElementById('val-total-tasks').innerText = cumulative.count_all_time || 0;
    document.getElementById('val-today').innerText = formatDDHHMMSS(cumulative.today);
    document.getElementById('val-today-tasks').innerText = cumulative.count_today || 0;

    const avgDur = cumulative.count_all_time > 0 ? (cumulative.all_time / cumulative.count_all_time) : 0;
    document.getElementById('val-avg-duration').innerText = formatDuration(avgDur);

    // Cumulative Breakdown Cards
    const asrCum = cumulative.asr || { count: 0, duration: 0.0 };
    const dlCum = cumulative.detectlang || { count: 0, duration: 0.0 };
    const audCum = cumulative.audio || { count: 0, duration: 0.0 };

    document.getElementById('val-asr-cumulative').innerText = formatDuration(asrCum.duration);
    document.getElementById('val-asr-count').innerText = `${asrCum.count} tasks`;

    document.getElementById('val-detectlang-cumulative').innerText = formatDuration(dlCum.duration);
    document.getElementById('val-detectlang-count').innerText = `${dlCum.count} tasks`;

    document.getElementById('val-audio-cumulative').innerText = formatDuration(audCum.duration);
    document.getElementById('val-audio-count').innerText = `${audCum.count} tasks`;

    // Sort Daily Breakdown by Date (oldest to newest for charts)
    const sortedDates = Object.keys(daily).sort();
    
    // Render Table (newest to oldest)
    const tbody = document.getElementById('table-body');
    if (sortedDates.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--md-sys-color-secondary)">No analytics data recorded yet.</td></tr>`;
    } else {
        tbody.innerHTML = [...sortedDates].reverse().map(date => {
            const info = daily[date] || {};
            const asr = info.asr || { count: 0, duration: 0.0 };
            const dl = info.detectlang || { count: 0, duration: 0.0 };
            const aud = info.audio || { count: 0, duration: 0.0 };
            const safeDate = escapeHtml(date);
            const safeCount = escapeHtml(info.count || 0);
            const safeAsrCount = escapeHtml(asr.count || 0);
            const safeDlCount = escapeHtml(dl.count || 0);
            const safeAudCount = escapeHtml(aud.count || 0);
            const safeAsrDur = escapeHtml(formatDuration(asr.duration));
            const safeDlDur = escapeHtml(formatDuration(dl.duration));
            const safeAudDur = escapeHtml(formatDuration(aud.duration));
            const safeTotalDur = escapeHtml(formatDuration(info.duration));
            return `
                <tr>
                    <td style="font-family: 'Roboto Mono', monospace; font-weight: 500;">${safeDate}</td>
                    <td><strong>${safeCount}</strong></td>
                    <td>${safeAsrCount} <span style="font-size:11px; color:var(--md-sys-color-outline)">(${safeAsrDur})</span></td>
                    <td>${safeDlCount} <span style="font-size:11px; color:var(--md-sys-color-outline)">(${safeDlDur})</span></td>
                    <td>${safeAudCount} <span style="font-size:11px; color:var(--md-sys-color-outline)">(${safeAudDur})</span></td>
                    <td><strong>${safeTotalDur}</strong></td>
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
    
    const asrCounts = dates.map(d => (daily[d].asr || {}).count || 0);
    const detectlangCounts = dates.map(d => (daily[d].detectlang || {}).count || 0);
    const audioCounts = dates.map(d => (daily[d].audio || {}).count || 0);

    const asrDurationsMin = dates.map(d => parseFloat((((daily[d].asr || {}).duration || 0) / 60).toFixed(2)));
    const detectlangDurationsMin = dates.map(d => parseFloat((((daily[d].detectlang || {}).duration || 0) / 60).toFixed(2)));
    const audioDurationsMin = dates.map(d => parseFloat((((daily[d].audio || {}).duration || 0) / 60).toFixed(2)));

    // 1. Tasks Chart (Stacked Bar Chart)
    const tasksOptions = {
        series: [
            { name: '/asr', data: asrCounts },
            { name: '/detect-language', data: detectlangCounts },
            { name: '/v1/audio/...', data: audioCounts }
        ],
        chart: {
            type: 'bar',
            stacked: true,
            height: 350,
            toolbar: { show: false },
            background: 'transparent'
        },
        theme: { mode: isDark ? 'dark' : 'light' },
        colors: ['#006495', '#e65100', '#2e7d32'],
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

    // 2. Duration Chart (Stacked Area Chart)
    const durationOptions = {
        series: [
            { name: '/asr', data: asrDurationsMin },
            { name: '/detect-language', data: detectlangDurationsMin },
            { name: '/v1/audio/...', data: audioDurationsMin }
        ],
        chart: {
            type: 'area',
            stacked: true,
            height: 350,
            toolbar: { show: false },
            background: 'transparent'
        },
        theme: { mode: isDark ? 'dark' : 'light' },
        colors: ['#006495', '#e65100', '#2e7d32'],
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
