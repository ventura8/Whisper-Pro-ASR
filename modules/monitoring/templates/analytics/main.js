let rawData = null;
let charts = {};
let analyticsFetchInFlight = false;
const logger = globalThis.logger || console;

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
    if (!_isValidSeconds(sec)) return "0s";
    if (sec < 60) return sec.toFixed(1) + "s";
    const minutes = Math.floor(sec / 60);
    if (minutes < 60) {
        return _formatMinuteSecond(minutes, sec);
    }
    const hours = Math.floor(minutes / 60);
    if (hours < 24) {
        return _formatHourMinute(hours, minutes);
    }
    return _formatDayHourMinute(hours, minutes);
}

function _isValidSeconds(sec) {
    return sec !== undefined && sec !== null && sec >= 0;
}

function _formatMinuteSecond(minutes, sec) {
    const seconds = Math.floor(sec % 60);
    return minutes + "m " + seconds + "s";
}

function _formatHourMinute(hours, minutes) {
    return hours + "h " + (minutes % 60) + "m";
}

function _formatDayHourMinute(hours, minutes) {
    const days = Math.floor(hours / 24);
    const remHours = hours % 24;
    return days + "d " + remHours + "h " + (minutes % 60) + "m";
}

function formatDDHHMMSS(sec) {
    if (sec === undefined || sec === null || sec < 0) return "0d 0h 0m";
    const d = Math.floor(sec / 86400);
    const h = Math.floor((sec % 86400) / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${d}d ${h}h ${m}m`;
}

function getDetectLangStat(source) {
    if (!source) return { count: 0, duration: 0.0 };
    return source.detectlang || source.isolations || { count: 0, duration: 0.0 };
}

function _analyticsAuthHeaders() {
    const headers = { Accept: "application/json" };
    try {
        const apiKey = localStorage.getItem("whisper_api_key");
        if (apiKey) {
            headers["X-API-Key"] = apiKey;
        }
    } catch (err) {
        logger.debug("Analytics API key storage is unavailable:", err);
    }
    return headers;
}

async function fetchAnalytics() {
    if (analyticsFetchInFlight) {
        return;
    }
    analyticsFetchInFlight = true;
    try {
        const res = await fetch('/analytics', {
            headers: _analyticsAuthHeaders()
        });
        if (!res.ok) {
            throw new Error(`Analytics request failed with status ${res.status}`);
        }
        rawData = await res.json();
        renderAnalytics(rawData);
    } catch (e) {
        logger.error("Failed to fetch analytics:", e);
    } finally {
        analyticsFetchInFlight = false;
    }
}

function renderAnalytics(data) {
    const cumulative = data.cumulative || {};
    const daily = data.daily || {};

    _updateTopCards(cumulative);
    _updateCumulativeBreakdown(cumulative);
    const sortedDates = Object.keys(daily)
        .filter((key) => !key.startsWith("__"))
        .sort();

    _renderDailyBreakdownTable(sortedDates, daily);
    renderCharts(sortedDates, daily);
    document.getElementById('last-update').innerText = `Updated: ${new Date().toLocaleTimeString()}`;
}

function _updateTopCards(cumulative) {
    document.getElementById('val-all-time').innerText = formatDDHHMMSS(cumulative.all_time);
    document.getElementById('val-this-month').innerText = formatDDHHMMSS(cumulative.this_month || 0);
    document.getElementById('val-total-tasks').innerText = cumulative.count_all_time || 0;
    document.getElementById('val-today').innerText = formatDDHHMMSS(cumulative.today);
    document.getElementById('val-today-tasks').innerText = cumulative.count_today || 0;
    const avgDur = cumulative.count_all_time > 0 ? (cumulative.all_time / cumulative.count_all_time) : 0;
    document.getElementById('val-avg-duration').innerText = formatDuration(avgDur);
}

function _updateCumulativeBreakdown(cumulative) {
    const asrCum = cumulative.asr || { count: 0, duration: 0.0 };
    const detectlangCum = getDetectLangStat(cumulative);
    const audCum = cumulative.audio || { count: 0, duration: 0.0 };
    _setCumulativeCard('asr', asrCum);
    _setCumulativeCard('detectlang', detectlangCum);
    _setCumulativeCard('audio', audCum);
}

function _setCumulativeCard(prefix, payload) {
    document.getElementById(`val-${prefix}-cumulative`).innerText = formatDuration(payload.duration);
    document.getElementById(`val-${prefix}-count`).innerText = `${payload.count} tasks`;
}

function _renderDailyBreakdownTable(sortedDates, daily) {
    const tbody = document.getElementById('table-body');
    if (sortedDates.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--md-sys-color-secondary)">No analytics data recorded yet.</td></tr>`;
        return;
    }
    tbody.innerHTML = [...sortedDates].reverse().map((date) => _buildDailyBreakdownRow(date, daily)).join('');
}

function _buildDailyBreakdownRow(date, daily) {
    const info = daily[date] || {};
    const asr = info.asr || { count: 0, duration: 0.0 };
    const detectlang = getDetectLangStat(info);
    const aud = info.audio || { count: 0, duration: 0.0 };
    const safe = _buildDailyBreakdownSafeView(date, info, asr, detectlang, aud);
    return `
        <tr>
            <td style="font-family: 'Roboto Mono', monospace; font-weight: 500;">${safe.date}</td>
            <td><strong>${safe.count}</strong></td>
            <td>${safe.asrCount} <span style="font-size:11px; color:var(--md-sys-color-outline)">(${safe.asrDur})</span></td>
            <td>${safe.detectlangCount} <span style="font-size:11px; color:var(--md-sys-color-outline)">(${safe.detectlangDur})</span></td>
            <td>${safe.audCount} <span style="font-size:11px; color:var(--md-sys-color-outline)">(${safe.audDur})</span></td>
            <td><strong>${safe.totalDur}</strong></td>
        </tr>
    `;
}

function _buildDailyBreakdownSafeView(date, info, asr, detectlang, aud) {
    return {
        date: escapeHtml(date),
        count: escapeHtml(info.count || 0),
        asrCount: escapeHtml(asr.count || 0),
        detectlangCount: escapeHtml(detectlang.count || 0),
        audCount: escapeHtml(aud.count || 0),
        asrDur: escapeHtml(formatDuration(asr.duration)),
        detectlangDur: escapeHtml(formatDuration(detectlang.duration)),
        audDur: escapeHtml(formatDuration(aud.duration)),
        totalDur: escapeHtml(formatDuration(info.duration))
    };
}

function renderCharts(dates, daily) {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const taskSeries = _buildTaskSeries(dates, daily);
    const durationSeries = _buildDurationSeries(dates, daily);
    const tasksOptions = _buildTasksChartOptions(dates, taskSeries, isDark);
    const durationOptions = _buildDurationChartOptions(dates, durationSeries, isDark);
    updateOrCreateChart('tasksChart', tasksOptions);
    updateOrCreateChart('durationChart', durationOptions);
}

function _buildTaskSeries(dates, daily) {
    return [
        { name: '/asr', data: dates.map((d) => (daily[d].asr || {}).count || 0) },
        { name: '/detect-language', data: dates.map((d) => getDetectLangStat(daily[d]).count || 0) },
        { name: '/v1/audio/...', data: dates.map((d) => (daily[d].audio || {}).count || 0) }
    ];
}

function _buildDurationSeries(dates, daily) {
    return [
        { name: '/asr', data: dates.map((d) => _toMinutes((daily[d].asr || {}).duration || 0)) },
        { name: '/detect-language', data: dates.map((d) => _toMinutes(getDetectLangStat(daily[d]).duration || 0)) },
        { name: '/v1/audio/...', data: dates.map((d) => _toMinutes((daily[d].audio || {}).duration || 0)) }
    ];
}

function _toMinutes(seconds) {
    return parseFloat((seconds / 60).toFixed(2));
}

function _buildTasksChartOptions(dates, series, isDark) {
    return {
        series: [
            series[0],
            series[1],
            series[2]
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
}

function _buildDurationChartOptions(dates, series, isDark) {
    return {
        series: [
            series[0],
            series[1],
            series[2]
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
