function changeChartWindow(val) {
    chartWindowMinutes = parseInt(val, 10);
    renderCharts();
}

function renderCharts() {
    const data = lastStatusData;
    if (!data || !rollingTelemetryBuffer || rollingTelemetryBuffer.length === 0) return;
    
    // Sort telemetry buffer by timestamp
    const sorted = [...rollingTelemetryBuffer].sort((a,b) => a.timestamp - b.timestamp);
    if (sorted.length === 0) return;

    // N is the number of points for the selected window
    // 2-second interval, so 30 points per minute
    const N = chartWindowMinutes * 30;

    // Take at most the last N elements
    let rawPoints = sorted;
    if (rawPoints.length > N) {
        rawPoints = rawPoints.slice(-N);
    }

    // Pad on the left with dummy entries if we have fewer than N points
    const paddedPoints = [];
    const padCount = N - rawPoints.length;
    for (let i = 0; i < padCount; i++) {
        paddedPoints.push({
            isDummy: true,
            system: { cpu_percent: 0, app_cpu_percent: 0, app_memory_gb: 0 },
            telemetry: { npu_load: 0, intel_gpu_load: 0, nvidia: [] },
            cpu_sys: 0,
            cpu_app: 0,
            mem_app_gb: 0,
            nvidia_util: []
        });
    }
    for (let i = 0; i < rawPoints.length; i++) {
        paddedPoints.push(rawPoints[i]);
    }

    // Assign perfectly spaced normalized timestamps ending at nowMs
    const nowMs = Date.now();
    const dataPoints = paddedPoints.map((p, i) => {
        const timestampMs = nowMs - (N - 1 - i) * 2000;
        return {
            timestampMs: timestampMs,
            system: p.system,
            telemetry: p.telemetry,
            cpu_sys: p.cpu_sys || 0,
            cpu_app: p.cpu_app || 0,
            mem_app_gb: p.mem_app_gb || 0,
            nvidia_util: p.nvidia_util || []
        };
    });

    // Extract CPU data for charts and stats
    const cpuSysData = dataPoints.map(p => (p && p.system) ? p.system.cpu_percent : (p ? p.cpu_sys : null));
    const cpuAppData = dataPoints.map(p => (p && p.system) ? p.system.app_cpu_percent : (p ? p.cpu_app : null));
    const memData = dataPoints.map(p => (p && p.system) ? p.system.app_memory_gb : (p ? p.mem_app_gb : null));
    
    // Calculate current and highest for CPU
    const cpuSysFiltered = cpuSysData.filter(v => v !== null && v !== undefined && !isNaN(v));
    const cpuSysCurrent = cpuSysFiltered.length > 0 ? cpuSysFiltered[cpuSysFiltered.length - 1] : 0;
    const cpuSysHighest = cpuSysFiltered.length > 0 ? Math.max(...cpuSysFiltered) : 0;
    const cpuAppFiltered = cpuAppData.filter(v => v !== null && v !== undefined && !isNaN(v));
    const cpuAppCurrent = cpuAppFiltered.length > 0 ? cpuAppFiltered[cpuAppFiltered.length - 1] : 0;
    const cpuAppHighest = cpuAppFiltered.length > 0 ? Math.max(...cpuAppFiltered) : 0;
    
    // Calculate current and highest for memory
    const memFiltered = memData.filter(v => v !== null && v !== undefined && !isNaN(v));
    const memCurrent = memFiltered.length > 0 ? memFiltered[memFiltered.length - 1] : 0;
    const memHighest = memFiltered.length > 0 ? Math.max(...memFiltered) : 0;
    
    // Update CPU stat displays
    const cpuSysCurrentEl = document.getElementById('cpu-sys-current');
    const cpuSysHighestEl = document.getElementById('cpu-sys-highest');
    const cpuAppCurrentEl = document.getElementById('cpu-app-current');
    const cpuAppHighestEl = document.getElementById('cpu-app-highest');
    if (cpuSysCurrentEl) cpuSysCurrentEl.textContent = cpuSysCurrent.toFixed(1) + '%';
    if (cpuSysHighestEl) cpuSysHighestEl.textContent = cpuSysHighest.toFixed(1) + '%';
    if (cpuAppCurrentEl) cpuAppCurrentEl.textContent = cpuAppCurrent.toFixed(1) + '%';
    if (cpuAppHighestEl) cpuAppHighestEl.textContent = cpuAppHighest.toFixed(1) + '%';
    
    // Update Memory stat displays
    const memCurrentEl = document.getElementById('mem-current');
    const memPeakEl = document.getElementById('mem-peak');
    if (memCurrentEl) memCurrentEl.textContent = memCurrent.toFixed(2) + ' GB';
    if (memPeakEl) memPeakEl.textContent = memHighest.toFixed(2) + ' GB';

    createOrUpdateLineChart('cpuChart', [
        { label: 'System CPU %', data: dataPoints.map(p => ({ x: p.timestampMs, y: p.system ? p.system.cpu_percent : p.cpu_sys })), color: COLORS[0] },
        { label: 'App CPU %', data: dataPoints.map(p => ({ x: p.timestampMs, y: p.system ? p.system.app_cpu_percent : p.cpu_app })), color: COLORS[3] }
    ], false);

    createOrUpdateLineChart('memChart', [
        { label: 'App Memory (GB)', data: dataPoints.map(p => ({ x: p.timestampMs, y: p.system ? p.system.app_memory_gb : p.mem_app_gb })), color: COLORS[1] }
    ], false);

    const hwDatasets = [];
    (data.hardware_units || []).forEach((u, index) => {
        if (u.type === 'CPU') return;
        
        // Determine label and color
        let label = u.name || `${u.type} Load %`;
        let color = COLORS[(index + 2) % COLORS.length];
        if (u.type === 'NPU') color = '#8e24aa';
        else if (u.type === 'GPU' && !u.name.includes('NVIDIA')) color = '#3949ab';
        
        hwDatasets.push({
            label: label,
            data: dataPoints.map(p => {
                let val = 0;
                if (p.telemetry && p.telemetry.hardware_util && p.telemetry.hardware_util[u.id] !== undefined) {
                    val = p.telemetry.hardware_util[u.id];
                } else {
                    // Fallback to legacy structure
                    if (u.type === 'CUDA' || (u.type === 'GPU' && u.name.includes('NVIDIA'))) {
                        const idx = parseInt(u.id.split(':')[1] || 0);
                        const nv = (p.telemetry && p.telemetry.nvidia) ? p.telemetry.nvidia : p.nvidia_util;
                        val = (nv && nv[idx]) ? (nv[idx].util !== undefined ? nv[idx].util : nv[idx]) : 0;
                    } else if (u.type === 'GPU') {
                        val = p.telemetry?.intel_gpu_load || p.intel_util || 0;
                    } else if (u.type === 'NPU') {
                        val = p.telemetry?.npu_load || p.npu_util || 0;
                    }
                }
                return { x: p.timestampMs, y: val };
            }),
            color: color,
            unitId: u.id,
            unitName: u.name || u.type
        });
    });
    
    // Update hardware stats
    updateHardwareStats(hwDatasets);
    
    createOrUpdateLineChart('hwChart', hwDatasets, true);
}

function updateHardwareStats(hwDatasets) {
    const hwStatsEl = document.getElementById('hw-stats');
    if (!hwStatsEl) return;

    const escapeHtml = (value) => String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    
    hwStatsEl.innerHTML = '';
    hwDatasets.forEach(dataset => {
        const values = dataset.data.map(d => d.y).filter(v => !isNaN(v));
        const current = values[values.length - 1] || 0;
        const highest = values.length > 0 ? Math.max(...values) : 0;
        
        const statsDiv = document.createElement('div');
        statsDiv.className = 'stat-box stat-box-compact';
        const safeUnitName = escapeHtml(dataset.unitName);
        statsDiv.innerHTML = `
            <div class="stat-label">${safeUnitName}</div>
            <div class="stat-value">${current.toFixed(1)}%</div>
            <div class="item-secondary">
                <span>Current load</span>
                <span class="meta-tag">Peak ${highest.toFixed(1)}%</span>
            </div>
        `;
        hwStatsEl.appendChild(statsDiv);
    });
}

function createOrUpdateLineChart(id, datasets, percent) {
    const el = document.getElementById(id);
    if (!el) return;
    
    // Re-map datasets to ApexCharts format
    let series = datasets.map(d => ({
        name: d.label,
        data: d.data
    }));
    
    // Ensure at least one series exists to prevent ApexCharts from failing to render
    if (series.length === 0) {
        series = [{ name: 'No Acceleration Detected', data: [] }];
    }

    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const themeMode = isDark ? 'dark' : 'light';
    const rangeMs = (chartWindowMinutes * 30 - 1) * 2000;

    // Calculate dynamic Y-axis maximum for non-percentage charts (like memory)
    let yaxisMax = undefined;
    if (percent) {
        yaxisMax = 100;
    } else {
        let maxVal = 0;
        datasets.forEach(d => {
            d.data.forEach(pt => {
                if (pt.y > maxVal) maxVal = pt.y;
            });
        });
        // Pad by 10% and ceiling to nearest integer to avoid constant updates
        yaxisMax = maxVal > 0 ? Math.ceil(maxVal * 1.1) : 1;
    }

    const options = {
        series: series,
        chart: {
            type: 'area',
            height: 350,
            toolbar: { show: false },
            zoom: { enabled: false },
            animations: {
                enabled: false
            },
            background: 'transparent'
        },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.25,
                opacityTo: 0.02,
                stops: [0, 90, 100]
            }
        },
        theme: {
            mode: themeMode
        },
        colors: datasets.map(d => d.color || '#006495'),
        dataLabels: { enabled: false },
        stroke: { curve: 'straight', width: 2 },
        markers: { size: 0 },
        xaxis: {
            type: 'datetime',
            range: rangeMs,
            labels: {
                datetimeUTC: false,
                format: 'HH:mm:ss',
                style: { fontSize: '10px', fontFamily: 'Outfit' },
                rotate: 0,
                hideOverlappingLabels: true
            },
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            min: 0,
            max: yaxisMax,
            labels: {
                style: { fontSize: '10px', fontFamily: 'Outfit' },
                formatter: (val) => percent ? val.toFixed(0) + '%' : val.toFixed(1)
            }
        },
        grid: {
            borderColor: isDark ? 'rgba(255, 255, 255, 0.08)' : '#e5e7eb',
            strokeDashArray: 3,
            xaxis: { lines: { show: true } },
            yaxis: { lines: { show: true } }
        },
        legend: {
            position: 'top',
            horizontalAlign: 'right',
            fontFamily: 'Outfit',
            fontSize: '12px',
            markers: { radius: 12 }
        },
        tooltip: {
            theme: themeMode,
            x: { show: true, format: 'HH:mm:ss' },
            y: { formatter: (val) => percent ? val.toFixed(1) + '%' : val.toFixed(2) + ' GB' }
        }
    };

    if (!lastChartStates[id]) {
        lastChartStates[id] = { rangeMs: null, theme: null, yaxisMax: null };
    }

    const state = lastChartStates[id];

    if (charts[id]) {
        if (state.rangeMs !== rangeMs || state.theme !== themeMode || state.yaxisMax !== yaxisMax) {
            state.rangeMs = rangeMs;
            state.theme = themeMode;
            state.yaxisMax = yaxisMax;
            charts[id].updateOptions({
                xaxis: { range: rangeMs },
                yaxis: {
                    min: 0,
                    max: yaxisMax,
                    labels: {
                        style: { fontSize: '10px', fontFamily: 'Outfit' },
                        formatter: (val) => percent ? val.toFixed(0) + '%' : val.toFixed(1)
                    }
                },
                series: series,
                theme: { mode: themeMode },
                grid: { borderColor: isDark ? 'rgba(255, 255, 255, 0.08)' : '#e5e7eb' },
                tooltip: { theme: themeMode }
            });
        } else {
            charts[id].updateSeries(series);
        }
    } else {
        state.rangeMs = rangeMs;
        state.theme = themeMode;
        state.yaxisMax = yaxisMax;
        charts[id] = new ApexCharts(el, options);
        charts[id].render();
    }
}
