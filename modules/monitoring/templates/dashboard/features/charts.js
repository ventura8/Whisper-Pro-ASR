function changeChartWindow(val) {
    chartWindowMinutes = parseInt(val, 10);
    renderCharts();
}

function renderCharts() {
    const data = lastStatusData;
    if (!data || !rollingTelemetryBuffer || rollingTelemetryBuffer.length === 0) return;
    const sorted = [...rollingTelemetryBuffer].sort((a,b) => a.timestamp - b.timestamp);
    if (sorted.length === 0) return;
    const pointsInfo = _buildDataPointsForWindow(sorted);
    const dataPoints = pointsInfo.dataPoints;
    const stats = _calculateChartStats(dataPoints);
    _updateCpuStatsDisplay(stats.cpu);
    _updateMemoryStatsDisplay(stats.mem);

    createOrUpdateLineChart('cpuChart', [
        { label: 'System CPU %', data: dataPoints.map((p) => _toChartPoint(p, _cpuSystemValue(p))), color: COLORS[0] },
        { label: 'App CPU %', data: dataPoints.map((p) => _toChartPoint(p, _cpuAppValue(p))), color: COLORS[3] }
    ], false);

    createOrUpdateLineChart('memChart', [
        { label: 'App Memory (GB)', data: dataPoints.map((p) => _toChartPoint(p, _memAppValue(p))), color: COLORS[1] },
        { label: 'System Memory (GB)', data: dataPoints.map((p) => _toChartPoint(p, _memSystemValue(p))), color: COLORS[2] }
    ], false);
    const hwDatasets = _buildHardwareDatasets(data, dataPoints);
    
    updateHardwareStats(hwDatasets);
    updateHardwareLegend(hwDatasets);
    createOrUpdateLineChart('hwChart', hwDatasets, true);
}

function _buildDataPointsForWindow(sorted) {
    const nowMs = Date.now();
    const windowStartSec = (nowMs - chartWindowMinutes * 60 * 1000) / 1000;
    const dataPoints = sorted
        .filter((point) => point && typeof point.timestamp === 'number' && point.timestamp >= windowStartSec)
        .map((point) => ({
            timestampMs: point.timestamp * 1000,
            system: point.system,
            telemetry: point.telemetry,
            cpu_sys: _orZero(point.cpu_sys),
            cpu_app: _orZero(point.cpu_app),
            mem_app_gb: _orZero(point.mem_app_gb),
            mem_sys_gb: _orZero(point.mem_sys_gb),
            nvidia_util: _orArray(point.nvidia_util)
        }));
    return {
        totalPoints: dataPoints.length,
        dataPoints: dataPoints
    };
}

function _toChartPoint(point, value) {
    return { x: point.timestampMs, y: value };
}

function _cpuSystemValue(point) {
    return point.system ? point.system.cpu_percent : point.cpu_sys;
}

function _cpuAppValue(point) {
    return point.system ? point.system.app_cpu_percent : point.cpu_app;
}

function _memAppValue(point) {
    if (point.system) {
        return point.system.app_memory_gb || 0;
    }
    return point.mem_app_gb;
}

function _memSystemValue(point) {
    if (point.system) {
        return point.system.memory_used_gb || 0;
    }
    return point.mem_sys_gb || 0;
}

function _leftPadDataPoints(rawPoints, totalPoints) {
    const padded = [];
    const padCount = totalPoints - rawPoints.length;
    for (let i = 0; i < padCount; i++) {
        padded.push(_dummyTelemetryPoint());
    }
    return padded.concat(rawPoints);
}

function _dummyTelemetryPoint() {
    return {
        isDummy: true,
        system: { cpu_percent: 0, app_cpu_percent: 0, app_memory_gb: 0 },
        telemetry: { npu_load: 0, intel_gpu_load: 0, nvidia: [] },
        cpu_sys: 0,
        cpu_app: 0,
        mem_app_gb: 0,
        nvidia_util: []
    };
}

function _normalizeDataPointTimestamps(points, totalPoints) {
    const nowMs = Date.now();
    return points.map((p, i) => _normalizedPoint(p, i, totalPoints, nowMs));
}

function _normalizedPoint(point, index, totalPoints, nowMs) {
    return {
        timestampMs: nowMs - (totalPoints - 1 - index) * 2000,
        system: point.system,
        telemetry: point.telemetry,
        cpu_sys: _orZero(point.cpu_sys),
        cpu_app: _orZero(point.cpu_app),
        mem_app_gb: _orZero(point.mem_app_gb),
        mem_sys_gb: _orZero(point.mem_sys_gb),
        nvidia_util: _orArray(point.nvidia_util)
    };
}

function _orZero(value) {
    return value === undefined || value === null ? 0 : value;
}

function _orArray(value) {
    return Array.isArray(value) ? value : [];
}

function _calculateChartStats(dataPoints) {
    const cpuSysData = dataPoints.map((p) => _readPointMetric(p, 'cpu_percent', 'cpu_sys'));
    const cpuAppData = dataPoints.map((p) => _readPointMetric(p, 'app_cpu_percent', 'cpu_app'));
    const memAppData = dataPoints.map((p) => _readPointMetric(p, 'app_memory_gb', 'mem_app_gb'));
    const memSysData = dataPoints.map((p) => _readPointMetric(p, 'memory_used_gb', 'mem_sys_gb'));
    return {
        cpu: {
            system: _currentAndPeak(cpuSysData),
            app: _currentAndPeak(cpuAppData)
        },
        mem: {
            system: _currentAndPeak(memSysData),
            app: _currentAndPeak(memAppData)
        }
    };
}

function _readPointMetric(point, systemKey, fallbackKey) {
    if (!point) return null;
    if (point.system) {
        return point.system[systemKey] || 0;
    }
    return point[fallbackKey] || 0;
}

function _currentAndPeak(values) {
    const filtered = values.filter((v) => _isFiniteMetric(v));
    if (filtered.length === 0) {
        return { current: 0, peak: 0 };
    }
    return {
        current: filtered[filtered.length - 1],
        peak: Math.max(...filtered)
    };
}

function _isFiniteMetric(value) {
    return value !== null && value !== undefined && !isNaN(value);
}

function _setTextIfExists(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function _updateCpuStatsDisplay(cpuStats) {
    _setTextIfExists('cpu-sys-current', cpuStats.system.current.toFixed(1) + '%');
    _setTextIfExists('cpu-sys-highest', cpuStats.system.peak.toFixed(1) + '%');
    _setTextIfExists('cpu-app-current', cpuStats.app.current.toFixed(1) + '%');
    _setTextIfExists('cpu-app-highest', cpuStats.app.peak.toFixed(1) + '%');
}

function _updateMemoryStatsDisplay(memStats) {
    _setTextIfExists('mem-sys-current', memStats.system.current.toFixed(2) + ' GB');
    _setTextIfExists('mem-sys-max', memStats.system.peak.toFixed(2) + ' GB');
    const memAppCurrentEl = document.getElementById('mem-app-current') || document.getElementById('mem-current');
    const memAppPeakEl = document.getElementById('mem-app-peak') || document.getElementById('mem-peak');
    if (memAppCurrentEl) memAppCurrentEl.textContent = memStats.app.current.toFixed(2) + ' GB';
    if (memAppPeakEl) memAppPeakEl.textContent = memStats.app.peak.toFixed(2) + ' GB';
}

function _buildHardwareDatasets(data, dataPoints) {
    const units = (data.hardware_units || []).filter((u) => u.type !== 'CPU');
    return units.map((u, index) => _buildHardwareDataset(u, index, dataPoints));
}

function _buildHardwareDataset(unit, index, dataPoints) {
    const style = _resolveHardwareDatasetStyle(unit, index);
    return {
        label: style.label,
        data: dataPoints.map((p) => ({ x: p.timestampMs, y: _resolveHardwareValue(p, unit) })),
        color: style.color,
        dashArray: style.dashArray,
        strokeWidth: style.strokeWidth,
        markerSize: style.markerSize,
        markerShape: style.markerShape,
        markerStep: style.markerStep,
        markerOffset: style.markerOffset,
        unitId: unit.id,
        unitName: _hardwareStatName(unit)
    };
}

function _resolveHardwareDatasetStyle(unit, index) {
    const bucket = _hardwareStyleBucket(unit);
    const color = _colorForHardwareBucket(bucket, unit.id, index);
    const dashArray = _dashForHardwareBucket(bucket);
    const markerShape = _markerForHardwareBucket(bucket);
    const markerStep = _hardwareMarkerStep(bucket, unit.id, index);
    const markerOffset = _hardwareMarkerOffset(unit.id, markerStep);
    return {
        label: _hardwareLegendLabel(unit),
        color: color,
        dashArray: dashArray,
        strokeWidth: 3,
        markerSize: 4,
        markerShape: markerShape,
        markerStep: markerStep,
        markerOffset: markerOffset
    };
}

function _hardwareLegendLabel(unit) {
    const unitType = (unit.type || 'UNKNOWN').toUpperCase();
    const unitId = unit.id || 'unknown';
    const unitName = unit.name ? ` - ${unit.name}` : '';
    return `${unitType} ${unitId}${unitName}`;
}

function _hardwareStatName(unit) {
    const unitId = unit.id || 'unknown';
    const baseName = unit.name || unit.type || 'Hardware';
    return `${baseName} (${unitId})`;
}

function _hardwareStyleBucket(unit) {
    if (_isNvidiaUnit(unit)) {
        return 'CUDA';
    }
    if (unit.type === 'NPU') {
        return 'NPU';
    }
    if (unit.type === 'GPU') {
        return 'GPU';
    }
    return 'OTHER';
}

function _colorForHardwareBucket(bucket, unitId, index) {
    const palettes = {
        CUDA: ['#ff6f00', '#f4511e', '#e64a19', '#fb8c00'],
        NPU: ['#8e24aa', '#ab47bc', '#6a1b9a', '#9c27b0'],
        GPU: ['#1e88e5', '#3949ab', '#00897b', '#2e7d32'],
        OTHER: ['#546e7a', '#6d4c41', '#455a64', '#78909c']
    };
    const palette = palettes[bucket] || palettes.OTHER;
    const hash = _stableTextHash(String(unitId || ''));
    return palette[(hash + index) % palette.length];
}

function _stableTextHash(text) {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
        hash = (hash * 31 + text.charCodeAt(i)) % 997;
    }
    return hash;
}

function _dashForHardwareBucket(bucket) {
    const byBucket = {
        CUDA: 0,
        GPU: 6,
        NPU: 2,
        OTHER: 8
    };
    return byBucket[bucket] ?? 4;
}

function _markerForHardwareBucket(bucket) {
    const byBucket = {
        CUDA: 'circle',
        GPU: 'square',
        NPU: 'triangle',
        OTHER: 'diamond'
    };
    return byBucket[bucket] || 'circle';
}

function _hardwareMarkerStep(bucket, unitId, index) {
    const baseByBucket = {
        CUDA: 1,
        GPU: 2,
        NPU: 3,
        OTHER: 5
    };
    const base = baseByBucket[bucket] || 2;
    const hash = _stableTextHash(String(unitId || ''));
    if (bucket === 'CUDA') {
        return base;
    }
    return base + ((hash + index) % 2);
}

function _hardwareMarkerOffset(unitId, markerStep) {
    if (!markerStep || markerStep <= 1) {
        return 0;
    }
    return _stableTextHash(String(unitId || '')) % markerStep;
}

function _resolveHardwareValue(point, unit) {
    const direct = _resolveHardwareValueFromTelemetryMap(point, unit);
    if (direct !== null) {
        return direct;
    }
    return _resolveHardwareValueFromLegacyTelemetry(point, unit);
}

function _resolveHardwareValueFromTelemetryMap(point, unit) {
    const telemetry = point.telemetry || {};
    const hwUtil = telemetry.hardware_util;
    if (hwUtil && hwUtil[unit.id] !== undefined) {
        return hwUtil[unit.id];
    }
    return null;
}

function _resolveHardwareValueFromLegacyTelemetry(point, unit) {
    const resolver = _legacyHardwareResolver(unit);
    return resolver(point, unit);
}

function _legacyHardwareResolver(unit) {
    if (_isNvidiaUnit(unit)) {
        return _resolveLegacyNvidiaValue;
    }
    return _legacyResolverByType(unit.type);
}

function _legacyResolverByType(unitType) {
    const resolverMap = {
        GPU: _resolveLegacyIntelGpuValue,
        NPU: _resolveLegacyNpuValue
    };
    return resolverMap[unitType] || _legacyZeroValue;
}

function _resolveLegacyIntelGpuValue(point, unit) {
    const unitId = unit?.id;
    const telemetryValue = point.telemetry ? point.telemetry.intel_gpu_load : null;
    const pointValue = point.intel_util;
    return _resolveLegacyUnitMetricValue(unitId, telemetryValue, pointValue);
}

function _resolveLegacyNpuValue(point, unit) {
    const unitId = unit?.id;
    const telemetryValue = point.telemetry ? point.telemetry.npu_load : null;
    const pointValue = point.npu_util;
    return _resolveLegacyUnitMetricValue(unitId, telemetryValue, pointValue);
}

function _legacyZeroValue() {
    return 0;
}

function _resolveLegacyNvidiaValue(point, unit) {
    const idx = _resolveUnitIndex(unit.id);
    const nv = _legacyNvidiaArray(point);
    if (!_hasLegacyNvidiaSample(nv, idx)) {
        return 0;
    }
    const metric = nv[idx];
    return _legacyNvidiaMetricValue(metric);
}

function _legacyNvidiaArray(point) {
    if (point.telemetry && point.telemetry.nvidia) {
        return point.telemetry.nvidia;
    }
    return point.nvidia_util;
}

function _hasLegacyNvidiaSample(nv, idx) {
    if (!nv) {
        return false;
    }
    return !!nv[idx];
}

function _legacyNvidiaMetricValue(metric) {
    if (metric.util !== undefined) {
        return metric.util;
    }
    return metric;
}

function _resolveLegacyUnitMetricValue(unitId, primaryValue, secondaryValue) {
    const resolved = _pickLegacyUnitMetric(unitId, primaryValue);
    if (resolved !== null) {
        return resolved;
    }
    const fallback = _pickLegacyUnitMetric(unitId, secondaryValue);
    if (fallback !== null) {
        return fallback;
    }
    return 0;
}

function _pickLegacyUnitMetricFromArray(unitId, value) {
    const idx = _resolveUnitIndex(unitId);
    return idx < value.length ? _legacyNvidiaMetricValue(value[idx]) : null;
}

function _pickLegacyUnitMetricFromMap(unitId, value) {
    if (unitId && value[unitId] !== undefined) {
        return _legacyNvidiaMetricValue(value[unitId]);
    }
    const idxKey = String(_resolveUnitIndex(unitId));
    return value[idxKey] !== undefined ? _legacyNvidiaMetricValue(value[idxKey]) : null;
}

function _pickLegacyUnitMetric(unitId, value) {
    if (value === null || value === undefined) {
        return null;
    }
    return _pickLegacyUnitMetricByType(unitId, value);
}

function _pickLegacyUnitMetricByType(unitId, value) {
    if (typeof value === 'number') {
        return value;
    }
    if (Array.isArray(value)) {
        return _pickLegacyUnitMetricFromArray(unitId, value);
    }
    if (typeof value === 'object') {
        return _pickLegacyUnitMetricFromMap(unitId, value);
    }
    return null;
}

function _resolveUnitIndex(unitId) {
    const match = String(unitId || '').match(/(\d+)(?!.*\d)/);
    return match ? parseInt(match[1], 10) : 0;
}

function _isNvidiaUnit(unit) {
    return unit.type === 'CUDA' || (unit.type === 'GPU' && typeof unit.name === 'string' && unit.name.includes('NVIDIA'));
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

function updateHardwareLegend(hwDatasets) {
    const legendEl = document.getElementById('hw-legend');
    if (!legendEl) return;
    legendEl.innerHTML = '';

    hwDatasets.forEach((dataset) => {
        const item = _buildHardwareLegendItem(dataset);
        legendEl.appendChild(item);
    });
}

function _hardwareLegendColor(dataset) {
    return dataset.color || '#006495';
}

function _hardwareLegendLabelText(dataset) {
    return dataset.label || dataset.unitName || dataset.unitId || 'Hardware Unit';
}

function _applyLegendLinePattern(line, linePattern) {
    if (!linePattern.pattern) {
        return;
    }
    line.style.setProperty('--legend-line-pattern', linePattern.pattern);
    line.style.setProperty('--legend-line-size', linePattern.size);
}

function _buildHardwareLegendItem(dataset) {
    const color = _hardwareLegendColor(dataset);
    const item = document.createElement('div');
    item.className = 'hw-legend-item';

    const swatch = document.createElement('span');
    swatch.className = 'hw-legend-swatch';
    swatch.style.setProperty('--legend-color', color);

    const line = document.createElement('span');
    line.className = 'hw-legend-line';
    _applyLegendLinePattern(line, _legendLinePattern(dataset.dashArray || 0, color));

    const marker = document.createElement('span');
    marker.className = `hw-legend-marker hw-marker-${dataset.markerShape || 'circle'}`;

    const label = document.createElement('span');
    label.className = 'hw-legend-label';
    label.textContent = _hardwareLegendLabelText(dataset);

    swatch.appendChild(line);
    swatch.appendChild(marker);
    item.appendChild(swatch);
    item.appendChild(label);
    return item;
}

function _legendLinePattern(dashArray, color) {
    if (!dashArray) {
        return { pattern: '', size: '' };
    }
    const gap = Math.max(2, Math.round(dashArray / 2));
    const dash = Math.max(2, dashArray);
    return {
        pattern: `repeating-linear-gradient(to right, ${color} 0 ${dash}px, transparent ${dash}px ${dash + gap}px)`,
        size: `${dash + gap}px 3px`
    };
}

function createOrUpdateLineChart(id, datasets, percent) {
    const el = document.getElementById(id);
    if (!el) return;
    let series = _buildApexSeries(datasets);
    series = _withFallbackSeries(series);
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const themeMode = isDark ? 'dark' : 'light';
    const rangeMs = (chartWindowMinutes * 30 - 1) * 2000;
    const yaxisMax = _resolveChartYaxisMax(datasets, percent);
    const options = _buildLineChartOptions(id, series, datasets, percent, isDark, themeMode, rangeMs, yaxisMax);
    const state = _getOrCreateChartState(id);
    if (charts[id]) {
        _updateExistingChart(id, state, series, datasets, percent, isDark, themeMode, rangeMs, yaxisMax);
        return;
    }
    _createNewChart(id, el, state, options, datasets, rangeMs, themeMode, yaxisMax);
}

function _getOrCreateChartState(id) {
    if (!lastChartStates[id]) {
        lastChartStates[id] = { rangeMs: null, theme: null, yaxisMax: null, styleSig: null };
    }
    return lastChartStates[id];
}

function _buildApexSeries(datasets) {
    return datasets.map((d) => ({ name: d.label, data: d.data }));
}

function _withFallbackSeries(series) {
    if (series.length === 0) {
        return [{ name: 'No Acceleration Detected', data: [] }];
    }
    return series;
}

function _resolveChartYaxisMax(datasets, percent) {
    if (percent) return 100;
    const maxVal = _maxDatasetValue(datasets);
    return maxVal > 0 ? Math.ceil(maxVal * 1.1) : 1;
}

function _maxDatasetValue(datasets) {
    let maxVal = 0;
    datasets.forEach((d) => {
        d.data.forEach((pt) => {
            if (pt.y > maxVal) maxVal = pt.y;
        });
    });
    return maxVal;
}

function _buildLineChartOptions(id, series, datasets, percent, isDark, themeMode, rangeMs, yaxisMax) {
    const strokeStyles = _seriesStrokeStyles(datasets);
    const markerStyles = _seriesMarkerStyles(id, datasets);
    const showLegend = id !== 'hwChart';
    return {
        series: series,
        chart: {
            type: 'area',
            height: 350,
            toolbar: { show: false },
            zoom: { enabled: false },
            animations: { enabled: false },
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
        theme: { mode: themeMode },
        colors: datasets.map((d) => d.color || '#006495'),
        dataLabels: { enabled: false },
        stroke: strokeStyles,
        markers: markerStyles,
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
        yaxis: _buildChartYaxis(percent, yaxisMax),
        grid: {
            borderColor: isDark ? 'rgba(255, 255, 255, 0.08)' : '#e5e7eb',
            strokeDashArray: 3,
            xaxis: { lines: { show: true } },
            yaxis: { lines: { show: true } }
        },
        legend: {
            show: showLegend,
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
}

function _seriesStrokeStyles(datasets) {
    return {
        curve: 'straight',
        width: datasets.map((d) => d.strokeWidth || 2),
        dashArray: datasets.map((d) => d.dashArray || 0)
    };
}

function _seriesMarkerStyles(id, datasets) {
    if (id === 'hwChart') {
        return {
            size: 0,
            strokeWidth: 0,
            hover: { sizeOffset: 2 },
            discrete: _buildDiscreteHardwareMarkers(datasets)
        };
    }
    return {
        size: datasets.map((d) => d.markerSize || 0),
        shape: datasets.map((d) => d.markerShape || 'circle'),
        strokeWidth: 0,
        hover: { sizeOffset: 2 }
    };
}

function _hardwareMarkerPointCount(dataset) {
    return Array.isArray(dataset.data) ? dataset.data.length : 0;
}

function _hardwareMarkerTemplate(dataset, seriesIndex) {
    const color = dataset.color || '#006495';
    return {
        seriesIndex: seriesIndex,
        fillColor: color,
        strokeColor: color,
        size: dataset.markerSize || 4,
        shape: dataset.markerShape || 'circle'
    };
}

function _buildMarkersForDataset(dataset, seriesIndex) {
    const points = _hardwareMarkerPointCount(dataset);
    const step = Math.max(1, dataset.markerStep || 4);
    const offset = Math.max(0, dataset.markerOffset || 0);
    const template = _hardwareMarkerTemplate(dataset, seriesIndex);
    const markers = [];
    for (let pointIndex = offset; pointIndex < points; pointIndex += step) {
        markers.push({ ...template, dataPointIndex: pointIndex });
    }
    return markers;
}

function _buildDiscreteHardwareMarkers(datasets) {
    const markers = [];
    datasets.forEach((dataset, seriesIndex) => {
        markers.push(..._buildMarkersForDataset(dataset, seriesIndex));
    });
    return markers;
}

function _buildChartYaxis(percent, yaxisMax) {
    return {
        min: 0,
        max: yaxisMax,
        labels: {
            style: { fontSize: '10px', fontFamily: 'Outfit' },
            formatter: (val) => percent ? val.toFixed(0) + '%' : val.toFixed(1)
        }
    };
}

function _datasetLineStyleParts(d) {
    return [d.color || '', d.dashArray || 0, d.strokeWidth || 2];
}

function _datasetMarkerStyleParts(d) {
    return [d.markerShape || 'circle', d.markerSize || 0, d.markerStep || 0, d.markerOffset || 0];
}

function _datasetStyleSignatureParts(d) {
    return [d.label, ..._datasetLineStyleParts(d), ..._datasetMarkerStyleParts(d)];
}

function _datasetStyleSignature(datasets) {
    return datasets
        .map((d) => _datasetStyleSignatureParts(d).join('|'))
        .join('||');
}

function _needsFullChartUpdate(state, rangeMs, themeMode, yaxisMax, styleSig) {
    return state.rangeMs !== rangeMs || state.theme !== themeMode || state.yaxisMax !== yaxisMax || state.styleSig !== styleSig;
}

function _updateExistingChart(id, state, series, datasets, percent, isDark, themeMode, rangeMs, yaxisMax) {
    const styleSig = _datasetStyleSignature(datasets);
    if (!_needsFullChartUpdate(state, rangeMs, themeMode, yaxisMax, styleSig)) {
        charts[id].updateSeries(series);
        return;
    }
    _saveChartState(state, rangeMs, themeMode, yaxisMax, styleSig);
    charts[id].updateOptions({
        xaxis: { range: rangeMs },
        yaxis: _buildChartYaxis(percent, yaxisMax),
        series: series,
        theme: { mode: themeMode },
        colors: datasets.map((d) => d.color || '#006495'),
        stroke: _seriesStrokeStyles(datasets),
        markers: _seriesMarkerStyles(id, datasets),
        grid: { borderColor: isDark ? 'rgba(255, 255, 255, 0.08)' : '#e5e7eb' },
        tooltip: { theme: themeMode }
    });
}

function _createNewChart(id, el, state, options, datasets, rangeMs, themeMode, yaxisMax) {
    const styleSig = _datasetStyleSignature(datasets);
    _saveChartState(state, rangeMs, themeMode, yaxisMax, styleSig);
    charts[id] = new ApexCharts(el, options);
    charts[id].render();
}

function _saveChartState(state, rangeMs, themeMode, yaxisMax, styleSig) {
    state.rangeMs = rangeMs;
    state.theme = themeMode;
    state.yaxisMax = yaxisMax;
    state.styleSig = styleSig;
}
