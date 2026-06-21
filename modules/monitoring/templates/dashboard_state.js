const expandedElements = new Set();
let currentTab = 'active';
let charts = {};
let currentTelemetry = [];
let rollingTelemetryBuffer = [];
let chartWindowMinutes = 1;
let lastChartStates = {};
let fullTaskHistory = [];
let lastStatusData = null;
let refreshEnabled = true;

const COLORS = [
    '#006495', '#2e7d32', '#e65100', '#d81b60', '#5e35b1',
    '#00acc1', '#fb8c00', '#43a047', '#3949ab', '#8e24aa'
];
