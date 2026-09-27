const expandedElements = new Set();
const activeTaskTimeline = {};
let charts = {};
let lastChartStates = {};
// Mutable state that the feature scripts reassign lives on globalThis rather than in
// script-scoped `let` bindings. Each feature file is its own script (concatenated into
// one bundle in production, loaded one by one into a vm context by the unit tests), so a
// reassignment such as `chartWindowMinutes = 5` from charts.js would otherwise be an
// implicit global write as far as that file is concerned. Writers go through
// `globalThis.<name> = ...`; readers keep using the bare name, which resolves to the
// same global property.
globalThis.currentTab = 'active';
globalThis.currentTelemetry = [];
globalThis.rollingTelemetryBuffer = [];
globalThis.chartWindowMinutes = 1;
globalThis.fullTaskHistory = [];
globalThis.lastStatusData = null;
globalThis.refreshEnabled = true;
globalThis.refreshTimer = null;
globalThis.currentRefreshInterval = 2000;
globalThis.activeTaskFilter = 'all';
globalThis.historyTaskFilter = 'all';

const COLORS = [
    '#006495', '#2e7d32', '#e65100', '#d81b60', '#5e35b1',
    '#00acc1', '#fb8c00', '#43a047', '#3949ab', '#8e24aa'
];
