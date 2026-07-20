const fs = require("fs");
const http = require("http");
const path = require("path");

const HOST = "127.0.0.1";
const PORT = 9614;

const templateDir = path.resolve(__dirname, "../../modules/monitoring/templates");
const eventState = {
  statusCalls: 0,
  settingsSaves: [],
  historyClears: 0,
  telemetryClears: 0,
  logDownloads: 0,
  lifecycleScenario: "default",
  lifecycleTick: 0,
};

const BASE_NOW = Math.floor(Date.now() / 1000);

function defaultTelemetryHistoryPayload() {
  return [
    {
      timestamp: BASE_NOW - 8,
      system: {
        app_cpu_percent: 9.5,
        cpu_percent: 21.2,
        app_memory_gb: 1.05,
        memory_total_gb: 16,
        memory_used_gb: 6.8,
        memory_percent: 42.5,
      },
      telemetry: {
        hardware_util: { CPU: 24 },
        nvidia: [],
      },
    },
    {
      timestamp: BASE_NOW - 6,
      system: {
        app_cpu_percent: 10.8,
        cpu_percent: 23.6,
        app_memory_gb: 1.11,
        memory_total_gb: 16,
        memory_used_gb: 6.9,
        memory_percent: 43.1,
      },
      telemetry: {
        hardware_util: { CPU: 27 },
        nvidia: [],
      },
    },
    {
      timestamp: BASE_NOW - 4,
      system: {
        app_cpu_percent: 11.7,
        cpu_percent: 25.4,
        app_memory_gb: 1.16,
        memory_total_gb: 16,
        memory_used_gb: 7.0,
        memory_percent: 43.8,
      },
      telemetry: {
        hardware_util: { CPU: 29 },
        nvidia: [],
      },
    },
    {
      timestamp: BASE_NOW - 2,
      system: {
        app_cpu_percent: 12.4,
        cpu_percent: 26.1,
        app_memory_gb: 1.19,
        memory_total_gb: 16,
        memory_used_gb: 7.1,
        memory_percent: 44.4,
      },
      telemetry: {
        hardware_util: { CPU: 30 },
        nvidia: [],
      },
    },
  ];
}

function defaultHistoryPayload() {
  return [
    {
      task_id: "h-asr",
      filename: "history-asr.mkv",
      type: "/asr",
      completed_at: "now",
      status: "completed",
      video_duration: 120,
      active_elapsed_sec: 20,
      queue_elapsed_sec: 2,
      unit_id: "CPU",
      logs: ["ok"],
      result: { text: "asr text", segments: [{ start: 0, end: 1, text: "asr text" }] },
    },
    {
      task_id: "h-dl",
      filename: "history-detect.mkv",
      type: "/detect-language",
      completed_at: "now",
      status: "completed",
      video_duration: 50,
      active_elapsed_sec: 5,
      queue_elapsed_sec: 1,
      unit_id: "CPU",
      logs: ["ok"],
      result: { detected_language: "en" },
    },
    {
      task_id: "h-v1",
      filename: "history-v1.mkv",
      type: "/v1/audio/transcriptions",
      completed_at: "now",
      status: "completed",
      video_duration: 150,
      active_elapsed_sec: 25,
      queue_elapsed_sec: 1,
      unit_id: "CPU",
      logs: ["ok"],
      result: { text: "v1 text", segments: [{ start: 0, end: 1, text: "v1 text" }] },
    },
  ];
}

function historyStatsFromHistory(history) {
  const count = history.length;
  const totalDuration = history.reduce((acc, entry) => acc + Number(entry.video_duration || 0), 0);
  return {
    today: totalDuration,
    count_today: count,
    this_month: totalDuration + 480,
    all_time: totalDuration + 960,
    count_all_time: count + 9,
  };
}

function lifecycleBasePayload(tasks, history) {
  const activeSessions = tasks.filter((t) => t.status === "active").length;
  const queuedSessions = tasks.filter((t) => t.status === "queued").length;
  return {
    version: "1.2.0-test",
    active_sessions: activeSessions,
    queued_sessions: queuedSessions,
    uptime_sec: 3600,
    tasks,
    history,
    history_stats: historyStatsFromHistory(history),
    system: {
      app_cpu_percent: 12,
      cpu_percent: 24,
      app_memory_gb: 1.2,
      memory_total_gb: 16,
      memory_used_gb: 7,
      memory_percent: 44,
    },
    telemetry: {
      hardware_util: { CPU: 30 },
      nvidia: [],
    },
    telemetry_history: defaultTelemetryHistoryPayload(),
    hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
  };
}

function defaultStatusPayload() {
  const call = eventState.statusCalls;
  const liveText = call >= 2 ? "segment 2" : "segment 1";
  const progress = call >= 2 ? 52 : 47;

  const tasks = [
    {
      task_id: "asr-1",
      filename: "movie-asr.mkv",
      type: "/asr",
      status: "active",
      stage: "Inference",
      progress,
      video_duration: 120,
      start_time: BASE_NOW - 20,
      start_active: BASE_NOW - 15,
      unit_id: "CPU",
      logs: ["ASR running"],
      live_text: liveText,
    },
    {
      task_id: "dl-1",
      filename: "movie-detect.mkv",
      type: "/detect-language",
      status: "queued",
      stage: "Paused for Priority Task",
      progress: 0,
      video_duration: 60,
      start_time: BASE_NOW - 30,
      unit_id: null,
      logs: [],
    },
    {
      task_id: "v1-1",
      filename: "movie-v1.mkv",
      type: "/v1/audio/transcriptions",
      status: "queued",
      stage: "Queued",
      progress: 0,
      video_duration: 180,
      start_time: BASE_NOW - 10,
      unit_id: null,
      logs: [],
    },
  ];

  return lifecycleBasePayload(tasks, []);
}

function detectLangLifecycleScenario(tick) {
  const phases = [
    {
      tasks: [
        {
          task_id: "dl-life-1",
          filename: "detect-life.mkv",
          type: "/detect-language",
          status: "queued",
          stage: "Paused for Priority Task",
          progress: 5,
          video_duration: 60,
          start_time: BASE_NOW - 120,
          unit_id: null,
          logs: ["Queued behind priority lock"],
        },
        {
          task_id: "asr-life-anchor",
          filename: "anchor-asr.mkv",
          type: "/asr",
          status: "active",
          stage: "Inference",
          progress: 40,
          video_duration: 100,
          start_time: BASE_NOW - 110,
          start_active: BASE_NOW - 100,
          unit_id: "CPU",
          logs: ["Anchor ASR running"],
          live_text: "anchor segment",
        },
      ],
      history: [],
    },
    {
      tasks: [
        {
          task_id: "dl-life-1",
          filename: "detect-life.mkv",
          type: "/detect-language",
          status: "queued",
          stage: "Queued",
          progress: 10,
          video_duration: 60,
          start_time: BASE_NOW - 120,
          unit_id: null,
          logs: ["Waiting for available hardware"],
        },
      ],
      history: [],
    },
    {
      tasks: [
        {
          task_id: "dl-life-1",
          filename: "detect-life.mkv",
          type: "/detect-language",
          status: "active",
          stage: "Language Detection",
          progress: 70,
          video_duration: 60,
          start_time: BASE_NOW - 120,
          start_active: BASE_NOW - 60,
          unit_id: "CPU",
          logs: ["Detecting language"],
        },
      ],
      history: [],
    },
    {
      tasks: [
        {
          task_id: "dl-life-1",
          filename: "detect-life.mkv",
          type: "/detect-language",
          status: "post-processing",
          stage: "Post-Processing",
          progress: 99,
          video_duration: 60,
          start_time: BASE_NOW - 120,
          start_active: BASE_NOW - 60,
          unit_id: "CPU",
          logs: ["Persisting detection result"],
        },
      ],
      history: [],
    },
    {
      tasks: [],
      history: [
        {
          task_id: "dl-life-1",
          filename: "detect-life.mkv",
          type: "/detect-language",
          completed_at: "now",
          status: "completed",
          video_duration: 60,
          active_elapsed_sec: 7,
          queue_elapsed_sec: 5,
          unit_id: "CPU",
          logs: ["Detection done"],
          result: { detected_language: "en" },
        },
      ],
    },
  ];
  return phases[Math.min(tick, phases.length - 1)];
}

function asrLifecycleScenario(tick) {
  const phases = [
    {
      tasks: [
        {
          task_id: "asr-life-1",
          filename: "asr-life.mkv",
          type: "/asr",
          status: "queued",
          stage: "Queued",
          progress: 0,
          video_duration: 140,
          start_time: BASE_NOW - 90,
          unit_id: null,
          logs: ["Awaiting hardware"],
        },
      ],
      history: [],
    },
    {
      tasks: [
        {
          task_id: "asr-life-1",
          filename: "asr-life.mkv",
          type: "/asr",
          status: "active",
          stage: "Inference",
          progress: 45,
          video_duration: 140,
          start_time: BASE_NOW - 90,
          start_active: BASE_NOW - 55,
          unit_id: "CPU",
          logs: ["ASR inference running"],
          live_text: "asr segment 1",
        },
      ],
      history: [],
    },
    {
      tasks: [
        {
          task_id: "asr-life-1",
          filename: "asr-life.mkv",
          type: "/asr",
          status: "post-processing",
          stage: "Post-Processing",
          progress: 99,
          video_duration: 140,
          start_time: BASE_NOW - 90,
          start_active: BASE_NOW - 55,
          unit_id: "CPU",
          logs: ["Preparing final subtitles"],
          live_text: "asr segment final",
        },
      ],
      history: [],
    },
    {
      tasks: [],
      history: [
        {
          task_id: "asr-life-1",
          filename: "asr-life.mkv",
          type: "/asr",
          completed_at: "now",
          status: "completed",
          video_duration: 140,
          active_elapsed_sec: 30,
          queue_elapsed_sec: 4,
          unit_id: "CPU",
          logs: ["ASR done"],
          result: { text: "asr lifecycle text", segments: [{ start: 0, end: 1, text: "asr lifecycle text" }] },
        },
      ],
    },
  ];
  return phases[Math.min(tick, phases.length - 1)];
}

function v1LifecycleScenario(tick) {
  const phases = [
    {
      tasks: [
        {
          task_id: "v1-life-1",
          filename: "v1-life.mkv",
          type: "/v1/audio/transcriptions",
          status: "queued",
          stage: "Queued",
          progress: 0,
          video_duration: 150,
          start_time: BASE_NOW - 80,
          unit_id: null,
          logs: ["V1 request queued"],
        },
      ],
      history: [],
    },
    {
      tasks: [
        {
          task_id: "v1-life-1",
          filename: "v1-life.mkv",
          type: "/v1/audio/transcriptions",
          status: "active",
          stage: "Inference",
          progress: 55,
          video_duration: 150,
          start_time: BASE_NOW - 80,
          start_active: BASE_NOW - 40,
          unit_id: "CPU",
          logs: ["V1 transcription running"],
          live_text: "v1 segment 1",
        },
      ],
      history: [],
    },
    {
      tasks: [
        {
          task_id: "v1-life-1",
          filename: "v1-life.mkv",
          type: "/v1/audio/transcriptions",
          status: "post-processing",
          stage: "Post-Processing",
          progress: 99,
          video_duration: 150,
          start_time: BASE_NOW - 80,
          start_active: BASE_NOW - 40,
          unit_id: "CPU",
          logs: ["V1 formatting output"],
          live_text: "v1 final",
        },
      ],
      history: [],
    },
    {
      tasks: [],
      history: [
        {
          task_id: "v1-life-1",
          filename: "v1-life.mkv",
          type: "/v1/audio/transcriptions",
          completed_at: "now",
          status: "completed",
          video_duration: 150,
          active_elapsed_sec: 27,
          queue_elapsed_sec: 3,
          unit_id: "CPU",
          logs: ["V1 done"],
          result: { text: "v1 lifecycle text", segments: [{ start: 0, end: 1, text: "v1 lifecycle text" }] },
        },
      ],
    },
  ];
  return phases[Math.min(tick, phases.length - 1)];
}

function mixedLifecycleScenario(tick) {
  const phases = [
    {
      tasks: [
        {
          task_id: "mix-a-active",
          filename: "mix-active-a.mkv",
          type: "/asr",
          status: "active",
          stage: "Inference",
          progress: 60,
          video_duration: 100,
          start_time: BASE_NOW - 120,
          start_active: BASE_NOW - 100,
          unit_id: "CPU",
          logs: ["active A"],
          live_text: "active A text",
        },
        {
          task_id: "mix-b-active",
          filename: "mix-active-b.mkv",
          type: "/v1/audio/transcriptions",
          status: "active",
          stage: "Inference",
          progress: 35,
          video_duration: 130,
          start_time: BASE_NOW - 110,
          start_active: BASE_NOW - 90,
          unit_id: "CPU",
          logs: ["active B"],
          live_text: "active B text",
        },
        {
          task_id: "mix-priority-queued",
          filename: "mix-priority-queued.mkv",
          type: "/detect-language",
          status: "queued",
          stage: "Paused for Priority Task",
          progress: 0,
          video_duration: 60,
          start_time: BASE_NOW - 105,
          is_priority: true,
          unit_id: null,
          logs: ["priority queued"],
        },
        {
          task_id: "mix-standard-queued",
          filename: "mix-standard-queued.mkv",
          type: "/asr",
          status: "queued",
          stage: "Queued",
          progress: 0,
          video_duration: 150,
          start_time: BASE_NOW - 100,
          is_priority: false,
          unit_id: null,
          logs: ["standard queued"],
        },
      ],
      history: [],
    },
    {
      tasks: [],
      history: defaultHistoryPayload(),
    },
  ];
  return phases[Math.min(tick, phases.length - 1)];
}

function lifecycleScenarioPayload() {
  const tick = eventState.lifecycleTick;
  const scenario = eventState.lifecycleScenario;

  let phase;
  if (scenario === "lifecycle-detectlang") {
    phase = detectLangLifecycleScenario(tick);
  } else if (scenario === "lifecycle-asr") {
    phase = asrLifecycleScenario(tick);
  } else if (scenario === "lifecycle-v1") {
    phase = v1LifecycleScenario(tick);
  } else if (scenario === "lifecycle-mixed") {
    phase = mixedLifecycleScenario(tick);
  } else {
    return defaultStatusPayload();
  }

  return lifecycleBasePayload(phase.tasks, phase.history);
}

function read(fileName) {
  return fs.readFileSync(path.join(templateDir, fileName), "utf8");
}

function readManifestLines(fileName) {
  return read(fileName)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
}

function buildDashboardHtml() {
  let html = read("dashboard.html");
  const css = read("dashboard.css");
  const jsBundle = readManifestLines("dashboard_js_files.txt")
    .map((name) => read(name))
    .join("\n\n");

  html = html.replace("/* {{DASHBOARD_CSS}} */", css);
  html = html.replace("// {{DASHBOARD_JS}}", jsBundle);

  // Avoid network dependency for chart library in E2E.
  html = html.replace(
    '<script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>',
    '<script src="/__apexcharts_stub.js"></script>'
  );

  return html;
}

const dashboardHtml = buildDashboardHtml();

function buildAnalyticsHtml() {
  let html = read("analytics.html");
  const css = read("analytics.css");
  const js = readManifestLines("analytics_js_files.txt")
    .map((name) => read(name))
    .join("\n\n");

  html = html.replace("/* {{ANALYTICS_CSS}} */", css);
  html = html.replace("// {{ANALYTICS_JS}}", js);
  html = html.replace(
    '<script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>',
    '<script src="/__apexcharts_stub.js"></script>'
  );

  return html;
}

const analyticsHtml = buildAnalyticsHtml();

function buildStatusPayload() {
  eventState.statusCalls += 1;
  return lifecycleScenarioPayload();
}

const analyticsPayload = {
  cumulative: {
    all_time: 7265,
    this_month: 3600,
    today: 1250,
    count_all_time: 15,
    count_today: 4,
    asr: { count: 8, duration: 4020 },
    isolations: { count: 3, duration: 930 },
    audio: { count: 4, duration: 2315 },
  },
  daily: {
    "2026-07-12": {
      count: 3,
      duration: 1500,
      asr: { count: 2, duration: 900 },
      detectlang: { count: 1, duration: 600 },
      audio: { count: 0, duration: 0 },
    },
    "2026-07-13": {
      count: 5,
      duration: 2500,
      asr: { count: 3, duration: 1400 },
      isolations: { count: 1, duration: 500 },
      audio: { count: 1, duration: 600 },
    },
    "2026-07-14": {
      count: 4,
      duration: 1800,
      asr: { count: 2, duration: 1000 },
      detectlang: { count: 1, duration: 300 },
      audio: { count: 1, duration: 500 },
    },
  },
};

function respondJson(res, payload) {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify(payload));
}

function readJsonBody(req) {
  return new Promise((resolve) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        resolve({});
      }
    });
  });
}

const apexStub = `
class ApexCharts {
  constructor() {}
  render() { return Promise.resolve(); }
  updateOptions() { return Promise.resolve(); }
  updateSeries() { return Promise.resolve(); }
  destroy() { return Promise.resolve(); }
}
window.ApexCharts = ApexCharts;
`;

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (url.pathname === "/healthz") {
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("ok");
    return;
  }

  if (url.pathname === "/dashboard" || url.pathname === "/") {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(dashboardHtml);
    return;
  }

  if (url.pathname === "/analytics") {
    const accept = String(req.headers.accept || "").toLowerCase();
    if (accept.includes("application/json")) {
      respondJson(res, analyticsPayload);
      return;
    }
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    res.end(analyticsHtml);
    return;
  }

  if (url.pathname === "/status") {
    respondJson(res, buildStatusPayload());
    return;
  }

  if (url.pathname === "/history") {
    if (eventState.lifecycleScenario === "default") {
      respondJson(res, defaultHistoryPayload());
      return;
    }
    const statusPayload = lifecycleScenarioPayload();
    respondJson(res, statusPayload.history || []);
    return;
  }

  if (url.pathname === "/settings") {
    if (req.method === "POST") {
      readJsonBody(req).then((payload) => {
        eventState.settingsSaves.push(payload);
        respondJson(res, { ok: true });
      });
      return;
    }
    respondJson(res, {});
    return;
  }

  if (url.pathname === "/system/history/clear" && req.method === "POST") {
    eventState.historyClears += 1;
    respondJson(res, { ok: true });
    return;
  }

  if (url.pathname === "/system/telemetry/clear" && req.method === "POST") {
    eventState.telemetryClears += 1;
    respondJson(res, { ok: true });
    return;
  }

  if (url.pathname === "/logs/download") {
    eventState.logDownloads += 1;
    res.writeHead(200, {
      "Content-Type": "text/plain; charset=utf-8",
      "Content-Disposition": 'attachment; filename="whisper_pro.log"',
      "Cache-Control": "no-store",
    });
    res.end("2026-07-14 09:00:00 INFO Whisper Pro fixture log line\n");
    return;
  }

  if (url.pathname === "/__reset" && req.method === "POST") {
    // Reset all event counters so parallel spec files start from a clean slate
    eventState.statusCalls = 0;
    eventState.settingsSaves = [];
    eventState.historyClears = 0;
    eventState.telemetryClears = 0;
    eventState.logDownloads = 0;
    eventState.lifecycleScenario = "default";
    eventState.lifecycleTick = 0;
    respondJson(res, { ok: true });
    return;
  }

  if (url.pathname === "/__lifecycle/reset" && req.method === "POST") {
    eventState.lifecycleScenario = "default";
    eventState.lifecycleTick = 0;
    respondJson(res, { ok: true, scenario: eventState.lifecycleScenario, tick: eventState.lifecycleTick });
    return;
  }

  if (url.pathname === "/__lifecycle/scenario" && req.method === "POST") {
    readJsonBody(req).then((payload) => {
      const requested = String((payload && payload.name) || "default");
      const allowed = new Set(["default", "lifecycle-detectlang", "lifecycle-asr", "lifecycle-v1", "lifecycle-mixed"]);
      eventState.lifecycleScenario = allowed.has(requested) ? requested : "default";
      eventState.lifecycleTick = 0;
      respondJson(res, { ok: true, scenario: eventState.lifecycleScenario, tick: eventState.lifecycleTick });
    });
    return;
  }

  if (url.pathname === "/__lifecycle/advance" && req.method === "POST") {
    readJsonBody(req).then((payload) => {
      const delta = Number((payload && payload.delta) || 1);
      const step = Number.isFinite(delta) && delta > 0 ? Math.floor(delta) : 1;
      eventState.lifecycleTick += step;
      respondJson(res, { ok: true, scenario: eventState.lifecycleScenario, tick: eventState.lifecycleTick });
    });
    return;
  }

  if (url.pathname === "/__lifecycle/state") {
    respondJson(res, { scenario: eventState.lifecycleScenario, tick: eventState.lifecycleTick });
    return;
  }

  if (url.pathname === "/__events") {
    respondJson(res, eventState);
    return;
  }

  if (url.pathname === "/__apexcharts_stub.js") {
    res.writeHead(200, { "Content-Type": "application/javascript" });
    res.end(apexStub);
    return;
  }

  res.writeHead(404, { "Content-Type": "text/plain" });
  res.end("Not Found");
});

server.listen(PORT, HOST, () => {
  console.log(`Dashboard fixture server running at http://${HOST}:${PORT}`);
});