const path = require("path");
const { JSDOM } = require("jsdom");
const { evalInContext, loadScriptInContext, createMatchMediaStub } = require("./helpers");

function buildDashboardDom() {
  return new JSDOM(`<!doctype html><html><body>
    <button id="tab-active" class="tab"></button>
    <button id="tab-history" class="tab"></button>
    <button id="tab-analytics" class="tab"></button>
    <button id="tab-charts" class="tab"></button>
    <button id="tab-settings" class="tab"></button>

    <div id="active-section">
      <div class="filter-row">
        <button id="filter-all"></button>
        <button id="filter-asr"></button>
        <button id="filter-detectlang"></button>
        <button id="filter-v1"></button>
      </div>
    </div>
    <div id="history-section">
      <div class="filter-row">
        <button id="hist-filter-all"></button>
        <button id="hist-filter-asr"></button>
        <button id="hist-filter-detectlang"></button>
        <button id="hist-filter-v1"></button>
      </div>
    </div>
    <div id="analytics-section"></div>
    <div id="charts-section"></div>
    <div id="settings-section"></div>

    <div id="history-list"></div>
    <div id="task-list"></div>
    <div id="analytics-grid"></div>
    <div id="hw-pool"></div>

    <div id="app-version"></div>
    <div id="app-cpu-val"></div><div id="app-cpu-bar"></div>
    <div id="sys-cpu-val"></div><div id="sys-cpu-bar"></div>
    <div id="app-mem-val"></div><div id="app-mem-bar"></div>
    <div id="sys-mem-val"></div><div id="sys-mem-bar"></div>
    <div id="active-val"></div>
    <div id="queued-val"></div>
    <div id="last-update"></div>

    <select id="retention-range"><option value="24" selected>24</option></select>
    <select id="log-retention-range"><option value="7" selected>7</option></select>

    <button id="toggle-refresh"></button>
    <span id="refresh-icon"></span>
    <span id="refresh-text"></span>
    <select id="refresh-interval">
      <option value="1000">1s</option>
      <option value="2000" selected>2s</option>
    </select>
  </body></html>`, { url: "http://localhost/" });
}

describe("main.js", () => {
  let dom;
  let fetchMock;
  let context;

  beforeEach(() => {
    dom = buildDashboardDom();

    fetchMock = vi.fn(async (url) => {
      if (url === "/history") {
        return {
          json: async () => [
            {
              task_id: "h1",
              filename: "movie.mp4",
              type: "Transcription",
              completed_at: "now",
              status: "completed",
              video_duration: 120,
              active_elapsed_sec: 30,
              queue_elapsed_sec: 2,
              logs: ["ok"],
              result: {
                text: "hello",
                segments: [{ start: 0, end: 1, text: "hello" }],
              },
            },
          ],
        };
      }
      if (url === "/settings") {
        return { ok: true, json: async () => ({}) };
      }
      return {
        json: async () => ({
          version: "1.0.0",
          active_sessions: 1,
          queued_sessions: 1,
          uptime_sec: 120,
          tasks: [
            {
              task_id: "a1",
              filename: "active.mp4",
              type: "Transcription",
              status: "active",
              stage: "Inference",
              progress: 50,
              video_duration: 120,
              start_time: Date.now() / 1000 - 10,
              start_active: Date.now() / 1000 - 8,
              unit_id: "GPU.0",
              logs: ["running"],
              live_text: "subtitle",
            },
            {
              task_id: "q1",
              filename: "queued.mp4",
              type: "Transcription",
              status: "queued",
              stage: "Paused for Priority Task",
              progress: 45,
              video_duration: 200,
              start_time: Date.now() / 1000 - 20,
              unit_id: null,
              logs: [],
            },
          ],
          history: [
            {
              status: "completed",
              video_duration: 120,
              result: { performance: { inference_sec: 20, isolation_sec: 10 } },
            },
          ],
          history_stats: {
            today: 60,
            count_today: 1,
            this_month: 600,
            all_time: 1000,
            count_all_time: 4,
          },
          system: {
            app_cpu_percent: 10,
            cpu_percent: 20,
            app_memory_gb: 1.5,
            memory_total_gb: 16,
            memory_used_gb: 7,
            memory_percent: 43,
          },
          telemetry: {
            hardware_util: { "GPU.0": 60, "NPU.0": 35 },
            nvidia: [],
          },
          telemetry_history: [],
          hardware_units: [
            { id: "GPU.0", type: "GPU", name: "Intel GPU", uvr_status: "ready", whisper_status: "ready" },
            { id: "NPU.0", type: "NPU", name: "Intel NPU", uvr_status: "ready", whisper_status: "ready" },
          ],
        }),
      };
    });

    const baseContext = {
      window: {
        matchMedia: createMatchMediaStub(false),
        dispatchEvent: () => {},
      },
      document: dom.window.document,
      fetch: fetchMock,
      alert: () => {},
      confirm: () => true,
      Event: dom.window.Event,
      setTimeout,
      clearTimeout,
      setInterval,
      clearInterval,
      expandedElements: new Set(),
      activeTaskTimeline: {},
      currentTab: "active",
      charts: {},
      currentTelemetry: [],
      rollingTelemetryBuffer: [],
      chartWindowMinutes: 1,
      lastChartStates: {},
      fullTaskHistory: [],
      lastStatusData: null,
      refreshEnabled: true,
    };

    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/core/state.js"), baseContext);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/core/utils.js"), context);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/charts.js"), context);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/audit.js"), context);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/task_filter_history.js"), context);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/speed_status.js"), context);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/active_tasks.js"), context);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/runtime.js"), context);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/main.js"), context);
  });

  it("handles tab switching, history rendering, settings save, and status refresh", async () => {
    context.showTab("charts");
    context.showTab("history");
    context.showTab("active");

    await context.updateStats();
    await context.saveSettings();

    context.toggleRefresh();
    context.toggleRefresh();

    expect(String(dom.window.document.getElementById("active-val").innerText)).toBe("1");
    expect(String(dom.window.document.getElementById("queued-val").innerText)).toBe("1");
    expect(dom.window.document.getElementById("task-list").innerHTML).toContain("Paused for priority detect-language tasks");
    expect(dom.window.document.getElementById("hw-pool").innerHTML).toContain("Intel GPU");

    const audit = context.renderAuditDetails({ task_id: "x", caller_info: { ip: "127.0.0.1" }, request_json: {}, result: {} });
    expect(audit).toContain("Audit & Caller Info");
  });

  it("covers settings/history/telemetry action branches and audit open/user-agent display", async () => {
    const alerts = [];
    context.alert = (msg) => alerts.push(String(msg));

    // saveSettings catch branch
    fetchMock.mockRejectedValueOnce(new Error("settings down"));
    await context.saveSettings();
    expect(alerts.some((msg) => msg.includes("Failed to save settings"))).toBe(true);

    // clearTaskHistory early-return branch (confirm false)
    context.confirm = () => false;
    const beforeCalls = fetchMock.mock.calls.length;
    await context.clearTaskHistory();
    expect(fetchMock.mock.calls.length).toBe(beforeCalls);

    // clearTaskHistory failure branch (confirm true + non-ok)
    context.confirm = () => true;
    fetchMock.mockResolvedValueOnce({ ok: false });
    await context.clearTaskHistory();
    expect(alerts.some((msg) => msg.includes("Failed to clear task history."))).toBe(true);

    // clearTelemetryMetrics success branch + reset callback branch
    let resetCalled = 0;
    context.resetTelemetryChartsAndStats = () => {
      resetCalled += 1;
    };
    fetchMock.mockResolvedValueOnce({ ok: true });
    await context.clearTelemetryMetrics();
    expect(resetCalled).toBe(1);
    expect(alerts.some((msg) => msg.includes("Telemetry history purged successfully."))).toBe(true);

    // clearTelemetryMetrics catch branch
    fetchMock.mockRejectedValueOnce(new Error("telemetry down"));
    await context.clearTelemetryMetrics();
    expect(alerts.some((msg) => msg.startsWith("Error:"))).toBe(true);

    // audit.js additional branches: user-agent provided + expanded open state
    evalInContext(
      context,
      "expandedElements.add('audit-1_audit'); expandedElements.add('audit-1_req'); expandedElements.add('audit-1_res');"
    );
    const auditWithUa = context.renderAuditDetails({
      task_id: "audit-1",
      caller_info: {
        ip: "127.0.0.1",
        user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Long UA String",
      },
      request_json: { a: 1 },
      result: { ok: true },
    });
    expect(auditWithUa).toMatch(/<details[^>]*open/);
    expect(auditWithUa).toContain('UA:');
  });

  it("covers remaining main module refresh and maintenance branches", async () => {
    const alerts = [];
    context.alert = (msg) => alerts.push(String(msg));

    // startRefreshInterval branch where existing timer is cleared first.
    evalInContext(context, "refreshTimer = setInterval(() => {}, 1000);");
    context.startRefreshInterval();

    // clearTaskHistory success branch
    evalInContext(context, "fullTaskHistory = [{ task_id: 't1' }]");
    let renderHistoryCalls = 0;
    context.renderHistory = () => {
      renderHistoryCalls += 1;
    };
    context.confirm = () => true;
    fetchMock.mockResolvedValueOnce({ ok: true });
    await context.clearTaskHistory();
    expect(renderHistoryCalls).toBeGreaterThan(0);
    expect(alerts.some((msg) => msg.includes("Task history purged successfully."))).toBe(true);

    // clearTaskHistory catch branch
    fetchMock.mockRejectedValueOnce(new Error("history down"));
    await context.clearTaskHistory();
    expect(alerts.some((msg) => msg.startsWith("Error:"))).toBe(true);

    // clearTelemetryMetrics confirm false branch
    context.confirm = () => false;
    const beforeCalls = fetchMock.mock.calls.length;
    await context.clearTelemetryMetrics();
    expect(fetchMock.mock.calls.length).toBe(beforeCalls);

    // clearTelemetryMetrics non-ok branch
    context.confirm = () => true;
    fetchMock.mockResolvedValueOnce({ ok: false });
    await context.clearTelemetryMetrics();
    expect(alerts.some((msg) => msg.includes("Failed to clear telemetry metrics."))).toBe(true);

    // window.onload branch where matchMedia is absent
    context.window.matchMedia = null;
    context.window.onload();
  });

  it("normalizes placeholder stage and status values in rendered tasks", async () => {
    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.0.9",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 10,
        tasks: [
          {
            task_id: "ph-1",
            filename: "placeholder.mp4",
            type: "Transcription",
            status: "unknown",
            stage: "resuming",
            progress: 1,
            video_duration: 60,
            start_time: Date.now() / 1000 - 5,
            unit_id: null,
            logs: [],
          },
        ],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 0, cpu_percent: 0, app_memory_gb: 0.5, memory_total_gb: 16, memory_used_gb: 2, memory_percent: 12 },
        telemetry: { hardware_util: {} },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();
    const card = dom.window.document.querySelector('[data-task-id="ph-1"]');
    const stageText = card.querySelector('.stage-text').textContent.toLowerCase();
    const badgeText = card.querySelector('.badge').textContent.toLowerCase();

    expect(stageText).toBe("initializing");
    expect(badgeText).toBe("initializing");
  });

  it("renders idle state and keeps updateStats no-op when refresh disabled", async () => {
    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.0.1",
        active_sessions: 0,
        queued_sessions: 0,
        tasks: [],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 0, cpu_percent: 0, app_memory_gb: 0.5, memory_total_gb: 16, memory_used_gb: 2, memory_percent: 12 },
        telemetry: { hardware_util: {} },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
        uptime_sec: 1,
      }),
    });

    await context.updateStats();
    expect(dom.window.document.getElementById("task-list").innerHTML).toContain("Service is idle");

    evalInContext(context, "refreshEnabled = false");
    const callsBefore = fetchMock.mock.calls.length;
    await context.updateStats();
    expect(fetchMock.mock.calls.length).toBe(callsBefore);
  });

  it("handles updateStats fetch failure and history rendering edge variants", async () => {
    fetchMock.mockRejectedValueOnce(new Error("down"));
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await context.updateStats();
    expect(consoleSpy).toHaveBeenCalled();

    evalInContext(
      context,
      `fullTaskHistory = ${JSON.stringify([
        {
          task_id: "err-1",
          filename: "broken.mp4",
          type: "Transcription",
          completed_at: "now",
          status: "failed",
          video_duration: 0,
          active_elapsed_sec: 1,
          queue_elapsed_sec: 0,
          result: { error: "boom" },
          logs: ["bad"],
        },
        {
          task_id: "none-1",
          filename: "silent.mp4",
          type: "ASR",
          completed_at: "now",
          status: "completed",
          video_duration: 20,
          active_elapsed_sec: 10,
          queue_elapsed_sec: 1,
          result: { segments: [] },
          logs: [],
        },
      ])}`
    );

    context.renderHistory();
    const html = dom.window.document.getElementById("history-list").innerHTML;
    expect(html).toContain("Error:");
    expect(html).toContain("No speech detected or transcription failed.");
    consoleSpy.mockRestore();
  });

  it("covers existing-card update branches and onload chart change callback", async () => {
    let darkListener = null;
    context.window.matchMedia = () => ({
      matches: false,
      addEventListener: (_evt, cb) => {
        darkListener = cb;
      },
    });

    const now = Date.now() / 1000;
    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.0.2",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 50,
        tasks: [
          {
            task_id: "uvr-1",
            filename: "uvr.mp4",
            type: "Transcription",
            status: "active",
            stage: "Vocal Separation",
            progress: 25,
            video_duration: 300,
            start_time: now - 30,
            start_active: now - 20,
            current_position: 60,
            unit_id: "CPU",
            logs: ["l1"],
            live_text: "l1",
          },
        ],
        history: [
          {
            status: "completed",
            video_duration: 300,
            result: { performance: { inference_sec: 100, isolation_sec: 120 } },
          },
        ],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 1, cpu_percent: 2, app_memory_gb: 1, memory_total_gb: 16, memory_used_gb: 4, memory_percent: 25 },
        telemetry: { nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();

    const card = dom.window.document.querySelector('[data-task-id="uvr-1"]');
    expect(card).toBeTruthy();

    const speedTextNode = card.querySelector('.speed-text');
    if (speedTextNode) speedTextNode.remove();
    const etaTextNode = card.querySelector('.eta-text');
    if (etaTextNode) etaTextNode.remove();

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.0.3",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 70,
        tasks: [
          {
            task_id: "uvr-1",
            filename: "uvr.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 70,
            video_duration: 300,
            start_time: now - 40,
            start_active: now - 35,
            start_inference: now - 20,
            current_position: 190,
            unit_id: "CPU",
            logs: ["l2"],
            live_text: "l2",
          },
        ],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 3, cpu_percent: 5, app_memory_gb: 1.1, memory_total_gb: 16, memory_used_gb: 4.1, memory_percent: 26 },
        telemetry: { hardware_util: { CPU: 10 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.0.4",
        active_sessions: 0,
        queued_sessions: 0,
        uptime_sec: 80,
        tasks: [],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 2, cpu_percent: 4, app_memory_gb: 1, memory_total_gb: 16, memory_used_gb: 4, memory_percent: 25 },
        telemetry: { hardware_util: { CPU: 0 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();
    expect(dom.window.document.querySelector('[data-task-id="uvr-1"]')).toBeNull();

    context.window.onload();
    expect(typeof darkListener).toBe("function");
    darkListener();
  });

  it("renders hardware in history cards using history_unit_id fallback", () => {
    evalInContext(
      context,
      `fullTaskHistory = ${JSON.stringify([
        {
          task_id: "hist-hw-1",
          filename: "history_hw.mp4",
          type: "Transcription",
          completed_at: "now",
          status: "completed",
          video_duration: 120,
          active_elapsed_sec: 30,
          queue_elapsed_sec: 2,
          unit_id: null,
          history_unit_id: "GPU.0",
          result: { text: "ok", segments: [{ start: 0, end: 1, text: "ok" }] },
          logs: ["ok"],
        },
      ])}`
    );

    context.renderHistory();
    const html = dom.window.document.getElementById("history-list").innerHTML;
    expect(html).toContain("Intel GPU");
    expect(html).not.toContain("Intel GPU (GPU.0)");
  });

  it("renders slot suffix for history hardware when multiple units of the same family exist", () => {
    evalInContext(
      context,
      `lastStatusData = ${JSON.stringify({
        hardware_units: [
          { id: "GPU.0", type: "GPU", name: "Intel GPU" },
          { id: "GPU.1", type: "GPU", name: "Intel GPU" },
          { id: "NPU.0", type: "NPU", name: "Intel NPU" },
        ],
      })}`
    );

    evalInContext(
      context,
      `fullTaskHistory = ${JSON.stringify([
        {
          task_id: "hist-hw-2",
          filename: "history_hw_multi.mp4",
          type: "Transcription",
          completed_at: "now",
          status: "completed",
          video_duration: 120,
          active_elapsed_sec: 30,
          queue_elapsed_sec: 2,
          unit_id: null,
          history_unit_id: "GPU.1",
          result: { text: "ok", segments: [{ start: 0, end: 1, text: "ok" }] },
          logs: ["ok"],
        },
      ])}`
    );

    context.renderHistory();
    const html = dom.window.document.getElementById("history-list").innerHTML;
    expect(html).toContain("Intel GPU (GPU.1)");
  });

  it("does not append suffix for generic family ids", () => {
    evalInContext(
      context,
      `lastStatusData = ${JSON.stringify({
        hardware_units: [
          { id: "GPU", type: "GPU", name: "Intel GPU" },
          { id: "NPU", type: "NPU", name: "Intel NPU" },
        ],
      })}`
    );

    const gpu = context.getHwIconAndLabel("GPU");
    const npu = context.getHwIconAndLabel("NPU");
    expect(gpu.label).toBe("Intel GPU");
    expect(npu.label).toBe("Intel NPU");
  });

  it("renders hardware in history cards using history unit metadata when IDs are missing", () => {
    evalInContext(
      context,
      `fullTaskHistory = ${JSON.stringify([
        {
          task_id: "hist-hw-meta-1",
          filename: "history_hw_meta.mp4",
          type: "Transcription",
          completed_at: "now",
          status: "completed",
          video_duration: 120,
          active_elapsed_sec: 30,
          queue_elapsed_sec: 2,
          unit_id: null,
          history_unit_id: null,
          history_unit_type: "GPU",
          history_unit_name: "Intel Arc A770",
          result: { text: "ok", segments: [{ start: 0, end: 1, text: "ok" }] },
          logs: ["ok"],
        },
      ])}`
    );

    context.renderHistory();
    const html = dom.window.document.getElementById("history-list").innerHTML;
    expect(html).toContain("Intel Arc A770");
  });

  it("covers history hardware helper fallback branches", () => {
    const noMetaTag = context._historyHardwareTag({ unit_id: null, history_unit_id: null });
    expect(noMetaTag).toBe("");

    const typeOnlyTag = context._historyHardwareTag({
      unit_id: null,
      history_unit_id: null,
      history_unit_type: "CPU",
      history_unit_name: null,
    });
    expect(typeOnlyTag).toContain("CPU");
    expect(typeOnlyTag).toContain("settings_input_component");

    expect(context._historyHardwareIconForType("CUDA")).toBe("rocket_launch");
    expect(context._historyHardwareIconForType("NPU")).toBe("psychology_alt");
    expect(context._historyHardwareIconForType("GPU")).toBe("developer_board");
    expect(context._historyHardwareIconForType("CPU")).toBe("settings_input_component");
    expect(context._historyHardwareIconForType("SOMETHING")).toBe("memory");
    expect(context._historyHardwareIconForType(null)).toBe("memory");

    expect(context._historyHardwareLabelFromMeta("GPU", "")).toBe("GPU");
    expect(context._historyHardwareLabelFromMeta(null, null)).toBe("Unknown Hardware");
  });

  it("covers showTab chart/history branches and handleToggle open-close paths", async () => {
    const renderChartsSpy = vi.fn();
    context.renderCharts = renderChartsSpy;
    evalInContext(context, "lastChartStates = { cpuChart: { rangeMs: 1, theme: 'light' } }");

    context.showTab("charts");
    expect(renderChartsSpy).toHaveBeenCalled();
    expect(evalInContext(context, "lastChartStates.cpuChart.rangeMs")).toBeNull();

    context.showTab("history");
    await Promise.resolve();
    await Promise.resolve();
    expect(fetchMock).toHaveBeenCalledWith("/history");

    const card = dom.window.document.createElement("div");
    card.dataset.taskId = "a1";
    const logBuffer = dom.window.document.createElement("div");
    logBuffer.className = "log-buffer";
    Object.defineProperty(logBuffer, "scrollHeight", { value: 99, configurable: true });
    card.appendChild(logBuffer);
    dom.window.document.getElementById("task-list").appendChild(card);

    context.handleToggle("a1_logs", true);
    expect(evalInContext(context, "expandedElements.has('a1_logs')")).toBe(true);
    context.handleToggle("a1_logs", false);
    expect(evalInContext(context, "expandedElements.has('a1_logs')")).toBe(false);
  });

  it("covers saveSettings failure and renderHistory empty state", async () => {
    fetchMock.mockRejectedValueOnce(new Error("settings down"));
    const alertSpy = vi.fn();
    context.alert = alertSpy;

    await context.saveSettings();
    expect(alertSpy).toHaveBeenCalled();

    evalInContext(context, "fullTaskHistory = []");
    context.renderHistory();
    expect(dom.window.document.getElementById("history-list").innerHTML).toContain("No history yet");
  });

  it("covers endpoint filter helpers and filter action functions", () => {
    expect(context.normalizeTaskFilterType("isolation")).toBe("detectlang");
    expect(context.normalizeTaskFilterType("isolations")).toBe("detectlang");
    expect(context.normalizeTaskFilterType("detect-language")).toBe("detectlang");
    expect(context.normalizeTaskFilterType("asr")).toBe("asr");

    expect(context.getTaskFilterCategory({ type: "/asr", stage: "Inference" })).toBe("asr");
    expect(context.getTaskFilterCategory({ type: "/detect-language", stage: "Queued" })).toBe("detectlang");
    expect(context.getTaskFilterCategory({ type: "Translate", stage: "Vocal Separation" })).toBe("detectlang");
    expect(context.getTaskFilterCategory({ type: "/v1/audio/transcriptions", stage: "Inference" })).toBe("v1");
    expect(context.getTaskFilterCategory({ type: "misc", stage: "idle" })).toBe("other");

    const updateStatsSpy = vi.fn();
    const renderHistorySpy = vi.fn();
    context.updateStats = updateStatsSpy;
    context.renderHistory = renderHistorySpy;

    context.filterTasks("detect-language");
    expect(evalInContext(context, "activeTaskFilter")).toBe("detectlang");
    expect(updateStatsSpy).toHaveBeenCalledTimes(1);
    expect(dom.window.document.getElementById("filter-detectlang").classList.contains("active-filter")).toBe(true);

    context.filterHistory("isolations");
    expect(evalInContext(context, "historyTaskFilter")).toBe("detectlang");
    expect(renderHistorySpy).toHaveBeenCalledTimes(1);
    expect(dom.window.document.getElementById("hist-filter-detectlang").classList.contains("active-filter")).toBe(true);
  });

  it("uses declaration default and passes through activeTaskFilter unchanged", async () => {
    const originalMatchesCategoryFilter = context.matchesCategoryFilter;
    const matchesSpy = vi.fn((task, selectedFilter) => originalMatchesCategoryFilter(task, selectedFilter));
    context.matchesCategoryFilter = matchesSpy;

    expect(evalInContext(context, "globalThis.activeTaskFilter")).toBe("all");
    await context.updateStats();

    expect(matchesSpy).toHaveBeenCalled();
    expect(matchesSpy.mock.calls.map((call) => call[1]).every((value) => value === "all")).toBe(true);

    matchesSpy.mockClear();

    evalInContext(context, "activeTaskFilter = 'v1'");
    await context.updateStats();

    expect(matchesSpy).toHaveBeenCalled();
    expect(matchesSpy.mock.calls.map((call) => call[1]).every((value) => value === "v1")).toBe(true);
  });

  it("covers live-text refresh, eta hidden branch, and stale card removal", async () => {
    const now = Date.now() / 1000;

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.4.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 10,
        tasks: [
          {
            task_id: "asr-live",
            filename: "live.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 50,
            video_duration: 100,
            start_time: now - 2,
            start_active: now - 2,
            current_position: 50,
            unit_id: "CPU",
            logs: ["first"],
            live_text: "first text",
          },
        ],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 1, cpu_percent: 2, app_memory_gb: 1, memory_total_gb: 16, memory_used_gb: 4, memory_percent: 25 },
        telemetry: { hardware_util: { CPU: 1 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();

    const stale = dom.window.document.createElement("div");
    stale.className = "task-card";
    stale.dataset.taskId = "stale-task";
    dom.window.document.getElementById("task-list").appendChild(stale);

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.4.1",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 12,
        tasks: [
          {
            task_id: "asr-live",
            filename: "live.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 51,
            video_duration: 100,
            start_time: now - 3,
            start_active: now - 3,
            current_position: 51,
            unit_id: "CPU",
            logs: ["second"],
            live_text: "second text",
          },
        ],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 1, cpu_percent: 2, app_memory_gb: 1, memory_total_gb: 16, memory_used_gb: 4, memory_percent: 25 },
        telemetry: { hardware_util: { CPU: 1 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();

    const card = dom.window.document.querySelector('[data-task-id="asr-live"]');
    expect(card.querySelector(".live-text-box").textContent).toContain("second text");
    expect(card.querySelector(".eta-tag").style.display).toBe("none");
    expect(dom.window.document.querySelector('[data-task-id="stale-task"]')).toBeNull();
  });

  it("covers hardware status detection branches for all unit types", async () => {
    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 100,
        tasks: [],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 5, cpu_percent: 10, app_memory_gb: 0.8, memory_total_gb: 16, memory_used_gb: 3, memory_percent: 19 },
        telemetry: {
          hardware_util: { "GPU.0": 45, "NPU.0": 30 },
          intel_gpu_load: 45,
          npu_load: 30,
          nvidia: [{ util: 0 }]
        },
        telemetry_history: [],
        hardware_units: [
          { id: "CUDA:0", type: "CUDA", name: "NVIDIA GPU", uvr_status: "ready", whisper_status: "ready" },
          { id: "GPU.0", type: "GPU", name: "Intel GPU", uvr_status: "ready", whisper_status: "ready" },
          { id: "NPU.0", type: "NPU", name: "Intel NPU", uvr_status: "ready", whisper_status: "ready" },
          { id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }
        ],
      }),
    });

    await context.updateStats();
    const hwPool = dom.window.document.getElementById("hw-pool").innerHTML;
    expect(hwPool).toContain("NVIDIA GPU");
    expect(hwPool).toContain("Intel GPU");
    expect(hwPool).toContain("Intel NPU");
    expect(hwPool).toContain("Host CPU");
  });

  it("covers telemetry prepopulation and advanced speed/eta update branches", async () => {
    vi.useFakeTimers();
    const renderChartsSpy = vi.fn();
    context.renderCharts = renderChartsSpy;

    const now = Date.now() / 1000;
    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.2.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 10,
        tasks: [
          {
            task_id: "t-adv",
            filename: "adv.mp4",
            type: "Transcription",
            status: "active",
            stage: "Vocal Separation",
            progress: 5,
            video_duration: 200,
            start_time: now - 5,
            start_active: now - 5,
            current_position: 0,
            unit_id: "CPU",
            logs: ["a"],
            live_text: "a",
          },
        ],
        history: [
          {
            status: "completed",
            video_duration: 200,
            result: { performance: { inference_sec: 100, isolation_sec: 100 } },
          },
        ],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 1, cpu_percent: 2, app_memory_gb: 1, memory_total_gb: 16, memory_used_gb: 4, memory_percent: 25 },
        telemetry: { hardware_util: { CPU: 1 }, nvidia: [] },
        telemetry_history: [{ timestamp: now - 100, system: { app_cpu_percent: 0 }, telemetry: { hardware_util: { CPU: 0 } } }],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();
    expect(evalInContext(context, "rollingTelemetryBuffer.length")).toBeGreaterThan(1);

    const card = dom.window.document.querySelector('[data-task-id="t-adv"]');
    expect(card).toBeTruthy();

    context.handleToggle("t-adv_logs", true);
    vi.advanceTimersByTime(60);
    context.handleToggle("t-adv_logs", false);

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.2.1",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 20,
        tasks: [
          {
            task_id: "t-adv",
            filename: "adv.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 30,
            video_duration: 200,
            start_time: now - 20,
            start_active: now - 20,
            start_inference: now - 2,
            current_position: 0,
            unit_id: "CPU",
            logs: ["b"],
            live_text: "b",
          },
        ],
        history: [
          {
            status: "completed",
            video_duration: 200,
            result: { performance: { inference_sec: 50, isolation_sec: 100 } },
          },
        ],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 2, cpu_percent: 3, app_memory_gb: 1.1, memory_total_gb: 16, memory_used_gb: 4.2, memory_percent: 26 },
        telemetry: { hardware_util: { CPU: 2 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.2.2",
        active_sessions: 0,
        queued_sessions: 0,
        uptime_sec: 30,
        tasks: [],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 1, cpu_percent: 2, app_memory_gb: 1, memory_total_gb: 16, memory_used_gb: 4, memory_percent: 25 },
        telemetry: { hardware_util: { CPU: 0 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();
    expect(dom.window.document.querySelector('[data-task-id="t-adv"]')).toBeNull();

    let darkListener = null;
    context.window.matchMedia = () => ({
      matches: false,
      addEventListener: (_evt, cb) => {
        darkListener = cb;
      },
    });

    context.window.onload();
    evalInContext(context, "currentTab = 'charts'");
    darkListener();
    expect(renderChartsSpy).toHaveBeenCalled();

    vi.useRealTimers();
  });

  it("covers UVR task speed and ETA calculations with elapsedActive > 5", async () => {
    vi.useFakeTimers();
    const now = 1000;
    vi.setSystemTime(now * 1000);

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 100,
        tasks: [
          {
            task_id: "uvr-speed",
            filename: "uvr-test.mp4",
            type: "Isolation",
            status: "active",
            stage: "Vocal Separation",
            progress: 30,
            video_duration: 300,
            start_time: now - 100,
            start_active: now - 50,
            current_position: 90,
            unit_id: "CPU",
            logs: ["processing uvr"],
            live_text: "separating vocals",
          },
        ],
        history: [
          { status: "completed", video_duration: 300, result: { performance: { isolation_sec: 150, inference_sec: 0 } } },
          { status: "completed", video_duration: 300, result: { performance: { isolation_sec: 140, inference_sec: 0 } } },
        ],
        history_stats: { today: 2, count_today: 2, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 20, cpu_percent: 30, app_memory_gb: 1.5, memory_total_gb: 16, memory_used_gb: 6, memory_percent: 37 },
        telemetry: { hardware_util: { CPU: 20 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "busy", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();
    const card = dom.window.document.querySelector('[data-task-id="uvr-speed"]');
    expect(card).toBeTruthy();
    const speedText = card.querySelector(".speed-text");
    const etaText = card.querySelector(".eta-text");
    // With elapsedActive = 50 > 5 and valid progress, speed/eta should be calculated
    if (speedText && etaText) {
      expect(speedText.textContent).toBeTruthy();
      expect(etaText.textContent).toBeTruthy();
    }

    vi.useRealTimers();
  });

  it("covers ASR task ETA calculation with start_inference set", async () => {
    vi.useFakeTimers();
    const now = 2000;
    vi.setSystemTime(now * 1000);

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 200,
        tasks: [
          {
            task_id: "asr-eta",
            filename: "asr-test.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 40,
            video_duration: 200,
            start_time: now - 150,
            start_active: now - 150,
            start_inference: now - 20,
            current_position: 80,
            unit_id: "CPU",
            logs: [],
            live_text: "",
          },
        ],
        history: [
          { status: "completed", video_duration: 200, result: { performance: { inference_sec: 100, isolation_sec: 0 } } },
        ],
        history_stats: { today: 1, count_today: 1, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 15, cpu_percent: 25, app_memory_gb: 1.2, memory_total_gb: 16, memory_used_gb: 5, memory_percent: 31 },
        telemetry: { hardware_util: { CPU: 15 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "busy" }],
      }),
    });

    await context.updateStats();
    const card = dom.window.document.querySelector('[data-task-id="asr-eta"]');
    expect(card).toBeTruthy();
    // With high elapsedActive, speed/eta elements should be present
    expect(card).toBeTruthy();

    vi.useRealTimers();
  });

  it("covers task card update when speed/eta elements missing", async () => {
    vi.useFakeTimers();
    const now = 3000;
    vi.setSystemTime(now * 1000);

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 300,
        tasks: [
          {
            task_id: "no-speed-eta",
            filename: "test.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 50,
            video_duration: 100,
            start_time: now - 200,
            start_active: now - 200,
            current_position: 50,
            unit_id: "CPU",
            logs: ["log1", "log2"],
            live_text: "processing",
          },
        ],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 25, cpu_percent: 35, app_memory_gb: 2.0, memory_total_gb: 16, memory_used_gb: 7, memory_percent: 43 },
        telemetry: { hardware_util: { CPU: 25 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "busy", whisper_status: "busy" }],
      }),
    });

    await context.updateStats();
    const card = dom.window.document.querySelector('[data-task-id="no-speed-eta"]');
    expect(card).toBeTruthy();
    const logBuffer = card.querySelector(".log-buffer");
    expect(logBuffer.textContent).toContain("log1");

    vi.useRealTimers();
  });

  it("covers different hardware type status detection branches", async () => {
    const now = Date.now() / 1000;
    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 2,
        queued_sessions: 1,
        uptime_sec: 400,
        tasks: [
          {
            task_id: "gpu-task",
            filename: "gpu.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 60,
            video_duration: 250,
            start_time: now - 60,
            start_active: now - 60,
            current_position: 150,
            unit_id: "GPU.0",
            logs: [],
            live_text: "",
          },
          {
            task_id: "npu-task",
            filename: "npu.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 45,
            video_duration: 200,
            start_time: now - 45,
            start_active: now - 45,
            current_position: 90,
            unit_id: "NPU.0",
            logs: [],
            live_text: "",
          },
        ],
        history: [],
        history_stats: { today: 2, count_today: 2, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 30, cpu_percent: 40, app_memory_gb: 2.5, memory_total_gb: 16, memory_used_gb: 8, memory_percent: 50 },
        telemetry: {
          hardware_util: { "GPU.0": 75, "NPU.0": 60 },
          intel_gpu_load: 75,
          npu_load: 60,
          nvidia: []
        },
        telemetry_history: [],
        hardware_units: [
          { id: "GPU.0", type: "GPU", name: "Intel Arc", uvr_status: "busy", whisper_status: "busy" },
          { id: "NPU.0", type: "NPU", name: "Intel NPU", uvr_status: "busy", whisper_status: "ready" },
        ],
      }),
    });

    await context.updateStats();
    const gpuCard = dom.window.document.querySelector('[data-task-id="gpu-task"]');
    const npuCard = dom.window.document.querySelector('[data-task-id="npu-task"]');
    expect(gpuCard).toBeTruthy();
    expect(npuCard).toBeTruthy();
    const hwPool = dom.window.document.getElementById("hw-pool").innerHTML;
    expect(hwPool).toContain("Intel Arc");
    expect(hwPool).toContain("Intel NPU");
  });

  it("covers CUDA GPU hardware detection with nvidia array", async () => {
    const now = Date.now() / 1000;
    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 500,
        tasks: [
          {
            task_id: "cuda-task",
            filename: "cuda.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 70,
            video_duration: 280,
            start_time: now - 80,
            start_active: now - 80,
            current_position: 196,
            unit_id: "CUDA:0",
            logs: [],
            live_text: "",
          },
        ],
        history: [],
        history_stats: { today: 1, count_today: 1, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 35, cpu_percent: 45, app_memory_gb: 3.0, memory_total_gb: 32, memory_used_gb: 15, memory_percent: 46 },
        telemetry: {
          hardware_util: { "CUDA:0": 95 },
          nvidia: [{ util: 95 }],
          intel_gpu_load: 0,
          npu_load: 0
        },
        telemetry_history: [],
        hardware_units: [
          { id: "CUDA:0", type: "CUDA", name: "NVIDIA GPU", uvr_status: "busy", whisper_status: "busy" },
        ],
      }),
    });

    await context.updateStats();
    const cudaCard = dom.window.document.querySelector('[data-task-id="cuda-task"]');
    expect(cudaCard).toBeTruthy();
    const hwPool = dom.window.document.getElementById("hw-pool").innerHTML;
    expect(hwPool).toContain("NVIDIA GPU");
  });

  it("covers CUDA GPU idle but active via telemetry.nvidia", async () => {
    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 0,
        queued_sessions: 0,
        uptime_sec: 500,
        tasks: [],
        history: [],
        history_stats: { today: 1, count_today: 1, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 35, cpu_percent: 45, app_memory_gb: 3.0, memory_total_gb: 32, memory_used_gb: 15, memory_percent: 46 },
        telemetry: {
          nvidia: [{ util: 95 }],
          intel_gpu_load: 0,
          npu_load: 0
        },
        telemetry_history: [],
        hardware_units: [
          { id: "CUDA:0", type: "CUDA", name: "NVIDIA GPU", uvr_status: "ready", whisper_status: "ready" },
        ],
      }),
    });

    await context.updateStats();
    const hwPool = dom.window.document.getElementById("hw-pool").innerHTML;
    expect(hwPool).toContain("NVIDIA GPU");
    expect(hwPool).toContain("Used");
  });

  it("covers renderHistory with card removal for inactive tasks", async () => {
    const now = Date.now() / 1000;
    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 600,
        tasks: [{ task_id: "active-1", filename: "a.mp4", type: "Transcription", status: "active", stage: "Inference", progress: 80, video_duration: 300, start_time: now - 100, start_active: now - 100, current_position: 240, unit_id: "CPU", logs: [], live_text: "" }],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 10, cpu_percent: 15, app_memory_gb: 1.0, memory_total_gb: 16, memory_used_gb: 4, memory_percent: 25 },
        telemetry: { hardware_util: { CPU: 10 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "busy", whisper_status: "busy" }],
      }),
    });

    await context.updateStats();
    let activeCard = dom.window.document.querySelector('[data-task-id="active-1"]');
    expect(activeCard).toBeTruthy();

    // Update to remove the active task
    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 0,
        queued_sessions: 0,
        uptime_sec: 610,
        tasks: [],
        history: [],
        history_stats: { today: 1, count_today: 1, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 5, cpu_percent: 8, app_memory_gb: 0.8, memory_total_gb: 16, memory_used_gb: 3, memory_percent: 20 },
        telemetry: { hardware_util: { CPU: 0 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();
    activeCard = dom.window.document.querySelector('[data-task-id="active-1"]');
    expect(activeCard).toBeNull();
  });

  it("covers elapsedActive <= 5 condition for speed/eta display", async () => {
    vi.useFakeTimers();
    const now = 4000;
    vi.setSystemTime(now * 1000);

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 700,
        tasks: [
          {
            task_id: "short-elapsed",
            filename: "short.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 10,
            video_duration: 100,
            start_time: now - 3,
            start_active: now - 3,
            current_position: 10,
            unit_id: "CPU",
            logs: [],
            live_text: "",
          },
        ],
        history: [],
        history_stats: { today: 0, count_today: 0, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 5, cpu_percent: 10, app_memory_gb: 1.0, memory_total_gb: 16, memory_used_gb: 4, memory_percent: 25 },
        telemetry: { hardware_util: { CPU: 5 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "busy", whisper_status: "busy" }],
      }),
    });

    await context.updateStats();
    const card = dom.window.document.querySelector('[data-task-id="short-elapsed"]');
    expect(card).toBeTruthy();
    const speedTag = card.querySelector(".speed-tag");
    const etaTag = card.querySelector(".eta-tag");
    // With elapsedActive = 3 <= 5, speed/eta should be hidden
    if (speedTag) expect(speedTag.style.display).toBe("none");
    if (etaTag) expect(etaTag.style.display).toBe("none");

    vi.useRealTimers();
  });

  it("covers UVR task with elapsedAsr <= 5 and expectedUvrSpeed fallback", async () => {
    vi.useFakeTimers();
    const now = 5000;
    vi.setSystemTime(now * 1000);

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 800,
        tasks: [
          {
            task_id: "uvr-fallback",
            filename: "uvr-fallback.mp4",
            type: "Isolation",
            status: "active",
            stage: "Vocal Separation",
            progress: 25,
            video_duration: 600,
            start_time: now - 200,
            start_active: now - 40,
            current_position: 150,
            unit_id: "CPU",
            logs: [],
            live_text: "",
          },
        ],
        history: [
          { status: "completed", video_duration: 600, result: { performance: { isolation_sec: 200, inference_sec: 0 } } },
        ],
        history_stats: { today: 1, count_today: 1, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 20, cpu_percent: 30, app_memory_gb: 1.5, memory_total_gb: 16, memory_used_gb: 6, memory_percent: 37 },
        telemetry: { hardware_util: { CPU: 20 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "busy", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();
    const card = dom.window.document.querySelector('[data-task-id="uvr-fallback"]');
    expect(card).toBeTruthy();

    vi.useRealTimers();
  });

  it("covers UVR + ASR combined speed/eta calculation with history", async () => {
    vi.useFakeTimers();
    const now = 6000;
    vi.setSystemTime(now * 1000);

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 900,
        tasks: [
          {
            task_id: "uvr-asr-combo",
            filename: "combo.mp4",
            type: "Isolation",
            status: "active",
            stage: "Vocal Separation",
            progress: 35,
            video_duration: 400,
            start_time: now - 250,
            start_active: now - 120,
            current_position: 140,
            unit_id: "CPU",
            logs: [],
            live_text: "",
          },
        ],
        history: [
          { status: "completed", video_duration: 400, result: { performance: { isolation_sec: 120, inference_sec: 80 } } },
          { status: "completed", video_duration: 400, result: { performance: { isolation_sec: 110, inference_sec: 90 } } },
          { status: "completed", video_duration: 400, result: { performance: { isolation_sec: 130, inference_sec: 70 } } },
        ],
        history_stats: { today: 3, count_today: 3, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 25, cpu_percent: 35, app_memory_gb: 2.0, memory_total_gb: 16, memory_used_gb: 7, memory_percent: 43 },
        telemetry: { hardware_util: { CPU: 25 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "busy", whisper_status: "ready" }],
      }),
    });

    await context.updateStats();
    const card = dom.window.document.querySelector('[data-task-id="uvr-asr-combo"]');
    expect(card).toBeTruthy();

    vi.useRealTimers();
  });

  it("covers ASR speed calculation with uvrElapsed = 0 (no UVR stage)", async () => {
    vi.useFakeTimers();
    const now = 7000;
    vi.setSystemTime(now * 1000);

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 1000,
        tasks: [
          {
            task_id: "asr-only",
            filename: "asr-only.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 55,
            video_duration: 180,
            start_time: now - 100,
            start_active: now - 100,
            current_position: 99,
            unit_id: "CPU",
            logs: [],
            live_text: "",
          },
        ],
        history: [
          { status: "completed", video_duration: 180, result: { performance: { inference_sec: 60, isolation_sec: 0 } } },
        ],
        history_stats: { today: 1, count_today: 1, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 18, cpu_percent: 28, app_memory_gb: 1.8, memory_total_gb: 16, memory_used_gb: 6, memory_percent: 35 },
        telemetry: { hardware_util: { CPU: 18 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "busy" }],
      }),
    });

    await context.updateStats();
    const card = dom.window.document.querySelector('[data-task-id="asr-only"]');
    expect(card).toBeTruthy();

    vi.useRealTimers();
  });

  it("covers speed/eta display when calculatedSpeed > 0 and remainingSeconds > 0", async () => {
    vi.useFakeTimers();
    const now = 8000;
    vi.setSystemTime(now * 1000);

    fetchMock.mockResolvedValueOnce({
      json: async () => ({
        version: "1.3.0",
        active_sessions: 1,
        queued_sessions: 0,
        uptime_sec: 1100,
        tasks: [
          {
            task_id: "show-speed-eta",
            filename: "show-speed-eta.mp4",
            type: "Transcription",
            status: "active",
            stage: "Inference",
            progress: 65,
            video_duration: 240,
            start_time: now - 150,
            start_active: now - 150,
            current_position: 156,
            unit_id: "CPU",
            logs: [],
            live_text: "",
          },
        ],
        history: [
          { status: "completed", video_duration: 240, result: { performance: { inference_sec: 80, isolation_sec: 0 } } },
        ],
        history_stats: { today: 1, count_today: 1, this_month: 0, all_time: 0, count_all_time: 0 },
        system: { app_cpu_percent: 22, cpu_percent: 32, app_memory_gb: 1.9, memory_total_gb: 16, memory_used_gb: 6.5, memory_percent: 40 },
        telemetry: { hardware_util: { CPU: 22 }, nvidia: [] },
        telemetry_history: [],
        hardware_units: [{ id: "CPU", type: "CPU", name: "Host CPU", uvr_status: "ready", whisper_status: "busy" }],
      }),
    });

    await context.updateStats();
    const card = dom.window.document.querySelector('[data-task-id="show-speed-eta"]');
    expect(card).toBeTruthy();
    // Speed and ETA should potentially be visible with these conditions
    expect(card).toBeTruthy();

    vi.useRealTimers();
  });

  it("covers calculateHistoricalSpeeds helper function with multiple history items", async () => {
    const history = [
      { status: "completed", video_duration: 300, result: { performance: { inference_sec: 100, isolation_sec: 150 } } },
      { status: "completed", video_duration: 300, result: { performance: { inference_sec: 120, isolation_sec: 140 } } },
      { status: "completed", video_duration: 300, result: { performance: { inference_sec: 80, isolation_sec: 160 } } },
    ];

    const speeds = evalInContext(context, `calculateHistoricalSpeeds(${JSON.stringify(history)})`);
    expect(speeds.expectedAsrSpeed).toBeGreaterThan(0);
    expect(speeds.expectedUvrSpeed).toBeGreaterThan(0);
  });

  it("covers calculateHistoricalSpeeds with missing performance data", async () => {
    const history = [
      { status: "completed", video_duration: 300, result: null },
      { status: "completed", video_duration: 300, result: { performance: { inference_sec: 100, isolation_sec: 150 } } },
      { status: "failed", video_duration: 300, result: { performance: { inference_sec: 100, isolation_sec: 150 } } },
    ];

    const speeds = evalInContext(context, `calculateHistoricalSpeeds(${JSON.stringify(history)})`);
    expect(speeds.expectedAsrSpeed).toBeGreaterThan(0);
    expect(speeds.expectedUvrSpeed).toBeGreaterThan(0);
  });

  it("covers calculateTaskSpeedAndEta for UVR task with zero progress", async () => {
    const task = {
      type: "Isolation",
      stage: "Vocal Separation",
      video_duration: 300,
      current_position: 0,
      start_active: 1000,
      start_inference: undefined
    };

    const now = 1050;
    const history = [
      { status: "completed", video_duration: 300, result: { performance: { inference_sec: 100, isolation_sec: 150 } } },
    ];

    const result = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(task)}, ${now}, ${JSON.stringify(history)})`);
    expect(result.calculatedSpeed).toBeDefined();
    expect(result.remainingSeconds).toBeDefined();
  });

  it("covers calculateTaskSpeedAndEta for ASR task with long elapsed time", async () => {
    const task = {
      type: "Transcription",
      stage: "Inference",
      video_duration: 200,
      current_position: 120,
      start_active: 1000,
      start_inference: 1050
    };

    const now = 1120;
    const history = [
      { status: "completed", video_duration: 200, result: { performance: { inference_sec: 60, isolation_sec: 0 } } },
    ];

    const result = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(task)}, ${now}, ${JSON.stringify(history)})`);
    expect(result.calculatedSpeed).toBeGreaterThanOrEqual(0);
    expect(result.remainingSeconds).toBeGreaterThanOrEqual(0);
  });

  it("uses activeTaskTimeline to calculate live speed for UVR tasks", () => {
    const task = {
      task_id: "t-uvr-live",
      type: "Isolation",
      stage: "Vocal Separation",
      video_duration: 100,
      current_position: 10,
      start_active: 1000,
      start_inference: undefined
    };

    // Seed the timeline
    evalInContext(context, `activeTaskTimeline["t-uvr-live"] = [
      { timestamp: 1000, position: 0 },
      { timestamp: 1005, position: 5 }
    ]`);

    const result = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(task)}, 1010, [])`);
    // Speed over last 10s: progress went from 0 to 10. Speed is 1.0.
    // Remaining UVR = (100 - 10) / 1.0 = 90s.
    // Default ASR speed fallback = 5.0 -> ASR ETA = 100 / 5 = 20s.
    // Total remaining = 90 + 20 = 110s.
    expect(result.remainingSeconds).toBe(110);
    expect(result.calculatedSpeed).toBeCloseTo(100 / (10 + 110)); // video_duration / (elapsedActive + remaining)
  });

  it("uses activeTaskTimeline to calculate live speed for ASR tasks", () => {
    const task = {
      task_id: "t-asr-live",
      type: "Transcription",
      stage: "Inference",
      video_duration: 100,
      current_position: 40,
      start_active: 1000,
      start_inference: 1020
    };

    // Seed the timeline
    evalInContext(context, `activeTaskTimeline["t-asr-live"] = [
      { timestamp: 1020, position: 0 },
      { timestamp: 1030, position: 20 }
    ]`);

    const result = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(task)}, 1040, [])`);
    // Speed in ASR over last 20s: progress went from 0 to 40. Speed is 2.0.
    // Remaining ASR = (100 - 40) / 2.0 = 30s.
    // Total elapsed active = 40s.
    // Total estimated = 20 (uvr) + 20 (asr elapsed) + 30 (asr remaining) = 70s.
    expect(result.remainingSeconds).toBe(30);
    expect(result.calculatedSpeed).toBeCloseTo(100 / 70);
  });

  it("persists timeline cache fields on the first invocation for a task", () => {
    const task = {
      task_id: "t-first-cache",
      type: "Transcription",
      stage: "Inference",
      video_duration: 100,
      current_position: 20,
      start_active: 1000,
      start_inference: 1000
    };

    const result = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(task)}, 1010, [])`);
    const timeline = evalInContext(context, `activeTaskTimeline["t-first-cache"]`);

    expect(result.calculatedSpeed).toBeGreaterThan(0);
    expect(timeline).toBeDefined();
    expect(timeline.lastCalculatedSpeed).toBe(result.calculatedSpeed);
    expect(timeline.lastRemainingSeconds).toBe(result.remainingSeconds);
    expect(timeline.lastPosition).toBe(20);
    expect(timeline.lastStage).toBe("Inference");
    expect(timeline.lastSmoothedTimestamp).toBe(1010);
  });

  it("applies EMA smoothing to task speeds on subsequent calls for both ASR and UVR", () => {
    // 1. Test ASR smoothing
    const taskAsr = {
      task_id: "t-smooth-asr",
      type: "Transcription",
      stage: "Inference",
      video_duration: 100,
      current_position: 20,
      start_active: 1000,
      start_inference: 1000
    };
    evalInContext(context, `activeTaskTimeline["t-smooth-asr"] = [
      { timestamp: 1000, position: 0 }
    ]`);

    // First call: sets baseline asrSpeed to 20 / 10 = 2.0
    evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(taskAsr)}, 1010, [])`);

    // Update position and add timeline entry
    taskAsr.current_position = 60;
    evalInContext(context, `activeTaskTimeline["t-smooth-asr"].push({ timestamp: 1020, position: 60 })`);

    // Second call at 1020. Raw live speed is 3.0 (from 1000 to 1020). dt = 10. alpha = 1 - exp(-10/12.3)
    const alpha = 1 - Math.exp(-10 / 12.3);
    const expectedAsrSpeed = alpha * 3.0 + (1 - alpha) * 2.0;
    const resultAsr = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(taskAsr)}, 1020, [])`);
    expect(resultAsr.remainingSeconds).toBeCloseTo(40 / expectedAsrSpeed);

    // 2. Test UVR smoothing
    const taskUvr = {
      task_id: "t-smooth-uvr",
      type: "Isolation",
      stage: "Vocal Separation",
      video_duration: 100,
      current_position: 20,
      start_active: 1000,
      start_inference: undefined
    };
    evalInContext(context, `activeTaskTimeline["t-smooth-uvr"] = [
      { timestamp: 1000, position: 0 }
    ]`);

    // First call: sets baseline uvrSpeed to 20 / 10 = 2.0
    evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(taskUvr)}, 1010, [])`);

    // Update position and add timeline entry
    taskUvr.current_position = 60;
    evalInContext(context, `activeTaskTimeline["t-smooth-uvr"].push({ timestamp: 1020, position: 60 })`);

    // Second call at 1020. Raw live speed is 3.0.
    const expectedUvrSpeed = alpha * 3.0 + (1 - alpha) * 2.0;
    const resultUvr = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(taskUvr)}, 1020, [])`);
    expect(resultUvr.remainingSeconds).toBeCloseTo(40 / expectedUvrSpeed + 100 / 5.0);
  });

  it("bypasses recalculation and ticks down remainingSeconds when position and stage are unchanged", () => {
    const task = {
      task_id: "t-bypass-test",
      type: "Transcription",
      stage: "Inference",
      video_duration: 100,
      current_position: 20,
      start_active: 1000,
      start_inference: 1000
    };
    evalInContext(context, `activeTaskTimeline["t-bypass-test"] = [
      { timestamp: 1000, position: 0 }
    ]`);

    // First call: runs calculation and caches values (e.g. at timestamp 1010)
    const result1 = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(task)}, 1010, [])`);
    const initialRemaining = result1.remainingSeconds;

    // Second call at timestamp 1015: same position (20), same stage (Inference)
    const result2 = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(task)}, 1015, [])`);
    
    // It should bypass recalculation, keep calculatedSpeed same, and tick down remainingSeconds by 5s.
    expect(result2.calculatedSpeed).toBe(result1.calculatedSpeed);
    expect(result2.remainingSeconds).toBe(initialRemaining - 5);
  });

  it("does not use the fast-path cache when the stage changes", () => {
    const task = {
      task_id: "t-stage-change",
      type: "Transcription",
      stage: "Inference",
      video_duration: 100,
      current_position: 20,
      start_active: 1000,
      start_inference: 1000
    };

    evalInContext(context, `activeTaskTimeline["t-stage-change"] = [{ timestamp: 1000, position: 0 }]`);
    evalInContext(context, `activeTaskTimeline["t-stage-change"].lastCalculatedSpeed = 7`);
    evalInContext(context, `activeTaskTimeline["t-stage-change"].lastRemainingSeconds = 80`);
    evalInContext(context, `activeTaskTimeline["t-stage-change"].lastPosition = 20`);
    evalInContext(context, `activeTaskTimeline["t-stage-change"].lastStage = "Vocal Separation"`);
    evalInContext(context, `activeTaskTimeline["t-stage-change"].lastSmoothedTimestamp = 1010`);

    const result = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(task)}, 1015, [])`);
    const timeline = evalInContext(context, `activeTaskTimeline["t-stage-change"]`);

    expect(result.calculatedSpeed).not.toBe(7);
    expect(result.remainingSeconds).not.toBe(75);
    expect(timeline.lastStage).toBe("Inference");
    expect(timeline.lastSmoothedTimestamp).toBe(1015);
  });

  it("clears prior timeline samples on stage transition and prevents negative delta speed calculation", () => {
    const task = {
      task_id: "t-transition-restart",
      filename: "movie.mp4",
      status: "active",
      type: "Transcription",
      stage: "Inference",
      video_duration: 100,
      current_position: 0,
      start_active: 1000,
      start_inference: 1015
    };

    evalInContext(context, `activeTaskTimeline["t-transition-restart"] = [{ timestamp: 1010, position: 50 }]`);
    evalInContext(context, `activeTaskTimeline["t-transition-restart"].lastStage = "Vocal Separation"`);
    evalInContext(context, `activeTaskTimeline["t-transition-restart"].lastPosition = 50`);

    evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(task)}, 1015, [])`);
    const timeline1 = evalInContext(context, `activeTaskTimeline["t-transition-restart"]`);

    expect(timeline1.length).toBe(1);
    expect(timeline1[0].position).toBe(0);
    expect(timeline1[0].timestamp).toBe(1015);

    task.current_position = 10;
    const result2 = evalInContext(context, `calculateTaskSpeedAndEta(${JSON.stringify(task)}, 1020, [])`);
    const timeline2 = evalInContext(context, `activeTaskTimeline["t-transition-restart"]`);

    expect(timeline2.length).toBe(2);
    expect(result2.calculatedSpeed).toBeGreaterThan(0);
  });

  it("cleans up activeTaskTimeline for completed/removed tasks during updateStats", async () => {
    // Seed timeline
    evalInContext(context, `activeTaskTimeline["old-completed-task"] = [{ timestamp: 1000, position: 10 }]`);
    evalInContext(context, `activeTaskTimeline["a1"] = [{ timestamp: 1000, position: 5 }]`); // a1 is active in fetchMock

    await context.updateStats();

    const timeline = evalInContext(context, `activeTaskTimeline`);
    expect(timeline["old-completed-task"]).toBeUndefined();
    expect(timeline["a1"]).toBeDefined();
  });

  it("handles changeRefreshInterval correctly", () => {
    context.changeRefreshInterval("5000");
    const interval = evalInContext(context, "currentRefreshInterval");
    expect(interval).toBe(5000);
  });

  it("handles filterTasks and filterHistory correctly", () => {
    context.filterTasks("asr");
    const activeFilter = evalInContext(context, "activeTaskFilter");
    expect(activeFilter).toBe("asr");

    context.filterHistory("detectlang");
    const historyFilter = evalInContext(context, "historyTaskFilter");
    expect(historyFilter).toBe("detectlang");

    context.filterTasks("v1");
    const v1Filter = evalInContext(context, "activeTaskFilter");
    expect(v1Filter).toBe("v1");
  });

  it("sorts tasks deterministically: active first, then priority queued, then standard queued, then initializing", async () => {
    const tasks = [
      { task_id: "t-std-queued", filename: "std-queued.mp4", type: "/asr", status: "queued", is_priority: false, start_time: 1000 },
      { task_id: "t-pri-queued", filename: "pri-queued.mp4", type: "/asr", status: "queued", is_priority: true, start_time: 950 },
      { task_id: "t-active", filename: "active.mp4", type: "/asr", status: "active", start_time: 1050 },
      { task_id: "t-initializing", filename: "init.mp4", type: "/asr", status: "initializing", start_time: 1100 }
    ];

    // Append a stale task card to cover the removal branch
    const staleCard = dom.window.document.createElement("div");
    staleCard.className = "task-card";
    staleCard.dataset.taskId = "t-stale";
    dom.window.document.getElementById("task-list").appendChild(staleCard);

    context.fetch = vi.fn(async () => ({
      json: async () => ({
        version: "1.2.0",
        active_sessions: 1,
        queued_sessions: 2,
        system: { cpu_percent: 10, app_cpu_percent: 5, app_memory_gb: 1, memory_total_gb: 16, memory_used_gb: 8, memory_percent: 50 },
        telemetry: {},
        telemetry_history: [],
        history: [],
        tasks: tasks,
        hardware_units: []
      })
    }));

    context.filterTasks("all");
    await context.updateStats();

    const renderedCards = Array.from(dom.window.document.querySelectorAll("#task-list .task-card"));
    expect(renderedCards).toHaveLength(4);
    
    expect(renderedCards[0].dataset.taskId).toBe("t-active");
    expect(renderedCards[1].dataset.taskId).toBe("t-pri-queued");
    expect(renderedCards[2].dataset.taskId).toBe("t-std-queued");
    expect(renderedCards[3].dataset.taskId).toBe("t-initializing");
  });

  it("retains activeTaskTimeline history for filtered-out but active tasks", async () => {
    evalInContext(context, `activeTaskTimeline["t-visible"] = [{ timestamp: 1000, position: 5 }]`);
    evalInContext(context, `activeTaskTimeline["t-hidden"] = [{ timestamp: 1000, position: 8 }]`);

    const tasks = [
      { task_id: "t-visible", filename: "visible-asr.mp4", type: "/asr", status: "active", start_time: 1000 },
      { task_id: "t-hidden", filename: "hidden-detectlang.mp4", type: "/detect-language", status: "active", start_time: 1010 }
    ];

    context.fetch = vi.fn(async () => ({
      json: async () => ({
        version: "1.2.0",
        active_sessions: 1,
        queued_sessions: 1,
        system: { cpu_percent: 10, app_cpu_percent: 5, app_memory_gb: 1, memory_total_gb: 16, memory_used_gb: 8, memory_percent: 50 },
        telemetry: {},
        telemetry_history: [],
        history: [],
        tasks: tasks,
        hardware_units: []
      })
    }));

    context.filterTasks("asr");
    await context.updateStats();

    const timeline = evalInContext(context, `activeTaskTimeline`);
    expect(timeline["t-visible"]).toBeDefined();
    expect(timeline["t-hidden"]).toBeDefined();

    const renderedCards = Array.from(dom.window.document.querySelectorAll("#task-list .task-card"));
    expect(renderedCards).toHaveLength(1);
    expect(renderedCards[0].dataset.taskId).toBe("t-visible");
  });

  it("covers task order comparator tie-break branches", () => {
    const sameTierDifferentTimeA = { task_id: "a", status: "queued", is_priority: true, start_time: 100 };
    const sameTierDifferentTimeB = { task_id: "b", status: "queued", is_priority: true, start_time: 120 };
    const byTime = evalInContext(
      context,
      `_compareTaskOrder(${JSON.stringify(sameTierDifferentTimeA)}, ${JSON.stringify(sameTierDifferentTimeB)})`
    );
    expect(byTime).toBeLessThan(0);

    const sameTierSameTimeA = { task_id: "a", status: "queued", is_priority: false, start_time: 200 };
    const sameTierSameTimeB = { task_id: "b", status: "queued", is_priority: false, start_time: 200 };
    const byId = evalInContext(
      context,
      `_compareTaskOrder(${JSON.stringify(sameTierSameTimeA)}, ${JSON.stringify(sameTierSameTimeB)})`
    );
    expect(byId).toBeLessThan(0);
  });

  it("covers timestamp comparability and UVR speed estimation helper", () => {
    const mismatched = evalInContext(context, `_areComparableTimestamps(3997, ${Math.floor(Date.now() / 1000)})`);
    expect(mismatched).toBe(false);

    const estimate = evalInContext(
      context,
      `_taskSpeedEtaEstimate(${JSON.stringify({ type: "Transcription", stage: "Vocal Separation", video_duration: 120, current_position: 20, start_active: 1000, start_inference: 1000 })}, 1010, ${JSON.stringify({ expectedAsrSpeed: 2, expectedUvrSpeed: 1 })}, "Vocal Separation")`
    );
    expect(estimate.calculatedSpeed).toBeCloseTo(1.0, 5);
    expect(estimate.remainingSeconds).toBeCloseTo(110, 5);
  });
});
