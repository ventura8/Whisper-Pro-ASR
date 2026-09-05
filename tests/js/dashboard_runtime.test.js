const path = require("path");
const { JSDOM } = require("jsdom");
const { loadScriptInContext, evalInContext } = require("./helpers");

function buildStatusData(overrides = {}) {
  return {
    version: "1.3.0",
    system: {
      app_cpu_percent: 5,
      cpu_percent: 20,
      app_memory_gb: 1.2,
      memory_total_gb: 16,
      memory_used_gb: 6,
      memory_percent: 37.5,
    },
    active_sessions: 1,
    queued_sessions: 0,
    telemetry_history: [],
    telemetry: {},
    history: [],
    history_stats: {
      today: 120,
      count_today: 2,
      this_month: 3600,
      all_time: 7200,
      count_all_time: 5,
    },
    uptime_sec: 500,
    hardware_units: [],
    tasks: [],
    ...overrides,
  };
}

describe("runtime.js", () => {
  let dom;
  let context;
  let fetchMock;

  beforeEach(() => {
    dom = new JSDOM(`<!doctype html><html><body>
      <div id="app-version"></div>
      <div id="app-cpu-val"></div>
      <div id="app-cpu-bar" style="width:0%"></div>
      <div id="sys-cpu-val"></div>
      <div id="sys-cpu-bar" style="width:0%"></div>
      <div id="app-mem-val"></div>
      <div id="app-mem-bar" style="width:0%"></div>
      <div id="sys-mem-val"></div>
      <div id="sys-mem-bar" style="width:0%"></div>
      <div id="active-val"></div>
      <div id="queued-val"></div>
      <div id="analytics-grid"></div>
      <div id="hw-pool"></div>
      <div id="last-update"></div>
      <div id="task-list"></div>
    </body></html>`);

    fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => buildStatusData(),
    });

    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/core/utils.js"), {
      document: dom.window.document,
      Date,
      Math,
      setTimeout,
      fetch: fetchMock,
      refreshEnabled: true,
      currentTab: "active",
      currentTelemetry: [],
      rollingTelemetryBuffer: [],
      lastStatusData: null,
      fullTaskHistory: [],
      renderCharts: vi.fn(),
      renderHistory: vi.fn(),
      calculateHistoricalSpeeds: vi.fn(() => ({})),
      _renderActiveTaskList: vi.fn(),
      bindToggleHandlers: vi.fn(),
      _cleanupTimelineForTasks: vi.fn(),
    });
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/runtime.js"), context);
  });

  describe("_hasValidStatusData", () => {
    it("is true only when data.system is present", () => {
      expect(context._hasValidStatusData({ system: {} })).toBe(true);
      expect(context._hasValidStatusData({})).toBe(false);
      expect(context._hasValidStatusData(null)).toBe(false);
    });
  });

  describe("_renderTopStats / _renderQueueCounters / _renderAnalyticsGrid", () => {
    it("renders top stats from status data", () => {
      context._renderTopStats(buildStatusData());
      expect(dom.window.document.getElementById("app-version").innerText).toBe("Version 1.3.0");
      expect(dom.window.document.getElementById("app-cpu-val").innerText).toBe("5%");
      expect(dom.window.document.getElementById("sys-cpu-val").innerText).toBe("20%");
      expect(dom.window.document.getElementById("app-mem-val").innerText).toBe("1.20 GB");
      expect(dom.window.document.getElementById("sys-mem-val").innerText).toBe("6.00 / 16.00 GB");
    });

    it("renders queue counters, defaulting to 0", () => {
      context._renderQueueCounters({ active_sessions: 3, queued_sessions: 2 });
      expect(dom.window.document.getElementById("active-val").innerText).toBe("3");
      expect(dom.window.document.getElementById("queued-val").innerText).toBe("2");

      context._renderQueueCounters({});
      expect(dom.window.document.getElementById("active-val").innerText).toBe("0");
      expect(dom.window.document.getElementById("queued-val").innerText).toBe("0");
    });

    it("renders the analytics grid from history_stats", () => {
      context._renderAnalyticsGrid(buildStatusData());
      const html = dom.window.document.getElementById("analytics-grid").innerHTML;
      expect(html).toContain("Tasks Today");
      expect(html).toContain("Total Tasks");
    });

    it("leaves analytics grid untouched when history_stats missing", () => {
      const grid = dom.window.document.getElementById("analytics-grid");
      grid.innerHTML = "sentinel";
      context._renderAnalyticsGrid({});
      expect(grid.innerHTML).toBe("sentinel");
    });
  });

  describe("_updateTelemetryState / _prepopulateTelemetryHistory", () => {
    it("filters the rolling buffer to the last 600 seconds", () => {
      const now = 10_000;
      evalInContext(context, `rollingTelemetryBuffer = [{ timestamp: ${now - 1000} }]`);
      context._updateTelemetryState(buildStatusData({ telemetry_history: [] }), now);
      const buffer = evalInContext(context, "rollingTelemetryBuffer");
      expect(buffer.every((entry) => entry.timestamp >= now - 600)).toBe(true);
      expect(buffer[buffer.length - 1].system).toBeTruthy();
    });

    it("backfills server telemetry history only on the first buffer entry", () => {
      const now = 10_000;
      evalInContext(context, "rollingTelemetryBuffer = []");
      const serverHistory = [{ timestamp: now - 100 }, { timestamp: now - 50 }];
      context._updateTelemetryState(buildStatusData({ telemetry_history: serverHistory }), now);
      const buffer = evalInContext(context, "rollingTelemetryBuffer");
      expect(buffer.length).toBe(3);
    });

    it("does not backfill when the buffer already has more than one entry", () => {
      const now = 10_000;
      evalInContext(context, `rollingTelemetryBuffer = [{ timestamp: ${now - 10} }, { timestamp: ${now - 5} }]`);
      context._updateTelemetryState(buildStatusData({ telemetry_history: [{ timestamp: now - 100 }] }), now);
      const buffer = evalInContext(context, "rollingTelemetryBuffer");
      expect(buffer.length).toBe(3);
    });

    it("skips backfill when server telemetry_history is empty", () => {
      const now = 10_000;
      evalInContext(context, "rollingTelemetryBuffer = []");
      context._updateTelemetryState(buildStatusData({ telemetry_history: [] }), now);
      const buffer = evalInContext(context, "rollingTelemetryBuffer");
      expect(buffer.length).toBe(1);
    });
  });

  describe("hardware-kind inference", () => {
    it("maps fixed unit types directly", () => {
      expect(context._hardwareKind({ type: "NPU" })).toBe("npu");
      expect(context._hardwareKind({ type: "CPU" })).toBe("cpu");
      expect(context._hardwareKind({ type: "CUDA" })).toBe("cuda");
      expect(context._hardwareKind({ type: "AMD" })).toBe("amd");
    });

    it("classifies GPU units as cuda or intel-gpu based on name", () => {
      expect(context._hardwareKind({ type: "GPU", name: "NVIDIA RTX 4090" })).toBe("cuda");
      expect(context._hardwareKind({ type: "GPU", name: "Intel Arc" })).toBe("intel-gpu");
    });

    it("falls back to other for unknown types", () => {
      expect(context._hardwareKind({ type: "TPU" })).toBe("other");
    });

    it("_isUnitUsedByActiveTask matches active tasks by unit id", () => {
      const tasks = [{ unit_id: "GPU.0", status: "active" }, { unit_id: "GPU.1", status: "queued" }];
      expect(context._isUnitUsedByActiveTask("GPU.0", tasks)).toBe(true);
      expect(context._isUnitUsedByActiveTask("GPU.1", tasks)).toBe(false);
      expect(context._isUnitUsedByActiveTask("GPU.2", tasks)).toBe(false);
    });

    it("_applyHardwareUtil overrides false usage using telemetry util", () => {
      const telemetry = { hardware_util: { "GPU.0": 42 } };
      expect(context._applyHardwareUtil(false, "GPU.0", telemetry)).toBe(true);
      expect(context._applyHardwareUtil(true, "GPU.0", telemetry)).toBe(true);
      expect(context._applyHardwareUtil(false, "GPU.1", telemetry)).toBe(false);
      expect(context._applyHardwareUtil(false, "GPU.0", { hardware_util: { "GPU.0": 0 } })).toBe(false);
    });

    it("_isCudaUtilUsed reads nvidia telemetry by index", () => {
      expect(context._isCudaUtilUsed({ nvidia: [{ util: 5 }] }, 0)).toBe(true);
      expect(context._isCudaUtilUsed({ nvidia: [{ util: 0 }] }, 0)).toBe(false);
      expect(context._isCudaUtilUsed({}, 0)).toBe(false);
      expect(context._isCudaUtilUsed({ nvidia: [] }, 0)).toBe(false);
    });

    it("_cpuVisual infers usage from active CPU tasks when not already used", () => {
      expect(context._cpuVisual(true, []).isUsed).toBe(true);
      expect(context._cpuVisual(false, [{ unit_id: "CPU", status: "active" }]).isUsed).toBe(true);
      expect(context._cpuVisual(false, [{ unit_id: "CPU", status: "queued" }]).isUsed).toBe(false);
    });

    it("dispatches visuals per kind via _hardwareKindVisual", () => {
      expect(context._hardwareKindVisual("npu", {}, true, {}, []).icon).toBe("psychology_alt");
      expect(context._hardwareKindVisual("cpu", {}, false, {}, []).icon).toBe("settings_input_component");
      expect(context._hardwareKindVisual("amd", {}, true, {}, []).icon).toBe("bolt");
      expect(context._hardwareKindVisual("intel-gpu", {}, true, {}, []).icon).toBe("developer_board");
      expect(context._hardwareKindVisual("cuda", { id: "CUDA:0" }, true, {}, []).icon).toBe("rocket_launch");
      expect(context._hardwareKindVisual("other", {}, false, {}, []).icon).toBe("memory");
    });

    it("infers intel-gpu/npu usage from legacy telemetry load when hardware_util absent", () => {
      expect(context._intelGpuVisual(false, { intel_gpu_load: 10 }).isUsed).toBe(true);
      expect(context._intelGpuVisual(false, { hardware_util: {}, intel_gpu_load: 10 }).isUsed).toBe(false);
      expect(context._npuVisual(false, { npu_load: 5 }).isUsed).toBe(true);
    });
  });

  describe("_renderHardwareCard / _resolveHardwareUsage", () => {
    it("renders a used-vs-idle badge and status text", () => {
      const data = { telemetry: {}, tasks: [{ unit_id: "CPU", status: "active" }] };
      const html = context._renderHardwareCard({ id: "CPU", type: "CPU", name: "Host CPU" }, data);
      expect(html).toContain("Used");
      expect(html).toContain("status-used");

      const idleHtml = context._renderHardwareCard({ id: "CPU", type: "CPU", name: "Host CPU" }, { telemetry: {}, tasks: [] });
      expect(idleHtml).toContain("Not used");
      expect(idleHtml).toContain("status-idle");
    });

    it("escapes an XSS-bearing unit name/type", () => {
      const html = context._renderHardwareCard(
        { id: "X", type: "<script>alert(1)</script>", name: "<img src=x onerror=alert(1)>" },
        { telemetry: {}, tasks: [] }
      );
      expect(html).not.toContain("<script>");
      expect(html).not.toContain("<img src=x");
    });

    it("defaults missing name/type to placeholder text", () => {
      const html = context._renderHardwareCard({ id: "X" }, { telemetry: {}, tasks: [] });
      expect(html).toContain("Unknown");
      expect(html).toContain("Unnamed Unit");
    });

    it("_renderHardwarePool renders one card per hardware unit", () => {
      context._renderHardwarePool({
        hardware_units: [
          { id: "CPU", type: "CPU", name: "Host CPU" },
          { id: "GPU.0", type: "GPU", name: "Intel GPU" },
        ],
        telemetry: {},
        tasks: [],
      });
      const html = dom.window.document.getElementById("hw-pool").innerHTML;
      expect(html).toContain("Host CPU");
      expect(html).toContain("Intel GPU");
    });
  });

  describe("_renderLastUpdate", () => {
    it("writes an 'Updated:' timestamp", () => {
      context._renderLastUpdate();
      expect(dom.window.document.getElementById("last-update").innerText).toContain("Updated:");
    });
  });

  describe("updateStats", () => {
    it("does nothing when refreshEnabled is false", async () => {
      evalInContext(context, "refreshEnabled = false");
      await context.updateStats();
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it("fetches status and renders top stats/queue/hardware on success", async () => {
      await context.updateStats();
      expect(fetchMock).toHaveBeenCalledWith("/status", { headers: expect.any(Object) });
      expect(dom.window.document.getElementById("app-version").innerText).toBe("Version 1.3.0");
      expect(evalInContext(context, "lastStatusData")).toBeTruthy();
    });

    it("does not render when the fetch response is not ok", async () => {
      fetchMock.mockResolvedValueOnce({ ok: false, status: 500, statusText: "Server Error" });
      await context.updateStats();
      expect(evalInContext(context, "lastStatusData")).toBeNull();
    });

    it("does not render when the response is missing data.system", async () => {
      fetchMock.mockResolvedValueOnce({ ok: true, json: async () => ({}) });
      await context.updateStats();
      expect(evalInContext(context, "lastStatusData")).toBeNull();
    });

    it("ignores a stale response that resolves after a newer request started", async () => {
      let resolveFirst;
      const firstPromise = new Promise((resolve) => {
        resolveFirst = () => resolve({ ok: true, json: async () => buildStatusData({ version: "stale" }) });
      });
      fetchMock.mockImplementationOnce(() => firstPromise);
      fetchMock.mockResolvedValueOnce({ ok: true, json: async () => buildStatusData({ version: "fresh" }) });

      const firstCall = context.updateStats();
      const secondCall = context.updateStats();
      await secondCall;
      resolveFirst();
      await firstCall;

      expect(dom.window.document.getElementById("app-version").innerText).toBe("Version fresh");
    });

    it("swallows unexpected errors without throwing", async () => {
      fetchMock.mockRejectedValueOnce(new Error("network down"));
      await expect(context.updateStats()).resolves.toBeUndefined();
    });
  });
});
