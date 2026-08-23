const path = require("path");
const { JSDOM } = require("jsdom");
const { evalInContext, loadScriptInContext, createMatchMediaStub } = require("./helpers");

function buildSettingsDom() {
  return new JSDOM(`<!doctype html><html><body>
    <div id="settings-section">
      <input type="range" id="retention-range" min="1" max="720" value="24">
      <span id="retention-label">24h</span>

      <input type="range" id="log-retention-range" min="1" max="90" value="7">
      <span id="log-retention-label">7d</span>

      <input type="password" id="api-key-input" />
      <input type="password" id="admin-api-key-input" />
    </div>

    <div id="history-list"></div>
    <div id="task-list"></div>
  </body></html>`, { url: "http://localhost/" });
}

// Slider-label live sync (the `oninput` attribute in dashboard.html) is not exercised here:
// once a JSDOM document is passed into vm.createContext (as loadScriptInContext does below to
// run main.js), inline event-handler attributes stop firing process-wide, even on unrelated
// fresh JSDOM instances. That behavior is covered in the real browser instead, by
// "updates settings range labels on input" in tests/e2e/dashboard-additional-user-flows.spec.cjs
// and by tests/e2e/dashboard-settings.spec.cjs.

describe("dashboard settings", () => {
  let dom;
  let fetchMock;
  let context;
  let alerts;

  beforeEach(() => {
    dom = buildSettingsDom();
    alerts = [];

    fetchMock = vi.fn(async (url) => {
      if (url === "/system/settings") {
        return { ok: true, json: async () => ({}) };
      }
      return { ok: true, json: async () => ({}) };
    });

    const baseContext = {
      window: {
        matchMedia: createMatchMediaStub(false),
        dispatchEvent: () => {},
      },
      document: dom.window.document,
      fetch: fetchMock,
      alert: (msg) => alerts.push(String(msg)),
      confirm: () => true,
      Event: dom.window.Event,
      setTimeout,
      clearTimeout,
      setInterval,
      clearInterval,
      expandedElements: new Set(),
      fullTaskHistory: [],
      rollingTelemetryBuffer: [],
      charts: {},
      localStorage: {
        getItem: () => null,
        setItem: () => {},
        removeItem: () => {},
      },
    };

    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/core/state.js"), baseContext);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/core/utils.js"), context);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/task_filter_history.js"), context);
    context = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/main.js"), context);
  });

  it("loadRetentionSettings applies TELEMETRY_RETENTION_HOURS and LOG_RETENTION_DAYS to the sliders", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ TELEMETRY_RETENTION_HOURS: 48, LOG_RETENTION_DAYS: 14 }),
    });
    dom.window.document.getElementById("admin-api-key-input").value = "admin_key_1";

    await context.loadRetentionSettings();

    expect(fetchMock).toHaveBeenCalledWith("/system/settings", expect.objectContaining({
      headers: expect.objectContaining({ "X-API-Key": "admin_key_1" }),
    }));
    expect(dom.window.document.getElementById("retention-range").value).toBe("48");
    expect(dom.window.document.getElementById("retention-label").textContent).toBe("48h");
    expect(dom.window.document.getElementById("log-retention-range").value).toBe("14");
    expect(dom.window.document.getElementById("log-retention-label").textContent).toBe("14d");
  });

  it("loadRetentionSettings clamps values above slider maxima consistently with labels", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ TELEMETRY_RETENTION_HOURS: 999, LOG_RETENTION_DAYS: 200 }),
    });

    await context.loadRetentionSettings();

    const retentionRange = dom.window.document.getElementById("retention-range");
    const logRange = dom.window.document.getElementById("log-retention-range");
    expect(retentionRange.value).toBe("720");
    expect(dom.window.document.getElementById("retention-label").textContent).toBe("720h");
    expect(logRange.value).toBe("90");
    expect(dom.window.document.getElementById("log-retention-label").textContent).toBe("90d");
  });

  it("loadRetentionSettings applies backend-max retention values without clamping", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ TELEMETRY_RETENTION_HOURS: 720, LOG_RETENTION_DAYS: 90 }),
    });

    await context.loadRetentionSettings();

    expect(dom.window.document.getElementById("retention-range").value).toBe("720");
    expect(dom.window.document.getElementById("retention-label").textContent).toBe("720h");
    expect(dom.window.document.getElementById("log-retention-range").value).toBe("90");
    expect(dom.window.document.getElementById("log-retention-label").textContent).toBe("90d");
  });

  it("loadRetentionSettings does not overwrite a slider the user edited while the request was pending", async () => {
    let resolveSettings;
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => {
      resolveSettings = resolve;
    }));

    const pending = context.loadRetentionSettings();
    const retentionRange = dom.window.document.getElementById("retention-range");
    retentionRange.value = "99";
    retentionRange.dataset.userEdited = "1";

    resolveSettings({
      ok: true,
      json: async () => ({ TELEMETRY_RETENTION_HOURS: 48, LOG_RETENTION_DAYS: 14 }),
    });
    await pending;

    // Untouched slider still hydrates normally from the server response.
    expect(retentionRange.value).toBe("99");
    expect(dom.window.document.getElementById("log-retention-range").value).toBe("14");
  });

  it("saveSettings sends the slider values as integers with auth headers, and alerts on success", async () => {
    dom.window.document.getElementById("retention-range").value = "48";
    dom.window.document.getElementById("log-retention-range").value = "14";
    dom.window.document.getElementById("admin-api-key-input").value = "admin_key_1";

    await context.saveSettings();

    const [url, opts] = fetchMock.mock.calls[fetchMock.mock.calls.length - 1];
    expect(url).toBe("/system/settings");
    expect(opts.method).toBe("POST");
    expect(opts.headers["X-API-Key"]).toBe("admin_key_1");
    expect(JSON.parse(opts.body)).toEqual({
      telemetry_retention_hours: 48,
      log_retention_days: 14,
    });
    expect(alerts.some((msg) => msg.includes("Configuration saved!"))).toBe(true);
  });

  it("saveSettings falls back to the regular API key when the admin key is empty", async () => {
    dom.window.document.getElementById("api-key-input").value = "regular_key";
    dom.window.document.getElementById("admin-api-key-input").value = "";

    await context.saveSettings();

    const [, opts] = fetchMock.mock.calls[fetchMock.mock.calls.length - 1];
    expect(opts.headers["X-API-Key"]).toBe("regular_key");
  });

  it("saveSettings alerts with the status code on a non-OK response", async () => {
    fetchMock.mockResolvedValueOnce({ ok: false, status: 403 });
    await context.saveSettings();
    expect(alerts.some((msg) => msg.includes("Failed to save settings (403)"))).toBe(true);
  });

  it("saveSettings alerts on a network/fetch rejection", async () => {
    fetchMock.mockRejectedValueOnce(new Error("network down"));
    await context.saveSettings();
    expect(alerts.some((msg) => msg.includes("Failed to save settings: Error: network down"))).toBe(true);
  });

  it("persists API keys in-session and restores them via loadDashboardApiKeys, but not across a fresh DOM", () => {
    dom.window.document.getElementById("api-key-input").value = "sess_key";
    dom.window.document.getElementById("admin-api-key-input").value = "sess_admin_key";
    context.persistDashboardApiKeys();

    dom.window.document.getElementById("api-key-input").value = "";
    dom.window.document.getElementById("admin-api-key-input").value = "";
    context.loadDashboardApiKeys();

    expect(dom.window.document.getElementById("api-key-input").value).toBe("sess_key");
    expect(dom.window.document.getElementById("admin-api-key-input").value).toBe("sess_admin_key");

    // A real page reload creates a brand-new JS context, so dashboardApiKey/
    // dashboardAdminApiKey (plain module-level `let`s in core/utils.js) reset to
    // '' there too. Verify against an actually fresh vm context + loadDashboardApiKeys
    // call, not just a fresh DOM's untouched default input values (which would pass
    // trivially regardless of whether the fix under test is real).
    const freshDom = buildSettingsDom();
    let freshContext = {
      window: { matchMedia: createMatchMediaStub(false), dispatchEvent: () => {} },
      document: freshDom.window.document,
      fetch: vi.fn(),
      alert: () => {},
      confirm: () => true,
      Event: freshDom.window.Event,
      setTimeout,
      clearTimeout,
      setInterval,
      clearInterval,
      expandedElements: new Set(),
      fullTaskHistory: [],
      rollingTelemetryBuffer: [],
      charts: {},
      localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
    };
    freshContext = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/core/state.js"), freshContext);
    freshContext = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/core/utils.js"), freshContext);
    freshContext = loadScriptInContext(
      path.join(__dirname, "../../modules/monitoring/templates/dashboard/features/task_filter_history.js"),
      freshContext
    );
    freshContext = loadScriptInContext(path.join(__dirname, "../../modules/monitoring/templates/dashboard/main.js"), freshContext);

    freshContext.loadDashboardApiKeys();

    expect(freshDom.window.document.getElementById("api-key-input").value).toBe("");
    expect(freshDom.window.document.getElementById("admin-api-key-input").value).toBe("");
  });

  it("clearTaskHistory does nothing when the confirm dialog is dismissed", async () => {
    context.confirm = () => false;
    const before = fetchMock.mock.calls.length;
    await context.clearTaskHistory();
    expect(fetchMock.mock.calls.length).toBe(before);
  });

  it("clearTaskHistory clears history and re-renders on success", async () => {
    evalInContext(context, "fullTaskHistory = [{ task_id: 't1' }]");
    let renderCalls = 0;
    context.renderHistory = () => {
      renderCalls += 1;
    };
    fetchMock.mockResolvedValueOnce({ ok: true });

    await context.clearTaskHistory();

    expect(alerts.some((msg) => msg.includes("Task history purged successfully."))).toBe(true);
    expect(renderCalls).toBe(1);
    expect(evalInContext(context, "fullTaskHistory.length")).toBe(0);
  });

  it("clearTaskHistory alerts on a non-OK response", async () => {
    fetchMock.mockResolvedValueOnce({ ok: false });
    await context.clearTaskHistory();
    expect(alerts.some((msg) => msg.includes("Failed to clear task history."))).toBe(true);
  });

  it("clearTaskHistory alerts on a fetch rejection", async () => {
    fetchMock.mockRejectedValueOnce(new Error("boom"));
    await context.clearTaskHistory();
    expect(alerts.some((msg) => msg.startsWith("Error:"))).toBe(true);
  });

  it("clearTelemetryMetrics does nothing when the confirm dialog is dismissed", async () => {
    context.confirm = () => false;
    const before = fetchMock.mock.calls.length;
    await context.clearTelemetryMetrics();
    expect(fetchMock.mock.calls.length).toBe(before);
  });

  it("clearTelemetryMetrics resets buffers and chart stats on success", async () => {
    evalInContext(context, "rollingTelemetryBuffer = [1, 2, 3]");
    let resetCalls = 0;
    context.resetTelemetryChartsAndStats = () => {
      resetCalls += 1;
    };
    context.renderCharts = () => {};
    fetchMock.mockResolvedValueOnce({ ok: true });

    await context.clearTelemetryMetrics();

    expect(alerts.some((msg) => msg.includes("Telemetry history purged successfully."))).toBe(true);
    expect(resetCalls).toBe(1);
    expect(evalInContext(context, "rollingTelemetryBuffer.length")).toBe(0);
  });

  it("clearTelemetryMetrics alerts on a non-OK response", async () => {
    fetchMock.mockResolvedValueOnce({ ok: false });
    await context.clearTelemetryMetrics();
    expect(alerts.some((msg) => msg.includes("Failed to clear telemetry metrics."))).toBe(true);
  });

  it("clearTelemetryMetrics alerts on a fetch rejection", async () => {
    fetchMock.mockRejectedValueOnce(new Error("telemetry boom"));
    await context.clearTelemetryMetrics();
    expect(alerts.some((msg) => msg.startsWith("Error:"))).toBe(true);
  });
});
