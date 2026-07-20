/* global describe, it, expect, beforeEach, vi */
const path = require("path");
const { JSDOM } = require("jsdom");
const { loadScriptInContext } = require("./helpers");

describe("utils.js", () => {
  let dom;
  let context;

  beforeEach(() => {
    dom = new JSDOM("<!doctype html><html><body></body></html>");
    const alertMock = vi.fn();
    context = loadScriptInContext(
      path.join(__dirname, "../../modules/monitoring/templates/dashboard/core/utils.js"),
      {
        window: dom.window,
        document: dom.window.document,
        Blob: dom.window.Blob,
        fullTaskHistory: [
          { task_id: "ok", filename: "movie.mkv", result: { text: "1\n00:00:00,000 --> 00:00:01,000\nHi" } },
          { filename: "only-name.mp4", result: { text: "caption" } },
        ],
        alert: alertMock,
      }
    );

    context.__alertMock = alertMock;
    context.window.URL.createObjectURL = vi.fn(() => "blob://test");
    context.window.URL.revokeObjectURL = vi.fn();
    vi.spyOn(dom.window.HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  });

  it("formats hardware labels", () => {
    expect(context.getHwIconAndLabel("CUDA:0").label).toContain("NVIDIA GPU");
    expect(context.getHwIconAndLabel("NPU.0").label).toContain("Intel NPU");
    expect(context.getHwIconAndLabel("GPU.0").label).toContain("Intel GPU");
    expect(context.getHwIconAndLabel("AMD:0").label).toContain("AMD GPU");
    expect(context.getHwIconAndLabel("CPU").label).toContain("Host CPU");
    expect(context.getHwIconAndLabel("ASIC-1").label).toBe("ASIC-1");
    expect(context.getHwIconAndLabel(null).label).toBe("Queued");
  });

  it("keeps API keys in session state and prefers admin headers for admin requests", () => {
    const store = {};
    context.localStorage = {
      getItem: (key) => store[key] || null,
      setItem: (key, value) => {
        store[key] = value;
      },
      removeItem: (key) => {
        delete store[key];
      },
    };
    context.document = {
      getElementById: (id) => {
        if (id === "api-key-input") {
          return { value: "api-secret" };
        }
        if (id === "admin-api-key-input") {
          return { value: "admin-secret" };
        }
        return null;
      },
    };

    context.persistDashboardApiKeys();
    expect(store.whisper_api_key).toBeUndefined();
    expect(store.whisper_admin_api_key).toBeUndefined();
    expect(context.getAuthHeaders()["X-API-Key"]).toBe("api-secret");
    expect(context.getAuthHeaders(null, true)["X-API-Key"]).toBe("admin-secret");
    expect(context.getAuthHeaders("application/json")["Content-Type"]).toBe("application/json");

    const loaded = { "api-key-input": { value: "" }, "admin-api-key-input": { value: "" } };
    context.document = {
      getElementById: (id) => loaded[id] || null,
    };
    context.loadDashboardApiKeys();
    expect(loaded["api-key-input"].value).toBe("api-secret");
    expect(loaded["admin-api-key-input"].value).toBe("admin-secret");

    context.document = {
      getElementById: (id) => (id === "api-key-input" ? { value: "" } : { value: "" }),
    };
    context.persistDashboardApiKeys();
    expect(store.whisper_api_key).toBeUndefined();
    expect(store.whisper_admin_api_key).toBeUndefined();
    expect(context.getAuthHeaders()["X-API-Key"]).toBeUndefined();
    expect(context.getAuthHeaders(null, true)["X-API-Key"]).toBeUndefined();
    expect(context.getAuthHeaders()["X-Requested-With"]).toBe("XMLHttpRequest");
    context.persistDashboardApiKeys();

    context.document = { getElementById: () => null };
    context.persistDashboardApiKeys();
    context.loadDashboardApiKeys();
  });

  it("escapes html and formats durations", () => {
    expect(context.escapeHtml('<x>&"\'' )).toContain("&lt;x&gt;");
    expect(context.escapeHtml("")).toBe("");
    expect(context.formatDur(3661)).toBe("01:01:01");
    expect(context.formatDur(-1)).toBe("00:00:00");
    expect(context.formatDDHHMMSS(90061)).toBe("1d 1h 1m");
    expect(context.formatDDHHMMSS(-1)).toBe("0d 0h 0m");
  });

  it("computes timer text for queued and active tasks", () => {
    const now = 100;
    expect(context.getTimerText({ status: "queued", start_time: 90 }, now)).toContain("Queued for");
    expect(context.getTimerText({ status: "active", start_time: 80, start_active: 90 }, now)).toContain("Running");
  });

  it("downloads srt content and handles missing task content", () => {
    context.downloadSrtById("ok");
    context.downloadSrtById("only-name.mp4");
    expect(context.window.URL.createObjectURL).toHaveBeenCalled();
    expect(context.window.URL.revokeObjectURL).toHaveBeenCalled();

    context.downloadSrtById("missing");
    expect(context.__alertMock).toHaveBeenCalled();
  });
});
