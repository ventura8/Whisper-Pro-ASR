const path = require("path");
const { JSDOM } = require("jsdom");
const { evalInContext, loadScriptInContext, createMatchMediaStub } = require("./helpers");

describe("analytics.js", () => {
  let dom;
  let context;
  let fetchMock;

  beforeEach(() => {
    dom = new JSDOM(`<!doctype html><html><body>
      <div id="val-all-time"></div>
      <div id="val-this-month"></div>
      <div id="val-total-tasks"></div>
      <div id="val-today"></div>
      <div id="val-today-tasks"></div>
      <div id="val-avg-duration"></div>
      <div id="val-asr-cumulative"></div>
      <div id="val-asr-count"></div>
      <div id="val-detectlang-cumulative"></div>
      <div id="val-detectlang-count"></div>
      <div id="val-audio-cumulative"></div>
      <div id="val-audio-count"></div>
      <table><tbody id="table-body"></tbody></table>
      <div id="tasksChart"></div>
      <div id="durationChart"></div>
      <div id="last-update"></div>
    </body></html>`);

    class FakeApexCharts {
      constructor(el, options) {
        this.el = el;
        this.options = options;
      }
      render() {}
      updateOptions(next) {
        this.options = { ...this.options, ...next };
      }
    }

    fetchMock = vi.fn(async () => ({ json: async () => ({}) }));

    context = loadScriptInContext(
      path.join(__dirname, "../../modules/monitoring/templates/analytics.js"),
      {
        window: {
          matchMedia: createMatchMediaStub(false),
        },
        document: dom.window.document,
        fetch: fetchMock,
        ApexCharts: FakeApexCharts,
      }
    );
  });

  it("renders analytics cards, table, and charts", () => {
    const data = {
      cumulative: {
        all_time: 3600,
        this_month: 1200,
        count_all_time: 3,
        today: 600,
        count_today: 1,
        asr: { count: 2, duration: 700 },
        detectlang: { count: 1, duration: 150 },
        audio: { count: 0, duration: 0 },
      },
      daily: {
        "2026-06-20": { count: 1, duration: 300, asr: { count: 1, duration: 300 } },
        "2026-06-21": { count: 2, duration: 800, detectlang: { count: 1, duration: 100 }, asr: { count: 1, duration: 700 } },
      },
    };

    context.renderAnalytics(data);

    expect(String(dom.window.document.getElementById("val-total-tasks").innerText)).toBe("3");
    expect(dom.window.document.getElementById("table-body").innerHTML).toContain("2026-06-21");
    expect(String(dom.window.document.getElementById("last-update").innerText)).toContain("Updated:");
  });

  it("formats durations consistently", () => {
    expect(context.formatDuration(5)).toBe("5.0s");
    expect(context.formatDuration(125)).toContain("2m");
    expect(context.formatDDHHMMSS(90061)).toBe("1d 1h 1m");
    expect(context.formatDuration(3600)).toContain("1h");
    expect(context.formatDuration(90000)).toContain("1d");
    expect(context.formatDuration(-1)).toBe("0s");
    expect(context.formatDDHHMMSS(-1)).toBe("0d 0h 0m");
  });

  it("fetches analytics and calls renderer", async () => {
    const payload = { cumulative: { all_time: 10, count_all_time: 1 }, daily: {} };
    fetchMock.mockResolvedValueOnce({ json: async () => payload });
    const renderSpy = vi.spyOn(context, "renderAnalytics");

    await context.fetchAnalytics();

    expect(fetchMock).toHaveBeenCalledWith("/analytics", expect.any(Object));
    expect(renderSpy).toHaveBeenCalled();
  });

  it("handles fetch analytics failure", async () => {
    fetchMock.mockRejectedValueOnce(new Error("network"));
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await context.fetchAnalytics();

    expect(errorSpy).toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  it("updates existing charts and exports json", () => {
    const chartEl = dom.window.document.getElementById("tasksChart");
    evalInContext(context, "charts.tasksChart = { updateOptions: (...args) => { globalThis.__chartUpdateArgs = args; } }");

    context.updateOrCreateChart("tasksChart", {
      xaxis: { categories: ["2026-06-23"] },
      series: [{ name: "x", data: [1] }],
      theme: { mode: "light" },
      grid: { borderColor: "#fff" },
    });

    expect(evalInContext(context, "Array.isArray(globalThis.__chartUpdateArgs)")).toBe(true);

    evalInContext(context, "rawData = { ok: true }");
    const appendSpy = vi.spyOn(dom.window.document.body, "appendChild");
    const removeSpy = vi.spyOn(dom.window.Element.prototype, "remove");

    context.exportJson();

    expect(appendSpy).toHaveBeenCalled();
    expect(removeSpy).toHaveBeenCalled();
    expect(chartEl).toBeTruthy();
  });

  it("renders empty daily state and handles missing chart target / empty export", () => {
    context.renderAnalytics({ cumulative: {}, daily: {} });
    expect(dom.window.document.getElementById("table-body").innerHTML).toContain("No analytics data recorded yet");

    context.updateOrCreateChart("missing", {
      xaxis: { categories: [] },
      series: [],
      theme: { mode: "light" },
      grid: { borderColor: "#000" },
    });

    const appendSpy = vi.spyOn(dom.window.document.body, "appendChild");
    evalInContext(context, "rawData = null");
    context.exportJson();
    expect(appendSpy).not.toHaveBeenCalled();
  });

  it("executes chart formatter callbacks and daily default branches", () => {
    context.window.matchMedia = createMatchMediaStub(true);
    context.renderAnalytics({
      cumulative: { all_time: 0, count_all_time: 0, today: 0 },
      daily: {
        "2026-06-01": { count: 1, duration: 70 },
        "2026-06-02": { duration: 0, asr: { duration: 0 }, detectlang: { duration: 0 }, audio: { duration: 0 } },
      },
    });

    const tasksChart = evalInContext(context, "charts.tasksChart");
    const durationChart = evalInContext(context, "charts.durationChart");

    expect(tasksChart.options.yaxis.labels.formatter(7.2)).toBe("7");
    expect(durationChart.options.yaxis.labels.formatter(1.2)).toBe("1.2 m");
  });

  it("onload registers dark-mode listener and rerenders only when rawData exists", async () => {
    let listener = null;
    const fetchSpy = vi.spyOn(context, "fetchAnalytics").mockResolvedValue();
    context.window.matchMedia = () => ({
      matches: false,
      addEventListener: (_evt, cb) => {
        listener = cb;
      },
    });

    const renderSpy = vi.spyOn(context, "renderAnalytics");
    context.window.onload();
    expect(fetchSpy).toHaveBeenCalled();
    expect(typeof listener).toBe("function");

    evalInContext(context, "rawData = null");
    listener();
    expect(renderSpy).not.toHaveBeenCalled();

    evalInContext(context, "rawData = { cumulative: {}, daily: {} }");
    listener();
    expect(renderSpy).toHaveBeenCalled();
  });
});
