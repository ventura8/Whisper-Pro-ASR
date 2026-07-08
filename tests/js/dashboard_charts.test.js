const path = require("path");
const { JSDOM } = require("jsdom");
const { evalInContext, loadScriptInContext, createMatchMediaStub } = require("./helpers");

describe("dashboard_charts.js", () => {
  let dom;
  let chartInstances;
  let context;

  beforeEach(() => {
    dom = new JSDOM(`<!doctype html><html><body>
      <div id="cpuChart"></div>
      <div id="memChart"></div>
      <div id="hw-stats"></div>
      <div id="hwChart"></div>
    </body></html>`);

    chartInstances = {};

    class FakeApexCharts {
      constructor(el, options) {
        this.el = el;
        this.options = options;
        this.updates = 0;
        chartInstances[el.id] = this;
      }
      render() {}
      updateOptions() {
        this.updates += 1;
      }
      updateSeries() {
        this.updates += 1;
      }
    }

    context = loadScriptInContext(
      path.join(__dirname, "../../modules/monitoring/templates/dashboard_charts.js"),
      {
        window: {
          matchMedia: createMatchMediaStub(false),
        },
        document: dom.window.document,
        Date,
        lastStatusData: {
          hardware_units: [
            { id: "NPU.0", type: "NPU", name: "Intel NPU" },
            { id: "GPU.0", type: "GPU", name: "Intel GPU" },
          ],
        },
        rollingTelemetryBuffer: [
          {
            timestamp: Date.now() / 1000,
            system: { cpu_percent: 10, app_cpu_percent: 5, app_memory_gb: 1.2 },
            telemetry: { hardware_util: { "NPU.0": 55, "GPU.0": 40 }, nvidia: [] },
          },
        ],
        chartWindowMinutes: 1,
        charts: {},
        lastChartStates: {},
        COLORS: ["#111", "#222", "#333", "#444"],
        ApexCharts: FakeApexCharts,
      }
    );
  });

  it("renders and updates charts", () => {
    context.renderCharts();
    expect(chartInstances.cpuChart).toBeTruthy();
    expect(chartInstances.memChart).toBeTruthy();
    expect(chartInstances.hwChart).toBeTruthy();
    expect(dom.window.document.querySelector('#hw-stats .stat-box')).toBeTruthy();

    context.changeChartWindow("2");
    context.renderCharts();

    expect(chartInstances.cpuChart.updates).toBeGreaterThanOrEqual(1);
  });

  it("covers early returns, fallback dataset branch, and updateSeries path", () => {
    evalInContext(context, "lastStatusData = null");
    context.renderCharts();
    expect(chartInstances.cpuChart).toBeUndefined();

    evalInContext(context, "lastStatusData = { hardware_units: [] }");
    evalInContext(context, "rollingTelemetryBuffer = []");
    context.renderCharts();
    expect(chartInstances.memChart).toBeUndefined();

    evalInContext(
      context,
      "lastStatusData = { hardware_units: [{ id: 'CUDA:0', type: 'CUDA', name: 'NVIDIA GPU 0' }, { id: 'GPU.0', type: 'GPU', name: 'Intel Arc' }, { id: 'NPU.0', type: 'NPU', name: 'Intel NPU' }] }"
    );
    evalInContext(
      context,
      "rollingTelemetryBuffer = [{ timestamp: Date.now() / 1000, system: { cpu_percent: 25, app_cpu_percent: 12, app_memory_gb: 2.5 }, telemetry: { nvidia: [{ util: 77 }], intel_gpu_load: 41, npu_load: 36 } }]"
    );

    context.renderCharts();
    expect(chartInstances.hwChart).toBeTruthy();

    // second call with same range/theme/y-axis should use updateSeries branch
    const before = chartInstances.memChart.updates;
    context.renderCharts();
    expect(chartInstances.memChart.updates).toBeGreaterThan(before);
  });

  it("covers createOrUpdateLineChart no-element and empty-series behavior", () => {
    context.createOrUpdateLineChart("missing", [], true);

    context.createOrUpdateLineChart("hwChart", [], true);
    expect(chartInstances.hwChart).toBeTruthy();
  });

  it("uses updateOptions when chart state changes", () => {
    context.renderCharts();
    const before = chartInstances.cpuChart.updates;

    context.changeChartWindow("2");
    expect(chartInstances.cpuChart.updates).toBeGreaterThan(before);
  });

  it("executes y-axis and tooltip formatter callbacks", () => {
    context.renderCharts();

    const cpu = chartInstances.cpuChart;
    const mem = chartInstances.memChart;
    const hw = chartInstances.hwChart;

    expect(cpu.options.yaxis.labels.formatter(55)).toBe("55.0");
    expect(mem.options.yaxis.labels.formatter(1.234)).toBe("1.2");
    expect(hw.options.yaxis.labels.formatter(55)).toBe("55%");
    expect(hw.options.tooltip.y.formatter(12.34)).toBe("12.3%");
    expect(mem.options.tooltip.y.formatter(3.456)).toBe("3.46 GB");
  });

  it("covers long-buffer sorting and slicing path", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 1");
    evalInContext(context, "lastStatusData = { hardware_units: [{ id: 'GPU.0', type: 'GPU', name: 'Intel GPU' }] }");
    evalInContext(
      context,
      `rollingTelemetryBuffer = Array.from({ length: 45 }, (_, i) => ({
        timestamp: ${now} - (45 - i),
        system: { cpu_percent: i, app_cpu_percent: i / 2, app_memory_gb: 1 + (i / 100) },
        telemetry: { hardware_util: { 'GPU.0': i }, nvidia: [] }
      }))`
    );

    context.renderCharts();
    expect(chartInstances.cpuChart).toBeTruthy();
  });

  it("covers all hardware type branches and legacy telemetry fallback paths", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 1");
    evalInContext(
      context,
      `lastStatusData = {
        hardware_units: [
          { id: 'CUDA:0', type: 'CUDA', name: 'NVIDIA GPU 0' },
          { id: 'GPU.0', type: 'GPU', name: 'Intel Arc GPU' },
          { id: 'NPU.0', type: 'NPU', name: 'Intel NPU' },
          { id: 'CPU', type: 'CPU', name: 'Host CPU' }
        ]
      }`
    );
    evalInContext(
      context,
      `rollingTelemetryBuffer = [{
        timestamp: ${now},
        system: { cpu_percent: 20, app_cpu_percent: 10, app_memory_gb: 2.0 },
        telemetry: { nvidia: [{ util: 75 }, { util: 60 }], intel_gpu_load: 50, npu_load: 40, hardware_util: {} },
        cpu_sys: 20,
        cpu_app: 10,
        mem_app_gb: 2.0,
        nvidia_util: [{ util: 75 }, { util: 60 }],
        intel_util: 50,
        npu_util: 40
      }]`
    );

    context.renderCharts();
    expect(chartInstances.hwChart).toBeTruthy();
  });

  it("covers legacy telemetry structure without hardware_util fallback", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 1");
    evalInContext(
      context,
      `lastStatusData = {
        hardware_units: [
          { id: 'CUDA:1', type: 'CUDA', name: 'Second GPU' },
          { id: 'GPU.0', type: 'GPU', name: 'Intel GPU Legacy' },
          { id: 'NPU.0', type: 'NPU', name: 'NPU Legacy' }
        ]
      }`
    );
    evalInContext(
      context,
      `rollingTelemetryBuffer = [{
        timestamp: ${now},
        system: null,
        telemetry: { nvidia: [undefined, { util: 55 }], intel_gpu_load: 35, npu_load: 25 },
        cpu_sys: 15,
        cpu_app: 8,
        mem_app_gb: 1.5,
        nvidia_util: [undefined, { util: 55 }],
        intel_util: 35,
        npu_util: 25
      }]`
    );

    context.renderCharts();
    expect(chartInstances.hwChart).toBeTruthy();
  });

  it("covers empty hwDatasets and No Acceleration fallback", () => {
    evalInContext(context, "chartWindowMinutes = 1");
    evalInContext(
      context,
      `lastStatusData = { hardware_units: [{ id: 'CPU', type: 'CPU', name: 'Host CPU' }] }`
    );
    evalInContext(
      context,
      `rollingTelemetryBuffer = [{
        timestamp: Date.now() / 1000,
        system: { cpu_percent: 5, app_cpu_percent: 2, app_memory_gb: 0.8 },
        telemetry: {}
      }]`
    );

    context.renderCharts();
    expect(chartInstances.hwChart).toBeTruthy();
  });

  it("covers dark mode theme change path", () => {
    context.window.matchMedia = createMatchMediaStub(true);
    context.renderCharts();
    expect(chartInstances.cpuChart.options.theme.mode).toBe("dark");
  });

  it("covers formatData callbacks for all chart types", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 5");
    evalInContext(
      context,
      `lastStatusData = {
        hardware_units: [
          { id: 'CUDA:0', type: 'CUDA', name: 'NVIDIA' },
          { id: 'CPU', type: 'CPU', name: 'Host CPU' }
        ]
      }`
    );
    evalInContext(
      context,
      `rollingTelemetryBuffer = Array.from({ length: 10 }, (_, i) => ({
        timestamp: ${now} - (10 - i),
        system: { cpu_percent: 20 + i, app_cpu_percent: 10 + i, app_memory_gb: 2.5 + (i * 0.1) },
        telemetry: { nvidia: [{ util: 60 + i }], hardware_util: { 'CUDA:0': 60 + i } }
      }))`
    );

    context.renderCharts();
    expect(chartInstances.cpuChart).toBeDefined();
    expect(chartInstances.memChart).toBeDefined();
    expect(chartInstances.hwChart).toBeDefined();
  });

  it("covers missing telemetry and null hardware_util branches", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 1");
    evalInContext(
      context,
      `lastStatusData = {
        hardware_units: [
          { id: 'GPU.0', type: 'GPU', name: 'Intel GPU' },
          { id: 'NPU.0', type: 'NPU', name: 'Intel NPU' }
        ]
      }`
    );
    evalInContext(
      context,
      `rollingTelemetryBuffer = [{
        timestamp: ${now},
        system: null,
        telemetry: null,
        nvidia_util: null,
        intel_util: null,
        npu_util: null
      }]`
    );

    context.renderCharts();
    expect(chartInstances.hwChart).toBeDefined();
  });

  it("covers multiple GPU and NPU units with varied utilization", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 2");
    evalInContext(
      context,
      `lastStatusData = {
        hardware_units: [
          { id: 'CUDA:0', type: 'CUDA', name: 'GPU 0' },
          { id: 'CUDA:1', type: 'CUDA', name: 'GPU 1' },
          { id: 'GPU.0', type: 'GPU', name: 'Intel Arc' },
          { id: 'NPU.0', type: 'NPU', name: 'Intel NPU' }
        ]
      }`
    );
    evalInContext(
      context,
      `rollingTelemetryBuffer = [{
        timestamp: ${now},
        system: { cpu_percent: 30, app_cpu_percent: 15, app_memory_gb: 3.0 },
        telemetry: {
          nvidia: [{ util: 85 }, { util: 70 }],
          intel_gpu_load: 55,
          npu_load: 45,
          hardware_util: { 'CUDA:0': 85, 'CUDA:1': 70, 'GPU.0': 55, 'NPU.0': 45 }
        }
      }]`
    );

    context.renderCharts();
    expect(chartInstances.hwChart).toBeDefined();
  });

  it("covers chart update with chart window change", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 10");
    evalInContext(
      context,
      `lastStatusData = { hardware_units: [{ id: 'CPU', type: 'CPU', name: 'Host CPU' }] }`
    );
    evalInContext(
      context,
      `rollingTelemetryBuffer = Array.from({ length: 20 }, (_, i) => ({
        timestamp: ${now} - (20 - i),
        system: { cpu_percent: 15 + Math.sin(i) * 5, app_cpu_percent: 8, app_memory_gb: 2.0 },
        telemetry: { hardware_util: { CPU: 0 } }
      }))`
    );

    context.renderCharts();
    expect(chartInstances.cpuChart).toBeDefined();

    // Change window size and re-render
    evalInContext(context, "chartWindowMinutes = 2");
    context.renderCharts();
    expect(chartInstances.cpuChart).toBeDefined();
  });

  it("covers new chart creation when buffer initially small", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 15");
    evalInContext(
      context,
      `lastStatusData = { hardware_units: [{ id: 'CPU', type: 'CPU', name: 'Host CPU' }] }`
    );
    evalInContext(
      context,
      `rollingTelemetryBuffer = [{
        timestamp: ${now},
        system: { cpu_percent: 10, app_cpu_percent: 5, app_memory_gb: 1.5 },
        telemetry: {}
      }]`
    );

    // Render which should create new charts since this is first render with small buffer
    context.renderCharts();
    expect(chartInstances.cpuChart).toBeDefined();
  });

  it("covers formatter functions with boundary values", () => {
    context.renderCharts();
    const cpu = chartInstances.cpuChart;
    const mem = chartInstances.memChart;
    const hw = chartInstances.hwChart;

    // Test edge values
    expect(cpu.options.yaxis.labels.formatter(0)).toBe("0.0");
    expect(cpu.options.yaxis.labels.formatter(100)).toBe("100.0");
    expect(mem.options.yaxis.labels.formatter(0)).toBe("0.0");
    expect(mem.options.yaxis.labels.formatter(32)).toBe("32.0");
    expect(hw.options.yaxis.labels.formatter(0)).toBe("0%");
    expect(hw.options.yaxis.labels.formatter(100)).toBe("100%");
  });

  it("covers changeChartWindow function directly", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 5");
    evalInContext(
      context,
      `lastStatusData = { hardware_units: [{ id: 'CPU', type: 'CPU', name: 'Host CPU' }] }`
    );
    evalInContext(
      context,
      `rollingTelemetryBuffer = Array.from({ length: 30 }, (_, i) => ({
        timestamp: ${now} - (30 - i),
        system: { cpu_percent: 20 + i, app_cpu_percent: 10, app_memory_gb: 2.0 },
        telemetry: {}
      }))`
    );

    // Call changeChartWindow which calls renderCharts
    context.changeChartWindow("10");
    expect(evalInContext(context, "chartWindowMinutes")).toBe(10);
    expect(chartInstances.cpuChart).toBeDefined();
  });

  it("covers createOrUpdateLineChart with empty datasets (No Acceleration fallback)", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 1");
    evalInContext(
      context,
      `lastStatusData = { hardware_units: [{ id: 'CPU', type: 'CPU', name: 'Host CPU' }] }`
    );
    evalInContext(
      context,
      `rollingTelemetryBuffer = [{
        timestamp: ${now},
        system: { cpu_percent: 10, app_cpu_percent: 5, app_memory_gb: 1.5 },
        telemetry: {}
      }]`
    );

    context.renderCharts();
    // When no hardware units except CPU, hwChart should render with empty dataset fallback
    expect(chartInstances.hwChart).toBeDefined();
  });

  it("covers tooltip y formatter for memory and percentage values", () => {
    context.renderCharts();
    const mem = chartInstances.memChart;
    const hw = chartInstances.hwChart;

    // Test tooltip formatters with various values
    expect(mem.options.tooltip.y.formatter(3.456)).toBe("3.46 GB");
    expect(mem.options.tooltip.y.formatter(0.123)).toBe("0.12 GB");
    expect(hw.options.tooltip.y.formatter(45.67)).toBe("45.7%");
    expect(hw.options.tooltip.y.formatter(99.99)).toBe("100.0%");
  });

  it("covers yaxis label formatter for different data types", () => {
    context.renderCharts();
    const mem = chartInstances.memChart;
    const hw = chartInstances.hwChart;

    // Memory chart uses GB format (percent = false)
    expect(mem.options.yaxis.labels.formatter(5.5)).toBe("5.5");
    // Hardware chart uses percentage format (percent = true)
    expect(hw.options.yaxis.labels.formatter(85)).toBe("85%");
  });

  it("covers CUDA GPU fallback path with legacy nvidia array structure", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 1");
    
    // Test the legacy CUDA fallback path
    evalInContext(
      context,
      `lastStatusData = {
        system: { cpu_percent: 35, app_cpu_percent: 20, app_memory_gb: 2.5, memory_total_gb: 16, memory_used_gb: 7 },
        hardware_units: [
          { id: 'CUDA:0', type: 'CUDA', name: 'NVIDIA GPU 0' },
          { id: 'CUDA:1', type: 'CUDA', name: 'NVIDIA GPU 1' }
        ],
        telemetry: { hardware_util: {}, nvidia: [] }
      }`
    );
    
    // Create telemetry buffer with legacy nvidia array (no hardware_util)
    // This forces the fallback path in hwDatasets.map()
    const telemetryPoints = [];
    for (let i = 0; i < 5; i++) {
      telemetryPoints.push({
        timestamp: now - (5 - i),
        system: { cpu_percent: 30, app_cpu_percent: 15, app_memory_gb: 2.0 },
        telemetry: {
          // NO hardware_util property, forcing fallback
          nvidia: [{ util: 75 }, { util: 65 }],
          intel_gpu_load: 0,
          npu_load: 0
        },
        // Legacy fallback fields
        nvidia_util: [{ util: 70 }, { util: 60 }]
      });
    }
    evalInContext(context, `rollingTelemetryBuffer = ${JSON.stringify(telemetryPoints)}`);

    context.renderCharts();
    
    // Verify hardware chart was created with fallback data
    expect(chartInstances.hwChart).toBeDefined();
    expect(chartInstances.hwChart.options.series.length).toBeGreaterThan(0);
  });

  it("covers null-valued telemetry points and verifies renderCharts still completes", () => {
    const now = Date.now() / 1000;
    evalInContext(context, "chartWindowMinutes = 1");
    evalInContext(
      context,
      `lastStatusData = {
        system: { cpu_percent: 35, app_cpu_percent: 20, app_memory_gb: 2.5, memory_total_gb: 16, memory_used_gb: 7 },
        hardware_units: [
          { id: 'CUDA:0', type: 'CUDA', name: 'NVIDIA GPU 0' }
        ],
        telemetry: { hardware_util: {}, nvidia: [] }
      }`
    );
    const telemetryPoints = [
      {
        timestamp: now - 2,
        system: null,
        cpu_sys: null,
        cpu_app: null,
        mem_app_gb: null,
        telemetry: {
          nvidia: [],
          intel_gpu_load: 0,
          npu_load: 0
        }
      },
      {
        timestamp: now - 1,
        system: { cpu_percent: 30, app_cpu_percent: 15, app_memory_gb: 2.0 },
        telemetry: {
          nvidia: [{ util: 75 }],
          intel_gpu_load: 0,
          npu_load: 0
        }
      }
    ];
    evalInContext(context, `rollingTelemetryBuffer = ${JSON.stringify(telemetryPoints)}`);
    context.renderCharts();
    expect(chartInstances.cpuChart).toBeDefined();
  });
});
