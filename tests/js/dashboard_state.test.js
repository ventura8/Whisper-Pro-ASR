const path = require("path");
const { evalInContext, loadScriptInContext } = require("./helpers");

describe("dashboard_state.js", () => {
  it("initializes dashboard state globals", () => {
    const context = loadScriptInContext(
      path.join(__dirname, "../../modules/monitoring/templates/dashboard_state.js")
    );

    expect(evalInContext(context, "currentTab")).toBe("active");
    expect(evalInContext(context, "Array.isArray(rollingTelemetryBuffer)")).toBe(true);
    expect(evalInContext(context, "typeof lastStatusData")).toBe("object");
    expect(evalInContext(context, "refreshEnabled")).toBe(true);
    expect(evalInContext(context, "Array.isArray(COLORS)")).toBe(true);
    expect(evalInContext(context, "COLORS.length")).toBeGreaterThan(0);
  });
});
