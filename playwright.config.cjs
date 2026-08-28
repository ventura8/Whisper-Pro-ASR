const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "tests/e2e",
  // tests/e2e/real/** is a separate project (playwright.real.config.cjs) that runs
  // against the real FastAPI app instead of this suite's fixture mock server.
  testIgnore: "real/**",
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI
    ? [["list", { printSteps: true }], ["html", { open: "never" }]]
    : [["list", { printSteps: true }]],
  use: {
    baseURL: "http://127.0.0.1:9614",
    browserName: "chromium",
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  webServer: {
    command: "node tests/e2e/dashboard-fixture-server.cjs",
    url: "http://127.0.0.1:9614/healthz",
    timeout: 30_000,
    reuseExistingServer: !process.env.CI,
  },
});