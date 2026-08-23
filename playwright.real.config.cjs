const os = require("os");
const path = require("path");
const { defineConfig } = require("@playwright/test");

// Real-backend e2e project: runs against the actual FastAPI app (whisper_pro_asr.create_app),
// with only ASR inference and language-detection voting patched to deterministic fakes (see
// tests/e2e/real_backend/serve_real_app.py). Routing, history_manager, telemetry_manager,
// settings persistence, and the auth/admin-key middleware are all genuinely live. Kept as a
// separate project/config from the fixture-mock suite (playwright.config.cjs) so both can run
// independently or together.
const stateDir = path.join(os.tmpdir(), `whisper-e2e-real-state-${process.pid}-${Date.now()}`);

module.exports = defineConfig({
  testDir: "tests/e2e/real",
  outputDir: "test-results-real",
  timeout: 30_000,
  expect: { timeout: 8_000 },
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI
    ? [["list", { printSteps: true }], ["html", { open: "never", outputFolder: "playwright-report-real" }]]
    : [["list", { printSteps: true }]],
  use: {
    baseURL: "http://127.0.0.1:9615",
    browserName: "chromium",
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  webServer: {
    command: `poetry run python tests/e2e/real_backend/serve_real_app.py`,
    url: "http://127.0.0.1:9615/asr",
    timeout: 60_000,
    reuseExistingServer: false,
    env: {
      WHISPER_STATE_DIR: stateDir,
      WHISPER_E2E_PORT: "9615",
      ADMIN_API_KEY: "e2e_real_admin_key",
      ASR_MODEL: "tiny",
      PYTHONPATH: __dirname,
    },
  },
});
