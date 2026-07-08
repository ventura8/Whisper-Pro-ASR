const { defineConfig } = require("vitest/config");

module.exports = defineConfig({
  test: {
    globals: true,
    environment: "jsdom",
    include: ["tests/js/**/*.test.js"],
    coverage: {
      provider: "v8",
      reportsDirectory: "coverage-js",
      reporter: ["text", "lcov"],
      include: ["modules/monitoring/templates/*.js"],
      thresholds: {
        perFile: true,
        lines: 90,
        statements: 90,
        branches: 68,
        functions: 83,
      },
    },
  },
});
