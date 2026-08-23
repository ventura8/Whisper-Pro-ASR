const { test, expect } = require("@playwright/test");
const { ADMIN_API_KEY } = require("./helpers");

test.describe("Real backend: admin auth middleware", () => {
  test("purge history is rejected with 401 when no API key is supplied", async ({ request }) => {
    const resp = await request.post("/system/history/clear");
    expect(resp.status()).toBe(401);
  });

  test("purge history is rejected with 401 for a wrong API key", async ({ request }) => {
    const resp = await request.post("/system/history/clear", { headers: { "X-API-Key": "not-the-real-key" } });
    expect(resp.status()).toBe(401);
  });

  test("purge history succeeds with the correct admin API key", async ({ request }) => {
    const resp = await request.post("/system/history/clear", { headers: { "X-API-Key": ADMIN_API_KEY } });
    expect(resp.ok()).toBeTruthy();
  });

  test("settings write is rejected with 401 for a missing/wrong key", async ({ request }) => {
    const missing = await request.post("/system/settings", {
      headers: { "Content-Type": "application/json" },
      data: { telemetry_retention_hours: 24 },
    });
    expect(missing.status()).toBe(401);

    const wrong = await request.post("/system/settings", {
      headers: { "X-API-Key": "wrong", "Content-Type": "application/json" },
      data: { telemetry_retention_hours: 24 },
    });
    expect(wrong.status()).toBe(401);
  });
});
