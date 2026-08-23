const { test, expect } = require("@playwright/test");
const { ADMIN_API_KEY, resetRealBackendState } = require("./helpers");

test.describe("Real backend: settings persistence", () => {
  test.beforeEach(async ({ request }) => {
    await resetRealBackendState(request);
  });

  test("saving retention settings via the real UI persists on the real backend", async ({ page, request }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");
    await page.locator("#admin-api-key-input").fill(ADMIN_API_KEY);

    await page.locator("#retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "48");
    await page.locator("#log-retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "14");

    page.on("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "Save Configuration" }).click();
    await expect(page.locator("#retention-label")).toHaveText("48h");

    // Backend persistence is the source of truth; also re-open Settings after reload
    // and assert the sliders rehydrate from GET /system/settings (loadRetentionSettings).
    await expect
      .poll(
        async () => {
          const resp = await request.get("/system/settings", { headers: { "X-API-Key": ADMIN_API_KEY } });
          const data = await resp.json();
          return {
            telemetry: data.TELEMETRY_RETENTION_HOURS,
            logDays: data.LOG_RETENTION_DAYS,
          };
        },
        { timeout: 5_000 }
      )
      .toEqual({ telemetry: 48, logDays: 14 });

    await page.reload();
    await expect(page.locator("#app-name")).toBeVisible();
    await page.click("#tab-settings");
    await page.locator("#admin-api-key-input").fill(ADMIN_API_KEY);
    // Re-enter Settings so loadRetentionSettings runs with the restored admin key.
    await page.click("#tab-active");
    await page.click("#tab-settings");
    await expect(page.locator("#retention-range")).toHaveValue("48");
    await expect(page.locator("#log-retention-range")).toHaveValue("14");
  });

  test("the real /system/settings endpoint rejects an out-of-range retention value with a real 400", async ({
    request,
  }) => {
    const resp = await request.post("/system/settings", {
      headers: { "X-API-Key": ADMIN_API_KEY, "Content-Type": "application/json" },
      data: { telemetry_retention_hours: 99999 },
    });
    expect(resp.status()).toBe(400);
  });
});
