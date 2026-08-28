const { test, expect } = require("@playwright/test");

test.describe("Dashboard settings tab", () => {
  test.beforeEach(async ({ request }) => {
    await request.post("/__reset");
  });

  test("renders default slider values and labels on load", async ({ page }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    await expect(page.locator("#retention-range")).toHaveValue("24");
    await expect(page.locator("#retention-label")).toHaveText("24h");
    await expect(page.locator("#log-retention-range")).toHaveValue("7");
    await expect(page.locator("#log-retention-label")).toHaveText("7d");
  });

  test("moves retention sliders to their min/max and updates labels live without saving", async ({ page, request }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    await page.locator("#retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "1");
    await expect(page.locator("#retention-label")).toHaveText("1h");

    await page.locator("#retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "720");
    await expect(page.locator("#retention-label")).toHaveText("720h");

    await page.locator("#log-retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "1");
    await expect(page.locator("#log-retention-label")).toHaveText("1d");

    await page.locator("#log-retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "90");
    await expect(page.locator("#log-retention-label")).toHaveText("90d");

    const eventsResponse = await request.get("/__events");
    const events = await eventsResponse.json();
    expect(events.settingsSaves.length).toBe(0);
  });

  test("sends the entered API key as X-API-Key when saving configuration", async ({ page, request }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    await page.locator("#admin-api-key-input").fill("e2e_admin_key_123");

    page.on("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "Save Configuration" }).click();

    await expect.poll(async () => {
      const resp = await request.get("/__events");
      const events = await resp.json();
      return events.lastSettingsHeaders && events.lastSettingsHeaders["x-api-key"];
    }).toBe("e2e_admin_key_123");
  });

  test("shows a success alert and records the payload on a successful save", async ({ page, request }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    await page.locator("#retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "96");
    await page.locator("#log-retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "10");

    let dialogMessage = "";
    page.on("dialog", (dialog) => {
      dialogMessage = dialog.message();
      dialog.accept();
    });

    await page.getByRole("button", { name: "Save Configuration" }).click();
    await expect.poll(() => dialogMessage).toBe("Configuration saved!");

    const eventsResponse = await request.get("/__events");
    const events = await eventsResponse.json();
    expect(events.settingsSaves[events.settingsSaves.length - 1]).toMatchObject({
      telemetry_retention_hours: 96,
      log_retention_days: 10,
    });
  });

  test("shows a failure alert containing the status code when the save request fails", async ({ page, request }) => {
    await request.post("/__settings/fail", { data: { status: 500 } });

    await page.goto("/dashboard");
    await page.click("#tab-settings");

    let dialogMessage = "";
    page.on("dialog", (dialog) => {
      dialogMessage = dialog.message();
      dialog.accept();
    });

    await page.getByRole("button", { name: "Save Configuration" }).click();
    await expect.poll(() => dialogMessage).toContain("Failed to save settings (500)");
  });

  test("Purge Task History is a no-op when the confirm dialog is dismissed", async ({ page }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    const clearRequests = [];
    await page.route("**/system/history/clear", (route) => {
      clearRequests.push(route.request());
      route.continue();
    });

    page.on("dialog", (dialog) => dialog.dismiss());
    await page.getByRole("button", { name: "Clear History" }).click();

    // window.confirm() blocks page JS execution until the dialog resolves, so by the
    // time click() resolves the dismissed-confirm code path (which returns before ever
    // calling fetch) has already fully run -- no wait is needed. Route interception
    // (rather than reading fixture event counters) gives a deterministic, direct
    // signal that no request was ever initiated.
    expect(clearRequests).toHaveLength(0);
  });

  test("Purge Task History clears history when the confirm dialog is accepted", async ({ page, request }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    let dialogMessage = "";
    page.on("dialog", (dialog) => {
      dialogMessage = dialog.message();
      dialog.accept();
    });

    await page.getByRole("button", { name: "Clear History" }).click();
    await expect.poll(() => dialogMessage).toBe("Task history purged successfully.");

    const eventsResponse = await request.get("/__events");
    const events = await eventsResponse.json();
    expect(events.historyClears).toBe(1);
  });

  test("Purge Telemetry Metrics is a no-op when the confirm dialog is dismissed", async ({ page }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    const clearRequests = [];
    await page.route("**/system/telemetry/clear", (route) => {
      clearRequests.push(route.request());
      route.continue();
    });

    page.on("dialog", (dialog) => dialog.dismiss());
    await page.getByRole("button", { name: "Clear Telemetry" }).click();

    // See the Task History dismiss test above for why no wait is needed here.
    expect(clearRequests).toHaveLength(0);
  });

  test("Purge Telemetry Metrics clears telemetry when the confirm dialog is accepted", async ({ page, request }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    let dialogMessage = "";
    page.on("dialog", (dialog) => {
      dialogMessage = dialog.message();
      dialog.accept();
    });

    await page.getByRole("button", { name: "Clear Telemetry" }).click();
    await expect.poll(() => dialogMessage).toBe("Telemetry history purged successfully.");

    const eventsResponse = await request.get("/__events");
    const events = await eventsResponse.json();
    expect(events.telemetryClears).toBe(1);
  });

  test("does not persist slider values or API keys across a fresh page load", async ({ page }) => {
    await page.goto("/dashboard");
    await page.click("#tab-settings");

    await page.locator("#retention-range").evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, "168");
    await page.locator("#api-key-input").fill("should-not-survive-reload");
    await page.locator("#admin-api-key-input").fill("should-not-survive-reload-admin");

    await page.reload();
    await page.click("#tab-settings");

    await expect(page.locator("#retention-range")).toHaveValue("24");
    await expect(page.locator("#retention-label")).toHaveText("24h");
    await expect(page.locator("#api-key-input")).toHaveValue("");
    await expect(page.locator("#admin-api-key-input")).toHaveValue("");
  });
});
