const { test, expect } = require("@playwright/test");

// Part 4 gap: responsive_shared.css defines three real breakpoints (768px/600px/480px)
// and dark/light theme logic exists, but no prior spec ever set a viewport or forced a
// color scheme - every dashboard e2e test ran at the implicit desktop/light default.
const BREAKPOINTS = [
  { name: "desktop", width: 1280, height: 800 },
  { name: "just-under-768", width: 767, height: 900 },
  { name: "just-under-600", width: 599, height: 900 },
  { name: "just-under-480", width: 479, height: 900 },
];

test.describe("Real backend: dashboard responsive layout", () => {
  for (const bp of BREAKPOINTS) {
    test(`dashboard has no horizontal overflow and key controls stay usable at ${bp.name} (${bp.width}px)`, async ({
      page,
    }) => {
      await page.setViewportSize({ width: bp.width, height: bp.height });
      await page.goto("/dashboard");
      // Wait for first-render / post-fetch UI before measuring overflow.
      await expect(page.locator("#app-name")).toBeVisible();
      await expect(page.locator("#tab-active")).toBeVisible();

      const hasHorizontalOverflow = await page.evaluate(
        () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
      );
      expect(hasHorizontalOverflow).toBe(false);

      await expect(page.locator("#tab-settings")).toBeVisible();
      await page.click("#tab-settings");
      await expect(page.locator("#retention-range")).toBeVisible();

      // Settings has its own layout (sliders, labels, buttons) that could overflow
      // independently of the active-tasks tab checked above -- verify it doesn't.
      const settingsHasHorizontalOverflow = await page.evaluate(
        () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
      );
      expect(settingsHasHorizontalOverflow).toBe(false);
    });
  }

  test(`analytics page has no horizontal overflow at a mobile width`, async ({ page }) => {
    await page.setViewportSize({ width: 479, height: 900 });
    await page.goto("/analytics");
    await expect(page.locator("header h1")).toBeVisible();
    const hasHorizontalOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
    );
    expect(hasHorizontalOverflow).toBe(false);
  });

  test("dark color scheme actually changes computed background/text colors on the dashboard", async ({
    page,
  }) => {
    await page.emulateMedia({ colorScheme: "light" });
    await page.goto("/dashboard");
    const lightBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    const lightText = await page.evaluate(() => getComputedStyle(document.getElementById("app-name")).color);

    await page.emulateMedia({ colorScheme: "dark" });
    await page.reload();
    const darkBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    const darkText = await page.evaluate(() => getComputedStyle(document.getElementById("app-name")).color);

    expect(darkBg).not.toBe(lightBg);
    expect(darkText).not.toBe(lightText);
  });

  test("dark color scheme actually changes computed colors on the analytics page", async ({ page }) => {
    await page.emulateMedia({ colorScheme: "light" });
    await page.goto("/analytics");
    const lightBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    const lightText = await page.evaluate(() => getComputedStyle(document.querySelector("header h1")).color);

    await page.emulateMedia({ colorScheme: "dark" });
    await page.reload();
    const darkBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    const darkText = await page.evaluate(() => getComputedStyle(document.querySelector("header h1")).color);

    expect(darkBg).not.toBe(lightBg);
    expect(darkText).not.toBe(lightText);
  });
});
