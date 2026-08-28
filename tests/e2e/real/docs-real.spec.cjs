const { test, expect } = require("@playwright/test");

// Part 4 gap: the custom-themed Swagger UI (/docs) previously had only backend
// string-assertion tests (test_server.py) and zero browser-level verification.
test.describe("Real backend: /docs Swagger UI", () => {
  test("Swagger UI mounts and lists the real app's routes", async ({ page }) => {
    await page.goto("/docs");
    await expect(page.locator(".swagger-ui")).toBeVisible({ timeout: 10_000 });

    // Real routes must actually appear, not a placeholder/empty spec.
    await expect(page.getByText("/asr", { exact: false }).first()).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("/v1/audio/transcriptions", { exact: false }).first()).toBeVisible();
  });

  test("theme CSS is actually applied, not just linked", async ({ page }) => {
    // swagger-theme.css only overrides colors inside `@media (prefers-color-scheme:
    // dark)`; the previous version of this test compared against getComputedStyle's
    // default browser return value, which is never an empty string even when the
    // theme CSS never loaded at all -- it always resolves to at least
    // "rgba(0, 0, 0, 0)", so that assertion could never actually fail. Force dark
    // mode and assert the exact themed color (`body { background-color: #1f1f1f }`)
    // so this fails if the stylesheet link is ever removed or broken.
    await page.emulateMedia({ colorScheme: "dark" });
    await page.goto("/docs");
    await expect(page.locator(".swagger-ui")).toBeVisible({ timeout: 10_000 });

    const bodyBackground = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    expect(bodyBackground).toBe("rgb(31, 31, 31)");
  });

  test("a 'Try it out' round trip against a real lightweight endpoint works interactively", async ({ page }) => {
    await page.goto("/docs");
    await expect(page.locator(".swagger-ui")).toBeVisible({ timeout: 10_000 });

    // Expand the GET /asr operation (status check - cheap, no file upload needed).
    // Constrained to opblock-get so this can't accidentally match the POST /asr
    // block (which also contains the "/asr" text and would require a file upload).
    const opBlock = page.locator(".opblock.opblock-get").filter({ hasText: "/asr" }).first();
    await opBlock.click();
    await opBlock.getByText("Try it out", { exact: true }).click();
    await opBlock.getByText("Execute", { exact: true }).click();

    const responsesTable = opBlock.locator(".live-responses-table").first();
    await expect(responsesTable).toBeVisible({ timeout: 10_000 });
    await expect(responsesTable.locator("tr.response td.response-col_status").first()).toContainText("200", {
      timeout: 10_000,
    });
  });
});
