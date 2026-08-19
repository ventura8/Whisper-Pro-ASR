const { test, expect } = require("@playwright/test");

async function setScenario(request, scenario) {
  await request.post("/__reset");
  const res = await request.post("/__lifecycle/scenario", { data: { name: scenario } });
  expect(res.ok()).toBeTruthy();
}

async function advanceLifecycle(request, delta = 1) {
  const res = await request.post("/__lifecycle/advance", { data: { delta } });
  expect(res.ok()).toBeTruthy();
}

function taskCard(page, taskId) {
  return page.locator(`#task-list .task-card[data-task-id="${taskId}"]`);
}

test.describe("Dashboard Concurrency & Preemption UI E2E", () => {
  test("renders multi-hardware cluster and active concurrent cards correctly", async ({ page, request }) => {
    await setScenario(request, "lifecycle-concurrency-burst");
    await page.goto("/dashboard");

    // Verify active & queued sessions overview metric cards reflect active concurrent tasks
    await expect(page.locator("#active-val")).toHaveText("3");
    await expect(page.locator("#queued-val")).toHaveText("2");

    // Verify multi-unit hardware badges rendered in task list
    await expect(page.locator("#task-list")).toContainText("NPU.0");
    await expect(page.locator("#task-list")).toContainText("GPU.0");
    await expect(page.locator("#task-list")).toContainText("CUDA.0");

    // Verify task cards render properly
    await expect(page.locator("#task-list .task-card")).toHaveCount(5);
  });

  test("filters tasks and renders empty history during concurrency burst", async ({ page, request }) => {
    await setScenario(request, "lifecycle-concurrency-burst");
    await page.goto("/dashboard");

    await expect.poll(() => page.locator("#task-list .task-card").count()).toBe(5);

    await page.click("#filter-asr");
    await expect.poll(() => page.locator("#task-list .task-card").count()).toBe(2);

    await page.click("#filter-detectlang");
    await expect.poll(() => page.locator("#task-list .task-card").count()).toBe(1);

    await page.click("#filter-v1");
    await expect.poll(() => page.locator("#task-list .task-card").count()).toBe(2);

    await page.click("#filter-all");
    await expect.poll(() => page.locator("#task-list .task-card").count()).toBe(5);

    await page.click("#tab-history");
    await expect(page.locator("#history-list")).toContainText("No history yet");
  });

  test("renders preemption pause hint banner during priority yield", async ({ page, request }) => {
    await setScenario(request, "lifecycle-concurrency-burst");
    await page.goto("/dashboard");

    // Advance to tick 1 where priority LD preempts NPU ASR
    await advanceLifecycle(request, 1);
    await page.evaluate(async () => {
      await window.updateStats();
    });

    const npuCard = taskCard(page, "conc-npu-asr-1");
    await expect(npuCard).toContainText("Paused for priority detect-language tasks");

    const prioCard = taskCard(page, "conc-prio-ld-1");
    await expect(prioCard).toContainText("active");
    await expect(prioCard).toContainText("Language Detection");
    await expect(prioCard).toContainText("NPU.0");
  });

  test("verifies zero placeholder strings rendered in DOM during concurrency", async ({ page, request }) => {
    await setScenario(request, "lifecycle-concurrency-burst");
    await page.goto("/dashboard");

    const textContent = await page.locator("#task-list").innerText();
    const normalized = textContent.replace(/\s+/g, "");
    expect(normalized).not.toContain("(0/0)");
    expect(textContent).not.toMatch(/\b(unknown|null|undefined|none)\b/i);
  });
});
