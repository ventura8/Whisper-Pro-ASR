const path = require("path");
const { test, expect } = require("@playwright/test");
const { resetRealBackendState } = require("./helpers");

// Runs against the REAL FastAPI app (see tests/e2e/real_backend/serve_real_app.py) with
// only ASR inference/language-detection patched to deterministic fakes. Proves the
// Bazarr/v1 ingestion path and the dashboard display path are actually wired together
// end-to-end, which no unit/integration/fixture-mock e2e test verifies.
const FIXTURE_WAV = path.join(__dirname, "..", "fixtures", "silence.wav");

test.describe("Real backend: task lifecycle end-to-end", () => {
  test.beforeEach(async ({ request }) => {
    await resetRealBackendState(request);
  });

  test("a task submitted via the real /v1/audio/transcriptions endpoint appears active, then in history and analytics", async ({
    page,
    request,
  }) => {
    await page.goto("/dashboard");
    await page.click("#tab-active");

    // Fire the real transcription request in the background (the fake engine sleeps
    // ~1.5s so there's a window to observe the 'active' state before it completes).
    const submitPromise = request.post("/v1/audio/transcriptions?output=json", {
      multipart: { file: { name: "clip.wav", mimeType: "audio/wav", buffer: require("fs").readFileSync(FIXTURE_WAV) } },
    });

    await expect
      .poll(
        async () => {
          const statusResp = await request.get("/status");
          const data = await statusResp.json();
          return data.active_sessions;
        },
        { timeout: 5_000 }
      )
      .toBeGreaterThan(0);

    // Assert the browser UI itself (not just the API) shows the task as active
    // while it's still in flight -- the fake ASR delay is kept above the
    // dashboard's own poll interval specifically so this has a real window to
    // observe, rather than only checking post-completion state.
    const taskList = page.locator("#task-list");
    await expect(taskList).toContainText("clip.wav", { timeout: 5_000 });

    const submitResponse = await submitPromise;
    expect(submitResponse.ok()).toBeTruthy();
    const body = await submitResponse.json();
    expect(body.text).toContain("real-backend e2e fixture");

    // History must reflect the completed task via the real history_manager.
    await expect
      .poll(async () => {
        const historyResp = await request.get("/history");
        const history = await historyResp.json();
        return history.length;
      }, { timeout: 5_000 })
      .toBeGreaterThan(0);

    const historyResp = await request.get("/history");
    const history = await historyResp.json();
    expect(history[0].status).toBe("completed");
    expect(history[0].endpoint).toBe("/v1/audio/transcriptions");

    // Analytics must reflect the same task via real aggregation.
    const analyticsResp = await request.get("/analytics", { headers: { Accept: "application/json" } });
    const analytics = await analyticsResp.json();
    expect(analytics.cumulative.count_all_time).toBeGreaterThan(0);

    // And the dashboard UI itself (not just the API) must show it in history, once
    // its next /status poll picks up the completed task. Assert the submitted
    // record's own type label is actually rendered, not just that the empty-state
    // placeholder is gone (which a stale/unrelated history entry would also satisfy).
    await page.reload();
    await page.click("#tab-history");
    const historyList = page.locator("#history-list");
    await expect(historyList).not.toContainText("No history yet", { timeout: 10_000 });
    await expect(historyList).toContainText("Transcription", { timeout: 10_000 });
    await expect(historyList).toContainText("Finished", { timeout: 10_000 });
  });

  test("a task submitted via the real Bazarr multipart wire format (encode=false, raw audio) flows through to history", async ({
    request,
  }) => {
    // Bazarr's real wire format for encode=false is headerless raw s16le PCM (already
    // decoded client-side), never a WAV container -- see the Python-side contract test
    // in tests/integration/test_bazarr_wire_format.py's `_raw_pcm_bytes()`. Reusing
    // silence.wav's RIFF header here would misrepresent that contract even though ASR
    // itself is faked in this e2e run.
    const rawPcmBuffer = Buffer.alloc(2000, 1);
    const resp = await request.post("/asr?task=transcribe&language=en&output=srt&encode=false", {
      multipart: {
        audio_file: { name: "audio.raw", mimeType: "application/octet-stream", buffer: rawPcmBuffer },
      },
    });
    expect(resp.ok()).toBeTruthy();

    await expect
      .poll(async () => {
        const historyResp = await request.get("/history");
        const history = await historyResp.json();
        return history.some((entry) => entry.endpoint === "/asr");
      }, { timeout: 5_000 })
      .toBeTruthy();
  });
});
