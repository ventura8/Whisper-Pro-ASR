const ADMIN_API_KEY = "e2e_real_admin_key";

async function resetRealBackendState(request) {
  const historyResp = await request.post("/system/history/clear", { headers: { "X-API-Key": ADMIN_API_KEY } });
  if (!historyResp.ok()) {
    throw new Error(`resetRealBackendState: /system/history/clear failed with status ${historyResp.status()}`);
  }
  const telemetryResp = await request.post("/system/telemetry/clear", { headers: { "X-API-Key": ADMIN_API_KEY } });
  if (!telemetryResp.ok()) {
    throw new Error(`resetRealBackendState: /system/telemetry/clear failed with status ${telemetryResp.status()}`);
  }
}

module.exports = { ADMIN_API_KEY, resetRealBackendState };
